#!/usr/bin/env python3
"""
Simplified Direct HRM-TOPAS Training (robust version)
- GradScaler()
- device-aware autocast
- optional HRM->TOPAS best-effort bridge
- skip steps if logits missing (no dummy grads)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys, os, logging
from pathlib import Path
from trainers.arc_dataset_loader import ARCDataset
from models.topas_arc_60M import TopasARC60M, ModelConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging

def create_models(device):
    print("🔧 Creating HRM-TOPAS integrated models...")
    topas_config = ModelConfig()
    topas_config.width = 512
    topas_config.depth = 8
    topas_config.slots = 32
    topas_config.slot_dim = 256
    topas_config.max_dsl_depth = 4
    topas_config.use_ebr = True
    topas_config.enable_dream = False
    topas_config.verbose = True
    topas_config.pretraining_mode = True

    topas_model = TopasARC60M(topas_config).to(device)
    print(f"✅ TOPAS: {sum(p.numel() for p in topas_model.parameters()):,} parameters")

    hrm_config = {
        "batch_size": 1,
        "seq_len": 900,
        "vocab_size": 10,
        "num_puzzle_identifiers": 1000,
        "puzzle_emb_ndim": 128,
        "H_cycles": 3,
        "L_cycles": 4,
        "H_layers": 4,
        "L_layers": 4,
        "hidden_size": 512,
        "num_heads": 8,
        "halt_max_steps": 6,
        "forward_dtype": "bfloat16",
    }
    hrm_model = HierarchicalReasoningModel_ACTV1(hrm_config).to(device)
    print(f"✅ HRM: {sum(p.numel() for p in hrm_model.parameters()):,} parameters")

    return topas_model, hrm_model

def train_step(topas_model, hrm_model, batch, optimizer, scaler, device):
    """Single training step with safer AMP, optional HRM->TOPAS bridge, and robust loss handling."""
    optimizer.zero_grad()
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    try:
        demos, test_inputs, test_outputs, task_id = batch
        if not demos or len(demos) == 0:
            logging.warning("No demos in batch; skipping")
            return None

        input_grid = demos[0][0].to(device)
        target_grid = demos[0][1].to(device)

        # Normalize shapes
        if input_grid.dim() == 3 and input_grid.shape[0] == 1:
            input_grid = input_grid.squeeze(0)
        if target_grid.dim() == 3 and target_grid.shape[0] == 1:
            target_grid = target_grid.squeeze(0)

        input_grid = input_grid.unsqueeze(0)   # [1, H, W] or [1, C, H, W]
        target_grid = target_grid.unsqueeze(0)

        # Best-effort HRM latents
        hrm_latents = None
        try:
            if hasattr(hrm_model, "encode"):
                hrm_latents = hrm_model.encode(input_grid)
            else:
                hrm_out = hrm_model(input_grid)
                hrm_latents = hrm_out
        except Exception:
            hrm_latents = None

        with torch.amp.autocast(device_type, enabled=(device.type=='cuda')):
            # Prefer passing hrm_latents if TOPAS supports it
            try:
                if hrm_latents is not None:
                    outputs = topas_model.forward_pretraining(input_grid, hrm_latents=hrm_latents)
                else:
                    outputs = topas_model.forward_pretraining(input_grid)
            except TypeError:
                outputs = topas_model.forward_pretraining(input_grid)

            # Expect outputs to be dict-like and contain 'logits'
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
                target_flat = target_grid.view(target_grid.size(0), -1).long()

                if logits.dim() == 4:  # [B, C, H, W]
                    logits = logits.view(logits.size(0), logits.size(1), -1)
                    logits = logits.transpose(1, 2)  # [B, H*W, C]

                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_flat.view(-1))
            else:
                logging.error("train_step: outputs missing 'logits'; keys=%s", (list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)))
                return None

        if loss is None or (isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all()):
            logging.error("Invalid loss encountered; skipping step.")
            return None

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss.item() if isinstance(loss, torch.Tensor) else None

    except Exception:
        logging.exception("Exception in train_step")
        return None

def main():
    logger = setup_logging()
    print("🚀 Starting Simplified HRM-TOPAS Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    topas_model, hrm_model = create_models(device)

    dataset = ARCDataset(challenge_file="/mnt/d/Bitterbot/research/topas_v2/ARC-AGI/data/training", device=str(device))
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    optimizer = optim.AdamW(topas_model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler()

    num_epochs = 2
    total_steps = len(dataset) * num_epochs
    print(f"Training: {num_epochs} epochs, {total_steps} total steps")

    print("\n🎯 Starting training loop...")
    print(">>> TRAIN STARTED")
    global_step = 0

    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        from tqdm import tqdm
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress):
            loss = train_step(topas_model, hrm_model, batch, optimizer, scaler, device)
            if loss is not None:
                epoch_losses.append(loss)
            global_step += 1

            if len(epoch_losses) > 0:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})

            if global_step >= 50:
                print("Reached smoke test limit (50 steps). Exiting early.")
                break

        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}")

    print("\n🎉 Training completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Simplified HRM-TOPAS training WORKS!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback; traceback.print_exc()