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

def compute_metrics(model, input_grid, target_grid, hrm_latents=None):
    """Compute evaluation metrics with optional EBR refinement."""
    # Use the model's evaluate_with_ebr method for comprehensive metrics
    with torch.no_grad():
        eval_outputs = model.evaluate_with_ebr(input_grid, target_grid, hrm_latents=hrm_latents)
        
        if eval_outputs.get('logits') is None:
            return {'exact_match': 0.0, 'accuracy': 0.0, 'mean_iou': 0.0, 'exact_match_refined': 0.0}
        
        logits = eval_outputs['logits']
        B = logits.size(0)
        preds = logits.argmax(dim=-1)  # [B, H*W]
        targets_flat = target_grid.view(B, -1)  # [B, H*W]
        
        # Basic metrics
        exact_match = (preds == targets_flat).all(dim=1).float().mean().item()
        accuracy = (preds == targets_flat).float().mean().item()
        
        # IoU per color
        ious = []
        for c in range(10):  # NUM_COLORS
            pred_c = (preds == c)
            target_c = (targets_flat == c)
            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            if union > 0:
                ious.append((intersection / union).item())
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        
        # EBR refined exact match
        exact_match_refined = eval_outputs.get('exact_match_refined', 0.0)
        
        return {
            'exact_match': exact_match,
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'exact_match_refined': exact_match_refined
        }

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
        "expansion": 3.0,
        "num_heads": 8,
        "pos_encodings": "rope",
        "halt_max_steps": 6,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "bfloat16",
    }
    hrm_model = HierarchicalReasoningModel_ACTV1(hrm_config).to(device)
    print(f"✅ HRM: {sum(p.numel() for p in hrm_model.parameters()):,} parameters")

    return topas_model, hrm_model

def train_step(topas_model, hrm_model, batch, optimizer, scaler, device, return_metrics=False, global_step=0):
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
            # Pass target shape to fix mismatches + optional HRM latents
            target_shape = target_grid.shape[-2:]  # (H, W)
            try:
                if hrm_latents is not None:
                    outputs = topas_model.forward_pretraining(input_grid, hrm_latents=hrm_latents, target_shape=target_shape)
                else:
                    outputs = topas_model.forward_pretraining(input_grid, target_shape=target_shape)
            except TypeError:
                # Fallback for older signature
                outputs = topas_model.forward_pretraining(input_grid)

            # Expect outputs to be dict-like and contain 'logits'
            if isinstance(outputs, dict) and 'logits' in outputs and outputs['logits'] is not None:
                logits = outputs['logits']  # Should already be [B, H*W, C]
                
                # Check for None return (model detected issues)
                if logits is None:
                    logging.warning("Model returned None logits, skipping batch")
                    return None
                
                # Ensure target is properly shaped
                B = logits.size(0)
                H, W = target_grid.shape[-2:]
                target_flat = target_grid.view(B, -1).long()
                
                # Auto-align shapes if needed (model should handle this now)
                if logits.size(1) != target_flat.size(1):
                    logging.warning(f"Shape mismatch: logits {logits.shape} vs target {target_flat.shape}, skipping")
                    return None
                
                # Sanity check targets
                assert (target_flat >= 0).all() and (target_flat < 10).all(), f"Invalid target values: {target_flat.unique()}"
                
                # Cross-entropy with label smoothing
                label_smoothing = 0.05
                ce_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    target_flat.reshape(-1),
                    label_smoothing=label_smoothing
                )
                
                # Add DSL losses if available
                total_loss = ce_loss
                if 'losses' in outputs and outputs['losses']:
                    for loss_name, loss_value in outputs['losses'].items():
                        if loss_name == 'dsl_loss':
                            total_loss = total_loss + 0.1 * loss_value  # lambda_dsl = 0.1
                            if global_step % 100 == 0:  # Log occasionally
                                logging.info(f"Step {global_step}: ce_loss={ce_loss:.3f}, dsl_loss={loss_value:.3f}")
                
                loss = total_loss
            else:
                logging.error("train_step: outputs missing 'logits'; keys=%s", (list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)))
                return None

        if loss is None or (isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all()):
            logging.error("Invalid loss (NaN/Inf) at global step %d, skipping", global_step)
            return None

        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(topas_model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        if return_metrics:
            # Pass model and grids for comprehensive metrics including EBR
            metrics = compute_metrics(topas_model, input_grid, target_grid, hrm_latents=hrm_latents)
            return loss.item(), metrics
        else:
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
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # Conservative hyperparameters for stable training
    optimizer = optim.AdamW(topas_model.parameters(), lr=5e-5, weight_decay=1e-5)  # Lower LR
    scaler = torch.amp.GradScaler()

    num_epochs = 150  # Extended run with stable hyperparams
    total_steps = len(dataset) * num_epochs
    print(f"Training: {num_epochs} epochs, {total_steps} total steps")
    
    # Time estimation
    estimated_time_hours = total_steps / (10 * 3600)  # Assuming ~10 it/s from smoke test
    print(f"⏱️  Estimated training time: {estimated_time_hours:.1f} hours ({estimated_time_hours*60:.0f} minutes)")

    print("\n🎯 Starting training loop...")
    print(">>> TRAIN STARTED")
    global_step = 0
    best_em = 0.0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        epoch_metrics = {'exact_match': [], 'accuracy': [], 'mean_iou': [], 'exact_match_refined': []}
        from tqdm import tqdm
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress):
            # Compute metrics every 10 steps
            compute_metrics_now = (global_step % 10 == 0)
            
            result = train_step(topas_model, hrm_model, batch, optimizer, scaler, device, return_metrics=compute_metrics_now, global_step=global_step)
            
            if result is not None:
                if compute_metrics_now and isinstance(result, tuple):
                    loss, metrics = result
                    epoch_losses.append(loss)
                    for k, v in metrics.items():
                        epoch_metrics[k].append(v)
                else:
                    loss = result
                    if loss is not None:
                        epoch_losses.append(loss)
            
            global_step += 1

            # Update progress bar
            if len(epoch_losses) > 0:
                postfix = {"loss": f"{sum(epoch_losses[-10:]) / min(10, len(epoch_losses)):.4f}", "step": global_step}
                if len(epoch_metrics['exact_match']) > 0:
                    postfix['EM'] = f"{sum(epoch_metrics['exact_match'][-5:]) / min(5, len(epoch_metrics['exact_match'])):.2%}"
                    postfix['acc'] = f"{sum(epoch_metrics['accuracy'][-5:]) / min(5, len(epoch_metrics['accuracy'])):.2%}"
                    if len(epoch_metrics['exact_match_refined']) > 0:
                        postfix['EM_ebr'] = f"{sum(epoch_metrics['exact_match_refined'][-5:]) / min(5, len(epoch_metrics['exact_match_refined'])):.2%}"
                progress.set_postfix(postfix)

            # Save checkpoint every 2 epochs (more frequent for monitoring)
            if global_step % (len(dataset) * 2) == 0 and global_step > 0:
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': topas_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': sum(epoch_losses[-100:]) / min(100, len(epoch_losses)) if epoch_losses else 0,
                    'best_em': best_em,
                    'best_acc': best_acc
                }
                torch.save(checkpoint, f'checkpoint_step_{global_step}.pt')
                print(f"💾 Saved checkpoint at step {global_step}")

        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            summary = f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}"
            
            if len(epoch_metrics['exact_match']) > 0:
                avg_em = sum(epoch_metrics['exact_match']) / len(epoch_metrics['exact_match'])
                avg_acc = sum(epoch_metrics['accuracy']) / len(epoch_metrics['accuracy'])
                avg_iou = sum(epoch_metrics['mean_iou']) / len(epoch_metrics['mean_iou'])
                summary += f", EM={avg_em:.2%}, acc={avg_acc:.2%}, IoU={avg_iou:.3f}"
                
                # Add EBR refined exact match if available
                if len(epoch_metrics['exact_match_refined']) > 0:
                    avg_em_refined = sum(epoch_metrics['exact_match_refined']) / len(epoch_metrics['exact_match_refined'])
                    summary += f", EM_ebr={avg_em_refined:.2%}"
                
                # Track best metrics
                if avg_em > best_em:
                    best_em = avg_em
                    print(f"🎯 New best EM: {best_em:.2%}")
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    print(f"🎯 New best accuracy: {best_acc:.2%}")
            
            print(summary)

    # Save final checkpoint
    final_checkpoint = {
        'epoch': num_epochs,
        'global_step': global_step,
        'model_state_dict': topas_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_em': best_em,
        'best_acc': best_acc
    }
    torch.save(final_checkpoint, 'checkpoint_final.pt')
    print(f"💾 Saved final checkpoint: best_em={best_em:.2%}, best_acc={best_acc:.2%}")
    
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