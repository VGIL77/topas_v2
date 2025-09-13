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
import torch.nn.functional as F
import sys, os, logging
import argparse
import threading
import time
import json
import traceback
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from trainers.arc_dataset_loader import ARCDataset
from models.topas_arc_60M import TopasARC60M, ModelConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from trainers.self_play import SelfPlayBuffer  # used for storing dopamine rewards
from trainers.self_critique.counterexamples import CounterexampleGenerator, Task  # Task wrapper + counterexamples
from trainers.self_critique.star_bootstrapper import STaRBootstrapper           # STaR trace gen + verification
from trainers.self_critique.consistency import ConsistencyEnforcer               # enforce consistency across valid traces
from trainers.augmentation.deep_program_discoverer import mine_deep_programs                  # deep DSL programs miner (6–10 ops)
from collections import Counter, deque
import math

# =========================
# Dopamine & Nightmare Core
# =========================

# Global state shared across training
op_success_count = Counter()          # track operations in successful traces (for planner op_bias)
recent_failures: List[Any] = []       # queue of failed counterexamples for nightmares
rolling_em = deque(maxlen=200)        # rolling window of EM to estimate failure pressure

def build_op_bias(temp: float = 0.7):
    """
    Convert op_success_count to a softmax prior (democratic → data-driven).
    STaR will accept planner_op_bias for roughly half the traces.
    """
    ops = list(op_success_count.keys())
    if not ops:
        # If no data yet, keep a democratic prior over 41 ops
        ops = [f"op_{i}" for i in range(41)]
        vals = [1.0] * len(ops)
    else:
        vals = [op_success_count.get(op, 1.0) for op in ops]
    mx = max(vals) if vals else 1.0
    exps = [math.exp((v - mx) / max(1e-6, temp)) for v in vals]
    Z = sum(exps) if exps else 1.0
    return {op: (e / Z) for op, e in zip(ops, exps)}

def dopamine_reward(task: Task, buffer: Optional[SelfPlayBuffer], logger, global_step: int):
    """
    Positive reinforcement: store solved puzzle pair in SelfPlayBuffer.
    We store (input, target) pairs; SelfPlayBuffer will generate templated variants later.
    """
    if buffer is None:
        return
    try:
        # Accept Task, dict, or (input, output) tuple/list
        if hasattr(task, "input") and hasattr(task, "output"):
            inp, out = task.input, task.output
        elif isinstance(task, (list, tuple)) and len(task) == 2:
            inp, out = task[0], task[1]
        elif isinstance(task, dict):
            inp, out = task.get("input"), task.get("output")
        else:
            raise TypeError(f"Unsupported task type for dopamine capture: {type(task)}")

        buffer.buffer.append((inp.detach().cpu(), out.detach().cpu()))
        logger.info(f"[Dopamine] Stored 1 puzzle pair → buffer size={len(buffer.buffer)} at step {global_step}")
    except Exception as e:
        logger.warning(f"[Dopamine] buffer append failed: {e}")

def nightmare_prune(model, failures: List[Any], optimizer, scaler, device, logger, global_step: int,
                    alpha: float = 0.08, max_failures: int = 8):
    """
    Negative replay: apply small negative gradients on recent failed counterexamples.
    Uses model.evaluate_with_ebr() logits pathway like compute_metrics() for stability.
    """
    if not failures:
        return
    # Limit the number per cycle for stability
    batch_failures = failures[:max_failures]
    del failures[:max_failures]

    model.train()
    optimizer.zero_grad(set_to_none=True)
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'

    neg_terms = []  # collect per-failure negative CE terms to ensure a real graph
    with torch.amp.autocast(device_type, enabled=(device.type == 'cuda')):
        for f in batch_failures:
            try:
                # Counterexample objects expose .task with .input/.output in Phase 3 code.
                task = getattr(f, "task", None)
                if task is None:
                    continue
                inp = task.input.to(device).unsqueeze(0)   # [1, H, W]
                tgt = task.output.to(device).unsqueeze(0)  # [1, H, W]

                logits = None
                # Prefer a grad-enabled path
                try:
                    out = model.forward_pretraining(inp, target_shape=tgt.shape[-2:])
                    logits = out.get('logits', None) if isinstance(out, dict) else None
                except Exception:
                    logits = None

                if logits is None:
                    # Fallback to evaluate_with_ebr (may be no-grad depending on implementation)
                    try:
                        eval_out = model.evaluate_with_ebr(inp, tgt)
                        logits = eval_out.get('logits', None)
                    except Exception:
                        logits = None

                if logits is None or (isinstance(logits, torch.Tensor) and not logits.requires_grad):
                    # Can't use this item for gradient-based pruning
                    continue

                # Negative CE: move away from these bad solutions
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     tgt.view(-1).long().clamp(0, 9))
                neg_terms.append(-alpha * ce)
            except Exception as e:
                logger.debug(f"[Nightmare] skip one failure: {e}")
                continue

    if len(neg_terms) == 0:
        logger.info(f"[Nightmare] No grad-capable failures to prune at step {global_step} (skipped)")
        return

    # Aggregate (mean) to keep magnitude small and stable
    total_neg = torch.stack(neg_terms).mean()

    if torch.isfinite(total_neg) and total_neg.requires_grad:
        scaler.scale(total_neg).backward()
        # modest step; grads are small
        scaler.step(optimizer)
        scaler.update()
        logger.info(
            f"[Nightmare] Applied negative replay on {len(neg_terms)}/{len(batch_failures)} "
            f"failures at step {global_step} (loss={float(total_neg.item()):.4f})"
        )
    else:
        logger.warning("[Nightmare] Skipped due to non-finite or no-grad loss")

def parse_args():
    parser = argparse.ArgumentParser(description="Train TOPAS+HRM (with DreamEngine controls)")
    # Dream engine controls
    parser.add_argument("--enable-dream", action="store_true", default=False,
                        help="Enable DreamEngine (micro ticks during forward) and offline cycles.")
    parser.add_argument("--dream-micro-ticks", type=int, default=1,
                        help="Number of micro dream ticks to run during each forward_pretraining call.")
    parser.add_argument("--dream-full-every", type=int, default=10,
                        help="Run a full offline dream consolidation every N epochs. 0 disables.")
    parser.add_argument("--dream-full-timeout", type=int, default=600,
                        help="Log timeout threshold (seconds) for full dream cycle; used for warnings only.")
    parser.add_argument("--dream-background", action="store_true", default=False,
                        help="If set, run full dream cycles in a background daemon thread (risky if dream touches model GPU state).")
    parser.add_argument("--dream-force-cpu", action="store_true", default=False,
                        help="Hint: prefer CPU for offline dream cycle if supported (not enforced here).")

    # RelMem configuration
    parser.add_argument("--relmem-reg-alpha", type=float, default=1e-3,
                        help="Weight for inverse_loss_safe regularization")
    parser.add_argument("--relmem-reg-beta", type=float, default=5e-4,
                        help="Weight for inheritance_pass regularization")
    parser.add_argument("--relmem-bind-iou", type=float, default=0.25,
                        help="IoU threshold for RelMem concept binding on success")
    parser.add_argument("--relmem-log-interval", type=int, default=200,
                        help="Log RelMem stats every N steps")
    
    # Progressive RelMem bias ramping parameters
    parser.add_argument("--relmem-bias-ramp-start", type=int, default=10,
                        help="Epoch to start ramping up RelMem bias")
    parser.add_argument("--relmem-bias-max", type=float, default=0.5,
                        help="Maximum RelMem bias weight after ramping")
    
    # Dream pretrain args
    parser.add_argument("--dream-pretrain-epochs", type=int, default=0,
                        help="Number of epochs for Dream/ETS pretraining (default 0 = disabled)")
    parser.add_argument("--dream-pretrain-lr", type=float, default=1e-4,
                        help="Learning rate for Dream pretraining")
    parser.add_argument("--dream-pretrain-batches", type=int, default=200,
                        help="Number of batches per Dream pretrain epoch")
    parser.add_argument("--dream-pretrain-freeze-model", action="store_true", default=False,
                        help="Freeze main model during Dream pretraining")
    
    # Self-play args
    parser.add_argument("--selfplay-enable", action="store_true", default=False,
                        help="Enable self-play with template-guided puzzles")
    parser.add_argument("--selfplay-interval", type=int, default=250,
                        help="Generate self-play puzzles every N steps")
    parser.add_argument("--selfplay-weight", type=float, default=0.1,
                        help="Weight for self-play loss")
    parser.add_argument("--selfplay-topk", type=int, default=3,
                        help="Number of puzzles to generate per self-play round")
    parser.add_argument("--selfplay-buffer-size", type=int, default=200,
                        help="Maximum size of self-play buffer")
    
    # RelMem / UKS-lite
    parser.add_argument('--relmem-loss-weight', type=float, default=0.01,
                        help="Weight for RelMem auxiliary loss")
    parser.add_argument('--relmem-loss-interval', type=int, default=25,
                        help="Apply RelMem loss every N steps")
    parser.add_argument('--uks-save-path', type=str, default='uks_state.pt',
                        help="Path to save UKS state")
    parser.add_argument('--uks-load-path', type=str, default='',
                        help="Path to load UKS state from")
    
    # Eval args
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Run evaluation every N epochs")
    
    # Training args
    parser.add_argument("--max-steps", type=int, default=60000,
                        help="Maximum training steps")

    # ---- Model-size & dopaminergic/nightmare controls ----
    parser.add_argument("--model-width", type=int, default=640,
                        help="Hidden width for TOPAS conv backbone (default 640)")
    parser.add_argument("--model-slots", type=int, default=64,
                        help="Number of slots for concept vectors (default 64)")
    parser.add_argument("--breakthrough-threshold", type=float, default=0.33,
                        help="EM threshold to trigger dopamine capture (default 0.33)")
    parser.add_argument("--nightmare-alpha", type=float, default=0.08,
                        help="Negative reinforcement strength for nightmares (0.05–0.10 recommended)")
    parser.add_argument("--nightmare-min-interval", type=int, default=200,
                        help="Minimum steps between nightmare cycles (when failure low)")
    parser.add_argument("--nightmare-max-interval", type=int, default=1000,
                        help="Maximum steps between nightmare cycles (when failure high)")
    # Mind-voice controls
    parser.add_argument("--monologue-interval", type=int, default=200,
                        help="Every N steps, sample traces and compute reasoning consistency")
    parser.add_argument("--monologue-min-traces", type=int, default=4,
                        help="Min number of traces to consider for consistency")
    parser.add_argument("--monologue-consistency-target", type=float, default=0.85,
                        help="Target overall consistency; below this we increase pruning, above this we ramp bias")
    parser.add_argument("--monologue-selfplay-bonus", type=float, default=0.05,
                        help="Increase self-play weight by this when consistency is high")
    
    args, _unknown = parser.parse_known_args()
    return args

def run_dream_cycle_safe(model,
                         timeout_sec: int = 600,
                         background: bool = False,
                         force_cpu: bool = False,
                         logger=None) -> Optional[Dict[str, Any]]:
    """
    Run model.run_dream_cycle() robustly.

    - Checks model has run_dream_cycle
    - Skips if no cached tokens and model.run_dream_cycle requires tokens
    - Logs start/end, duration, returned stats (if any)
    - Optionally runs in a background daemon thread (default False).

    Returns the dict returned by run_dream_cycle() if called and returned; otherwise None.
    """
    logger = logger or logging.getLogger(__name__)

    if not hasattr(model, "run_dream_cycle"):
        logger.warning("[Dream] model has no run_dream_cycle() method; skipping.")
        return None

    # Best-effort: prefer to pass cached tokens if available
    tokens = getattr(model, "_dream_tokens", None)

    def _call_cycle():
        try:
            t0 = time.time()
            logger.info("[Dream] Full cycle started (force_cpu=%s) ...", force_cpu)
            # Call with tokens if available; wrap in try/except
            try:
                if tokens is not None:
                    stats = model.run_dream_cycle(tokens=tokens)
                else:
                    # Some implementations accept no args
                    stats = model.run_dream_cycle()
            except TypeError:
                # Older signature - call without tokens
                stats = model.run_dream_cycle()
            duration = time.time() - t0
            logger.info("[Dream] Full cycle finished in %.1f s. stats=%s", duration, repr(stats))
            return stats
        except Exception as e:
            logger.warning("[Dream] Full cycle failed: %s\n%s", repr(e), traceback.format_exc())
            return None

    if background:
        # Run asynchronously in a daemon thread and return immediately.
        logger.warning("[Dream] Running full dream cycle in background thread (thread-safety warning).")
        th = threading.Thread(target=_call_cycle, daemon=True, name="dream_cycle_thread")
        th.start()
        return None
    else:
        # Run synchronously (blocking) and return stats. Use a simple watchdog for long runs (warning only).
        t_start = time.time()
        stats = _call_cycle()
        t_total = time.time() - t_start
        if t_total > timeout_sec:
            logger.warning("[Dream] Full cycle exceeded timeout threshold %ds (took %.1fs)", timeout_sec, t_total)
        return stats

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

def create_models(device, cli_args=None):
    print("🔧 Creating HRM-TOPAS integrated models...")
    topas_config = ModelConfig()
    
    # Apply CLI dream settings to config before model creation
    if cli_args and getattr(cli_args, "enable_dream", False):
        topas_config.enable_dream = True
        if hasattr(cli_args, "dream_micro_ticks"):
            topas_config.dream_micro_ticks = cli_args.dream_micro_ticks
    # Allow moderate scaling to use GPU headroom safely
    topas_config.width = int(getattr(cli_args, "model_width", 640))
    topas_config.depth = 8
    topas_config.slots = int(getattr(cli_args, "model_slots", 64))
    topas_config.slot_dim = 256
    topas_config.max_dsl_depth = 4
    topas_config.use_ebr = True
    
    # Progressive RelMem bias ramping - start stable, then boost
    base_bias_w = 0.2  # Proven stable value
    max_bias_w = getattr(cli_args, 'relmem_bias_max', 0.5)
    ramp_start = getattr(cli_args, 'relmem_bias_ramp_start', 10)
    
    # Boosted RelMem DSL influence parameters (will be ramped progressively)
    topas_config.relmem_op_bias_w = base_bias_w        # Start with proven stable
    topas_config.relmem_op_bias_scale = 1.0           # Double from 0.5 → full scaling  
    topas_config.relmem_bias_beta = 0.4               # Double from 0.2 → aggressive bias
    
    # Store ramping parameters for training loop
    topas_config._bias_ramp_start = ramp_start
    topas_config._bias_max = max_bias_w
    topas_config._bias_base = base_bias_w
    
    # Don't override enable_dream - respect CLI setting
    topas_config.verbose = True
    topas_config.pretraining_mode = True

    topas_model = TopasARC60M(topas_config).to(device)
    
    # Ensure Dream system is also moved to the correct device
    if hasattr(topas_model, 'dream') and topas_model.dream is not None:
        if hasattr(topas_model.dream, 'to'):
            topas_model.dream.to(device)
    
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

def dream_pretrain_loop(topas_model, dataset, cli_args, device, logger):
    """
    Tiny Dream-Pretrain phase: train Dream/ETS only for a few epochs
    """
    if not cli_args or cli_args.dream_pretrain_epochs <= 0:
        return
        
    logger.info(f"[Dream-Pretrain] Starting {cli_args.dream_pretrain_epochs} epochs")
    
    # Check if model has dream engine
    if not hasattr(topas_model, 'dream') or topas_model.dream is None:
        logger.warning("[Dream-Pretrain] No DreamEngine found, skipping pretrain")
        return
    
    dream = topas_model.dream
    # Move dream to the same device as the model
    if hasattr(dream, 'to'):
        # Use the proper to() method which handles device and generator
        dream.to(device)
    elif hasattr(dream, 'device'):
        # Fallback: Update device and move internal modules
        dream.device = device
        # Move internal modules if they exist
        if hasattr(dream, '_dream_color_head') and dream._dream_color_head is not None:
            dream._dream_color_head = dream._dream_color_head.to(device)
        if hasattr(dream, '_dream_theme_head') and dream._dream_theme_head is not None:
            dream._dream_theme_head = dream._dream_theme_head.to(device)
        if hasattr(dream, 'nmda') and dream.nmda is not None:
            for attr_name in dir(dream.nmda):
                attr = getattr(dream.nmda, attr_name)
                if isinstance(attr, torch.nn.Module):
                    setattr(dream.nmda, attr_name, attr.to(device))
    
    # robustly collect dream params
    dream_params = []
    if hasattr(dream, 'parameters') and callable(getattr(dream, 'parameters')):
        try:
            dream_params = [p for p in dream.parameters() if p is not None]
        except Exception:
            dream_params = []
    # if still empty, attempt recursive attribute scan (already implemented in dream.parameters())
    if len(dream_params) == 0:
        logging.getLogger(__name__).warning("[Dream-Pretrain] No dream params found (will attempt fallback or skip)")
        return
    
    if not dream_params:
        logger.warning("[Dream-Pretrain] No trainable Dream/ETS parameters found")
        return
    
    dream_optimizer = torch.optim.Adam(dream_params, lr=cli_args.dream_pretrain_lr)
    
    # Freeze main model if requested
    if cli_args.dream_pretrain_freeze_model:
        topas_model.eval()
        for param in topas_model.parameters():
            param.requires_grad = False
    
    # Use proven single-sample approach but benefit from larger Dream buffer (512)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    
    for epoch in range(cli_args.dream_pretrain_epochs):
        total_loss = 0.0
        motifs_added = 0
        buffer_len = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # ARCDataLoader returns (demos, test_inputs, test_outputs, task_id)
            demos, test_inputs, test_outputs, task_id = batch
            if batch_idx >= cli_args.dream_pretrain_batches:
                break

            # Pick the first available test input/output (if any)
            test_grid = test_inputs[0] if test_inputs else None
            target_grid = test_outputs[0] if test_outputs else None

            # Safety: fallback to minimal tensor if dataset gives nothing
            if test_grid is None:
                test_grid = torch.zeros((1, 3, 3), device=device, dtype=torch.long)
            if target_grid is None:
                target_grid = torch.zeros_like(test_grid)
                    
            # Ensure tensors are on the right device
            test_grid = test_grid.to(device)
            target_grid = target_grid.to(device)
                
            # Get slot features from model (no grad if frozen)
            with torch.no_grad() if cli_args.dream_pretrain_freeze_model else torch.enable_grad():
                extras = {}
                if hasattr(topas_model, 'encoder'):
                    # Ensure batch dimension
                    if test_grid.dim() == 2:
                        test_grid = test_grid.unsqueeze(0)
                    enc_in = test_grid.float() / 9.0  # Normalize
                    feat, glob = topas_model.encoder(enc_in)
                    if hasattr(topas_model, 'slots'):
                        slot_vecs = topas_model.slots(feat)
                        if isinstance(slot_vecs, tuple):
                            slot_vecs = slot_vecs[0]
                        # Ensure slot_vecs has correct shape [B, num_slots, slot_dim]
                        if slot_vecs.dim() == 2:
                            slot_vecs = slot_vecs.unsqueeze(1)  # Add slot dimension
                        
                        # Concatenate global features with slot vectors to match state_dim
                        # DreamEngine expects state_dim = ctrl_dim (width + slot_dim + puzzle_emb if present)
                        B, K, D = slot_vecs.shape
                        glob_expanded = glob.unsqueeze(1).expand(B, K, -1)  # [B, K, width]
                        
                        # Check if we need to add puzzle_emb to match ctrl_dim
                        # The model's ctrl_dim includes puzzle_emb (128) when planner is present
                        if hasattr(topas_model, '_has_planner') and topas_model._has_planner:
                            # Add dummy puzzle_emb for dream pretraining (128D)
                            puzzle_emb = torch.zeros(B, K, 128, device=device)
                            combined_state = torch.cat([glob_expanded, slot_vecs, puzzle_emb], dim=-1)  # [B, K, ctrl_dim]
                        else:
                            combined_state = torch.cat([glob_expanded, slot_vecs], dim=-1)  # [B, K, width+slot_dim]
                        
                        # Verify dimension matches DreamEngine's expectation
                        expected_dim = topas_model.ctrl_dim if hasattr(topas_model, 'ctrl_dim') else 768
                        if combined_state.shape[-1] != expected_dim:
                            logger.warning(f"[Dream-Pretrain] Dimension mismatch: latent has {combined_state.shape[-1]}D, expected {expected_dim}D")
                        
                        extras['latent'] = combined_state
            
            # Train Dream/ETS
            if hasattr(dream, 'train_step'):
                latent = extras.get('latent')
                if latent is not None:
                    print(f"[Dream-Debug] latent shape: {latent.shape}, target shape: {target_grid.shape if target_grid is not None else None}")
                loss = dream.train_step(latent, target=target_grid)
            else:
                # Minimal fallback: reconstruction loss
                if 'latent' in extras and extras['latent'] is not None:
                    # Simple projection loss
                    proj = torch.nn.functional.linear(extras['latent'].mean(dim=1), 
                                                     torch.randn(10, extras['latent'].size(-1), device=device))
                    loss = torch.nn.functional.cross_entropy(proj.view(-1, 10), 
                                                            target_grid.view(-1).long())
                else:
                    loss = torch.tensor(0.0, device=device)
            
            if loss.requires_grad:
                dream_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dream_params, max_norm=1.0)
                dream_optimizer.step()
            
            total_loss += loss.item()
            
            # Track metrics
            if hasattr(dream, 'nmda') and hasattr(dream.nmda, 'buffer'):
                buffer_len = len(dream.nmda.buffer)
            if hasattr(dream, 'theme') and hasattr(dream.theme, 'synthesis_count'):
                motifs_added = dream.theme.synthesis_count
        
        avg_loss = total_loss / min(batch_idx + 1, cli_args.dream_pretrain_batches)
        logger.info(f"[Dream-Pretrain] Epoch {epoch+1}/{cli_args.dream_pretrain_epochs}: "
                   f"loss={avg_loss:.4f}, buffer_len={buffer_len}, motifs_added={motifs_added}")
    
    # Save pretrained Dream/ETS
    try:
        dream.save_state('dream_pretrain.pth')
        logger.info("[Dream-Pretrain] saved dream_pretrain.pth")
    except Exception:
        logging.getLogger(__name__).exception("[Dream-Pretrain] Could not save dream state")

def train_step(topas_model, hrm_model, batch, optimizer, scaler, device, return_metrics=False, global_step=0):
    """Single training step with safer AMP, optional HRM->TOPAS bridge, and robust loss handling."""
    optimizer.zero_grad()
    device_type = 'cuda' if str(device).startswith('cuda') else 'cpu'

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
            # Pass target shape, demos, and global_step for complete DSL+EBR integration
            target_shape = target_grid.shape[-2:]  # (H, W)
            try:
                outputs = topas_model.forward_pretraining(
                    input_grid, 
                    hrm_latents=hrm_latents, 
                    target_shape=target_shape,
                    demos=demos,
                    global_step=global_step
                )
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
                
                # Batch debug probe for CE spikes
                if global_step % 100 == 0 and ce_loss > 2.0:  # Log when CE loss is high
                    batch_shapes = [tuple(input_grid.shape), tuple(target_grid.shape)]
                    logging.info(f"[BATCH DEBUG] CE_spike={ce_loss.item():.3f} batch_shapes={batch_shapes}")
                
                # Dream health check - log cached tokens if available
                if global_step % 100 == 0 and getattr(topas_model, "_dream_tokens", None) is not None:
                    tokens_shape = getattr(topas_model, "_dream_tokens").shape
                    logging.info(f"[Dream] _dream_tokens shape: {tokens_shape}")
                
                # Add DSL losses (weight already annealed in model)
                total_loss = ce_loss
                if 'losses' in outputs and outputs['losses']:
                    for loss_name, loss_value in outputs['losses'].items():
                        if loss_name == 'dsl_loss':
                            total_loss = total_loss + loss_value  # Weight already applied in model
                            if global_step % 100 == 0:  # Log occasionally
                                logging.info(f"Step {global_step}: ce_loss={ce_loss:.3f}, dsl_loss={loss_value:.3f}")
                
                # ---- RelMem auxiliary loss every N steps (with warm-up) ----
                # Delay RelMem until after 5 epochs (~2000 steps)
                relmem_warmup_epochs = 5
                current_epoch = global_step // 400  # assumes ~400 steps/epoch
                relmem_loss_interval = getattr(args, 'relmem_loss_interval', 25) if 'args' in locals() else 25
                if (hasattr(topas_model, "relmem") and topas_model.relmem is not None and 
                    current_epoch >= relmem_warmup_epochs and (global_step % relmem_loss_interval == 0)):
                    try:
                        reg_alpha = getattr(args, 'relmem_reg_alpha', 1e-4) if 'args' in locals() else 1e-4  # Reduced for stability
                        reg_beta = getattr(args, 'relmem_reg_beta', 1e-4) if 'args' in locals() else 1e-4   # Reduced for stability
                        
                        # Replace broken inverse_loss() with inverse_loss_safe()
                        relmem_aux = torch.tensor(0.0, device=device)
                        if hasattr(topas_model.relmem, 'inverse_loss_safe'):
                            relmem_aux = relmem_aux + reg_alpha * topas_model.relmem.inverse_loss_safe()
                        
                        # Add inheritance_pass regularization
                        if hasattr(topas_model.relmem, 'inheritance_pass'):
                            relmem_aux = relmem_aux + reg_beta * topas_model.relmem.inheritance_pass()
                        elif hasattr(topas_model.relmem, 'inheritance_pass_plus'):
                            relmem_aux = relmem_aux + reg_beta * topas_model.relmem.inheritance_pass_plus()
                        
                        if torch.is_tensor(relmem_aux) and relmem_aux.item() > 0:
                            total_loss = total_loss + relmem_aux
                            if global_step % 100 == 0:
                                logging.info(f"Step {global_step}: RelMem aux loss={relmem_aux.item():.6f}")
                    except Exception as e:
                        if global_step % 100 == 0:
                            logging.warning(f"RelMem auxiliary loss failed: {e}")
                
                # RelMem binding on success (check if we have metrics to evaluate)
                if return_metrics:
                    try:
                        # Quick metrics for binding decision
                        preds = logits.argmax(dim=-1)  # [B, H*W]
                        targets_flat = target_grid.view(B, -1)  # [B, H*W]
                        exact_match = (preds == targets_flat).all(dim=1).float().mean().item()
                        
                        # Compute IoU for binding threshold
                        ious = []
                        for c in range(10):  # NUM_COLORS
                            pred_c = (preds == c)
                            target_c = (targets_flat == c)
                            intersection = (pred_c & target_c).sum().float()
                            union = (pred_c | target_c).sum().float()
                            if union > 0:
                                ious.append((intersection / union).item())
                        mean_iou = sum(ious) / len(ious) if ious else 0.0
                        
                        # Enhanced RelMem concept binding on success
                        if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                            # Extract brain latent from outputs
                            brain_latent = None
                            if hasattr(outputs, 'brain') and outputs['brain'] is not None:
                                brain_latent = outputs['brain']
                            elif 'brain' in outputs and outputs['brain'] is not None:
                                brain_latent = outputs['brain']
                            elif 'latent' in outputs and outputs['latent'] is not None:
                                brain_latent = outputs['latent']
                            
                            if brain_latent is not None:
                                # Check for success criteria with configurable threshold
                                bind_iou_threshold = getattr(args, 'relmem_bind_iou', 0.25) if 'args' in locals() else 0.25
                                em_success = exact_match > 0
                                iou_success = mean_iou >= bind_iou_threshold
                                
                                if (em_success or iou_success):
                                    # Perform concept binding
                                    try:
                                        if hasattr(topas_model, '_relmem_try_bind'):
                                            binding_stats = topas_model._relmem_try_bind(
                                                brain=brain_latent,
                                                ops_used=outputs.get('extras', {}).get('ops_used', []),
                                                iou=mean_iou,
                                                em=em_success
                                            )
                                        else:
                                            # Fallback binding method
                                            binding_stats = {'relmem_bound': False}
                                            if hasattr(topas_model.relmem, 'bind_concept'):
                                                topas_model.relmem.bind_concept(brain_latent, success_metric=mean_iou)
                                                binding_stats['relmem_bound'] = True
                                                binding_stats['relmem_concept_id'] = 'auto'
                                        
                                        # --- Wormhole integration ---
                                        try:
                                            if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole"):
                                                rule_info = outputs.get('extras', {}).get('rule_info')
                                                ops = outputs.get('extras', {}).get('ops_used', [])
                                                programs = rule_info.get("programs", [ops]) if isinstance(rule_info, dict) else [ops]
                                                mined = topas_model.dream.wormhole.mine_from_programs(programs, top_k=5)
                                                if mined:
                                                    logging.getLogger(__name__).info(f"[Wormhole] mined {len(mined)} templates")
                                                    # Bind mined templates into RelMem
                                                    for tpl in mined:
                                                        try:
                                                            tpl_sig = str(tpl)[:32]
                                                            if hasattr(topas_model.relmem, "bind_concept_by_vector"):
                                                                topas_model.relmem.bind_concept_by_vector(
                                                                    brain_latent.squeeze(0), op_name="wormhole",
                                                                    meta={"template": tpl_sig}, alpha=0.5
                                                                )
                                                        except Exception as e:
                                                            logging.getLogger(__name__).warning(f"[Wormhole] RelMem bind failed: {e}")
                                        except Exception as e:
                                            logging.getLogger(__name__).warning(f"[Wormhole] integration failed: {e}")
                                        
                                        if binding_stats.get('relmem_bound') and global_step % 100 == 0:
                                            concept_id = binding_stats.get('relmem_concept_id', 'unknown')
                                            logging.info(f"RelMem bound concept {concept_id} on success (EM={exact_match:.3f}, IoU={mean_iou:.3f})")
                                    except Exception as e:
                                        if global_step % 200 == 0:
                                            logging.warning(f"RelMem concept binding failed: {e}")
                    except Exception as e:
                        if global_step % 100 == 0:
                            logging.warning(f"RelMem binding failed: {e}")
                
                # RelMem regularization
                relmem_reg_loss = torch.tensor(0.0, device=device)
                if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                    try:
                        # Compute RelMem regularization
                        reg_alpha = getattr(args, 'relmem_reg_alpha', 0.01) if 'args' in locals() else 0.01
                        reg_beta = getattr(args, 'relmem_reg_beta', 0.02) if 'args' in locals() else 0.02
                        
                        if hasattr(topas_model.relmem, 'compute_regularization'):
                            relmem_reg_loss = topas_model.relmem.compute_regularization(alpha=reg_alpha, beta=reg_beta)
                        elif hasattr(topas_model.relmem, 'regularization_loss'):
                            relmem_reg_loss = topas_model.relmem.regularization_loss() * reg_alpha
                            
                        if torch.is_tensor(relmem_reg_loss) and relmem_reg_loss.item() > 0:
                            total_loss = total_loss + relmem_reg_loss
                            
                    except Exception as e:
                        if global_step % 100 == 0:
                            logging.warning(f"RelMem regularization failed: {e}")
                
                loss = total_loss
            else:
                logging.error("train_step: outputs missing 'logits'; keys=%s", (list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)))
                return None

        if loss is None or (isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all()):
            logging.error("Invalid loss (NaN/Inf) at global step %d, skipping", global_step)
            return None

        scaler.scale(loss).backward()
        
        # Cache batch and programs for dream seeding (every 50 steps to reduce overhead)
        if global_step % 50 == 0:
            try:
                # Extract programs from outputs
                programs = outputs.get("extras", {}).get("programs", [])
                
                # Store in a global that main() can access
                globals()['last_batch_for_dream'] = {
                    'test_grid': input_grid.detach().cpu(),
                    'task_id': task_id,
                    'programs': programs
                }
            except Exception:
                pass
        
        # Tighter gradient clipping for RelMem stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(topas_model.parameters(), max_norm=0.5)  # Tighter clipping
        
        scaler.step(optimizer)
        scaler.update()
        
        # Apply post-optimizer hooks for RelMem
        if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
            try:
                if hasattr(topas_model.relmem, 'apply_post_optimizer_hooks'):
                    topas_model.relmem.apply_post_optimizer_hooks()
                elif hasattr(topas_model.relmem, 'post_optimizer_step'):
                    topas_model.relmem.post_optimizer_step()
            except Exception as e:
                if global_step % 500 == 0:
                    logging.warning(f"RelMem post-optimizer hooks failed: {e}")

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
    
    # Apply CLI dream settings
    try:
        cli_args = parse_args()
    except Exception:
        cli_args = None

    topas_model, hrm_model = create_models(device, cli_args)

    # CLI args are already applied in create_models, just log the status
    if cli_args and getattr(cli_args, "enable_dream", False):
        dream_status = "enabled" if (hasattr(topas_model, 'dream') and topas_model.dream is not None) else "failed"
        print(f"✅ DreamEngine {dream_status} (micro_ticks={getattr(cli_args, 'dream_micro_ticks', 1)})")

    # store cli_args in trainer scope for later reference
    trainer_cli_args = cli_args
    
    # Optional UKS load
    if cli_args and cli_args.uks_load_path:
        try:
            if hasattr(topas_model, "relmem") and topas_model.relmem is not None:
                topas_model.relmem.load_uks(cli_args.uks_load_path)
                print(f"[UKS] Loaded RelMem state from {cli_args.uks_load_path}")
        except Exception as e:
            print(f"[UKS] Could not load RelMem state: {e}")

    dataset = ARCDataset(
        challenge_file="/mnt/d/Bitterbot/research/topas_v2/ARC-AGI/data/training",
        device=str(device)
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # === Self-play & critique stack ===
    self_play_buffer = SelfPlayBuffer(maxlen=cli_args.selfplay_buffer_size if cli_args else 200)  # used by dopamine rewards
    counterexample_gen = CounterexampleGenerator(device=str(device))  # for nightmare queue
    star_bootstrapper = STaRBootstrapper(topas_model, device=str(device))  # trace generator/validator
    consistency_enforcer = ConsistencyEnforcer(device=str(device))         # optional consistency step
    
    # Run Dream pretrain if requested
    if cli_args and cli_args.dream_pretrain_epochs > 0:
        dream_pretrain_loop(topas_model, dataset, cli_args, device, logger)
        # Load pretrained Dream/ETS
        if os.path.exists('dream_pretrain.pth') and hasattr(topas_model, 'dream') and topas_model.dream:
            try:
                topas_model.dream.load_state('dream_pretrain.pth')
                logger.info("[Main] Loaded pretrained Dream/ETS")
            except Exception as e:
                logger.warning(f"[Main] Failed to load dream pretrain: {e}")
    
    # Self-play buffer initialized above in critique stack

    # Conservative hyperparameters for stable training with RelMem
    optimizer = optim.AdamW(topas_model.parameters(), lr=2e-5, weight_decay=1e-5)  # Even lower LR for stability
    scaler = torch.amp.GradScaler("cuda")  # Fixed FutureWarning

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
    stable_breakthrough_steps = 0  # counts metric windows at/above threshold for curriculum trigger

    for epoch in range(num_epochs):
        # Progressive RelMem bias ramping
        if hasattr(topas_model.config, '_bias_ramp_start') and epoch >= topas_model.config._bias_ramp_start:
            # Ramp up RelMem bias for stronger breakthrough push (but only if planner-aligned success shows up)
            ramp_progress = min(1.0, (epoch - topas_model.config._bias_ramp_start) / 10.0)
            current_bias_w = topas_model.config._bias_base + ramp_progress * (topas_model.config._bias_max - topas_model.config._bias_base)
            recent_successes = sum(1 for v in list(rolling_em)[-10:] if v >= getattr(cli_args, "breakthrough_threshold", 0.33))
            if hasattr(topas_model.config, 'relmem_op_bias_w') and recent_successes >= 3:
                topas_model.config.relmem_op_bias_w = current_bias_w
                print(f"🎯 RelMem bias ramped to {current_bias_w:.3f} (epoch {epoch+1})")
        
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        epoch_metrics = {'exact_match': [], 'accuracy': [], 'mean_iou': [], 'exact_match_refined': []}
        from tqdm import tqdm
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress):
            # Compute metrics every 10 steps
            compute_metrics_now = (global_step % 10 == 0)
            
            result = train_step(
                topas_model, hrm_model, batch,
                optimizer, scaler, device,
                return_metrics=compute_metrics_now,
                global_step=global_step
            )

            # === Dopamine capture & planner priors ===
            if compute_metrics_now and isinstance(result, tuple):
                try:
                    loss, metrics = result
                    em_val = float(metrics.get("exact_match", 0.0))
                    rolling_em.append(em_val)
                    # reconstruct current task from batch for trace gen / buffer
                    demos, test_inputs, test_outputs, task_id = batch
                    if demos and len(demos) > 0:
                        grid_in = demos[0][0].to(device)
                        grid_out = demos[0][1].to(device)
                        task = Task(input=grid_in, output=grid_out, constraints={}, metadata={})
                    else:
                        task = None

                    if em_val >= getattr(cli_args, "breakthrough_threshold", 0.33) and task is not None:
                        logger.info(f"[Breakthrough] EM={em_val:.2%} at step={global_step} → dopamine capture")
                        # 1) Planner prior from successful ops (democratic→peaked)
                        planner_bias = build_op_bias(temp=0.7)
                        # 2) Generate/verify traces (half use planner bias internally)
                        traces = star_bootstrapper.generate_diverse_traces(task, n_traces=8, planner_op_bias=planner_bias)
                        validations = star_bootstrapper.verify_traces(traces, task)
                        good_traces = [t for t, v in zip(traces, validations) if v.is_valid]
                        # 3) Update op priors from successful traces
                        for t in good_traces:
                            if hasattr(t, "operations") and t.operations:
                                op_success_count.update(t.operations)
                        # 4) Dopamine: store solved pair into buffer
                        dopamine_reward(task, self_play_buffer, logger, global_step)
                        # 5) Counterexamples → nightmare queue
                        cex = counterexample_gen.generate_from_failure(task, topas_model, n_counterexamples=5)
                        if cex:
                            recent_failures.extend(cex)
                        # 6) Optional: consistency enforcement across valid traces
                        if len(good_traces) > 1:
                            try:
                                consistency_enforcer.enforce_consistency(good_traces, task)
                            except Exception:
                                pass
                        # 7) Stable-breakthrough counter (for curriculum)
                        stable_breakthrough_steps += 1
                    else:
                        stable_breakthrough_steps = max(0, stable_breakthrough_steps - 1)
                except Exception as e:
                    logger.warning(f"[Dopamine] capture pipeline skipped: {e}")

            # --- Internal monologue (mind-voice) control-plane ---
            try:
                if cli_args and cli_args.monologue_interval > 0 and (global_step % cli_args.monologue_interval == 0):
                    if 'task' not in locals() or task is None:
                        # Build a minimal Task from current batch if not present
                        demos, test_inputs, test_outputs, task_id = batch
                        if demos and len(demos) > 0:
                            grid_in = demos[0][0].to(device)
                            grid_out = demos[0][1].to(device)
                            task = Task(input=grid_in, output=grid_out, constraints={}, metadata={})
                    if task is not None:
                        # Half of traces guided by current op_bias (from dopamine counts)
                        planner_bias = build_op_bias(temp=0.7)
                        traces = star_bootstrapper.generate_diverse_traces(task, n_traces=max(6, cli_args.monologue_min_traces), planner_op_bias=planner_bias)
                        vals = star_bootstrapper.verify_traces(traces, task)
                        valid_traces = [t for t, v in zip(traces, vals) if v.is_valid or v.similarity_score >= 0.90]
                        if len(valid_traces) >= 2:
                            c_res = consistency_enforcer.enforce_consistency(valid_traces, task)
                            monolog_score = c_res['metrics'].overall_consistency if c_res.get('metrics') else 0.0
                            # Use monologue score to steer schedule:
                            target = float(getattr(cli_args, "monologue_consistency_target", 0.85))
                            if monolog_score >= target:
                                # Confidence ↑ : gently ramp RelMem op-bias and reward self-play
                                if hasattr(topas_model.config, 'relmem_op_bias_w'):
                                    topas_model.config.relmem_op_bias_w = min(
                                        getattr(topas_model.config, 'relmem_op_bias_w', 0.2) + 0.02,
                                        getattr(topas_model.config, '_bias_max', 0.5)
                                    )
                                # Nudge self-play weight slightly
                                if hasattr(cli_args, "selfplay_weight"):
                                    cli_args.selfplay_weight = float(cli_args.selfplay_weight + cli_args.monologue_selfplay_bonus)
                            else:
                                # Reasoning shaky → increase nightmare pressure & shorten interval
                                recent = getattr(cli_args, "nightmare_min_interval", 200)
                                cli_args.nightmare_min_interval = max(100, int(0.75 * recent))
                                # Queue a few counterexamples immediately
                                try:
                                    cex = counterexample_gen.generate_from_failure(task, topas_model, n_counterexamples=4)
                                    if cex:
                                        recent_failures.extend(cex)
                                except Exception:
                                    pass
                            logger.info(f"[Monologue] consistency={monolog_score:.3f}, relmem_bias_w={getattr(topas_model.config,'relmem_op_bias_w',None)}")
            except Exception as e:
                logger.debug(f"[Monologue] skipped: {e}")

            # === Curriculum: escalate difficulty when breakthroughs persist ===
            if stable_breakthrough_steps >= 100:
                try:
                    # Use the last seen task to mine deep programs
                    demos, test_inputs, test_outputs, task_id = batch
                    if demos and len(demos) > 0:
                        grid_in = demos[0][0]
                        grid_out = demos[0][1]
                        deep = mine_deep_programs({"test": {"input": grid_in, "output": grid_out}}, max_depth=10)
                        # Only keep exact matches to avoid incorrect targets
                        exact_deep = [dp for dp in deep if dp.get("exact_match")]
                        for _dp in exact_deep:
                            self_play_buffer.buffer.append((grid_in.detach().cpu(), grid_out.detach().cpu()))
                        logger.info(f"[Curriculum] Injected {len(exact_deep)} deep-program exemplars")
                except Exception as e:
                    logger.warning(f"[Curriculum] deep mining failed: {e}")
                finally:
                    stable_breakthrough_steps = 0

            # === Adaptive Nightmare cycle ===
            if len(rolling_em) >= 50 and cli_args:
                fail_rate = 1.0 - (sum(rolling_em) / len(rolling_em))  # higher → worse
                min_iv = int(getattr(cli_args, "nightmare_min_interval", 200))
                max_iv = int(getattr(cli_args, "nightmare_max_interval", 1000))
                # Map fail_rate∈[0,1] → interval [max_iv, min_iv]
                interval = int(max_iv - (max_iv - min_iv) * max(0.0, min(1.0, fail_rate)))
                if interval < min_iv: interval = min_iv
                if global_step % interval == 0 and recent_failures:
                    nightmare_prune(topas_model, recent_failures, optimizer, scaler, device, logger,
                                    global_step, alpha=float(getattr(cli_args, "nightmare_alpha", 0.08)))
            
            # === Self-play training integration (existing) ===
            sp_loss_contribution = 0.0
            if cli_args and cli_args.selfplay_enable and self_play_buffer and global_step % cli_args.selfplay_interval == 0:
                try:
                    # Generate new puzzles from current batch
                    demos, test_inputs, test_outputs, task_id = batch
                    current_demos = [(test_inputs, test_outputs)] if test_inputs is not None and test_outputs is not None else []
                    
                    if current_demos:
                        new_puzzles = self_play_buffer.generate_batch(
                            current_demos, 
                            getattr(topas_model, 'wormhole', None), 
                            top_k=cli_args.selfplay_topk
                        )
                        if new_puzzles:
                            print(f"[SelfPlay] Generated {len(new_puzzles)} puzzles at step={global_step}")
                        else:
                            print(f"[SelfPlay] No puzzles generated (fallback used) at step={global_step}")
                except Exception as e:
                    import traceback
                    logging.getLogger(__name__).exception("[SelfPlay] failure: %s", e)
                
                # Sample and compute self-play loss
                sp_samples = self_play_buffer.sample_batch(4)
                if sp_samples:
                    print(f"[SelfPlay] Training on {len(sp_samples)} puzzles")
                    for sp_input, sp_target in sp_samples:
                        try:
                            sp_output = topas_model.forward_pretraining(sp_input.unsqueeze(0), target_shape=sp_target.shape[-2:])
                            if 'logits' in sp_output:
                                sp_loss = F.cross_entropy(sp_output['logits'].view(-1, 10), sp_target.view(-1).long().clamp(0, 9))
                                sp_loss_contribution += sp_loss * cli_args.selfplay_weight
                        except Exception:
                            continue
                    if sp_loss_contribution > 0:
                        print(f"[SelfPlay] applied sp_loss={float(sp_loss_contribution.item()):.4f}")
                        # Simple reinforcement to RelMem
                        try:
                            if hasattr(topas_model, "relmem") and topas_model.relmem is not None:
                                if float(sp_loss_contribution.item()) < 1.0:
                                    topas_model.relmem.queue_hebbian_update("pattern", sid=0, oid=0, eta=0.05)
                                else:
                                    topas_model.relmem.add_exception(0, "has_attr", 0)
                        except Exception:
                            pass
                else:
                    print(f"[SelfPlay] No samples available for training")
            
            if result is not None:
                if compute_metrics_now and isinstance(result, tuple):
                    loss, metrics = result
                    # Add self-play contribution to loss
                    if sp_loss_contribution > 0:
                        loss = loss + sp_loss_contribution
                    epoch_losses.append(loss)
                    for k, v in metrics.items():
                        epoch_metrics[k].append(v)
                else:
                    loss = result
                    # Add self-play contribution to loss
                    if sp_loss_contribution > 0:
                        loss = loss + sp_loss_contribution
                    if loss is not None:
                        epoch_losses.append(loss)
            
            # Enhanced RelMem stats logging
            relmem_log_interval = getattr(trainer_cli_args, 'relmem_log_interval', 200) if 'trainer_cli_args' in globals() else 200
            if global_step % relmem_log_interval == 0:
                try:
                    relmem_stats = {}
                    if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
                        if hasattr(topas_model.relmem, 'get_stats'):
                            relmem_stats = topas_model.relmem.get_stats()
                        elif hasattr(topas_model.relmem, 'stats'):
                            relmem_stats = topas_model.relmem.stats()
                        else:
                            # Enhanced basic stats collection
                            relmem_stats = {
                                'concepts_count': getattr(topas_model.relmem, 'concepts_count', 0),
                                'relations_count': len(getattr(topas_model.relmem, 'relations', [])),
                                'last_binding_success': getattr(topas_model.relmem, 'last_binding_success', False),
                                'regularization_strength': getattr(topas_model.relmem, 'regularization_strength', 0.0),
                                'active_concepts': getattr(topas_model.relmem, 'active_concepts', 0)
                            }
                    
                    if relmem_stats:
                        stats_str = ', '.join([f"{k}={v}" for k, v in relmem_stats.items()])
                        logging.info(f"[Step {global_step}] RelMem: {stats_str}")
                        
                except Exception as e:
                    if global_step % 1000 == 0:  # Less frequent warning
                        logging.warning(f"RelMem stats logging failed: {e}")
            
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

        # ---- Epoch end: refinement + optional UKS save ----
        try:
            if hasattr(topas_model, "relmem") and topas_model.relmem is not None:
                topas_model.relmem.refinement_step()
                if trainer_cli_args and trainer_cli_args.uks_save_path:
                    topas_model.relmem.save_uks(trainer_cli_args.uks_save_path)
                    if epoch % 10 == 0:
                        print(f"[UKS] Saved RelMem state to {trainer_cli_args.uks_save_path}")
        except Exception as e:
            pass
        
        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            summary = f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}"
            
            # ---- Logging: include RelMem stats ----
            try:
                if hasattr(topas_model, "relmem") and topas_model.relmem is not None:
                    relmem_info = topas_model.relmem.stats()
                    if relmem_info:
                        summary += f" | RelMem active={int(relmem_info.get('relmem_active', 0))} "
                        summary += f"depth={relmem_info.get('relmem_depth', 0):.2f} "
                        summary += f"exceptions={int(relmem_info.get('relmem_exceptions', 0))}"
            except Exception:
                pass

            # --- Wormhole consolidation ---
            try:
                if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole"):
                    consolidator = getattr(topas_model.dream.wormhole, "consolidator", None)
                    if consolidator is not None:
                        # Consolidate programs collected during the epoch
                        # Assume extras logged some programs; you can adapt source
                        all_programs = []
                        if hasattr(topas_model, "task_history"):
                            for tid, perf in topas_model.task_history.items():
                                if "programs" in perf:
                                    all_programs.extend(perf["programs"])
                        if all_programs:
                            new_templates = consolidator.consolidate(all_programs, top_k=20)
                            logging.getLogger(__name__).info(
                                f"[Wormhole] Consolidated {len(new_templates)} templates at epoch {epoch+1}"
                            )
            except Exception as e:
                logging.getLogger(__name__).warning(f"[Wormhole] consolidation failed: {e}")
            
            if len(epoch_metrics['exact_match']) > 0:
                avg_em = sum(epoch_metrics['exact_match']) / len(epoch_metrics['exact_match'])
                avg_acc = sum(epoch_metrics['accuracy']) / len(epoch_metrics['accuracy'])
                avg_iou = sum(epoch_metrics['mean_iou']) / len(epoch_metrics['mean_iou'])
                summary += f", EM={avg_em:.2%}, acc={avg_acc:.2%}, IoU={avg_iou:.3f}"
                
                # Add EBR refined exact match if available
                if len(epoch_metrics['exact_match_refined']) > 0:
                    avg_em_refined = sum(epoch_metrics['exact_match_refined']) / len(epoch_metrics['exact_match_refined'])
                    summary += f", EM_ebr={avg_em_refined:.2%}"
                
                # Track best metrics and save checkpoints
                if avg_em > best_em:
                    best_em = avg_em
                    print(f"🎯 New best EM: {best_em:.2%}")
                    
                    # Save best EM checkpoint with metric in filename
                    best_em_filename = f"best_em_{best_em*100:.1f}.pt"
                    torch.save(topas_model.state_dict(), best_em_filename)
                    print(f"💾 Saved best EM checkpoint: {best_em_filename}")
                    
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    print(f"🎯 New best accuracy: {best_acc:.2%}")
                    
                    # Save best accuracy checkpoint with metric in filename
                    best_acc_filename = f"best_acc_{best_acc*100:.1f}.pt"
                    torch.save(topas_model.state_dict(), best_acc_filename)
                    print(f"💾 Saved best accuracy checkpoint: {best_acc_filename}")
                
                # Periodic evaluation checkpoints (every eval_interval epochs)
                if cli_args and hasattr(cli_args, 'eval_interval') and (epoch + 1) % cli_args.eval_interval == 0:
                    eval_filename = f"eval_epoch_{epoch+1}_em_{avg_em*100:.1f}.pt"
                    torch.save(topas_model.state_dict(), eval_filename)
                    print(f"📊 Saved evaluation checkpoint: {eval_filename}")
            
            print(summary)

        # === DREAM: full offline consolidation on schedule ===
        try:
            if trainer_cli_args:
                full_every = getattr(trainer_cli_args, "dream_full_every", 0)
                if full_every and ((epoch + 1) % int(full_every) == 0):
                    # Check we enabled dream in config
                    try:
                        enabled = bool(getattr(topas_model.config, "enable_dream", False))
                    except Exception:
                        enabled = False

                    if enabled:
                        timeout = int(getattr(trainer_cli_args, "dream_full_timeout", 600))
                        bg = bool(getattr(trainer_cli_args, "dream_background", False))
                        force_cpu = bool(getattr(trainer_cli_args, "dream_force_cpu", False))
                        logging.info("[Dream-Trainer] Triggering full dream cycle (epoch %d)", epoch+1)
                        # epoch boundary: build canonical 1152-dim dream tokens and run dream cycle
                        stats = None
                        try:
                            cached_batch = globals().get('last_batch_for_dream', None)
                            outputs = topas_model.forward_pretraining(
                                cached_batch['test_grid'].to(device),
                                target_shape=cached_batch['test_grid'].shape[-2:]
                            ) if cached_batch is not None else None

                            hemi_stats = {}
                            if outputs is not None and "brain" in outputs and "slot_vecs" in outputs:
                                brain = outputs["brain"]         # [B, 1152] (symbolic/global)
                                slot_vecs = outputs["slot_vecs"] # [B, T, 512] (perceptual/objects)
                                B, T, _ = slot_vecs.shape
                                # Left: symbolic (brain expanded to slots)
                                left_tokens  = brain.unsqueeze(1).expand(B, T, -1)
                                # Right: perceptual (use slot vectors; if Dream expects ctrl_dim, pad w/ zeros)
                                if left_tokens.size(-1) != slot_vecs.size(-1):
                                    pad = torch.zeros(B, T, left_tokens.size(-1) - slot_vecs.size(-1), device=slot_vecs.device)
                                    right_tokens = torch.cat([slot_vecs, pad], dim=-1)
                                else:
                                    right_tokens = slot_vecs
                                # Run both cycles
                                hemi_stats['left']  = topas_model.run_dream_cycle(tokens=left_tokens,  demos_programs=outputs.get("extras", {}).get("programs") if outputs else None)
                                hemi_stats['right'] = topas_model.run_dream_cycle(tokens=right_tokens, demos_programs=outputs.get("extras", {}).get("programs") if outputs else None)
                                # Choose a winner (prefer refined EM if present)
                                def score(s): 
                                    if not isinstance(s, dict): return 0.0
                                    return float(s.get("exact_match_refined") or s.get("em_ebr") or s.get("EM_ebr") or 0.0)
                                left_score, right_score = score(hemi_stats['left']), score(hemi_stats['right'])
                                winner = 'left' if left_score >= right_score else 'right'
                                logging.info(f"[Dream-UniHemi] left={left_score:.3f} right={right_score:.3f} → winner={winner}")
                                # Small, bounded nudge to priors based on winner
                                if hasattr(topas_model.config, 'relmem_op_bias_w'):
                                    delta = 0.02 if winner == 'left' else -0.01
                                    topas_model.config.relmem_op_bias_w = float(
                                        max(0.15, min(getattr(topas_model.config, 'relmem_op_bias_w', 0.2) + delta,
                                                      getattr(topas_model.config, '_bias_max', 0.5)))
                                    )
                                # Optionally skew planner op_bias success counts slightly
                                if winner == 'left':
                                    op_success_count.update(["planner_align_bonus"])
                                else:
                                    op_success_count.update(["percept_align_bonus"])
                                stats = hemi_stats[winner]
                            else:
                                # fallback: single-hemisphere like before
                                dream_tokens = getattr(topas_model, "_dream_tokens", None)
                                stats = topas_model.run_dream_cycle(tokens=dream_tokens, demos_programs=outputs.get("extras", {}).get("programs") if outputs else None)
                        except Exception as e:
                            logging.exception("[Dream] Full cycle failed: %s", e)
                        # If stats contains EM or other metrics, push into epoch_metrics for visibility
                        if isinstance(stats, dict):
                            em_ebr = stats.get("exact_match_refined") or stats.get("em_ebr") or stats.get("EM_ebr")
                            if em_ebr is not None:
                                epoch_metrics.setdefault('exact_match_refined', []).append(float(em_ebr))
                            # log other stats
                            logging.info("[Dream-Trainer] Full dream stats keys: %s", list(stats.keys()))
                        
                        # Generate self-play puzzles after dream cycle
                        if self_play_buffer and cli_args and cli_args.selfplay_enable:
                            try:
                                # Sample recent training examples for transformation
                                recent_samples = []
                                sample_count = 0
                                for batch_idx, batch in enumerate(dataloader):
                                    if sample_count >= cli_args.selfplay_topk:
                                        break
                                    try:
                                        demos, test_inputs, test_outputs, task_id = batch
                                        if test_inputs is not None and test_outputs is not None:
                                            recent_samples.append((test_inputs, test_outputs))
                                            sample_count += 1
                                    except:
                                        continue
                                
                                if recent_samples and hasattr(topas_model, 'wormhole'):
                                    new_puzzles = self_play_buffer.generate_from_wormhole(
                                        recent_samples, 
                                        topas_model.wormhole,
                                        themes=getattr(topas_model.dream, 'theme', None) if hasattr(topas_model, 'dream') else None,
                                        top_k=cli_args.selfplay_topk
                                    )
                                    if new_puzzles:
                                        print(f"🎮 Generated {len(new_puzzles)} self-play puzzles (buffer: {len(self_play_buffer.buffer)})")
                                        
                            except Exception as e:
                                print(f"⚠️  Self-play generation failed: {e}")
                    else:
                        logging.info("[Dream-Trainer] Dream disabled in model config; skipping full cycle.")
        except Exception as e:
            import traceback
            logging.warning("[Dream-Trainer] Dream scheduling failed: %s", traceback.format_exc())

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