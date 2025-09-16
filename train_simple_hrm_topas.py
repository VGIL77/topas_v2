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
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import math
from trainers.arc_dataset_loader import ARCDataset
try:
    from arc2_dataset_loader import ARC2Dataset
except ImportError:
    ARC2Dataset = None
from models.topas_arc_60M import TopasARC60M, ModelConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from trainers.self_play import SelfPlayBuffer  # used for storing dopamine rewards
from trainers.self_critique.counterexamples import CounterexampleGenerator, Task  # Task wrapper + counterexamples
from trainers.self_critique.star_bootstrapper import STaRBootstrapper           # STaR trace gen + verification
from trainers.self_critique.consistency import ConsistencyEnforcer               # enforce consistency across valid traces
from trainers.augmentation.deep_program_discoverer import mine_deep_programs                  # deep DSL programs miner (6–10 ops)
from models.policy_nets import OpPolicyNet, op_logits_to_bias                              # policy-guided search
from collections import Counter, deque
from typing import Callable
import math, statistics, time
import numpy as np
import random
import hashlib   # NEW

# Alpha-ARC X additions
try:
    from trainers.puct_search import puct_search
except Exception:
    puct_search = None
try:
    from trainers.replay import PrioritizedReplay
except Exception:
    PrioritizedReplay = None
try:
    from trainers.near_miss import near_miss_repair
except Exception:
    near_miss_repair = None

def _canonical_puzzle_id(task_id, modulo: int = 1000) -> int:
    """
    Map any task identifier (int/str/uuid) to a stable int in [0, modulo).
    Matches HRM config num_puzzle_identifiers=1000.
    """
    try:
        return int(task_id) % modulo
    except Exception:
        h = int(hashlib.sha1(str(task_id).encode("utf-8")).hexdigest()[:8], 16)
        return h % modulo

# =========================
# Dopamine & Nightmare Core
# =========================

# Global state shared across training
op_success_count = Counter()          # track operations in successful traces (for planner op_bias)
recent_failures: List[Any] = []       # queue of failed counterexamples for nightmares
rolling_em = deque(maxlen=200)        # rolling window of EM to estimate failure pressure

# =========================
# Additional Dopamine Helpers (Production-Grade)
# =========================

def _extract_entropy_reduction(dream_stats) -> float:
    try:
        if isinstance(dream_stats, dict):
            if "entropy_reduction" in dream_stats:
                return float(dream_stats["entropy_reduction"])
            if "entropy_before" in dream_stats and "entropy_after" in dream_stats:
                eb = float(dream_stats["entropy_before"]); ea = float(dream_stats["entropy_after"])
                return max(0.0, eb - ea)
    except Exception:
        pass
    return 0.0

def _extract_mdl_gain(mined_templates) -> float:
    try:
        if not mined_templates:
            return 0.0
        gains = []
        for t in mined_templates:
            if isinstance(t, dict) and "mdl_gain" in t:
                gains.append(float(t["mdl_gain"]))
            else:
                gains.append(1.0)
        return float(sum(gains))
    except Exception:
        return 0.0

def _novelty_estimate(enc_inp: Tuple[Tuple[int,int], Tuple[int,...]], buffer, k: int = 64) -> float:
    try:
        (_, fa) = enc_inp
        fa = list(fa)
        if not hasattr(buffer, "buffer") or len(buffer.buffer) == 0:
            return 1.0
        sample = buffer.buffer[-k:] if len(buffer.buffer) > k else buffer.buffer
        def sim(a, b):
            enc_b = b[0]   # (enc_inp, enc_out) OR (enc_inp, enc_out, score)
            (_, fb) = enc_b
            n = min(len(fa), len(fb))
            if n == 0: return 0.0
            eq = sum(1 for i in range(n) if fa[i] == fb[i])
            return eq / n
        mx = 0.0
        for item in sample:
            try:
                mx = max(mx, sim(enc_inp, item))
            except Exception:
                continue
        return max(0.0, 1.0 - mx)
    except Exception:
        return 0.5

def _squash(x: float, temp: float = 1.0) -> float:
    try:
        return math.tanh(x / max(1e-6, temp))
    except Exception:
        return 0.0

def _dopamine_score(em: float, acc: float, iou: float, program_len: int,
                    entropy_red: float, mdl_gain: float, novelty: float,
                    Lmax: int = 12) -> Tuple[float, Dict[str, float]]:
    w_em, w_acc, w_iou = 1.0, 0.25, 0.5
    w_len, w_ent, w_mdl, w_nov = -0.10, 0.30, 0.50, 0.20
    L = min(max(0, program_len), Lmax)
    len_pen = L / max(1, Lmax)
    raw = (w_em * em) + (w_acc * acc) + (w_iou * iou) + (w_ent * entropy_red) \
          + (w_mdl * mdl_gain) + (w_nov * novelty) + (w_len * len_pen)
    R = _squash(raw, temp=1.0)
    comps = dict(em=em, acc=acc, iou=iou, len=L, len_pen=len_pen,
                 entropy_red=entropy_red, mdl_gain=mdl_gain, novelty=novelty,
                 raw=raw, squashed=R)
    return R, comps

def _stringify_ops(ops: Any) -> List[str]:
    out = []
    if not ops:
        return out
    for op in ops:
        if isinstance(op, dict):
            try:
                key = f"composite_{hash(str(sorted(op.items())))}"
            except Exception:
                key = f"composite_{hash(str(op))}"
            out.append(key)
        else:
            out.append(str(op))
    return out

def _hamming(a: torch.Tensor, b: torch.Tensor) -> int:
    """
    Compute Hamming distance between two tensors.
    """
    if a is None or b is None or a.shape != b.shape:
        return 10**9
    return (a.view(-1) != b.view(-1)).sum().item()

def _sc_run_star(star_bootstrapper, task, planner_bias, n: int) -> list:
    """
    Self-consistency booster: run STaR N times with small jitter in priors.
    Return list of valid traces.
    """
    traces_all = []
    import copy, random
    for _ in range(max(1, n)):
        pb = None
        if isinstance(planner_bias, dict):
            # tiny jitter (root-Dirichlet-like) for diversity
            pb = {k: max(1e-6, v * (0.9 + 0.2 * random.random())) for k, v in planner_bias.items()}
        tr = star_bootstrapper.generate_diverse_traces(task, n_traces=8, planner_op_bias=pb or planner_bias)
        traces_all.extend(tr)
    return traces_all

def safe_replay_add(buffer, ops, priority, logger=None):
    """Safely stringify ops and add them to replay buffer."""
    if buffer is None or not ops:
        return
    try:
        safe_ops = _stringify_ops(ops)
        if safe_ops:
            buffer.add(safe_ops, priority)
    except Exception as e:
        if logger: logger.warning(f"[Replay] add failed for ops={ops}: {e}")

def _puct_plan_stepwise(model, demos, test_input, target_grid, cli_args, device) -> list:
    """
    Compose a program by running PUCT selection sequentially for puct_depth steps.
    Each step: run small PUCT on current grid to pick next op, apply via DSL shim.
    """
    if puct_search is None:
        return []
    state_grid = test_input.clone()
    program_ops = []

    def priors_fn(grid_s):
        # one forward to get policy priors (dsl_op_logits) on this state
        try:
            out = model.forward_pretraining(grid_s.unsqueeze(0), target_shape=target_grid.shape[-2:], demos=demos)
            extras = out.get("extras", {}) if isinstance(out, dict) else {}
            logits = extras.get("dsl_op_logits") or extras.get("policy_logits")
            if logits is None: return {}
            probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy().tolist()
            from models.dsl_registry import DSL_OPS
            prior = {op: float(p) for op, p in zip(DSL_OPS, probs)}
            # root Dirichlet
            import numpy as np
            if len(prior) and cli_args.root_dirichlet_eps > 0:
                eps = cli_args.root_dirichlet_eps
                alpha = cli_args.root_dirichlet_alpha
                noise = np.random.dirichlet([alpha]*len(prior))
                keys = list(prior.keys())
                for i, k in enumerate(keys):
                    prior[k] = float((1-eps)*prior[k] + eps*noise[i])
            return prior
        except Exception:
            return {}

    def value_fn(grid_s):
        try:
            out = model.forward_pretraining(grid_s.unsqueeze(0), target_shape=target_grid.shape[-2:], demos=demos)
            v = out.get("extras", {}).get("value_logit")
            return torch.sigmoid(v)[0].item() if v is not None else 0.1
        except Exception:
            return 0.1

    def expand_fn(grid_s, op):
        try:
            return model.dsl.apply(op, grid_s)
        except Exception:
            return grid_s

    for _ in range(max(1, int(cli_args.puct_depth))):
        best_op, _ = puct_search(
            state_grid, priors_fn, value_fn, expand_fn,
            max_nodes=int(cli_args.puct_nodes),
            c_puct=float(cli_args.c_puct)
        )
        try:
            # Always stringify ops before appending
            safe_ops = _stringify_ops([best_op]) if best_op is not None else []
            if safe_ops:
                program_ops.extend(safe_ops)
            state_grid = expand_fn(state_grid, best_op)
        except Exception:
            continue
    return program_ops
    return out

# =========================
# Dopamine state (EMA + refractory)
# =========================
class _EMA:
    def __init__(self, beta=0.9, init=0.0):
        self.beta = beta
        self.m = init
        self.initialized = False
    def update(self, x: float) -> float:
        if not self.initialized:
            self.m = x
            self.initialized = True
        else:
            self.m = self.beta * self.m + (1 - self.beta) * x
        return self.m
    def value(self) -> float:
        return self.m

_dopamine_ema = _EMA(beta=0.9, init=0.0)
_last_dopamine_step = -10**9  # refractory tracking

# =========================
# Phase 6: PriorNet for Neural DSL Operation Priors
# =========================

class PriorNet(nn.Module):
    """Neural network for learning adaptive DSL operation priors."""

    def __init__(self, input_dim=128, hidden=[256, 128, 64], num_ops=41):
        super().__init__()
        layers = []
        last_dim = input_dim

        for h in hidden:
            layers.extend([nn.Linear(last_dim, h), nn.ReLU()])
            last_dim = h

        layers.append(nn.Linear(last_dim, num_ops))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================
# Enhanced Dopamine Helper Functions
# =========================
def _extract_program_len(programs: List) -> int:
    """Extract representative program length from operations list."""
    if not programs:
        return 0
    total_len = 0
    count = 0
    for prog in programs:
        if isinstance(prog, (list, tuple)):
            total_len += len(prog)
            count += 1
        elif hasattr(prog, '__len__'):
            total_len += len(prog)
            count += 1
    return total_len // max(1, count)

def _safe_iou(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> float:
    """Safely compute IoU between prediction and target grids."""
    try:
        if pred_grid.shape != target_grid.shape:
            return 0.0
        pred_flat = pred_grid.view(-1)
        target_flat = target_grid.view(-1)
        intersection = (pred_flat == target_flat).sum().float()
        union = pred_flat.numel()
        return (intersection / union).item() if union > 0 else 0.0
    except Exception:
        return 0.0

def _extract_entropy_reduction(dream_stats: Optional[Dict]) -> float:
    """Extract entropy reduction metric from dream statistics."""
    if not isinstance(dream_stats, dict):
        return 0.0
    return float(dream_stats.get('entropy_reduction', 0.0))

def _extract_mdl_gain(mined_templates: List) -> float:
    """Extract MDL gain from mined program templates."""
    if not mined_templates:
        return 0.0
    # Simple heuristic: more complex templates = higher MDL gain
    total_complexity = sum(len(str(t)) for t in mined_templates)
    return min(1.0, total_complexity / 100.0)

def _novelty_estimate(encoded_input: tuple, buffer: Any, k: int = 64) -> float:
    """Estimate novelty of input relative to buffer contents."""
    if not hasattr(buffer, 'buffer') or len(buffer.buffer) < k:
        return 1.0  # High novelty if buffer sparse
    try:
        # Count similar patterns in recent buffer
        recent = buffer.buffer[-k:]
        matches = sum(1 for inp, _ in recent if inp == encoded_input)
        return max(0.0, 1.0 - matches / k)
    except Exception:
        return 0.5

def _dopamine_score(em: float, acc: float, iou: float, program_len: int, 
                   entropy_red: float, mdl_gain: float, novelty: float, Lmax: int = 12) -> tuple:
    """
    Multi-factor dopamine scoring with component breakdown.
    Returns: (total_score, components_dict)
    """
    # Core performance (0.6 weight)
    perf_score = 0.4 * em + 0.3 * acc + 0.3 * iou
    
    # Program efficiency bonus (0.2 weight) 
    eff_score = max(0.0, 1.0 - program_len / Lmax) if program_len > 0 else 0.0
    
    # Learning signal (0.2 weight)
    learn_score = 0.5 * entropy_red + 0.3 * mdl_gain + 0.2 * novelty
    
    total = 0.6 * perf_score + 0.2 * eff_score + 0.2 * learn_score
    
    components = {
        'perf': perf_score, 'eff': eff_score, 'learn': learn_score,
        'em': em, 'acc': acc, 'iou': iou, 'prog_len': program_len,
        'ent_red': entropy_red, 'mdl': mdl_gain, 'novel': novelty
    }
    
    return total, components

# =========================
# Canonical grid encoding
# =========================
def _encode_grid_tensor(grid: torch.Tensor) -> tuple:
    """
    Canonical, hashable encoding for ARC grids.
    Returns: ((H, W), tuple(flat_int_values))
    Fixed: Ensure complete detachment from GPU and any unhashable references
    """
    if isinstance(grid, torch.Tensor):
        # Ensure complete detachment and conversion to CPU
        g = grid.detach().cpu().clone().long()
        if g.dim() == 3 and g.size(0) == 1:
            g = g.squeeze(0)
        assert g.dim() == 2, f"Expected [H,W], got {tuple(g.shape)}"
        H, W = g.shape
        # Ensure all values are plain Python ints, not tensor scalars
        flat_values = []
        for x in g.view(-1):
            flat_values.append(int(x.item()))  # Use .item() to extract scalar value
        
        # Create the final tuple and verify it's hashable before returning
        result = ((int(H), int(W)), tuple(flat_values))
        try:
            hash(result)  # Verify hashability
        except TypeError as e:
            raise ValueError(f"Generated unhashable grid encoding: H={H}, W={W}, values_type={type(flat_values)}, error={e}")
        
        return result
    # already encoded?
    if isinstance(grid, tuple) and len(grid) == 2 and isinstance(grid[0], tuple):
        return grid
    raise TypeError(f"Unsupported grid type for encoding: {type(grid)}")

def _decode_grid(enc: tuple) -> torch.Tensor:
    """
    Decode canonical grid encoding back to torch.LongTensor [H,W].
    """
    (H, W), flat = enc
    arr = np.array(flat, dtype=np.int64).reshape(H, W)
    return torch.from_numpy(arr).long()

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

def build_policy_guided_bias(grid_in: torch.Tensor, grid_out: torch.Tensor, 
                           op_policy: Optional[Any], device, temp: float = 0.7):
    """
    Enhanced op-bias that combines historical success counts with policy net predictions.
    Returns hybrid bias: 50% historical + 50% policy-guided.
    """
    # Start with historical bias (proven successful)
    historical_bias = build_op_bias(temp)
    
    # Add policy-guided bias if available
    if op_policy is not None:
        try:
            # Build minimal context for policy prediction  
            # Ensure grid is properly formatted [B, H, W] for policy
            if grid_in.dim() == 4:  # [B, C, H, W] → remove channel dim if present
                policy_grid = grid_in.squeeze(1) if grid_in.size(1) == 1 else grid_in[:, 0]
            elif grid_in.dim() == 3:  # [B, H, W] or [C, H, W]
                if grid_in.size(0) == 1:  # [1, H, W]
                    policy_grid = grid_in
                else:  # [C, H, W] → [1, H, W]  
                    policy_grid = grid_in[0:1]
            elif grid_in.dim() == 2:  # [H, W] → [1, H, W]
                policy_grid = grid_in.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected grid dimensions: {grid_in.shape}")
                
            B, H, W = policy_grid.shape
            rel_features = None  # Will use zeros fallback in policy
            size_oracle = torch.tensor([[H, W, H, W]], device=device).float()
            theme_priors = torch.zeros(1, 10, device=device)
            
            # Get policy prediction with properly shaped grid
            with torch.no_grad():
                pred = op_policy(policy_grid, rel_features, size_oracle, theme_priors, program_ops=[])
                raw_policy_bias = op_logits_to_bias(pred.op_logits)
                
                # Ensure policy_bias is a proper dict (robust type checking)
                if isinstance(raw_policy_bias, dict):
                    policy_bias = raw_policy_bias
                elif hasattr(raw_policy_bias, 'keys'):
                    policy_bias = dict(raw_policy_bias)  # Convert dict-like to dict
                else:
                    # Fallback: create uniform policy bias if conversion fails
                    policy_bias = {f"op_{i}": 1.0/41 for i in range(41)}
            
            # Hybrid: 50% historical + 50% policy-guided (with safe key access)
            hybrid_bias = {}
            all_ops = set(historical_bias.keys()) | set(policy_bias.keys())
            for op in all_ops:
                hist_val = historical_bias.get(op, 1.0 / len(all_ops))
                policy_val = policy_bias.get(op, 1.0 / len(all_ops))
                hybrid_bias[op] = 0.5 * hist_val + 0.5 * policy_val
                
            return hybrid_bias
        except Exception as e:
            logging.getLogger(__name__).warning(f"[Policy] guided bias failed: {e}")
    
    # Fallback to historical only
    return historical_bias

def dopamine_reward(task, buffer, logger, global_step, score: float = 1.0, components: Dict = None):
    """
    Enhanced dopamine capture with importance scoring:
    - Always store as canonical hashable tuples: ((H,W), tuple(flattened))
    - Enriched buffer with (enc_in, enc_out, score) for importance replay
    - Bypass ALL unhashable dict/Tensor issues
    - Log buffer growth and reward details
    """
    if buffer is None:
        return
    try:
        inp_t = task['input'] if isinstance(task, dict) else getattr(task, 'input', None)
        out_t = task['output'] if isinstance(task, dict) else getattr(task, 'output', None)
        if inp_t is None or out_t is None:
            raise ValueError("dopamine_reward: task missing input/output")
        # Ensure [H,W] tensors for encoding
        if isinstance(inp_t, torch.Tensor) and inp_t.dim() == 3 and inp_t.size(0) == 1:
            inp_t = inp_t.squeeze(0)
        if isinstance(out_t, torch.Tensor) and out_t.dim() == 3 and out_t.size(0) == 1:
            out_t = out_t.squeeze(0)
        enc_inp = _encode_grid_tensor(inp_t)
        enc_out = _encode_grid_tensor(out_t)
        
        # Verify encodings are truly hashable before storage
        try:
            hash(enc_inp)
            hash(enc_out)
            hash(score)  # Ensure score is also hashable
        except TypeError as hash_err:
            raise ValueError(f"Generated unhashable encoding: enc_inp={type(enc_inp)}, enc_out={type(enc_out)}, score={type(score)}, error={hash_err}")
        
        # Enhanced buffer storage with importance score
        if score > 0:
            buffer.buffer.append((enc_inp, enc_out, score))
        else:
            buffer.buffer.append((enc_inp, enc_out))  # Fallback to old format
            
        if logger:
            # Safely format components to avoid any unhashable issues in logging
            comp_str = ""
            if components:
                try:
                    # Convert components dict to a safe string representation
                    safe_components = {k: float(v) if hasattr(v, '__float__') else str(v) 
                                     for k, v in components.items() if v is not None}
                    comp_str = f" components={safe_components}"
                except Exception as comp_err:
                    comp_str = f" components=<error: {comp_err}>"
            logger.info(f"[Dopamine] Stored pair (score={score:.3f}) → buffer size={len(buffer.buffer)} at step {global_step}{comp_str}")
    except Exception as e:
        if logger:
            logger.warning(f"[Dopamine] capture pipeline skipped at step {global_step}: {e}")

def _as_star_task(task):
    """
    Normalize any input into a proper Task dataclass.
    Guarantees .input/.output attributes for downstream (STaR, dopamine, Wormhole).
    """
    if task is None:
        return None

    # Already Task
    if isinstance(task, Task):
        return task

    # Dict-like
    if isinstance(task, dict):
        return Task(
            input=task.get("input") or task.get("test"),
            output=task.get("output") or task.get("target"),
            constraints=task.get("constraints", {}),
            metadata=task.get("metadata", {})
        )

    # Tuple/list
    if isinstance(task, (list, tuple)) and len(task) >= 2:
        return Task(input=task[0], output=task[1], constraints={}, metadata={})

    raise TypeError(f"Unsupported task type: {type(task)}")

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
    parser.add_argument("--eval-beam-width", type=int, default=None,
                        help="Beam width for evaluation beam search (overrides model config).")
    parser.add_argument("--ebr-steps", type=int, default=None,
                        help="Number of EBR refinement steps (overrides model config).")
    
    # Training args
    parser.add_argument("--max-steps", type=int, default=60000,
                        help="Maximum training steps")
    parser.add_argument("--bucket-unlock-patience", type=int, default=0,
                        help="Epochs with no EM improvement before unlocking next task bucket (0 disables).")
    parser.add_argument("--dataset", type=str, default="arc1", choices=["arc1", "arc2"],
                        help="Dataset to use: arc1 (original training) or arc2 (ARC Prize 2025)")

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

    # Alpha-ARC X Neural-Guided Search 2.0 parameters
    parser.add_argument("--search-alg", type=str, default="beam", choices=["beam", "puct"],
                        help="Search algorithm: beam or puct")
    parser.add_argument("--puct-nodes", type=int, default=100,
                        help="Number of PUCT search nodes")
    parser.add_argument("--c-puct", type=float, default=1.414,
                        help="PUCT exploration constant")
    parser.add_argument("--puct-depth", type=int, default=8,
                        help="Maximum PUCT search depth")
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha for root node")
    parser.add_argument("--root-dirichlet-eps", type=float, default=0.25,
                        help="Dirichlet noise epsilon for root node")
    parser.add_argument("--sc-star", action="store_true", default=False,
                        help="Enable Self-Critique STaR wrapper")
    parser.add_argument("--near-miss-hamming-pct", type=float, default=0.15,
                        help="Near-miss Hamming distance threshold percentage")
    parser.add_argument("--replay-cap", type=int, default=2000,
                        help="Replay buffer capacity")

    # Phase 2: Meta-learning (MAML-lite inner/outer loop)
    parser.add_argument("--enable-meta-loop", action="store_true", default=False,
                        help="Enable MAML-lite inner/outer meta-learning loop")
    parser.add_argument("--meta-inner-steps", type=int, default=1,
                        help="Inner-loop adaptation steps per meta-task")
    parser.add_argument("--meta-tasks-per-batch", type=int, default=4,
                        help="Number of tasks per meta-batch")
    parser.add_argument("--meta-adapt-lr", type=float, default=5e-5,
                        help="Learning rate for inner-loop adaptation")

    # Phase 6: Neuro-priors (PriorNet for DSL op bias evolution)
    parser.add_argument("--enable-priornet", action="store_true", default=False,
                        help="Enable PriorNet for DSL op bias learning")
    parser.add_argument("--priornet-hidden", type=str, default="256,128,64",
                        help="Comma-separated hidden layer sizes for PriorNet")
    parser.add_argument("--priornet-reg-weight", type=float, default=0.01,
                        help="Regularization weight for PriorNet outputs")

    args, _unknown = parser.parse_known_args()
    return args

# =========================
# Phase 2: Meta-Learning (MAML-lite)
# =========================

def run_meta_loop(topas_model, hrm_model, dataset, cli_args, device):
    """MAML-lite meta-learning loop for fast adaptation."""
    import numpy as np
    from torch import optim
    from models.topas_arc_60M import TopasARC60M  # import your model class

    # Sample meta-batch of tasks
    tasks = [dataset[i] for i in np.random.choice(len(dataset), cli_args.meta_tasks_per_batch)]
    meta_grads = [{n: torch.zeros_like(p) for n, p in topas_model.named_parameters() if p.requires_grad}]

    for task in tasks:
        # Create inner model by re-instantiating + loading weights (instead of deepcopy)
        inner_model = TopasARC60M(topas_model.config).to(device)
        inner_model.load_state_dict(topas_model.state_dict(), strict=False)
        inner_opt = optim.SGD(inner_model.parameters(), lr=cli_args.meta_adapt_lr)

        # Inner loop adaptation
        for _ in range(cli_args.meta_inner_steps):
            loss = train_step(inner_model, hrm_model, task, inner_opt, torch.amp.GradScaler("cuda"), device)
            if loss is None:
                continue
            if not torch.is_tensor(loss):  # convert float back to tensor
                loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True, device=device)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()

        # Accumulate meta-gradients (difference between adapted and base model)
        for (n, p_base), (_, p_inner) in zip(topas_model.named_parameters(), inner_model.named_parameters()):
            if p_base.requires_grad:
                meta_grads[0][n] += (p_inner.data - p_base.data)

    # Apply averaged meta-gradients to base model
    for n, p in topas_model.named_parameters():
        if p.requires_grad:
            p.grad = meta_grads[0][n] / float(cli_args.meta_tasks_per_batch)

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
        
        # Ensure Wormhole is initialized for template mining
        if not hasattr(topas_model.dream, 'wormhole') or topas_model.dream.wormhole is None:
            try:
                from wormhole_offline import WormholeTemplateMiner
                topas_model.dream.wormhole = WormholeTemplateMiner()
                print("[TOPAS] Wormhole template miner initialized")
            except ImportError as e:
                print(f"[TOPAS] Warning: Could not initialize Wormhole: {e}")
        else:
            print("[TOPAS] Wormhole template miner already initialized")
    
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

    # Initialize PriorNet if enabled
    if cli_args and getattr(cli_args, "enable_priornet", False):
        hidden = [int(x) for x in cli_args.priornet_hidden.split(",")]
        topas_model.priornet = PriorNet(input_dim=topas_model.config.width, hidden=hidden).to(device)
        topas_model._priornet_reg_weight = cli_args.priornet_reg_weight
        print(f"✅ PriorNet: {sum(p.numel() for p in topas_model.priornet.parameters()):,} parameters (reg_weight={cli_args.priornet_reg_weight})")

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

    # CRITICAL: Unfreeze model after dream pretraining if it was frozen
    if cli_args.dream_pretrain_freeze_model:
        logger.info("[Dream-Pretrain] Unfreezing model parameters for main training")
        topas_model.train()
        for param in topas_model.parameters():
            param.requires_grad = True

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
                ce_loss = ce_loss.float()  # Ensure float32 for AMP/GradScaler
                
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
                        if isinstance(loss_value, torch.Tensor) and loss_value.requires_grad:
                            total_loss = total_loss + loss_value  # Weight already applied in model
                            if global_step % 100 == 0:  # Log occasionally
                                logging.info(f"Step {global_step}: ce_loss={ce_loss:.3f}, {loss_name}={loss_value:.3f}")

                # === Joint HRM Training with ACT Loss ===
                if hasattr(topas_model, 'planner_loss_head') and global_step > 1000:  # After initial warmup
                    try:
                        # Ensure HRM planner_loss_head is on correct device
                        if hasattr(topas_model.planner_loss_head, 'to'):
                            topas_model.planner_loss_head = topas_model.planner_loss_head.to(device)

                        # HRM supervision targets (reuse grid tokens)
                        tokens = topas_model.grid_to_tokens(input_grid)
                        # CRITICAL: Ensure tokens are on correct device immediately
                        tokens = tokens.to(device)

                        # task_id was already unpacked above from the dataloader tuple
                        raw_pid = task_id
                        if isinstance(raw_pid, torch.Tensor):
                            pid = int(raw_pid.item())
                        elif isinstance(raw_pid, (int, np.integer)):
                            pid = int(raw_pid)
                        elif isinstance(raw_pid, str):
                            # map string → stable integer in [0, 999]
                            pid = int(hashlib.sha1(raw_pid.encode()).hexdigest(), 16) % 1000
                        else:
                            pid = int(hashlib.sha1(str(raw_pid).encode()).hexdigest(), 16) % 1000
                        puzzle_ids = torch.tensor([pid], device=device, dtype=torch.long)
                        
                        # --- HRM token normalization ---
                        # Use tokens.size(1) as source of truth (already may include puzzle_emb_len).
                        # Add +1 EOS slot to match HRM logits.
                        base_len = tokens.size(1)
                        target_len = base_len + 1

                        inputs_norm = tokens.to(device)
                        labels_norm = tokens.to(device)

                        # Truncate/pad to exactly target_len
                        if inputs_norm.size(1) > target_len:
                            inputs_norm = inputs_norm[:, :target_len]
                            labels_norm = labels_norm[:, :target_len]
                        elif inputs_norm.size(1) < target_len:
                            pad_len = target_len - inputs_norm.size(1)
                            pad_inputs = torch.zeros(inputs_norm.size(0), pad_len, device=device, dtype=inputs_norm.dtype)
                            pad_labels = torch.full((labels_norm.size(0), pad_len), -100, device=device, dtype=labels_norm.dtype)
                            inputs_norm = torch.cat([inputs_norm, pad_inputs], dim=1)
                            labels_norm = torch.cat([labels_norm, pad_labels], dim=1)

                        # Debug probe
                        if global_step % 200 == 0:
                            # seq_len variable no longer exists; log base_len instead
                            logging.info(
                                f"[HRM-Supervision] aligned inputs={inputs_norm.shape[1]}, labels={labels_norm.shape[1]}, base_len={base_len}, target_len={target_len}"
                            )

                        # Ensure all tensors are on the same device
                        batch_dict = {
                            "inputs": inputs_norm,
                            "labels": labels_norm,
                            "puzzle_identifiers": puzzle_ids.to(device)
                        }
                        carry = topas_model.planner_loss_head.initial_carry(batch_dict)

                        # Ensure carry is on correct device
                        if isinstance(carry, dict):
                            carry = {k: v.to(device) if torch.is_tensor(v) else v for k, v in carry.items()}
                        elif torch.is_tensor(carry):
                            carry = carry.to(device)

                        new_carry, hrm_loss, hrm_metrics, _, _ = topas_model.planner_loss_head(
                            return_keys=[], carry=carry, batch=batch_dict
                        )

                        # Ensure hrm_loss is on correct device
                        if isinstance(hrm_loss, torch.Tensor):
                            hrm_loss = hrm_loss.to(device)

                        # Only add HRM loss if it has gradients
                        if isinstance(hrm_loss, torch.Tensor) and hrm_loss.requires_grad:
                            total_loss = total_loss + 0.2 * hrm_loss  # Start with modest weight
                        
                        # Log HRM metrics
                        if global_step % 100 == 0:
                            lm_loss = hrm_metrics.get("lm_loss", 0.0)
                            q_halt_loss = hrm_metrics.get("q_halt_loss", 0.0)
                            logging.info(f"[HRM-Joint] hrm_loss={hrm_loss:.3f}, lm_loss={lm_loss:.3f}, q_halt_loss={q_halt_loss:.3f}")
                    except Exception as e:
                        if global_step % 100 == 0:
                            logging.warning(f"[HRM-Joint] Supervision failed: {e}")
                
                # ---- RelMem auxiliary loss every N steps (with warm-up) ----
                # Delay RelMem until after 5 epochs (~2000 steps)
                relmem_warmup_epochs = 5
                current_epoch = global_step // 400  # assumes ~400 steps/epoch
                relmem_loss_interval = getattr(args, 'relmem_loss_interval', 25) if 'args' in locals() else 25
                if (hasattr(topas_model, "relmem") and topas_model.relmem is not None and 
                    current_epoch >= relmem_warmup_epochs and (global_step % relmem_loss_interval == 0)):
                    try:
                        reg_alpha = getattr(args, 'relmem_reg_alpha', 1e-4) if 'args' in locals() else 1e-4
                        reg_beta  = getattr(args, 'relmem_reg_beta', 1e-4)  if 'args' in locals() else 1e-4

                        # Safe aggregation: only add terms that are tensors with ≥1D.
                        relmem_aux = torch.tensor(0.0, device=device)
                        if hasattr(topas_model.relmem, 'inverse_loss_safe'):
                            inv_loss = topas_model.relmem.inverse_loss_safe()
                            if torch.is_tensor(inv_loss) and inv_loss.dim() >= 1:
                                relmem_aux = relmem_aux + reg_alpha * inv_loss.sum()
                        elif hasattr(topas_model.relmem, 'inverse_loss'):
                            inv_loss = topas_model.relmem.inverse_loss()
                            if torch.is_tensor(inv_loss) and inv_loss.dim() >= 1:
                                relmem_aux = relmem_aux + reg_alpha * inv_loss.sum()

                        # Only call inheritance_pass if a safe variant exists; otherwise skip to avoid 0D@0D matmul
                        if hasattr(topas_model.relmem, 'inheritance_pass_safe'):
                            inh = topas_model.relmem.inheritance_pass_safe()
                            if torch.is_tensor(inh) and inh.dim() >= 1:
                                relmem_aux = relmem_aux + reg_beta * inh.sum()
                        else:
                            # No safe variant; keep silent and skip to avoid console spam
                            pass
                        
                        if torch.is_tensor(relmem_aux) and relmem_aux.item() > 0:
                            total_loss = total_loss + relmem_aux
                            if global_step % 100 == 0:
                                logging.info(f"Step {global_step}: RelMem aux loss={relmem_aux.item():.6f}")
                    except Exception as e:
                        # Downgrade to debug to avoid spam when RelMem lacks safe hooks
                        logging.debug(f"RelMem auxiliary loss skipped: {e}")
                
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

                # === Phase 6: PriorNet DSL operation bias learning ===
                if getattr(topas_model, "priornet", None):
                    try:
                        context = outputs.get("brain") or outputs.get("latent")
                        if context is not None:
                            prior_logits = topas_model.priornet(context.mean(dim=1))
                            prior_bias = torch.softmax(prior_logits, dim=-1)
                            # Apply PriorNet regularization to prevent overfitting
                            prior_reg = getattr(topas_model, "_priornet_reg_weight", 0.01) * (prior_logits**2).mean()
                            total_loss = total_loss + prior_reg
                            if global_step % 200 == 0:
                                logging.info(f"[PriorNet] reg={prior_reg.item():.6f}, bias_entropy={torch.sum(-prior_bias * torch.log(prior_bias + 1e-8)).item():.3f}")
                    except Exception as e:
                        if global_step % 200 == 0:
                            logging.debug(f"[PriorNet] skipped: {e}")

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
        
        # Store outputs for enhanced dopamine scoring (when metrics computed)
        if return_metrics:
            try:
                globals()['last_outputs_for_dopamine'] = {
                    'outputs': outputs,
                    'input_grid': input_grid.detach().cpu(),
                    'target_grid': target_grid.detach().cpu(),
                    'global_step': global_step
                }
            except Exception:
                pass
        
        # Tighter gradient clipping for RelMem stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(topas_model.parameters(), max_norm=0.5)  # Tighter clipping
        
        scaler.step(optimizer)
        scaler.update()
        
        # Apply post-optimizer hooks for RelMem + exemplar consolidation
        if hasattr(topas_model, 'relmem') and topas_model.relmem is not None:
            try:
                if hasattr(topas_model.relmem, 'apply_post_optimizer_hooks'):
                    topas_model.relmem.apply_post_optimizer_hooks()
                elif hasattr(topas_model.relmem, 'post_optimizer_step'):
                    topas_model.relmem.post_optimizer_step()

                # Consolidate exemplar memory every 500 steps
                if hasattr(topas_model.relmem, "consolidate_exemplars") and global_step % 500 == 0:
                    topas_model.relmem.consolidate_exemplars()
            except Exception as e:
                if global_step % 500 == 0:
                    logging.warning(f"RelMemExemplar post-optimizer failed: {e}")

        if return_metrics:
            # Pass model and grids for comprehensive metrics including EBR
            metrics = compute_metrics(topas_model, input_grid, target_grid, hrm_latents=hrm_latents)
            return loss.item(), metrics
        else:
            return loss.item() if isinstance(loss, torch.Tensor) else None

    except Exception:
        logging.exception("Exception in train_step")
        return None


# === Lightweight curriculum heuristics ===
def _heuristic_difficulty(demo_in: torch.Tensor, demo_out: torch.Tensor) -> str:
    """Cheap proxy: size, palette, and rough difference."""
    try:
        H, W = demo_in.shape[-2], demo_in.shape[-1]
        size_score = int(H*W >= 16) + int(H*W >= 64) + int(H*W >= 196)
        palette = int(torch.unique(demo_in).numel())
        color_score = int(palette >= 3) + int(palette >= 5)
        diff = (demo_in.view(-1) != demo_out.view(-1)).float().mean().item()
        diff_score = int(diff > 0.4) + int(diff > 0.7)
        s = size_score + color_score + diff_score
        return "easy" if s <= 2 else ("medium" if s <= 4 else "hard")
    except Exception:
        return "medium"


def main():
    logger = setup_logging()
    print("🚀 Starting Simplified HRM-TOPAS Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Apply CLI dream + dataset settings
    try:
        cli_args = parse_args()
    except Exception:
        cli_args = None

    dataset_choice = getattr(cli_args, "dataset", "arc1") if cli_args else "arc1"

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

    if dataset_choice == "arc2" and ARC2Dataset is not None:
        print("📦 Using ARC-II dataset (arc-agi_training/evaluation/test)")
        dataset = ARC2Dataset(
            "arc-agi_training_challenges.json",
            "arc-agi_training_solutions.json",
            device=str(device)
        )
    else:
        print("📦 Using ARC-1 dataset")
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
    
    # === Policy-guided search integration ===
    try:
        op_policy = OpPolicyNet().to(device)
        logger.info("[Policy] OpPolicyNet initialized for guided search")
    except Exception as e:
        logger.warning(f"[Policy] OpPolicyNet initialization failed: {e}")
        op_policy = None

    # === Alpha-ARC X Neural-Guided Search 2.0 initialization ===
    replay_buffer = None
    if cli_args and PrioritizedReplay is not None:
        try:
            replay_buffer = PrioritizedReplay(capacity=cli_args.replay_cap)
            logger.info(f"[Alpha-ARC X] PrioritizedReplay initialized with capacity {cli_args.replay_cap}")
        except Exception as e:
            logger.warning(f"[Alpha-ARC X] PrioritizedReplay initialization failed: {e}")

    # Set search mode on model
    if cli_args and hasattr(topas_model, 'config'):
        try:
            topas_model.config.search_alg = cli_args.search_alg
            logger.info(f"[Alpha-ARC X] Search algorithm set to: {cli_args.search_alg}")
        except Exception as e:
            logger.warning(f"[Alpha-ARC X] Failed to set search algorithm: {e}")
    
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

    # === Enhanced Curriculum state ===
    active_bucket = 1
    stagnation_epochs = 0
    bucket_count = 8
    bucket_size = math.ceil(len(dataset) / bucket_count) if len(dataset) > 0 else 1
    
    for epoch in range(num_epochs):
        # === RelMem bias scheduling ===
        if hasattr(topas_model.config, 'relmem_op_bias_w'):
            if epoch < 5:
                # keep minimal bias during warmup
                topas_model.config.relmem_op_bias_w = 0.2
            elif 5 <= epoch < 30:
                # ramp up toward stronger bias
                progress = (epoch - 5) / 25.0
                topas_model.config.relmem_op_bias_w = 0.2 + progress * (0.5 - 0.2)
            else:
                # decay bias slightly after stabilization
                topas_model.config.relmem_op_bias_w = 0.3
            print(f"[RelMem] Epoch {epoch}: bias_w={topas_model.config.relmem_op_bias_w:.3f}")
        
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        if cli_args.bucket_unlock_patience and cli_args.bucket_unlock_patience > 0:
            print(f"[Bucket] Active bucket: {active_bucket}/{bucket_count}")
        epoch_losses = []
        epoch_metrics = {'exact_match': [], 'accuracy': [], 'mean_iou': [], 'exact_match_refined': []}
        from tqdm import tqdm
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress):
            
            # --- Curriculum gating by heuristic difficulty ---
            try:
                demos, test_inputs, test_outputs, task_id = batch
                if demos and len(demos) > 0:
                    d_in, d_out = demos[0][0], demos[0][1]
                    difficulty = _heuristic_difficulty(d_in, d_out)
                    if active_bucket == "easy" and difficulty != "easy":
                        continue
                    if active_bucket == "medium" and difficulty == "hard":
                        continue
            except Exception:
                pass
            
                        # Compute metrics every 10 steps
            compute_metrics_now = (global_step % 10 == 0)

            # === Phase 2: Meta-learning loop ===
            if getattr(trainer_cli_args, "enable_meta_loop", False) and global_step % 200 == 0:
                try:
                    # deepcopy fails on weight_norm layers — use state_dict clone
                    from models.topas_arc_60M import TopasARC60M
                    inner_model = TopasARC60M(topas_model.config).to(device)
                    inner_model.load_state_dict(topas_model.state_dict(), strict=False)
                    run_meta_loop(inner_model, hrm_model, dataset, trainer_cli_args, device)
                    del inner_model
                    print(f"[Meta-Learning] Applied MAML-lite update at step {global_step}")
                except Exception as e:
                    print(f"[Meta-Learning] Error at step {global_step}: {e}")

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
                    
                    # Curriculum unlock: promote buckets when stable high EM
                    try:
                        if em_val >= 0.90:
                            bucket_streak += 1
                        else:
                            bucket_streak = 0
                        if bucket_streak >= bucket_unlock_patience:
                            if active_bucket == "easy":
                                active_bucket = "medium"
                                logger.info("[Curriculum] Unlocked MEDIUM tasks")
                            elif active_bucket == "medium":
                                active_bucket = "hard"
                                logger.info("[Curriculum] Unlocked HARD tasks")
                            bucket_streak = 0
                    except Exception:
                        pass
                    
                                        # reconstruct current task from batch for trace gen / buffer
                    demos, test_inputs, test_outputs, task_id = batch
                    if demos and len(demos) > 0:
                        grid_in = demos[0][0].to(device)
                        grid_out = demos[0][1].to(device)
                        if not torch.is_tensor(grid_in): 
                            grid_in = torch.tensor(grid_in, device=device)
                        if not torch.is_tensor(grid_out):
                            grid_out = torch.tensor(grid_out, device=device)
                        task = Task(input=grid_in, output=grid_out, constraints={}, metadata={})
                    else:
                        task = None

                    if em_val >= getattr(cli_args, "breakthrough_threshold", 0.33) and task is not None:
                        logger.info(f"[Breakthrough] EM={em_val:.2%} at step={global_step} → dopamine capture")
                        # 1) Enhanced policy-guided planner bias (hybrid approach)
                        try:
                            planner_bias = build_policy_guided_bias(grid_in, grid_out, op_policy, device, temp=0.7)
                            if global_step % 100 == 0:
                                logger.info(f"[Policy] guided bias active: {len(planner_bias)} ops")
                        except Exception as e:
                            logger.warning(f"[Dopamine] policy-guided bias failed: {e}")
                            planner_bias = build_op_bias(temp=0.7)  # Fallback to historical
                        
                        # 2) Generate/verify traces (Alpha-ARC X enhanced with SC-STaR + PUCT)
                        try:
                            star_task = _as_star_task(task)

                            # Alpha-ARC X: SC-STaR wrapper for enhanced trace generation
                            if cli_args and cli_args.sc_star:
                                traces = _sc_run_star(star_bootstrapper, star_task, planner_bias, n=3)
                                logger.info(f"[Alpha-ARC X] SC-STaR generated {len(traces)} traces")
                            else:
                                traces = star_bootstrapper.generate_diverse_traces(star_task, n_traces=8, planner_op_bias=planner_bias)

                            # Alpha-ARC X: PUCT planning for additional program discovery
                            if cli_args and cli_args.search_alg == "puct" and puct_search is not None:
                                try:
                                    puct_programs = _puct_plan_stepwise(topas_model, demos, grid_in, grid_out, cli_args, device)
                                    if puct_programs:
                                        # Convert PUCT programs to trace format
                                        from types import SimpleNamespace
                                        puct_trace = SimpleNamespace(operations=puct_programs)
                                        traces.append(puct_trace)
                                        logger.info(f"[Alpha-ARC X] PUCT contributed program with {len(puct_programs)} ops")
                                except Exception as e:
                                    logger.warning(f"[Alpha-ARC X] PUCT planning failed: {e}")

                            # Belt-and-suspenders: sanitize *before* verify to avoid any internal hashing pitfalls
                            for t in traces:
                                if hasattr(t, "operations") and t.operations:
                                    t.operations = _stringify_ops(t.operations)
                            validations = star_bootstrapper.verify_traces(traces, star_task)
                            good_traces = [t for t, v in zip(traces, validations) if v.is_valid]

                        except Exception as e:
                            logger.warning(f"[Dopamine] STaR trace generation failed: {e}")
                            good_traces = []
                            star_task = _as_star_task(task)
                        
                        # 3) Update op priors from successful traces (with safety)
                        try:
                            for t in good_traces:
                                if hasattr(t, "operations") and t.operations:
                                    # Use _stringify_ops to ensure all operations are hashable strings
                                    safe_ops = _stringify_ops(t.operations)
                                    if safe_ops:
                                        op_success_count.update(safe_ops)
                        except Exception as e:
                            logger.warning(f"[Dopamine] op_success_count update failed: {e}")
                        # 4) Enhanced Dopamine: score + robust storage
                        outputs = globals().get('last_outputs_for_dopamine', {}).get('outputs', {})
                        programs = []
                        try:
                            if isinstance(outputs, dict):
                                programs = outputs.get("extras", {}).get("programs", []) or []
                        except Exception:
                            pass
                        prog_len = _extract_program_len(programs or [getattr(t, "operations", []) for t in good_traces])
                        iou_val = float(metrics.get("iou", 0.0))
                        if iou_val == 0.0:
                            try:
                                if isinstance(outputs, dict) and "grid" in outputs:
                                    iou_val = _safe_iou(outputs["grid"][0], grid_out)
                            except Exception:
                                pass

                        # Alpha-ARC X: Near-miss repair
                        if cli_args and near_miss_repair is not None and outputs and iou_val > 0.0:
                            try:
                                pred_grid = outputs.get("grid", [None])[0] if isinstance(outputs, dict) else None
                                if pred_grid is not None and torch.is_tensor(pred_grid):
                                    hamming_threshold = int(cli_args.near_miss_hamming_pct * pred_grid.numel())
                                    hamming_dist = _hamming(pred_grid, grid_out)
                                    if hamming_dist <= hamming_threshold:
                                        repaired_programs = near_miss_repair(pred_grid, grid_out, programs)
                                        if repaired_programs:
                                            # Add repaired programs to good_traces for further processing
                                            from types import SimpleNamespace
                                            for prog in repaired_programs:
                                                repair_trace = SimpleNamespace(operations=prog)
                                                good_traces.append(repair_trace)
                                            logger.info(f"[Alpha-ARC X] Near-miss repair added {len(repaired_programs)} programs")
                            except Exception as e:
                                logger.warning(f"[Alpha-ARC X] Near-miss repair failed: {e}")
                        dream_stats = None
                        try:
                            if hasattr(topas_model, "dream") and topas_model.dream is not None:
                                dream_stats = getattr(topas_model.dream, "last_stats", None)
                        except Exception:
                            pass
                        ent_red = _extract_entropy_reduction(dream_stats)
                        mdl_gain = 0.0
                        try:
                            if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole") and programs:
                                mined = topas_model.dream.wormhole.mine_from_programs(programs, top_k=3)
                                mdl_gain = _extract_mdl_gain(mined)
                        except Exception:
                            pass
                        enc_inp = _encode_grid_tensor(grid_in)
                        novelty = _novelty_estimate(enc_inp, self_play_buffer, k=64)
                        R, comps = _dopamine_score(em=1.0 if em_val >= 0.999 else em_val,
                                                   acc=float(metrics.get("accuracy", 0.0)),
                                                   iou=iou_val, program_len=prog_len,
                                                   entropy_red=ent_red, mdl_gain=mdl_gain,
                                                   novelty=novelty, Lmax=12)
                        ema = _dopamine_ema.update(R)
                        advantage = R - ema
                        global _last_dopamine_step
                        refractory = (global_step - _last_dopamine_step) < 20
                        threshold = 0.10
                        accept = (em_val >= 0.999) or (advantage >= threshold and not refractory)
                        if accept:
                            dopamine_reward(star_task, self_play_buffer, logger, global_step,
                                            score=float(advantage), components=comps)

                            # Alpha-ARC X: Add successful programs to replay buffer (always stringified)
                            if replay_buffer is not None:
                                priority = float(advantage) + 0.1  # Ensure positive priority
                                try:
                                    for t in good_traces:
                                        if hasattr(t, "operations") and t.operations:
                                            safe_replay_add(replay_buffer, t.operations, priority, logger)
                                    if programs:
                                        for prog in programs:
                                            if prog:
                                                safe_replay_add(replay_buffer, prog, priority, logger)
                                    logger.info(f"[Alpha-ARC X] Added {len(good_traces) + len(programs)} programs to replay buffer")
                                except Exception as e:
                                    logger.warning(f"[Alpha-ARC X] Replay buffer addition failed: {e}")

                            try:
                                if hasattr(topas_model, "dream") and hasattr(topas_model.dream, "wormhole") and programs:
                                    topas_model.dream.wormhole.refresh_ttl_from_programs(programs, bonus_ttl=3)
                            except Exception:
                                pass
                            try:
                                for t in good_traces:
                                    if hasattr(t, "operations") and t.operations:
                                        # Use _stringify_ops to ensure all operations are hashable strings
                                        safe_ops = _stringify_ops(t.operations)
                                        if safe_ops:
                                            for sop in safe_ops:
                                                op_success_count.update({sop: max(1, int(10 * max(0.0, advantage)))})
                            except Exception as op_err:
                                logger.warning(f"[Dopamine] advantage-based op_count update failed: {op_err}")
                            _last_dopamine_step = global_step
                        else:
                            logger.info(f"[Dopamine] Skipped (R={R:+.3f}, adv={advantage:+.3f}, ema={ema:+.3f}, refractory={refractory})")
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
                            if not torch.is_tensor(grid_in): 
                                grid_in = torch.tensor(grid_in, device=device)
                            if not torch.is_tensor(grid_out):
                                grid_out = torch.tensor(grid_out, device=device)
                            task = Task(input=grid_in, output=grid_out, constraints={}, metadata={})
                    if task is not None:
                        # Half of traces guided by policy-enhanced op_bias
                        planner_bias = build_policy_guided_bias(grid_in, grid_out, op_policy, device, temp=0.7)
                        star_task = _as_star_task(task)
                        traces = star_bootstrapper.generate_diverse_traces(star_task, n_traces=max(6, cli_args.monologue_min_traces), planner_op_bias=planner_bias)
                        vals = star_bootstrapper.verify_traces(traces, star_task)
                        valid_traces = [t for t, v in zip(traces, vals) if v.is_valid or v.similarity_score >= 0.90]
                        if len(valid_traces) >= 2:
                            c_res = consistency_enforcer.enforce_consistency(valid_traces, star_task)
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
                                
                                # Enhanced: Dopamine reward for strong consistency scores
                                if monolog_score >= 0.90:
                                    try:
                                        # Calculate consistency-based dopamine score
                                        consistency_reward = min(1.0, monolog_score * 1.2)  # Scale and cap at 1.0
                                        good_traces = [t for t, v in zip(traces, vals) if v.is_valid]
                                        prog_len = _extract_program_len([getattr(t, "operations", []) for t in good_traces])
                                        
                                        # Use enhanced scoring with consistency bonus
                                        R, comps = _dopamine_score(
                                            em=consistency_reward,
                                            acc=consistency_reward,
                                            iou=0.8,  # Reasonable default for consistency
                                            program_len=prog_len,
                                            entropy_red=0.1,
                                            mdl_gain=0.1,
                                            novelty=0.5,
                                            Lmax=12
                                        )
                                        
                                        ema = _dopamine_ema.update(R)
                                        advantage = R - ema
                                        
                                        # More lenient threshold for consistency rewards
                                        if advantage >= 0.10:
                                            logger.info(f"[Monologue] Consistency dopamine: score={monolog_score:.3f}, R={R:.3f}, adv={advantage:.3f}")
                                            dopamine_reward(star_task, self_play_buffer, logger, global_step,
                                                            score=float(advantage), components=comps)
                                            
                                            # Update operation success counts from consistent traces
                                            for t in good_traces:
                                                if hasattr(t, "operations") and t.operations:
                                                    # Use _stringify_ops to ensure all operations are hashable strings
                                                    safe_ops = _stringify_ops(t.operations)
                                                    if safe_ops:
                                                        for sop in safe_ops:
                                                            op_success_count.update({sop: max(1, int(5 * max(0.0, advantage)))})
                                    except Exception as e:
                                        logger.debug(f"[Monologue] Consistency dopamine failed: {e}")
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
                            # Store canonical encoded samples
                            enc_inp = _encode_grid_tensor(grid_in)
                            enc_out = _encode_grid_tensor(grid_out)
                            self_play_buffer.buffer.append((enc_inp, enc_out))
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
                            print(f"[SelfPlay] No puzzles generated at step={global_step} → trying Dream motifs")
                            if hasattr(topas_model, "dream") and hasattr(topas_model, "painter"):
                                try:
                                    # Get last Dream features
                                    dream_feat = getattr(topas_model.dream, "last_features", None)
                                    if dream_feat is not None:
                                        grid, logits, size = topas_model.painter(dream_feat)
                                        enc_in = _encode_grid_tensor(grid)
                                        # Cheap target: identity reconstruction
                                        enc_out = _encode_grid_tensor(grid.clone())
                                        self_play_buffer.buffer.append((enc_in, enc_out, 0.1))
                                        print(f"[Dream→SelfPlay] Injected 1 painter-motif puzzle (buffer={len(self_play_buffer.buffer)})")
                                except Exception as e:
                                    print(f"[Dream→SelfPlay] Painter fallback failed: {e}")
                except Exception as e:
                    import traceback
                    logging.getLogger(__name__).exception("[SelfPlay] failure: %s", e)
                
                # Sample and compute self-play loss with importance replay
                def extract_score(sample):
                    """
                    Extract score from sample tuple, handling both (input, output) and (input, output, score) formats.
                    Dream→Painter samples carry a default score of 0.1, but we give them a small novelty bonus so they don't get drowned out.
                    """
                    score = 0.0
                    if len(sample) >= 3:
                        raw = sample[2]
                        score = float(raw) if hasattr(raw, "__float__") else 0.0
                    # Detect Dream-derived motifs (tiny 0.1 baseline score) and up-weight them
                    if abs(score - 0.1) < 1e-6:
                        score += 0.2  # novelty bonus
                    return score
                
                # Sample larger batch for stochastic importance weighting
                sp_candidates = self_play_buffer.sample_batch(32)
                if sp_candidates:
                    # Extract scores and compute stochastic importance weights
                    sp_candidates_with_scores = [(s, extract_score(s)) for s in sp_candidates]
                    scores = np.array([score for _, score in sp_candidates_with_scores])
                    
                    # Use stochastic importance sampling with softmax weighting
                    if np.sum(scores) > 0:
                        # Apply softmax with numerical stability
                        exp_scores = np.exp(scores - np.max(scores))
                        probabilities = exp_scores / np.sum(exp_scores)
                        # Sample up to 4 puzzles using importance weights (boundary check)
                        sample_size = min(4, len(sp_candidates))
                        selected_indices = np.random.choice(len(sp_candidates), size=sample_size, replace=False, p=probabilities)
                    else:
                        # Fallback to uniform random sampling if no scores available (boundary check)
                        sample_size = min(4, len(sp_candidates))
                        selected_indices = np.random.choice(len(sp_candidates), size=sample_size, replace=False)
                    
                    sp_samples = [sp_candidates_with_scores[i][0] for i in selected_indices]
                    
                    print(f"[SelfPlay] Training on {sample_size} importance-weighted puzzles from {len(sp_candidates)} candidates")
                    for sp_sample in sp_samples:
                        # Handle both tuple formats: (input, output) and (input, output, score)
                        sp_input = sp_sample[0]
                        sp_target = sp_sample[1]
                        try:
                            # Decode if samples are encoded tuples
                            if not torch.is_tensor(sp_input):
                                sp_input = _decode_grid(sp_input)
                            if not torch.is_tensor(sp_target):
                                sp_target = _decode_grid(sp_target)
                            sp_output = topas_model.forward_pretraining(sp_input.unsqueeze(0),
                                                                        target_shape=sp_target.shape[-2:])
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

                    # Alpha-ARC X: Sample from replay buffer for additional learning
                    if replay_buffer is not None and len(replay_buffer) > 0:
                        try:
                            replay_samples = replay_buffer.sample(2)  # Sample 2 programs
                            for program_ops, priority in replay_samples:
                                if program_ops:
                                    # Update op_success_count with replay programs
                                    for op in program_ops:
                                        op_success_count.update({op: max(1, int(priority * 5))})
                            logger.info(f"[Alpha-ARC X] Updated op_success_count with {len(replay_samples)} replay programs")
                        except Exception as e:
                            logger.warning(f"[Alpha-ARC X] Replay sampling failed: {e}")
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
            
            # === RelMem exemplar stats logging ===
            if hasattr(topas_model, "relmem") and hasattr(topas_model.relmem, "exemplar_stats"):
                if global_step % 500 == 0:
                    try:
                        ex_stats = topas_model.relmem.exemplar_stats()
                        logging.info(f"[RelMemExemplar] step={global_step} total={ex_stats['exemplar_total']}, avg_per_concept={ex_stats['exemplar_avg_per_concept']:.2f}")
                    except Exception:
                        pass

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