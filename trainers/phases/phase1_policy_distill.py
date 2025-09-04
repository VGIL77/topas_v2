#!/usr/bin/env python3
"""
Phase 1 – Policy Distillation
Distill beams into OpPolicyNet + ValueNet with strict ARC enforcement.
Now integrated with RelMem for inheritance + inverse-consistency regularization.
"""

import torch
import torch.nn.functional as F
from trainers.train_logger import TrainLogger
from trainers.trainer_utils import (
    safe_model_forward, compute_ce_loss, default_sample_fn,
    get_from_state
)
from models.policy_nets import OpPolicyNet
from models.value_net import ValueNet
from relational_memory_neuro import RelationalMemoryNeuro   # ✅ NEW
from validation.eval_runner import EvalRunner
from trainers.augmentation.alpha_trainer import generate_self_play_traces
# Try HRM imports with fallback for testing
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../docs/HRM-main'))
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    from models.losses import ACTLossHead
    _HAS_HRM = True
except ImportError:
    _HAS_HRM = False
    class HierarchicalReasoningModel_ACTV1:
        def __init__(self, *args, **kwargs): pass
        def initial_carry(self, *args): return {}
        def __call__(self, *args, **kwargs): return {}, {"q_halt_logits": torch.tensor(0.5), "z_H": torch.randn(1, 512)}
        def parameters(self): return iter([])
        def to(self, device): return self
    class ACTLossHead:
        def __init__(self, *args, **kwargs): pass

def run(config, state):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # === Model + Logger ===
    model = get_from_state(state, "model", required=True)
    logger = state.get("logger") or TrainLogger(config.get("log_dir", "./logs"))

    # === Initialize RelMem if missing ===
    if "relmem" not in state:
        state["relmem"] = RelationalMemoryNeuro(
            hidden_dim=model.config.slot_dim,
            max_concepts=4096,
            device=device
        ).to(device)
    relmem = state["relmem"]

    # === Planner rail ===
    planner = state.get("planner")
    if planner is None:
        # Expect planner to be pre-trained already (HRM-pretrain step)
        hrm_cfg = dict(
            batch_size=1, seq_len=30*30, vocab_size=10,
            num_puzzle_identifiers=1000, puzzle_emb_ndim=128,
            H_cycles=3, L_cycles=4, H_layers=4, L_layers=4,
            hidden_size=512, expansion=3.0, num_heads=8,
            pos_encodings="rope", halt_max_steps=6,
            halt_exploration_prob=0.1, forward_dtype="bfloat16"
        )
        planner = HierarchicalReasoningModel_ACTV1(hrm_cfg).to(device)
        state["planner"] = planner

    planner_loss_head = state.get("planner_loss_head")
    if planner_loss_head is None:
        planner_loss_head = ACTLossHead(planner, loss_type="softmax_cross_entropy")
        state["planner_loss_head"] = planner_loss_head

    # Freeze planner weights during Phase-1 distill
    for p in planner.parameters():
        p.requires_grad = False

    # Projection from planner latent → op_bias head
    if "planner_op_bias" not in state:
        state["planner_op_bias"] = torch.nn.Linear(
            hrm_cfg["hidden_size"], model.config.dsl_vocab_size
        ).to(device)
    planner_op_bias = state["planner_op_bias"]

    # === Nets + Optimizer ===
    policy_net = OpPolicyNet(model.config.slot_dim, model.config.dsl_vocab_size).to(device)
    value_net = ValueNet(model.config.slot_dim).to(device)
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()) + list(planner_op_bias.parameters()), 
        lr=1e-4
    )

    dataset = state.get("dataset") or None
    state["dataset"] = dataset

    steps = config.get("steps", 500)
    global_step = state.get("global_step", 0)
    
    # Self-play tracking
    total_self_play_traces = 0
    successful_self_play_traces = 0

    for step in range(steps):
        try:
            demos, test = default_sample_fn(dataset, device)
            grid, logits, extras = safe_model_forward(model, demos, test, device, training_mode=True)

            # === Base losses ===
            target = test.get("output")
            if target is None:
                raise RuntimeError("[Phase 1] No test output found in ARC data. Cannot train policy distill.")
            target = target.long().clamp(0, 9)

            policy_logits = policy_net(extras.get("latent", torch.zeros(1, model.config.slot_dim, device=device)))
            value_preds = value_net(extras.get("latent", torch.zeros(1, model.config.slot_dim, device=device)))

            loss_policy = compute_ce_loss(policy_logits, target)
            loss_value = F.mse_loss(value_preds.squeeze(), torch.tensor([1.0], device=device))  # placeholder
            loss = loss_policy + 0.1 * loss_value

            # === RelMem aux losses === ✅
            inherit_loss = relmem.inheritance_pass()
            inverse_loss = relmem.inverse_loss()
            loss = loss + 0.05 * inherit_loss + 0.05 * inverse_loss

            # === Distill from Planner → Policy/Value ===
            if hasattr(model, 'grid_to_tokens'):
                tokens = model.grid_to_tokens(test["input"])   # reuse adapter
            else:
                # Fallback if grid_to_tokens not available
                tokens = test["input"].view(test["input"].size(0), -1).clamp(0, 9)
            puzzle_ids = torch.zeros(tokens.size(0), dtype=torch.long, device=device)

            planner_batch = {
                "inputs": tokens,
                "labels": tokens,
                "puzzle_identifiers": puzzle_ids
            }
            carry = planner.initial_carry(planner_batch)
            carry, planner_out = planner(carry=carry, batch=planner_batch)

            if "z_H" in planner_out:
                with torch.no_grad():
                    z_H = planner_out["z_H"]
                # Distill latent op bias
                teacher_logits = planner_op_bias(z_H)
                student_logits = policy_logits  # Use already computed policy logits
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits.detach(), dim=-1),
                    reduction="batchmean"
                )
            else:
                distill_loss = torch.tensor(0.0, device=device)

            # === Total loss ===
            loss = loss + 0.5 * distill_loss

            # === Compute planner metrics for logging ===
            planner_logits_entropy = 0.0
            q_halt_mean = 0.0
            if "z_H" in planner_out and planner_out.get("q_halt_logits") is not None:
                with torch.no_grad():
                    # Compute teacher logits entropy
                    if 'teacher_logits' in locals() and teacher_logits is not None:
                        teacher_probs = F.softmax(teacher_logits, dim=-1)
                        planner_logits_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(-1).mean().item()
                    
                    # Compute halting confidence
                    q_halt_mean = float(planner_out["q_halt_logits"].mean().item()) if planner_out.get("q_halt_logits") is not None else 0.0

            # === Backward ===
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(value_net.parameters()) + list(planner_op_bias.parameters()), 
                max_norm=1.0
            )
            optimizer.step()

            # === RelMem post-opt hooks === ✅
            if hasattr(relmem, "apply_post_optimizer_hooks"):
                relmem.apply_post_optimizer_hooks()

            global_step += 1

            # === Alpha Self-Play Augmentation ===
            if step % config.get("self_play_interval", 50) == 0 and step > 0:
                try:
                    # Generate self-play traces for current task
                    task = {"test": {"input": test_grid[0] if test_grid.dim() > 2 else test_grid, 
                                    "output": test.get("output", torch.zeros_like(test_grid[0] if test_grid.dim() > 2 else test_grid))}}
                    self_play_traces = generate_self_play_traces(task, n_games=3, depth=4)
                    total_self_play_traces += len(self_play_traces)
                    
                    # Train on successful self-play traces
                    step_successful_traces = 0
                    for trace in self_play_traces:
                        if trace["score"] > 0.3:  # Only use high-quality traces
                            step_successful_traces += 1
                            # Convert trace to policy training data
                            ops_tensor = torch.tensor([DSL_OPS.index(op) if op in DSL_OPS else 0 
                                                     for op in trace["program"]]).to(device)
                            
                            # Generate policy logits for this trace
                            trace_logits = policy_net(latents.detach())
                            policy_loss_trace = F.cross_entropy(trace_logits, ops_tensor[:trace_logits.size(0)])
                            
                            # Backprop self-play loss
                            optimizer.zero_grad()
                            (0.2 * policy_loss_trace).backward()  # Weighted self-play loss
                            optimizer.step()
                    
                    successful_self_play_traces += step_successful_traces
                    print(f"[Phase 1] Self-play step {step}: {step_successful_traces}/{len(self_play_traces)} successful traces")
                except Exception as e:
                    print(f"[Phase 1] Self-play error: {e}")

            # === Progressive Evaluation ===
            if global_step % config.get("eval_interval_steps", 1000) == 0 and global_step > 0:
                try:
                    print(f"[Phase 1] Running evaluation at step {global_step}...")
                    eval_runner = EvalRunner(model=model, device=device)
                    metrics = eval_runner.run(
                        "ARC/arc-agi_evaluation_challenges.json",
                        "ARC/arc-agi_evaluation_solutions.json"
                    )
                    logger.log_batch(global_step, {
                        "eval_exact1": metrics.get("exact@1", 0.0),
                        "eval_exact_k": metrics.get("exact@k", 0.0),
                        "eval_iou": metrics.get("iou", 0.0),
                        "phase": "1_policy_eval"
                    })
                    print(f"[Phase 1] Eval Results - Exact@1: {metrics.get('exact@1', 0.0):.2%}, IoU: {metrics.get('iou', 0.0):.3f}")
                except Exception as e:
                    print(f"[Phase 1] Evaluation error at step {global_step}: {e}")

            # === Logging ===
            if step % config.get("log_interval", 20) == 0:
                # Enhanced batch logging with planner + RelMem signals
                losses = {
                    "total": loss.item(),
                    "policy": loss_policy.item(),
                    "value": loss_value.item(),
                    "distill": distill_loss.item() if distill_loss is not None else 0.0
                }
                
                relmem_info = {
                    "inherit_loss": inherit_loss.item(),
                    "inverse_loss": inverse_loss.item()
                }
                
                planner_info = {
                    "q_halt": q_halt_mean,
                    "entropy": planner_logits_entropy,
                    "kl_loss": distill_loss.item() if distill_loss is not None else 0.0
                }
                
                alpha_info = {
                    "total_traces": total_self_play_traces,
                    "successful_traces": successful_self_play_traces,
                    "success_rate": successful_self_play_traces / max(total_self_play_traces, 1)
                }
                
                # Use enhanced logging with separate info dicts
                logger.log_batch(global_step, losses, relmem_info=relmem_info, planner_info=planner_info, alpha_info=alpha_info)
            if step % 100 == 0:  # ✅ Optional RelMem + Planner + Alpha logging
                print(f"[RelMem] inherit={inherit_loss.item():.4f}, inverse={inverse_loss.item():.4f}")
                print(f"[Planner Distill] step={step} | KL={distill_loss.item():.4f}")
                success_rate = successful_self_play_traces / max(total_self_play_traces, 1)
                print(f"[Alpha Self-Play] traces={total_self_play_traces}, success_rate={success_rate:.2%}")

        except Exception as e:
            print(f"[Phase 1] Error in step {step}: {e}")
            continue

    state.update({
        "policy_net": policy_net,
        "value_net": value_net,
        "optimizer": optimizer,
        "logger": logger,
        "global_step": global_step,
        "phase1_complete": True,
        "relmem": relmem,  # ✅ Keep relmem in state
        "planner": planner,  # ✅ Keep planner in state
        "planner_loss_head": planner_loss_head,  # ✅ Keep planner loss head in state
        "planner_op_bias": planner_op_bias,  # ✅ Keep planner projection in state
        
        # ✅ State passing for Phase-3 consumption
        "planner_outputs": planner_out if 'planner_out' in locals() else {},
        "relmem_metrics": {
            "inherit": inherit_loss.item() if 'inherit_loss' in locals() else 0.0,
            "inverse": inverse_loss.item() if 'inverse_loss' in locals() else 0.0
        },
        "planner_metrics": {
            "q_halt_mean": q_halt_mean if 'q_halt_mean' in locals() else 0.0,
            "entropy": planner_logits_entropy if 'planner_logits_entropy' in locals() else 0.0,
            "kl_loss": distill_loss.item() if 'distill_loss' in locals() and distill_loss is not None else 0.0
        }
    })

    return state