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
from models.policy_nets import OpPolicyNet, ValueNet
from relational_memory_neuro import RelationalMemoryNeuro   # ✅ NEW

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

    # === Nets + Optimizer ===
    policy_net = OpPolicyNet(model.config.slot_dim, model.config.dsl_vocab_size).to(device)
    value_net = ValueNet(model.config.slot_dim).to(device)
    optimizer = torch.optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=1e-4)

    dataset = state.get("dataset") or None
    state["dataset"] = dataset

    steps = config.get("steps", 500)
    global_step = state.get("global_step", 0)

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

            # === Backward ===
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy_net.parameters()) + list(value_net.parameters()), max_norm=1.0)
            optimizer.step()

            # === RelMem post-opt hooks === ✅
            if hasattr(relmem, "apply_post_optimizer_hooks"):
                relmem.apply_post_optimizer_hooks()

            global_step += 1

            # === Logging ===
            if step % config.get("log_interval", 20) == 0:
                logger.log_batch(global_step, {
                    "total": loss.item(),
                    "policy": loss_policy.item(),
                    "value": loss_value.item(),
                    "inherit": inherit_loss.item(),
                    "inverse": inverse_loss.item()
                })
            if step % 100 == 0:  # ✅ Optional RelMem logging
                print(f"[RelMem] inherit={inherit_loss.item():.4f}, inverse={inverse_loss.item():.4f}")

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
        "relmem": relmem  # ✅ Keep relmem in state
    })

    return state