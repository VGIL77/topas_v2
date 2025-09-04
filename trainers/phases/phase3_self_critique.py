#!/usr/bin/env python3
"""
Phase 3 – Self-Critique
Implements STaR methodology:
  1. Generate counterexamples
  2. Analyze reasoning traces
  3. Enforce RelMem consistency
  4. Bootstrap training with self-corrected traces
"""

import torch
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.train_logger import TrainLogger
from trainers.trainer_utils import safe_model_forward, default_sample_fn, get_from_state
from relational_memory_neuro import RelationalMemoryNeuro

# ✅ Self-critique subsystems
from trainers.self_critique.counterexamples import CounterexampleGenerator
from trainers.self_critique.trace_analysis import TraceAnalyzer
from trainers.self_critique.consistency import ConsistencyEnforcer
from trainers.self_critique.star_bootstrapper import STaRBootstrapper
from trainers.self_critique.critique_trainer import SelfCritiqueTrainer
from validation.eval_runner import EvalRunner

def run(config, state):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = state.get("model")
    if model is None:
        raise RuntimeError("[Phase 3] Model not found in state. Run previous phases first.")
    
    dataset = state.get("dataset")
    logger = state.get("logger") or TrainLogger()

    # === Initialize RelMem ===
    if "relmem" not in state:
        state["relmem"] = RelationalMemoryNeuro(
            hidden_dim=model.config.slot_dim,
            max_concepts=4096,
            device=device
        ).to(device)
    relmem = state["relmem"]

    # === Initialize self-critique components ===
    print("[Phase 3] Initializing self-critique components...")
    try:
        counter_gen = CounterexampleGenerator(device=device)
        trace_analyzer = TraceAnalyzer(device=device)
        consistency_enforcer = ConsistencyEnforcer(device=device)
        star_bootstrapper = STaRBootstrapper(model)
        critique_trainer = SelfCritiqueTrainer(model)
        print("[Phase 3] ✓ All self-critique components initialized")
    except Exception as e:
        print(f"[Phase 3] Warning: Could not initialize all self-critique components: {e}")
        # Fallback to basic training
        counter_gen = None
        trace_analyzer = None
        consistency_enforcer = None
        star_bootstrapper = None
        critique_trainer = None

    epochs = config.get("epochs", 3)
    steps_per_epoch = config.get("steps_per_epoch", 200)
    global_step = state.get("global_step", 0)
    
    # Initialize optimizer
    optimizer = state.get("optimizer") or torch.optim.AdamW(
        model.parameters(), 
        lr=config.get("learning_rate", 5e-5)
    )

    # === Main Self-Critique Loop ===
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_steps = 0
        counterexample_count = 0
        successful_trace_count = 0
        
        for step in range(steps_per_epoch):
            try:
                demos, test = default_sample_fn(dataset, device)
                
                # Ensure valid targets
                if test.get("output") is not None:
                    test["output"] = test["output"].long().clamp(0, 9)
                
                grid, logits, extras = safe_model_forward(model, demos, test, device, training_mode=True)

                # === RelMem baseline losses ===
                inherit_loss = relmem.inheritance_pass()
                inverse_loss = relmem.inverse_loss()
                base_loss = inherit_loss * 0.05 + inverse_loss * 0.05

                # === Self-critique pipeline ===
                consistency_loss = torch.tensor(0.0, device=device)
                counterexamples = []
                trace_report = {}
                
                if counter_gen is not None and trace_analyzer is not None:
                    try:
                        # Generate counterexamples
                        counterexamples = counter_gen.generate_from_failure(model, demos, test)
                        counterexample_count += len(counterexamples)
                        
                        # Analyze reasoning traces (includes planner + RelMem)
                        trace_report = trace_analyzer.analyze_traces(
                            model, demos, test, 
                            relmem=relmem,
                            planner=getattr(model, 'planner', None)
                        )
                        
                        # Enforce consistency
                        if consistency_enforcer is not None:
                            consistency_loss = consistency_enforcer.enforce_consistency(
                                relmem, counterexamples, trace_report
                            )
                            
                    except Exception as e:
                        print(f"[Phase 3] Self-critique error at step {step}: {e}")
                        consistency_loss = torch.tensor(0.0, device=device)

                # === Total loss ===
                total_loss = base_loss + 0.1 * consistency_loss

                # === Backward pass ===
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # === STaR Bootstrapping ===
                if star_bootstrapper is not None and trace_report.get("successful_traces"):
                    try:
                        star_bootstrapper.reinforce(traces=trace_report["successful_traces"])
                        successful_trace_count += len(trace_report["successful_traces"])
                    except Exception as e:
                        print(f"[Phase 3] STaR bootstrapping error: {e}")

                epoch_loss += total_loss.item()
                num_steps += 1
                global_step += 1

                # === Logging ===
                if step % config.get("log_interval", 20) == 0:
                    logger.log_batch(global_step, {
                        "phase": "3_self_critique",
                        "epoch": epoch,
                        "step": step,
                        "total_loss": total_loss.item(),
                        "inherit_loss": float(inherit_loss.item()) if hasattr(inherit_loss, 'item') else 0.0,
                        "inverse_loss": float(inverse_loss.item()) if hasattr(inverse_loss, 'item') else 0.0,
                        "consistency_loss": float(consistency_loss.item()) if hasattr(consistency_loss, 'item') else 0.0,
                        "num_counterexamples": len(counterexamples),
                        "num_successful_traces": len(trace_report.get("successful_traces", []))
                    })

            except Exception as e:
                print(f"[Phase 3] Error in training step {step}: {e}")
                continue
        
        # === End of epoch: Critique trainer ===
        if critique_trainer is not None and num_steps > 0:
            try:
                critique_trainer.apply_critiques(trace_report, consistency_loss)
                print(f"[Phase 3] Applied epoch-level critiques for epoch {epoch}")
            except Exception as e:
                print(f"[Phase 3] Critique trainer error: {e}")
        
        # Epoch summary
        if num_steps > 0:
            avg_loss = epoch_loss / num_steps
            print(f"[Phase 3] Epoch {epoch+1}/{epochs} complete:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Counterexamples Generated: {counterexample_count}")
            print(f"  Successful Traces: {successful_trace_count}")
            
            logger.log_epoch(epoch, "3_self_critique", avg_loss, {
                "counterexamples": counterexample_count,
                "successful_traces": successful_trace_count
            })
            
            # === Evaluate after each self-critique epoch ===
            try:
                print(f"[Phase 3] Running evaluation after epoch {epoch+1}...")
                eval_runner = EvalRunner(model=model, device=device)
                metrics = eval_runner.run(
                    "ARC/arc-agi_evaluation_challenges.json",
                    "ARC/arc-agi_evaluation_solutions.json"
                )
                logger.log_batch(global_step, {
                    "eval_after_critique_exact1": metrics.get("exact@1", 0.0),
                    "eval_after_critique_exact_k": metrics.get("exact@k", 0.0),
                    "eval_after_critique_iou": metrics.get("iou", 0.0),
                    "phase": "3_critique_eval",
                    "epoch": epoch
                })
                print(f"[Phase 3] Post-Critique Eval - Exact@1: {metrics.get('exact@1', 0.0):.2%}, IoU: {metrics.get('iou', 0.0):.3f}")
            except Exception as e:
                print(f"[Phase 3] Evaluation error after epoch {epoch}: {e}")

    # === RelMem post-training plasticity ===
    if hasattr(relmem, "apply_post_optimizer_hooks"):
        try:
            relmem.apply_post_optimizer_hooks()
            print("[Phase 3] Applied RelMem post-training plasticity")
        except Exception as e:
            print(f"[Phase 3] RelMem plasticity error: {e}")

    state.update({
        "model": model,
        "relmem": relmem,
        "optimizer": optimizer,
        "logger": logger,
        "global_step": global_step,
        "phase3_complete": True
    })

    print("[Phase 3] Self-Critique with STaR methodology complete!")
    return state