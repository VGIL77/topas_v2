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

# ✅ Near-miss repair integration
from trainers.near_miss import integrate_near_miss_learning, hamming_distance
from trainers.replay import PrioritizedReplay, create_replay_buffer_from_config

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

    # === Initialize shared replay buffer for near-miss repairs ===
    if "replay_buffer" not in state:
        replay_config = {
            'capacity': config.get('near_miss_buffer_capacity', 5000),
            'alpha': config.get('near_miss_priority_alpha', 0.6),
            'novelty_weight': config.get('near_miss_novelty_weight', 0.3),
            'eviction_strategy': 'oldest_low_priority',
            'min_score_threshold': config.get('near_miss_min_score', 0.1)
        }
        state["replay_buffer"] = create_replay_buffer_from_config(replay_config)
        print(f"[Phase 3] Initialized replay buffer with capacity {replay_config['capacity']}")
    replay_buffer = state["replay_buffer"]

    # === Near-miss repair configuration ===
    near_miss_config = {
        'enabled': config.get('near_miss_repair_enabled', True),
        'distance_threshold': config.get('near_miss_distance_threshold', 20),
        'similarity_threshold': config.get('near_miss_similarity_threshold', 0.6),
        'min_improvement': config.get('near_miss_min_improvement', 0.15),
        'max_repairs': config.get('near_miss_max_repairs', 2),
        'enable_complex_repairs': config.get('near_miss_enable_complex', False),
        'batch_frequency': config.get('near_miss_batch_frequency', 5)  # Check every 5 steps
    }
    print(f"[Phase 3] Near-miss repair {'enabled' if near_miss_config['enabled'] else 'disabled'}")
    if near_miss_config['enabled']:
        print(f"[Phase 3] Near-miss config: distance≤{near_miss_config['distance_threshold']}, "
              f"similarity≥{near_miss_config['similarity_threshold']:.2f}, "
              f"improvement≥{near_miss_config['min_improvement']:.2f}")

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

        # Near-miss repair tracking
        near_miss_failed_predictions = []
        near_miss_target_outputs = []
        near_miss_task_ids = []
        near_miss_repairs_found = 0
        near_miss_traces_added = 0
        
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

                # === Collect Near-Miss Candidates ===
                if (near_miss_config['enabled'] and
                    test.get("output") is not None and
                    grid is not None):
                    try:
                        # Check if this is a near-miss (close but not exact)
                        target_grid = test["output"]
                        pred_grid = grid

                        # Ensure grids have same shape for comparison
                        if (pred_grid.shape == target_grid.shape and
                            not torch.equal(pred_grid, target_grid)):

                            # Calculate Hamming distance for near-miss detection
                            distance = hamming_distance(pred_grid, target_grid)

                            if distance <= near_miss_config['distance_threshold']:
                                # This is a near-miss candidate
                                task_id = f"epoch{epoch}_step{step}_task{len(near_miss_task_ids)}"
                                near_miss_failed_predictions.append(pred_grid.clone())
                                near_miss_target_outputs.append(target_grid.clone())
                                near_miss_task_ids.append(task_id)

                                logger.log_batch(global_step, {
                                    "near_miss_detected": 1,
                                    "near_miss_distance": distance,
                                    "phase": "3_near_miss_detection"
                                })
                    except Exception as e:
                        print(f"[Phase 3] Near-miss detection error at step {step}: {e}")

                # === Process Near-Miss Repairs (Batch Processing) ===
                if (near_miss_config['enabled'] and
                    near_miss_failed_predictions and
                    step % near_miss_config['batch_frequency'] == 0):
                    try:
                        print(f"[Phase 3] Processing {len(near_miss_failed_predictions)} near-miss candidates...")

                        # Call near-miss repair integration
                        repair_results = integrate_near_miss_learning(
                            model=model,
                            failed_predictions=near_miss_failed_predictions,
                            target_outputs=near_miss_target_outputs,
                            task_ids=near_miss_task_ids,
                            replay_buffer=replay_buffer,
                            batch_info={
                                'epoch': epoch,
                                'step': step,
                                'global_step': global_step,
                                'phase': '3_self_critique'
                            },
                            config=near_miss_config
                        )

                        # Update tracking metrics
                        near_miss_repairs_found += repair_results.get('repairs_found', 0)
                        near_miss_traces_added += repair_results.get('traces_added', 0)

                        # Enhanced logging
                        if repair_results.get('repairs_found', 0) > 0:
                            logger.log_batch(global_step, {
                                "near_miss_repairs_found": repair_results['repairs_found'],
                                "near_miss_traces_added": repair_results['traces_added'],
                                "near_miss_high_priority": repair_results.get('high_priority_traces', 0),
                                "near_miss_success_rate": repair_results.get('batch_success_rate', 0.0),
                                "near_miss_processing_time": repair_results.get('processing_time', 0.0),
                                "phase": "3_near_miss_repair"
                            })

                            print(f"[Phase 3] Near-miss repair: {repair_results['repairs_found']} repairs, "
                                  f"{repair_results['traces_added']} traces added, "
                                  f"{repair_results.get('batch_success_rate', 0.0):.1%} success rate")

                        # Clear batch for next processing cycle
                        near_miss_failed_predictions.clear()
                        near_miss_target_outputs.clear()
                        near_miss_task_ids.clear()

                    except Exception as e:
                        print(f"[Phase 3] Near-miss repair error at step {step}: {e}")
                        # Clear failed batch to prevent accumulation
                        near_miss_failed_predictions.clear()
                        near_miss_target_outputs.clear()
                        near_miss_task_ids.clear()

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
                        "num_successful_traces": len(trace_report.get("successful_traces", [])),
                        "near_miss_candidates": len(near_miss_failed_predictions),
                        "replay_buffer_size": len(replay_buffer.traces) if replay_buffer else 0
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
        
        # === Final Near-Miss Processing (End of Epoch) ===
        if (near_miss_config['enabled'] and near_miss_failed_predictions):
            try:
                print(f"[Phase 3] Final near-miss processing: {len(near_miss_failed_predictions)} remaining candidates...")
                repair_results = integrate_near_miss_learning(
                    model=model,
                    failed_predictions=near_miss_failed_predictions,
                    target_outputs=near_miss_target_outputs,
                    task_ids=near_miss_task_ids,
                    replay_buffer=replay_buffer,
                    batch_info={'epoch': epoch, 'step': 'final', 'phase': '3_self_critique'},
                    config=near_miss_config
                )
                near_miss_repairs_found += repair_results.get('repairs_found', 0)
                near_miss_traces_added += repair_results.get('traces_added', 0)

                if repair_results.get('repairs_found', 0) > 0:
                    print(f"[Phase 3] Final near-miss results: {repair_results['repairs_found']} repairs, "
                          f"{repair_results['traces_added']} traces added")
            except Exception as e:
                print(f"[Phase 3] Final near-miss processing error: {e}")

        # Epoch summary
        if num_steps > 0:
            avg_loss = epoch_loss / num_steps
            print(f"[Phase 3] Epoch {epoch+1}/{epochs} complete:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Counterexamples Generated: {counterexample_count}")
            print(f"  Successful Traces: {successful_trace_count}")
            if near_miss_config['enabled']:
                print(f"  Near-Miss Repairs Found: {near_miss_repairs_found}")
                print(f"  Near-Miss Traces Added to Buffer: {near_miss_traces_added}")
                print(f"  Replay Buffer Size: {len(replay_buffer.traces)}")

            logger.log_epoch(epoch, "3_self_critique", avg_loss, {
                "counterexamples": counterexample_count,
                "successful_traces": successful_trace_count,
                "near_miss_repairs": near_miss_repairs_found if near_miss_config['enabled'] else 0,
                "near_miss_traces_added": near_miss_traces_added if near_miss_config['enabled'] else 0,
                "replay_buffer_size": len(replay_buffer.traces) if replay_buffer else 0
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
                    "epoch": epoch,
                    "replay_buffer_size": len(replay_buffer.traces) if replay_buffer else 0
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

    # === Final Replay Buffer Statistics ===
    if replay_buffer and near_miss_config['enabled']:
        buffer_stats = replay_buffer.get_statistics()
        print(f"[Phase 3] Final replay buffer statistics:")
        print(f"  Total traces: {buffer_stats['size']}/{buffer_stats['capacity']}")
        print(f"  Source distribution: {buffer_stats['source_distribution']}")
        print(f"  Average score: {buffer_stats['score_stats'].get('mean', 0.0):.3f}")
        print(f"  Average novelty: {buffer_stats['novelty_stats'].get('mean', 0.0):.3f}")

        # Export high-quality traces for analysis
        try:
            high_quality_traces = replay_buffer.get_top_traces(n=20)
            if high_quality_traces:
                print(f"[Phase 3] Top repair trace priorities: "
                      f"{[f'{t[1]:.3f}' for t in high_quality_traces[:5]]}")
        except Exception as e:
            print(f"[Phase 3] Error getting top traces: {e}")

    state.update({
        "model": model,
        "relmem": relmem,
        "optimizer": optimizer,
        "logger": logger,
        "global_step": global_step,
        "replay_buffer": replay_buffer,
        "phase3_complete": True,
        "near_miss_config": near_miss_config
    })

    print("[Phase 3] Self-Critique with STaR methodology and near-miss repair integration complete!")
    if near_miss_config['enabled']:
        print(f"[Phase 3] Near-miss repair summary: {near_miss_repairs_found} total repairs found across all epochs")
    return state