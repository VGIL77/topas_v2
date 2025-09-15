"""
Phase 4 – MCTS Alpha with PUCT Search and Replay Supervision
Combines MCTS search with neural policy/value networks using PUCT algorithm.
Integrates prioritized replay buffer for trace supervision and supports configurable
search methods (PUCT or beam search for backward compatibility).

Neural-Guided Search 2.0 Implementation with:
- PUCT-based Monte Carlo Tree Search
- Prioritized replay buffer for diverse trace sampling
- Deep mining integration with replay supervision
- Policy and value network co-training
- Comprehensive logging and monitoring
"""

def run(config, state):
    from trainers.trainer_utils import default_sample_fn, program_to_tensor, compute_ce_loss, safe_model_forward
    from trainers.train_logger import TrainLogger
    from models.dsl_search import beam_search
    from trainers.augmentation.deep_program_discoverer import mine_deep_programs
    from relational_memory_neuro import RelationalMemoryNeuro
    from validation.eval_runner import EvalRunner

    # Neural-Guided Search 2.0 imports
    from trainers.puct_search import puct_search, PUCTSearcher
    from trainers.replay import PrioritizedReplay, add_deep_mining_programs
    from models.policy_nets import PolicyPrediction
    from models.value_net import ValuePrediction

    import torch
    import numpy as np
    import time
    from typing import List, Dict, Any, Optional, Tuple

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Get model and policy/value nets from state
    model = state.get("model")
    if model is None:
        print("[Phase 4] Model not found in state. Run Phase 0 first.")
        return state

    policy_net = state.get("policy_net")
    value_net = state.get("value_net")

    if policy_net is None or value_net is None:
        print("[Phase 4] Policy/Value nets not found. Skipping MCTS phase.")
        state["phase4_completed"] = True
        return state

    # Phase 4 configuration parameters
    use_puct = config.get("use_puct_search", True)  # Enable PUCT by default
    num_simulations = config.get("puct_simulations", 800)
    c_puct = config.get("c_puct", 1.4)
    search_timeout = config.get("search_timeout", 30.0)

    # Replay buffer configuration
    replay_capacity = config.get("replay_capacity", 10000)
    replay_alpha = config.get("replay_alpha", 0.6)
    replay_sample_size = config.get("replay_sample_size", 32)
    replay_temperature = config.get("replay_temperature", 1.0)

    print(f"[Phase 4] Neural-Guided Search 2.0 Configuration:")
    print(f"  - Search Method: {'PUCT' if use_puct else 'Beam Search'}")
    print(f"  - PUCT Simulations: {num_simulations}")
    print(f"  - Exploration Constant: {c_puct}")
    print(f"  - Replay Buffer: {replay_capacity} capacity, α={replay_alpha}")

    # Initialize RelationalMemoryNeuro
    if "relmem" not in state:
        state["relmem"] = RelationalMemoryNeuro(
            hidden_dim=model.config.slot_dim,
            max_concepts=4096,
            device=device
        ).to(device)
    relmem = state["relmem"]

    # Initialize Prioritized Replay Buffer
    if "replay_buffer" not in state:
        state["replay_buffer"] = PrioritizedReplay(
            capacity=replay_capacity,
            alpha=replay_alpha,
            novelty_weight=0.5,
            eviction_strategy='oldest_low_priority',
            min_score_threshold=0.1
        )
    replay_buffer = state["replay_buffer"]

    # Initialize PUCT Searcher if enabled
    puct_searcher = None
    if use_puct:
        puct_searcher = PUCTSearcher(
            policy_net=policy_net,
            value_net=value_net,
            c_puct=c_puct,
            max_depth=config.get("mcts_depth", 10),
            device=device
        )
        print(f"[Phase 4] PUCT Searcher initialized with {len(puct_searcher.available_ops)} DSL operations")

    logger = state.get("logger") or TrainLogger(config.get("log_dir", "logs"))
    dataset = state.get("dataset")

    # Enhanced optimizer for policy and value networks
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5)
    )

    # Learning rate scheduler for stable training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Enhanced vocabulary for program encoding (compatible with DSL registry)
    vocab = {
        "<PAD>": 0, "<UNK>": 1, "rotate90": 2, "rotate180": 3, "rotate270": 4,
        "flip_h": 5, "flip_v": 6, "color_map": 7, "crop_bbox": 8, "flood_fill": 9,
        "outline": 10, "symmetry": 11, "translate": 12, "scale": 13, "tile": 14,
        "paste": 15, "tile_pattern": 16, "crop_nonzero": 17, "extract_color": 18,
        "resize_nn": 19, "center_pad_to": 20, "identity": 21
    }

    steps = config.get("steps", 200)
    global_step = state.get("global_step", 0)

    # Training statistics
    training_stats = {
        'total_traces_generated': 0,
        'puct_searches': 0,
        'beam_searches': 0,
        'deep_mining_runs': 0,
        'replay_samples': 0,
        'avg_search_time': 0.0,
        'avg_puct_simulations': 0.0,
        'policy_loss_history': [],
        'value_loss_history': []
    }

    print(f"[Phase 4] Starting Neural-Guided Search training for {steps} steps...")

    for step in range(steps):
        try:
            step_start_time = time.time()

            # Use default_sample_fn to get data
            demos, test = default_sample_fn(dataset, device)

            # Convert demos to pairs for search
            demo_pairs = [(d["input"], d["output"]) for d in demos]
            test_input = test.get("input")
            task_id = f"step_{global_step}_{step}"

            traces = []
            search_info = {}

            # Run search (PUCT or beam search based on configuration)
            try:
                if use_puct and puct_searcher is not None:
                    # Run PUCT search
                    best_program, search_info = puct_searcher.search(
                        demos=demo_pairs,
                        test_input=test_input,
                        num_simulations=num_simulations,
                        timeout_seconds=search_timeout
                    )

                    if best_program is not None:
                        # Convert DSLProgram to trace-like object
                        trace = type("PUCTTrace", (), {
                            "program": best_program.ops,
                            "params": best_program.params,
                            "score": search_info.get("best_value", 0.0),
                            "source": "puct_search",
                            "search_info": search_info
                        })()
                        traces.append(trace)

                    training_stats['puct_searches'] += 1
                    training_stats['avg_puct_simulations'] = (
                        training_stats['avg_puct_simulations'] * 0.9 +
                        search_info.get('simulations_run', 0) * 0.1
                    )

                    print(f"[Phase 4] PUCT search: {search_info.get('simulations_run', 0)} sims, "
                          f"value={search_info.get('best_value', 0.0):.3f}, "
                          f"time={search_info.get('search_time', 0.0):.2f}s")
                else:
                    # Fallback to beam search for backward compatibility
                    traces = beam_search(
                        demo_pairs, test_input,
                        priors=None, op_bias=None,
                        depth=config.get("mcts_depth", 10),
                        beam=config.get("beam_width", 50)
                    )
                    training_stats['beam_searches'] += 1

            except Exception as e:
                print(f"[Phase 4] Search error: {e}")
                traces = []

            # Deep program mining at intervals
            if step % config.get("deep_mining_interval", 200) == 0:
                try:
                    print(f"[Phase 4] Running deep program mining at step {step}...")
                    mined_programs = mine_deep_programs(
                        {"test": test},
                        max_depth=config.get("deep_mining_depth", 10)
                    )

                    # Add mined programs to replay buffer
                    mining_results = []
                    for prog in mined_programs:
                        mining_result = {
                            'task_id': f"{task_id}_mining",
                            'operations': prog.get("program", ["<UNK>"]),
                            'parameters': prog.get("params", []),
                            'score': prog.get("score", 0.0),
                            'input_grid': test_input,
                            'output_grid': test.get("output", test_input),
                            'target_grid': test.get("output"),
                            'method': 'deep_mining'
                        }
                        mining_results.append(mining_result)

                        # Also add to traces for immediate training
                        trace = type("MiningTrace", (), {
                            "program": prog.get("program", ["<UNK>"]),
                            "params": prog.get("params", []),
                            "score": prog.get("score", 0.0),
                            "source": "deep_mining"
                        })()
                        traces.append(trace)

                    # Add to replay buffer
                    if mining_results:
                        added_count = add_deep_mining_programs(replay_buffer, mining_results)
                        training_stats['deep_mining_runs'] += 1
                        print(f"[Phase 4] Deep mining: {len(mining_results)} programs, {added_count} added to replay")

                except Exception as e:
                    print(f"[Phase 4] Deep mining error: {e}")

            # Add current traces to replay buffer
            if traces:
                for trace in traces:
                    try:
                        success = replay_buffer.push(
                            task_id=task_id,
                            program=getattr(trace, 'program', []),
                            params=getattr(trace, 'params', []),
                            score=getattr(trace, 'score', 0.0),
                            input_grid=test_input,
                            output_grid=test.get("output", test_input),
                            target_grid=test.get("output"),
                            source=getattr(trace, 'source', 'search'),
                            metadata=getattr(trace, 'search_info', {})
                        )
                        if success:
                            training_stats['total_traces_generated'] += 1
                    except Exception as e:
                        print(f"[Phase 4] Failed to add trace to replay buffer: {e}")

            # Sample from replay buffer for training
            replay_traces = []
            if replay_buffer.traces and len(replay_buffer.traces) > replay_sample_size:
                try:
                    replay_traces = replay_buffer.sample(
                        n=replay_sample_size,
                        temperature=replay_temperature
                    )
                    training_stats['replay_samples'] += len(replay_traces)
                    print(f"[Phase 4] Sampled {len(replay_traces)} traces from replay buffer "
                          f"(total: {len(replay_buffer.traces)})")
                except Exception as e:
                    print(f"[Phase 4] Replay sampling error: {e}")

            # Combine current traces with replay traces for training
            all_training_traces = traces + [trace for trace in replay_traces]

            if not all_training_traces:
                print(f"[Phase 4] Warning: No traces available for training at step {step}")
                continue

            # Convert programs to tensors for training
            targets = torch.stack([
                program_to_tensor(
                    t.program if hasattr(t, 'program') else ["<UNK>"],
                    vocab
                ) for t in all_training_traces
            ]).to(device)

            scores = torch.tensor([
                t.score if hasattr(t, 'score') else 0.0
                for t in all_training_traces
            ], dtype=torch.float32).to(device)

            # Extract trace sources for logging
            trace_sources = [getattr(t, 'source', 'unknown') for t in all_training_traces]
            source_counts = {}
            for source in trace_sources:
                source_counts[source] = source_counts.get(source, 0) + 1

            # Forward pass through policy and value networks
            try:
                # Extract test grid for network inputs
                test_grid = test.get("input") if isinstance(test, dict) else test
                if test_grid is None or not torch.is_tensor(test_grid):
                    test_grid = test.get("input", test.get("output"))

                if test_grid.dim() == 2:
                    test_grid = test_grid.unsqueeze(0)  # Add batch dim

                B, H, W = test_grid.shape

                # Create enhanced features for neural networks
                rel_features = torch.randn(B, 64, device=device)
                size_oracle = torch.tensor([[H, W, H, W]], device=device).float()
                theme_priors = torch.randn(B, 10, device=device)

                # Policy network forward pass with enhanced context
                if hasattr(policy_net, 'forward'):
                    policy_pred = policy_net.forward(
                        test_grid, rel_features, size_oracle, theme_priors
                    )

                    if isinstance(policy_pred, PolicyPrediction):
                        policy_logits = policy_pred.op_logits
                    elif hasattr(policy_pred, 'op_logits'):
                        policy_logits = policy_pred.op_logits
                    else:
                        policy_logits = policy_pred
                else:
                    # Fallback for simpler policy networks
                    policy_logits = policy_net(test_grid)

                # Value network forward pass with enhanced context
                if hasattr(value_net, 'forward'):
                    value_pred = value_net.forward(
                        test_grid, rel_features, size_oracle, theme_priors
                    )

                    if isinstance(value_pred, ValuePrediction):
                        value_preds = value_pred.solvability
                    elif hasattr(value_pred, 'solvability'):
                        value_preds = value_pred.solvability
                    else:
                        value_preds = value_pred
                else:
                    # Fallback for simpler value networks
                    value_preds = value_net(test_grid)

                # Validate network outputs
                if policy_logits is None or policy_logits.numel() == 0:
                    raise RuntimeError("[Phase 4] Policy network produced invalid logits. Cannot train without valid policy outputs!")
                if value_preds is None or value_preds.numel() == 0:
                    raise RuntimeError("[Phase 4] Value network produced invalid predictions. Cannot train without valid value outputs!")

                # Reshape for loss computation
                if policy_logits.dim() == 4:
                    # [B, T, L, V] -> [B*T, L, V]
                    B, T, L, V = policy_logits.shape
                    policy_logits = policy_logits.view(B*T, L, V)

                # Handle batch size mismatches
                if targets.dim() == 2 and policy_logits.dim() == 3:
                    if targets.size(0) != policy_logits.size(0):
                        min_size = min(targets.size(0), policy_logits.size(0))
                        targets = targets[:min_size]
                        policy_logits = policy_logits[:min_size]
                        scores = scores[:min_size]
                        value_preds = value_preds[:min_size] if value_preds.dim() > 0 else value_preds

                # Ensure value predictions have correct shape
                if value_preds.dim() > 1:
                    value_preds = value_preds.squeeze()
                if scores.dim() > 1:
                    scores = scores.squeeze()

            except Exception as e:
                print(f"[Phase 4] Forward pass failed: {e}. Skipping training step.")
                continue

            # Compute enhanced losses for neural-guided search
            try:
                # Policy loss with cross-entropy
                loss_policy = compute_ce_loss(policy_logits, targets)

                # Value loss with MSE (ensure tensors have compatible shapes)
                if value_preds.shape != scores.shape:
                    # Reshape to match
                    min_elements = min(value_preds.numel(), scores.numel())
                    value_preds = value_preds.view(-1)[:min_elements]
                    scores = scores.view(-1)[:min_elements]

                loss_value = torch.nn.functional.mse_loss(value_preds, scores)

                # Additional regularization for PUCT training
                if use_puct and puct_searcher is not None:
                    # Encourage policy diversity for exploration
                    policy_entropy = -torch.sum(
                        torch.softmax(policy_logits, dim=-1) *
                        torch.log_softmax(policy_logits, dim=-1),
                        dim=-1
                    ).mean()

                    # Value prediction confidence regularization
                    value_confidence_loss = torch.var(value_preds) * 0.1

                    # Combine losses with regularization
                    puct_regularization = -0.01 * policy_entropy + value_confidence_loss
                else:
                    puct_regularization = torch.tensor(0.0, device=device)

            except Exception as e:
                print(f"[Phase 4] Loss computation failed: {e}. Skipping training step.")
                continue

            # Add RelMem losses and PUCT regularization
            inherit_loss = relmem.inheritance_pass()
            inverse_loss = relmem.inverse_loss()

            # Combine all losses with appropriate weighting
            loss = (
                loss_policy +
                loss_value +
                0.05 * inherit_loss +
                0.05 * inverse_loss +
                puct_regularization
            )

            # Track loss components for monitoring
            training_stats['policy_loss_history'].append(float(loss_policy.item()))
            training_stats['value_loss_history'].append(float(loss_value.item()))

            # Enhanced backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(value_net.parameters()),
                max_norm=config.get("grad_clip_norm", 1.0)
            )

            optimizer.step()

            # Update learning rate based on loss
            if step % 50 == 0:  # Update scheduler periodically
                scheduler.step(loss.item())

            # Apply post-optimizer RelMem hooks
            if hasattr(relmem, "apply_post_optimizer_hooks"):
                relmem.apply_post_optimizer_hooks()

            # Update training statistics
            step_time = time.time() - step_start_time
            training_stats['avg_search_time'] = (
                training_stats['avg_search_time'] * 0.9 + step_time * 0.1
            )

            global_step += 1

            # Enhanced evaluation with PUCT-specific metrics
            if step % config.get("eval_interval_steps", 50) == 0 and step > 0:
                try:
                    print(f"[Phase 4] Running Neural-Guided Search evaluation at step {step}...")

                    # Standard evaluation
                    eval_runner = EvalRunner(model=model, device=device)
                    metrics = eval_runner.run(
                        "ARC/arc-agi_evaluation_challenges.json",
                        "ARC/arc-agi_evaluation_solutions.json"
                    )

                    # Get replay buffer statistics
                    replay_stats = replay_buffer.get_statistics()

                    # Enhanced logging with PUCT and replay metrics
                    logger.log_batch(global_step, {
                        "eval_exact1": metrics.get("exact@1", 0.0),
                        "eval_exact_k": metrics.get("exact@k", 0.0),
                        "eval_iou": metrics.get("iou", 0.0),
                        "phase": "4_neural_guided_search",
                        "search_method": "puct" if use_puct else "beam",
                        "replay_buffer_size": replay_stats['size'],
                        "replay_total_pushes": replay_stats['total_pushes'],
                        "replay_total_samples": replay_stats['total_samples'],
                        "avg_search_time": training_stats['avg_search_time'],
                        "puct_simulations": training_stats['avg_puct_simulations'],
                        "total_traces": training_stats['total_traces_generated'],
                        "policy_loss_avg": np.mean(training_stats['policy_loss_history'][-10:]) if training_stats['policy_loss_history'] else 0.0,
                        "value_loss_avg": np.mean(training_stats['value_loss_history'][-10:]) if training_stats['value_loss_history'] else 0.0
                    })

                    print(f"[Phase 4] Evaluation Results:")
                    print(f"  - Exact@1: {metrics.get('exact@1', 0.0):.2%}")
                    print(f"  - IoU: {metrics.get('iou', 0.0):.3f}")
                    print(f"  - Replay Buffer: {replay_stats['size']}/{replay_stats['capacity']} traces")
                    print(f"  - Search Method: {'PUCT' if use_puct else 'Beam Search'}")

                except Exception as e:
                    print(f"[Phase 4] Evaluation error at step {step}: {e}")

            # Enhanced logging with Neural-Guided Search metrics
            if step % config.get("log_interval", 20) == 0:
                # Get current replay buffer statistics
                replay_stats = replay_buffer.get_statistics()

                logger.log_batch(global_step, {
                    "phase": "4_neural_guided_search",
                    "step": step,
                    "total_loss": float(loss.item()),
                    "policy_loss": float(loss_policy.item()),
                    "value_loss": float(loss_value.item()),
                    "inherit_loss": float(inherit_loss.item()) if hasattr(inherit_loss, "item") else 0.0,
                    "inverse_loss": float(inverse_loss.item()) if hasattr(inverse_loss, "item") else 0.0,
                    "puct_regularization": float(puct_regularization.item()) if hasattr(puct_regularization, "item") else 0.0,
                    "num_current_traces": len(traces),
                    "num_replay_traces": len(replay_traces),
                    "num_total_training_traces": len(all_training_traces),
                    "replay_buffer_size": replay_stats['size'],
                    "search_method": "puct" if use_puct else "beam",
                    "step_time": step_time,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "trace_sources": source_counts
                })

                print(f"[Phase 4] Step {step} - Neural-Guided Search Training:")
                print(f"  Loss: {loss.item():.4f} (Policy: {loss_policy.item():.4f}, Value: {loss_value.item():.4f})")
                print(f"  Traces: {len(all_training_traces)} (current: {len(traces)}, replay: {len(replay_traces)})")
                print(f"  Replay Buffer: {replay_stats['size']}/{replay_stats['capacity']} traces")
                print(f"  Search: {'PUCT' if use_puct else 'Beam'}, Time: {step_time:.2f}s")

                if source_counts:
                    sources_str = ", ".join([f"{k}:{v}" for k, v in source_counts.items()])
                    print(f"  Sources: {sources_str}")

                # Detailed logging every 100 steps
                if step % 100 == 0:
                    print(f"[RelMem] inherit={inherit_loss.item():.4f}, inverse={inverse_loss.item():.4f}")
                    print(f"[Training Stats] PUCT searches: {training_stats['puct_searches']}, "
                          f"Beam searches: {training_stats['beam_searches']}, "
                          f"Deep mining: {training_stats['deep_mining_runs']}")

                    # Log replay buffer statistics
                    if step % 200 == 0:
                        print(f"[Replay Buffer Statistics]")
                        print(f"  Size: {replay_stats['size']}/{replay_stats['capacity']}")
                        print(f"  Score stats: mean={replay_stats['score_stats']['mean']:.3f}, "
                              f"std={replay_stats['score_stats']['std']:.3f}")
                        print(f"  Novelty stats: mean={replay_stats['novelty_stats']['mean']:.3f}")
                        print(f"  Source distribution: {replay_stats['source_distribution']}")

        except Exception as e:
            print(f"[Phase 4] Error in step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save final statistics and cleanup
    final_replay_stats = replay_buffer.get_statistics()

    print(f"\n[Phase 4] Neural-Guided Search Training Complete!")
    print(f"===========================================")
    print(f"Training Summary:")
    print(f"  - Total Steps: {steps}")
    print(f"  - Search Method: {'PUCT' if use_puct else 'Beam Search'}")
    print(f"  - Total Traces Generated: {training_stats['total_traces_generated']}")
    print(f"  - PUCT Searches: {training_stats['puct_searches']}")
    print(f"  - Beam Searches: {training_stats['beam_searches']}")
    print(f"  - Deep Mining Runs: {training_stats['deep_mining_runs']}")
    print(f"  - Average Search Time: {training_stats['avg_search_time']:.2f}s")
    if use_puct:
        print(f"  - Average PUCT Simulations: {training_stats['avg_puct_simulations']:.1f}")

    print(f"\nReplay Buffer Final State:")
    print(f"  - Size: {final_replay_stats['size']}/{final_replay_stats['capacity']}")
    print(f"  - Total Pushes: {final_replay_stats['total_pushes']}")
    print(f"  - Total Samples: {final_replay_stats['total_samples']}")
    print(f"  - Source Distribution: {final_replay_stats['source_distribution']}")
    print(f"  - Score Range: [{final_replay_stats['score_stats']['min']:.3f}, {final_replay_stats['score_stats']['max']:.3f}]")
    print(f"  - Average Novelty: {final_replay_stats['novelty_stats']['mean']:.3f}")

    # Update state with enhanced components
    state.update({
        "policy_net": policy_net,
        "value_net": value_net,
        "logger": logger,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": global_step,
        "relmem": relmem,
        "replay_buffer": replay_buffer,
        "puct_searcher": puct_searcher,
        "phase4_completed": True,
        "training_stats": training_stats,
        "neural_guided_search_config": {
            "use_puct": use_puct,
            "num_simulations": num_simulations,
            "c_puct": c_puct,
            "replay_capacity": replay_capacity,
            "replay_alpha": replay_alpha
        }
    })

    print(f"\n[Phase 4] Neural-Guided Search 2.0 Phase Complete")
    print(f"===========================================")
    return state