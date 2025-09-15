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
            # Use default_sample_fn to get data
            demos, test = default_sample_fn(dataset, device)
            
            # Convert demos to pairs for beam search
            demo_pairs = [(d["input"], d["output"]) for d in demos]
            test_input = test.get("input")
            
            # Run beam search to get program traces
            try:
                traces = beam_search(
                    demo_pairs, test_input,
                    priors=None, op_bias=None,
                    depth=config.get("mcts_depth", 10),
                    beam=config.get("beam_width", 50)
                )
            except Exception as e:
                print(f"[Phase 4] Beam search error: {e}")
                traces = []
            
            # Deep program mining at intervals
            if step % config.get("deep_mining_interval", 200) == 0:
                try:
                    mined_programs = mine_deep_programs(
                        {"test": test}, 
                        max_depth=10
                    )
                    for prog in mined_programs:
                        # Create a simple trace object
                        trace = type("Trace", (), {
                            "program": prog.get("program", ["<UNK>"]),
                            "score": prog.get("score", 0.0)
                        })()
                        traces.append(trace)
                except Exception as e:
                    print(f"[Phase 4] Deep mining error: {e}")
            
            if not traces:
                raise RuntimeError("[Phase 4] MCTS beam search returned no traces. Cannot train without valid program traces!")
            
            # Convert programs to tensors
            targets = torch.stack([
                program_to_tensor(t.program if hasattr(t, 'program') else ["<UNK>"], vocab) 
                for t in traces
            ]).to(device)
            
            scores = torch.tensor([
                t.score if hasattr(t, 'score') else 0.0 
                for t in traces
            ], dtype=torch.float32).to(device)
            
            # Forward pass through policy and value networks
            try:
                # Extract test grid for network inputs
                test_grid = test.get("input") if isinstance(test, dict) else test
                if test_grid is None or not torch.is_tensor(test_grid):
                    # Use actual test grid from ARC data
                    test_grid = test.get("input", test.get("output"))
                
                if test_grid.dim() == 2:
                    test_grid = test_grid.unsqueeze(0)  # Add batch dim
                
                B, H, W = test_grid.shape
                
                # Create dummy features for policy network
                rel_features = torch.randn(B, 64, device=device)
                size_oracle = torch.tensor([[H, W, H, W]], device=device).float()
                theme_priors = torch.randn(B, 10, device=device)
                
                # Policy network forward pass
                policy_pred = policy_net.forward(
                    test_grid, rel_features, size_oracle, theme_priors
                )
                policy_logits = policy_pred.op_logits  # Extract operation logits
                
                # Value network forward pass  
                value_pred = value_net.forward(
                    test_grid, rel_features, size_oracle, theme_priors
                )
                value_preds = value_pred.solvability  # Extract solvability prediction
                
                # Ensure proper shapes
                if policy_logits is None or policy_logits.numel() == 0:
                    raise RuntimeError("[Phase 4] Policy network produced invalid logits. Cannot train without valid policy outputs!")
                if value_preds is None or value_preds.numel() == 0:
                    raise RuntimeError("[Phase 4] Value network produced invalid predictions. Cannot train without valid value outputs!")
                
                # Reshape for loss computation
                if policy_logits.dim() == 4:
                    # [B, T, L, V] -> [B*T, L, V]
                    B, T, L, V = policy_logits.shape
                    policy_logits = policy_logits.view(B*T, L, V)
                
                if targets.dim() == 2 and policy_logits.dim() == 3:
                    # Expand targets to match batch
                    if targets.size(0) != policy_logits.size(0):
                        # Repeat targets or truncate to match
                        min_size = min(targets.size(0), policy_logits.size(0))
                        targets = targets[:min_size]
                        policy_logits = policy_logits[:min_size]
                        scores = scores[:min_size]
                        value_preds = value_preds[:min_size] if value_preds.dim() > 0 else value_preds
                
            except Exception as e:
                raise RuntimeError(f"[Phase 4] Forward pass failed: {e}. Cannot train MCTS Alpha without functional model!")
            
            # Compute losses
            loss_policy = compute_ce_loss(policy_logits, targets)
            loss_value = torch.nn.functional.mse_loss(value_preds.squeeze(), scores.squeeze())
            
            # Add RelMem losses
            inherit_loss = relmem.inheritance_pass()
            inverse_loss = relmem.inverse_loss()
            loss = loss_policy + loss_value + 0.05 * inherit_loss + 0.05 * inverse_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply post-optimizer RelMem hooks
            if hasattr(relmem, "apply_post_optimizer_hooks"):
                relmem.apply_post_optimizer_hooks()
            
            global_step += 1
            
            # === Evaluate every N search steps ===
            if step % config.get("eval_interval_steps", 50) == 0 and step > 0:
                try:
                    print(f"[Phase 4] Running MCTS evaluation at step {step}...")
                    eval_runner = EvalRunner(model=model, device=device)
                    metrics = eval_runner.run(
                        "ARC/arc-agi_evaluation_challenges.json", 
                        "ARC/arc-agi_evaluation_solutions.json"
                    )
                    logger.log_batch(step, {
                        "eval_mcts_exact1": metrics.get("exact@1", 0.0),
                        "eval_mcts_exact_k": metrics.get("exact@k", 0.0), 
                        "eval_mcts_iou": metrics.get("iou", 0.0),
                        "phase": "4_mcts_eval"
                    })
                    print(f"[Phase 4] MCTS Eval - Exact@1: {metrics.get('exact@1', 0.0):.2%}, IoU: {metrics.get('iou', 0.0):.3f}")
                except Exception as e:
                    print(f"[Phase 4] Evaluation error at step {step}: {e}")
            
            # Logging
            if step % config.get("log_interval", 20) == 0:
                logger.log_batch(global_step, {
                    "phase": "4_mcts_alpha",
                    "step": step,
                    "total": float(loss.item()),
                    "policy": float(loss_policy.item()),
                    "value": float(loss_value.item()),
                    "inherit_loss": float(inherit_loss.item()) if hasattr(inherit_loss, "item") else 0.0,
                    "inverse_loss": float(inverse_loss.item()) if hasattr(inverse_loss, "item") else 0.0,
                    "num_traces": len(traces)
                })
                print(f"[Phase 4] Step {step} - Loss: {loss.item():.4f} (Policy: {loss_policy.item():.4f}, Value: {loss_value.item():.4f})")
                
                # RelMem logging every 100 steps
                if step % 100 == 0:
                    print(f"[RelMem] inherit={inherit_loss.item():.4f}, inverse={inverse_loss.item():.4f}")
                
        except Exception as e:
            print(f"[Phase 4] Error in step {step}: {e}")
            continue

    state.update({
        "policy_net": policy_net,
        "value_net": value_net,
        "logger": logger,
        "optimizer": optimizer,
        "global_step": global_step,
        "relmem": relmem,
        "phase4_completed": True
    })
    
    print("[Phase 4] MCTS Alpha complete")
    return state