"""
Phase 4 – MCTS Alpha
Combines MCTS search with neural policy/value networks.
Fixed to use trainer_utils helpers consistently.
"""

def run(config, state):
    from trainers.trainer_utils import default_sample_fn, program_to_tensor, compute_ce_loss, safe_model_forward
    from trainers.train_logger import TrainLogger
    from models.dsl_search import beam_search
    from trainers.augmentation.deep_program_discoverer import mine_deep_programs
    from relational_memory_neuro import RelationalMemoryNeuro
    import torch

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
    
    # Initialize RelationalMemoryNeuro
    if "relmem" not in state:
        state["relmem"] = RelationalMemoryNeuro(
            hidden_dim=model.config.slot_dim,
            max_concepts=4096,
            device=device
        ).to(device)
    relmem = state["relmem"]
    
    logger = state.get("logger") or TrainLogger(config.get("log_dir", "logs"))
    dataset = state.get("dataset")

    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()), 
        lr=config.get("learning_rate", 1e-4)
    )
    
    # Setup vocab for program encoding
    vocab = {
        "<PAD>": 0, "<UNK>": 1, "rotate": 2, "flip": 3, "color_map": 4,
        "copy": 5, "fill": 6, "move": 7, "resize": 8, "crop": 9
    }

    steps = config.get("steps", 200)
    global_step = state.get("global_step", 0)
    
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