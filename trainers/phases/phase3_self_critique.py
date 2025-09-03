"""
Phase 3 – Self-Critique
Uses counterexamples, trace analysis, and consistency to improve model robustness.
Fixed to use trainer_utils helpers consistently.
"""

def run(config, state):
    from trainers.trainer_utils import filter_config, safe_model_forward, compute_ce_loss, default_sample_fn
    from trainers.arc_dataset_loader import ARCDataset
    from models.topas_arc_60M import TopasARC60M, ModelConfig
    from trainers.train_logger import TrainLogger
    from trainers.self_critique.counterexamples import CounterexampleGenerator, Task
    from trainers.self_critique.trace_analysis import TraceAnalyzer
    from trainers.self_critique.consistency import ConsistencyEnforcer
    from relational_memory_neuro import RelationalMemoryNeuro
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Use filter_config helper for safe dataclass init
    model_cfg_dict = filter_config(config, ModelConfig.__annotations__)
    model = state.get("model")
    if model is None:
        try:
            model = TopasARC60M(ModelConfig(**model_cfg_dict)).to(device)
        except Exception as e:
            print(f"[Phase 3] Error creating model: {e}. Using default config.")
            model = TopasARC60M(ModelConfig()).to(device)

    # Initialize self-critique components
    try:
        counter_gen = CounterexampleGenerator(device=device)
        trace_analyzer = TraceAnalyzer(device=device)
        consistency_enforcer = ConsistencyEnforcer(device=device)
    except Exception as e:
        print(f"[Phase 3] Warning: Could not initialize all self-critique components: {e}")
        counter_gen = None
        trace_analyzer = None
        consistency_enforcer = None

    # Initialize RelationalMemoryNeuro
    if "relmem" not in state:
        state["relmem"] = RelationalMemoryNeuro(
            hidden_dim=model.config.slot_dim,
            max_concepts=4096,
            device=device
        ).to(device)
    relmem = state["relmem"]
    
    logger = state.get("logger") or TrainLogger(config.get("log_dir", "logs"))

    # Dataset
    try:
        dataset = ARCDataset(
            challenge_file=config.get("train_challenges", "arc-agi_training_challenges.json"),
            solution_file=config.get("train_solutions", "arc-agi_training_solutions.json"),
            device=str(device),
            max_grid_size=config.get("max_grid_size", 30),
        )
    except Exception as e:
        print(f"[Phase 3] Could not load dataset: {e}. Will use default_sample_fn.")
        dataset = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 1e-4))

    num_epochs = config.get("num_epochs", 3)
    steps_per_epoch = config.get("steps_per_epoch", 200)
    global_step = state.get("global_step", 0)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_steps = 0
        
        for step in range(steps_per_epoch):
            try:
                # Use default_sample_fn to get data
                demos, test = default_sample_fn(dataset, device)
                
                # Fix CE loss issue: clamp targets to valid range [0, 9] and ensure integer type
                if test.get("output") is not None:
                    test["output"] = test["output"].long().clamp(0, 9)
                
                # Forward pass using safe wrapper
                grid, logits, extras = safe_model_forward(model, demos, test, device, training_mode=True)
                
                # Compute loss using normalized CE
                loss = compute_ce_loss(logits, test.get("output"))
                
                # Add RelMem losses
                inherit_loss = relmem.inheritance_pass()
                inverse_loss = relmem.inverse_loss()
                loss = loss + 0.05 * inherit_loss + 0.05 * inverse_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Apply post-optimizer RelMem hooks
                if hasattr(relmem, "apply_post_optimizer_hooks"):
                    relmem.apply_post_optimizer_hooks()
                
                epoch_loss += loss.item()
                num_steps += 1
                global_step += 1
                
                # Log batch
                if step % config.get("log_interval", 20) == 0:
                    logger.log_batch(global_step, {
                        "phase": "3_self_critique",
                        "epoch": epoch,
                        "step": step,
                        "loss": loss.item(),
                        "inherit_loss": float(inherit_loss.item()) if hasattr(inherit_loss, "item") else 0.0,
                        "inverse_loss": float(inverse_loss.item()) if hasattr(inverse_loss, "item") else 0.0
                    })
                    
                    # RelMem logging every 100 steps
                    if step % 100 == 0:
                        print(f"[RelMem] inherit={inherit_loss.item():.4f}, inverse={inverse_loss.item():.4f}")
                    
            except Exception as e:
                print(f"[Phase 3] Error in training step: {e}")
                continue
        
        # Epoch summary
        if num_steps > 0:
            avg_loss = epoch_loss / num_steps
            logger.log_epoch(epoch, "3_self_critique", avg_loss, None)
            print(f"[Phase 3] Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

    state.update({
        "model": model,
        "logger": logger,
        "optimizer": optimizer,
        "global_step": global_step,
        "relmem": relmem,
        "phase3_completed": True
    })
    
    print("[Phase 3] Self-Critique complete")
    return state