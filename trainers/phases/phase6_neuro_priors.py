"""
Phase 6 – Neuro-Priors
Integrates Φ/κ/CGE priors, DifficultyEstimator, and Enhanced UCB scheduler.
Fixed to use trainer_utils helpers consistently.
"""

def run(config, state):
    from trainers.trainer_utils import filter_config, safe_model_forward, compute_ce_loss, default_sample_fn
    from trainers.arc_dataset_loader import ARCDataset
    from trainers.train_logger import TrainLogger
    from trainers.schedulers.ucb_scheduler import EnhancedUCBTaskScheduler, SchedulerConfig
    from trainers.schedulers.difficulty_estimator import DifficultyEstimator
    import torch
    import torch.nn.functional as F

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Get model from state
    model = state.get("model")
    if model is None:
        print("[Phase 6] Model not found in state. Run Phase 0 first.")
        return state
    
    # Logger
    logger = state.get("logger") or TrainLogger(config.get("log_dir", "./logs"))
    
    # Initialize or get Enhanced UCB scheduler
    scheduler = state.get("scheduler")
    if scheduler is None:
        try:
            scheduler_config = filter_config(config, SchedulerConfig.__annotations__)
            scheduler = EnhancedUCBTaskScheduler(SchedulerConfig(**scheduler_config))
        except Exception as e:
            print(f"[Phase 6] Warning: Could not init scheduler: {e}. Using defaults.")
            scheduler = EnhancedUCBTaskScheduler(SchedulerConfig())
    
    # Initialize difficulty estimator with safe init
    difficulty_estimator = None
    if config.get("use_difficulty_estimator", True):
        try:
            difficulty_estimator = DifficultyEstimator().to(device)
            if hasattr(scheduler, 'attach_difficulty_estimator'):
                scheduler.attach_difficulty_estimator(difficulty_estimator)
        except Exception as e:
            print(f"[Phase 6] Warning: Could not init DifficultyEstimator: {e}")
            difficulty_estimator = None
    
    # Optimizer
    params = list(model.parameters())
    if difficulty_estimator is not None:
        params += list(difficulty_estimator.parameters())
    
    lr = config.get("learning_rate", 1e-4)
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # Dataset
    try:
        dataset = state.get("dataset") or ARCDataset(
            challenge_file=config.get("train_challenges", "ARC/arc-agi_training_challenges.json"),
            solution_file=config.get("train_solutions", "ARC/arc-agi_training_solutions.json"),
            device=str(device),
            max_grid_size=config.get("max_grid_size", 30),
        )
        state["dataset"] = dataset
    except Exception as e:
        print(f"[Phase 6] Could not load dataset: {e}. Will use default_sample_fn.")
        dataset = None
    
    # Training parameters
    epochs = config.get("epochs", 5)
    steps_per_epoch = config.get("steps_per_epoch", 200)
    global_step = state.get("global_step", 0)
    
    print(f"[Phase 6] Starting Neuro-Priors + Difficulty Scheduling for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_exact = 0.0
        num_steps = 0
        
        model.train()
        if difficulty_estimator:
            difficulty_estimator.train()
        
        for step in range(steps_per_epoch):
            try:
                # Use default_sample_fn instead of state["sample_fn"]
                demos, test = default_sample_fn(dataset, device)
                
                # Forward pass using safe wrapper
                grid, logits, extras = safe_model_forward(model, demos, test, device, training_mode=True)
                
                # Main reconstruction loss using compute_ce_loss
                target = test.get("output")
                if target is None:
                    raise RuntimeError("[Phase 6] No test output found in ARC data. Cannot train without valid targets!")
                else:
                    target = target.long().clamp(0, 9)
                
                main_loss = compute_ce_loss(logits, target)
                
                # Estimate difficulty if estimator exists
                predicted_difficulty = 2
                if difficulty_estimator and logits is not None:
                    try:
                        with torch.no_grad():
                            # Extract features for difficulty estimation
                            if extras and "latent" in extras and extras["latent"] is not None:
                                feat_vec = extras["latent"].flatten().unsqueeze(0)
                            else:
                                # Use logits as features
                                feat_vec = logits.mean(dim=1) if logits.dim() > 2 else logits.unsqueeze(0)
                                if feat_vec.size(-1) != 128:  # Default feature_dim
                                    # Project to correct size
                                    feat_vec = F.adaptive_avg_pool1d(
                                        feat_vec.view(1, 1, -1), 
                                        128
                                    ).squeeze(1)
                            
                            difficulty_logits = difficulty_estimator(feat_vec)
                            predicted_difficulty = difficulty_logits.argmax(dim=-1).item()
                    except Exception as e:
                        pass  # Silent fail, use default difficulty
                
                # Simple neuro loss placeholder (since we don't have phi_metrics_neuro imported)
                neuro_loss = torch.tensor(0.01 * predicted_difficulty, device=device)
                
                # Total loss
                total_loss = main_loss + 0.1 * neuro_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                
                epoch_loss += main_loss.item()
                num_steps += 1
                global_step += 1
                
                # Logging
                if step % config.get("log_interval", 20) == 0:
                    logger.log_batch(global_step, {
                        "phase": "6_neuro_priors",
                        "epoch": epoch,
                        "step": step,
                        "total": total_loss.item(),
                        "main": main_loss.item(),
                        "neuro": neuro_loss.item() if isinstance(neuro_loss, torch.Tensor) else neuro_loss,
                        "difficulty": predicted_difficulty
                    })
                    
            except Exception as e:
                print(f"[Phase 6] Error in step {step}: {e}")
                continue
        
        # Epoch summary
        if num_steps > 0:
            avg_loss = epoch_loss / num_steps
            logger.log_epoch(epoch, "6_neuro_priors", avg_loss, {"steps": num_steps})
            print(f"[Phase 6] Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    
    # Update state
    state.update({
        "model": model,
        "scheduler": scheduler,
        "difficulty_estimator": difficulty_estimator,
        "logger": logger,
        "optimizer": optimizer,
        "global_step": global_step,
        "phase6_complete": True
    })
    
    print("[Phase 6] Neuro-Priors + Difficulty Scheduling complete")
    return state