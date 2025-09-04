"""
Phase 10 – Production Trainer
Full SGI/neuromorphic training pipeline for TOPAS.
Enhanced with complete neuroscientific dream system.
"""

def run(config, state):
    from trainers.trainer_utils import default_sample_fn, safe_model_forward, compute_ce_loss, filter_config
    from trainers.train_logger import TrainLogger
    from trainers.sgi_optimizer import SGIOptimizer, OptimizerConfig
    from trainers.ensemble_solver import EnsembleSolver
    from validation.eval_runner import EvalRunner
    from gccrf_curiosity import GCCRFCuriosity
    from nmda_dreaming import NMDAGatedDreaming
    from emergent_theme_synthesis import EmergentThemeSynthesis
    from ripple_substrate import create_default_ripple_substrate
    from wormhole_offline import WormholeTemplateMiner
    import torch
    import os

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Get model from state
    model = state.get("model")
    if model is None:
        print("[Phase 10] Model not found in state. Run Phase 0 first.")
        return state
    
    model = model.to(device)
    logger = state.get("logger") or TrainLogger(log_path="logs/phase10.jsonl")
    
    # === Initialize Dream subsystems if missing ===
    if "dream" not in state:
        from dream_engine import DreamEngine
        state["dream"] = DreamEngine(device=device)

    if "gccrf" not in state:
        state["gccrf"] = GCCRFCuriosity(state_dim=model.config.slot_dim).to(device)

    if "nmda" not in state:
        state["nmda"] = NMDAGatedDreaming(state_dim=model.config.slot_dim,
                                          action_dim=model.config.dsl_vocab_size,
                                          device=device).to(device)

    if "themes" not in state:
        state["themes"] = EmergentThemeSynthesis(embedding_dim=model.config.slot_dim)

    if "ripple" not in state:
        state["ripple"] = create_default_ripple_substrate()

    if "wormhole" not in state:
        state["wormhole"] = WormholeTemplateMiner()

    dream, gccrf, nmda, themes, ripple, wormhole = (
        state["dream"], state["gccrf"], state["nmda"], state["themes"], state["ripple"], state["wormhole"]
    )

    # Attach optimizer with filtered config
    try:
        optimizer_config_dict = filter_config(
            config.get("optimizer_config", {}), 
            OptimizerConfig.__annotations__
        )
        optimizer_config = OptimizerConfig(**optimizer_config_dict)
        sgi_opt = SGIOptimizer(model, optimizer_config)
    except Exception as e:
        print(f"[Phase 10] Error creating SGI optimizer: {e}. Using defaults.")
        optimizer_config = OptimizerConfig()
        sgi_opt = SGIOptimizer(model, optimizer_config)
    
    state["sgi_optimizer"] = sgi_opt

    # Ensemble solver
    ensemble = state.get("ensemble_solver")
    if ensemble is None:
        try:
            ensemble = EnsembleSolver(model, config.get("ensemble_config", {}))
        except Exception as e:
            print(f"[Phase 10] Could not create ensemble solver: {e}")
            ensemble = None
    state["ensemble_solver"] = ensemble

    # DreamEngine, RelMem, Scheduler should already be attached in state
    dream = state.get("dream")
    scheduler = state.get("scheduler")
    
    # Dataset
    dataset = state.get("dataset")

    epochs = config.get("epochs", 20)
    steps_per_epoch = config.get("steps_per_epoch", 400)
    save_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    global_step = state.get("global_step", 0)
    
    print(f"[Phase 10] Starting Production Training for {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0.0
        num_steps = 0
        
        for step in range(steps_per_epoch):
            try:
                # Use default_sample_fn instead of state["sample_fn"]
                demos, test = default_sample_fn(dataset, device)
                
                # Fix CE loss issue: clamp targets to valid range [0, 9] and ensure integer type
                if test.get("output") is not None:
                    test["output"] = test["output"].long().clamp(0, 9)
                
                # Forward pass using safe wrapper
                grid, logits, extras = safe_model_forward(model, demos, test, device, training_mode=True)
                
                # Get target
                target = test.get("output")
                if target is None:
                    raise RuntimeError("[Phase 10] No test output found in ARC data. Cannot run production without valid targets!")
                
                # Extract latents for dream system
                latents = extras.get("latent") if extras else None
                
                # Curiosity-driven valence/arousal
                if latents is not None:
                    R_i, curiosity_info = gccrf.compute_reward(latents.flatten(1))
                    valence = curiosity_info["prediction_error"].item()
                    arousal = curiosity_info["novelty"].item()
                    
                    # Run ripple update
                    ripple.update(step * config.get("dt", 0.01))
                else:
                    valence, arousal = 0.5, 0.3  # fallback
                    curiosity_info = {"prediction_error": torch.tensor(0.0), "novelty": torch.tensor(0.0)}
                
                # Main CE loss using compute_ce_loss helper
                ce_loss = compute_ce_loss(logits, target)
                
                # SGI optimization (with advanced regularization)
                try:
                    metrics = sgi_opt.step(ce_loss)
                except Exception as e:
                    print(f"[Phase 10] SGI step error: {e}")
                    # Fallback to regular optimization
                    ce_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    sgi_opt.optimizer.step()
                    sgi_opt.optimizer.zero_grad()
                    metrics = {"lr": 1e-4, "grad_norm_before": 0.0}
                
                # Hebbian/WTA plasticity post-step if RelMem exists
                if hasattr(model, "relmem") and model.relmem is not None:
                    try:
                        if hasattr(model.relmem, "apply_hebbian"):
                            model.relmem.apply_hebbian()
                        if hasattr(model.relmem, "apply_wta"):
                            model.relmem.apply_wta()
                    except Exception as e:
                        print(f"[Phase 10] RelMem plasticity error: {e}")
                
                # Advanced dream system integration
                if step % config.get("dream_interval", 100) == 0 and latents is not None:
                    try:
                        # Dream replay with real latents
                        replay_stats = dream.cycle_offline(latents, demos, valence=valence, arousal=arousal)

                        # NMDA consolidation
                        nmda_loss = nmda.dream_consolidation(valence, arousal, ripple_ctx=ripple.get_current_context())

                        # Theme synthesis
                        if target is not None:
                            motifs = themes.process_dream_themes(latents.unsqueeze(0), target.view(-1))
                            themes.synthesize_emergent_themes(motifs)

                        # Wormhole mining every N epochs
                        if epoch % config.get("wormhole_interval", 10) == 0:
                            wormhole_results = wormhole.mine_from_programs(demos, top_k=5)

                    except Exception as e:
                        print(f"[Phase 10] Advanced dream system error: {e}")
                
                total_loss += ce_loss.item()
                num_steps += 1
                global_step += 1
                
                # Enhanced logging with dream metrics
                if step % config.get("log_interval", 50) == 0:
                    logger.log_batch(global_step, {
                        "phase": "10_production",
                        "epoch": epoch,
                        "step": step,
                        "total": ce_loss.item(),
                        "lr": metrics.get("lr", 0.0),
                        "grad_norm": metrics.get("grad_norm_before", 0.0)
                    }, dream_info={
                        "dream_valence": valence,
                        "dream_arousal": arousal,
                        "nmda_loss": float(nmda_loss) if 'nmda_loss' in locals() and isinstance(nmda_loss, torch.Tensor) else 0.0,
                        "num_themes": len(themes.themes) if hasattr(themes, 'themes') else 0,
                        "curiosity_pred_error": curiosity_info["prediction_error"].item(),
                        "curiosity_novelty": curiosity_info["novelty"].item(),
                        "ripple_phase": ripple.get_current_context() if hasattr(ripple, 'get_current_context') else 0.0
                    })
                    
            except Exception as e:
                print(f"[Phase 10] Error in step {step}: {e}")
                continue
        
        # Epoch summary
        if num_steps > 0:
            avg_loss = total_loss / num_steps
            logger.log_epoch(epoch, "10_production", avg_loss, {"steps": num_steps})
            print(f"[Phase 10] Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
        
        # Checkpointing
        if epoch % config.get("save_interval", 5) == 0:
            ckpt_path = os.path.join(save_dir, f"topas_epoch_{epoch}.pt")
            try:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "avg_loss": avg_loss if num_steps > 0 else 0.0
                }, ckpt_path)
                print(f"[Phase 10] Saved checkpoint to {ckpt_path}")
            except Exception as e:
                print(f"[Phase 10] Checkpoint save error: {e}")

        # Ensemble solving + validation at eval interval
        if (epoch + 1) % config.get("eval_interval", 5) == 0:
            try:
                eval_runner = EvalRunner(model=model, device=device)
                results = eval_runner.run(
                    "ARC/arc-agi_evaluation_challenges.json",
                    "ARC/arc-agi_evaluation_solutions.json"
                )
                
                print(f"[Phase 10] Validation Results:")
                print(f"  Exact@1: {results.get('exact@1', 0.0):.2%}")
                print(f"  Exact@K: {results.get('exact@k', 0.0):.2%}")
                print(f"  IoU: {results.get('iou', 0.0):.3f}")
                
                logger.log_milestone("Validation", results)
            except Exception as e:
                print(f"[Phase 10] Validation error: {e}")

    state.update({
        "model": model,
        "logger": logger,
        "global_step": global_step,
        "dream": dream,
        "gccrf": gccrf,
        "nmda": nmda,
        "themes": themes,
        "ripple": ripple,
        "wormhole": wormhole,
        "phase10_complete": True
    })
    
    print("[Phase 10] Production Training complete")
    return state