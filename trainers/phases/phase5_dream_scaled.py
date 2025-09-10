"""
Phase 5 – Dream-Scaled Training
Incorporates Dream micro-ticks and full cycles with enhanced logging.
"""

def run(config, state):
    from trainers.trainer_utils import default_sample_fn, safe_model_forward, compute_ce_loss, filter_config
    from trainers.train_logger import TrainLogger
    from trainers.sgi_optimizer import SGIOptimizer, OptimizerConfig
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
        print("[Phase 5] Model not found in state. Run Phase 0 first.")
        return state
    
    model = model.to(device)
    logger = state.get("logger") or TrainLogger(log_path="logs/phase5_dream.jsonl")
    
    # === Initialize Dream subsystems if missing ===
    if "dream" not in state:
        from dream_engine import DreamEngine, DreamConfig
        dream_cfg = DreamConfig(state_dim=model.config.slot_dim, device=str(device))
        state["dream"] = DreamEngine(dream_cfg)

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

    # Attach optimizer
    try:
        optimizer_config_dict = filter_config(
            config.get("optimizer_config", {}), 
            OptimizerConfig.__annotations__
        )
        optimizer_config = OptimizerConfig(**optimizer_config_dict)
        sgi_opt = SGIOptimizer(model, optimizer_config)
    except Exception as e:
        print(f"[Phase 5] Error creating SGI optimizer: {e}. Using defaults.")
        optimizer_config = OptimizerConfig()
        sgi_opt = SGIOptimizer(model, optimizer_config)
    
    state["sgi_optimizer"] = sgi_opt
    
    # Dataset
    dataset = state.get("dataset")

    epochs = config.get("epochs", 20)
    steps_per_epoch = config.get("steps_per_epoch", 400)
    save_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    global_step = state.get("global_step", 0)
    
    print(f"[Phase 5] Starting Dream-Scaled Training for {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0.0
        
        # Track motif events
        motifs_added_epoch = 0
        motifs_rejected_epoch = 0
        
        for step in range(steps_per_epoch):
            global_step += 1
            
            # Sample from dataset
            demos, test, target = default_sample_fn(dataset, device)
            
            # Core forward pass
            loss, info = safe_model_forward(model, demos, test, target, device)
            
            # Compute enhanced dream info
            dream_info = {}
            if nmda and hasattr(nmda, 'buffer'):
                dream_info['buffer_len'] = len(nmda.buffer)
            if nmda and hasattr(nmda, 'compute_loss'):
                with torch.no_grad():
                    dream_info['nmda_loss'] = nmda.compute_loss().item() if nmda.buffer else 0.0
            if themes:
                dream_info['num_themes'] = len(themes.themes)
                # Track synthesis events
                if hasattr(themes, 'synthesis_count'):
                    new_added = themes.synthesis_count - motifs_added_epoch
                    if new_added > 0:
                        motifs_added_epoch = themes.synthesis_count
                        dream_info['motifs_added'] = new_added
                    else:
                        dream_info['motifs_rejected'] = 1  # Simplified tracking
            if gccrf and hasattr(gccrf, 'prediction_error'):
                dream_info['curiosity_pred_error'] = gccrf.prediction_error
                dream_info['curiosity_novelty'] = getattr(gccrf, 'novelty_score', 0.0)
            if ripple and hasattr(ripple, 'phase'):
                dream_info['ripple_phase'] = ripple.phase
            
            # Log with enhanced info
            if step % config.get("log_interval", 50) == 0:
                relmem_info = None
                try:
                    if hasattr(model, "relmem") and model.relmem is not None:
                        relmem_info = model.relmem.stats()
                except Exception:
                    relmem_info = None
                logger.log_batch(global_step, {"total": loss.item()}, dream_info=dream_info, relmem_info=relmem_info)
            
            # Dream micro-ticks
            if dream and config.get("dream_micro_ticks", 0) > 0:
                for _ in range(config.get("dream_micro_ticks", 1)):
                    dream.tick()
            
            # Backward pass
            sgi_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            sgi_opt.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / steps_per_epoch
        
        # Full dream cycle
        if dream and epoch % config.get("dream_cycle_interval", 10) == 0:
            print(f"[Phase 5] Running full dream cycle at epoch {epoch}")
            dream_stats = dream.full_cycle(num_iterations=config.get("dream_iters", 50))
            print(f"[Phase 5] Dream cycle stats: {dream_stats}")
            
            # Log dream cycle completion
            logger.log_epoch(epoch, {"dream_cycle": 1, "avg_loss": avg_loss})
        
        # Save checkpoint
        if epoch % config.get("save_interval", 5) == 0:
            checkpoint_path = os.path.join(save_dir, f"phase5_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Phase 5] Saved checkpoint to {checkpoint_path}")
    
    print(f"[Phase 5] Training complete. Global step: {global_step}")
    state["global_step"] = global_step
    
    return state