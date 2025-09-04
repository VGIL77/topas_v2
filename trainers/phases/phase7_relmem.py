"""
Phase 7 – Relational Memory Integration
Adds RelMem residuals, aux inverse-consistency loss, and Hebbian/WTA plasticity.
Fixed to use trainer_utils helpers consistently.
"""

def run(config, state):
    from trainers.trainer_utils import default_sample_fn, safe_model_forward, compute_ce_loss
    from trainers.train_logger import TrainLogger
    from validation.eval_runner import EvalRunner
    import torch

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Get model from state
    model = state.get("model")
    if model is None:
        print("[Phase 7] Model not found in state. Run Phase 0 first.")
        return state
    
    logger = state.get("logger") or TrainLogger(config.get("log_dir", "logs"))
    
    # Get or create optimizer
    optimizer = state.get("optimizer") or torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Dataset
    dataset = state.get("dataset")
    
    epochs = config.get("epochs", 5)
    steps_per_epoch = config.get("steps_per_epoch", 200)
    global_step = state.get("global_step", 0)
    
    print(f"[Phase 7] Starting RelMem Integration for {epochs} epochs")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
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
                    raise RuntimeError("[Phase 7] No test output found in ARC data. Cannot train relational memory without valid targets!")
                
                # Main CE loss using compute_ce_loss helper
                ce_loss = compute_ce_loss(logits, target)
                
                # Add RelMem inverse-consistency aux loss if enabled
                aux_loss = torch.tensor(0.0, device=device)
                if hasattr(model, "relmem") and model.relmem is not None:
                    try:
                        inv_w = float(config.get("inverse_loss_w", 0.05))
                        if hasattr(model.relmem, "inverse_loss"):
                            aux_loss = inv_w * model.relmem.inverse_loss()
                    except Exception as e:
                        print(f"[Phase 7] RelMem aux loss error: {e}")
                        aux_loss = torch.tensor(0.0, device=device)
                
                # Total loss
                loss = ce_loss + aux_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Post-opt Hebbian/WTA plasticity
                if hasattr(model, "relmem") and model.relmem is not None:
                    try:
                        if hasattr(model.relmem, "apply_hebbian"):
                            model.relmem.apply_hebbian()
                        if hasattr(model.relmem, "apply_wta"):
                            model.relmem.apply_wta()
                    except Exception as e:
                        print(f"[Phase 7] RelMem plasticity error: {e}")
                
                epoch_loss += ce_loss.item()
                num_steps += 1
                global_step += 1
                
                # Logging
                if step % config.get("log_interval", 20) == 0:
                    logger.log_batch(global_step, {
                        "phase": "7_relmem",
                        "epoch": epoch,
                        "step": step,
                        "total": loss.item(),
                        "painter": ce_loss.item(),
                        "relmem_aux": float(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)
                    })
                    
            except Exception as e:
                print(f"[Phase 7] Error in step {step}: {e}")
                continue
        
        # Epoch summary
        if num_steps > 0:
            avg_loss = epoch_loss / num_steps
            logger.log_epoch(epoch, "7_relmem", avg_loss, {"steps": num_steps})
            print(f"[Phase 7] Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
            
            # === Evaluate at the end of each RelMem epoch ===
            try:
                print(f"[Phase 7] Running evaluation after epoch {epoch+1}...")
                eval_runner = EvalRunner(model=model, device=device)
                metrics = eval_runner.run(
                    "ARC/arc-agi_evaluation_challenges.json",
                    "ARC/arc-agi_evaluation_solutions.json"
                )
                logger.log_batch(global_step, {
                    "eval_relmem_exact1": metrics.get("exact@1", 0.0),
                    "eval_relmem_exact_k": metrics.get("exact@k", 0.0),
                    "eval_relmem_iou": metrics.get("iou", 0.0),
                    "phase": "7_relmem_eval",
                    "epoch": epoch
                })
                print(f"[Phase 7] RelMem Eval - Exact@1: {metrics.get('exact@1', 0.0):.2%}, IoU: {metrics.get('iou', 0.0):.3f}")
            except Exception as e:
                print(f"[Phase 7] Evaluation error after epoch {epoch}: {e}")
    
    state.update({
        "model": model,
        "optimizer": optimizer,
        "logger": logger,
        "global_step": global_step,
        "phase7_complete": True
    })
    
    print("[Phase 7] RelMem Integration complete")
    return state