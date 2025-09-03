"""
Phase 2 – Meta-Learning (MAML/Reptile)
Enable fast adaptation to new tasks with few updates.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.arc_dataset_loader import ARCDataset
from models.topas_arc_60M import TopasARC60M, ModelConfig
from trainers.meta_learner import MetaLearner
from relational_memory_neuro import RelationalMemoryNeuro

def run(config, state):
    """
    Phase 2: Meta Learning with MAML/Reptile
    Fixed to use trainer_utils helpers consistently.
    
    Args:
        config: Configuration dict with hyperparameters
        state: Persistent state dict across phases
        
    Returns:
        Updated state dict
    """
    from trainers.trainer_utils import filter_config, safe_model_forward, compute_ce_loss, default_sample_fn
    
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Use filter_config helper to safely filter ModelConfig fields
    model_cfg_dict = filter_config(config, ModelConfig.__annotations__)
    
    # Model: reuse if passed from state or create new
    model = state.get("model")
    if model is None:
        try:
            cfg = ModelConfig(**model_cfg_dict)
            cfg.validate()
            model = TopasARC60M(cfg).to(device)
        except Exception as e:
            print(f"[Phase 2] Error creating model: {e}. Using default config.")
            cfg = ModelConfig()  # Use defaults
            model = TopasARC60M(cfg).to(device)
    
    # Create MetaLearningConfig
    from trainers.meta_learner import MetaLearningConfig
    meta_config = MetaLearningConfig(
        outer_lr=config.get("meta_lr", 1e-3),
        inner_lr=config.get("inner_lr", 1e-2),
        inner_steps=config.get("inner_steps", 5),
        first_order=config.get("first_order", True)
    )
    
    # Initialize RelationalMemoryNeuro
    if "relmem" not in state:
        state["relmem"] = RelationalMemoryNeuro(
            hidden_dim=model.config.slot_dim,
            max_concepts=4096,
            device=device
        ).to(device)
    relmem = state["relmem"]
    
    # Wrap model with MetaLearner (correct parameter name is base_model)
    meta_learner = MetaLearner(
        base_model=model,
        config=meta_config
    )
    
    # Logger: reuse or create
    logger = state.get("logger")
    if logger is None:
        from trainers.train_logger import TrainLogger
        logger = TrainLogger(config.get("log_dir", "logs"))
    
    # Initialize if dry_run
    if config.get("dry_run", False):
        print("[Phase 2] Dry run mode - skipping actual training")
        state["model"] = model
        state["meta_learner"] = meta_learner
        state["logger"] = logger
        state["phase2_completed"] = True
        return state
    
    # Dataset setup
    try:
        dataset = ARCDataset(
            challenge_file=config.get("train_challenges", "arc-agi_training-challenges.json"),
            solution_file=config.get("train_solutions", "arc-agi_training-solutions.json"),
            device=str(device),
            max_grid_size=config.get("max_grid_size", 30)
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"[Phase 2] ARC dataset files not found: {e}. Cannot train ARC solver without ARC data! Fix dataset paths in config.")
    
    # Training loop
    num_epochs = config.get("num_epochs", 5)
    tasks_per_meta_batch = config.get("tasks_per_meta_batch", 4)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_tasks = 0
        
        # Collect tasks for meta-batch
        task_batch = []
        
        for task_idx, task_data in enumerate(dataloader):
            if len(dataloader) == 0:
                break
                
            task_batch.append(task_data)
            
            # Process meta-batch when full
            if len(task_batch) >= tasks_per_meta_batch:
                # Convert task data to Episodes for MetaLearner
                episodes = []
                for task_data in task_batch:
                    demos, test_inputs, test_outputs, task_id = task_data
                    
                    # PROPER FIDELITY: Convert tuple demos to dict format for MetaLearner
                    demo_dicts = []
                    for demo in demos:
                        if isinstance(demo, tuple) and len(demo) >= 2:
                            # Convert (input_tensor, output_tensor) → {"input": ..., "output": ...}
                            demo_dicts.append({"input": demo[0], "output": demo[1]})
                        elif isinstance(demo, dict) and 'input' in demo and 'output' in demo:
                            # Already in correct format
                            demo_dicts.append(demo)
                        else:
                            print(f"[Phase 2] Skipping invalid demo format: {type(demo)}")
                            continue
                    
                    # Clamp support/query targets for each demo
                    for demo in demo_dicts:
                        if isinstance(demo, dict):
                            if "output" in demo and torch.is_tensor(demo["output"]):
                                num_classes = 10  # ARC has 10 categorical classes
                                if (demo["output"] < 0).any() or (demo["output"] >= num_classes).any():
                                    print(f"[DEBUG][Phase2] Clamping invalid support/query outputs: min={demo['output'].min().item()}, max={demo['output'].max().item()}")
                                demo["output"] = demo["output"].clamp(0, num_classes - 1)
                    
                    # Split demos into support (first part) and query (rest) 
                    support_set = demo_dicts[:2] if len(demo_dicts) > 2 else demo_dicts
                    query_set = demo_dicts[2:] if len(demo_dicts) > 2 else []
                    
                    # Add test data to query if available
                    if test_inputs and test_outputs:
                        for test_in, test_out in zip(test_inputs, test_outputs):
                            query_set.append({"input": test_in, "output": test_out})
                    
                    # Create Episode
                    from trainers.meta_learner import Episode
                    episode = Episode(
                        task_id=task_id,
                        support_set=support_set,
                        query_set=query_set,
                        difficulty=0.5,  # Default difficulty
                        composition_type="unknown"
                    )
                    episodes.append(episode)
                
                # Run meta-learning step
                meta_metrics = meta_learner.outer_loop(episodes)
                meta_loss = meta_metrics.get("meta_loss", 0.0)
                
                # Add RelMem losses
                inherit_loss = relmem.inheritance_pass()
                inverse_loss = relmem.inverse_loss()
                meta_loss = meta_loss + 0.05 * inherit_loss + 0.05 * inverse_loss
                
                # Apply post-optimizer hooks (meta-learner handles optimization internally)
                if hasattr(relmem, "apply_post_optimizer_hooks"):
                    relmem.apply_post_optimizer_hooks()
                
                epoch_loss += meta_loss
                num_tasks += len(task_batch)
                
                # Log progress
                avg_loss = epoch_loss / max(1, num_tasks)
                print(f"[Phase 2] Epoch {epoch+1}/{num_epochs}, Tasks {num_tasks}, Meta Loss: {avg_loss:.4f}")
                logger.log({
                    "phase": 2,
                    "epoch": epoch,
                    "num_tasks": num_tasks,
                    "meta_loss": avg_loss,
                    "adaptation_success_rate": meta_metrics.get("adaptation_success_rate", 0.0),
                    "inherit_loss": float(inherit_loss.item()) if hasattr(inherit_loss, "item") else 0.0,
                    "inverse_loss": float(inverse_loss.item()) if hasattr(inverse_loss, "item") else 0.0
                })
                
                # RelMem logging every 100 tasks
                if num_tasks % 100 == 0:
                    print(f"[RelMem] inherit={inherit_loss.item():.4f}, inverse={inverse_loss.item():.4f}")
                
                # Clear batch
                task_batch = []
            
            # Early stop for dry run
            if num_tasks >= 10 and config.get("dry_run", False):
                break
        
        # Process remaining tasks
        if task_batch:
            episodes = []
            for demos, test_inputs, test_outputs, task_id in task_batch:
                demo_dicts = [{"input": d[0], "output": d[1]} for d in demos if isinstance(d, tuple) and len(d) >= 2]
                query_set = []
                if test_inputs and test_outputs:
                    for ti, to in zip(test_inputs, test_outputs):
                        query_set.append({"input": ti, "output": to})
                from trainers.meta_learner import Episode
                episode = Episode(task_id=task_id, support_set=demo_dicts, query_set=query_set, difficulty=0.5, composition_type="unknown")
                episodes.append(episode)
            meta_metrics = meta_learner.outer_loop(episodes)
            meta_loss = meta_metrics.get("meta_loss", 0.0)
            
            # Add RelMem losses for remaining batch
            inherit_loss = relmem.inheritance_pass()
            inverse_loss = relmem.inverse_loss()
            meta_loss = meta_loss + 0.05 * inherit_loss + 0.05 * inverse_loss
            
            # Apply post-optimizer hooks
            if hasattr(relmem, "apply_post_optimizer_hooks"):
                relmem.apply_post_optimizer_hooks()
            epoch_loss += meta_loss
            num_tasks += len(task_batch)
        
        # Epoch summary
        if num_tasks > 0:
            avg_epoch_loss = epoch_loss / num_tasks
            print(f"[Phase 2] Epoch {epoch+1} completed. Avg Meta Loss: {avg_epoch_loss:.4f}")
        else:
            avg_epoch_loss = 0.0
            print(f"[Phase 2] Epoch {epoch+1} completed. No tasks processed.")
    
    # Update state
    state["model"] = model
    state["meta_learner"] = meta_learner
    state["logger"] = logger
    state["relmem"] = relmem
    state["phase2_completed"] = True
    state["phase2_epochs"] = num_epochs
    state["phase2_final_loss"] = avg_epoch_loss if 'avg_epoch_loss' in locals() else 0.0
    
    print("[Phase 2] Meta Learning completed!")
    return state