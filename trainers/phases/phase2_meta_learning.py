"""
Phase 2 – Meta-Learning with HRM Integration (MAML/Reptile + HRM Fast Adaptation)
Enable fast adaptation to new tasks with few updates, combining HRM puzzle embeddings
with traditional MAML/Reptile approaches for improved task context understanding.
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

# HRM imports for fast adaptation
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../docs/HRM-main'))
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    _HAS_HRM = True
except ImportError:
    _HAS_HRM = False
    class HierarchicalReasoningModel_ACTV1:
        def __init__(self, *args, **kwargs): pass
        def initial_carry(self, *args): return {}
        def __call__(self, *args, **kwargs): return {}, {"puzzle_embedding": torch.randn(1, 128)}
        def to(self, device): return self

def _create_task_embedding(task_data, puzzle_embedder=None):
    """Create task embedding using HRM puzzle embedder or fallback."""
    import hashlib
    import numpy as np
    
    if puzzle_embedder is not None and _HAS_HRM:
        try:
            # Extract first example as task representation
            if 'support_set' in task_data and task_data['support_set']:
                example = task_data['support_set'][0]
                input_grid = example['input']
                if torch.is_tensor(input_grid):
                    tokens = input_grid.flatten().unsqueeze(0)
                else:
                    tokens = torch.tensor(input_grid).flatten().unsqueeze(0)
                
                puzzle_ids = torch.zeros(1, dtype=torch.long, device=tokens.device)
                batch = {
                    "inputs": tokens,
                    "labels": tokens,
                    "puzzle_identifiers": puzzle_ids
                }
                carry = puzzle_embedder.initial_carry(batch)
                carry, outputs = puzzle_embedder(carry=carry, batch=batch)
                
                if "puzzle_embedding" in outputs:
                    return outputs["puzzle_embedding"]
        except Exception as e:
            print(f"[Phase 2] HRM task embedding failed: {e}")
    
    # Fallback: deterministic embedding from task ID
    task_id = getattr(task_data, 'task_id', str(hash(str(task_data))))
    hash_val = int(hashlib.md5(str(task_id).encode()).hexdigest()[:8], 16)
    np.random.seed(hash_val % (2**31))
    embedding = np.random.randn(128)
    return torch.tensor(embedding, dtype=torch.float32)

def _hrm_fast_adapt(model, support_set, task_embedding, inner_steps=3, inner_lr=1e-2):
    """
    HRM-inspired fast adaptation using task embeddings.
    Combines gradient-based adaptation with task context.
    """
    import torch.nn.functional as F
    from copy import deepcopy
    
    # Clone model for fast adaptation
    adapted_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            adapted_params[name] = param.clone()
    
    # Task-specific adaptation with HRM context
    for step in range(inner_steps):
        total_loss = 0.0
        
        for example in support_set:
            try:
                input_grid = example['input']
                target_grid = example['output']
                
                # Convert to tensors if needed
                if not torch.is_tensor(input_grid):
                    input_grid = torch.tensor(input_grid, dtype=torch.long)
                if not torch.is_tensor(target_grid):
                    target_grid = torch.tensor(target_grid, dtype=torch.long)
                
                # Add batch dimension if missing
                if input_grid.dim() == 2:
                    input_grid = input_grid.unsqueeze(0)
                if target_grid.dim() == 2:
                    target_grid = target_grid.unsqueeze(0)
                
                # Forward pass with task embedding context
                # Note: This is a simplified version - in practice, you'd inject task_embedding into model
                context = [{"input": input_grid, "output": target_grid}]
                query = {"input": input_grid, "output": target_grid}
                
                with torch.enable_grad():
                    pred_grid, pred_logits, extras = model(context, query)
                    
                    if pred_logits is not None:
                        # Cross-entropy loss for adaptation
                        loss = F.cross_entropy(
                            pred_logits.view(-1, pred_logits.size(-1)),
                            target_grid.view(-1).clamp(0, 9)
                        )
                        total_loss += loss
            except Exception as e:
                print(f"[Phase 2] HRM fast adapt step error: {e}")
                continue
        
        if total_loss > 0:
            # Compute gradients for adaptation
            grads = torch.autograd.grad(total_loss, adapted_params.values(), create_graph=True, allow_unused=True)
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - inner_lr * grad
    
    return adapted_params

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
    
    # Initialize HRM puzzle embedder for task context
    puzzle_embedder = state.get("puzzle_embedder")
    if puzzle_embedder is None and _HAS_HRM and config.get("use_hrm_embeddings", True):
        try:
            hrm_config = {
                'batch_size': 1, 'seq_len': 400, 'vocab_size': 10,
                'num_puzzle_identifiers': 1000, 'puzzle_emb_ndim': 128,
                'H_cycles': 2, 'L_cycles': 2, 'H_layers': 2, 'L_layers': 2,
                'hidden_size': 256, 'expansion': 2.0, 'num_heads': 4,
                'pos_encodings': "rope", 'halt_max_steps': 4,
                'halt_exploration_prob': 0.1, 'forward_dtype': "bfloat16"
            }
            puzzle_embedder = HierarchicalReasoningModel_ACTV1(hrm_config).to(device)
            print(f"[Phase 2] Initialized HRM puzzle embedder for meta-learning")
        except Exception as e:
            print(f"[Phase 2] Failed to initialize HRM puzzle embedder: {e}")
            puzzle_embedder = None
    
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
                
                # === Enhanced Meta-Learning with HRM Fast Adaptation ===
                hrm_adapted_episodes = []
                
                # Apply HRM fast adaptation to each episode
                for episode in episodes:
                    try:
                        # Create task embedding for this episode
                        task_embedding = _create_task_embedding(episode, puzzle_embedder)
                        
                        # Apply HRM-inspired fast adaptation
                        if config.get("use_hrm_fast_adapt", True) and len(episode.support_set) > 0:
                            adapted_params = _hrm_fast_adapt(
                                model=model,
                                support_set=episode.support_set,
                                task_embedding=task_embedding,
                                inner_steps=config.get("hrm_inner_steps", 2),
                                inner_lr=config.get("hrm_inner_lr", 5e-3)
                            )
                            
                            # Store adapted parameters in episode for meta-learner
                            episode.hrm_adapted_params = adapted_params
                            episode.task_embedding = task_embedding
                        
                        hrm_adapted_episodes.append(episode)
                        
                    except Exception as e:
                        print(f"[Phase 2] HRM adaptation failed for episode {episode.task_id}: {e}")
                        hrm_adapted_episodes.append(episode)  # Use original episode
                
                # Run meta-learning step with HRM-adapted episodes
                meta_metrics = meta_learner.outer_loop(hrm_adapted_episodes)
                meta_loss = meta_metrics.get("meta_loss", 0.0)
                
                # Add task embedding consistency loss
                embedding_consistency_loss = 0.0
                if puzzle_embedder is not None and len(hrm_adapted_episodes) > 1:
                    embeddings = []
                    for episode in hrm_adapted_episodes:
                        if hasattr(episode, 'task_embedding'):
                            embeddings.append(episode.task_embedding)
                    
                    if len(embeddings) > 1:
                        # Encourage diverse task embeddings (prevent collapse)
                        embedding_stack = torch.stack(embeddings)
                        similarity_matrix = torch.mm(embedding_stack, embedding_stack.t())
                        # Penalize high similarity between different tasks
                        off_diagonal = similarity_matrix - torch.diag(torch.diag(similarity_matrix))
                        embedding_consistency_loss = 0.1 * off_diagonal.abs().mean()
                
                # Add RelMem losses and HRM consistency
                inherit_loss = relmem.inheritance_pass()
                inverse_loss = relmem.inverse_loss()
                meta_loss = meta_loss + 0.05 * inherit_loss + 0.05 * inverse_loss + embedding_consistency_loss
                
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
                    "inverse_loss": float(inverse_loss.item()) if hasattr(inverse_loss, "item") else 0.0,
                    "embedding_consistency_loss": float(embedding_consistency_loss.item()) if hasattr(embedding_consistency_loss, "item") else 0.0,
                    "hrm_adapted_episodes": len([ep for ep in hrm_adapted_episodes if hasattr(ep, 'hrm_adapted_params')]),
                    "total_episodes": len(hrm_adapted_episodes)
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
    
    # Add HRM-specific state
    if puzzle_embedder is not None:
        state["puzzle_embedder"] = puzzle_embedder
    state["hrm_meta_learning_enabled"] = config.get("use_hrm_fast_adapt", True)
    state["hrm_embedding_enabled"] = config.get("use_hrm_embeddings", True)
    
    print("[Phase 2] Meta Learning completed!")
    return state