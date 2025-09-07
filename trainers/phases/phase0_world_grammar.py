"""
Phase 0 – World Grammar Pretrain with HRM Integration
Train directly on ARC-AGI data with curriculum learning and HRM puzzle embeddings.
Implements ARC data loading, curriculum progression, and HRM task identity features.
"""

from typing import Any, Dict, List, Tuple, Optional
import os
import sys
import json
import random
import numpy as np

# Try to make both repo layouts work (with/without "trainers." prefix)
def _import_datasets():
    try:
        from arc_dataset_loader import SyntheticGrammarDataset, ARCDataset  # type: ignore
        return SyntheticGrammarDataset, ARCDataset
    except Exception:
        from trainers.arc_dataset_loader import SyntheticGrammarDataset, ARCDataset  # type: ignore
        return SyntheticGrammarDataset, ARCDataset

# HRM imports for puzzle embedding
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

def _import_logger():
    try:
        from train_logger import TrainLogger  # type: ignore
        return TrainLogger
    except Exception:
        from trainers.train_logger import TrainLogger  # type: ignore
        return TrainLogger

def _import_model_factory():
    # Optional helper to create a default model if state has none
    try:
        # prefer a factory if present
        from models.topas_arc_60M import create_model  # type: ignore
        return ("factory", create_model)
    except Exception:
        try:
            from models.topas_arc_60M import TopasARC60M, ModelConfig  # type: ignore
            def _mk(**kwargs):
                cfg = ModelConfig()
                # allow device override
                dev = kwargs.get("device", "cpu")
                m = TopasARC60M(cfg).to(dev)
                return m
            return ("class", _mk)
        except Exception:
            # Try without models prefix for different directory structures
            try:
                from topas_arc_60M import TopasARC60M, ModelConfig  # type: ignore
                def _mk(**kwargs):
                    cfg = ModelConfig()
                    dev = kwargs.get("device", "cpu")
                    m = TopasARC60M(cfg).to(dev)
                    return m
                return ("class", _mk)
            except Exception:
                return (None, None)

def _to_device(t, device):
    import torch
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        return type(t)(_to_device(x, device) for x in t)
    if isinstance(t, dict):
        return {k: _to_device(v, device) for k, v in t.items()}
    if torch.is_tensor(t):
        return t.to(device)
    return t

def _compute_ce_loss(logits, target_grid):
    """
    Cross-entropy over flattened H*W with C classes (0..9).
    Expects logits: [B, H*W, C] or [B, C, H, W]; target_grid: [B,H,W] or [H,W]
    """
    import torch
    import torch.nn.functional as F
    if target_grid is None:
        return torch.tensor(0.0, device=logits.device if hasattr(logits, "device") else "cpu")

    if target_grid.dim() == 2:
        target_grid = target_grid.unsqueeze(0)  # [1,H,W]
    B = target_grid.shape[0]

    # Normalize logits to [B, H*W, C]
    if logits is None:
        return torch.tensor(0.0, device=target_grid.device, dtype=torch.float32)

    if logits.dim() == 4 and logits.shape[1] > 1:
        # [B,C,H,W] -> [B, H*W, C]
        B_, C, H, W = logits.shape
        logits = logits.view(B_, C, H*W).transpose(1, 2)
    elif logits.dim() == 3:
        pass  # [B, H*W, C]
    else:
        # Unknown shape, try to coerce
        logits = logits.view(B, -1, 10)

    # Flatten for CE
    target_flat = target_grid.reshape(B, -1).long().clamp(0, 9)
    logits_flat = logits.reshape(B, -1, logits.size(-1))
    # Align lengths if they mismatch (crop to min)
    L = min(target_flat.shape[1], logits_flat.shape[1])
    target_flat = target_flat[:, :L]
    logits_flat = logits_flat[:, :L, :]

    loss = F.cross_entropy(logits_flat.reshape(-1, logits_flat.size(-1)),
                           target_flat.reshape(-1))
    return loss

def _canonize_grid(x, device, num_colors: int = 10):
    """
    Make any ARC-ish grid acceptable to the encoder/painter:
    - Densify sparse tensors
    - Remove stray singleton dims to get [B,*,H,W]
    - Convert categorical ints to one-hot channels => [B, C(=num_colors), H, W] (float)
    - If already channelized (C==num_colors), pass through
    """
    import torch
    import torch.nn.functional as F

    # to device + densify if sparse
    x = x.to(device)
    if x.is_sparse:
        x = x.to_dense()

    # squeeze extra leading singleton dims except batch (ONE-SHOT + ASSERT)
    # target is either [B,H,W] or [B,1,H,W] or [B,C,H,W]
    if x.ndim == 5 and x.shape[1] == 1:
        x = x.squeeze(1)

    # sanity check – don't allow runaway shapes
    assert x.ndim <= 4, f"[canonize] Unexpected shape after squeeze: {x.shape}"

    # ensure batch dim
    if x.ndim == 2:                # [H,W]
        x = x.unsqueeze(0)         # [1,H,W]
    if x.ndim == 3:                # [B,H,W] -> one-hot to channels
        x = F.one_hot(x.long().clamp(0, num_colors-1), num_classes=num_colors)  # [B,H,W,C]
        x = x.permute(0, 3, 1, 2).float()                                       # [B,C,H,W]
        return x

    if x.ndim == 4:
        B, C, H, W = x.shape
        if C == 1:                  # [B,1,H,W] -> squeeze then one-hot
            x_ = x.squeeze(1).long().clamp(0, num_colors-1)                     # [B,H,W]
            x_ = F.one_hot(x_, num_classes=num_colors).permute(0,3,1,2).float() # [B,C,H,W]
            return x_
        if C == num_colors:         # already channelized
            return x.float()
        # uncommon: other C -> try to treat as logits over colors per pixel
        if C > 1:
            # softmax over channels -> argmax -> one-hot
            idx = x.argmax(dim=1)                                                # [B,H,W]
            x_ = F.one_hot(idx.long().clamp(0, num_colors-1), num_classes=num_colors).permute(0,3,1,2).float()
            return x_
    # fallback: flatten/reshape into one-hot best-effort
    x = x.reshape(1, *x.shape[-2:]) if x.ndim >= 2 else x.view(1, 8, 8)
    x = torch.clamp(torch.round(x), 0, num_colors-1).long()
    x = torch.nn.functional.one_hot(x, num_classes=num_colors).permute(0,3,1,2).float()
    return x

def _load_arc_tasks_with_curriculum(arc_data_path: str, curriculum_level: int = 0) -> List[Dict]:
    """
    Load ARC tasks with curriculum learning progression.
    
    Args:
        arc_data_path: Path to ARC data directory (e.g., "ARC-AGI/data/training")
        curriculum_level: 0 (easy) to 4 (hard)
    
    Returns:
        List of task dictionaries sorted by difficulty
    """
    tasks = []
    
    if not os.path.exists(arc_data_path):
        print(f"[Phase 0] ARC data path not found: {arc_data_path}")
        return tasks
    
    # Load all task files
    for filename in os.listdir(arc_data_path):
        if filename.endswith('.json'):
            filepath = os.path.join(arc_data_path, filename)
            try:
                with open(filepath, 'r') as f:
                    task_data = json.load(f)
                    task_data['task_id'] = filename.replace('.json', '')
                    tasks.append(task_data)
            except Exception as e:
                print(f"[Phase 0] Failed to load {filename}: {e}")
                continue
    
    # Sort tasks by difficulty (curriculum learning)
    def _calculate_difficulty(task):
        """Calculate task difficulty based on grid size, number of objects, etc."""
        difficulty = 0
        
        # Check training examples for complexity
        train_examples = task.get('train', [])
        for example in train_examples:
            input_grid = example.get('input', [])
            output_grid = example.get('output', [])
            
            # Grid size factor
            input_size = len(input_grid) * len(input_grid[0]) if input_grid else 0
            output_size = len(output_grid) * len(output_grid[0]) if output_grid else 0
            difficulty += (input_size + output_size) / 200.0  # Normalize by ~average size
            
            # Number of unique colors (complexity)
            input_colors = set()
            output_colors = set()
            for row in input_grid:
                input_colors.update(row)
            for row in output_grid:
                output_colors.update(row)
            difficulty += (len(input_colors) + len(output_colors)) / 20.0
        
        # Number of training examples (more examples = easier to learn pattern)
        difficulty -= len(train_examples) * 0.1
        
        return max(0.0, difficulty)
    
    # Sort by difficulty
    tasks_with_difficulty = [(task, _calculate_difficulty(task)) for task in tasks]
    tasks_with_difficulty.sort(key=lambda x: x[1])
    
    # Select tasks based on curriculum level
    total_tasks = len(tasks_with_difficulty)
    if curriculum_level == 0:  # Easiest 20%
        selected_tasks = tasks_with_difficulty[:int(total_tasks * 0.2)]
    elif curriculum_level == 1:  # Easiest 40%
        selected_tasks = tasks_with_difficulty[:int(total_tasks * 0.4)]
    elif curriculum_level == 2:  # Easiest 60%
        selected_tasks = tasks_with_difficulty[:int(total_tasks * 0.6)]
    elif curriculum_level == 3:  # Easiest 80%
        selected_tasks = tasks_with_difficulty[:int(total_tasks * 0.8)]
    else:  # All tasks
        selected_tasks = tasks_with_difficulty
    
    result_tasks = [task for task, _ in selected_tasks]
    print(f"[Phase 0] Curriculum level {curriculum_level}: Selected {len(result_tasks)} tasks (difficulty range: {selected_tasks[0][1]:.2f} - {selected_tasks[-1][1]:.2f})")
    
    return result_tasks

def _create_puzzle_embedding(task_data: Dict, puzzle_embedder=None):
    """Create HRM-style puzzle embedding for task identity."""
    import torch
    
    if puzzle_embedder is not None and _HAS_HRM:
        # Use actual HRM puzzle embedder
        try:
            # Convert task to tokens for HRM processing
            train_grids = []
            for example in task_data.get('train', []):
                input_grid = torch.tensor(example['input'], dtype=torch.long)
                train_grids.append(input_grid.flatten())
            
            if train_grids:
                # Use first training example as task representation
                tokens = train_grids[0].unsqueeze(0)  # [1, seq_len]
                puzzle_ids = torch.zeros(1, dtype=torch.long)
                
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
            print(f"[Phase 0] HRM puzzle embedding failed: {e}")
    
    # Fallback: create simple hash-based embedding
    task_id = task_data.get('task_id', 'unknown')
    
    # Create deterministic embedding from task features
    feature_vector = []
    
    # Hash task ID to create base embedding
    import hashlib
    hash_val = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
    np.random.seed(hash_val % (2**31))
    base_embedding = np.random.randn(64)
    
    # Add task-specific features
    train_examples = task_data.get('train', [])
    if train_examples:
        # Average grid sizes
        avg_input_size = np.mean([len(ex['input']) * len(ex['input'][0]) for ex in train_examples])
        avg_output_size = np.mean([len(ex['output']) * len(ex['output'][0]) for ex in train_examples])
        
        # Color diversity
        all_colors = set()
        for ex in train_examples:
            for row in ex['input'] + ex['output']:
                all_colors.update(row)
        color_diversity = len(all_colors)
        
        # Task complexity features
        complexity_features = np.array([
            avg_input_size / 100.0,  # Normalize grid size
            avg_output_size / 100.0,
            color_diversity / 10.0,  # Normalize color count
            len(train_examples) / 10.0  # Normalize example count
        ])
        
        # Combine with base embedding
        full_embedding = np.concatenate([base_embedding, complexity_features])[:128]
        if len(full_embedding) < 128:
            full_embedding = np.pad(full_embedding, (0, 128 - len(full_embedding)))
    else:
        full_embedding = np.pad(base_embedding, (0, 64))[:128]
    
    return torch.tensor(full_embedding, dtype=torch.float32)

def _painter_only_forward(base_model, demo_in, demo_out, device):
    """
    Painter-only forward pass:
    encoder -> slots -> painter
    (bypasses DSL/EBR/Relations entirely)
    """
    try:
        if hasattr(base_model, "encoder") and hasattr(base_model, "slots") and hasattr(base_model, "painter"):
            xin = _canonize_grid(demo_in, device)     # [B,C,H,W] float
            # Belt-and-suspenders assertion
            assert xin.ndim == 4, f"canonize failed, got {xin.shape}"
            xin = xin.contiguous()  # guard against odd strides
            
            # MASTER'S DEBUG WISDOM: Monitor shapes during testing
            if hasattr(_painter_only_forward, '_debug_step'):
                _painter_only_forward._debug_step = getattr(_painter_only_forward, '_debug_step', 0) + 1
                if _painter_only_forward._debug_step % 20 == 0:
                    print(f"[dbg] step={_painter_only_forward._debug_step}, xin.shape={xin.shape}, xin.device={xin.device}")
            else:
                _painter_only_forward._debug_step = 1
            
            feat = base_model.encoder(xin)            # [B,*,H,W] or latent map
            slots = base_model.slots(feat)            # [B,K,slot_dim]
            pred_grid, pred_logits, *_ = base_model.painter(feat, slots)
            return pred_grid, pred_logits
        else:
            return _model_forward(base_model, demo_in, demo_out, device)
    except Exception as e:
        print(f"[PAINTER_ONLY] Direct forward failed: {e}")
        return _model_forward(base_model, demo_in, demo_out, device)

def _model_forward(base_model, demo_in, demo_out, device):
    """
    Forward wrapper tolerant to model signature variations.
    Returns (pred_grid, pred_logits)
    """
    import torch
    # Build context/query per sacred signature assumption
    context = [(demo_in, demo_out)]
    query = {"input": demo_in, "output": demo_out}

    try:
        out = base_model(context, query)
    except TypeError:
        # Some models expect a list of dict demos
        demo_dicts = [{"input": demo_in, "output": demo_out}]
        out = base_model(demo_dicts, {"input": demo_in})

    # Unpack 3–4 return values robustly
    if isinstance(out, (list, tuple)):
        if len(out) >= 2:
            pred_grid, pred_logits = out[0], out[1]
        elif len(out) == 1:
            pred_grid, pred_logits = out[0], None
        else:
            pred_grid, pred_logits = None, None
    elif isinstance(out, dict):
        pred_grid = out.get("grid", None)
        pred_logits = out.get("logits", None)
    else:
        pred_grid, pred_logits = out, None

    # Ensure on device
    pred_grid = _to_device(pred_grid, device)
    pred_logits = _to_device(pred_logits, device)
    return pred_grid, pred_logits

def run(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 0 runner with fidelity:
    - Respects painter_only, use_dsl, use_ebr, use_relations flags
    - Runs painter cleanly when painter_only=True
    """
    import torch
    from torch.utils.data import DataLoader

    # === Setup ===
    device = config.get("device", "cpu")
    
    # DIVINE FIDELITY FLAGS - Respect the config!
    use_dsl = config.get("use_dsl", False)
    use_ebr = config.get("use_ebr", False)
    use_rel = config.get("use_relations", False)
    painter_only = config.get("painter_only", False)
    
    if painter_only:
        use_dsl = False
        use_ebr = False
        use_rel = False
        print("[Phase 0] DIVINE PAINTER-ONLY MODE: DSL/EBR/Relations disabled for clean learning")
    
    SyntheticGrammarDataset, ARCDataset = _import_datasets()
    TrainLogger = _import_logger()

    base_model = state.get("model")
    if base_model is None:
        kind, maker = _import_model_factory()
        if maker is not None:
            base_model = maker(device=device)
            state["model"] = base_model

    base_model_device = next((p.device for p in base_model.parameters()), torch.device(device)) if base_model else torch.device(device)

    logger = state.get("logger")
    if logger is None:
        log_path = os.path.join(config.get("log_dir", "logs"), "topas_train.jsonl")
        logger = TrainLogger(log_path=log_path, verbose=True)
        state["logger"] = logger

    # === Dataset with Curriculum Learning ===
    # Force use of ARC data as specified in requirements
    arc_data_path = config.get("arc_data_path", "/mnt/d/Bitterbot/research/topas_v2/ARC-AGI/data/training")
    curriculum_level = config.get("curriculum_level", 0)  # Start with easiest tasks
    
    print(f"[Phase 0] Using ARC-AGI data with curriculum learning (level {curriculum_level})")
    
    # Load tasks with curriculum
    arc_tasks = _load_arc_tasks_with_curriculum(arc_data_path, curriculum_level)
    
    if not arc_tasks:
        print("[Phase 0] No ARC tasks loaded! Falling back to synthetic data for testing.")
        dataset = SyntheticGrammarDataset(
            num_samples=int(config.get("dataset_size", 50)),
            max_grid_size=int(config.get("max_grid_size", 15)),  # Start smaller for curriculum
            device=str(device),
        )
    else:
        # Use ARC tasks directly
        dataset = None  # We'll handle iteration manually
    
    # Initialize HRM puzzle embedder if available
    puzzle_embedder = None
    if _HAS_HRM and config.get("use_hrm_embeddings", True):
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
            print(f"[Phase 0] Initialized HRM puzzle embedder")
        except Exception as e:
            print(f"[Phase 0] Failed to initialize HRM puzzle embedder: {e}")
            puzzle_embedder = None

    if dataset is not None:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        dataloader = None

    # === Training loop ===
    epochs = int(config.get("epochs", 1))
    base_model_train = getattr(base_model, "train", None)
    if callable(base_model_train):
        base_model.train(True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        if arc_tasks:
            # Train on ARC tasks directly
            random.shuffle(arc_tasks)
            task_iterator = arc_tasks[:int(config.get("steps_per_epoch", 200))]
        else:
            # Use synthetic dataloader
            task_iterator = dataloader

        for batch_idx, item in enumerate(task_iterator):
            # MASTER'S WISDOM: Respect steps_per_epoch to prevent runaway training
            if batch_idx >= int(config.get("steps_per_epoch", 200)):
                break
            
            if arc_tasks:
                # --- Process ARC task directly ---
                task_data = item
                task_id = task_data.get('task_id', f'task_{batch_idx}')
                
                # Create puzzle embedding for task identity
                puzzle_embedding = _create_puzzle_embedding(task_data, puzzle_embedder)
                
                # Extract training examples as demo pairs
                demo_pairs = []
                train_examples = task_data.get('train', [])
                
                for example in train_examples:
                    try:
                        input_grid = torch.tensor(example['input'], dtype=torch.long).to(base_model_device)
                        output_grid = torch.tensor(example['output'], dtype=torch.long).to(base_model_device)
                        demo_pairs.append((input_grid, output_grid))
                    except Exception as e:
                        print(f"[Phase 0] Failed to process example in {task_id}: {e}")
                        continue
                
                if not demo_pairs:
                    continue
            else:
                # --- Unpack synthetic dataloader batch (robust against collate quirks) ---
                batch_data = item
                try:
                    demos, test_inputs, test_outputs, task_ids = batch_data
                except Exception:
                    # Some DataLoader configs return a single element already unbatched
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 4:
                        demos, test_inputs, test_outputs, task_ids = batch_data
                    else:
                        # Skip malformed batch
                        continue

                # De-batch for batch_size=1
                try:
                    demos = demos[0]
                except Exception:
                    pass

                if isinstance(test_inputs, list) and len(test_inputs) > 0:
                    test_inputs = test_inputs[0]
                if isinstance(test_outputs, list) and len(test_outputs) > 0:
                    test_outputs = test_outputs[0]
                if isinstance(task_ids, (list, tuple)) and len(task_ids) > 0:
                    task_id = task_ids[0]
                else:
                    task_id = task_ids

                # --- Normalize demos to list of (in,out) pairs ---
                demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
                if isinstance(demos, (list, tuple)) and len(demos) == 2 and torch.is_tensor(demos[0]):
                    # Synthetic single pair got collated into [in, out]
                    demo_pairs = [(demos[0], demos[1])]
                elif isinstance(demos, (list, tuple)):
                    # Already a list of pairs (ARCDataset)
                    demo_pairs = []
                    for entry in demos:
                        if isinstance(entry, (list, tuple)) and len(entry) == 2:
                            demo_pairs.append((entry[0], entry[1]))
                else:
                    # Skip malformed
                    continue

            # --- Iterate demo pairs, compute loss, average ---
            valid_pairs = 0
            batch_loss_accum = 0.0

            for pair in demo_pairs:
                try:
                    demo_in, demo_out = pair
                except Exception:
                    # Skip malformed entries
                    continue

                # Move to device
                demo_in = demo_in.to(base_model_device) if torch.is_tensor(demo_in) else demo_in
                demo_out = demo_out.to(base_model_device) if torch.is_tensor(demo_out) else demo_out

                # Forward
                try:
                    # DIVINE FIDELITY: Use painter-only mode when requested
                    if painter_only:
                        # Call painter directly, skip DSL/EBR/Relations
                        pred_grid, pred_logits = _painter_only_forward(base_model, demo_in, demo_out, base_model_device)
                    else:
                        # Full model pipeline
                        pred_grid, pred_logits = _model_forward(base_model, demo_in, demo_out, base_model_device)
                except Exception as e:
                    print(f"[Phase 0] Model forward failed: {e}")
                    # If model forward fails completely, skip this pair
                    continue

                # Compute CE loss (primary)
                loss = _compute_ce_loss(pred_logits, demo_out)
                if loss.requires_grad is False:
                    loss.requires_grad_()  # Ensure backprop works

                # Backprop if optimizer provided in state
                optimizer = state.get("optimizer")
                if optimizer is not None and loss.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                # Accumulate
                batch_loss_accum += float(loss.item())
                valid_pairs += 1

            if valid_pairs == 0:
                continue

            loss_val = batch_loss_accum / valid_pairs
            epoch_loss += loss_val
            num_batches += 1

            # --- Logging per-batch ---
            try:
                logger.log_batch(batch_idx, {"loss": float(loss_val)})
            except Exception:
                # Be resilient; don't crash on logging
                pass

        avg_epoch_loss = float(epoch_loss / max(1, num_batches))
        # --- Epoch logging with correct signature ---
        try:
            logger.log_epoch(epoch, "Phase0_WorldGrammar", avg_epoch_loss, None)
        except Exception:
            pass

    # === State updates ===
    state["model"] = base_model
    state["logger"] = logger
    state["phase0_completed"] = True
    
    # Add HRM-specific state
    if puzzle_embedder is not None:
        state["puzzle_embedder"] = puzzle_embedder
    state["curriculum_level"] = curriculum_level
    state["arc_tasks_processed"] = len(arc_tasks) if arc_tasks else 0
    
    print(f"[Phase 0] Completed! Processed {state['arc_tasks_processed']} ARC tasks at curriculum level {curriculum_level}")
    return state