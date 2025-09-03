"""
Trainer utilities shared across all phases.
Ensures consistent model forward calls, CE loss, config filtering, and program encoding.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional

# -------------------------
# Forward wrapper
# -------------------------
def safe_model_forward(model, demos, test, device, training_mode=False):
    """
    Call model forward safely with proper demo format handling.
    Returns (grid, logits, extras) consistently.
    """
    try:
        # PROPER FIDELITY: Normalize demo format before calling model
        normalized_demos = []
        for d in demos:
            if isinstance(d, tuple) and len(d) >= 2:
                # Convert tuple (input, output) → dict {"input": ..., "output": ...}
                normalized_demos.append({"input": d[0], "output": d[1]})
            elif isinstance(d, dict) and 'input' in d and 'output' in d:
                # Already in correct format
                normalized_demos.append(d)
            else:
                print(f"[SAFE_FORWARD] Skipping invalid demo format: {type(d)}")
                continue
        
        # Call model with normalized demos
        out = model(normalized_demos, test, training_mode=training_mode)
    except TypeError:
        try:
            # Fallback: try without training_mode
            out = model(normalized_demos, test)
        except:
            # Last resort: pass demos as-is
            out = model(demos, test)

    grid, logits, extras = None, None, {}
    if isinstance(out, (list, tuple)):
        if len(out) == 4:
            grid, logits, size, extras = out
        elif len(out) == 3:
            grid, logits, extras = out
        elif len(out) == 2:
            grid, logits = out
        else:
            grid = out[0] if len(out) > 0 else None
    elif isinstance(out, dict):
        grid = out.get("grid")
        logits = out.get("logits")
        extras = out
    else:
        grid = out

    # PROPER FIDELITY FIX: Shape normalization and range clamping
    if grid is not None and torch.is_tensor(grid):
        # Fix 5D → 3D shape issues properly
        while grid.ndim > 3:
            if grid.shape[1] == 1:
                grid = grid.squeeze(1)  # Remove singleton dimensions
            else:
                grid = grid[0]  # Take first in batch
        
        # Ensure valid ARC color range [0-9]
        grid = torch.clamp(torch.round(grid), 0, 9).long()
    
    # ADULT WISDOM: Generate logits from grid if missing
    if logits is None and grid is not None and torch.is_tensor(grid):
        # Convert predicted grid to logits
        logits = torch.nn.functional.one_hot(grid.long(), num_classes=10).permute(0, 3, 1, 2).float()
    
    # Ensure everything is on the right device
    if grid is not None and hasattr(grid, 'to'):
        grid = grid.to(device)
    if logits is not None and hasattr(logits, 'to'):
        logits = logits.to(device)
    
    return grid, logits, extras


# -------------------------
# CE Loss wrapper
# -------------------------
def compute_ce_loss(logits, target):
    """
    Cross-entropy loss wrapper. Ensures logits/targets are valid for ARC.
    """
    import torch
    import torch.nn.functional as F

    if logits is None or target is None:
        return torch.tensor(0.0, device=target.device if torch.is_tensor(target) else "cuda")
    
    # GLOBAL GUARD: Clamp bad targets before any processing
    if target is not None and torch.is_tensor(target) and logits is not None:
        num_classes = logits.size(-1)
        if (target < 0).any() or (target >= num_classes).any():
            print(f"[DEBUG][Global CE] Clamping bad targets: min={target.min().item()}, max={target.max().item()}")
        target = target.clamp(0, num_classes - 1)

    # Normalize logits → always [B, C, H, W]
    if logits.dim() == 3:  # [B, H, W]
        logits = F.one_hot(logits.long(), num_classes=10).permute(0, 3, 1, 2).float()
    elif logits.dim() == 4 and logits.shape[1] == 1:  # [B,1,H,W]
        logits = F.one_hot(logits.squeeze(1).long(), num_classes=10).permute(0, 3, 1, 2).float()

    B, C, H, W = logits.shape
    logits = logits.view(B, C, -1).transpose(1, 2)   # [B, H*W, C]
    target = target.view(B, -1).long().clamp(0, C-1)

    return F.cross_entropy(logits.reshape(-1, C), target.reshape(-1))


# -------------------------
# Config filter
# -------------------------
def filter_config(config: Dict[str, Any], annotations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter configuration dict to only include keys that exist in annotations.
    Used for safely passing configs to classes with specific parameter requirements.
    
    Args:
        config: Source configuration dictionary
        annotations: Target class annotations (e.g., ModelConfig.__annotations__)
        
    Returns:
        Filtered configuration dict with only valid keys
    """
    if config is None:
        return {}
    
    if annotations is None:
        return config.copy() if isinstance(config, dict) else {}
    
    # Handle both set and dict annotations
    valid_keys = annotations if isinstance(annotations, set) else set(annotations.keys())
    
    # Filter to only valid annotation keys
    filtered = {k: v for k, v in config.items() if k in valid_keys}
    
    return filtered


# -------------------------
# Program encoder
# -------------------------
def program_to_tensor(program, vocab: Dict[str, int], max_len: int = 32):
    """
    Map DSL program (list of ops) to a tensor of token ids.
    Pads/truncates to fixed length for batching.
    """
    if vocab is None:
        vocab = {}
    
    pad_id = vocab.get("<PAD>", 0)
    unk_id = vocab.get("<UNK>", 1)

    ids = []
    
    # Handle different program formats
    if isinstance(program, str):
        # Single operation
        ids.append(vocab.get(program, unk_id))
    elif isinstance(program, (list, tuple)):
        # List of operations
        for op in program:
            if isinstance(op, str):
                ids.append(vocab.get(op, unk_id))
            elif isinstance(op, dict) and 'op' in op:
                # Operation with parameters
                ids.append(vocab.get(op['op'], unk_id))
            else:
                ids.append(unk_id)
    else:
        # Unknown format
        ids.append(unk_id)
    
    # Pad or truncate to max_len
    ids = ids[:max_len]
    ids += [pad_id] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)


# -------------------------
# Dataset unpacking helper
# -------------------------
def unpack_batch_data(batch_data, batch_size: int = 1):
    """
    Safely unpack batch data from dataloader.
    Returns (demos, test_inputs, test_outputs, task_ids)
    """
    # Default values
    demos, test_inputs, test_outputs, task_ids = None, None, None, None
    
    # Try to unpack 4-tuple
    try:
        demos, test_inputs, test_outputs, task_ids = batch_data
    except ValueError:
        # Not a 4-tuple, try other formats
        if isinstance(batch_data, (list, tuple)):
            if len(batch_data) == 3:
                demos, test_inputs, test_outputs = batch_data
                task_ids = ["unknown"]
            elif len(batch_data) == 2:
                demos, test_inputs = batch_data
                test_outputs = None
                task_ids = ["unknown"]
            elif len(batch_data) == 1:
                demos = batch_data[0]
                test_inputs = None
                test_outputs = None
                task_ids = ["unknown"]
        else:
            # Single item
            demos = batch_data
            test_inputs = None
            test_outputs = None
            task_ids = ["unknown"]
    
    # Unwrap batch dimension if batch_size=1
    if batch_size == 1:
        if isinstance(demos, (list, tuple)) and len(demos) == 1:
            demos = demos[0]
        if isinstance(test_inputs, (list, tuple)) and len(test_inputs) == 1:
            test_inputs = test_inputs[0]
        if isinstance(test_outputs, (list, tuple)) and len(test_outputs) == 1:
            test_outputs = test_outputs[0]
        if isinstance(task_ids, (list, tuple)) and len(task_ids) == 1:
            task_ids = task_ids[0]
    
    return demos, test_inputs, test_outputs, task_ids


# -------------------------
# Demo normalization helper
# -------------------------
def normalize_demos(demos) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Normalize demos to a list of (input, output) pairs.
    Handles both synthetic single pairs and ARC multiple pairs.
    """
    demo_pairs = []
    
    if demos is None:
        return demo_pairs
    
    # Check if it's a single pair [input_tensor, output_tensor]
    if isinstance(demos, (list, tuple)) and len(demos) == 2:
        # Check if first element is a tensor (single pair)
        if torch.is_tensor(demos[0]):
            demo_pairs = [(demos[0], demos[1])]
        else:
            # Might be a list of two pairs
            for item in demos:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    demo_pairs.append((item[0], item[1]))
    # Check if it's already a list of pairs
    elif isinstance(demos, (list, tuple)):
        for item in demos:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                demo_pairs.append((item[0], item[1]))
            elif isinstance(item, dict):
                # Handle dict format {"input": ..., "output": ...}
                if "input" in item and "output" in item:
                    demo_pairs.append((item["input"], item["output"]))
    
    return demo_pairs


# -------------------------
# State validation helper
# -------------------------
def get_from_state(state: Dict[str, Any], key: str, default=None, required: bool = False):
    """
    Safely get value from state dictionary.
    """
    value = state.get(key, default)
    if required and value is None:
        raise ValueError(f"Required state key '{key}' not found. Run previous phases first.")
    return value


# -------------------------
# Belt-and-suspenders safety helper
# -------------------------
def _to_long_0_9(x):
    import torch
    if x is None: return None
    if torch.is_tensor(x):
        return x.long().clamp(0, 9)
    return x


# -------------------------
# Default sample function
# -------------------------
def default_sample_fn(dataset_or_tasks, device="cpu"):
    """
    Default sampling function that returns (demos, test) from dataset.
    This removes the hidden dependency on state["sample_fn"].
    
    Args:
        dataset_or_tasks: Either a dataset object or list of tasks
        device: Device to place tensors on
        
    Returns:
        (demos, test) tuple where:
        - demos: List of {"input": tensor, "output": tensor} dicts
        - test: {"input": tensor, "output": tensor} dict
    """
    import torch
    import random
    
    # Handle None dataset
    if dataset_or_tasks is None:
        # In Phase 1+, ARC dataset is mandatory
        raise RuntimeError("ARC dataset not available — cannot continue training.")
    
    # Handle dataset with __getitem__
    if hasattr(dataset_or_tasks, '__getitem__') and hasattr(dataset_or_tasks, '__len__'):
        idx = random.randint(0, len(dataset_or_tasks) - 1)
        try:
            # Try standard ARC dataset format
            demos_raw, test_inputs, test_outputs, task_id = dataset_or_tasks[idx]
            
            # Normalize demos to list of dicts
            demos = []
            demo_pairs = normalize_demos(demos_raw)
            for inp, out in demo_pairs:
                if torch.is_tensor(inp) and torch.is_tensor(out):
                    demos.append({
                        "input": inp.to(device),
                        "output": out.to(device)
                    })
            
            # Create test dict
            test = {}
            if test_inputs is not None:
                test["input"] = test_inputs.to(device) if torch.is_tensor(test_inputs) else test_inputs
            if test_outputs is not None:
                test["output"] = test_outputs.to(device) if torch.is_tensor(test_outputs) else test_outputs
                
            # Strict validation - ARC data must be complete
            if len(demos) == 0:
                raise RuntimeError("ARC dataset returned empty demos. Check dataset integrity.")
            if "input" not in test:
                raise RuntimeError("ARC dataset missing test input. Check dataset integrity.")
            if "output" not in test:
                raise RuntimeError("ARC dataset missing test output. Check dataset integrity.")
            # Final IO sanitization
            for d in demos:
                if "input" in d:  d["input"]  = d["input"].long().clamp(0,9)
                if "output" in d: d["output"] = d["output"].long().clamp(0,9)
            if "input" in test:  test["input"]  = test["input"].long().clamp(0,9)
            if "output" in test: test["output"] = test["output"].long().clamp(0,9)
            return demos, test
            
        except Exception as e:
            # Strict enforcement: fail immediately on dataset errors
            raise RuntimeError(f"Failed to load ARC dataset sample: {e}. Fix dataset loading before continuing.")
    
    # Handle list of tasks
    if isinstance(dataset_or_tasks, list) and len(dataset_or_tasks) > 0:
        task = random.choice(dataset_or_tasks)
        if isinstance(task, dict):
            demos = task.get("demos", [])
            test = task.get("test", {})
            
            # Ensure proper format
            if not isinstance(demos, list):
                demos = [demos]
            if not isinstance(test, dict):
                test = {"input": test}
            
            # Belt-and-suspenders safety clamping
            for d in demos:
                if "input" in d and torch.is_tensor(d["input"]):  
                    d["input"] = _to_long_0_9(d["input"])
                if "output" in d and torch.is_tensor(d["output"]): 
                    d["output"] = _to_long_0_9(d["output"])
            if "output" in test and torch.is_tensor(test["output"]):
                test["output"] = _to_long_0_9(test["output"])
                
            return demos, test
    
    # Strict enforcement: no fallback to dummy data in phases 1+
    raise RuntimeError(f"Invalid dataset format: {type(dataset_or_tasks)}. ARC dataset required for training phases.")