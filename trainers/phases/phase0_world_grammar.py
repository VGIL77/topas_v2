"""
Phase 0 – World Grammar Pretrain
Warm up encoder/painter rails on synthetic ARC grammar tasks.
Implements robust batching, demo normalization, CE loss, and correct logging.
"""

from typing import Any, Dict, List, Tuple, Optional
import os
import sys

# Try to make both repo layouts work (with/without "trainers." prefix)
def _import_datasets():
    try:
        from arc_dataset_loader import SyntheticGrammarDataset, ARCDataset  # type: ignore
        return SyntheticGrammarDataset, ARCDataset
    except Exception:
        from trainers.arc_dataset_loader import SyntheticGrammarDataset, ARCDataset  # type: ignore
        return SyntheticGrammarDataset, ARCDataset

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

    # === Dataset ===
    dataset_name = (config.get("dataset", "synthetic") or "synthetic").lower()
    if dataset_name in ("arc", "arcdataset"):
        challenge_file = config.get("challenge_file", "arc-agi_training_challenges.json")
        solution_file = config.get("solution_file", "arc-agi_training_solutions.json")
        if not os.path.exists(challenge_file) and os.path.exists(os.path.join("ARC", "arc-agi_training-challenges.json")):
            # try common subdir
            challenge_file = os.path.join("ARC", "arc-agi_training-challenges.json")
            solution_file = os.path.join("ARC", "arc-agi_training-solutions.json")
        dataset = ARCDataset(challenge_file, solution_file, device=str(device))
    else:
        dataset = SyntheticGrammarDataset(
            num_samples=int(config.get("dataset_size", 200)),
            max_grid_size=int(config.get("max_grid_size", 30)),
            device=str(device),
        )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # === Training loop ===
    epochs = int(config.get("epochs", 1))
    base_model_train = getattr(base_model, "train", None)
    if callable(base_model_train):
        base_model.train(True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            # MASTER'S WISDOM: Respect steps_per_epoch to prevent runaway training
            if batch_idx >= int(config.get("steps_per_epoch", 200)):
                break
            
            # --- Unpack dataloader batch (robust against collate quirks) ---
            # Expect 4-tuple
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
    return state