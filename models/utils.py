import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import math

# Sacred Signature utility functions

def verify_sacred_signature(grid: torch.Tensor, logits: torch.Tensor, 
                           size: torch.Tensor, extras: Dict[str, Any]) -> bool:
    """
    Verify the Sacred Signature format is correct
    
    Args:
        grid: [B, H, W] tensor of predicted grids
        logits: [B, H*W, C] tensor of logits
        size: [B, 2] tensor of output sizes [H, W]
        extras: Dictionary with optional tensors
        
    Returns:
        bool: True if signature is valid
    """
    try:
        # Check grid shape
        if len(grid.shape) != 3:
            return False
        B, H, W = grid.shape
        
        # Check logits shape
        if logits.shape[0] != B or logits.shape[1] != H*W:
            return False
            
        # Check size shape
        if size.shape != (B, 2):
            return False
            
        # Check extras is a dict
        if not isinstance(extras, dict):
            return False
            
        return True
    except:
        return False


def logits_from_grid_validated(grid: torch.Tensor, num_colors: int = 10, operation: str = "logits_conversion") -> torch.Tensor:
    """
    Convert grid to logits with Sacred Signature validation.
    
    Args:
        grid: Input grid [B,H,W]
        num_colors: Number of color classes
        operation: Operation name for error reporting
        
    Returns:
        logits: [B,H*W,C] logits tensor
        
    Raises:
        RuntimeError: On signature violation
    """
    if not isinstance(grid, torch.Tensor):
        raise RuntimeError(f"[{operation}] INPUT VIOLATION: grid must be torch.Tensor, got {type(grid)}")
    
    if grid.dim() != 3:
        raise RuntimeError(f"[{operation}] INPUT VIOLATION: grid must be [B,H,W], got shape {grid.shape}")
    
    B, H, W = grid.shape
    
    if B <= 0 or H <= 0 or W <= 0:
        raise RuntimeError(f"[{operation}] INPUT VIOLATION: dimensions must be positive, got [{B},{H},{W}]")
    
    # Clamp grid to valid range and ensure integer
    if grid.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
        grid = grid.round().long()
    
    grid_clamped = torch.clamp(grid, 0, num_colors - 1).long()
    
    # Convert to logits
    flat = grid_clamped.reshape(B, H*W, 1)
    logits = torch.zeros(B, H*W, num_colors, device=grid.device, dtype=torch.float32)
    logits.scatter_(2, flat, 1.0)
    
    # Validation
    assert isinstance(logits, torch.Tensor), f"[{operation}] OUTPUT VIOLATION: logits must be tensor"
    assert logits.shape == (B, H*W, num_colors), f"[{operation}] OUTPUT VIOLATION: logits shape expected ({B},{H*W},{num_colors}), got {logits.shape}"
    assert logits.dtype == torch.float32, f"[{operation}] OUTPUT VIOLATION: logits must be float32, got {logits.dtype}"
    assert logits.min() >= 0.0 and logits.max() <= 1.0, f"[{operation}] OUTPUT VIOLATION: logits range violation [{logits.min()},{logits.max()}] not in [0,1]"
    
    return logits

def size_tensor_from_grid_validated(grid: torch.Tensor, operation: str = "size_conversion") -> torch.Tensor:
    """
    Create size tensor from grid with Sacred Signature validation.
    
    Args:
        grid: Input grid [B,H,W]
        operation: Operation name for error reporting
        
    Returns:
        size: [B,2] size tensor
        
    Raises:
        RuntimeError: On signature violation
    """
    if not isinstance(grid, torch.Tensor):
        raise RuntimeError(f"[{operation}] INPUT VIOLATION: grid must be torch.Tensor, got {type(grid)}")
    
    if grid.dim() != 3:
        raise RuntimeError(f"[{operation}] INPUT VIOLATION: grid must be [B,H,W], got shape {grid.shape}")
    
    B, H, W = grid.shape
    
    if B <= 0 or H <= 0 or W <= 0:
        raise RuntimeError(f"[{operation}] INPUT VIOLATION: dimensions must be positive, got [{B},{H},{W}]")
    
    size = torch.tensor([H, W], device=grid.device, dtype=torch.long).unsqueeze(0).expand(B, -1).contiguous()
    
    # Validation
    assert isinstance(size, torch.Tensor), f"[{operation}] OUTPUT VIOLATION: size must be tensor"
    assert size.shape == (B, 2), f"[{operation}] OUTPUT VIOLATION: size shape expected ({B},2), got {size.shape}"
    assert size.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long], f"[{operation}] OUTPUT VIOLATION: size must be integer, got {size.dtype}"
    assert size.min() > 0, f"[{operation}] OUTPUT VIOLATION: size values must be positive, got min={size.min()}"
    
    # VERIFY VALUES
    expected_values = torch.tensor([H, W], device=grid.device, dtype=size.dtype)
    for i in range(B):
        if not torch.equal(size[i], expected_values):
            raise RuntimeError(f"[{operation}] OUTPUT VIOLATION: size[{i}] expected {expected_values.tolist()}, got {size[i].tolist()}")
    
    return size

# Sacred Signature validation functions

def validate_grid_signature(grid: torch.Tensor, operation: str = "grid_validation") -> None:
    """
    Validate grid conforms to Sacred Signature [B,H,W] with integer values in [0,10).
    
    Raises:
        RuntimeError: On any violation
    """
    if not isinstance(grid, torch.Tensor):
        raise RuntimeError(f"[{operation}] GRID VIOLATION: must be torch.Tensor, got {type(grid)}")
    
    if grid.dim() != 3:
        raise RuntimeError(f"[{operation}] GRID VIOLATION: must be [B,H,W], got shape {grid.shape}")
    
    B, H, W = grid.shape
    if B <= 0 or H <= 0 or W <= 0:
        raise RuntimeError(f"[{operation}] GRID VIOLATION: dimensions must be positive, got [{B},{H},{W}]")
    
    if grid.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
        raise RuntimeError(f"[{operation}] GRID VIOLATION: must be integer tensor, got {grid.dtype}")
    
    if grid.min() < 0 or grid.max() >= 10:
        raise RuntimeError(f"[{operation}] GRID VIOLATION: values must be in [0,10), got [{grid.min()},{grid.max()}]")

def validate_logits_signature(logits: torch.Tensor, B: int, H: int, W: int, num_colors: int = 10, operation: str = "logits_validation") -> None:
    """
    Validate logits conform to Sacred Signature [B,H*W,C] with float values.
    
    Raises:
        RuntimeError: On any violation
    """
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(f"[{operation}] LOGITS VIOLATION: must be torch.Tensor, got {type(logits)}")
    
    if logits.dim() != 3:
        raise RuntimeError(f"[{operation}] LOGITS VIOLATION: must be [B,H*W,C], got shape {logits.shape}")
    
    B_logits, HW_logits, C_logits = logits.shape
    if B_logits != B:
        raise RuntimeError(f"[{operation}] LOGITS VIOLATION: batch mismatch, expected {B}, got {B_logits}")
    
    if HW_logits != H * W:
        raise RuntimeError(f"[{operation}] LOGITS VIOLATION: spatial mismatch, expected {H*W}, got {HW_logits}")
    
    if C_logits != num_colors:
        raise RuntimeError(f"[{operation}] LOGITS VIOLATION: channels mismatch, expected {num_colors}, got {C_logits}")
    
    if not logits.dtype.is_floating_point:
        raise RuntimeError(f"[{operation}] LOGITS VIOLATION: must be float tensor, got {logits.dtype}")

def validate_size_signature(size: torch.Tensor, B: int, H: int, W: int, operation: str = "size_validation") -> None:
    """
    Validate size tensor conforms to Sacred Signature [B,2] with correct H,W values.
    
    Raises:
        RuntimeError: On any violation
    """
    if not isinstance(size, torch.Tensor):
        raise RuntimeError(f"[{operation}] SIZE VIOLATION: must be torch.Tensor, got {type(size)}")
    
    if size.dim() != 2:
        raise RuntimeError(f"[{operation}] SIZE VIOLATION: must be [B,2], got shape {size.shape}")
    
    B_size, dim_size = size.shape
    if B_size != B:
        raise RuntimeError(f"[{operation}] SIZE VIOLATION: batch mismatch, expected {B}, got {B_size}")
    
    if dim_size != 2:
        raise RuntimeError(f"[{operation}] SIZE VIOLATION: must have 2 dimensions, got {dim_size}")
    
    # Verify values
    expected_size = torch.tensor([H, W], device=size.device, dtype=size.dtype)
    for i in range(B):
        if not torch.equal(size[i], expected_size):
            raise RuntimeError(f"[{operation}] SIZE VIOLATION: size[{i}] expected {expected_size.tolist()}, got {size[i].tolist()}")

# ============================================================================# ============================================================================


# ============================================================================
# METRIC COMPUTATION UTILITY
# ============================================================================

@torch.no_grad()
def compute_eval_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute Exact@1, Exact@K (same as @1 for single attempt),
    and IoU between predicted and ground truth grids.
    """
    eval_metrics = {"exact@1": 0.0, "exact@k": 0.0, "iou": 0.0}
    try:
        if pred.shape[-2:] != target.shape[-2:]:
            return eval_metrics
        exact_match = float((pred.detach() == target.detach()).all().item())
        eval_metrics["exact@1"] = exact_match
        eval_metrics["exact@k"] = exact_match
        pred_bool = (pred.detach() > 0)
        target_bool = (target.detach() > 0)
        overlap = (pred_bool & target_bool).sum().item()
        union = (pred_bool | target_bool).sum().item()
        eval_metrics["iou"] = overlap / max(1.0, union)
    except Exception:
        pass
    return eval_metrics


# ============================================================================
# OBJECT AUXILIARY LOSS FUNCTIONS - ACTIVE LEARNING COMPONENTS
# ============================================================================


def object_count_loss(pred_counts: torch.Tensor, true_counts: torch.Tensor, 
                     weight: float = 1.0) -> torch.Tensor:
    """
    Predict number of objects per color
    
    Args:
        pred_counts: Predicted object counts [B, num_colors]
        true_counts: True object counts [B, num_colors]
        weight: Loss weight
        
    Returns:
        MSE loss for object count prediction
    """
    return weight * F.mse_loss(pred_counts, true_counts)

def color_correspondence_loss(pred_mapping: torch.Tensor, true_mapping: torch.Tensor,
                             weight: float = 1.0) -> torch.Tensor:
    """
    Predict color mappings between objects
    
    Args:
        pred_mapping: Predicted color mappings [B, num_colors, num_colors] (softmax over target colors)
        true_mapping: True color mappings [B, num_colors] (target color indices)
        weight: Loss weight
        
    Returns:
        Cross-entropy loss for color mapping prediction
    """
    return weight * F.cross_entropy(pred_mapping.view(-1, pred_mapping.size(-1)), 
                                   true_mapping.view(-1))

def relation_prediction_loss(pred_relations: Dict[str, torch.Tensor], 
                           true_relations: Dict[str, torch.Tensor],
                           relation_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """
    Predict object relationships
    
    Args:
        pred_relations: Predicted relationship probabilities
        true_relations: True relationship labels
        relation_weights: Per-relation loss weights
        
    Returns:
        Weighted binary cross-entropy loss for relationships
    """
    if relation_weights is None:
        relation_weights = {
            'touching': 1.2,
            'contained': 1.0,
            'aligned': 0.8,
            'same_shape': 0.6,
            'same_color': 1.0
        }
    
    total_loss = 0.0
    loss_count = 0
    
    for rel_type in ['touching', 'contained', 'aligned', 'same_shape', 'same_color']:
        if rel_type in pred_relations and rel_type in true_relations:
            weight = relation_weights.get(rel_type, 1.0)
            loss = F.binary_cross_entropy(pred_relations[rel_type], true_relations[rel_type])
            total_loss += weight * loss
            loss_count += 1
    
    return total_loss / max(loss_count, 1)

def object_consistency_loss(object_masks: List[torch.Tensor], 
                           attention_weights: torch.Tensor,
                           weight: float = 0.5) -> torch.Tensor:
    """
    Ensure object masks are consistent with attention weights
    
    Args:
        object_masks: List of object binary masks [H, W]
        attention_weights: Slot attention weights [B, K, H*W]
        weight: Loss weight
        
    Returns:
        Consistency loss between masks and attention
    """
    if not object_masks or attention_weights.numel() == 0:
        return torch.tensor(0.0, device=attention_weights.device if attention_weights.numel() > 0 else 'cpu')
    
    B, K, HW = attention_weights.shape
    H = W = int(math.sqrt(HW))
    
    # Reshape attention to spatial dimensions
    attn_spatial = attention_weights[0].view(K, H, W)  # Take first batch
    
    total_loss = 0.0
    valid_pairs = 0
    
    for mask in object_masks:
        if mask.shape != (H, W):
            continue
        
        # Find best matching attention slot
        best_overlap = -1.0
        best_slot_idx = 0
        
        for k in range(K):
            attn_mask = (attn_spatial[k] > 0.1).float()
            overlap = (mask * attn_mask).sum() / (mask.sum() + attn_mask.sum() - (mask * attn_mask).sum() + 1e-8)
            if overlap > best_overlap:
                best_overlap = overlap
                best_slot_idx = k
        
        # Compute consistency loss with best matching slot
        if best_overlap > 0.1:  # Only if there's reasonable overlap
            target_attn = attn_spatial[best_slot_idx]
            consistency = F.mse_loss(target_attn, mask.float())
            total_loss += consistency
            valid_pairs += 1
    
    return weight * (total_loss / max(valid_pairs, 1))

def shape_preservation_loss(original_shapes: torch.Tensor, transformed_shapes: torch.Tensor,
                           weight: float = 0.3) -> torch.Tensor:
    """
    Encourage preservation of object shapes during transformations
    
    Args:
        original_shapes: Original shape signatures [N, shape_dim]
        transformed_shapes: Transformed shape signatures [N, shape_dim]
        weight: Loss weight
        
    Returns:
        MSE loss for shape preservation
    """
    if original_shapes.numel() == 0 or transformed_shapes.numel() == 0:
        return torch.tensor(0.0)
    
    return weight * F.mse_loss(transformed_shapes, original_shapes)

def spatial_coherence_loss(object_positions: torch.Tensor, predicted_positions: torch.Tensor,
                          weight: float = 0.4) -> torch.Tensor:
    """
    Encourage spatial coherence in object arrangements
    
    Args:
        object_positions: True object centroids [N, 2]
        predicted_positions: Predicted object centroids [N, 2]
        weight: Loss weight
        
    Returns:
        MSE loss for spatial positions
    """
    if object_positions.numel() == 0 or predicted_positions.numel() == 0:
        return torch.tensor(0.0)
    
    return weight * F.mse_loss(predicted_positions, object_positions)

def object_compactness_loss(object_masks: List[torch.Tensor], 
                           target_compactness: Optional[torch.Tensor] = None,
                           weight: float = 0.2) -> torch.Tensor:
    """
    Encourage compact object representations
    
    Args:
        object_masks: List of object binary masks
        target_compactness: Target compactness values [N] (optional)
        weight: Loss weight
        
    Returns:
        Compactness regularization loss
    """
    if not object_masks:
        return torch.tensor(0.0)
    
    total_loss = 0.0
    valid_objects = 0
    
    for i, mask in enumerate(object_masks):
        if mask.sum() < 3:  # Skip very small objects
            continue
        
        # Compute actual compactness
        area = mask.sum().item()
        perimeter = compute_perimeter(mask)
        compactness = 4 * math.pi * area / max(perimeter ** 2, 1) if perimeter > 0 else 0.0
        
        if target_compactness is not None and i < len(target_compactness):
            # Loss against target compactness
            target = target_compactness[i].item()
            loss = (compactness - target) ** 2
        else:
            # Regularize towards reasonable compactness (not too fragmented)
            ideal_compactness = 0.5  # Moderate compactness
            loss = (compactness - ideal_compactness) ** 2
        
        total_loss += loss
        valid_objects += 1
    
    return weight * (total_loss / max(valid_objects, 1))

def compute_perimeter(mask: torch.Tensor) -> int:
    """Compute perimeter of binary mask"""
    if mask.sum() == 0:
        return 0
    
    H, W = mask.shape
    perimeter = 0
    
    for i in range(H):
        for j in range(W):
            if mask[i, j] > 0:
                # Check if on boundary
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= H or nj < 0 or nj >= W or mask[ni, nj] == 0:
                        perimeter += 1
                        break
    
    return perimeter

def symmetry_preservation_loss(original_symmetries: Dict[str, torch.Tensor], 
                              predicted_symmetries: Dict[str, torch.Tensor],
                              weight: float = 0.3) -> torch.Tensor:
    """
    Encourage preservation of symmetry properties
    
    Args:
        original_symmetries: Original symmetry flags
        predicted_symmetries: Predicted symmetry flags
        weight: Loss weight
        
    Returns:
        BCE loss for symmetry preservation
    """
    total_loss = 0.0
    symmetry_count = 0
    
    for sym_type in ['horizontal', 'vertical', 'rotational']:
        if sym_type in original_symmetries and sym_type in predicted_symmetries:
            loss = F.binary_cross_entropy(predicted_symmetries[sym_type], 
                                         original_symmetries[sym_type])
            total_loss += loss
            symmetry_count += 1
    
    return weight * (total_loss / max(symmetry_count, 1))

class ObjectAuxiliaryLossManager:
    """
    Manager class for combining multiple auxiliary losses
    """
    
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        self.loss_weights = loss_weights or {
            'object_count': 0.5,
            'color_correspondence': 0.3,
            'relation_prediction': 0.4,
            'object_consistency': 0.2,
            'shape_preservation': 0.1,
            'spatial_coherence': 0.2,
            'object_compactness': 0.1,
            'symmetry_preservation': 0.15
        }
    
    def compute_auxiliary_losses(self, batch: Dict[str, Any], 
                                predictions: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute all auxiliary losses for a batch
        
        Args:
            batch: Batch data containing targets
            predictions: Model predictions
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        
        # Object count loss
        if 'object_counts' in batch and 'pred_object_counts' in predictions:
            losses['object_count'] = object_count_loss(
                predictions['pred_object_counts'],
                batch['object_counts'],
                weight=self.loss_weights['object_count']
            )
        
        # Color correspondence loss
        if 'color_mappings' in batch and 'pred_color_mappings' in predictions:
            losses['color_correspondence'] = color_correspondence_loss(
                predictions['pred_color_mappings'],
                batch['color_mappings'],
                weight=self.loss_weights['color_correspondence']
            )
        
        # Relation prediction loss
        if 'object_relations' in batch and 'pred_relations' in predictions:
            losses['relation_prediction'] = relation_prediction_loss(
                predictions['pred_relations'],
                batch['object_relations']
            ) * self.loss_weights['relation_prediction']
        
        # Object consistency loss
        if 'object_masks' in batch and 'attention_weights' in predictions:
            losses['object_consistency'] = object_consistency_loss(
                batch['object_masks'],
                predictions['attention_weights'],
                weight=self.loss_weights['object_consistency']
            )
        
        # Shape preservation loss
        if 'original_shapes' in batch and 'transformed_shapes' in predictions:
            losses['shape_preservation'] = shape_preservation_loss(
                batch['original_shapes'],
                predictions['transformed_shapes'],
                weight=self.loss_weights['shape_preservation']
            )
        
        # Spatial coherence loss
        if 'object_positions' in batch and 'pred_positions' in predictions:
            losses['spatial_coherence'] = spatial_coherence_loss(
                batch['object_positions'],
                predictions['pred_positions'],
                weight=self.loss_weights['spatial_coherence']
            )
        
        # Object compactness loss
        if 'object_masks' in batch:
            target_compactness = batch.get('target_compactness', None)
            losses['object_compactness'] = object_compactness_loss(
                batch['object_masks'],
                target_compactness,
                weight=self.loss_weights['object_compactness']
            )
        
        # Symmetry preservation loss
        if 'original_symmetries' in batch and 'pred_symmetries' in predictions:
            losses['symmetry_preservation'] = symmetry_preservation_loss(
                batch['original_symmetries'],
                predictions['pred_symmetries'],
                weight=self.loss_weights['symmetry_preservation']
            )
        
        return losses
    
    def compute_total_auxiliary_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine all auxiliary losses into total loss"""
        if not losses:
            return torch.tensor(0.0)
        
        total_loss = sum(losses.values())
        return total_loss

def add_object_losses_to_training(model, batch: Dict[str, Any], 
                                 predictions: Dict[str, Any]) -> torch.Tensor:
    """
    Convenience function to add auxiliary losses during training
    
    Args:
        model: TOPAS model (contains object extraction components)
        batch: Training batch with auxiliary targets
        predictions: Model predictions
        
    Returns:
        Total auxiliary loss
    """
    # Create loss manager
    loss_manager = ObjectAuxiliaryLossManager()
    
    # Extract objects from input if not provided
    if 'object_masks' not in batch and 'input' in batch:
        # Use model's object extraction capabilities
        try:
            if hasattr(model, 'slots') and hasattr(model.slots, 'extract_object_masks'):
                # Extract attention weights and use them to get object masks
                input_grid = batch['input']
                with torch.no_grad():
                    feat, _ = model.encoder(input_grid.float() / 9.0)  # Normalize
                    _, attention_weights = model.slots(feat)
                    
                # Extract object masks for first sample in batch
                if len(input_grid.shape) == 4:  # [B, C, H, W] or [B, H, W, C]
                    grid_2d = input_grid[0, 0] if input_grid.shape[1] < input_grid.shape[-1] else input_grid[0, :, :, 0]
                else:
                    grid_2d = input_grid[0]
                
                object_masks = model.slots.extract_object_masks(grid_2d, attention_weights)
                batch['object_masks'] = object_masks
                
                # Also add attention weights to predictions
                predictions['attention_weights'] = attention_weights
                
        except Exception as e:
            print(f"[WARN] Could not extract object masks for auxiliary loss: {e}")
    
    # Compute auxiliary losses
    losses = loss_manager.compute_auxiliary_losses(batch, predictions)
    
    # Return total auxiliary loss
    return loss_manager.compute_total_auxiliary_loss(losses)

def create_object_training_targets(input_grids: torch.Tensor, 
                                  output_grids: torch.Tensor) -> Dict[str, Any]:
    """
    Create auxiliary training targets from input/output grid pairs
    
    Args:
        input_grids: Input grids [B, H, W]
        output_grids: Output grids [B, H, W]
        
    Returns:
        Dictionary of auxiliary training targets
    """
    targets = {}
    B = input_grids.shape[0]
    
    # Object count targets (count objects per color)
    object_counts = []
    for b in range(B):
        inp = input_grids[b]
        out = output_grids[b]
        
        # Count objects of each color in input and output
        inp_counts = torch.zeros(10)  # ARC has colors 0-9
        out_counts = torch.zeros(10)
        
        for color in range(10):
            # Simple connected component counting (approximation)
            inp_mask = (inp == color).float()
            out_mask = (out == color).float()
            
            # Count approximate connected components
            inp_counts[color] = count_connected_components(inp_mask)
            out_counts[color] = count_connected_components(out_mask)
        
        object_counts.append(torch.stack([inp_counts, out_counts]))
    
    targets['object_counts'] = torch.stack(object_counts)
    
    # Color correspondence targets (learn color mappings)
    color_mappings = []
    for b in range(B):
        inp = input_grids[b]
        out = output_grids[b]
        
        # Simple color mapping detection
        mapping = torch.arange(10).long()  # Default identity mapping
        
        inp_colors = torch.unique(inp)
        out_colors = torch.unique(out)
        
        # Try to find consistent color mappings (simplified)
        for inp_color in inp_colors:
            inp_positions = (inp == inp_color)
            if inp_positions.sum() > 0:
                # Find most common output color at those positions
                out_values = out[inp_positions]
                if len(out_values) > 0:
                    most_common_out = torch.mode(out_values)[0]
                    mapping[inp_color] = most_common_out
        
        color_mappings.append(mapping)
    
    targets['color_mappings'] = torch.stack(color_mappings)
    
    return targets

def count_connected_components(mask: torch.Tensor) -> int:
    """Simple connected component counting for auxiliary targets"""
    if mask.sum() == 0:
        return 0
    
    visited = torch.zeros_like(mask, dtype=torch.bool)
    count = 0
    H, W = mask.shape
    
    for i in range(H):
        for j in range(W):
            if mask[i, j] > 0 and not visited[i, j]:
                count += 1
                # Flood fill
                stack = [(i, j)]
                while stack:
                    ci, cj = stack.pop()
                    if ci < 0 or ci >= H or cj < 0 or cj >= W or visited[ci, cj] or mask[ci, cj] == 0:
                        continue
                    visited[ci, cj] = True
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        stack.append((ci + di, cj + dj))
    
    return count
# Export validated versions directly
logits_from_grid = logits_from_grid_validated
size_tensor_from_grid = size_tensor_from_grid_validated
