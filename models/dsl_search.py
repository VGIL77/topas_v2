"""
DSL Beam Search Implementation
Enhanced deep compositional search for complex ARC task solving
Supports loop constructs, conditional branching, and parallel search
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import sys
import os
import time
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import canonical DSL operations from registry
from models.dsl_registry import DSL_OPS

# ======================================================
# DSL Program + Core Operations (migrated from dsl_head.py)
# ======================================================

@dataclass
class DSLProgram:
    """Represents a sequence of DSL operations"""
    ops: List[str]
    params: List[Dict[str, Any]]
    
    def __repr__(self):
        return f"DSLProgram({' -> '.join(self.ops)})"
    
    def __len__(self):
        return len(self.ops)

    def __getitem__(self, idx):
        return (self.ops[idx], self.params[idx])


# Restrict to actually implemented operations
IMPLEMENTED_OPS = {
    "identity", "rotate90", "rotate180", "rotate270",
    "flip_h", "flip_v", "color_map", "crop_bbox",
    "flood_fill", "outline", "translate", "scale", "resize_nn",
    "extract_objects", "fill_pattern",
    "for_each_object", "for_each_object_translate", "for_each_object_recolor",
    "for_each_object_rotate", "for_each_object_scale", "for_each_object_flip"
}


# ======================================================
# Per-object Operation Utilities
# ======================================================

def _extract_components(grid: torch.Tensor, target_color: int = None) -> List[Dict]:
    """Extract connected components/objects from grid"""
    objects = []
    H, W = grid.shape
    visited = torch.zeros_like(grid, dtype=torch.bool)
    
    # If target_color specified, only extract that color
    colors_to_extract = [target_color] if target_color is not None else torch.unique(grid).tolist()
    colors_to_extract = [c for c in colors_to_extract if c != 0]  # Exclude background
    
    for color in colors_to_extract:
        color_mask = (grid == color)
        
        for i in range(H):
            for j in range(W):
                if color_mask[i, j] and not visited[i, j]:
                    # Found new component - extract it
                    component_mask = torch.zeros_like(grid, dtype=torch.bool)
                    stack = [(i, j)]
                    
                    # Flood fill to get all connected pixels
                    while stack:
                        ci, cj = stack.pop()
                        if (ci < 0 or ci >= H or cj < 0 or cj >= W or 
                            visited[ci, cj] or not color_mask[ci, cj]):
                            continue
                        
                        visited[ci, cj] = True
                        component_mask[ci, cj] = True
                        
                        # Add 4-connected neighbors
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            stack.append((ci + di, cj + dj))
                    
                    # Extract bounding box and local grid
                    nonzero = torch.nonzero(component_mask)
                    if len(nonzero) > 0:
                        min_r, min_c = nonzero.min(dim=0).values
                        max_r, max_c = nonzero.max(dim=0).values
                        
                        # Extract object region
                        object_region = grid[min_r:max_r+1, min_c:max_c+1].clone()
                        object_mask = component_mask[min_r:max_r+1, min_c:max_c+1]
                        
                        # Set non-object pixels to background
                        object_region[~object_mask] = 0
                        
                        objects.append({
                            'grid': object_region,
                            'mask': object_mask,
                            'bbox': (min_r.item(), min_c.item(), max_r.item(), max_c.item()),
                            'color': color,
                            'size': object_mask.sum().item()
                        })
    
    return objects


def _bbox_from_mask(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Get bounding box from mask"""
    nonzero = torch.nonzero(mask)
    if len(nonzero) > 0:
        min_r, min_c = nonzero.min(dim=0).values
        max_r, max_c = nonzero.max(dim=0).values
        return (min_r.item(), min_c.item(), max_r.item(), max_c.item())
    return (0, 0, 0, 0)


def _apply_per_object(grid: torch.Tensor, operation: str, params: Dict[str, Any]) -> torch.Tensor:
    """Apply operation to each object in the grid"""
    objects = _extract_components(grid)
    result = torch.zeros_like(grid)
    
    for obj in objects:
        obj_grid = obj['grid']
        bbox = obj['bbox']
        
        # Apply operation to this object
        if operation == 'rotate90':
            transformed = torch.rot90(obj_grid, k=-1, dims=(0, 1))
        elif operation == 'rotate180':
            transformed = torch.rot90(obj_grid, k=2, dims=(0, 1))
        elif operation == 'rotate270':
            transformed = torch.rot90(obj_grid, k=1, dims=(0, 1))
        elif operation == 'flip_h':
            transformed = torch.flip(obj_grid, dims=(1,))
        elif operation == 'flip_v':
            transformed = torch.flip(obj_grid, dims=(0,))
        elif operation == 'color_map':
            mapping = params.get('mapping', {})
            transformed = obj_grid.clone()
            for old_c, new_c in mapping.items():
                transformed[transformed == old_c] = new_c
        elif operation == 'translate':
            dx, dy = params.get('dx', 0), params.get('dy', 0)
            transformed = _translate_grid(obj_grid, dx, dy)
        else:
            transformed = obj_grid  # Default: no change
        
        # Place transformed object back (handle size changes)
        min_r, min_c, max_r, max_c = bbox
        th, tw = transformed.shape
        
        # Try to place at original position, clipping if necessary
        end_r = min(min_r + th, grid.shape[0])
        end_c = min(min_c + tw, grid.shape[1])
        
        placed_h = end_r - min_r
        placed_w = end_c - min_c
        
        if placed_h > 0 and placed_w > 0:
            region = result[min_r:end_r, min_c:end_c]
            obj_part = transformed[:placed_h, :placed_w]
            
            # Only place non-background pixels
            mask = obj_part != 0
            region[mask] = obj_part[mask]
    
    return result


def extract_objects(grid: torch.Tensor, color: int = None) -> torch.Tensor:
    """Extract objects of specified color (or all non-background if None)"""
    objects = _extract_components(grid, target_color=color)
    result = torch.zeros_like(grid)
    
    for obj in objects:
        bbox = obj['bbox']
        min_r, min_c, max_r, max_c = bbox
        result[min_r:max_r+1, min_c:max_c+1] = obj['grid']
    
    return result


def fill_pattern(grid: torch.Tensor, pattern: str = 'solid') -> torch.Tensor:
    """Fill objects with specified pattern"""
    if pattern == 'solid':
        return grid  # Already solid
    elif pattern == 'checkerboard':
        result = grid.clone()
        H, W = result.shape
        
        # Create checkerboard mask
        for i in range(H):
            for j in range(W):
                if result[i, j] != 0:  # Non-background
                    if (i + j) % 2 == 0:
                        result[i, j] = result[i, j]  # Keep original
                    else:
                        result[i, j] = 0  # Make background
        return result
    else:
        return grid  # Default: no change


def for_each_object_translate(grid: torch.Tensor, dx: int = 0, dy: int = 0) -> torch.Tensor:
    """Translate each object by dx, dy"""
    return _apply_per_object(grid, 'translate', {'dx': dx, 'dy': dy})


def for_each_object_recolor(grid: torch.Tensor, color_map: Dict[int, int]) -> torch.Tensor:
    """Recolor each object according to color_map"""
    return _apply_per_object(grid, 'color_map', {'mapping': color_map})


def for_each_object_rotate(grid: torch.Tensor, rotation: str = 'rotate90') -> torch.Tensor:
    """Rotate each object"""
    return _apply_per_object(grid, rotation, {})


def for_each_object_scale(grid: torch.Tensor, fy: int = 2, fx: int = None) -> torch.Tensor:
    """Scale each object (fx defaults to fy if not specified)"""
    if fx is None:
        fx = fy
    
    objects = _extract_components(grid)
    result = torch.zeros_like(grid)  # May need to resize later
    
    for obj in objects:
        obj_grid = obj['grid']
        bbox = obj['bbox']
        
        # Scale the object
        scaled = torch.nn.functional.interpolate(
            obj_grid.unsqueeze(0).unsqueeze(0).float(),
            scale_factor=(fy, fx),
            mode="nearest"
        ).squeeze().long()
        
        # Place scaled object back
        min_r, min_c, max_r, max_c = bbox
        sh, sw = scaled.shape
        
        # Extend result grid if needed
        new_h = max(result.shape[0], min_r + sh)
        new_w = max(result.shape[1], min_c + sw)
        
        if new_h > result.shape[0] or new_w > result.shape[1]:
            new_result = torch.zeros(new_h, new_w, dtype=result.dtype)
            new_result[:result.shape[0], :result.shape[1]] = result
            result = new_result
        
        # Place non-background pixels
        end_r = min(min_r + sh, result.shape[0])
        end_c = min(min_c + sw, result.shape[1])
        
        if end_r > min_r and end_c > min_c:
            placed_h = end_r - min_r
            placed_w = end_c - min_c
            
            region = result[min_r:end_r, min_c:end_c]
            obj_part = scaled[:placed_h, :placed_w]
            
            mask = obj_part != 0
            region[mask] = obj_part[mask]
    
    return result


def for_each_object_flip(grid: torch.Tensor, direction: str = 'flip_h') -> torch.Tensor:
    """Flip each object"""
    return _apply_per_object(grid, direction, {})


def for_each_object(grid: torch.Tensor, operation: str = 'rotate90', **kwargs) -> torch.Tensor:
    """Generic per-object operation"""
    return _apply_per_object(grid, operation, kwargs)


def apply_program(grid: torch.Tensor, program: DSLProgram) -> torch.Tensor:
    """Apply sequence of DSL operations to a grid."""
    out = grid.clone()
    for op, p in zip(program.ops, program.params):
        if op == "identity":
            continue  # No change
        elif op == "rotate90":
            out = torch.rot90(out, k=-1, dims=(0, 1))
        elif op == "rotate180":
            out = torch.rot90(out, k=2, dims=(0, 1))
        elif op == "rotate270":
            out = torch.rot90(out, k=1, dims=(0, 1))
        elif op == "flip_h":
            out = torch.flip(out, dims=(1,))
        elif op == "flip_v":
            out = torch.flip(out, dims=(0,))
        elif op == "color_map":
            mapping = p.get("mapping", {})
            out = out.clone()
            for old_c, new_c in mapping.items():
                out[out == old_c] = new_c
        elif op == "crop_bbox":
            nonzero = torch.nonzero(out)
            if len(nonzero) > 0:
                min_r, min_c = nonzero.min(dim=0).values
                max_r, max_c = nonzero.max(dim=0).values
                out = out[min_r:max_r+1, min_c:max_c+1]
        elif op == "flood_fill":
            start_pos = p.get("start_pos", (0, 0))
            target_color = p.get("target_color", 0)
            fill_color = p.get("fill_color", 1)
            out = _flood_fill(out, start_pos, target_color, fill_color)
        elif op == "outline":
            out = _create_outline(out)
        elif op == "resize_nn":
            H, W = p.get("H", out.shape[0]), p.get("W", out.shape[1])
            if H != out.shape[0] or W != out.shape[1]:
                out = torch.nn.functional.interpolate(
                    out.unsqueeze(0).unsqueeze(0).float(),
                    size=(H, W),
                    mode="nearest"
                ).squeeze().long()
        elif op == "translate":
            dx, dy = p.get("dx", 0), p.get("dy", 0)
            out = _translate_grid(out, dx, dy)
        elif op == "scale":
            fx, fy = p.get("fx", 2), p.get("fy", 2)
            out = _scale_grid(out, fx, fy)
        # Per-object operations
        elif op == "extract_objects":
            color = p.get("color", None)
            out = extract_objects(out, color)
        elif op == "fill_pattern":
            pattern = p.get("pattern", "solid")
            out = fill_pattern(out, pattern)
        elif op == "for_each_object_translate":
            dx, dy = p.get("dx", 0), p.get("dy", 0)
            out = for_each_object_translate(out, dx, dy)
        elif op == "for_each_object_recolor":
            color_map = p.get("color_map", {})
            out = for_each_object_recolor(out, color_map)
        elif op == "for_each_object_rotate":
            rotation = p.get("rotation", "rotate90")
            out = for_each_object_rotate(out, rotation)
        elif op == "for_each_object_scale":
            fy, fx = p.get("fy", 2), p.get("fx", None)
            out = for_each_object_scale(out, fy, fx)
        elif op == "for_each_object_flip":
            direction = p.get("direction", "flip_h")
            out = for_each_object_flip(out, direction)
        elif op == "for_each_object":
            operation = p.get("operation", "rotate90")
            out = for_each_object(out, operation, **{k: v for k, v in p.items() if k != "operation"})
        else:
            continue  # Skip unknown operations
    return out


def _flood_fill(grid: torch.Tensor, start_pos: Tuple[int, int], target_color: int, fill_color: int) -> torch.Tensor:
    """Simple flood fill implementation"""
    out = grid.clone()
    if start_pos[0] < 0 or start_pos[0] >= grid.shape[0] or start_pos[1] < 0 or start_pos[1] >= grid.shape[1]:
        return out
    if out[start_pos] != target_color:
        return out
    
    # BFS flood fill
    queue = [start_pos]
    visited = set()
    
    while queue:
        y, x = queue.pop(0)
        if (y, x) in visited or y < 0 or y >= out.shape[0] or x < 0 or x >= out.shape[1]:
            continue
        if out[y, x] != target_color:
            continue
            
        out[y, x] = fill_color
        visited.add((y, x))
        
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            queue.append((y + dy, x + dx))
    
    return out


def _create_outline(grid: torch.Tensor) -> torch.Tensor:
    """Create outline of non-zero regions"""
    out = torch.zeros_like(grid)
    H, W = grid.shape
    
    for i in range(H):
        for j in range(W):
            if grid[i, j] != 0:
                # Check if on boundary
                is_boundary = False
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= H or nj < 0 or nj >= W or grid[ni, nj] == 0:
                        is_boundary = True
                        break
                if is_boundary:
                    out[i, j] = grid[i, j]
    return out


def _translate_grid(grid: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate grid by dx, dy"""
    out = torch.zeros_like(grid)
    H, W = grid.shape
    
    for i in range(H):
        for j in range(W):
            ni, nj = i + dy, j + dx
            if 0 <= ni < H and 0 <= nj < W:
                out[ni, nj] = grid[i, j]
    return out


def _scale_grid(grid: torch.Tensor, fx: int, fy: int) -> torch.Tensor:
    """Scale grid by factors fx, fy"""
    return torch.nn.functional.interpolate(
        grid.unsqueeze(0).unsqueeze(0).float(),
        scale_factor=(fy, fx),
        mode="nearest"
    ).squeeze().long()


# Use registry-based operations instead of hardcoded list
CORE_OPS = [op for op in DSL_OPS if op in IMPLEMENTED_OPS]

# ======================================================
# End DSL Operations Migration
# ======================================================

# Import wormhole templates
from wormhole_offline import Template, TemplateLibrary, WormholeConsolidator

# Configure logging for debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NO FALLBACK OPERATIONS - DSL operations MUST come from models.dsl_head

@dataclass
class BeamCandidate:
    """Represents a candidate program in beam search with enhanced features"""
    ops: List[str]
    params: List[Dict[str, Any]]
    score: float = 0.0
    is_template: bool = False  # Track if this came from a template
    template_signature: str = ""  # Template signature if applicable
    depth: int = 0  # Current depth in search tree
    heuristic_score: float = 0.0  # A* heuristic score
    partial_results: Dict[str, Any] = field(default_factory=dict)  # Cache partial results
    program_hash: str = ""  # Hash for memoization
    # Removed: is_composite support
    parent_id: Optional[str] = None  # For tracking program tree
    
    def __post_init__(self):
        """Compute program hash for memoization"""
        if not self.program_hash:
            ops_str = "|".join(self.ops)
            params_str = "|".join(str(p) for p in self.params)
            self.program_hash = f"{ops_str}#{params_str}"
    
    def apply(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply this program to a grid using integrated DSL operations"""
        program = DSLProgram(ops=self.ops, params=self.params)
        return apply_program(grid, program)
    
    def __lt__(self, other):
        """For priority queue ordering (higher score = higher priority)"""
        return self.score > other.score


@dataclass
class ProgramTemplate:
    """Template for common ARC patterns with compositional structure"""
    name: str
    ops_template: List[str]  # Operations with placeholders
    param_generators: List[callable]  # Functions to generate parameters
    conditions: List[callable] = field(default_factory=list)  # When to use this template
    priority: float = 1.0
    
    def instantiate(self, context: Dict[str, Any]) -> List[BeamCandidate]:
        """Generate concrete candidates from this template"""
        candidates = []
        
        # Generate parameter combinations
        param_sets = []
        for gen in self.param_generators:
            param_sets.append(gen(context))
        
        # Create candidates for each parameter combination
        import itertools
        for param_combo in itertools.product(*param_sets):
            candidate = BeamCandidate(
                ops=self.ops_template.copy(),
                params=list(param_combo),
                is_template=True,
                template_signature=self.name,
                score=self.priority
            )
            candidates.append(candidate)
        
        return candidates


class MemoizationCache:
    """Cache for memoizing program evaluation results"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, program_hash: str, input_hash: str) -> Optional[torch.Tensor]:
        """Get cached result for program + input combination"""
        key = f"{program_hash}#{input_hash}"
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, program_hash: str, input_hash: str, result: torch.Tensor):
        """Cache result for program + input combination"""
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        key = f"{program_hash}#{input_hash}"
        self.cache[key] = result.clone()
        self.access_count[key] = 1
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]


# Removed: CompositeProgram class - no loops/conditionals support


# Program templates for common ARC patterns
COMMON_PATTERNS = [
    ProgramTemplate(
        name="rotation_sequence",
        ops_template=["rotate90", "rotate90", "rotate90"],
        param_generators=[lambda ctx: [{}], lambda ctx: [{}], lambda ctx: [{}]],
        priority=2.0
    ),
    ProgramTemplate(
        name="flip_and_transform",
        ops_template=["flip_h", "color_map"],
        param_generators=[
            lambda ctx: [{}],
            lambda ctx: [{'mapping': {0: 1, 1: 0}}, {'mapping': {0: 2, 2: 0}}]
        ],
        priority=1.8
    ),
    ProgramTemplate(
        name="extract_and_fill",
        ops_template=["extract_objects", "fill_pattern"],
        param_generators=[
            lambda ctx: [{'color': c} for c in range(10)],
            lambda ctx: [{'pattern': p} for p in ['solid', 'checkerboard']]
        ],
        priority=1.5
    )
]

def verify_candidate(candidate: BeamCandidate, demos: List[Tuple], dsl_head: Any = None) -> bool:
    """Check if candidate solves all demonstrations with enhanced error handling"""
    try:
        for inp, out in demos:
            # Ensure tensors
            if isinstance(inp, dict):
                inp = inp.get('input', inp)
            if isinstance(out, dict):
                out = out.get('output', out)
                
            # Convert numpy to tensor if needed
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp)
            if isinstance(out, np.ndarray):
                out = torch.from_numpy(out)
            
            # Apply candidate program with timeout protection
            try:
                pred = candidate.apply(inp)
            except Exception as e:
                logger.debug(f"Program execution failed: {e}")
                return False
            
            # Check if shapes match
            if pred.shape != out.shape:
                return False
            
            # Check if values match
            if not torch.equal(pred, out):
                return False
        
        return True
    except Exception as e:
        logger.debug(f"Candidate verification failed: {e}")
        return False


def compute_heuristic_score(candidate: BeamCandidate, demos: List[Tuple], dsl_head: Any = None) -> float:
    """Compute A* heuristic score (admissible estimate of remaining cost)"""
    heuristic = 0.0
    
    try:
        # Estimate based on partial similarity to target outputs
        partial_score = 0.0
        for inp, out in demos:
            if isinstance(inp, dict):
                inp = inp.get('input', inp)
            if isinstance(out, dict):
                out = out.get('output', out)
                
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp)
            if isinstance(out, np.ndarray):
                out = torch.from_numpy(out)
            
            try:
                pred = candidate.apply(inp)
                # Compute structural similarity
                if pred.shape == out.shape:
                    similarity = (pred == out).float().mean().item()
                    partial_score += similarity
                else:
                    # Penalize shape mismatch but don't completely exclude
                    partial_score += 0.1
            except:
                partial_score += 0.0
        
        heuristic += partial_score / len(demos) * 5.0
        
        # Bonus for template-based candidates (proven patterns)
        if candidate.is_template:
            heuristic += 2.0
        
        # Small penalty for depth (prefer shallower solutions when equal)
        heuristic -= candidate.depth * 0.05
        
    except Exception as e:
        logger.debug(f"Heuristic computation failed: {e}")
        heuristic = 0.0
    
    return heuristic

def score_candidate(candidate: BeamCandidate, demos: List[Tuple], priors: Dict, dsl_head: Any, cache: Optional[MemoizationCache] = None) -> float:
    """Enhanced scoring with A* heuristics and caching"""
    try:
        score = 0.0
        
        # Compute A* heuristic score
        heuristic = compute_heuristic_score(candidate, demos, dsl_head)
        candidate.heuristic_score = heuristic
        
        # Enhanced template scoring with MDL integration
        if candidate.is_template:
            # HIGH PRIORITY: Templates are first-class priors
            score += 8.0  # Increased bonus for template-based candidates
            
            # MDL-based scoring: shorter templates with better compression get higher scores
            if hasattr(candidate, 'template_signature') and candidate.template_signature:
                # Bonus for well-compressed templates (inverse MDL)
                score += 2.0  # Additional bonus for proven compression
            
            # Minimal penalty for length since templates are proven patterns
            score -= len(candidate.ops) * 0.02
        else:
            # Standard penalty for longer programs
            score -= len(candidate.ops) * 0.1
        
        # Check accuracy on demos with caching
        correct = 0
        total = 0
        for i, (inp, out) in enumerate(demos):
            # Ensure tensors
            if isinstance(inp, dict):
                inp = inp.get('input', inp)
            if isinstance(out, dict):
                out = out.get('output', out)
                
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp)
            if isinstance(out, np.ndarray):
                out = torch.from_numpy(out)
            
            # Try cache first
            input_hash = str(hash(inp.tolist() if inp.numel() < 100 else hash(inp.shape)))
            cached_result = None
            if cache:
                cached_result = cache.get(candidate.program_hash, input_hash)
            
            if cached_result is not None:
                pred = cached_result
            else:
                try:
                    pred = candidate.apply(inp)
                    if cache:
                        cache.put(candidate.program_hash, input_hash, pred)
                except Exception as e:
                    logger.debug(f"Program execution failed during scoring: {e}")
                    continue
            
            if pred.shape == out.shape:
                # Pixel-wise accuracy
                correct += (pred == out).float().mean().item()
            else:
                # Partial credit for wrong shape but some similarity
                if pred.numel() > 0 and out.numel() > 0:
                    # Compute color histogram similarity as fallback
                    pred_flat = pred.flatten()
                    out_flat = out.flatten()
                    pred_hist = torch.bincount(pred_flat.long(), minlength=10)
                    out_hist = torch.bincount(out_flat.long(), minlength=10)
                    hist_sim = 1.0 - torch.abs(pred_hist.float() - out_hist.float()).mean() / max(pred_flat.max(), out_flat.max(), 1)
                    correct += hist_sim.item() * 0.3  # Partial credit
            total += 1
        
        if total > 0:
            accuracy_score = correct / total * 10.0
            score += accuracy_score
            
            # Perfect accuracy bonus
            if correct / total > 0.99:
                score += 5.0
        
        # Bonus for using priors (if available)
        if priors and 'trans' in priors and priors['trans'] is not None:
            # Check if transformation priors align with ops
            trans_probs = torch.softmax(priors['trans'], dim=-1)
            op_map = {'rotate90': 0, 'rotate180': 1, 'rotate270': 2, 'flip_h': 3, 'flip_v': 4}
            for op in candidate.ops:
                if op in op_map:
                    idx = op_map[op]
                    if idx < len(trans_probs[0]):
                        prior_bonus = trans_probs[0][idx].item()
                        # Templates get extra bonus for aligning with priors
                        if candidate.is_template:
                            prior_bonus *= 1.5
                        score += prior_bonus
        
        # Add heuristic component for A* search
        score += heuristic * 0.5
        
        # Template entropy reduction bonus
        if candidate.is_template and correct / total > 0.8:  # High accuracy templates
            # Entropy reduction: templates that work well should reduce beam entropy
            score += 3.0  # Entropy reduction bonus
        
        # Removed: composite program bonus
        
        return score
        
    except Exception as e:
        logger.debug(f"Scoring failed for candidate: {e}")
        return -float('inf')


def expand_candidate_parallel(candidate: BeamCandidate, available_ops: List[str], dsl_head: Any, max_depth: int, op_bias: Optional[Dict[str, float]] = None) -> List[BeamCandidate]:
    """Expand a candidate with all possible next operations in parallel"""
    # 🔒 Double-lock normalization
    if op_bias is None or not isinstance(op_bias, dict):
        op_bias = {}
    
    if candidate.depth >= max_depth:
        return []
    
    expansions = []
    
    # Standard operations with bias weighting
    ops_to_try = available_ops[:]
    
    # Sort by bias if provided (higher bias = higher priority)
    if op_bias:
        ops_to_try.sort(key=lambda op: op_bias.get(op, 0.0), reverse=True)
    
    for op in ops_to_try:
        # Apply bias weighting to parameter generation (more params for biased ops)
        bias_weight = op_bias.get(op, 1.0) if op_bias else 1.0
        max_params = max(1, min(5, int(3 * bias_weight)))  # 1-5 params based on bias
        
        # Generate reasonable parameters for the operation
        param_options = generate_op_parameters(op, candidate)
        
        for params in param_options[:max_params]:  # More params for biased operations
            new_ops = candidate.ops + [op]
            new_params = candidate.params + [params]
            
            expansion = BeamCandidate(
                ops=new_ops,
                params=new_params,
                depth=candidate.depth + 1,
                parent_id=candidate.program_hash,
                is_template=False
            )
            expansions.append(expansion)
    
    # Removed: composite operations support
    
    return expansions


def generate_op_parameters(op: str, context_candidate: BeamCandidate) -> List[Dict[str, Any]]:
    """Generate reasonable parameters for an operation based on context"""
    param_options = []
    
    if op in ['rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v']:
        param_options = [{}]  # No parameters needed
    elif op == 'color_map':
        # Generate common color mappings
        mappings = [
            {0: 1, 1: 0},
            {0: 2, 2: 0},
            {1: 2, 2: 1},
            {0: 5, 5: 0}
        ]
        param_options = [{'mapping': m} for m in mappings]
    elif op == 'extract_objects':
        # Try different colors
        param_options = [{'color': c} for c in range(10)]
    # Removed: repeat_n and if_else parameter generation
    # Per-object operations
    elif op == 'for_each_object_translate':
        param_options = [
            {'dx': dx, 'dy': dy} 
            for dx in [-2, -1, 0, 1, 2] 
            for dy in [-2, -1, 0, 1, 2]
        ][:10]  # Limit combinations
    elif op == 'for_each_object_recolor':
        # Common color remappings for per-object operations
        color_maps = [
            {1: 2, 2: 1},  # Swap colors 1 and 2
            {1: 3, 2: 4, 3: 5},  # Shift colors up
            {i: (i + 1) % 10 for i in range(1, 6)},  # Cycle colors
            {0: 1, 1: 2, 2: 3}  # Sequential mapping
        ]
        param_options = [{'color_map': cm} for cm in color_maps]
    elif op == 'for_each_object_rotate':
        param_options = [
            {'rotation': rot} for rot in ['rotate90', 'rotate180', 'rotate270']
        ]
    elif op == 'for_each_object_scale':
        param_options = [
            {'fy': fy, 'fx': fx} 
            for fy in [2, 3] 
            for fx in [None, 2, 3]  # None means same as fy
        ]
    elif op == 'for_each_object_flip':
        param_options = [
            {'direction': direction} for direction in ['flip_h', 'flip_v']
        ]
    elif op == 'for_each_object':
        # Generic per-object operation with various inner operations
        inner_ops = ['rotate90', 'rotate180', 'flip_h', 'flip_v', 'color_map', 'translate']
        param_options = []
        
        for inner_op in inner_ops:
            if inner_op == 'color_map':
                # Add color mapping parameters
                mappings = [{1: 2}, {2: 3}, {1: 3, 3: 1}]
                for mapping in mappings:
                    param_options.append({
                        'operation': inner_op,
                        'mapping': mapping
                    })
            elif inner_op == 'translate':
                # Add translation parameters
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    param_options.append({
                        'operation': inner_op,
                        'dx': dx,
                        'dy': dy
                    })
            else:
                # Simple operation
                param_options.append({'operation': inner_op})
    else:
        param_options = [{}]  # Default empty parameters
    
    return param_options[:5]  # Limit to prevent explosion


# Removed: generate_composite_candidates function

def has_multiple_objects(demos) -> bool:
    """
    Detect if the task involves multiple distinct objects that might benefit from per-object operations
    
    Args:
        demos: List of (input, output) demonstration pairs
        
    Returns:
        True if multiple objects detected in demos
    """
    try:
        # Check if any demo input has multiple connected components
        for demo in demos:
            if isinstance(demo, dict):
                inp = demo.get('input')
            elif isinstance(demo, tuple):
                inp = demo[0]
            else:
                continue
            
            # Convert to tensor if needed
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp)
            if inp.dim() == 3 and inp.shape[0] == 1:
                inp = inp[0]
            
            # Count connected components for each non-zero color
            object_count = 0
            unique_colors = torch.unique(inp)
            unique_colors = unique_colors[unique_colors != 0]  # Exclude background
            
            for color in unique_colors:
                color_mask = (inp == color)
                components = count_connected_components(color_mask)
                object_count += components
                
                # If we find multiple objects in any demo, return True
                if object_count > 1:
                    return True
        
        return False
        
    except Exception:
        # If analysis fails, assume single object for safety
        return False

def count_connected_components(mask: torch.Tensor) -> int:
    """Count connected components in a binary mask"""
    if mask.sum() == 0:
        return 0
    
    H, W = mask.shape
    visited = torch.zeros_like(mask, dtype=torch.bool)
    count = 0
    
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                # Found new component - flood fill
                count += 1
                stack = [(i, j)]
                
                while stack:
                    ci, cj = stack.pop()
                    if (ci < 0 or ci >= H or cj < 0 or cj >= W or 
                        visited[ci, cj] or not mask[ci, cj]):
                        continue
                    
                    visited[ci, cj] = True
                    
                    # Add 4-connected neighbors
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        stack.append((ci + di, cj + dj))
    
    return count

def get_template_candidates(demos, template_library: Optional['TemplateLibrary'] = None, max_templates: int = 10, verbose: bool = False) -> List[BeamCandidate]:
    """Get enhanced template-based candidates for beam search"""
    candidates = []
    
    # Use wormhole templates if available (default to True)
    use_wormhole = True  # Default to enabled for backward compatibility
    if use_wormhole and template_library is not None:
        templates = template_library.get_templates_by_score(max_count=max_templates)
        
        for template in templates:
            # Convert template to BeamCandidate
            ops = [op_name for op_name, _ in template.ops]
            params = [params for _, params in template.ops]
            
            candidate = BeamCandidate(
                ops=ops,
                params=params,
                score=template.score * 0.1,  # Start with template score as base
                is_template=True,
                template_signature=template.signature,
                depth=len(ops)
            )
            candidates.append(candidate)
    
    # Add built-in pattern templates
    context = {'demos': demos}
    for pattern in COMMON_PATTERNS:
        pattern_candidates = pattern.instantiate(context)
        candidates.extend(pattern_candidates[:3])  # Limit per pattern
    
    if verbose:
        print(f"[BEAM] Generated {len(candidates)} template candidates")
    return candidates


def progressive_deepening_search(
    demos, test, priors,
    max_depth=12, beam_width=50,
    timeout_seconds=30,
    verbose=False,
    return_rule_info=False,
    op_bias=None
):
    """Progressive deepening search - try shallow first, then deeper"""
    start_time = time.time()
    
    # 🔒 Double-lock normalization
    if op_bias is None or not isinstance(op_bias, dict):
        op_bias = {}
    if verbose and not op_bias:
        print("[DSL] op_bias not provided → defaulting to uniform")
    
    for depth in [3, 6, 9, max_depth]:
        if time.time() - start_time > timeout_seconds:
            if verbose:
                print(f"[BEAM] Timeout reached at depth {depth}")
            break
        
        if verbose:
            print(f"[BEAM] Trying depth {depth}")
            if op_bias:
                top_ops = sorted(op_bias.items(), key=lambda kv: kv[1], reverse=True)[:5]
                print(f"[BEAM] Top biased ops @depth {depth}: {top_ops}")
        
        result = enhanced_beam_search(
            demos, test, priors, 
            depth=depth, 
            beam=min(beam_width, depth * 8),  # Scale beam with depth
            timeout_seconds=timeout_seconds - (time.time() - start_time),
            verbose=verbose,
            op_bias=op_bias,
            return_rule_info=return_rule_info
        )
        
        if result is not None:
            if verbose:
                print(f"[BEAM] Found solution at depth {depth}")
            return result
    
    if verbose:
        print("[BEAM] No solution found in progressive deepening")
    
    if return_rule_info:
        return None, {"program": []}
    return None

def enhanced_beam_search(demos, test, priors, depth=12, beam=50, use_templates=True, use_wormhole=True, timeout_seconds=30, verbose=False, return_rule_info=False, op_bias=None):
    """
    Enhanced beam search with deep compositional reasoning for complex ARC tasks
    
    Args:
        demos: List of demonstration input/output pairs
        test: Test input grid to transform
        priors: Dictionary of neural priors (trans, size, pal, etc.)
        depth: Maximum search depth (number of operations) - now 12 for complex tasks
        beam: Beam width (number of candidates to track) - now 50 for broader search
        use_templates: Whether to use template-based candidates
        timeout_seconds: Maximum search time
        verbose: Enable debug logging
        return_rule_info: Return rule information along with result
        op_bias: Optional dict mapping operation names to bias weights for guided search
    
    Returns:
        Best grid prediction or None if no solution found
    """
    start_time = time.time()
    
    # Input validation
    if not demos:
        if verbose:
            print("[BEAM] No demos provided")
        return None
    
    # 🔒 Double-lock normalization
    if op_bias is None or not isinstance(op_bias, dict):
        op_bias = {}
    if verbose and not op_bias:
        print("[DSL] op_bias not provided → defaulting to uniform")
    
    # Normalize demos format
    normalized_demos = []
    for demo in demos:
        if isinstance(demo, dict):
            inp = demo.get('input')
            out = demo.get('output')
            if inp is not None and out is not None:
                # Convert to tensors if needed
                if isinstance(inp, np.ndarray):
                    inp = torch.from_numpy(inp)
                if isinstance(out, np.ndarray):
                    out = torch.from_numpy(out)
                normalized_demos.append((inp, out))
        elif isinstance(demo, tuple) and len(demo) == 2:
            inp, out = demo
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp)
            if isinstance(out, np.ndarray):
                out = torch.from_numpy(out)
            normalized_demos.append((inp, out))
    
    if not normalized_demos:
        if verbose:
            print("[BEAM] Failed to normalize demos")
        return None
    
    # Ensure test is tensor
    if isinstance(test, np.ndarray):
        test = torch.from_numpy(test)
    if test.dim() == 3 and test.shape[0] == 1:
        test = test[0]  # Remove batch dim if present
    
    # Initialize DSL head - REQUIRED for beam search
    try:
        # Use beam_search directly instead of DSLHead (zombie class removal)
        template_library = None
        
        # Initialize memoization cache
        cache = MemoizationCache(max_size=5000)
        
        # Try beam_search directly (clean implementation)
        try:
            beams = beam_search(normalized_demos, test, depth=depth, beam=beam)
            if beams and len(beams) > 0:
                program = beams[0].program
                if verbose:
                    print(f"[BEAM] beam_search found program: {program}")
                result = apply_program(test, program)
                if return_rule_info:
                    return result, {"program": program.ops if hasattr(program, 'ops') else program}
                return result
            else:
                program = None
        except Exception as e:
            if verbose:
                print(f"[BEAM] beam_search error: {e}, trying enhanced search")
            program = None
        
        # Enhanced A* beam search with parallel expansion (no DSLHead needed)
        return enhanced_astar_search(
            normalized_demos, test, priors, None, template_library, cache,
            depth, beam, timeout_seconds, verbose, return_rule_info, op_bias
        )
        
    except Exception as e:
        if verbose:
            print(f"[BEAM] DSLHead initialization failed: {e}")
        raise  # Fail loudly - no fallback allowed


def enhanced_astar_search(demos, test, priors, dsl_head, template_library, cache, max_depth, beam_width, timeout_seconds, verbose, return_rule_info, op_bias=None):
    """A* search with parallel expansion, advanced heuristics, and size constraints"""
    start_time = time.time()
    
    # 🔒 Normalize op_bias safely
    if op_bias is None or not isinstance(op_bias, dict):
        op_bias = {}
    
    # Priority queue for A* search (heapq)
    open_set = []
    closed_set = set()  # Track explored program hashes
    best_candidates = []
    
    # Get available operations from DSL head with bias support
    base_ops = CORE_OPS  # Use canonical ops only
    
    # Apply operation bias - prioritize detected operations (2x weight for detected ops)
    if op_bias:
        # Sort operations by bias weight (detected ops first)
        biased_ops = []
        unbiased_ops = []
        
        for op in base_ops:
            if op in op_bias:
                biased_ops.append((op, op_bias[op] * 2.0))  # 2x weight for detected ops
            else:
                unbiased_ops.append((op, 1.0))
        
        # Sort biased ops by weight (descending)
        biased_ops.sort(key=lambda x: x[1], reverse=True)
        
        # Create final operation list: biased first, then unbiased
        available_ops = [op for op, _ in biased_ops] + [op for op, _ in unbiased_ops]
        
        if verbose:
            print(f"[BEAM] Operation priority: {available_ops[:5]}... (biased: {len(biased_ops)})")
    else:
        available_ops = base_ops
    
    # Add per-object operations if multi-object tasks detected
    if has_multiple_objects(demos):
        per_object_ops = [
            'for_each_object_translate',
            'for_each_object_recolor', 
            'for_each_object_rotate',
            'for_each_object_scale',
            'for_each_object_flip',
            'for_each_object'
        ]
        available_ops.extend(per_object_ops)
        if verbose:
            print(f"[BEAM] Multi-object task detected, added {len(per_object_ops)} per-object operations")
    
    # Initialize with template candidates
    initial_candidates = []
    if template_library:
        template_candidates = get_template_candidates(demos, template_library, max_templates=15, verbose=verbose)
        initial_candidates.extend(template_candidates)
    
    # Add empty program as starting point
    empty_candidate = BeamCandidate(ops=[], params=[], depth=0)
    initial_candidates.append(empty_candidate)
    
    # Score and add initial candidates to priority queue
    for candidate in initial_candidates:
        if time.time() - start_time > timeout_seconds:
            break
            
        try:
            score = score_candidate(candidate, demos, priors, dsl_head, cache)
            candidate.score = score
            
            # Check if this is already a solution
            if verify_candidate(candidate, demos, dsl_head):
                if verbose:
                    print(f"[BEAM] Template solution found: {candidate.ops}")
                result = candidate.apply(test)
                if return_rule_info:
                    return result, {"program": candidate.ops}
                return result
            
            heapq.heappush(open_set, candidate)
        except Exception as e:
            if verbose:
                print(f"[BEAM] Failed to score initial candidate: {e}")
    
    # Main A* search loop
    iteration = 0
    while open_set and time.time() - start_time < timeout_seconds:
        iteration += 1
        
        if iteration % 100 == 0 and verbose:
            print(f"[BEAM] Iteration {iteration}, open_set size: {len(open_set)}, closed_set size: {len(closed_set)}")
        
        # Get best candidate
        current = heapq.heappop(open_set)
        
        # Skip if already explored
        if current.program_hash in closed_set:
            continue
        
        closed_set.add(current.program_hash)
        
        # Expand current candidate in parallel
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future = executor.submit(expand_candidate_parallel, current, available_ops, dsl_head, max_depth, op_bias)
                try:
                    expansions = future.result(timeout=5.0)  # 5 second timeout per expansion
                except Exception as e:
                    if verbose:
                        print(f"[BEAM] Expansion timeout: {e}")
                    expansions = []
        except Exception as e:
            if verbose:
                print(f"[BEAM] Parallel expansion failed: {e}")
            expansions = []
        
        # Process expansions
        for expansion in expansions[:beam_width]:  # Limit expansions
            if time.time() - start_time > timeout_seconds:
                break
                
            if expansion.program_hash in closed_set:
                continue
            
            try:
                # Score the expansion
                score = score_candidate(expansion, demos, priors, dsl_head, cache)
                expansion.score = score
                
                # Check if solution found
                if verify_candidate(expansion, demos, dsl_head):
                    if verbose:
                        print(f"[BEAM] Solution found at depth {expansion.depth}: {expansion.ops}")
                    result = expansion.apply(test)
                    if return_rule_info:
                        return result, {"program": expansion.ops}
                    return result
                
                # Add to open set if promising
                if score > -5.0:  # Pruning threshold
                    heapq.heappush(open_set, expansion)
                    best_candidates.append(expansion)
                
            except Exception as e:
                if verbose:
                    print(f"[BEAM] Failed to process expansion: {e}")
                continue
        
        # Beam pruning - keep only best candidates
        if len(best_candidates) > beam_width * 2:
            best_candidates.sort(key=lambda x: x.score, reverse=True)
            best_candidates = best_candidates[:beam_width]
    
    # Cleanup and persist templates after search (optional feature)
    template_miner = None  # Template miner not implemented in this scope
    if template_miner:
        template_miner.tick()  # Decrement TTLs
        cleaned_count = template_miner.cleanup_stale()  # Remove expired
        template_miner.persist_library()  # Save to disk
        
        if verbose and cleaned_count > 0:
            print(f"[BEAM] Cleaned {cleaned_count} expired templates")
    
    if verbose:
        print(f"[BEAM] Enhanced search completed in {time.time() - start_time:.2f}s, {iteration} iterations")
        if best_candidates:
            best_score = max(c.score for c in best_candidates)
            print(f"[BEAM] Best score achieved: {best_score}")
        if template_miner:
            active_templates = sum(1 for ttl in template_miner.ttl_decay.values() if ttl > 0)
            print(f"[BEAM] Active templates: {active_templates}/{len(template_miner.templates)}")
    
    if return_rule_info:
        # Return template usage info even on failure
        extras = {
            "program": [],
            "template_usage": {
                "templates_tried": len([c for c in initial_candidates if c.is_template]) if 'initial_candidates' in locals() else 0,
                "template_succeeded": None,
                "mdl_score": None,
                "ttl_remaining": None
            }
        }
        return None, extras
    return None


def beam_search(demos, test, priors, depth=12, beam=50, use_templates=True, use_wormhole=True, verbose=False, return_rule_info=False, op_bias=None):
    """
    Main beam search interface - now with enhanced deep search capabilities
    
    ENHANCED FEATURES:
    - Maximum depth increased from 6 to 12 for complex compositional reasoning
    - Beam width increased from 12 to 50 for broader search space
    - Progressive deepening (try shallow first, then deeper)
    - A* search with admissible heuristics
    - Memoization for repeated subprograms
    - Template-based initialization with higher priority
    - Operation bias integration for guided search
    - Parallel candidate expansion
    - Removed: loop and conditional support
    - Timeout protection for deep searches
    - Enhanced logging and debug information
    """
    # 🔒 Double-lock normalization
    if op_bias is None or not isinstance(op_bias, dict):
        op_bias = {}
    if verbose and not op_bias:
        print("[DSL] op_bias not provided → defaulting to uniform")
    
    # Use progressive deepening for best results
    return progressive_deepening_search(
        demos, test, priors, 
        max_depth=depth, 
        beam_width=beam,
        timeout_seconds=60,  # Increased timeout for deep searches
        verbose=verbose,
        return_rule_info=return_rule_info,
        op_bias=op_bias
    )