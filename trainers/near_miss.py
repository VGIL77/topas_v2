"""
Near-Miss Miner for ARC Program Synthesis
Converts almost-correct solutions into teachable repair macros

Enhanced version with sophisticated error analysis, targeted repair strategies,
and production-grade features for systematic learning from near-miss attempts.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import logging
import math

class ErrorType(Enum):
    """Types of errors detected in near-miss predictions"""
    COLOR_MISMATCH = "color_mismatch"
    SPATIAL_SHIFT = "spatial_shift"
    ROTATION_ERROR = "rotation_error"
    SCALE_ERROR = "scale_error"
    SHAPE_DEFORMATION = "shape_deformation"
    PATTERN_INCOMPLETE = "pattern_incomplete"
    SIZE_MISMATCH = "size_mismatch"
    UNKNOWN = "unknown"

@dataclass
class ErrorAnalysis:
    """Analysis of errors between predicted and target grids"""
    error_types: List[ErrorType]
    hamming_distance: int
    color_differences: Dict[int, int]  # old_color -> new_color frequency
    spatial_offset: Tuple[int, int]    # (dy, dx) most likely offset
    similarity_score: float            # 0.0 to 1.0
    repair_complexity: str            # "simple", "moderate", "complex"

def hamming_distance(grid_a: torch.Tensor, grid_b: torch.Tensor) -> int:
    """Compute Hamming distance between two grids"""
    if grid_a.shape != grid_b.shape:
        return float('inf')
    return (grid_a != grid_b).sum().item()

def iou_score(grid_a: torch.Tensor, grid_b: torch.Tensor) -> float:
    """Compute Intersection over Union score between two grids"""
    if grid_a.shape != grid_b.shape:
        return 0.0
    intersection = (grid_a == grid_b).sum().item()
    total_cells = grid_a.numel()
    return intersection / total_cells if total_cells > 0 else 0.0

def analyze_errors(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> ErrorAnalysis:
    """Analyze the types of errors between prediction and target"""
    if pred_grid.shape != target_grid.shape:
        return ErrorAnalysis(
            error_types=[ErrorType.SIZE_MISMATCH],
            hamming_distance=float('inf'),
            color_differences={},
            spatial_offset=(0, 0),
            similarity_score=0.0,
            repair_complexity="complex"
        )

    ham_dist = hamming_distance(pred_grid, target_grid)
    similarity = iou_score(pred_grid, target_grid)
    total_cells = pred_grid.numel()

    error_types = []
    color_diffs = Counter()

    # Detect color mismatches
    mismatch_mask = pred_grid != target_grid
    if mismatch_mask.sum() > 0:
        for i in range(pred_grid.shape[0]):
            for j in range(pred_grid.shape[1]):
                if mismatch_mask[i, j]:
                    pred_color = pred_grid[i, j].item()
                    target_color = target_grid[i, j].item()
                    color_diffs[(pred_color, target_color)] += 1

        # Check if it's primarily color differences with same spatial pattern
        unique_color_pairs = len(color_diffs)
        if unique_color_pairs <= 3 and ham_dist < total_cells * 0.5:
            error_types.append(ErrorType.COLOR_MISMATCH)

    # Detect spatial shifts
    spatial_offset = detect_spatial_shift(pred_grid, target_grid)
    if abs(spatial_offset[0]) > 0 or abs(spatial_offset[1]) > 0:
        error_types.append(ErrorType.SPATIAL_SHIFT)

    # Detect rotation errors
    if is_likely_rotation_error(pred_grid, target_grid):
        error_types.append(ErrorType.ROTATION_ERROR)

    # Detect scale errors
    if is_likely_scale_error(pred_grid, target_grid):
        error_types.append(ErrorType.SCALE_ERROR)

    # Determine repair complexity
    if ham_dist <= total_cells * 0.1:
        complexity = "simple"
    elif ham_dist <= total_cells * 0.3:
        complexity = "moderate"
    else:
        complexity = "complex"

    if not error_types:
        error_types.append(ErrorType.UNKNOWN)

    return ErrorAnalysis(
        error_types=error_types,
        hamming_distance=ham_dist,
        color_differences=dict(color_diffs),
        spatial_offset=spatial_offset,
        similarity_score=similarity,
        repair_complexity=complexity
    )

def detect_spatial_shift(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> Tuple[int, int]:
    """Detect most likely spatial offset between grids"""
    if pred_grid.shape != target_grid.shape:
        return (0, 0)

    H, W = pred_grid.shape
    best_match = 0
    best_offset = (0, 0)

    # Try small shifts (-2 to +2)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue

            # Create shifted version of pred_grid
            shifted = torch.zeros_like(pred_grid)

            # Calculate valid region after shift
            start_y = max(0, dy)
            end_y = min(H, H + dy)
            start_x = max(0, dx)
            end_x = min(W, W + dx)

            pred_start_y = max(0, -dy)
            pred_start_x = max(0, -dx)

            shifted[start_y:end_y, start_x:end_x] = pred_grid[
                pred_start_y:pred_start_y + (end_y - start_y),
                pred_start_x:pred_start_x + (end_x - start_x)
            ]

            # Count matches
            matches = (shifted == target_grid).sum().item()
            if matches > best_match:
                best_match = matches
                best_offset = (dy, dx)

    return best_offset

def is_likely_rotation_error(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> bool:
    """Check if target could be a rotation of prediction"""
    if pred_grid.shape != target_grid.shape:
        return False

    # Check 90, 180, 270 degree rotations
    for k in [1, 2, 3]:  # 90, 180, 270 degrees
        rotated = torch.rot90(pred_grid, k, dims=(0, 1))
        if rotated.shape == target_grid.shape:
            matches = (rotated == target_grid).sum().item()
            total = target_grid.numel()
            if matches / total > 0.8:  # 80% match indicates likely rotation
                return True
    return False

def is_likely_scale_error(pred_grid: torch.Tensor, target_grid: torch.Tensor) -> bool:
    """Check if grids have similar patterns but different scales"""
    # This is a simplified check - in practice you'd want more sophisticated analysis
    pred_h, pred_w = pred_grid.shape
    targ_h, targ_w = target_grid.shape

    # Check if dimensions are simple multiples
    if pred_h != targ_h or pred_w != targ_w:
        h_ratio = targ_h / pred_h if pred_h > 0 else 1
        w_ratio = targ_w / pred_w if pred_w > 0 else 1

        # Check for common scaling factors
        if abs(h_ratio - 2.0) < 0.1 or abs(h_ratio - 0.5) < 0.1:
            if abs(w_ratio - 2.0) < 0.1 or abs(w_ratio - 0.5) < 0.1:
                return True

    return False

def get_targeted_repair_ops(error_analysis: ErrorAnalysis) -> List[str]:
    """Get prioritized list of repair operations based on error analysis"""
    repair_ops = []

    for error_type in error_analysis.error_types:
        if error_type == ErrorType.COLOR_MISMATCH:
            repair_ops.extend(["color_map", "for_each_object_recolor"])
        elif error_type == ErrorType.SPATIAL_SHIFT:
            repair_ops.extend(["translate", "for_each_object_translate"])
        elif error_type == ErrorType.ROTATION_ERROR:
            repair_ops.extend(["rotate90", "rotate180", "rotate270", "for_each_object_rotate"])
        elif error_type == ErrorType.SCALE_ERROR:
            repair_ops.extend(["scale", "resize_nn", "for_each_object_scale"])
        elif error_type == ErrorType.SHAPE_DEFORMATION:
            repair_ops.extend(["flip_h", "flip_v", "for_each_object_flip"])

    # Add general purpose operations
    repair_ops.extend(["identity", "crop_bbox", "flood_fill", "outline"])

    # Remove duplicates while preserving order
    seen = set()
    unique_ops = []
    for op in repair_ops:
        if op not in seen:
            seen.add(op)
            unique_ops.append(op)

    return unique_ops

def generate_repair_params(op: str, error_analysis: ErrorAnalysis, pred_grid: torch.Tensor, target_grid: torch.Tensor) -> List[Dict[str, Any]]:
    """Generate intelligent parameter sets for repair operations based on error analysis"""
    params_list = []

    if op == "translate" and ErrorType.SPATIAL_SHIFT in error_analysis.error_types:
        dy, dx = error_analysis.spatial_offset
        if abs(dy) <= 3 and abs(dx) <= 3:  # Reasonable shift
            params_list.append({"dx": dx, "dy": dy})

    elif op == "color_map" and ErrorType.COLOR_MISMATCH in error_analysis.error_types:
        # Generate color mappings based on detected differences
        for (old_color, new_color), freq in error_analysis.color_differences.items():
            if freq > 1:  # Only use frequent color changes
                params_list.append({"mapping": {old_color: new_color}})

    elif op in ["rotate90", "rotate180", "rotate270"] and ErrorType.ROTATION_ERROR in error_analysis.error_types:
        params_list.append({})  # No parameters needed for rotation

    elif op == "resize_nn" and ErrorType.SCALE_ERROR in error_analysis.error_types:
        H_target, W_target = target_grid.shape
        params_list.append({"H": H_target, "W": W_target})

    elif op in ["flip_h", "flip_v"] and ErrorType.SHAPE_DEFORMATION in error_analysis.error_types:
        params_list.append({})  # No parameters needed

    else:
        # Default parameters for other operations
        params_list.append({})

    return params_list if params_list else [{}]

def near_miss_repair(pred_grid: torch.Tensor,
                    target_grid: torch.Tensor,
                    dsl_ops: List[str],
                    dsl_shim: Any,
                    max_repairs: int = 2,
                    distance_threshold: int = 15,
                    similarity_threshold: float = 0.7) -> Tuple[torch.Tensor, List[str], float, ErrorAnalysis]:
    """
    Enhanced near-miss repair using error analysis for targeted repair strategies.

    Args:
        pred_grid: Predicted output grid
        target_grid: Target ground truth grid
        dsl_ops: Available DSL operations
        dsl_shim: DSL shim for operation application
        max_repairs: Maximum repair operations to try (1-3)
        distance_threshold: Maximum acceptable Hamming distance for near-miss
        similarity_threshold: Minimum similarity score to attempt repair

    Returns:
        repaired_grid: Best repaired grid found
        repair_ops: List of operations used for repair
        improvement: Improvement score (0.0 to 1.0)
        error_analysis: Analysis of the original errors
    """
    # Analyze errors first
    error_analysis = analyze_errors(pred_grid, target_grid)
    initial_dist = error_analysis.hamming_distance

    # Quick exit conditions
    if initial_dist == 0:
        return pred_grid, [], 1.0, error_analysis

    if (initial_dist > distance_threshold or
        error_analysis.similarity_score < similarity_threshold or
        error_analysis.repair_complexity == "complex"):
        return pred_grid, [], 0.0, error_analysis

    # Get targeted repair operations based on error analysis
    targeted_ops = get_targeted_repair_ops(error_analysis)

    # Combine with provided ops, prioritizing targeted ones
    available_ops = targeted_ops + [op for op in dsl_ops if op not in targeted_ops]

    repairs = []
    best_dist = initial_dist

    # Try single operation repairs with intelligent parameters
    for op in available_ops[:15]:  # Focus on most relevant operations
        param_sets = generate_repair_params(op, error_analysis, pred_grid, target_grid)

        for params in param_sets:
            try:
                repaired = dsl_shim.apply(op, pred_grid.clone(), **params)
                if repaired is not None and repaired.shape == target_grid.shape:
                    new_dist = hamming_distance(repaired, target_grid)
                    if new_dist < best_dist:
                        repairs.append(([op], new_dist, repaired.clone(), params))
                        best_dist = new_dist
                        if new_dist == 0:
                            # Perfect repair found!
                            improvement = 1.0
                            return repaired, [op], improvement, error_analysis
            except Exception as e:
                logging.debug(f"Single repair {op} failed: {e}")
                continue

    # Try two-step repairs for moderate complexity errors
    if max_repairs >= 2 and error_analysis.repair_complexity in ["simple", "moderate"]:
        for op1 in available_ops[:8]:  # First operation
            param_sets1 = generate_repair_params(op1, error_analysis, pred_grid, target_grid)

            for params1 in param_sets1:
                try:
                    intermediate = dsl_shim.apply(op1, pred_grid.clone(), **params1)
                    if intermediate is None or intermediate.shape != pred_grid.shape:
                        continue

                    # Reanalyze errors after first step
                    intermediate_analysis = analyze_errors(intermediate, target_grid)

                    for op2 in available_ops[:8]:  # Second operation
                        param_sets2 = generate_repair_params(op2, intermediate_analysis, intermediate, target_grid)

                        for params2 in param_sets2:
                            try:
                                repaired = dsl_shim.apply(op2, intermediate.clone(), **params2)
                                if repaired is not None and repaired.shape == target_grid.shape:
                                    new_dist = hamming_distance(repaired, target_grid)
                                    if new_dist < best_dist:
                                        repairs.append(([op1, op2], new_dist, repaired.clone(), [params1, params2]))
                                        best_dist = new_dist
                                        if new_dist == 0:
                                            # Perfect repair found!
                                            improvement = 1.0
                                            return repaired, [op1, op2], improvement, error_analysis
                            except Exception as e:
                                logging.debug(f"Two-step repair {op1}->{op2} failed: {e}")
                                continue
                except Exception as e:
                    logging.debug(f"Two-step repair {op1} intermediate failed: {e}")
                    continue

    # Return best repair found
    if repairs:
        repairs.sort(key=lambda x: x[1])  # Sort by distance (lowest first)
        best_repair = repairs[0]
        ops = best_repair[0]
        improvement = 1.0 - (best_repair[1] / initial_dist) if initial_dist > 0 else 0.0
        return best_repair[2], ops, improvement, error_analysis

    return pred_grid, [], 0.0, error_analysis


@dataclass
class RepairMacro:
    """Enhanced repair macro with comprehensive information"""
    task_id: str
    original_pred: torch.Tensor
    target_grid: torch.Tensor
    repaired_grid: torch.Tensor
    repair_ops: List[str]
    repair_params: List[Dict[str, Any]]
    improvement: float
    error_analysis: ErrorAnalysis
    initial_distance: int
    final_distance: int
    repair_confidence: float = 0.0
    timestamp: float = 0.0

class NearMissMiner:
    """
    Enhanced near-miss mining system with sophisticated error analysis and targeted repairs.

    Features:
    - Error type classification and targeted repair strategies
    - Intelligent parameter generation based on error patterns
    - Configurable thresholds and complexity limits
    - Comprehensive repair macro storage and analysis
    - Production-grade error handling and logging
    """

    def __init__(self,
                 distance_threshold: int = 15,
                 similarity_threshold: float = 0.7,
                 min_improvement: float = 0.2,
                 max_repairs: int = 2,
                 enable_complex_repairs: bool = False):
        """
        Initialize near-miss miner with configurable parameters.

        Args:
            distance_threshold: Maximum Hamming distance to attempt repairs
            similarity_threshold: Minimum similarity score to attempt repairs
            min_improvement: Minimum improvement score to store repair
            max_repairs: Maximum number of repair operations to chain
            enable_complex_repairs: Whether to attempt repairs on complex errors
        """
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.min_improvement = min_improvement
        self.max_repairs = max_repairs
        self.enable_complex_repairs = enable_complex_repairs

        # Storage for successful repairs
        self.repair_buffer: List[RepairMacro] = []
        self.repair_stats = defaultdict(int)  # Operation success counts
        self.error_type_stats = defaultdict(int)  # Error type frequency

        # Performance metrics
        self.total_attempts = 0
        self.successful_repairs = 0
        self.perfect_repairs = 0  # Exact matches after repair

        logging.info(f"[NearMissMiner] Initialized with thresholds: "
                    f"distance={distance_threshold}, similarity={similarity_threshold:.2f}, "
                    f"improvement={min_improvement:.2f}, max_repairs={max_repairs}")

    def mine_repairs(self,
                    failed_outputs: List[torch.Tensor],
                    target_grids: List[torch.Tensor],
                    task_ids: List[str],
                    dsl_ops: List[str],
                    dsl_shim: Any,
                    batch_info: Optional[Dict[str, Any]] = None) -> List[RepairMacro]:
        """
        Mine repair macros from failed attempts using enhanced error analysis.

        Args:
            failed_outputs: List of predicted grids that didn't match targets
            target_grids: List of ground truth grids
            task_ids: List of task identifiers
            dsl_ops: Available DSL operations
            dsl_shim: DSL shim for operation application
            batch_info: Optional batch metadata for improved logging

        Returns:
            repair_macros: List of successful RepairMacro objects
        """
        repair_macros = []
        import time

        for i, (pred, target, task_id) in enumerate(zip(failed_outputs, target_grids, task_ids)):
            self.total_attempts += 1

            try:
                # Skip if grids have incompatible shapes
                if pred.shape != target.shape:
                    logging.debug(f"[NearMiss] Skipping {task_id}: shape mismatch {pred.shape} vs {target.shape}")
                    continue

                # Enhanced repair with error analysis
                repaired_grid, repair_ops, improvement, error_analysis = near_miss_repair(
                    pred, target, dsl_ops, dsl_shim,
                    max_repairs=self.max_repairs,
                    distance_threshold=self.distance_threshold,
                    similarity_threshold=self.similarity_threshold
                )

                # Update error type statistics
                for error_type in error_analysis.error_types:
                    self.error_type_stats[error_type.value] += 1

                # Check if repair meets minimum improvement threshold
                if improvement >= self.min_improvement and repair_ops:
                    # Calculate repair confidence based on multiple factors
                    repair_confidence = self._calculate_repair_confidence(
                        improvement, error_analysis, repair_ops
                    )

                    # Create enhanced repair macro
                    macro = RepairMacro(
                        task_id=task_id,
                        original_pred=pred.clone(),
                        target_grid=target.clone(),
                        repaired_grid=repaired_grid.clone(),
                        repair_ops=repair_ops,
                        repair_params=self._extract_repair_params(repair_ops, error_analysis),
                        improvement=improvement,
                        error_analysis=error_analysis,
                        initial_distance=error_analysis.hamming_distance,
                        final_distance=hamming_distance(repaired_grid, target),
                        repair_confidence=repair_confidence,
                        timestamp=time.time()
                    )

                    repair_macros.append(macro)
                    self.repair_buffer.append(macro)
                    self.successful_repairs += 1

                    if macro.final_distance == 0:
                        self.perfect_repairs += 1

                    # Update repair operation statistics
                    for op in repair_ops:
                        self.repair_stats[op] += 1

                    # Enhanced logging
                    error_types_str = ", ".join([et.value for et in error_analysis.error_types])
                    logging.info(f"[NearMiss] Repair found for {task_id}: {' -> '.join(repair_ops)} "
                               f"(improvement: {improvement:.3f}, confidence: {repair_confidence:.3f}, "
                               f"errors: {error_types_str})")

                else:
                    logging.debug(f"[NearMiss] No viable repair for {task_id} "
                                f"(improvement: {improvement:.3f}, threshold: {self.min_improvement:.3f})")

            except Exception as e:
                logging.warning(f"[NearMiss] Failed to process {task_id}: {e}")
                continue

        # Log batch summary
        if repair_macros:
            success_rate = len(repair_macros) / len(failed_outputs) if failed_outputs else 0.0
            logging.info(f"[NearMiss] Batch complete: {len(repair_macros)}/{len(failed_outputs)} repairs found "
                        f"(success rate: {success_rate:.2%})")

        return repair_macros

    def _calculate_repair_confidence(self, improvement: float, error_analysis: ErrorAnalysis,
                                   repair_ops: List[str]) -> float:
        """Calculate confidence score for a repair based on multiple factors"""
        confidence = 0.0

        # Base confidence from improvement
        confidence += improvement * 0.4

        # Bonus for simple repairs (fewer operations)
        if len(repair_ops) == 1:
            confidence += 0.2
        elif len(repair_ops) == 2:
            confidence += 0.1

        # Bonus for well-understood error types
        understood_errors = {ErrorType.COLOR_MISMATCH, ErrorType.SPATIAL_SHIFT,
                           ErrorType.ROTATION_ERROR, ErrorType.SCALE_ERROR}
        if any(et in understood_errors for et in error_analysis.error_types):
            confidence += 0.2

        # Penalty for complex repair scenarios
        if error_analysis.repair_complexity == "complex":
            confidence -= 0.1

        # Bonus for high similarity
        if error_analysis.similarity_score > 0.9:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _extract_repair_params(self, repair_ops: List[str],
                              error_analysis: ErrorAnalysis) -> List[Dict[str, Any]]:
        """Extract parameters used in repair operations for future reference"""
        params = []
        for op in repair_ops:
            if op == "translate" and ErrorType.SPATIAL_SHIFT in error_analysis.error_types:
                dy, dx = error_analysis.spatial_offset
                params.append({"dx": dx, "dy": dy})
            elif op == "color_map" and error_analysis.color_differences:
                # Use the most frequent color mapping
                most_frequent = max(error_analysis.color_differences.items(),
                                  key=lambda x: x[1], default=((0, 0), 0))
                old_c, new_c = most_frequent[0]
                params.append({"mapping": {old_c: new_c}})
            else:
                params.append({})  # Default empty params
        return params

    def get_repair_priorities(self) -> Dict[str, float]:
        """Get operation priorities based on repair effectiveness"""
        if not self.repair_stats:
            return {}

        total_repairs = sum(self.repair_stats.values())
        return {op: count/total_repairs for op, count in self.repair_stats.items()}

    def get_error_type_distribution(self) -> Dict[str, float]:
        """Get distribution of error types encountered"""
        if not self.error_type_stats:
            return {}

        total_errors = sum(self.error_type_stats.values())
        return {error_type: count/total_errors for error_type, count in self.error_type_stats.items()}

    def get_training_samples(self, max_samples: int = 100,
                           min_confidence: float = 0.5,
                           sort_by_confidence: bool = True) -> List[RepairMacro]:
        """
        Get high-quality repair macros for training with enhanced filtering.

        Args:
            max_samples: Maximum number of samples to return
            min_confidence: Minimum confidence threshold for samples
            sort_by_confidence: Whether to sort by confidence (highest first)

        Returns:
            List of high-quality RepairMacro objects
        """
        if not self.repair_buffer:
            return []

        # Filter by confidence threshold
        high_quality_samples = [
            macro for macro in self.repair_buffer
            if macro.repair_confidence >= min_confidence
        ]

        # Sort by confidence if requested
        if sort_by_confidence:
            high_quality_samples.sort(key=lambda x: x.repair_confidence, reverse=True)
        else:
            # Sort by timestamp (most recent first)
            high_quality_samples.sort(key=lambda x: x.timestamp, reverse=True)

        return high_quality_samples[:max_samples]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the near-miss miner"""
        total_attempts = max(self.total_attempts, 1)  # Avoid division by zero

        return {
            "total_attempts": self.total_attempts,
            "successful_repairs": self.successful_repairs,
            "perfect_repairs": self.perfect_repairs,
            "success_rate": self.successful_repairs / total_attempts,
            "perfect_rate": self.perfect_repairs / total_attempts,
            "average_confidence": np.mean([m.repair_confidence for m in self.repair_buffer]) if self.repair_buffer else 0.0,
            "buffer_size": len(self.repair_buffer),
            "top_operations": dict(Counter(self.repair_stats).most_common(5)),
            "top_error_types": dict(Counter(self.error_type_stats).most_common(5)),
            "configuration": {
                "distance_threshold": self.distance_threshold,
                "similarity_threshold": self.similarity_threshold,
                "min_improvement": self.min_improvement,
                "max_repairs": self.max_repairs,
                "enable_complex_repairs": self.enable_complex_repairs
            }
        }

    def export_repair_dataset(self, filename: str, format: str = "json") -> None:
        """Export repair macros to file for external analysis"""
        import json
        import pickle
        import time

        export_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "total_samples": len(self.repair_buffer),
                "configuration": self.get_performance_metrics()["configuration"]
            },
            "repairs": []
        }

        for macro in self.repair_buffer:
            repair_data = {
                "task_id": macro.task_id,
                "repair_ops": macro.repair_ops,
                "repair_params": macro.repair_params,
                "improvement": macro.improvement,
                "error_types": [et.value for et in macro.error_analysis.error_types],
                "repair_confidence": macro.repair_confidence,
                "initial_distance": macro.initial_distance,
                "final_distance": macro.final_distance,
                "timestamp": macro.timestamp
            }
            export_data["repairs"].append(repair_data)

        if format.lower() == "json":
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "pickle":
            with open(filename, 'wb') as f:
                pickle.dump(export_data, f)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logging.info(f"[NearMiss] Exported {len(self.repair_buffer)} repairs to {filename}")

    def clear_buffer(self, keep_recent: int = 0):
        """
        Clear the repair buffer, optionally keeping recent samples.

        Args:
            keep_recent: Number of most recent samples to keep (0 = clear all)
        """
        if keep_recent > 0 and self.repair_buffer:
            # Keep the most recent samples
            self.repair_buffer = self.repair_buffer[-keep_recent:]
            logging.info(f"[NearMiss] Buffer cleared, keeping {len(self.repair_buffer)} recent samples")
        else:
            self.repair_buffer.clear()
            logging.info("[NearMiss] Buffer completely cleared")

        # Reset statistics
        self.repair_stats.clear()
        self.error_type_stats.clear()
        self.total_attempts = 0
        self.successful_repairs = 0
        self.perfect_repairs = 0

    def update_configuration(self, **kwargs):
        """Update miner configuration parameters"""
        updated = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                updated.append(f"{key}={value}")

        if updated:
            logging.info(f"[NearMiss] Configuration updated: {', '.join(updated)}")


def integrate_near_miss_learning(model,
                                failed_predictions: List[torch.Tensor],
                                target_outputs: List[torch.Tensor],
                                task_ids: List[str],
                                replay_buffer: Any,
                                batch_info: Optional[Dict[str, Any]] = None,
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced integration of near-miss learning into training pipeline.

    Args:
        model: The model with DSL capabilities
        failed_predictions: List of failed prediction grids
        target_outputs: List of ground truth grids
        task_ids: List of task identifiers
        replay_buffer: Replay buffer for storing training traces
        batch_info: Optional batch metadata
        config: Optional configuration overrides

    Returns:
        Dictionary with comprehensive results and metrics
    """
    import time
    from models.dsl_registry import DSL_OPS

    start_time = time.time()

    # Initialize or get existing near-miss miner
    if not hasattr(model, '_near_miss_miner'):
        miner_config = {
            'distance_threshold': 15,
            'similarity_threshold': 0.7,
            'min_improvement': 0.2,
            'max_repairs': 2,
            'enable_complex_repairs': False
        }

        # Apply config overrides
        if config:
            miner_config.update(config)

        model._near_miss_miner = NearMissMiner(**miner_config)
        logging.info(f"[NearMiss] Initialized miner with config: {miner_config}")

    # Mine repairs from failed attempts
    repair_macros = model._near_miss_miner.mine_repairs(
        failed_outputs=failed_predictions,
        target_grids=target_outputs,
        task_ids=task_ids,
        dsl_ops=DSL_OPS,
        dsl_shim=model.dsl,
        batch_info=batch_info
    )

    # Convert repair macros to training traces and add to replay buffer
    traces_added = 0
    high_priority_traces = 0

    for macro in repair_macros:
        try:
            # Convert RepairMacro to trace format compatible with replay buffer
            trace_data = {
                'task_id': macro.task_id,
                'operations': macro.repair_ops,
                'input_grid': macro.original_pred,
                'output_grid': macro.repaired_grid,
                'target_grid': macro.target_grid,
                'success_score': macro.improvement,
                'confidence_score': macro.repair_confidence,
                'error_types': [et.value for et in macro.error_analysis.error_types],
                'trace_type': 'repair_macro',
                'repair_distance': macro.final_distance,
                'timestamp': macro.timestamp
            }

            # Determine priority based on improvement and confidence
            priority = (macro.improvement * 0.6 + macro.repair_confidence * 0.4)

            # Add to replay buffer with appropriate priority
            if hasattr(replay_buffer, 'add_priority_trace'):
                replay_buffer.add_priority_trace(trace_data, priority=priority)
                if priority > 0.8:
                    high_priority_traces += 1
            elif hasattr(replay_buffer, 'add_trace'):
                replay_buffer.add_trace(trace_data)
            elif hasattr(replay_buffer, 'append'):
                replay_buffer.append(trace_data)
            else:
                logging.warning("[NearMiss] Replay buffer has no compatible add method")

            traces_added += 1

        except Exception as e:
            logging.warning(f"[NearMiss] Failed to add repair macro to buffer: {e}")

    # Compile comprehensive results
    processing_time = time.time() - start_time
    metrics = model._near_miss_miner.get_performance_metrics()

    results = {
        'repairs_found': len(repair_macros),
        'traces_added': traces_added,
        'high_priority_traces': high_priority_traces,
        'processing_time': processing_time,
        'batch_success_rate': len(repair_macros) / len(failed_predictions) if failed_predictions else 0.0,
        'miner_metrics': metrics,
        'repair_types': model._near_miss_miner.get_error_type_distribution(),
        'operation_priorities': model._near_miss_miner.get_repair_priorities()
    }

    # Enhanced logging
    if repair_macros:
        perfect_repairs = sum(1 for m in repair_macros if m.final_distance == 0)
        avg_improvement = np.mean([m.improvement for m in repair_macros])
        avg_confidence = np.mean([m.repair_confidence for m in repair_macros])

        logging.info(f"[NearMiss] Integration complete: {len(repair_macros)} repairs found, "
                    f"{perfect_repairs} perfect, avg improvement: {avg_improvement:.3f}, "
                    f"avg confidence: {avg_confidence:.3f}, time: {processing_time:.2f}s")
    else:
        logging.debug(f"[NearMiss] No repairs found in batch of {len(failed_predictions)} failures")

    return results


# Utility functions for integration with existing systems
def create_repair_trace_dataset(repair_macros: List[RepairMacro],
                               output_format: str = "jsonl") -> str:
    """Create a dataset file from repair macros for external training"""
    import json
    import tempfile
    import os

    if output_format.lower() == "jsonl":
        fd, temp_path = tempfile.mkstemp(suffix='.jsonl', prefix='repair_traces_')
        with os.fdopen(fd, 'w') as f:
            for macro in repair_macros:
                trace_entry = {
                    "task_id": macro.task_id,
                    "input": macro.original_pred.tolist(),
                    "target": macro.target_grid.tolist(),
                    "output": macro.repaired_grid.tolist(),
                    "operations": macro.repair_ops,
                    "parameters": macro.repair_params,
                    "improvement": macro.improvement,
                    "confidence": macro.repair_confidence,
                    "error_types": [et.value for et in macro.error_analysis.error_types]
                }
                f.write(json.dumps(trace_entry) + '\n')

        logging.info(f"[NearMiss] Created repair trace dataset: {temp_path} ({len(repair_macros)} traces)")
        return temp_path
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def analyze_repair_patterns(repair_macros: List[RepairMacro]) -> Dict[str, Any]:
    """Analyze patterns in successful repairs for insights"""
    if not repair_macros:
        return {"error": "No repair macros provided"}

    analysis = {
        "total_repairs": len(repair_macros),
        "operation_frequency": Counter(),
        "error_type_frequency": Counter(),
        "improvement_distribution": [],
        "confidence_distribution": [],
        "repair_length_distribution": Counter(),
        "perfect_repairs": 0,
        "common_patterns": []
    }

    for macro in repair_macros:
        # Operation frequency
        for op in macro.repair_ops:
            analysis["operation_frequency"][op] += 1

        # Error type frequency
        for error_type in macro.error_analysis.error_types:
            analysis["error_type_frequency"][error_type.value] += 1

        # Distributions
        analysis["improvement_distribution"].append(macro.improvement)
        analysis["confidence_distribution"].append(macro.repair_confidence)
        analysis["repair_length_distribution"][len(macro.repair_ops)] += 1

        if macro.final_distance == 0:
            analysis["perfect_repairs"] += 1

    # Find common operation patterns
    pattern_counter = Counter()
    for macro in repair_macros:
        pattern = " -> ".join(macro.repair_ops)
        pattern_counter[pattern] += 1

    analysis["common_patterns"] = pattern_counter.most_common(10)

    # Statistical summaries
    if analysis["improvement_distribution"]:
        analysis["avg_improvement"] = np.mean(analysis["improvement_distribution"])
        analysis["avg_confidence"] = np.mean(analysis["confidence_distribution"])

    return analysis