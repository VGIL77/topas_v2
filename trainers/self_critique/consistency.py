"""
Consistency Enforcer for TOPAS ARC Solver - Phase 3
Ensures different valid traces produce consistent results and reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import itertools
from dataclasses import dataclass
from collections import defaultdict
import logging
from scipy.stats import entropy

@dataclass
class ConsistencyMetrics:
    """Metrics for trace consistency analysis"""
    grid_consistency: float
    probability_consistency: float
    operation_consistency: float
    reasoning_consistency: float
    overall_consistency: float
    inconsistency_sources: List[str]

@dataclass
class ConsistencyViolation:
    """Record of consistency violation between traces"""
    trace1_id: int
    trace2_id: int
    violation_type: str
    severity: float
    description: str
    grid_diff: Optional[torch.Tensor] = None

class ConsistencyEnforcer:
    """Ensure different valid traces agree on solutions and reasoning"""
    
    def __init__(self, device='cpu', consistency_threshold=0.95):
        self.device = device
        self.consistency_threshold = consistency_threshold
        self.violation_history = []
        
        # Consistency weights
        self.weights = {
            'grid': 0.4,
            'probability': 0.2, 
            'operation': 0.2,
            'reasoning': 0.2
        }
        
        # Tolerances for different types of consistency
        self.tolerances = {
            'grid_l1': 0.01,
            'grid_pixel': 0.95,  # 95% pixel agreement required
            'prob_kl': 0.1,
            'operation_jaccard': 0.7,
            'reasoning_semantic': 0.8
        }
        
        self.logger = logging.getLogger(__name__)
        
    def compute_consistency_loss(self, trace1, trace2, relmem_vectors1=None, relmem_vectors2=None) -> Tuple[torch.Tensor, ConsistencyMetrics]:
        """Compute consistency loss between two valid traces with RelMem awareness"""
        
        # Grid consistency
        grid_loss, grid_consistency = self._compute_grid_consistency(trace1, trace2)
        
        # Probability consistency  
        prob_loss, prob_consistency = self._compute_probability_consistency(trace1, trace2)
        
        # Operation consistency
        op_loss, op_consistency = self._compute_operation_consistency(trace1, trace2)
        
        # Reasoning consistency
        reason_loss, reason_consistency = self._compute_reasoning_consistency(trace1, trace2)
        
        # ✅ NEW: RelMem consistency (relational structure should be similar for equivalent traces)
        relmem_loss, relmem_consistency = self._compute_relmem_consistency(relmem_vectors1, relmem_vectors2)
        
        # Updated weights to include RelMem
        weights = {
            'grid': 0.35,
            'probability': 0.15, 
            'operation': 0.2,
            'reasoning': 0.15,
            'relmem': 0.15  # New weight for relational consistency
        }
        
        # Weighted total loss
        total_loss = (
            weights['grid'] * grid_loss +
            weights['probability'] * prob_loss +
            weights['operation'] * op_loss +
            weights['reasoning'] * reason_loss +
            weights['relmem'] * relmem_loss
        )
        
        # Overall consistency score
        overall_consistency = (
            weights['grid'] * grid_consistency +
            weights['probability'] * prob_consistency +
            weights['operation'] * op_consistency +
            weights['reasoning'] * reason_consistency +
            weights['relmem'] * relmem_consistency
        )
        
        # Identify inconsistency sources
        inconsistency_sources = []
        if grid_consistency < self.tolerances['grid_pixel']:
            inconsistency_sources.append('grid_mismatch')
        if prob_consistency < 1.0 - self.tolerances['prob_kl']:
            inconsistency_sources.append('probability_divergence')
        if op_consistency < self.tolerances['operation_jaccard']:
            inconsistency_sources.append('operation_difference')
        if reason_consistency < self.tolerances['reasoning_semantic']:
            inconsistency_sources.append('reasoning_mismatch')
        if relmem_consistency < 0.8:  # RelMem consistency threshold
            inconsistency_sources.append('relational_mismatch')
        
        # Extended metrics with RelMem
        metrics = ConsistencyMetrics(
            grid_consistency=grid_consistency,
            probability_consistency=prob_consistency,
            operation_consistency=op_consistency,
            reasoning_consistency=reason_consistency,
            overall_consistency=overall_consistency,
            inconsistency_sources=inconsistency_sources
        )
        
        # Add RelMem consistency to metrics (extend dataclass dynamically)
        metrics.relmem_consistency = relmem_consistency
        
        return total_loss, metrics
    
    def _compute_grid_consistency(self, trace1, trace2) -> Tuple[torch.Tensor, float]:
        """Ensure final grids are identical for valid traces"""
        
        if trace1.final_grid is None or trace2.final_grid is None:
            return torch.tensor(10.0, requires_grad=True), 0.0
        
        g1, g2 = trace1.final_grid, trace2.final_grid
        
        # Shape consistency
        if g1.shape != g2.shape:
            return torch.tensor(5.0, requires_grad=True), 0.0
        
        # L1 loss for gradients
        l1_loss = F.l1_loss(g1.float(), g2.float())
        
        # Pixel-wise exact match
        exact_matches = (g1 == g2).float()
        pixel_consistency = exact_matches.mean().item()
        
        # Structural consistency (connected components, symmetries)
        structural_loss = self._compute_structural_consistency(g1, g2)
        
        total_loss = l1_loss + structural_loss
        
        return total_loss, pixel_consistency
    
    def _compute_probability_consistency(self, trace1, trace2) -> Tuple[torch.Tensor, float]:
        """Ensure probability distributions are similar"""
        
        if trace1.probabilities is None or trace2.probabilities is None:
            return torch.tensor(0.0, requires_grad=True), 1.0
        
        if len(trace1.probabilities) == 0 or len(trace2.probabilities) == 0:
            return torch.tensor(0.0, requires_grad=True), 1.0
        
        p1, p2 = trace1.probabilities, trace2.probabilities
        
        # Align lengths by padding or truncating
        max_len = max(len(p1), len(p2))
        if len(p1) < max_len:
            p1 = F.pad(p1, (0, max_len - len(p1)))
        if len(p2) < max_len:
            p2 = F.pad(p2, (0, max_len - len(p2)))
            
        # KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(p1, dim=-1), 
            F.softmax(p2, dim=-1), 
            reduction='batchmean'
        )
        
        # Jensen-Shannon divergence for symmetry
        js_loss = self._jensen_shannon_divergence(p1, p2)
        
        total_loss = kl_loss + js_loss
        
        # Consistency score (inverse of KL divergence)
        consistency_score = max(0.0, 1.0 - kl_loss.item())
        
        return total_loss, consistency_score
    
    def _compute_operation_consistency(self, trace1, trace2) -> Tuple[torch.Tensor, float]:
        """Ensure operation sequences have semantic similarity"""
        
        ops1 = trace1.operations if trace1.operations else []
        ops2 = trace2.operations if trace2.operations else []
        
        if not ops1 and not ops2:
            return torch.tensor(0.0, requires_grad=True), 1.0
        
        # Jaccard similarity for operation sets
        set1, set2 = set(ops1), set(ops2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Sequence alignment similarity
        seq_similarity = self._compute_sequence_similarity(ops1, ops2)
        
        # Operation semantic similarity
        semantic_similarity = self._compute_operation_semantic_similarity(ops1, ops2)
        
        # Combined similarity
        overall_similarity = 0.4 * jaccard + 0.3 * seq_similarity + 0.3 * semantic_similarity
        
        # Loss is inverse of similarity
        loss = torch.tensor(1.0 - overall_similarity, requires_grad=True)
        
        return loss, overall_similarity
    
    def _compute_reasoning_consistency(self, trace1, trace2) -> Tuple[torch.Tensor, float]:
        """Ensure reasoning steps are semantically consistent"""
        
        steps1 = trace1.reasoning_steps if trace1.reasoning_steps else []
        steps2 = trace2.reasoning_steps if trace2.reasoning_steps else []
        
        if not steps1 and not steps2:
            return torch.tensor(0.0, requires_grad=True), 1.0
        
        # Semantic similarity between reasoning chains
        semantic_sim = self._compute_reasoning_semantic_similarity(steps1, steps2)
        
        # Logical consistency check
        logical_consistency = self._check_logical_consistency(steps1, steps2)
        
        # Goal alignment
        goal_alignment = self._check_goal_alignment(steps1, steps2)
        
        overall_consistency = 0.5 * semantic_sim + 0.3 * logical_consistency + 0.2 * goal_alignment
        
        loss = torch.tensor(1.0 - overall_consistency, requires_grad=True)
        
        return loss, overall_consistency
    
    def _compute_relmem_consistency(self, relmem_vectors1, relmem_vectors2) -> Tuple[torch.Tensor, float]:
        """Compute RelMem consistency - similar tasks should have similar relational structure"""
        
        if relmem_vectors1 is None or relmem_vectors2 is None:
            return torch.tensor(0.0, requires_grad=True), 1.0
        
        try:
            # Convert to tensors if needed
            if not isinstance(relmem_vectors1, torch.Tensor):
                relmem_vectors1 = torch.tensor(relmem_vectors1, dtype=torch.float32)
            if not isinstance(relmem_vectors2, torch.Tensor):
                relmem_vectors2 = torch.tensor(relmem_vectors2, dtype=torch.float32)
            
            # Ensure same shape
            if relmem_vectors1.shape != relmem_vectors2.shape:
                min_dim = min(relmem_vectors1.size(-1), relmem_vectors2.size(-1))
                relmem_vectors1 = relmem_vectors1[..., :min_dim]
                relmem_vectors2 = relmem_vectors2[..., :min_dim]
            
            # Cosine similarity for relational structure
            cos_sim = F.cosine_similarity(
                relmem_vectors1.flatten(), 
                relmem_vectors2.flatten(), 
                dim=0
            ).clamp(-1, 1)
            
            # Convert to consistency score [0, 1]
            consistency_score = (cos_sim + 1) / 2
            
            # L2 loss for training
            l2_loss = F.mse_loss(relmem_vectors1, relmem_vectors2)
            
            return l2_loss, consistency_score.item()
            
        except Exception as e:
            self.logger.warning(f"RelMem consistency computation failed: {e}")
            return torch.tensor(0.0, requires_grad=True), 1.0
    
    def _compute_structural_consistency(self, grid1: torch.Tensor, grid2: torch.Tensor) -> torch.Tensor:
        """Compute structural similarity between grids"""
        
        # Connected components
        cc1 = self._count_connected_components(grid1)
        cc2 = self._count_connected_components(grid2)
        cc_loss = abs(cc1 - cc2) * 0.1
        
        # Color distribution
        def color_dist(grid):
            dist = torch.zeros(10)
            for i in range(10):
                dist[i] = (grid == i).sum().float()
            return dist / (dist.sum() + 1e-8)
        
        dist1, dist2 = color_dist(grid1), color_dist(grid2)
        color_loss = F.l1_loss(dist1, dist2)
        
        return torch.tensor(cc_loss) + color_loss
    
    def _jensen_shannon_divergence(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Compute Jensen-Shannon divergence"""
        
        p1_soft = F.softmax(p1, dim=-1)
        p2_soft = F.softmax(p2, dim=-1)
        m = 0.5 * (p1_soft + p2_soft)
        
        kl1 = F.kl_div(F.log_softmax(p1, dim=-1), m, reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(p2, dim=-1), m, reduction='batchmean')
        
        return 0.5 * (kl1 + kl2)
    
    def _compute_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Compute similarity between operation sequences"""
        
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0
        
        # Longest common subsequence
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs_len = lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        
        return lcs_len / max_len if max_len > 0 else 0.0
    
    def _compute_operation_semantic_similarity(self, ops1: List[str], ops2: List[str]) -> float:
        """Compute semantic similarity between operations"""
        
        # Define operation categories and similarities
        operation_categories = {
            'spatial': ['rotate', 'flip', 'transpose', 'mirror'],
            'color': ['color_map', 'fill', 'replace_color'],
            'structural': ['extract', 'overlay', 'connect'],
            'logical': ['and', 'or', 'xor', 'not'],
            'pattern': ['repeat', 'tile', 'extend']
        }
        
        def categorize_ops(ops):
            categories = defaultdict(int)
            for op in ops:
                for cat, cat_ops in operation_categories.items():
                    if any(cat_op in op.lower() for cat_op in cat_ops):
                        categories[cat] += 1
                        break
            return categories
        
        cat1 = categorize_ops(ops1)
        cat2 = categorize_ops(ops2)
        
        # Compute category overlap
        all_cats = set(cat1.keys()) | set(cat2.keys())
        if not all_cats:
            return 1.0
        
        overlap = 0.0
        for cat in all_cats:
            count1, count2 = cat1.get(cat, 0), cat2.get(cat, 0)
            overlap += min(count1, count2) / max(count1, count2, 1)
        
        return overlap / len(all_cats)
    
    def _compute_reasoning_semantic_similarity(self, steps1: List[str], steps2: List[str]) -> float:
        """Compute semantic similarity between reasoning steps"""
        
        if not steps1 and not steps2:
            return 1.0
        if not steps1 or not steps2:
            return 0.0
        
        # Simple keyword-based similarity
        def extract_keywords(steps):
            keywords = set()
            for step in steps:
                # Extract key concepts
                words = step.lower().split()
                keywords.update(w for w in words if len(w) > 3)
            return keywords
        
        kw1 = extract_keywords(steps1)
        kw2 = extract_keywords(steps2)
        
        if not kw1 and not kw2:
            return 1.0
        
        intersection = len(kw1 & kw2)
        union = len(kw1 | kw2)
        
        return intersection / union if union > 0 else 0.0
    
    def _check_logical_consistency(self, steps1: List[str], steps2: List[str]) -> float:
        """Check if reasoning steps are logically consistent"""
        
        # Simple implementation: check for contradictions
        # In practice, would need more sophisticated logical analysis
        
        def has_contradiction(steps):
            # Look for explicit contradictions
            text = ' '.join(steps).lower()
            contradictions = [
                ('increase', 'decrease'),
                ('add', 'remove'),
                ('connect', 'separate'),
                ('same', 'different')
            ]
            
            for pos, neg in contradictions:
                if pos in text and neg in text:
                    return True
            return False
        
        consistent1 = not has_contradiction(steps1)
        consistent2 = not has_contradiction(steps2)
        
        # Both should be internally consistent
        return float(consistent1 and consistent2)
    
    def _check_goal_alignment(self, steps1: List[str], steps2: List[str]) -> float:
        """Check if reasoning steps aim toward the same goal"""
        
        # Extract goal-related keywords
        goal_keywords = ['output', 'result', 'final', 'target', 'goal', 'solve']
        
        def extract_goals(steps):
            goals = []
            for step in steps:
                if any(kw in step.lower() for kw in goal_keywords):
                    goals.append(step)
            return goals
        
        goals1 = extract_goals(steps1)
        goals2 = extract_goals(steps2)
        
        if not goals1 and not goals2:
            return 1.0
        if not goals1 or not goals2:
            return 0.5
        
        # Simple similarity based on common words
        all_words1 = set(' '.join(goals1).lower().split())
        all_words2 = set(' '.join(goals2).lower().split())
        
        intersection = len(all_words1 & all_words2)
        union = len(all_words1 | all_words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _count_connected_components(self, grid: torch.Tensor) -> int:
        """Count connected components in grid (simplified)"""
        
        # Simple flood-fill counting
        visited = torch.zeros_like(grid, dtype=torch.bool)
        components = 0
        
        h, w = grid.shape[-2:]
        
        for i in range(h):
            for j in range(w):
                if grid[i, j] != 0 and not visited[i, j]:
                    components += 1
                    # Flood fill (simplified)
                    stack = [(i, j)]
                    while stack:
                        ci, cj = stack.pop()
                        if ci < 0 or ci >= h or cj < 0 or cj >= w:
                            continue
                        if visited[ci, cj] or grid[ci, cj] == 0:
                            continue
                        
                        visited[ci, cj] = True
                        
                        # Add neighbors
                        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            stack.append((ci + di, cj + dj))
        
        return components
    
    def enforce_consistency(self, traces_batch: List, task) -> Dict:
        """Apply consistency enforcement across all valid traces"""
        
        if len(traces_batch) < 2:
            return {'consistency_loss': 0.0, 'violations': [], 'metrics': None}
        
        total_loss = torch.tensor(0.0, requires_grad=True)
        violations = []
        all_metrics = []
        
        # Pairwise consistency enforcement
        for i, trace1 in enumerate(traces_batch):
            for j, trace2 in enumerate(traces_batch[i+1:], i+1):
                
                loss, metrics = self.compute_consistency_loss(trace1, trace2)
                total_loss = total_loss + loss
                all_metrics.append(metrics)
                
                # Record violations
                if metrics.overall_consistency < self.consistency_threshold:
                    violation = ConsistencyViolation(
                        trace1_id=i,
                        trace2_id=j,
                        violation_type='overall_inconsistency',
                        severity=1.0 - metrics.overall_consistency,
                        description=f"Traces {i} and {j} inconsistent: {', '.join(metrics.inconsistency_sources)}",
                        grid_diff=self._compute_grid_diff(trace1, trace2) if trace1.final_grid is not None and trace2.final_grid is not None else None
                    )
                    violations.append(violation)
        
        # Normalize by number of pairs
        n_pairs = len(traces_batch) * (len(traces_batch) - 1) // 2
        if n_pairs > 0:
            total_loss = total_loss / n_pairs
        
        # Aggregate metrics
        avg_metrics = self._aggregate_consistency_metrics(all_metrics) if all_metrics else None
        
        # Store violations
        self.violation_history.extend(violations)
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-500:]
        
        self.logger.info(f"Consistency enforcement: {len(violations)} violations from {n_pairs} pairs")
        
        return {
            'consistency_loss': total_loss.item(),
            'violations': violations,
            'metrics': avg_metrics,
            'n_pairs': n_pairs
        }
    
    def _compute_grid_diff(self, trace1, trace2) -> torch.Tensor:
        """Compute visual difference between grids"""
        
        if trace1.final_grid is None or trace2.final_grid is None:
            return torch.tensor([])
        
        g1, g2 = trace1.final_grid, trace2.final_grid
        
        if g1.shape != g2.shape:
            return torch.tensor([])
        
        # Difference mask
        diff = (g1 != g2).float()
        return diff
    
    def _aggregate_consistency_metrics(self, metrics_list: List[ConsistencyMetrics]) -> ConsistencyMetrics:
        """Aggregate consistency metrics across all pairs"""
        
        if not metrics_list:
            return ConsistencyMetrics(0, 0, 0, 0, 0, [])
        
        avg_grid = np.mean([m.grid_consistency for m in metrics_list])
        avg_prob = np.mean([m.probability_consistency for m in metrics_list])
        avg_op = np.mean([m.operation_consistency for m in metrics_list])
        avg_reason = np.mean([m.reasoning_consistency for m in metrics_list])
        avg_overall = np.mean([m.overall_consistency for m in metrics_list])
        
        # Collect all inconsistency sources
        all_sources = []
        for m in metrics_list:
            all_sources.extend(m.inconsistency_sources)
        
        unique_sources = list(set(all_sources))
        
        return ConsistencyMetrics(
            grid_consistency=avg_grid,
            probability_consistency=avg_prob,
            operation_consistency=avg_op,
            reasoning_consistency=avg_reason,
            overall_consistency=avg_overall,
            inconsistency_sources=unique_sources
        )
    
    def get_consistency_statistics(self) -> Dict:
        """Get statistics about consistency violations"""
        
        if not self.violation_history:
            return {'total_violations': 0}
        
        stats = {
            'total_violations': len(self.violation_history),
            'violation_types': defaultdict(int),
            'avg_severity': np.mean([v.severity for v in self.violation_history]),
            'max_severity': max([v.severity for v in self.violation_history]),
            'recent_violations': len([v for v in self.violation_history[-100:]]) if len(self.violation_history) >= 100 else len(self.violation_history)
        }
        
        for violation in self.violation_history:
            stats['violation_types'][violation.violation_type] += 1
        
        return stats
    
    def should_enforce_consistency(self, metrics: ConsistencyMetrics) -> bool:
        """Determine if consistency enforcement is needed"""
        
        return (
            metrics.overall_consistency < self.consistency_threshold or
            len(metrics.inconsistency_sources) > 0 or
            metrics.grid_consistency < self.tolerances['grid_pixel']
        )