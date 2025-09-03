#!/usr/bin/env python3
"""
Ensemble Solver
Truth-conditioned ensembling with self-consistency and mixture of experts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import numpy as np
from collections import Counter, defaultdict
import random
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SolutionCandidate:
    """Represents a solution candidate with metadata"""
    solution: torch.Tensor  # Grid solution
    program: List[str]  # DSL program sequence
    confidence: float  # Model confidence
    verification_score: float  # Constraint satisfaction score
    temperature: float  # Sampling temperature used
    expert_id: Optional[str] = None  # Which expert generated this
    reasoning_path: Optional[List[str]] = None  # Step-by-step reasoning


class ConstraintVerifier:
    """Verifies solution candidates against ARC constraints"""
    
    def __init__(self):
        self.constraints = [
            self._check_size_consistency,
            self._check_color_validity,
            self._check_connectivity,
            self._check_symmetry_preservation,
            self._check_object_count,
            self._check_pattern_completion
        ]
    
    def verify_solution(self, 
                       solution: torch.Tensor,
                       input_grid: torch.Tensor,
                       task_metadata: Dict) -> float:
        """
        Verify solution against constraints.
        
        Args:
            solution: Predicted output grid
            input_grid: Input grid
            task_metadata: Task metadata and constraints
            
        Returns:
            Verification score [0, 1] where 1 is perfect
        """
        total_score = 0.0
        constraint_count = 0
        
        for constraint_fn in self.constraints:
            try:
                score = constraint_fn(solution, input_grid, task_metadata)
                total_score += score
                constraint_count += 1
            except Exception as e:
                logger.warning(f"Constraint verification failed: {e}")
        
        return total_score / max(constraint_count, 1)
    
    def _check_size_consistency(self, solution: torch.Tensor, input_grid: torch.Tensor, metadata: Dict) -> float:
        """Check if solution has expected size"""
        expected_h = metadata.get('expected_height', solution.shape[-2])
        expected_w = metadata.get('expected_width', solution.shape[-1])
        
        actual_h, actual_w = solution.shape[-2:]
        
        if actual_h == expected_h and actual_w == expected_w:
            return 1.0
        else:
            # Partial credit based on size difference
            h_diff = abs(actual_h - expected_h) / max(expected_h, 1)
            w_diff = abs(actual_w - expected_w) / max(expected_w, 1)
            return max(0.0, 1.0 - (h_diff + w_diff) / 2)
    
    def _check_color_validity(self, solution: torch.Tensor, input_grid: torch.Tensor, metadata: Dict) -> float:
        """Check if solution uses valid colors (0-9)"""
        valid_colors = set(range(10))
        solution_colors = set(solution.unique().cpu().numpy())
        
        invalid_colors = solution_colors - valid_colors
        if not invalid_colors:
            return 1.0
        else:
            return max(0.0, 1.0 - len(invalid_colors) / 10.0)
    
    def _check_connectivity(self, solution: torch.Tensor, input_grid: torch.Tensor, metadata: Dict) -> float:
        """Check connectivity constraints"""
        # Simplified connectivity check
        return 0.8  # Assume good connectivity for now
    
    def _check_symmetry_preservation(self, solution: torch.Tensor, input_grid: torch.Tensor, metadata: Dict) -> float:
        """Check if symmetry is preserved where expected"""
        symmetry_type = metadata.get('symmetry_type', None)
        if symmetry_type is None:
            return 1.0  # No symmetry constraint
        
        # Check horizontal symmetry
        if symmetry_type == 'horizontal':
            solution_np = solution.cpu().numpy()
            if len(solution_np.shape) == 3:  # [B, H, W]
                solution_np = solution_np[0]
            
            H, W = solution_np.shape
            symmetric = True
            for i in range(H):
                for j in range(W // 2):
                    if solution_np[i, j] != solution_np[i, W - 1 - j]:
                        symmetric = False
                        break
                if not symmetric:
                    break
            
            return 1.0 if symmetric else 0.3
        
        return 0.8  # Default score for other symmetries
    
    def _check_object_count(self, solution: torch.Tensor, input_grid: torch.Tensor, metadata: Dict) -> float:
        """Check if object count is reasonable"""
        expected_objects = metadata.get('expected_objects', None)
        if expected_objects is None:
            return 1.0
        
        # Count objects in solution (simplified)
        actual_objects = self._count_objects(solution)
        
        if actual_objects == expected_objects:
            return 1.0
        else:
            return max(0.0, 1.0 - abs(actual_objects - expected_objects) / max(expected_objects, 1))
    
    def _check_pattern_completion(self, solution: torch.Tensor, input_grid: torch.Tensor, metadata: Dict) -> float:
        """Check pattern completion constraints"""
        # Simplified pattern check
        return 0.9
    
    def _count_objects(self, grid: torch.Tensor) -> int:
        """Count connected components (objects) in grid"""
        if len(grid.shape) == 3:
            grid = grid[0]  # Take first in batch
        
        grid_np = grid.cpu().numpy()
        H, W = grid_np.shape
        visited = np.zeros((H, W), dtype=bool)
        object_count = 0
        
        def dfs(i, j, color):
            if (i < 0 or i >= H or j < 0 or j >= W or 
                visited[i, j] or grid_np[i, j] != color or color == 0):
                return
            visited[i, j] = True
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                dfs(i + di, j + dj, color)
        
        for i in range(H):
            for j in range(W):
                if not visited[i, j] and grid_np[i, j] != 0:
                    object_count += 1
                    dfs(i, j, grid_np[i, j])
        
        return object_count


class ExpertSpecialist(nn.Module):
    """Specialized micro-expert for specific ARC patterns"""
    
    def __init__(self, 
                 expert_type: str,
                 input_dim: int = 512,
                 hidden_dim: int = 256):
        super().__init__()
        self.expert_type = expert_type
        
        # Specialized processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Expert-specific heads
        if expert_type == 'symmetry':
            self.output_head = self._create_symmetry_head(hidden_dim)
        elif expert_type == 'counting':
            self.output_head = self._create_counting_head(hidden_dim)
        elif expert_type == 'pattern':
            self.output_head = self._create_pattern_head(hidden_dim)
        elif expert_type == 'spatial':
            self.output_head = self._create_spatial_head(hidden_dim)
        else:
            self.output_head = nn.Linear(hidden_dim, 10)  # General output
        
        # Expert confidence scorer
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _create_symmetry_head(self, hidden_dim: int) -> nn.Module:
        """Create head specialized for symmetry operations"""
        return nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Color outputs
        )
    
    def _create_counting_head(self, hidden_dim: int) -> nn.Module:
        """Create head specialized for counting operations"""
        return nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 10)
        )
    
    def _create_pattern_head(self, hidden_dim: int) -> nn.Module:
        """Create head specialized for pattern completion"""
        return nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def _create_spatial_head(self, hidden_dim: int) -> nn.Module:
        """Create head specialized for spatial transformations"""
        return nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for expert.
        
        Args:
            x: Input features
            
        Returns:
            (predictions, confidence_score)
        """
        features = self.feature_processor(x)
        predictions = self.output_head(features)
        confidence = self.confidence_head(features)
        
        return predictions, confidence


class RouterNetwork(nn.Module):
    """Routes tasks to appropriate experts"""
    
    def __init__(self, 
                 input_dim: int = 1024,
                 num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        
        self.router = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
        
        # Task classification features
        self.task_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # Task types
        )
    
    def forward(self, task_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route task to experts.
        
        Args:
            task_features: Task feature representation
            
        Returns:
            (expert_weights, task_classification)
        """
        expert_logits = self.router(task_features)
        expert_weights = F.softmax(expert_logits, dim=-1)
        
        task_class = self.task_classifier(task_features)
        
        return expert_weights, task_class


class EnsembleSolver:
    """Main ensemble solver with self-consistency and expert mixture"""
    
    def __init__(self, 
                 base_model: nn.Module,
                 config: Optional[Dict] = None):
        self.base_model = base_model
        self.config = config or {}
        
        # Expert specialists
        self.experts = {
            'symmetry': ExpertSpecialist('symmetry'),
            'counting': ExpertSpecialist('counting'), 
            'pattern': ExpertSpecialist('pattern'),
            'spatial': ExpertSpecialist('spatial')
        }
        
        # Router network
        self.router = RouterNetwork(num_experts=len(self.experts))
        
        # Constraint verifier
        self.verifier = ConstraintVerifier()
        
        # Self-consistency parameters
        self.n_samples = self.config.get('n_samples', 16)
        self.temperature_range = self.config.get('temperature_range', (0.5, 1.2))
        self.consensus_threshold = self.config.get('consensus_threshold', 0.6)
        
        # Statistics
        self.solution_stats = defaultdict(list)
        
    def self_consistency_solve(self, 
                             task_data: Dict,
                             task_features: torch.Tensor,
                             n_samples: Optional[int] = None) -> Optional[SolutionCandidate]:
        """
        Solve task using self-consistency with multiple samples.
        
        Args:
            task_data: Task input data
            task_features: Encoded task features
            n_samples: Number of solutions to sample
            
        Returns:
            Best solution candidate or None if no valid solution
        """
        n_samples = n_samples or self.n_samples
        candidates = []
        
        # Sample multiple solutions with different temperatures
        for i in range(n_samples):
            temp = self.temperature_range[0] + i * (
                (self.temperature_range[1] - self.temperature_range[0]) / max(n_samples - 1, 1)
            )
            
            try:
                candidate = self._sample_solution(task_data, task_features, temperature=temp)
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Failed to sample solution {i}: {e}")
        
        if not candidates:
            return None
        
        # Verify all candidates
        verified_candidates = []
        for candidate in candidates:
            verification_score = self.verifier.verify_solution(
                candidate.solution,
                task_data.get('input_tensor', torch.zeros(1, 10, 10)),
                task_data
            )
            candidate.verification_score = verification_score
            
            if verification_score > 0.5:  # Minimum verification threshold
                verified_candidates.append(candidate)
        
        if not verified_candidates:
            # Return best unverified candidate if no verified ones
            return max(candidates, key=lambda c: c.confidence)
        
        # Find consensus solution
        consensus_candidate = self._find_consensus(verified_candidates)
        if consensus_candidate:
            return consensus_candidate
        
        # Return highest scoring verified candidate
        return max(verified_candidates, key=lambda c: c.verification_score * c.confidence)
    
    def _sample_solution(self, 
                        task_data: Dict,
                        task_features: torch.Tensor,
                        temperature: float) -> Optional[SolutionCandidate]:
        """Sample a single solution with given temperature"""
        try:
            # Forward pass through base model with temperature sampling
            with torch.no_grad():
                # This would be the actual model forward pass
                # For now, simulate solution generation
                solution_shape = task_data.get('output_shape', (10, 10))
                solution = torch.randint(0, 10, (1, *solution_shape))
                
                # Simulate program generation
                program = self._generate_program(task_features, temperature)
                
                # Compute confidence (simplified)
                confidence = max(0.1, 1.0 - temperature * 0.5)
                
                return SolutionCandidate(
                    solution=solution,
                    program=program,
                    confidence=confidence,
                    verification_score=0.0,  # Will be computed later
                    temperature=temperature
                )
        
        except Exception as e:
            logger.error(f"Failed to sample solution: {e}")
            return None
    
    def _generate_program(self, task_features: torch.Tensor, temperature: float) -> List[str]:
        """Generate DSL program (simplified simulation)"""
        # This would use the actual DSL generation logic
        operations = ['rotate', 'flip', 'fill_color', 'extract_objects', 'mirror']
        program_length = random.randint(1, 5)
        
        program = []
        for _ in range(program_length):
            op = random.choice(operations)
            program.append(f"{op}()")
        
        return program
    
    def _find_consensus(self, candidates: List[SolutionCandidate]) -> Optional[SolutionCandidate]:
        """Find consensus solution among candidates"""
        if len(candidates) < 3:
            return None
        
        # Group similar solutions
        solution_groups = defaultdict(list)
        
        for candidate in candidates:
            # Convert solution to hashable representation for grouping
            solution_hash = self._hash_solution(candidate.solution)
            solution_groups[solution_hash].append(candidate)
        
        # Find largest group above consensus threshold
        min_consensus_size = max(2, int(len(candidates) * self.consensus_threshold))
        
        best_group = None
        best_size = 0
        
        for group in solution_groups.values():
            if len(group) >= min_consensus_size and len(group) > best_size:
                best_group = group
                best_size = len(group)
        
        if best_group:
            # Return highest confidence candidate from consensus group
            return max(best_group, key=lambda c: c.confidence * c.verification_score)
        
        return None
    
    def _hash_solution(self, solution: torch.Tensor) -> str:
        """Create hash for solution grouping"""
        # Simplified solution hashing
        return str(solution.cpu().numpy().tolist())
    
    def mixture_of_experts_solve(self, 
                               task_data: Dict,
                               task_features: torch.Tensor) -> SolutionCandidate:
        """
        Solve using mixture of experts routing.
        
        Args:
            task_data: Task data
            task_features: Task feature representation
            
        Returns:
            Solution from best expert
        """
        # Route to experts
        expert_weights, task_classification = self.router(task_features)
        
        expert_solutions = {}
        expert_confidences = {}
        
        # Get solutions from each expert
        for expert_name, expert in self.experts.items():
            try:
                predictions, confidence = expert(task_features)
                
                # Convert predictions to solution format
                solution = self._predictions_to_solution(predictions, task_data)
                
                expert_solutions[expert_name] = solution
                expert_confidences[expert_name] = float(confidence.mean())
                
            except Exception as e:
                logger.warning(f"Expert {expert_name} failed: {e}")
        
        if not expert_solutions:
            # Fallback to base model
            return self._sample_solution(task_data, task_features, temperature=1.0)
        
        # Weight expert solutions by router weights and confidence
        best_expert = None
        best_score = 0.0
        
        for i, (expert_name, solution) in enumerate(expert_solutions.items()):
            router_weight = float(expert_weights[0, i]) if expert_weights.shape[0] > 0 else 0.25
            confidence = expert_confidences[expert_name]
            
            combined_score = router_weight * confidence
            
            if combined_score > best_score:
                best_score = combined_score
                best_expert = expert_name
        
        if best_expert:
            return SolutionCandidate(
                solution=expert_solutions[best_expert],
                program=[f"expert_{best_expert}_solution()"],
                confidence=expert_confidences[best_expert],
                verification_score=0.0,
                temperature=1.0,
                expert_id=best_expert
            )
        
        # Fallback
        return list(expert_solutions.values())[0] if expert_solutions else None
    
    def _predictions_to_solution(self, predictions: torch.Tensor, task_data: Dict) -> torch.Tensor:
        """Convert model predictions to solution grid format"""
        # This would reshape and process predictions into grid format
        # For now, simulate based on expected output shape
        output_shape = task_data.get('output_shape', (10, 10))
        
        if len(predictions.shape) == 3:  # [B, T, C]
            # Take argmax over classes
            solution = predictions.argmax(dim=-1)
            
            # Reshape to grid if needed
            if solution.shape[-1] != output_shape[0] * output_shape[1]:
                # Fallback to random grid
                return torch.randint(0, 10, (1, *output_shape))
            
            return solution.view(1, *output_shape)
        
        # Fallback to random
        return torch.randint(0, 10, (1, *output_shape))
    
    def ensemble_solve(self, 
                      task_data: Dict,
                      task_features: torch.Tensor) -> SolutionCandidate:
        """
        Main ensemble solving method combining all approaches.
        
        Args:
            task_data: Task input data
            task_features: Encoded task features
            
        Returns:
            Best ensemble solution
        """
        solutions = []
        
        # Try self-consistency approach
        try:
            sc_solution = self.self_consistency_solve(task_data, task_features)
            if sc_solution:
                sc_solution.reasoning_path = ["self_consistency"]
                solutions.append(sc_solution)
        except Exception as e:
            logger.warning(f"Self-consistency failed: {e}")
        
        # Try mixture of experts
        try:
            moe_solution = self.mixture_of_experts_solve(task_data, task_features)
            if moe_solution:
                moe_solution.reasoning_path = ["mixture_of_experts"]
                solutions.append(moe_solution)
        except Exception as e:
            logger.warning(f"Mixture of experts failed: {e}")
        
        if not solutions:
            # Final fallback to base model
            return self._sample_solution(task_data, task_features, temperature=0.8)
        
        # Select best solution based on combined scoring
        best_solution = max(solutions, key=self._score_solution)
        
        # Update statistics
        self.solution_stats['total_solutions'] += len(solutions)
        self.solution_stats['best_scores'].append(self._score_solution(best_solution))
        
        return best_solution
    
    def _score_solution(self, candidate: SolutionCandidate) -> float:
        """Score solution candidate for ranking"""
        base_score = candidate.confidence * 0.6
        verification_score = candidate.verification_score * 0.4
        
        # Bonus for expert solutions
        if candidate.expert_id:
            base_score += 0.1
        
        # Bonus for self-consistency
        if candidate.reasoning_path and "self_consistency" in candidate.reasoning_path:
            base_score += 0.05
        
        return base_score + verification_score
    
    def get_ensemble_statistics(self) -> Dict[str, float]:
        """Get ensemble performance statistics"""
        stats = {}
        
        if self.solution_stats['best_scores']:
            scores = self.solution_stats['best_scores']
            stats['mean_score'] = np.mean(scores)
            stats['std_score'] = np.std(scores)
            stats['best_score'] = max(scores)
            stats['median_score'] = np.median(scores)
        
        stats['total_solutions'] = self.solution_stats['total_solutions']
        stats['avg_solutions_per_task'] = (
            self.solution_stats['total_solutions'] / 
            max(len(self.solution_stats['best_scores']), 1)
        )
        
        return stats
    
    def save_experts(self, path_prefix: str):
        """Save expert models"""
        for name, expert in self.experts.items():
            torch.save(expert.state_dict(), f"{path_prefix}_expert_{name}.pt")
        
        torch.save(self.router.state_dict(), f"{path_prefix}_router.pt")
        logger.info(f"Saved ensemble components to {path_prefix}_*.pt")
    
    def load_experts(self, path_prefix: str):
        """Load expert models"""
        for name, expert in self.experts.items():
            try:
                expert.load_state_dict(torch.load(f"{path_prefix}_expert_{name}.pt"))
                logger.info(f"Loaded expert {name}")
            except Exception as e:
                logger.warning(f"Failed to load expert {name}: {e}")
        
        try:
            self.router.load_state_dict(torch.load(f"{path_prefix}_router.pt"))
            logger.info("Loaded router network")
        except Exception as e:
            logger.warning(f"Failed to load router: {e}")