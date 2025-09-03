#!/usr/bin/env python3
"""
Curriculum Manager
Curriculum shaping with adaptive difficulty progression.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import random
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class TaskMetadata:
    """Metadata about an ARC task for curriculum learning"""
    task_id: str
    program_length: int = 0
    object_count: int = 0
    symmetry_depth: float = 0.0
    size_delta: float = 0.0  # abs(H_out*W_out - H_in*W_in)
    color_entropy: float = 0.0
    theme: str = "unknown"
    transformation_type: str = "unknown"
    success_history: List[bool] = field(default_factory=list)
    last_attempt_epoch: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate from history"""
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)
    
    @property
    def difficulty_score(self) -> float:
        """Compute multi-factor difficulty score"""
        score = 0.0
        score += self.program_length * 0.5  # Program complexity
        score += self.object_count * 0.3    # Object complexity
        score += self.symmetry_depth * 0.2  # Symmetry complexity
        score += abs(self.size_delta) * 0.1  # Size change complexity
        score += self.color_entropy * 0.1   # Color complexity
        return score


class DifficultyBucket:
    """Container for tasks of similar difficulty"""
    
    def __init__(self, name: str, min_difficulty: float, max_difficulty: float):
        self.name = name
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.tasks: List[TaskMetadata] = []
        self.sample_weights: List[float] = []
        
    def add_task(self, task: TaskMetadata):
        """Add task to bucket"""
        if self.min_difficulty <= task.difficulty_score <= self.max_difficulty:
            self.tasks.append(task)
            # Initial weight based on inverse success rate (focus on harder tasks)
            initial_weight = max(0.1, 1.0 - task.success_rate)
            self.sample_weights.append(initial_weight)
    
    def remove_task(self, task_id: str):
        """Remove task from bucket"""
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                del self.tasks[i]
                del self.sample_weights[i]
                break
    
    def update_weights(self):
        """Update sampling weights based on recent performance"""
        for i, task in enumerate(self.tasks):
            # Focus on tasks with low success rates
            base_weight = max(0.1, 1.0 - task.success_rate)
            
            # Boost recently failed tasks
            recent_failures = sum(1 for success in task.success_history[-5:] if not success)
            failure_boost = 1.0 + 0.2 * recent_failures
            
            # Reduce weight for consistently successful tasks
            if len(task.success_history) >= 5 and task.success_rate > 0.8:
                success_penalty = 0.5
            else:
                success_penalty = 1.0
            
            self.sample_weights[i] = base_weight * failure_boost * success_penalty
    
    def sample_tasks(self, n: int) -> List[TaskMetadata]:
        """Sample n tasks from this bucket"""
        if not self.tasks:
            return []
        
        n = min(n, len(self.tasks))
        
        # Update weights before sampling
        self.update_weights()
        
        # Weighted sampling without replacement
        if sum(self.sample_weights) == 0:
            # Uniform sampling fallback
            return random.sample(self.tasks, n)
        
        # Normalize weights
        total_weight = sum(self.sample_weights)
        normalized_weights = [w / total_weight for w in self.sample_weights]
        
        # Sample with weights
        sampled_indices = np.random.choice(
            len(self.tasks), 
            size=n, 
            replace=False,
            p=normalized_weights
        )
        
        return [self.tasks[i] for i in sampled_indices]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get bucket statistics"""
        if not self.tasks:
            return {"num_tasks": 0}
        
        success_rates = [task.success_rate for task in self.tasks]
        difficulties = [task.difficulty_score for task in self.tasks]
        
        return {
            "num_tasks": len(self.tasks),
            "avg_success_rate": np.mean(success_rates),
            "avg_difficulty": np.mean(difficulties),
            "min_difficulty": min(difficulties),
            "max_difficulty": max(difficulties)
        }


class AdversarialGenerator:
    """Generates adversarial examples based on model failures"""
    
    def __init__(self, max_mutations: int = 5):
        self.max_mutations = max_mutations
        self.failure_patterns = defaultdict(list)
        
    def analyze_failure(self, 
                       task: TaskMetadata, 
                       predicted_program: List[str],
                       correct_program: List[str]):
        """Analyze why a task failed"""
        failure_type = self.classify_failure(predicted_program, correct_program)
        self.failure_patterns[failure_type].append({
            'task': task,
            'predicted': predicted_program,
            'correct': correct_program
        })
    
    def classify_failure(self, predicted: List[str], correct: List[str]) -> str:
        """Classify type of failure"""
        if len(predicted) != len(correct):
            return "length_mismatch"
        
        # Check operation type errors
        pred_ops = [op.split('(')[0] for op in predicted]
        correct_ops = [op.split('(')[0] for op in correct]
        
        if pred_ops != correct_ops:
            return "operation_error"
        
        # Check parameter errors
        return "parameter_error"
    
    def generate_adversarial_tasks(self, 
                                 base_tasks: List[TaskMetadata],
                                 model,
                                 n_adversarial: int = 100) -> List[TaskMetadata]:
        """
        Generate adversarial tasks by minimal mutation until model fails.
        
        Args:
            base_tasks: Tasks to mutate
            model: Model to test against
            n_adversarial: Number of adversarial examples to generate
            
        Returns:
            List of adversarial task variants
        """
        adversarial_tasks = []
        
        for _ in range(n_adversarial):
            if not base_tasks:
                break
                
            # Select base task
            base_task = random.choice(base_tasks)
            
            # Try mutations
            for mutation_count in range(1, self.max_mutations + 1):
                mutated_task = self.mutate_task(base_task, mutation_count)
                
                # Test if model fails on mutated task
                if self.test_model_failure(mutated_task, model):
                    mutated_task.task_id = f"{base_task.task_id}_adv_{len(adversarial_tasks)}"
                    adversarial_tasks.append(mutated_task)
                    break
        
        logger.info(f"Generated {len(adversarial_tasks)} adversarial tasks")
        return adversarial_tasks
    
    def mutate_task(self, task: TaskMetadata, mutation_count: int) -> TaskMetadata:
        """Apply minimal mutations to create task variant"""
        mutated = TaskMetadata(
            task_id=f"{task.task_id}_mut",
            program_length=task.program_length,
            object_count=task.object_count,
            symmetry_depth=task.symmetry_depth,
            size_delta=task.size_delta,
            color_entropy=task.color_entropy,
            theme=task.theme,
            transformation_type=task.transformation_type
        )
        
        # Apply random mutations
        for _ in range(mutation_count):
            mutation_type = random.choice([
                'change_colors', 'add_noise', 'change_size', 
                'rotate', 'flip', 'translate'
            ])
            
            # Update task properties based on mutation
            if mutation_type == 'change_colors':
                mutated.color_entropy *= 1.1
            elif mutation_type == 'change_size':
                mutated.size_delta *= 1.2
            elif mutation_type in ['rotate', 'flip']:
                mutated.symmetry_depth *= 1.1
            
        return mutated
    
    def test_model_failure(self, task: TaskMetadata, model) -> bool:
        """Test if model fails on given task"""
        # This would normally run the model on the task
        # For now, return probabilistic failure based on difficulty
        failure_probability = min(0.8, task.difficulty_score / 10.0)
        return random.random() < failure_probability


class CurriculumManager:
    """Main curriculum learning manager"""
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 save_path: Optional[str] = None):
        self.config = config or {}
        self.save_path = save_path
        
        # Initialize difficulty buckets
        self.buckets = {
            'easy': DifficultyBucket('easy', 0.0, 2.0),
            'medium': DifficultyBucket('medium', 2.0, 4.0),
            'hard': DifficultyBucket('hard', 4.0, 6.0),
            'extreme': DifficultyBucket('extreme', 6.0, float('inf'))
        }
        
        # Adversarial generator
        self.adversarial_gen = AdversarialGenerator()
        
        # Curriculum progression parameters
        self.progression_schedule = {
            'early': {'easy': 0.7, 'medium': 0.2, 'hard': 0.1, 'extreme': 0.0},
            'middle': {'easy': 0.25, 'medium': 0.35, 'hard': 0.3, 'extreme': 0.1},
            'late': {'easy': 0.1, 'medium': 0.2, 'hard': 0.4, 'extreme': 0.3}
        }
        
        # Statistics
        self.epoch_stats = []
        
        # Load previous state if available
        self.load_state()
    
    def add_task(self, task: TaskMetadata):
        """Add task to appropriate difficulty bucket"""
        difficulty = task.difficulty_score
        
        for bucket in self.buckets.values():
            if bucket.min_difficulty <= difficulty <= bucket.max_difficulty:
                bucket.add_task(task)
                break
    
    def compute_difficulty(self, 
                         task_data: Dict[str, Any],
                         program: Optional[List[str]] = None) -> TaskMetadata:
        """
        Compute comprehensive difficulty score for a task.
        
        Args:
            task_data: Raw task data containing grids and metadata
            program: Optional program sequence if known
            
        Returns:
            TaskMetadata with computed difficulty
        """
        task_id = task_data.get('task_id', 'unknown')
        
        # Extract features from task data
        input_grid = task_data.get('input', [])
        output_grid = task_data.get('output', [])
        
        # Compute features
        program_length = len(program) if program else self.estimate_program_length(task_data)
        object_count = self.count_objects(input_grid)
        symmetry_depth = self.compute_symmetry_depth(input_grid, output_grid)
        size_delta = abs(len(output_grid) * len(output_grid[0]) - len(input_grid) * len(input_grid[0])) if input_grid and output_grid else 0
        color_entropy = self.compute_color_entropy(input_grid, output_grid)
        theme = self.classify_theme(task_data)
        transformation_type = self.classify_transformation(input_grid, output_grid)
        
        return TaskMetadata(
            task_id=task_id,
            program_length=program_length,
            object_count=object_count,
            symmetry_depth=symmetry_depth,
            size_delta=float(size_delta),
            color_entropy=color_entropy,
            theme=theme,
            transformation_type=transformation_type
        )
    
    def estimate_program_length(self, task_data: Dict) -> int:
        """Estimate program length from task complexity"""
        input_grid = task_data.get('input', [])
        output_grid = task_data.get('output', [])
        
        if not input_grid or not output_grid:
            return 1
        
        # Heuristic based on grid differences
        differences = 0
        min_h = min(len(input_grid), len(output_grid))
        min_w = min(len(input_grid[0]) if input_grid else 0, len(output_grid[0]) if output_grid else 0)
        
        for i in range(min_h):
            for j in range(min_w):
                if input_grid[i][j] != output_grid[i][j]:
                    differences += 1
        
        # Simple heuristic: more differences = longer program
        return max(1, differences // 10 + 1)
    
    def count_objects(self, grid: List[List[int]]) -> int:
        """Count distinct objects in grid using connected components"""
        if not grid:
            return 0
        
        H, W = len(grid), len(grid[0])
        visited = [[False] * W for _ in range(H)]
        object_count = 0
        
        def dfs(i, j, color):
            if (i < 0 or i >= H or j < 0 or j >= W or 
                visited[i][j] or grid[i][j] != color or color == 0):
                return
            visited[i][j] = True
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                dfs(i + di, j + dj, color)
        
        for i in range(H):
            for j in range(W):
                if not visited[i][j] and grid[i][j] != 0:
                    object_count += 1
                    dfs(i, j, grid[i][j])
        
        return object_count
    
    def compute_symmetry_depth(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> float:
        """Compute symmetry complexity"""
        symmetry_score = 0.0
        
        for grid in [input_grid, output_grid]:
            if not grid:
                continue
                
            H, W = len(grid), len(grid[0])
            
            # Check horizontal symmetry
            h_symmetric = True
            for i in range(H):
                for j in range(W // 2):
                    if grid[i][j] != grid[i][W - 1 - j]:
                        h_symmetric = False
                        break
                if not h_symmetric:
                    break
            
            # Check vertical symmetry  
            v_symmetric = True
            for i in range(H // 2):
                for j in range(W):
                    if grid[i][j] != grid[H - 1 - i][j]:
                        v_symmetric = False
                        break
                if not v_symmetric:
                    break
            
            # Check rotational symmetry
            r_symmetric = True
            if H == W:  # Only check for square grids
                for i in range(H):
                    for j in range(W):
                        if grid[i][j] != grid[W - 1 - j][i]:
                            r_symmetric = False
                            break
                    if not r_symmetric:
                        break
            else:
                r_symmetric = False
            
            # Accumulate symmetry score
            symmetry_score += (0.3 if h_symmetric else 0) + \
                            (0.3 if v_symmetric else 0) + \
                            (0.4 if r_symmetric else 0)
        
        return symmetry_score / 2  # Average over input and output
    
    def compute_color_entropy(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> float:
        """Compute color entropy as measure of complexity"""
        all_colors = []
        
        for grid in [input_grid, output_grid]:
            if not grid:
                continue
            for row in grid:
                all_colors.extend(row)
        
        if not all_colors:
            return 0.0
        
        # Color frequency
        from collections import Counter
        color_counts = Counter(all_colors)
        total_cells = len(all_colors)
        
        # Shannon entropy
        entropy = 0.0
        for count in color_counts.values():
            p = count / total_cells
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def classify_theme(self, task_data: Dict) -> str:
        """Classify visual theme of task"""
        # Placeholder theme classification
        themes = ['symmetry', 'counting', 'pattern_completion', 'object_detection', 
                 'color_mapping', 'spatial_transformation']
        return random.choice(themes)
    
    def classify_transformation(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> str:
        """Classify transformation type"""
        if not input_grid or not output_grid:
            return "unknown"
        
        input_h, input_w = len(input_grid), len(input_grid[0])
        output_h, output_w = len(output_grid), len(output_grid[0])
        
        if input_h == output_h and input_w == output_w:
            return "same_size"
        elif output_h * output_w > input_h * input_w:
            return "expansion"
        else:
            return "contraction"
    
    def sample_batch(self, 
                    batch_size: int,
                    epoch: int,
                    max_epochs: int) -> List[TaskMetadata]:
        """
        Sample a curriculum-aware batch of tasks.
        
        Args:
            batch_size: Number of tasks to sample
            epoch: Current training epoch
            max_epochs: Total training epochs
            
        Returns:
            List of sampled tasks
        """
        progress = epoch / max_epochs
        
        # Determine curriculum phase
        if progress < 0.3:
            phase = 'early'
        elif progress < 0.7:
            phase = 'middle'
        else:
            phase = 'late'
        
        # Get sampling weights for this phase
        weights = self.progression_schedule[phase]
        
        # Sample from each bucket according to weights
        batch = []
        for bucket_name, weight in weights.items():
            n_samples = int(batch_size * weight)
            if n_samples > 0:
                bucket_samples = self.buckets[bucket_name].sample_tasks(n_samples)
                batch.extend(bucket_samples)
        
        # Fill remaining slots randomly
        while len(batch) < batch_size:
            # Sample from non-empty buckets
            non_empty = [name for name, bucket in self.buckets.items() if bucket.tasks]
            if not non_empty:
                break
            
            bucket_name = random.choice(non_empty)
            extra_samples = self.buckets[bucket_name].sample_tasks(1)
            batch.extend(extra_samples)
        
        # Record statistics
        self.record_epoch_stats(epoch, batch, phase)
        
        return batch[:batch_size]
    
    def update_task_result(self, 
                          task: TaskMetadata,
                          success: bool,
                          epoch: int,
                          predicted_program: Optional[List[str]] = None,
                          correct_program: Optional[List[str]] = None):
        """Update task with training result"""
        task.success_history.append(success)
        task.last_attempt_epoch = epoch
        
        # Keep only recent history to avoid memory bloat
        if len(task.success_history) > 50:
            task.success_history = task.success_history[-50:]
        
        # Analyze failures for adversarial generation
        if not success and predicted_program and correct_program:
            self.adversarial_gen.analyze_failure(task, predicted_program, correct_program)
    
    def generate_adversarial_batch(self, 
                                 model,
                                 n_adversarial: int = 50) -> List[TaskMetadata]:
        """Generate adversarial examples and add to extreme bucket"""
        # Collect base tasks from all buckets
        base_tasks = []
        for bucket in self.buckets.values():
            base_tasks.extend(bucket.tasks)
        
        if not base_tasks:
            return []
        
        # Generate adversarial examples
        adversarial_tasks = self.adversarial_gen.generate_adversarial_tasks(
            base_tasks, model, n_adversarial
        )
        
        # Add to extreme bucket
        for task in adversarial_tasks:
            self.buckets['extreme'].add_task(task)
        
        return adversarial_tasks
    
    def record_epoch_stats(self, epoch: int, batch: List[TaskMetadata], phase: str):
        """Record statistics for this epoch"""
        stats = {
            'epoch': epoch,
            'phase': phase,
            'batch_size': len(batch),
            'difficulty_distribution': {},
            'theme_distribution': {}
        }
        
        # Compute distributions
        for bucket_name, bucket in self.buckets.items():
            bucket_count = sum(1 for task in batch if bucket.min_difficulty <= task.difficulty_score <= bucket.max_difficulty)
            stats['difficulty_distribution'][bucket_name] = bucket_count
        
        theme_counts = defaultdict(int)
        for task in batch:
            theme_counts[task.theme] += 1
        stats['theme_distribution'] = dict(theme_counts)
        
        self.epoch_stats.append(stats)
    
    def get_curriculum_statistics(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics"""
        stats = {
            'bucket_stats': {},
            'total_tasks': 0,
            'adversarial_patterns': len(self.adversarial_gen.failure_patterns)
        }
        
        for name, bucket in self.buckets.items():
            bucket_stats = bucket.get_statistics()
            stats['bucket_stats'][name] = bucket_stats
            stats['total_tasks'] += bucket_stats.get('num_tasks', 0)
        
        # Recent epoch statistics
        if self.epoch_stats:
            recent_stats = self.epoch_stats[-10:]  # Last 10 epochs
            stats['recent_phase_distribution'] = {}
            for stat in recent_stats:
                phase = stat['phase']
                stats['recent_phase_distribution'][phase] = stats['recent_phase_distribution'].get(phase, 0) + 1
        
        return stats
    
    def save_state(self):
        """Save curriculum state to disk"""
        if not self.save_path:
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Serialize curriculum state
        state = {
            'buckets': {},
            'epoch_stats': self.epoch_stats,
            'failure_patterns': dict(self.adversarial_gen.failure_patterns)
        }
        
        # Save bucket tasks (simplified)
        for name, bucket in self.buckets.items():
            state['buckets'][name] = {
                'tasks': [
                    {
                        'task_id': task.task_id,
                        'difficulty_score': task.difficulty_score,
                        'success_rate': task.success_rate,
                        'theme': task.theme
                    }
                    for task in bucket.tasks
                ]
            }
        
        with open(self.save_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved curriculum state to {self.save_path}")
    
    def load_state(self):
        """Load curriculum state from disk"""
        if not self.save_path or not os.path.exists(self.save_path):
            return
        
        try:
            with open(self.save_path, 'r') as f:
                state = json.load(f)
            
            self.epoch_stats = state.get('epoch_stats', [])
            
            # Load failure patterns
            failure_patterns = state.get('failure_patterns', {})
            for pattern_type, failures in failure_patterns.items():
                self.adversarial_gen.failure_patterns[pattern_type] = failures
            
            logger.info(f"Loaded curriculum state from {self.save_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load curriculum state: {e}")