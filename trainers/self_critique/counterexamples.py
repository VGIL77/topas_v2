"""
Counterexample Generator for TOPAS ARC Solver - Phase 3
Generates minimal failure cases to teach self-correction
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Task:
    """ARC task representation"""
    input: torch.Tensor
    output: torch.Tensor
    constraints: Dict
    metadata: Dict

@dataclass
class Counterexample:
    """Generated counterexample with perturbation info"""
    task: Task
    perturbation_type: str
    perturbation_strength: float
    expected_behavior: str
    should_succeed: bool

class CounterexampleGenerator:
    """Generate minimal failure cases for self-critique learning"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.perturbation_functions = {
            'color_swap': self._swap_colors,
            'mirror_h': self._mirror_horizontal,
            'mirror_v': self._mirror_vertical,
            'rotate_90': self._rotate_90,
            'resize_expand': self._resize_expand,
            'resize_shrink': self._resize_shrink,
            'object_permute': self._permute_objects,
            'noise_add': self._add_noise,
            'pattern_shift': self._shift_pattern,
            'symmetry_break': self._break_symmetry
        }
        
    def generate_from_failure(self, failed_task: Task, model, n_counterexamples: int = 8) -> List[Counterexample]:
        """Generate counterexamples from a failed task attempt"""
        counterexamples = []
        
        # Test permutation invariance
        if self._has_color_symmetry(failed_task):
            counter = self._generate_color_swap_counter(failed_task)
            if counter:
                counterexamples.append(counter)
                
        # Test spatial invariance  
        if self._has_spatial_symmetry(failed_task):
            for transform in ['mirror_h', 'mirror_v', 'rotate_90']:
                counter = self._generate_spatial_counter(failed_task, transform)
                if counter:
                    counterexamples.append(counter)
                    
        # Test size robustness
        if self._can_resize(failed_task):
            for resize_type in ['resize_expand', 'resize_shrink']:
                counter = self._generate_resize_counter(failed_task, resize_type)
                if counter:
                    counterexamples.append(counter)
                    
        # Test object permutations
        if self._has_multiple_objects(failed_task):
            counter = self._generate_permutation_counter(failed_task)
            if counter:
                counterexamples.append(counter)
                
        # Test pattern robustness
        counter = self._generate_pattern_shift_counter(failed_task)
        if counter:
            counterexamples.append(counter)
            
        return counterexamples[:n_counterexamples]
    
    def find_minimal_perturbation(self, success_task: Task, model, max_iterations: int = 20) -> Optional[Counterexample]:
        """Find smallest change that breaks the model via binary search"""
        
        # Try different perturbation types
        for pert_type in self.perturbation_functions.keys():
            minimal = self._binary_search_perturbation(success_task, model, pert_type, max_iterations)
            if minimal:
                return minimal
                
        return None
    
    def _binary_search_perturbation(self, task: Task, model, pert_type: str, max_iter: int) -> Optional[Counterexample]:
        """Binary search for minimal perturbation strength"""
        
        low_strength = 0.0
        high_strength = 1.0
        minimal_failing = None
        
        for iteration in range(max_iter):
            mid_strength = (low_strength + high_strength) / 2.0
            
            # Generate perturbed task
            perturbed_task = self._apply_perturbation(task, pert_type, mid_strength)
            if not perturbed_task:
                high_strength = mid_strength
                continue
                
            # Test if model fails on this perturbation
            try:
                trace = model.forward(perturbed_task)
                is_correct = self._verify_solution(trace, perturbed_task)
                
                if is_correct:
                    # Need stronger perturbation
                    low_strength = mid_strength
                else:
                    # Found failing case, try weaker
                    high_strength = mid_strength
                    minimal_failing = Counterexample(
                        task=perturbed_task,
                        perturbation_type=pert_type,
                        perturbation_strength=mid_strength,
                        expected_behavior="should succeed with invariance",
                        should_succeed=True
                    )
                    
            except Exception as e:
                # Model crashed, this is a failing case
                high_strength = mid_strength
                minimal_failing = Counterexample(
                    task=perturbed_task,
                    perturbation_type=pert_type,
                    perturbation_strength=mid_strength,
                    expected_behavior="should not crash",
                    should_succeed=True
                )
                
            # Convergence check
            if abs(high_strength - low_strength) < 0.01:
                break
                
        return minimal_failing
    
    def _apply_perturbation(self, task: Task, pert_type: str, strength: float) -> Optional[Task]:
        """Apply perturbation of given type and strength"""
        try:
            pert_func = self.perturbation_functions[pert_type]
            return pert_func(task, strength)
        except Exception:
            return None
    
    def _swap_colors(self, task: Task, strength: float = 1.0) -> Task:
        """Swap colors randomly (should preserve solution)"""
        new_task = copy.deepcopy(task)
        
        # Get unique colors
        input_colors = torch.unique(task.input).tolist()
        output_colors = torch.unique(task.output).tolist()
        all_colors = list(set(input_colors + output_colors))
        
        if len(all_colors) < 2:
            return new_task
            
        # Create random permutation
        np.random.shuffle(all_colors)
        color_map = {old: new for old, new in zip(torch.unique(torch.cat([task.input.flatten(), task.output.flatten()])).tolist(), all_colors)}
        
        # Apply color mapping
        new_input = task.input.clone()
        new_output = task.output.clone()
        
        for old_color, new_color in color_map.items():
            new_input[task.input == old_color] = new_color
            new_output[task.output == old_color] = new_color
            
        new_task.input = new_input
        new_task.output = new_output
        new_task.metadata['perturbation'] = 'color_swap'
        
        return new_task
    
    def _mirror_horizontal(self, task: Task, strength: float = 1.0) -> Task:
        """Mirror horizontally (should preserve solution for symmetric patterns)"""
        new_task = copy.deepcopy(task)
        new_task.input = torch.flip(task.input, dims=[-1])
        new_task.output = torch.flip(task.output, dims=[-1])
        new_task.metadata['perturbation'] = 'mirror_h'
        return new_task
    
    def _mirror_vertical(self, task: Task, strength: float = 1.0) -> Task:
        """Mirror vertically"""
        new_task = copy.deepcopy(task)
        new_task.input = torch.flip(task.input, dims=[-2])
        new_task.output = torch.flip(task.output, dims=[-2])
        new_task.metadata['perturbation'] = 'mirror_v'
        return new_task
    
    def _rotate_90(self, task: Task, strength: float = 1.0) -> Task:
        """Rotate 90 degrees"""
        new_task = copy.deepcopy(task)
        new_task.input = torch.rot90(task.input, k=1, dims=[-2, -1])
        new_task.output = torch.rot90(task.output, k=1, dims=[-2, -1])
        new_task.metadata['perturbation'] = 'rotate_90'
        return new_task
    
    def _resize_expand(self, task: Task, strength: float = 0.5) -> Task:
        """Expand grid size by adding padding"""
        new_task = copy.deepcopy(task)
        
        pad_size = max(1, int(strength * 3))  # 1-3 pixels padding
        
        # Pad with zeros (background)
        new_task.input = torch.nn.functional.pad(task.input, (pad_size, pad_size, pad_size, pad_size), value=0)
        new_task.output = torch.nn.functional.pad(task.output, (pad_size, pad_size, pad_size, pad_size), value=0)
        new_task.metadata['perturbation'] = 'resize_expand'
        
        return new_task
    
    def _resize_shrink(self, task: Task, strength: float = 0.5) -> Task:
        """Shrink grid by cropping (if possible)"""
        new_task = copy.deepcopy(task)
        
        h, w = task.input.shape[-2:]
        crop_size = max(1, int(strength * min(h, w) * 0.2))  # Crop up to 20%
        
        if h > crop_size * 2 and w > crop_size * 2:
            new_task.input = task.input[..., crop_size:-crop_size, crop_size:-crop_size]
            new_task.output = task.output[..., crop_size:-crop_size, crop_size:-crop_size]
            new_task.metadata['perturbation'] = 'resize_shrink'
            
        return new_task
    
    def _permute_objects(self, task: Task, strength: float = 1.0) -> Task:
        """Permute object positions (should preserve solution)"""
        new_task = copy.deepcopy(task)
        
        # Simple implementation: randomly shuffle non-zero regions
        # This is a placeholder - more sophisticated object detection needed
        new_task.metadata['perturbation'] = 'object_permute'
        return new_task
    
    def _add_noise(self, task: Task, strength: float = 0.1) -> Task:
        """Add small amount of noise"""
        new_task = copy.deepcopy(task)
        
        # Add noise to a few random pixels
        h, w = task.input.shape[-2:]
        n_noise = max(1, int(strength * h * w * 0.05))  # Up to 5% of pixels
        
        for _ in range(n_noise):
            i, j = np.random.randint(0, h), np.random.randint(0, w)
            if np.random.random() < 0.5:  # Only change some
                new_color = np.random.randint(0, 10)
                new_task.input[i, j] = new_color
                
        new_task.metadata['perturbation'] = 'noise_add'
        return new_task
    
    def _shift_pattern(self, task: Task, strength: float = 0.5) -> Task:
        """Shift pattern by small offset"""
        new_task = copy.deepcopy(task)
        
        max_shift = max(1, int(strength * 3))
        shift_h = np.random.randint(-max_shift, max_shift + 1)
        shift_w = np.random.randint(-max_shift, max_shift + 1)
        
        # Use roll to shift
        new_task.input = torch.roll(task.input, shifts=(shift_h, shift_w), dims=(-2, -1))
        new_task.output = torch.roll(task.output, shifts=(shift_h, shift_w), dims=(-2, -1))
        new_task.metadata['perturbation'] = 'pattern_shift'
        
        return new_task
    
    def _break_symmetry(self, task: Task, strength: float = 0.3) -> Task:
        """Slightly break symmetry to test robustness"""
        new_task = copy.deepcopy(task)
        
        # Add asymmetric noise
        h, w = task.input.shape[-2:]
        n_breaks = max(1, int(strength * h * w * 0.02))
        
        for _ in range(n_breaks):
            i, j = np.random.randint(0, h), np.random.randint(0, w)
            # Only break in one half
            if j < w // 2:
                new_task.input[i, j] = (task.input[i, j] + 1) % 10
                
        new_task.metadata['perturbation'] = 'symmetry_break'
        return new_task
    
    # Helper methods for analyzing task properties
    def _has_color_symmetry(self, task: Task) -> bool:
        """Check if task likely has color permutation invariance"""
        # Simple heuristic: multiple colors present
        return len(torch.unique(task.input)) > 2
    
    def _has_spatial_symmetry(self, task: Task) -> bool:
        """Check if task likely has spatial invariance"""
        # Simple heuristic: roughly square and not too small
        h, w = task.input.shape[-2:]
        return abs(h - w) <= 2 and min(h, w) >= 4
    
    def _can_resize(self, task: Task) -> bool:
        """Check if task can handle resizing"""
        h, w = task.input.shape[-2:]
        return min(h, w) >= 6  # Large enough to resize
    
    def _has_multiple_objects(self, task: Task) -> bool:
        """Check if task has multiple distinct objects"""
        # Rough heuristic: multiple connected components
        unique_colors = len(torch.unique(task.input))
        return unique_colors > 3
    
    def _verify_solution(self, trace, task: Task) -> bool:
        """Verify if trace solves the task correctly"""
        try:
            if hasattr(trace, 'final_grid'):
                return torch.equal(trace.final_grid, task.output)
            elif hasattr(trace, 'output'):
                return torch.equal(trace.output, task.output)
            else:
                return False
        except Exception:
            return False
    
    # Generator methods for specific counter types
    def _generate_color_swap_counter(self, task: Task) -> Optional[Counterexample]:
        """Generate color swap counterexample"""
        try:
            swapped_task = self._swap_colors(task)
            return Counterexample(
                task=swapped_task,
                perturbation_type='color_swap',
                perturbation_strength=1.0,
                expected_behavior='should succeed with color invariance',
                should_succeed=True
            )
        except Exception:
            return None
    
    def _generate_spatial_counter(self, task: Task, transform: str) -> Optional[Counterexample]:
        """Generate spatial transformation counterexample"""
        try:
            transform_func = self.perturbation_functions[transform]
            transformed_task = transform_func(task)
            return Counterexample(
                task=transformed_task,
                perturbation_type=transform,
                perturbation_strength=1.0,
                expected_behavior='should succeed with spatial invariance',
                should_succeed=True
            )
        except Exception:
            return None
    
    def _generate_resize_counter(self, task: Task, resize_type: str) -> Optional[Counterexample]:
        """Generate resize counterexample"""
        try:
            resize_func = self.perturbation_functions[resize_type]
            resized_task = resize_func(task)
            return Counterexample(
                task=resized_task,
                perturbation_type=resize_type,
                perturbation_strength=0.5,
                expected_behavior='should succeed with size robustness',
                should_succeed=True
            )
        except Exception:
            return None
    
    def _generate_permutation_counter(self, task: Task) -> Optional[Counterexample]:
        """Generate object permutation counterexample"""
        try:
            permuted_task = self._permute_objects(task)
            return Counterexample(
                task=permuted_task,
                perturbation_type='object_permute',
                perturbation_strength=1.0,
                expected_behavior='should succeed with object order invariance',
                should_succeed=True
            )
        except Exception:
            return None
    
    def _generate_pattern_shift_counter(self, task: Task) -> Optional[Counterexample]:
        """Generate pattern shift counterexample"""
        try:
            shifted_task = self._shift_pattern(task)
            return Counterexample(
                task=shifted_task,
                perturbation_type='pattern_shift',
                perturbation_strength=0.5,
                expected_behavior='should succeed with minor position shifts',
                should_succeed=True
            )
        except Exception:
            return None

    def analyze_counterexample_results(self, counterexamples: List[Counterexample], model_results: List[bool]) -> Dict:
        """Analyze what types of counterexamples the model fails on"""
        analysis = {
            'total_generated': len(counterexamples),
            'failure_by_type': defaultdict(int),
            'success_by_type': defaultdict(int),
            'insights': []
        }
        
        for counter, success in zip(counterexamples, model_results):
            if success:
                analysis['success_by_type'][counter.perturbation_type] += 1
            else:
                analysis['failure_by_type'][counter.perturbation_type] += 1
        
        # Generate insights
        for pert_type, failures in analysis['failure_by_type'].items():
            total = failures + analysis['success_by_type'][pert_type]
            if total > 0:
                failure_rate = failures / total
                if failure_rate > 0.5:
                    analysis['insights'].append(f"Model struggles with {pert_type} invariance ({failure_rate:.2%} failure rate)")
        
        return analysis