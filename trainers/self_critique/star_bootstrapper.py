"""
Self-Taught Reasoner (STaR) Bootstrapper for TOPAS ARC Solver - Phase 3
Implements diverse trace generation and self-reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import copy
from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class Trace:
    """Execution trace with program and intermediate states"""
    program: List[str]
    grid_states: List[torch.Tensor]
    probabilities: torch.Tensor
    operations: List[str]
    confidence: float
    reasoning_steps: List[str]
    final_grid: torch.Tensor
    execution_time: float
    
@dataclass
class TraceValidation:
    """Validation result for a trace"""
    is_valid: bool
    exact_match: bool
    constraint_violations: List[str]
    similarity_score: float
    explanation: str

class STaRBootstrapper:
    """Self-Taught Reasoner implementation for diverse trace generation"""
    
    def __init__(self, model, device='cpu', trace_diversity_target=0.7):
        self.model = model
        self.device = device
        self.trace_diversity_target = trace_diversity_target
        self.valid_trace_buffer = []
        self.trace_statistics = defaultdict(list)
        
        # STaR parameters
        self.temperature_range = (0.3, 1.2)
        self.max_program_length = 50
        self.diversity_weight = 0.3
        self.confidence_threshold = 0.6
        
        self.logger = logging.getLogger(__name__)
        
    def generate_diverse_traces(self, task, n_traces: int = 16) -> List[Trace]:
        """Generate N diverse solution attempts for a task"""
        
        traces = []
        temperatures = np.linspace(self.temperature_range[0], self.temperature_range[1], n_traces)
        
        # Track diversity metrics
        program_hashes = set()
        operation_sequences = set()
        
        for i in range(n_traces):
            temp = temperatures[i]
            
            # Generate trace with varied sampling
            trace = self._generate_single_trace(task, temperature=temp, attempt_id=i)
            
            if trace:
                # Check for diversity
                prog_hash = hash(tuple(trace.program))
                ops_hash = hash(tuple(trace.operations))
                
                # Add diversity bonus for novel approaches
                if prog_hash not in program_hashes or ops_hash not in operation_sequences:
                    trace.confidence += self.diversity_weight * (1.0 - len(program_hashes) / n_traces)
                
                program_hashes.add(prog_hash)
                operation_sequences.add(ops_hash)
                traces.append(trace)
                
        # Sort by confidence and diversity
        traces = self._rank_traces_by_quality(traces, task)
        
        self.logger.info(f"Generated {len(traces)} traces with diversity {len(program_hashes)/max(1,len(traces)):.3f}")
        
        return traces
    
    def _generate_single_trace(self, task, temperature: float = 0.8, attempt_id: int = 0) -> Optional[Trace]:
        """Generate a single trace with specified temperature"""
        
        try:
            # Set model to sampling mode
            original_training = self.model.training
            self.model.eval()
            
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            with torch.no_grad():
                # Forward pass with temperature sampling
                if hasattr(self.model, 'sample_trace'):
                    trace_data = self.model.sample_trace(task, temperature=temperature, max_length=self.max_program_length)
                else:
                    # Fallback: use standard forward pass with temperature
                    trace_data = self._manual_trace_generation(task, temperature)
                
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                exec_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                exec_time = 0.0
            
            # Restore model training state
            self.model.train(original_training)
            
            # Create trace object
            trace = Trace(
                program=trace_data.get('program', []),
                grid_states=trace_data.get('grid_states', []),
                probabilities=trace_data.get('probabilities', torch.tensor([])),
                operations=trace_data.get('operations', []),
                confidence=trace_data.get('confidence', 0.0),
                reasoning_steps=trace_data.get('reasoning_steps', []),
                final_grid=trace_data.get('final_grid', torch.zeros_like(task.input)),
                execution_time=exec_time
            )
            
            return trace
            
        except Exception as e:
            self.logger.warning(f"Failed to generate trace {attempt_id}: {str(e)}")
            return None
    
    def _manual_trace_generation(self, task, temperature: float) -> Dict:
        """Manual trace generation when model doesn't have sample_trace method"""
        
        # This is a simplified implementation - in practice would need model-specific code
        trace_data = {
            'program': [],
            'grid_states': [task.input.clone()],
            'probabilities': torch.tensor([]),
            'operations': [],
            'confidence': 0.5,
            'reasoning_steps': [],
            'final_grid': task.input.clone()
        }
        
        # Use model's forward pass
        try:
            if hasattr(self.model, 'forward_with_trace'):
                output = self.model.forward_with_trace(task.input)
                trace_data.update(output)
            else:
                output = self.model(task.input)
                trace_data['final_grid'] = output
                trace_data['confidence'] = 0.5
                
        except Exception as e:
            self.logger.warning(f"Manual trace generation failed: {e}")
            
        return trace_data
    
    def verify_traces(self, traces: List[Trace], task) -> List[TraceValidation]:
        """Verify which traces are valid and provide detailed feedback"""
        
        validations = []
        
        for i, trace in enumerate(traces):
            validation = self._verify_single_trace(trace, task)
            validations.append(validation)
            
            # Log validation results
            if validation.is_valid:
                self.logger.debug(f"Trace {i}: VALID ({'exact' if validation.exact_match else 'approximate'})")
            else:
                self.logger.debug(f"Trace {i}: INVALID - {validation.explanation}")
                
        return validations
    
    def _verify_single_trace(self, trace: Trace, task) -> TraceValidation:
        """Verify a single trace against task constraints"""
        
        validation = TraceValidation(
            is_valid=False,
            exact_match=False,
            constraint_violations=[],
            similarity_score=0.0,
            explanation=""
        )
        
        try:
            # Check if trace has valid output
            if trace.final_grid is None:
                validation.explanation = "No final grid produced"
                return validation
            
            # Check exact match
            if torch.equal(trace.final_grid, task.output):
                validation.exact_match = True
                validation.is_valid = True
                validation.similarity_score = 1.0
                validation.explanation = "Exact match"
                return validation
            
            # Check constraints
            violations = self._check_constraints(trace, task)
            validation.constraint_violations = violations
            
            # Compute similarity
            similarity = self._compute_similarity(trace.final_grid, task.output)
            validation.similarity_score = similarity
            
            # Determine validity based on constraints and similarity
            if len(violations) == 0 and similarity > 0.9:
                validation.is_valid = True
                validation.explanation = f"High similarity ({similarity:.3f}) with no constraint violations"
            elif len(violations) == 0:
                validation.explanation = f"No violations but low similarity ({similarity:.3f})"
            else:
                validation.explanation = f"Constraint violations: {', '.join(violations)}"
                
        except Exception as e:
            validation.explanation = f"Verification error: {str(e)}"
            
        return validation
    
    def _check_constraints(self, trace: Trace, task) -> List[str]:
        """Check various constraints on the trace"""
        
        violations = []
        
        try:
            # Shape constraints
            if trace.final_grid.shape != task.output.shape:
                violations.append(f"Shape mismatch: got {trace.final_grid.shape}, expected {task.output.shape}")
            
            # Value range constraints
            min_val, max_val = trace.final_grid.min(), trace.final_grid.max()
            expected_min, expected_max = task.output.min(), task.output.max()
            
            if min_val < 0 or max_val > 9:
                violations.append(f"Invalid color range: [{min_val}, {max_val}]")
            
            # Color usage constraints (for some tasks)
            trace_colors = set(torch.unique(trace.final_grid).tolist())
            expected_colors = set(torch.unique(task.output).tolist())
            
            if hasattr(task, 'constraints') and 'preserve_colors' in task.constraints:
                if trace_colors != expected_colors:
                    violations.append(f"Color set mismatch: got {trace_colors}, expected {expected_colors}")
            
            # Grid sparsity constraints
            trace_nonzero = (trace.final_grid != 0).sum().item()
            expected_nonzero = (task.output != 0).sum().item()
            
            if abs(trace_nonzero - expected_nonzero) > max(3, expected_nonzero * 0.2):
                violations.append(f"Sparsity mismatch: got {trace_nonzero}, expected ~{expected_nonzero}")
                
        except Exception as e:
            violations.append(f"Constraint check error: {str(e)}")
            
        return violations
    
    def _compute_similarity(self, output1: torch.Tensor, output2: torch.Tensor) -> float:
        """Compute similarity between two grids"""
        
        try:
            if output1.shape != output2.shape:
                return 0.0
                
            # Exact pixel match
            exact_matches = (output1 == output2).float().mean().item()
            
            # Color distribution similarity
            def color_hist(grid):
                hist = torch.zeros(10)
                for i in range(10):
                    hist[i] = (grid == i).sum().float()
                return hist / hist.sum()
            
            hist1 = color_hist(output1)
            hist2 = color_hist(output2)
            hist_similarity = 1.0 - 0.5 * torch.sum(torch.abs(hist1 - hist2)).item()
            
            # Pattern similarity (structural)
            def get_structure(grid):
                # Simple structural features
                return {
                    'nonzero_count': (grid != 0).sum().item(),
                    'unique_colors': len(torch.unique(grid)),
                    'center_of_mass': torch.nonzero(grid != 0).float().mean(dim=0).tolist() if (grid != 0).any() else [0, 0]
                }
            
            struct1 = get_structure(output1)
            struct2 = get_structure(output2)
            
            # Combine similarities
            final_similarity = 0.6 * exact_matches + 0.2 * hist_similarity + 0.2 * (
                1.0 - abs(struct1['nonzero_count'] - struct2['nonzero_count']) / max(struct1['nonzero_count'], struct2['nonzero_count'], 1)
            )
            
            return max(0.0, min(1.0, final_similarity))
            
        except Exception:
            return 0.0
    
    def reinforce_valid_traces(self, valid_traces: List[Trace], task) -> Dict:
        """Train on successful traces to increase their probability"""
        
        if not valid_traces:
            return {'reinforced': 0, 'loss': 0.0}
        
        self.model.train()
        
        total_loss = 0.0
        reinforced_count = 0
        
        for trace in valid_traces:
            try:
                # Supervised learning on program sequence
                if trace.program and hasattr(self.model, 'program_head'):
                    prog_loss = self._compute_program_loss(trace, task)
                    prog_loss.backward(retain_graph=True)
                    total_loss += prog_loss.item()
                
                # Supervised learning on final grid
                if trace.final_grid is not None:
                    grid_loss = self._compute_grid_loss(trace, task)
                    grid_loss.backward()
                    total_loss += grid_loss.item()
                
                # Increase trace probability
                if trace.probabilities is not None and len(trace.probabilities) > 0:
                    prob_loss = -torch.mean(torch.log(trace.probabilities + 1e-8))
                    prob_loss.backward()
                    total_loss += prob_loss.item()
                
                reinforced_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to reinforce trace: {str(e)}")
                continue
        
        # Store successful traces for future reference
        self.valid_trace_buffer.extend(valid_traces[-10:])  # Keep only recent ones
        if len(self.valid_trace_buffer) > 100:
            self.valid_trace_buffer = self.valid_trace_buffer[-50:]
        
        avg_loss = total_loss / max(1, reinforced_count)
        
        self.logger.info(f"Reinforced {reinforced_count}/{len(valid_traces)} traces, avg loss: {avg_loss:.4f}")
        
        return {
            'reinforced': reinforced_count,
            'loss': avg_loss,
            'buffer_size': len(self.valid_trace_buffer)
        }
    
    def _compute_program_loss(self, trace: Trace, task) -> torch.Tensor:
        """Compute loss for program sequence prediction"""
        # Placeholder - would need model-specific implementation
        return torch.tensor(0.0, requires_grad=True)
    
    def _compute_grid_loss(self, trace: Trace, task) -> torch.Tensor:
        """Compute loss for grid prediction"""
        if trace.final_grid is None:
            return torch.tensor(0.0, requires_grad=True)
            
        # L1 loss for exact pixel matching
        l1_loss = F.l1_loss(trace.final_grid.float(), task.output.float())
        
        # Cross-entropy loss treating as classification
        ce_loss = F.cross_entropy(
            trace.final_grid.view(1, -1), 
            task.output.view(-1).long(),
            ignore_index=-1
        ) if trace.final_grid.numel() == task.output.numel() else torch.tensor(0.0)
        
        return l1_loss + 0.5 * ce_loss
    
    def _rank_traces_by_quality(self, traces: List[Trace], task) -> List[Trace]:
        """Rank traces by quality metrics"""
        
        def quality_score(trace):
            score = trace.confidence
            
            # Bonus for reasonable program length
            if trace.program:
                length_penalty = max(0, len(trace.program) - 20) * 0.01
                score -= length_penalty
            
            # Bonus for diverse operations
            if trace.operations:
                diversity_bonus = len(set(trace.operations)) / max(len(trace.operations), 1) * 0.1
                score += diversity_bonus
            
            # Bonus for fast execution
            if trace.execution_time > 0:
                time_bonus = max(0, 2.0 - trace.execution_time) * 0.05
                score += time_bonus
            
            return score
        
        traces.sort(key=quality_score, reverse=True)
        return traces
    
    def analyze_trace_patterns(self, traces: List[Trace], validations: List[TraceValidation]) -> Dict:
        """Analyze patterns in successful vs failed traces"""
        
        analysis = {
            'total_traces': len(traces),
            'valid_traces': sum(1 for v in validations if v.is_valid),
            'exact_matches': sum(1 for v in validations if v.exact_match),
            'avg_similarity': np.mean([v.similarity_score for v in validations]),
            'common_operations': defaultdict(int),
            'success_patterns': defaultdict(list),
            'failure_patterns': defaultdict(list)
        }
        
        # Analyze successful traces
        for trace, validation in zip(traces, validations):
            # Count operations
            for op in trace.operations:
                analysis['common_operations'][op] += 1
            
            if validation.is_valid:
                analysis['success_patterns']['program_lengths'].append(len(trace.program))
                analysis['success_patterns']['confidences'].append(trace.confidence)
                analysis['success_patterns']['execution_times'].append(trace.execution_time)
            else:
                analysis['failure_patterns']['program_lengths'].append(len(trace.program))
                analysis['failure_patterns']['confidences'].append(trace.confidence)
                analysis['failure_patterns']['execution_times'].append(trace.execution_time)
        
        # Compute statistics
        for pattern_type in ['success_patterns', 'failure_patterns']:
            for metric, values in analysis[pattern_type].items():
                if values:
                    analysis[f'{pattern_type}_{metric}_mean'] = np.mean(values)
                    analysis[f'{pattern_type}_{metric}_std'] = np.std(values)
        
        return analysis
    
    def get_trace_diversity_metrics(self, traces: List[Trace]) -> Dict:
        """Compute diversity metrics for generated traces"""
        
        metrics = {
            'program_diversity': 0.0,
            'operation_diversity': 0.0,
            'confidence_spread': 0.0,
            'unique_programs': 0,
            'unique_operations': 0
        }
        
        if not traces:
            return metrics
        
        # Program diversity
        program_hashes = set()
        for trace in traces:
            if trace.program:
                program_hashes.add(hash(tuple(trace.program)))
        
        metrics['unique_programs'] = len(program_hashes)
        metrics['program_diversity'] = len(program_hashes) / len(traces)
        
        # Operation diversity
        operation_seqs = set()
        for trace in traces:
            if trace.operations:
                operation_seqs.add(tuple(trace.operations))
        
        metrics['unique_operations'] = len(operation_seqs)
        metrics['operation_diversity'] = len(operation_seqs) / len(traces)
        
        # Confidence spread
        confidences = [trace.confidence for trace in traces]
        metrics['confidence_spread'] = np.std(confidences) if confidences else 0.0
        
        return metrics