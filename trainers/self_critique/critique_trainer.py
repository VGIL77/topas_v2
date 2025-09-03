"""
Self-Critique Trainer for TOPAS ARC Solver - Phase 3
Orchestrates the complete self-critique training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
import time
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import os

from self_critique.counterexamples import CounterexampleGenerator, Counterexample, Task
from self_critique.star_bootstrapper import STaRBootstrapper, Trace, TraceValidation
from self_critique.consistency import ConsistencyEnforcer, ConsistencyMetrics, ConsistencyViolation

@dataclass
class CritiqueSession:
    """Record of a complete critique session"""
    task_id: str
    original_success: bool
    counterexamples_generated: int
    counterexamples_solved: int
    traces_generated: int
    valid_traces: int
    consistency_violations: int
    learning_updates: int
    session_duration: float
    improvements: Dict[str, float]

@dataclass 
class CritiqueMetrics:
    """Comprehensive metrics for critique training"""
    total_tasks: int
    success_rate: float
    counterexample_success_rate: float
    trace_diversity: float
    consistency_score: float
    learning_efficiency: float
    error_reduction: float
    reasoning_quality: float

class SelfCritiqueTrainer:
    """Orchestrate self-critique loop for ARC problem solving"""
    
    def __init__(self, model, device='cpu', config=None):
        self.model = model
        self.device = device
        
        # Default configuration
        self.config = config or {
            'n_traces_per_task': 16,
            'n_counterexamples': 8,
            'consistency_threshold': 0.95,
            'learning_rate': 1e-4,
            'critique_frequency': 10,  # Every N tasks
            'max_critique_iterations': 3,
            'trace_diversity_target': 0.7,
            'min_improvement_threshold': 0.05
        }
        
        # Initialize components
        self.counter_generator = CounterexampleGenerator(device=device)
        self.star_bootstrapper = STaRBootstrapper(model, device=device)
        self.consistency_enforcer = ConsistencyEnforcer(device=device)
        
        # Tracking
        self.critique_sessions = []
        self.performance_history = deque(maxlen=1000)
        self.learning_curves = defaultdict(list)
        self.error_analysis = defaultdict(int)
        
        # Optimizer for critique-driven learning
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Performance baselines
        self.baseline_metrics = None
        self.current_metrics = None
        
    def critique_loop(self, dataset: List[Task], max_iterations: int = None) -> Dict:
        """Main critique loop over dataset"""
        
        max_iterations = max_iterations or len(dataset)
        critique_results = {
            'sessions': [],
            'overall_metrics': None,
            'improvements': {},
            'error_patterns': {}
        }
        
        # Establish baseline if not set
        if self.baseline_metrics is None:
            self.baseline_metrics = self._evaluate_baseline(dataset[:50])  # Sample for speed
            self.logger.info(f"Baseline metrics: {self.baseline_metrics}")
        
        session_count = 0
        
        for task_idx, task in enumerate(dataset[:max_iterations]):
            
            # Decide whether to run critique on this task
            should_critique = (
                task_idx % self.config['critique_frequency'] == 0 or
                self._should_prioritize_task(task)
            )
            
            if not should_critique:
                continue
                
            session_start = time.time()
            
            self.logger.info(f"Starting critique session {session_count} on task {task_idx}")
            
            # Run critique session
            session = self._run_critique_session(task, task_idx)
            session.session_duration = time.time() - session_start
            
            self.critique_sessions.append(session)
            critique_results['sessions'].append(asdict(session))
            
            # Update learning curves
            self._update_learning_curves(session)
            
            # Check for improvements
            if session_count % 10 == 0:
                current_performance = self._evaluate_performance(dataset[task_idx-10:task_idx+10])
                improvement = self._compute_improvement(current_performance)
                
                if improvement > self.config['min_improvement_threshold']:
                    self.logger.info(f"Significant improvement detected: {improvement:.3f}")
                    critique_results['improvements'][f'session_{session_count}'] = improvement
                
            session_count += 1
            
            # Early stopping if convergence reached
            if self._check_convergence():
                self.logger.info(f"Convergence reached after {session_count} sessions")
                break
        
        # Final evaluation
        self.current_metrics = self._evaluate_performance(dataset)
        critique_results['overall_metrics'] = asdict(self.current_metrics)
        critique_results['error_patterns'] = dict(self.error_analysis)
        
        self.logger.info(f"Critique training completed: {session_count} sessions")
        self.logger.info(f"Final metrics: {self.current_metrics}")
        
        return critique_results
    
    def _run_critique_session(self, task: Task, task_id: int) -> CritiqueSession:
        """Run complete critique session on single task"""
        
        session = CritiqueSession(
            task_id=str(task_id),
            original_success=False,
            counterexamples_generated=0,
            counterexamples_solved=0,
            traces_generated=0,
            valid_traces=0,
            consistency_violations=0,
            learning_updates=0,
            session_duration=0.0,
            improvements={}
        )
        
        # Step 1: Try to solve original task
        try:
            original_trace = self.model.forward(task)
            session.original_success = self._is_correct(original_trace, task)
            
            if not session.original_success:
                # Generate counterexamples for failed task
                counterexamples = self.counter_generator.generate_from_failure(
                    task, self.model, n_counterexamples=self.config['n_counterexamples']
                )
                session.counterexamples_generated = len(counterexamples)
                
                # Try to solve counterexamples
                counter_successes = []
                for counter in counterexamples:
                    try:
                        counter_trace = self.model.forward(counter.task)
                        success = self._is_correct(counter_trace, counter.task)
                        counter_successes.append(success)
                        
                        if success:
                            session.counterexamples_solved += 1
                            # Learn from successful counterexample
                            self._learn_from_success(counter_trace, counter.task, original_trace, task)
                            session.learning_updates += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Counterexample solving failed: {e}")
                        counter_successes.append(False)
                
                # Analyze counterexample patterns
                self._analyze_counterexample_patterns(counterexamples, counter_successes)
        
        except Exception as e:
            self.logger.error(f"Original task solving failed: {e}")
            
        # Step 2: Generate diverse traces
        traces = self.star_bootstrapper.generate_diverse_traces(
            task, n_traces=self.config['n_traces_per_task']
        )
        session.traces_generated = len(traces)
        
        # Step 3: Verify traces
        validations = self.star_bootstrapper.verify_traces(traces, task)
        valid_traces = [trace for trace, val in zip(traces, validations) if val.is_valid]
        session.valid_traces = len(valid_traces)
        
        # Step 4: Reinforce valid traces
        if valid_traces:
            reinforcement_result = self.star_bootstrapper.reinforce_valid_traces(valid_traces, task)
            session.learning_updates += reinforcement_result.get('reinforced', 0)
        
        # Step 5: Enforce consistency
        if len(valid_traces) > 1:
            consistency_result = self.consistency_enforcer.enforce_consistency(valid_traces, task)
            session.consistency_violations = len(consistency_result['violations'])
            
            # Apply consistency loss
            if consistency_result['consistency_loss'] > 0:
                consistency_loss = torch.tensor(consistency_result['consistency_loss'], requires_grad=True)
                consistency_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                session.learning_updates += 1
        
        # Step 6: Analyze improvements
        session.improvements = self._analyze_session_improvements(
            task, original_trace if 'original_trace' in locals() else None,
            traces, validations
        )
        
        return session
    
    def _is_correct(self, trace, task: Task) -> bool:
        """Check if trace correctly solves task"""
        try:
            if hasattr(trace, 'final_grid') and trace.final_grid is not None:
                return torch.equal(trace.final_grid, task.output)
            elif hasattr(trace, 'output') and trace.output is not None:
                return torch.equal(trace.output, task.output)
            else:
                return False
        except Exception:
            return False
    
    def _learn_from_success(self, success_trace, success_task: Task, failed_trace, failed_task: Task):
        """Learn from successful counterexample vs failed original"""
        
        try:
            # Compare successful and failed approaches
            if hasattr(success_trace, 'program') and hasattr(failed_trace, 'program'):
                # Learn what operations work
                success_ops = set(success_trace.operations or [])
                failed_ops = set(failed_trace.operations or [])
                
                # Operations that work should be reinforced
                good_ops = success_ops - failed_ops
                for op in good_ops:
                    self.learning_curves['good_operations'].append(op)
                
                # Operations that fail should be penalized
                bad_ops = failed_ops - success_ops
                for op in bad_ops:
                    self.learning_curves['bad_operations'].append(op)
            
            # Learn grid transformations that work
            if (hasattr(success_trace, 'final_grid') and 
                hasattr(failed_trace, 'final_grid')):
                
                # Analyze what made the difference
                self._analyze_transformation_difference(
                    success_trace.final_grid, success_task.output,
                    failed_trace.final_grid, failed_task.output
                )
            
        except Exception as e:
            self.logger.warning(f"Learning from success failed: {e}")
    
    def _analyze_counterexample_patterns(self, counterexamples: List[Counterexample], successes: List[bool]):
        """Analyze what types of counterexamples are being solved/failed"""
        
        for counter, success in zip(counterexamples, successes):
            pattern_key = f"{counter.perturbation_type}_{success}"
            self.error_analysis[pattern_key] += 1
            
            if not success:
                # Record what types of invariances the model lacks
                self.error_analysis[f'lacks_{counter.perturbation_type}_invariance'] += 1
    
    def _analyze_transformation_difference(self, success_grid, success_target, failed_grid, failed_target):
        """Analyze differences between successful and failed transformations"""
        
        try:
            # How close did each get?
            success_accuracy = (success_grid == success_target).float().mean().item()
            failed_accuracy = (failed_grid == failed_target).float().mean().item()
            
            accuracy_diff = success_accuracy - failed_accuracy
            self.learning_curves['transformation_accuracy_diff'].append(accuracy_diff)
            
            # What operations would fix the failed case?
            if success_accuracy > failed_accuracy:
                # The successful approach is better - learn from it
                self.learning_curves['better_transformations'].append({
                    'success_acc': success_accuracy,
                    'failed_acc': failed_accuracy,
                    'improvement': accuracy_diff
                })
                
        except Exception as e:
            self.logger.warning(f"Transformation analysis failed: {e}")
    
    def _analyze_session_improvements(self, task: Task, original_trace, traces: List[Trace], validations: List[TraceValidation]) -> Dict[str, float]:
        """Analyze improvements made during the session"""
        
        improvements = {}
        
        try:
            # Trace quality improvement
            if traces:
                avg_confidence = np.mean([t.confidence for t in traces])
                improvements['avg_trace_confidence'] = avg_confidence
                
                # Diversity improvement
                diversity_metrics = self.star_bootstrapper.get_trace_diversity_metrics(traces)
                improvements['trace_diversity'] = diversity_metrics.get('program_diversity', 0.0)
            
            # Validation success rate
            if validations:
                success_rate = sum(1 for v in validations if v.is_valid) / len(validations)
                improvements['validation_success_rate'] = success_rate
                
                avg_similarity = np.mean([v.similarity_score for v in validations])
                improvements['avg_similarity'] = avg_similarity
            
            # Original vs best trace comparison
            if original_trace and traces:
                best_trace = max(traces, key=lambda t: t.confidence)
                if hasattr(original_trace, 'confidence') and hasattr(best_trace, 'confidence'):
                    improvements['confidence_improvement'] = best_trace.confidence - original_trace.confidence
                    
        except Exception as e:
            self.logger.warning(f"Improvement analysis failed: {e}")
            
        return improvements
    
    def _should_prioritize_task(self, task: Task) -> bool:
        """Determine if task should be prioritized for critique"""
        
        # Prioritize tasks that model recently failed on
        # Prioritize tasks with specific error patterns
        # This is a simple heuristic - could be more sophisticated
        
        return np.random.random() < 0.1  # 10% random prioritization
    
    def _update_learning_curves(self, session: CritiqueSession):
        """Update learning curves with session results"""
        
        self.learning_curves['success_rate'].append(float(session.original_success))
        self.learning_curves['counterexample_success'].append(
            session.counterexamples_solved / max(session.counterexamples_generated, 1)
        )
        self.learning_curves['trace_validity'].append(
            session.valid_traces / max(session.traces_generated, 1)
        )
        self.learning_curves['consistency_violations'].append(session.consistency_violations)
        self.learning_curves['learning_updates'].append(session.learning_updates)
        
        # Add session improvements
        for key, value in session.improvements.items():
            self.learning_curves[f'session_{key}'].append(value)
    
    def _evaluate_baseline(self, sample_dataset: List[Task]) -> CritiqueMetrics:
        """Evaluate baseline performance"""
        
        successes = 0
        total = len(sample_dataset)
        
        for task in sample_dataset:
            try:
                trace = self.model.forward(task)
                if self._is_correct(trace, task):
                    successes += 1
            except Exception:
                pass
        
        baseline = CritiqueMetrics(
            total_tasks=total,
            success_rate=successes / total,
            counterexample_success_rate=0.0,
            trace_diversity=0.0,
            consistency_score=0.0,
            learning_efficiency=0.0,
            error_reduction=0.0,
            reasoning_quality=0.0
        )
        
        return baseline
    
    def _evaluate_performance(self, dataset: List[Task]) -> CritiqueMetrics:
        """Evaluate current performance"""
        
        successes = 0
        total_traces = 0
        total_valid = 0
        consistency_scores = []
        
        for task in dataset[:50]:  # Sample for efficiency
            try:
                # Main task
                trace = self.model.forward(task)
                if self._is_correct(trace, task):
                    successes += 1
                
                # Generate some traces for diversity/consistency metrics
                traces = self.star_bootstrapper.generate_diverse_traces(task, n_traces=4)
                validations = self.star_bootstrapper.verify_traces(traces, task)
                
                total_traces += len(traces)
                total_valid += sum(1 for v in validations if v.is_valid)
                
                # Consistency
                if len([t for t, v in zip(traces, validations) if v.is_valid]) > 1:
                    valid_traces = [t for t, v in zip(traces, validations) if v.is_valid]
                    consistency_result = self.consistency_enforcer.enforce_consistency(valid_traces, task)
                    if consistency_result.get('metrics'):
                        consistency_scores.append(consistency_result['metrics'].overall_consistency)
                        
            except Exception as e:
                self.logger.warning(f"Performance evaluation failed for task: {e}")
                
        # Compute metrics
        success_rate = successes / len(dataset[:50])
        trace_diversity = self._compute_current_diversity()
        consistency_score = np.mean(consistency_scores) if consistency_scores else 0.0
        
        return CritiqueMetrics(
            total_tasks=len(dataset[:50]),
            success_rate=success_rate,
            counterexample_success_rate=self._compute_counterexample_success_rate(),
            trace_diversity=trace_diversity,
            consistency_score=consistency_score,
            learning_efficiency=self._compute_learning_efficiency(),
            error_reduction=self._compute_error_reduction(),
            reasoning_quality=self._compute_reasoning_quality()
        )
    
    def _compute_improvement(self, current: CritiqueMetrics) -> float:
        """Compute improvement over baseline"""
        
        if self.baseline_metrics is None:
            return 0.0
            
        improvements = []
        improvements.append(current.success_rate - self.baseline_metrics.success_rate)
        improvements.append(current.trace_diversity - self.baseline_metrics.trace_diversity)
        improvements.append(current.consistency_score - self.baseline_metrics.consistency_score)
        
        return np.mean([imp for imp in improvements if imp > 0])
    
    def _compute_current_diversity(self) -> float:
        """Compute current trace diversity"""
        if 'trace_diversity' in self.learning_curves and self.learning_curves['trace_diversity']:
            return np.mean(self.learning_curves['trace_diversity'][-10:])  # Recent average
        return 0.0
    
    def _compute_counterexample_success_rate(self) -> float:
        """Compute current counterexample success rate"""
        if 'counterexample_success' in self.learning_curves and self.learning_curves['counterexample_success']:
            return np.mean(self.learning_curves['counterexample_success'][-10:])
        return 0.0
    
    def _compute_learning_efficiency(self) -> float:
        """Compute learning efficiency (improvements per update)"""
        if not self.learning_curves['learning_updates']:
            return 0.0
            
        total_updates = sum(self.learning_curves['learning_updates'])
        total_improvements = len([x for x in self.learning_curves['success_rate'] if x > 0.5])
        
        return total_improvements / max(total_updates, 1)
    
    def _compute_error_reduction(self) -> float:
        """Compute error reduction rate"""
        if len(self.learning_curves['success_rate']) < 10:
            return 0.0
            
        early_success = np.mean(self.learning_curves['success_rate'][:10])
        recent_success = np.mean(self.learning_curves['success_rate'][-10:])
        
        return recent_success - early_success
    
    def _compute_reasoning_quality(self) -> float:
        """Compute reasoning quality based on consistency and diversity"""
        consistency = np.mean(self.learning_curves['consistency_violations'][-10:]) if self.learning_curves['consistency_violations'] else 0
        diversity = self._compute_current_diversity()
        
        # Quality is high diversity with low inconsistency
        return diversity * (1.0 - min(consistency / 10.0, 1.0))
    
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        
        if len(self.learning_curves['success_rate']) < 50:
            return False
            
        # Check if success rate has plateaued
        recent_success = self.learning_curves['success_rate'][-20:]
        success_variance = np.var(recent_success)
        
        # Check if improvements have stalled
        recent_improvements = [
            session.improvements.get('validation_success_rate', 0) 
            for session in self.critique_sessions[-10:]
            if session.improvements
        ]
        
        improvement_trend = np.mean(recent_improvements) if recent_improvements else 0
        
        converged = (
            success_variance < 0.01 and  # Stable success rate
            improvement_trend < 0.01     # No significant improvements
        )
        
        return converged
    
    def save_critique_state(self, filepath: str):
        """Save critique training state"""
        
        state = {
            'config': self.config,
            'critique_sessions': [asdict(s) for s in self.critique_sessions],
            'learning_curves': dict(self.learning_curves),
            'error_analysis': dict(self.error_analysis),
            'baseline_metrics': asdict(self.baseline_metrics) if self.baseline_metrics else None,
            'current_metrics': asdict(self.current_metrics) if self.current_metrics else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
        self.logger.info(f"Critique state saved to {filepath}")
    
    def load_critique_state(self, filepath: str):
        """Load critique training state"""
        
        if not os.path.exists(filepath):
            self.logger.warning(f"State file not found: {filepath}")
            return
            
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config.update(state.get('config', {}))
        
        # Restore sessions
        for session_data in state.get('critique_sessions', []):
            session = CritiqueSession(**session_data)
            self.critique_sessions.append(session)
        
        # Restore curves
        for key, values in state.get('learning_curves', {}).items():
            self.learning_curves[key] = values
            
        # Restore error analysis
        for key, count in state.get('error_analysis', {}).items():
            self.error_analysis[key] = count
            
        self.logger.info(f"Critique state loaded from {filepath}")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        
        summary = {
            'total_sessions': len(self.critique_sessions),
            'total_learning_updates': sum(s.learning_updates for s in self.critique_sessions),
            'avg_session_duration': np.mean([s.session_duration for s in self.critique_sessions]) if self.critique_sessions else 0,
            'baseline_metrics': asdict(self.baseline_metrics) if self.baseline_metrics else None,
            'current_metrics': asdict(self.current_metrics) if self.current_metrics else None,
            'improvement': self._compute_improvement(self.current_metrics) if self.current_metrics else 0,
            'convergence_status': self._check_convergence(),
            'error_patterns': dict(self.error_analysis),
            'learning_curve_summary': {
                key: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0
                }
                for key, values in self.learning_curves.items() 
                if values and len(values) > 0
            }
        }
        
        return summary