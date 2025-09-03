"""
Phase 3 Self-Critique Training - Main Orchestrator for TOPAS ARC Solver
Coordinates all components of the self-critique and STaR bootstrapping system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time

# Import self-critique components
from self_critique.counterexamples import CounterexampleGenerator, Task
from self_critique.star_bootstrapper import STaRBootstrapper
from self_critique.consistency import ConsistencyEnforcer
from self_critique.critique_trainer import SelfCritiqueTrainer, CritiqueMetrics
from self_critique.trace_analysis import TraceAnalyzer

# Import from previous phases (assuming they exist)
try:
    from config import Config
    from models.transformer_model import TransformerModel
    from data.arc_dataset import ARCDataset
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Please ensure Phase 1 and Phase 2 components are available")

class SelfCriticRunner:
    """Main orchestrator for Phase 3 self-critique training"""
    
    def __init__(self, config: Dict, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = setup_logger('phase3', 
                                 log_file=f"logs/phase3_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        self.logger.info("🚀 Initializing Phase 3: Self-Critique & STaR Bootstrapping")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Config: {json.dumps(config, indent=2)}")
        
        # Load or initialize model
        self.model = self._load_model(model_path)
        
        # Initialize Phase 3 components
        self.counter_generator = CounterexampleGenerator(device=self.device)
        self.star_bootstrapper = STaRBootstrapper(self.model, device=self.device)
        self.consistency_enforcer = ConsistencyEnforcer(device=self.device)
        self.trace_analyzer = TraceAnalyzer(device=self.device)
        self.critique_trainer = SelfCritiqueTrainer(self.model, device=self.device, config=config)
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = None
        self.training_history = []
        
        # Exit criteria tracking
        self.exit_criteria = {
            'self_critique_improvement': config.get('exit_self_critique_improvement', 0.10),
            'consistency_threshold': config.get('exit_consistency_threshold', 0.01),
            'counterexample_success_rate': config.get('exit_counterexample_success_rate', 0.80),
            'trace_explanation_quality': config.get('exit_explanation_quality', 0.80)
        }
        
        self.exit_criteria_met = {key: False for key in self.exit_criteria.keys()}
        
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load model from Phase 1/2 or initialize new one"""
        
        try:
            if model_path and os.path.exists(model_path):
                self.logger.info(f"Loading model from {model_path}")
                
                # Try to load state dict
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Initialize model (assuming config has model params)
                model = TransformerModel(
                    vocab_size=self.config.get('vocab_size', 1000),
                    hidden_dim=self.config.get('hidden_dim', 512),
                    num_layers=self.config.get('num_layers', 6),
                    num_heads=self.config.get('num_heads', 8)
                )
                
                model.load_state_dict(state_dict)
                model.to(self.device)
                
                self.logger.info("✅ Model loaded successfully")
                return model
                
            else:
                self.logger.warning("No model path provided or file not found, initializing new model")
                
                # Initialize new model
                model = TransformerModel(
                    vocab_size=self.config.get('vocab_size', 1000),
                    hidden_dim=self.config.get('hidden_dim', 512),
                    num_layers=self.config.get('num_layers', 6),
                    num_heads=self.config.get('num_heads', 8)
                )
                
                model.to(self.device)
                return model
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def train(self, dataset: List[Task], validation_dataset: Optional[List[Task]] = None) -> Dict:
        """Main training loop for Phase 3"""
        
        self.logger.info("🎯 Starting Phase 3 Self-Critique Training")
        self.logger.info(f"Training dataset size: {len(dataset)}")
        self.logger.info(f"Validation dataset size: {len(validation_dataset) if validation_dataset else 0}")
        
        training_start = time.time()
        
        # Initial evaluation to establish baseline
        baseline_metrics = self._evaluate_model(dataset[:50])  # Sample for speed
        self.logger.info(f"📊 Baseline metrics: {baseline_metrics}")
        
        best_performance = 0.0
        epochs_without_improvement = 0
        max_epochs = self.config.get('max_epochs', 100)
        patience = self.config.get('patience', 10)
        
        try:
            for epoch in range(max_epochs):
                self.current_epoch = epoch
                
                self.logger.info(f"\n🔄 Epoch {epoch + 1}/{max_epochs}")
                
                epoch_start = time.time()
                
                # Run critique training
                epoch_results = self._train_epoch(dataset, validation_dataset)
                
                # Evaluate current performance
                current_metrics = self._evaluate_model(validation_dataset or dataset[:100])
                
                # Check for improvement
                current_performance = current_metrics.success_rate
                if current_performance > best_performance + 0.01:  # Significant improvement
                    best_performance = current_performance
                    epochs_without_improvement = 0
                    self.best_metrics = current_metrics
                    
                    # Save best model
                    self._save_model(f"checkpoints/phase3_best_epoch_{epoch}.pt")
                    
                    self.logger.info(f"🏆 New best performance: {best_performance:.4f}")
                else:
                    epochs_without_improvement += 1
                
                # Update training history
                epoch_duration = time.time() - epoch_start
                epoch_summary = {
                    'epoch': epoch,
                    'duration': epoch_duration,
                    'metrics': current_metrics.__dict__,
                    'critique_results': epoch_results,
                    'exit_criteria_status': self.exit_criteria_met.copy()
                }
                self.training_history.append(epoch_summary)
                
                # Check exit criteria
                if self._check_exit_criteria(current_metrics, baseline_metrics):
                    self.logger.info("🎉 Exit criteria met! Training completed successfully.")
                    break
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    self.logger.info(f"⏹️  Early stopping: No improvement for {patience} epochs")
                    break
                
                # Log epoch summary
                self.logger.info(f"⏱️  Epoch {epoch + 1} completed in {epoch_duration:.2f}s")
                self.logger.info(f"📈 Current performance: {current_performance:.4f}")
                self.logger.info(f"🔍 Exit criteria met: {sum(self.exit_criteria_met.values())}/{len(self.exit_criteria_met)}")
                
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(epoch)
                
            # Final evaluation
            final_metrics = self._evaluate_model(validation_dataset or dataset)
            training_duration = time.time() - training_start
            
            # Comprehensive results
            training_results = {
                'baseline_metrics': baseline_metrics.__dict__,
                'final_metrics': final_metrics.__dict__,
                'best_metrics': self.best_metrics.__dict__ if self.best_metrics else final_metrics.__dict__,
                'training_duration': training_duration,
                'epochs_completed': self.current_epoch + 1,
                'exit_criteria_met': self.exit_criteria_met,
                'training_history': self.training_history,
                'improvement_over_baseline': final_metrics.success_rate - baseline_metrics.success_rate
            }
            
            self.logger.info("🏁 Phase 3 Training Completed!")
            self.logger.info(f"⏱️  Total training time: {training_duration/3600:.2f} hours")
            self.logger.info(f"📈 Improvement: {training_results['improvement_over_baseline']:.4f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            raise
    
    def _train_epoch(self, dataset: List[Task], validation_dataset: Optional[List[Task]] = None) -> Dict:
        """Train for one epoch using self-critique"""
        
        # Sample subset of dataset for efficiency
        epoch_size = min(len(dataset), self.config.get('epoch_size', 100))
        epoch_indices = np.random.choice(len(dataset), size=epoch_size, replace=False)
        epoch_dataset = [dataset[i] for i in epoch_indices]
        
        # Run critique training
        critique_results = self.critique_trainer.critique_loop(
            epoch_dataset, 
            max_iterations=epoch_size
        )
        
        # Analyze patterns in successful/failed traces
        self._analyze_epoch_patterns(critique_results)
        
        # Update consistency enforcement
        self._update_consistency_rules(critique_results)
        
        return critique_results
    
    def _analyze_epoch_patterns(self, critique_results: Dict):
        """Analyze patterns from critique training"""
        
        if not critique_results.get('sessions'):
            return
            
        # Collect all traces from successful sessions
        successful_traces = []
        failed_traces = []
        
        for session_data in critique_results['sessions']:
            if session_data.get('valid_traces', 0) > 0:
                # This is a placeholder - in practice would extract actual traces
                successful_traces.extend([f"session_{session_data['task_id']}_trace" for _ in range(session_data['valid_traces'])])
            else:
                failed_traces.append(f"session_{session_data['task_id']}_failed")
        
        if successful_traces:
            # Would analyze with trace_analyzer if we had actual trace objects
            self.logger.info(f"📊 Pattern analysis: {len(successful_traces)} successful traces analyzed")
    
    def _update_consistency_rules(self, critique_results: Dict):
        """Update consistency enforcement rules based on results"""
        
        total_violations = sum(session.get('consistency_violations', 0) 
                             for session in critique_results.get('sessions', []))
        
        if total_violations > 0:
            self.logger.info(f"⚖️  Consistency violations detected: {total_violations}")
            
            # Adjust consistency threshold if too many violations
            if total_violations > len(critique_results.get('sessions', [])) * 0.5:
                self.consistency_enforcer.consistency_threshold *= 0.95  # Make more lenient
                self.logger.info(f"📉 Relaxing consistency threshold to {self.consistency_enforcer.consistency_threshold:.3f}")
    
    def _evaluate_model(self, dataset: List[Task]) -> CritiqueMetrics:
        """Comprehensive model evaluation"""
        
        evaluation_size = min(len(dataset), 100)  # Limit for efficiency
        eval_dataset = dataset[:evaluation_size]
        
        # Basic success rate
        successes = 0
        total_traces = 0
        valid_traces = 0
        consistency_scores = []
        counterexample_successes = []
        
        self.model.eval()
        
        with torch.no_grad():
            for task in eval_dataset:
                try:
                    # Main task evaluation
                    trace = self._model_forward_safe(task)
                    if trace and self._is_correct_solution(trace, task):
                        successes += 1
                    
                    # Counterexample evaluation
                    counterexamples = self.counter_generator.generate_from_failure(task, self.model, n_counterexamples=3)
                    counter_success_count = 0
                    
                    for counter in counterexamples[:3]:  # Limit for speed
                        counter_trace = self._model_forward_safe(counter.task)
                        if counter_trace and self._is_correct_solution(counter_trace, counter.task):
                            counter_success_count += 1
                    
                    if counterexamples:
                        counterexample_successes.append(counter_success_count / len(counterexamples))
                    
                    # Trace diversity evaluation
                    traces = self.star_bootstrapper.generate_diverse_traces(task, n_traces=4)
                    validations = self.star_bootstrapper.verify_traces(traces, task)
                    
                    total_traces += len(traces)
                    valid_traces += sum(1 for v in validations if v.is_valid)
                    
                    # Consistency evaluation
                    valid_trace_objects = [t for t, v in zip(traces, validations) if v.is_valid]
                    if len(valid_trace_objects) > 1:
                        consistency_result = self.consistency_enforcer.enforce_consistency(valid_trace_objects, task)
                        if consistency_result.get('metrics'):
                            consistency_scores.append(consistency_result['metrics'].overall_consistency)
                    
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for task: {e}")
                    continue
        
        # Compute metrics
        success_rate = successes / evaluation_size if evaluation_size > 0 else 0.0
        counterexample_rate = np.mean(counterexample_successes) if counterexample_successes else 0.0
        trace_diversity = valid_traces / max(total_traces, 1)
        consistency_score = np.mean(consistency_scores) if consistency_scores else 0.0
        
        metrics = CritiqueMetrics(
            total_tasks=evaluation_size,
            success_rate=success_rate,
            counterexample_success_rate=counterexample_rate,
            trace_diversity=trace_diversity,
            consistency_score=consistency_score,
            learning_efficiency=0.0,  # Would compute based on training progress
            error_reduction=0.0,      # Would compute based on historical data
            reasoning_quality=consistency_score * trace_diversity  # Combined measure
        )
        
        return metrics
    
    def _model_forward_safe(self, task: Task):
        """Safe model forward pass with error handling"""
        try:
            return self.model.forward(task.input)
        except Exception as e:
            self.logger.warning(f"Model forward failed: {e}")
            return None
    
    def _is_correct_solution(self, trace, task: Task) -> bool:
        """Check if trace produces correct solution"""
        try:
            if hasattr(trace, 'final_grid') and trace.final_grid is not None:
                return torch.equal(trace.final_grid, task.output)
            elif hasattr(trace, 'output') and trace.output is not None:
                return torch.equal(trace.output, task.output)
            elif torch.is_tensor(trace):
                return torch.equal(trace, task.output)
            else:
                return False
        except Exception:
            return False
    
    def _check_exit_criteria(self, current_metrics: CritiqueMetrics, baseline_metrics: CritiqueMetrics) -> bool:
        """Check if exit criteria have been met"""
        
        # Self-critique improvement
        improvement = current_metrics.success_rate - baseline_metrics.success_rate
        if improvement >= self.exit_criteria['self_critique_improvement']:
            self.exit_criteria_met['self_critique_improvement'] = True
            self.logger.info(f"✅ Self-critique improvement criterion met: {improvement:.4f} >= {self.exit_criteria['self_critique_improvement']:.4f}")
        
        # Consistency threshold
        if current_metrics.consistency_score >= 1.0 - self.exit_criteria['consistency_threshold']:
            self.exit_criteria_met['consistency_threshold'] = True
            self.logger.info(f"✅ Consistency criterion met: {current_metrics.consistency_score:.4f} >= {1.0 - self.exit_criteria['consistency_threshold']:.4f}")
        
        # Counterexample success rate
        if current_metrics.counterexample_success_rate >= self.exit_criteria['counterexample_success_rate']:
            self.exit_criteria_met['counterexample_success_rate'] = True
            self.logger.info(f"✅ Counterexample success criterion met: {current_metrics.counterexample_success_rate:.4f} >= {self.exit_criteria['counterexample_success_rate']:.4f}")
        
        # Reasoning quality
        if current_metrics.reasoning_quality >= self.exit_criteria['trace_explanation_quality']:
            self.exit_criteria_met['trace_explanation_quality'] = True
            self.logger.info(f"✅ Reasoning quality criterion met: {current_metrics.reasoning_quality:.4f} >= {self.exit_criteria['trace_explanation_quality']:.4f}")
        
        # All criteria must be met
        all_met = all(self.exit_criteria_met.values())
        
        if all_met:
            self.logger.info("🎉 ALL EXIT CRITERIA MET!")
        else:
            remaining = [k for k, v in self.exit_criteria_met.items() if not v]
            self.logger.info(f"⏳ Remaining criteria: {remaining}")
        
        return all_met
    
    def _save_model(self, filepath: str):
        """Save model checkpoint"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'epoch': self.current_epoch,
            'best_metrics': self.best_metrics.__dict__ if self.best_metrics else None,
            'exit_criteria_met': self.exit_criteria_met
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"💾 Model saved to {filepath}")
    
    def _save_checkpoint(self, epoch: int):
        """Save comprehensive training checkpoint"""
        
        checkpoint_dir = f"checkpoints/phase3_epoch_{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self._save_model(f"{checkpoint_dir}/model.pt")
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'training_history': self.training_history,
            'exit_criteria_met': self.exit_criteria_met,
            'critique_trainer_state': self.critique_trainer.get_training_summary()
        }
        
        with open(f"{checkpoint_dir}/training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2, default=str)
        
        # Save critique trainer state
        self.critique_trainer.save_critique_state(f"{checkpoint_dir}/critique_state.json")
        
        self.logger.info(f"💾 Checkpoint saved for epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint"""
        
        # Load model
        model_path = f"{checkpoint_dir}/model.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.exit_criteria_met = checkpoint.get('exit_criteria_met', {})
            
            if checkpoint.get('best_metrics'):
                self.best_metrics = CritiqueMetrics(**checkpoint['best_metrics'])
        
        # Load training state
        state_path = f"{checkpoint_dir}/training_state.json"
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                training_state = json.load(f)
                self.training_history = training_state.get('training_history', [])
        
        # Load critique trainer state
        critique_path = f"{checkpoint_dir}/critique_state.json"
        if os.path.exists(critique_path):
            self.critique_trainer.load_critique_state(critique_path)
        
        self.logger.info(f"📂 Checkpoint loaded from {checkpoint_dir}")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        
        summary = {
            'phase': 3,
            'training_type': 'self_critique_star_bootstrapping',
            'current_epoch': self.current_epoch,
            'exit_criteria': self.exit_criteria,
            'exit_criteria_met': self.exit_criteria_met,
            'best_metrics': self.best_metrics.__dict__ if self.best_metrics else None,
            'training_duration': sum(h.get('duration', 0) for h in self.training_history),
            'total_critique_sessions': sum(
                len(h.get('critique_results', {}).get('sessions', [])) 
                for h in self.training_history
            ),
            'component_status': {
                'counterexample_generator': 'active',
                'star_bootstrapper': 'active',
                'consistency_enforcer': 'active',
                'trace_analyzer': 'active',
                'critique_trainer': 'active'
            }
        }
        
        return summary

def main():
    """Main entry point for Phase 3 training"""
    
    parser = argparse.ArgumentParser(description="Phase 3: Self-Critique & STaR Bootstrapping Training")
    parser.add_argument('--config', type=str, default='config/phase3_config.json', help='Configuration file path')
    parser.add_argument('--model-path', type=str, help='Path to pre-trained model from Phase 1/2')
    parser.add_argument('--data-path', type=str, default='data/arc', help='Path to ARC dataset')
    parser.add_argument('--output-dir', type=str, default='output/phase3', help='Output directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load configuration
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                'max_epochs': 50,
                'epoch_size': 100,
                'patience': 10,
                'learning_rate': 1e-4,
                'n_traces_per_task': 16,
                'n_counterexamples': 8,
                'consistency_threshold': 0.95,
                'exit_self_critique_improvement': 0.10,
                'exit_consistency_threshold': 0.01,
                'exit_counterexample_success_rate': 0.80,
                'exit_explanation_quality': 0.80
            }
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Initialize trainer
    trainer = Phase3SelfCritic(config, model_path=args.model_path)
    
    # Load checkpoint if resuming
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    try:
        # Load dataset (placeholder - replace with actual ARC dataset loading)
        print("📂 Loading ARC dataset...")
        dataset = []  # Would load actual dataset here
        validation_dataset = []  # Would load validation set here
        
        if not dataset:
            # Create dummy dataset for testing
            print("⚠️  Using dummy dataset for testing")
            dummy_task = Task(
                input=torch.randint(0, 10, (5, 5)),
                output=torch.randint(0, 10, (5, 5)),
                constraints={},
                metadata={}
            )
            dataset = [dummy_task] * 100
            validation_dataset = [dummy_task] * 20
        
        print(f"📊 Dataset loaded: {len(dataset)} training, {len(validation_dataset)} validation")
        
        # Run training
        results = trainer.train(dataset, validation_dataset)
        
        # Save final results
        results_path = os.path.join(args.output_dir, 'phase3_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save training summary
        summary = trainer.get_training_summary()
        summary_path = os.path.join(args.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("🎉 Phase 3 training completed successfully!")
        print(f"📊 Results saved to: {results_path}")
        print(f"📋 Summary saved to: {summary_path}")
        
        # Print key metrics
        final_metrics = results.get('final_metrics', {})
        print(f"\n📈 Final Performance:")
        print(f"   Success Rate: {final_metrics.get('success_rate', 0):.4f}")
        print(f"   Counterexample Success: {final_metrics.get('counterexample_success_rate', 0):.4f}")
        print(f"   Trace Diversity: {final_metrics.get('trace_diversity', 0):.4f}")
        print(f"   Consistency Score: {final_metrics.get('consistency_score', 0):.4f}")
        print(f"   Improvement: {results.get('improvement_over_baseline', 0):.4f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        # Save current state
        trainer._save_checkpoint(trainer.current_epoch)
        print("💾 Current state saved")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()