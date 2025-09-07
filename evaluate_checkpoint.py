#!/usr/bin/env python3
"""
Evaluate HRM-TOPAS Checkpoints

This script evaluates saved checkpoints on the ARC evaluation set to:
1. Load model checkpoints from different training phases
2. Run evaluation on ARC evaluation tasks
3. Calculate accuracy and performance metrics
4. Compare performance across different checkpoints
5. Generate detailed evaluation reports
6. Validate that models meet 50% baseline and 85% goal targets

Usage:
  python evaluate_checkpoint.py --checkpoint checkpoints/hrm_integrated_v1/latest_checkpoint.pt
  python evaluate_checkpoint.py --checkpoint-dir checkpoints/hrm_integrated_v1 --eval-all
  python evaluate_checkpoint.py --config configs/hrm_integrated_training.json --phase-comparison
"""

import torch
import os
import sys
import json
import argparse
import traceback
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time
from collections import defaultdict
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "HRM-main"))

# Import models and utilities
from trainers.arc_dataset_loader import ARCDataset
from models.hrm_topas_bridge import HRMTOPASBridge, HRMTOPASIntegrationConfig
from models.topas_arc_60M import TopasArc60M

# HRM imports (with fallback)
try:
    from models.hrm.hrm_act_v1 import HRMActV1
    from models.common import HRMConfig
    HRM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  HRM modules not available: {e}")
    HRM_AVAILABLE = False


class CheckpointEvaluator:
    """Evaluates HRM-TOPAS model checkpoints on ARC tasks."""
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        self.config = config
        self.device = torch.device(device)
        
        # Models
        self.topas_model = None
        self.hrm_model = None
        self.hrm_topas_bridge = None
        
        # Evaluation results
        self.evaluation_results = {}
        
    def load_models_from_config(self):
        """Load models from configuration (without checkpoint)."""
        print("🔧 Initializing models from config...")
        
        try:
            # Initialize TOPAS model
            model_config = self.config.get("model_config", {})
            topas_config = {
                "slot_dim": model_config.get("slot_dim", 128),
                "dsl_vocab_size": model_config.get("dsl_vocab_size", 64),
                "use_dsl": model_config.get("use_dsl", True),
                "use_ebr": model_config.get("use_ebr", True),
                "use_relations": model_config.get("use_relations", True)
            }
            
            self.topas_model = TopasArc60M(**topas_config).to(self.device)
            print(f"  ✅ TOPAS model: {sum(p.numel() for p in self.topas_model.parameters()):,} parameters")
            
            # Initialize HRM model (if available)
            if HRM_AVAILABLE:
                hrm_config = self.config.get("hrm_config", {})
                hrm_model_config = HRMConfig(
                    batch_size=hrm_config.get("batch_size", 1),
                    seq_len=hrm_config.get("seq_len", 400),
                    vocab_size=hrm_config.get("vocab_size", 10),
                    num_puzzle_identifiers=hrm_config.get("num_puzzle_identifiers", 1000),
                    puzzle_emb_ndim=hrm_config.get("puzzle_emb_ndim", 128),
                    H_cycles=hrm_config.get("H_cycles", 3),
                    L_cycles=hrm_config.get("L_cycles", 4),
                    H_layers=hrm_config.get("H_layers", 4),
                    L_layers=hrm_config.get("L_layers", 4),
                    hidden_size=hrm_config.get("hidden_size", 512),
                    expansion=hrm_config.get("expansion", 3.0),
                    num_heads=hrm_config.get("num_heads", 8),
                    pos_encodings=hrm_config.get("pos_encodings", "rope"),
                    halt_max_steps=hrm_config.get("halt_max_steps", 6),
                    halt_exploration_prob=hrm_config.get("halt_exploration_prob", 0.1),
                    forward_dtype=hrm_config.get("forward_dtype", "bfloat16")
                )
                
                self.hrm_model = HRMActV1(hrm_model_config).to(self.device)
                print(f"  ✅ HRM model: {sum(p.numel() for p in self.hrm_model.parameters()):,} parameters")
                
                # Initialize HRM-TOPAS bridge
                bridge_config = HRMTOPASIntegrationConfig(
                    hrm_hidden_size=hrm_config.get("hidden_size", 512),
                    topas_width=model_config.get("slot_dim", 128),
                    num_attention_heads=8,
                    cross_attention_dropout=0.1,
                    puzzle_emb_dim=hrm_config.get("puzzle_emb_ndim", 128),
                    dsl_ops_count=model_config.get("dsl_vocab_size", 64),
                    adaptive_halting_threshold=0.5,
                    max_planning_steps=hrm_config.get("halt_max_steps", 6)
                )
                
                self.hrm_topas_bridge = HRMTOPASBridge(bridge_config).to(self.device)
                print(f"  ✅ HRM-TOPAS bridge: {sum(p.numel() for p in self.hrm_topas_bridge.parameters()):,} parameters")
                
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            raise
            
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint and return metadata."""
        print(f"📁 Loading checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract metadata
            metadata = {
                "path": checkpoint_path,
                "global_step": checkpoint.get("global_step", 0),
                "current_phase": checkpoint.get("current_phase", 0),
                "phase_name": checkpoint.get("phase_name", "unknown"),
                "training_complete": checkpoint.get("training_complete", False),
                "hrm_integration_enabled": checkpoint.get("hrm_integration_enabled", False)
            }
            
            # Load model states
            if self.topas_model and "topas_model_state_dict" in checkpoint:
                self.topas_model.load_state_dict(checkpoint["topas_model_state_dict"])
                print(f"  ✅ TOPAS model state loaded")
                
            if self.hrm_model and "hrm_model_state_dict" in checkpoint:
                self.hrm_model.load_state_dict(checkpoint["hrm_model_state_dict"])
                print(f"  ✅ HRM model state loaded")
                
            if self.hrm_topas_bridge and "bridge_model_state_dict" in checkpoint:
                self.hrm_topas_bridge.load_state_dict(checkpoint["bridge_model_state_dict"])
                print(f"  ✅ HRM-TOPAS bridge state loaded")
                
            # Set models to eval mode
            if self.topas_model:
                self.topas_model.eval()
            if self.hrm_model:
                self.hrm_model.eval()
            if self.hrm_topas_bridge:
                self.hrm_topas_bridge.eval()
                
            print(f"  📊 Checkpoint metadata: {metadata}")
            return metadata
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            raise
            
    def load_evaluation_dataset(self) -> ARCDataset:
        """Load ARC evaluation dataset."""
        print("📊 Loading ARC evaluation dataset...")
        
        try:
            eval_challenges = self.config.get("eval_challenges")
            eval_solutions = self.config.get("eval_solutions", eval_challenges)  # May be same file
            
            if not eval_challenges:
                raise ValueError("eval_challenges path not specified in config")
                
            if not os.path.exists(eval_challenges):
                raise FileNotFoundError(f"Evaluation challenges not found: {eval_challenges}")
                
            dataset = ARCDataset(
                challenge_file=eval_challenges,
                solution_file=eval_solutions,
                device=str(self.device),
                max_grid_size=30
            )
            
            print(f"  ✅ Loaded {len(dataset)} evaluation tasks")
            return dataset
            
        except Exception as e:
            print(f"❌ Failed to load evaluation dataset: {e}")
            raise
            
    def evaluate_single_task(self, task_data: Tuple[List, List, List, str], 
                           timeout: float = 30.0) -> Dict[str, Any]:
        """Evaluate model on a single ARC task."""
        demos, test_inputs, test_outputs, task_id = task_data
        
        result = {
            "task_id": task_id,
            "success": False,
            "predictions": [],
            "correct_predictions": 0,
            "total_test_cases": 0,
            "accuracy": 0.0,
            "error": None,
            "processing_time": 0.0
        }
        
        try:
            start_time = time.time()
            
            if not demos or len(demos) == 0:
                raise ValueError(f"No demonstration data for task {task_id}")
                
            if not test_inputs or len(test_inputs) == 0:
                raise ValueError(f"No test inputs for task {task_id}")
                
            result["total_test_cases"] = len(test_inputs)
            
            # Process each test case
            for test_idx, test_input in enumerate(test_inputs):
                try:
                    prediction = self._predict_output(demos, test_input, task_id, timeout)
                    result["predictions"].append(prediction)
                    
                    # Check if prediction is correct (if we have ground truth)
                    if test_outputs and test_idx < len(test_outputs):
                        ground_truth = test_outputs[test_idx]
                        if self._compare_outputs(prediction, ground_truth):
                            result["correct_predictions"] += 1
                            
                except Exception as e:
                    result["predictions"].append(None)
                    print(f"    ❌ Test case {test_idx} failed: {e}")
                    
            result["processing_time"] = time.time() - start_time
            result["accuracy"] = result["correct_predictions"] / max(1, result["total_test_cases"])
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
            
        return result
        
    def _predict_output(self, demos: List[Dict], test_input: torch.Tensor, 
                       task_id: str, timeout: float) -> Optional[torch.Tensor]:
        """Predict output for a test input using the loaded model."""
        try:
            # Prepare input
            if test_input.dim() == 2:
                test_input = test_input.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif test_input.dim() == 3:
                test_input = test_input.unsqueeze(0)  # Add batch dim
                
            test_input = test_input.to(self.device).float()
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):  # Disable for stable evaluation
                    
                    # Run TOPAS model
                    topas_outputs = self.topas_model(test_input)
                    
                    # If HRM is available, run integrated pipeline
                    if self.hrm_model and self.hrm_topas_bridge:
                        # Convert to HRM input format
                        batch_size = test_input.size(0)
                        grid_sequence = test_input.view(batch_size, -1).long() % 10
                        
                        # Pad/truncate to sequence length
                        seq_len = self.config.get("hrm_config", {}).get("seq_len", 400)
                        if grid_sequence.size(1) > seq_len:
                            grid_sequence = grid_sequence[:, :seq_len]
                        elif grid_sequence.size(1) < seq_len:
                            padding = torch.zeros(batch_size, seq_len - grid_sequence.size(1),
                                                dtype=grid_sequence.dtype, device=self.device)
                            grid_sequence = torch.cat([grid_sequence, padding], dim=1)
                            
                        puzzle_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                        
                        # Run HRM model
                        hrm_outputs = self.hrm_model(grid_sequence, puzzle_ids)
                        
                        # Run bridge integration
                        _, _, h, w = test_input.shape
                        topas_width = self.config.get("model_config", {}).get("slot_dim", 128)
                        grid_features = torch.randn(batch_size, topas_width, h, w, device=self.device)
                        
                        bridge_outputs = self.hrm_topas_bridge(
                            grid_features=grid_features,
                            hrm_outputs=hrm_outputs,
                            current_search_depth=1
                        )
                        
                        # Use bridge outputs to guide prediction
                        # This is a simplified approach - in practice, you'd use the DSL operation biases
                        # and enhanced features to generate the actual output grid
                        
                    # Extract prediction from model outputs
                    # This is a placeholder - actual prediction logic would depend on model architecture
                    if "output_grid" in topas_outputs:
                        prediction = topas_outputs["output_grid"]
                    elif "reconstructed" in topas_outputs:
                        prediction = topas_outputs["reconstructed"]
                    else:
                        # Fallback: return input (identity prediction)
                        prediction = test_input.squeeze(0).squeeze(0).long()
                        
                    return prediction
                    
        except Exception as e:
            print(f"      ❌ Prediction failed: {e}")
            return None
            
    def _compare_outputs(self, prediction: Optional[torch.Tensor], 
                        ground_truth: torch.Tensor) -> bool:
        """Compare predicted output with ground truth."""
        if prediction is None:
            return False
            
        try:
            # Ensure same device and type
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.to(ground_truth.device).long()
            if isinstance(ground_truth, torch.Tensor):
                ground_truth = ground_truth.long()
                
            # Compare shapes
            if prediction.shape != ground_truth.shape:
                return False
                
            # Compare values
            return torch.equal(prediction, ground_truth)
            
        except Exception:
            return False
            
    def evaluate_checkpoint_on_dataset(self, checkpoint_path: str, 
                                     max_tasks: Optional[int] = None,
                                     save_results: bool = True) -> Dict[str, Any]:
        """Evaluate a checkpoint on the full evaluation dataset."""
        print(f"🧪 Evaluating checkpoint on ARC evaluation set...")
        
        # Load models and checkpoint
        self.load_models_from_config()
        checkpoint_metadata = self.load_checkpoint(checkpoint_path)
        
        # Load evaluation dataset
        eval_dataset = self.load_evaluation_dataset()
        
        # Determine number of tasks to evaluate
        num_tasks = len(eval_dataset)
        if max_tasks and max_tasks < num_tasks:
            num_tasks = max_tasks
            print(f"  📊 Evaluating on subset: {num_tasks}/{len(eval_dataset)} tasks")
        else:
            print(f"  📊 Evaluating on full dataset: {num_tasks} tasks")
            
        # Evaluation results
        evaluation_results = {
            "checkpoint_metadata": checkpoint_metadata,
            "dataset_size": len(eval_dataset),
            "tasks_evaluated": num_tasks,
            "task_results": [],
            "overall_accuracy": 0.0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_processing_time": 0.0,
            "meets_baseline_50": False,
            "meets_goal_85": False
        }
        
        # Process tasks
        successful_tasks = 0
        total_accuracy = 0.0
        start_time = time.time()
        
        for i in range(num_tasks):
            print(f"  🔍 Task {i+1}/{num_tasks}", end=" ")
            
            try:
                task_data = eval_dataset[i]
                task_result = self.evaluate_single_task(task_data)
                
                evaluation_results["task_results"].append(task_result)
                
                if task_result["success"]:
                    successful_tasks += 1
                    total_accuracy += task_result["accuracy"]
                    print(f"✅ {task_result['task_id']} (acc: {task_result['accuracy']:.1%})")
                else:
                    evaluation_results["failed_tasks"] += 1
                    print(f"❌ {task_result['task_id']} (error)")
                    
            except Exception as e:
                evaluation_results["failed_tasks"] += 1
                print(f"❌ Task {i}: {e}")
                
        evaluation_results["successful_tasks"] = successful_tasks
        evaluation_results["total_processing_time"] = time.time() - start_time
        
        # Calculate overall metrics
        if successful_tasks > 0:
            evaluation_results["overall_accuracy"] = total_accuracy / successful_tasks
        else:
            evaluation_results["overall_accuracy"] = 0.0
            
        # Check performance targets
        evaluation_results["meets_baseline_50"] = evaluation_results["overall_accuracy"] >= 0.5
        evaluation_results["meets_goal_85"] = evaluation_results["overall_accuracy"] >= 0.85
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"Phase: {checkpoint_metadata.get('phase_name', 'unknown')}")
        print(f"Global Step: {checkpoint_metadata.get('global_step', 0):,}")
        print(f"Tasks Evaluated: {evaluation_results['tasks_evaluated']}")
        print(f"Successful Tasks: {evaluation_results['successful_tasks']}")
        print(f"Failed Tasks: {evaluation_results['failed_tasks']}")
        print(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.1%}")
        print(f"Processing Time: {evaluation_results['total_processing_time']:.1f}s")
        print(f"Avg Time/Task: {evaluation_results['total_processing_time']/max(1, num_tasks):.2f}s")
        
        print(f"\n🎯 PERFORMANCE TARGETS:")
        baseline_status = "✅" if evaluation_results["meets_baseline_50"] else "❌"
        goal_status = "✅" if evaluation_results["meets_goal_85"] else "❌"
        print(f"Baseline (50%): {baseline_status} {evaluation_results['overall_accuracy']:.1%}")
        print(f"Goal (85%): {goal_status} {evaluation_results['overall_accuracy']:.1%}")
        
        # Save results if requested
        if save_results:
            results_path = Path(checkpoint_path).parent / f"eval_results_{int(time.time())}.json"
            with open(results_path, 'w') as f:
                # Convert tensors to lists for JSON serialization
                serializable_results = self._make_json_serializable(evaluation_results)
                json.dump(serializable_results, f, indent=2)
            print(f"💾 Results saved: {results_path}")
            
        return evaluation_results
        
    def _make_json_serializable(self, obj):
        """Make evaluation results JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
            
    def compare_checkpoints(self, checkpoint_paths: List[str], 
                          max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Compare multiple checkpoints on the evaluation dataset."""
        print(f"📈 Comparing {len(checkpoint_paths)} checkpoints...")
        
        comparison_results = {
            "checkpoint_results": {},
            "comparison_summary": {},
            "best_checkpoint": None,
            "performance_progression": []
        }
        
        # Evaluate each checkpoint
        for i, checkpoint_path in enumerate(checkpoint_paths):
            print(f"\n🔍 Evaluating checkpoint {i+1}/{len(checkpoint_paths)}")
            print(f"📁 {checkpoint_path}")
            
            try:
                results = self.evaluate_checkpoint_on_dataset(
                    checkpoint_path, max_tasks, save_results=False
                )
                comparison_results["checkpoint_results"][checkpoint_path] = results
                
            except Exception as e:
                print(f"❌ Failed to evaluate {checkpoint_path}: {e}")
                comparison_results["checkpoint_results"][checkpoint_path] = {
                    "error": str(e),
                    "overall_accuracy": 0.0
                }
                
        # Analyze comparison
        valid_results = {
            path: results for path, results in comparison_results["checkpoint_results"].items()
            if "error" not in results
        }
        
        if valid_results:
            # Find best checkpoint
            best_path = max(valid_results.keys(), 
                          key=lambda p: valid_results[p]["overall_accuracy"])
            comparison_results["best_checkpoint"] = {
                "path": best_path,
                "accuracy": valid_results[best_path]["overall_accuracy"],
                "phase": valid_results[best_path]["checkpoint_metadata"].get("phase_name", "unknown")
            }
            
            # Create performance progression
            progression = []
            for path, results in valid_results.items():
                progression.append({
                    "checkpoint": os.path.basename(path),
                    "phase": results["checkpoint_metadata"].get("phase_name", "unknown"),
                    "global_step": results["checkpoint_metadata"].get("global_step", 0),
                    "accuracy": results["overall_accuracy"],
                    "meets_baseline": results["meets_baseline_50"],
                    "meets_goal": results["meets_goal_85"]
                })
                
            # Sort by global step
            progression.sort(key=lambda x: x["global_step"])
            comparison_results["performance_progression"] = progression
            
            # Print comparison summary
            print(f"\n{'='*80}")
            print(f"📊 CHECKPOINT COMPARISON SUMMARY")
            print(f"{'='*80}")
            
            print(f"{'Checkpoint':<30} {'Phase':<20} {'Step':<10} {'Accuracy':<10} {'Baseline':<8} {'Goal':<8}")
            print(f"{'-'*30} {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
            
            for result in progression:
                baseline = "✅" if result["meets_baseline"] else "❌"
                goal = "✅" if result["meets_goal"] else "❌"
                print(f"{result['checkpoint']:<30} {result['phase']:<20} {result['global_step']:<10,} "
                      f"{result['accuracy']:<9.1%} {baseline:<8} {goal:<8}")
                      
            best = comparison_results["best_checkpoint"]
            print(f"\n🏆 Best checkpoint: {os.path.basename(best['path'])} ({best['accuracy']:.1%})")
            
        return comparison_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate HRM-TOPAS checkpoints")
    parser.add_argument("--config", type=str, default="configs/hrm_integrated_training.json",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to specific checkpoint to evaluate")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Directory containing checkpoints")
    parser.add_argument("--eval-all", action="store_true",
                       help="Evaluate all checkpoints in directory")
    parser.add_argument("--phase-comparison", action="store_true",
                       help="Compare checkpoints from different phases")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    parser.add_argument("--max-tasks", type=int, default=None,
                       help="Maximum number of tasks to evaluate")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = json.load(f)
        
    print(f"📊 HRM-TOPAS Checkpoint Evaluator")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"HRM Available: {HRM_AVAILABLE}")
    if args.max_tasks:
        print(f"Max tasks: {args.max_tasks}")
    print()
    
    # Initialize evaluator
    evaluator = CheckpointEvaluator(config, args.device)
    
    try:
        if args.checkpoint:
            # Evaluate single checkpoint
            if not os.path.exists(args.checkpoint):
                print(f"❌ Checkpoint not found: {args.checkpoint}")
                sys.exit(1)
                
            results = evaluator.evaluate_checkpoint_on_dataset(
                args.checkpoint, args.max_tasks
            )
            
            # Check if performance targets are met
            if results["meets_baseline_50"]:
                print("\n✅ Checkpoint evaluation PASSED (meets 50% baseline)")
                exit_code = 0
            else:
                print(f"\n❌ Checkpoint evaluation FAILED (accuracy: {results['overall_accuracy']:.1%} < 50%)")
                exit_code = 1
                
            sys.exit(exit_code)
            
        elif args.checkpoint_dir:
            # Evaluate checkpoints in directory
            checkpoint_dir = Path(args.checkpoint_dir)
            if not checkpoint_dir.exists():
                print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
                sys.exit(1)
                
            # Find checkpoint files
            checkpoint_files = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("**/*.pt"))
            
            if not checkpoint_files:
                print(f"❌ No checkpoint files found in: {checkpoint_dir}")
                sys.exit(1)
                
            if args.eval_all or args.phase_comparison:
                # Compare multiple checkpoints
                checkpoint_paths = [str(p) for p in checkpoint_files]
                results = evaluator.compare_checkpoints(checkpoint_paths, args.max_tasks)
                
                # Check if best checkpoint meets baseline
                if results["best_checkpoint"] and results["best_checkpoint"]["accuracy"] >= 0.5:
                    print("\n✅ Checkpoint comparison PASSED (best meets 50% baseline)")
                    sys.exit(0)
                else:
                    best_acc = results["best_checkpoint"]["accuracy"] if results["best_checkpoint"] else 0.0
                    print(f"\n❌ Checkpoint comparison FAILED (best accuracy: {best_acc:.1%} < 50%)")
                    sys.exit(1)
            else:
                # Evaluate latest checkpoint
                latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Using latest checkpoint: {latest_checkpoint}")
                
                results = evaluator.evaluate_checkpoint_on_dataset(
                    str(latest_checkpoint), args.max_tasks
                )
                
                if results["meets_baseline_50"]:
                    print("\n✅ Latest checkpoint evaluation PASSED")
                    sys.exit(0)
                else:
                    print(f"\n❌ Latest checkpoint evaluation FAILED")
                    sys.exit(1)
                    
        else:
            print("❌ Must specify either --checkpoint or --checkpoint-dir")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()