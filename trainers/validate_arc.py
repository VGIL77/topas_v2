#!/usr/bin/env python3
"""
ARC Validation Harness
Validates model performance on ARC dataset with exact/near-miss accuracy metrics
"""
import sys
import os
import torch
import json
import numpy as np
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Import model components
from models.topas_arc_60M import TopasARC60M, ModelConfig
from models.dsl_search import DSLProgram, apply_program
from arc_dataset_loader import ARCDataset, ARCDataLoader

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Results from ARC validation"""
    exact_match: int = 0
    near_miss_98: int = 0  # >= 98% accuracy
    near_miss_95: int = 0  # >= 95% accuracy  
    near_miss_90: int = 0  # >= 90% accuracy
    near_miss_80: int = 0  # >= 80% accuracy
    total_tasks: int = 0
    avg_accuracy: float = 0.0
    median_accuracy: float = 0.0
    std_accuracy: float = 0.0
    processing_time: float = 0.0
    dsl_successes: int = 0
    neural_successes: int = 0
    hybrid_successes: int = 0


class ARCValidationHarness:
    """Comprehensive ARC validation with exact and near-miss metrics"""
    
    def __init__(self, model_path: Optional[str] = None, use_dsl: bool = True, device: str = "cpu"):
        self.device = torch.device(device)
        self.use_dsl = use_dsl
        
        # DSLHead removed — fallback to dsl_search apply_program
        if self.use_dsl:
            from models.dsl_search import apply_program
            self.dsl_apply = apply_program
        
        # Initialize model if path provided
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = self._load_model(model_path)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        
        # Initialize dataset loader
        self.dataset_loader = ARCDataLoader(data_dir="ARC", device=str(device))
        
    def _load_model(self, model_path: str) -> TopasARC60M:
        """Load TOPAS model from checkpoint"""
        config = ModelConfig()
        model = TopasARC60M(config).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def compute_accuracy(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """Compute pixel-wise accuracy between prediction and target"""
        if prediction.shape != target.shape:
            return 0.0
        
        # Convert to same device and dtype
        prediction = prediction.to(target.device).to(target.dtype)
        
        # Compute pixel-wise accuracy
        correct_pixels = (prediction == target).sum().item()
        total_pixels = target.numel()
        
        return correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def solve_task_dsl(self, demos: List[Tuple[torch.Tensor, torch.Tensor]], 
                       test_input: torch.Tensor) -> Tuple[Optional[torch.Tensor], str]:
        """Solve task using DSL head"""
        if not self.use_dsl:
            return None, "DSL_DISABLED"
        
        try:
            # Attempt DSL solution
            # DSL solution disabled - use neural model instead
            result = None
            
            if result is not None:
                return result, "DSL_SUCCESS"
            else:
                return None, "DSL_FAILED"
                
        except Exception as e:
            return None, f"DSL_ERROR_{type(e).__name__}"
    
    def solve_task_neural(self, demos: List[Tuple[torch.Tensor, torch.Tensor]], 
                         test_input: torch.Tensor) -> Tuple[Optional[torch.Tensor], str]:
        """Solve task using neural model"""
        if self.model is None:
            return None, "NO_MODEL"
        
        try:
            # Prepare demos in the format expected by TopasARC60M
            demo_dicts = []
            for inp, out in demos:
                demo_dicts.append({
                    'input': inp.unsqueeze(0) if inp.dim() == 2 else inp,
                    'output': out.unsqueeze(0) if out.dim() == 2 else out
                })
            
            # Prepare test input
            test_dict = {'input': test_input.unsqueeze(0) if test_input.dim() == 2 else test_input}
            
            with torch.no_grad():
                # Forward pass through the model
                grid_pred, logits, size_pred, extras = self.model(demo_dicts, test_dict)
                
                # Use the grid prediction
                if grid_pred is not None:
                    # Remove batch dimension if present
                    result = grid_pred.squeeze(0) if grid_pred.dim() == 3 else grid_pred
                    return result, "NEURAL_SUCCESS"
                else:
                    return None, "NEURAL_NO_OUTPUT"
            
        except Exception as e:
            logger.debug(f"Neural solving failed: {e}")
            return None, f"NEURAL_ERROR_{type(e).__name__}"
    
    def solve_task_hybrid(self, demos: List[Tuple[torch.Tensor, torch.Tensor]], 
                         test_input: torch.Tensor) -> Tuple[Optional[torch.Tensor], str]:
        """Solve task using hybrid DSL + neural approach"""
        # Try DSL first
        dsl_result, dsl_status = self.solve_task_dsl(demos, test_input)
        if dsl_result is not None:
            return dsl_result, f"HYBRID_DSL_{dsl_status}"
        
        # Fall back to neural
        neural_result, neural_status = self.solve_task_neural(demos, test_input)
        if neural_result is not None:
            return neural_result, f"HYBRID_NEURAL_{neural_status}"
        
        return None, f"HYBRID_FAILED_DSL_{dsl_status}_NEURAL_{neural_status}"
    
    def validate_single_task(self, demos: List[Tuple[torch.Tensor, torch.Tensor]], 
                           test_input: torch.Tensor, test_output: torch.Tensor, 
                           task_id: str) -> Dict[str, Any]:
        """Validate a single ARC task"""
        
        # Try different solving approaches
        results = {}
        
        # DSL approach
        if self.use_dsl:
            dsl_pred, dsl_status = self.solve_task_dsl(demos, test_input)
            if dsl_pred is not None:
                dsl_acc = self.compute_accuracy(dsl_pred, test_output)
                results['dsl'] = {'accuracy': dsl_acc, 'status': dsl_status}
            else:
                results['dsl'] = {'accuracy': 0.0, 'status': dsl_status}
        
        # Neural approach
        neural_pred, neural_status = self.solve_task_neural(demos, test_input)
        if neural_pred is not None:
            neural_acc = self.compute_accuracy(neural_pred, test_output)
            results['neural'] = {'accuracy': neural_acc, 'status': neural_status}
        else:
            results['neural'] = {'accuracy': 0.0, 'status': neural_status}
        
        # Hybrid approach
        hybrid_pred, hybrid_status = self.solve_task_hybrid(demos, test_input)
        if hybrid_pred is not None:
            hybrid_acc = self.compute_accuracy(hybrid_pred, test_output)
            results['hybrid'] = {'accuracy': hybrid_acc, 'status': hybrid_status}
        else:
            results['hybrid'] = {'accuracy': 0.0, 'status': hybrid_status}
        
        # Best result
        best_acc = max(r['accuracy'] for r in results.values())
        best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
        
        return {
            'task_id': task_id,
            'best_accuracy': best_acc,
            'best_method': best_method,
            'results': results,
            'target_shape': list(test_output.shape)
        }
    
    def validate_dataset(self, dataset: str = 'evaluation', max_tasks: Optional[int] = None) -> ValidationResults:
        """
        Validate model on ARC dataset
        
        Args:
            dataset: 'training', 'evaluation', or 'test'
            max_tasks: Maximum number of tasks to validate
        """
        logger.info(f"Validating on ARC {dataset} dataset...")
        
        # Load dataset
        try:
            if dataset == 'training':
                if self.dataset_loader.train_dataset is None:
                    logger.error("Training dataset not available")
                    return ValidationResults()
                arc_dataset = self.dataset_loader.train_dataset
            elif dataset == 'evaluation':
                if self.dataset_loader.eval_dataset is None:
                    logger.error("Evaluation dataset not available")
                    return ValidationResults()
                arc_dataset = self.dataset_loader.eval_dataset
            elif dataset == 'test':
                if self.dataset_loader.test_dataset is None:
                    logger.error("Test dataset not available")
                    return ValidationResults()
                arc_dataset = self.dataset_loader.test_dataset
            else:
                logger.error(f"Unknown dataset: {dataset}")
                return ValidationResults()
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return ValidationResults()
        
        # Limit tasks if requested
        num_tasks = len(arc_dataset)
        if max_tasks:
            num_tasks = min(num_tasks, max_tasks)
            logger.info(f"Limited to {num_tasks} tasks")
        
        logger.info(f"Loaded {num_tasks} tasks")
        
        # Initialize results
        results = ValidationResults()
        results.total_tasks = num_tasks
        
        # Track per-task results
        task_accuracies = []
        dsl_successes = 0
        neural_successes = 0
        hybrid_successes = 0
        
        start_time = time.time()
        
        # Process each task
        for i in range(num_tasks):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{num_tasks} ({100*i/num_tasks:.1f}%)")
            
            # Get task data
            demos, test_inputs, test_outputs, task_id = arc_dataset[i]
            
            # Use first test case
            if test_inputs and test_outputs:
                test_input = test_inputs[0]
                test_output = test_outputs[0]
                
                # Validate the task
                task_result = self.validate_single_task(demos, test_input, test_output, task_id)
                accuracy = task_result['best_accuracy']
                best_method = task_result['best_method']
                
                task_accuracies.append(accuracy)
                
                # Count exact matches and near misses
                if accuracy >= 1.0:
                    results.exact_match += 1
                elif accuracy >= 0.98:
                    results.near_miss_98 += 1
                elif accuracy >= 0.95:
                    results.near_miss_95 += 1
                elif accuracy >= 0.90:
                    results.near_miss_90 += 1
                elif accuracy >= 0.80:
                    results.near_miss_80 += 1
                
                # Count method successes (exact matches only)
                if accuracy >= 1.0:
                    if 'dsl' in best_method.lower():
                        dsl_successes += 1
                    elif 'neural' in best_method.lower():
                        neural_successes += 1
                    elif 'hybrid' in best_method.lower():
                        hybrid_successes += 1
                
                # Verbose logging
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Task {task_id}: {accuracy:.3f} ({best_method})")
        
        # Calculate statistics
        results.processing_time = time.time() - start_time
        results.dsl_successes = dsl_successes
        results.neural_successes = neural_successes
        results.hybrid_successes = hybrid_successes
        
        if task_accuracies:
            results.avg_accuracy = np.mean(task_accuracies)
            results.median_accuracy = np.median(task_accuracies)
            results.std_accuracy = np.std(task_accuracies)
        
        return results
    
    def print_results(self, results: ValidationResults):
        """Print formatted validation results"""
        print("\n" + "="*80)
        print("ARC VALIDATION RESULTS")
        print("="*80)
        
        total = results.total_tasks
        if total == 0:
            print("No tasks processed")
            return
        
        print(f"Dataset Statistics:")
        print(f"   Total tasks processed: {total}")
        print(f"   Processing time: {results.processing_time:.1f}s")
        print(f"   Average time per task: {results.processing_time/total:.2f}s")
        
        print(f"\nAccuracy Breakdown:")
        print(f"   Exact matches (100%):     {results.exact_match:4d} ({100*results.exact_match/total:5.1f}%)")
        print(f"   Near miss (>=98%):        {results.near_miss_98:4d} ({100*results.near_miss_98/total:5.1f}%)")
        print(f"   Near miss (>=95%):        {results.near_miss_95:4d} ({100*results.near_miss_95/total:5.1f}%)")
        print(f"   Near miss (>=90%):        {results.near_miss_90:4d} ({100*results.near_miss_90/total:5.1f}%)")
        print(f"   Near miss (>=80%):        {results.near_miss_80:4d} ({100*results.near_miss_80/total:5.1f}%)")
        
        high_quality = results.exact_match + results.near_miss_98 + results.near_miss_95
        print(f"   High quality (>=95%):     {high_quality:4d} ({100*high_quality/total:5.1f}%)")
        
        print(f"\nStatistical Metrics:")
        print(f"   Average accuracy:         {results.avg_accuracy:.4f} ({100*results.avg_accuracy:.1f}%)")
        print(f"   Median accuracy:          {results.median_accuracy:.4f} ({100*results.median_accuracy:.1f}%)")
        print(f"   Standard deviation:       {results.std_accuracy:.4f}")
        
        print(f"\nMethod Breakdown (exact matches only):")
        print(f"   DSL successes:            {results.dsl_successes:4d}")
        print(f"   Neural successes:         {results.neural_successes:4d}")
        print(f"   Hybrid successes:         {results.hybrid_successes:4d}")
        
        # Performance summary
        exact_pct = 100 * results.exact_match / total
        high_quality_pct = 100 * high_quality / total
        
        print(f"\nPerformance Summary:")
        print(f"   Exact match accuracy: {exact_pct:.1f}%")
        print(f"   High-quality (≥95%): {high_quality_pct:.1f}%")


def main():
    """Main validation script"""
    parser = argparse.ArgumentParser(description="ARC Validation Harness")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--dataset", choices=['training', 'evaluation', 'test'], 
                       default='evaluation', help="Dataset to validate on")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks")
    parser.add_argument("--no-dsl", action="store_true", help="Disable DSL head")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    logger.info("ARC Validation Harness")
    logger.info("="*60)
    
    # Initialize harness
    harness = ARCValidationHarness(
        model_path=args.model,
        use_dsl=not args.no_dsl,
        device=args.device
    )
    
    # Run validation
    results = harness.validate_dataset(
        dataset=args.dataset,
        max_tasks=args.max_tasks
    )
    
    # Print results
    harness.print_results(results)
    
    # Save results if requested
    if args.output:
        output_data = {
            'results': results.__dict__,
            'validation_config': {
                'dataset': args.dataset,
                'max_tasks': args.max_tasks,
                'use_dsl': not args.no_dsl,
                'device': args.device
            },
            'timestamp': time.time()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()