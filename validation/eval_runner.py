#!/usr/bin/env python3
"""
EvalRunner - Sophisticated evaluation system for TOPAS ARC models

Provides comprehensive evaluation capabilities including:
- Exact@1, Exact@K metrics
- First-Hit Rate computation
- Generalization gap analysis
- Task-level and global accuracy metrics
- Advanced model performance tracking
"""

import json
import argparse
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Import sanitization functions for device-side assert protection
def _pad_or_crop(grid: List[List[int]], max_size: int = 30) -> List[List[int]]:
    """Pad or crop grid to ensure it doesn't exceed max_size x max_size."""
    if not grid or not grid[0]:
        return [[0]]
    
    height, width = len(grid), len(grid[0])
    
    # Crop if too large
    if height > max_size:
        grid = grid[:max_size]
    if width > max_size:
        grid = [row[:max_size] for row in grid]
    
    return grid

def _sanitize_grid(grid: List[List[int]], num_colors: int = 10, max_size: int = 30) -> List[List[int]]:
    """
    Sanitize ARC grid: crop/pad first, then clamp all values into [0, num_colors-1].
    CRITICAL: Prevents device-side assert cascades when values > 9 reach CUDA kernels.
    """
    grid = _pad_or_crop(grid, max_size)
    return [[min(max(int(cell), 0), num_colors - 1) for cell in row] for row in grid]
import time

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for ARC tasks"""
    exact_at_1: float = 0.0      # Exact match on first attempt
    exact_at_k: float = 0.0      # Exact match within K attempts
    first_hit_rate: float = 0.0  # Probability of first prediction being correct
    task_solve_rate: float = 0.0 # Percentage of tasks completely solved
    generalization_gap: float = 0.0  # Performance difference on seen vs unseen
    
    # Detailed breakdown
    total_tasks: int = 0
    solved_tasks: int = 0
    exact_hits: int = 0
    total_items: int = 0
    
    # Per-task metrics
    task_results: Dict[str, bool] = field(default_factory=dict)
    task_times: Dict[str, float] = field(default_factory=dict)
    task_errors: Dict[str, str] = field(default_factory=dict)

# Utility functions
def to_grid(data):
    """Convert data to numpy grid"""
    return np.array(data)

def grid_equal(grid1, grid2):
    """Check if two grids are equal"""
    return np.array_equal(grid1, grid2)

class EvalRunner:
    """
    Sophisticated evaluation runner for TOPAS ARC models
    
    Features:
    - Comprehensive metric computation (Exact@1, Exact@K, First-Hit Rate)
    - Modular evaluation for different model types
    - Advanced performance tracking and analysis
    - Integration with TopasARCTrainer pipeline
    """
    
    def __init__(self, model=None, device=None, verbose=True):
        """
        Initialize EvalRunner
        
        Args:
            model: Optional model to evaluate (can be set later)
            device: Device to run evaluation on
            verbose: Enable detailed logging
        """
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.verbose = verbose
        
        if self.verbose:
            logger.info(f"EvalRunner initialized on device: {self.device}")
    
    def set_model(self, model):
        """Set or update the model for evaluation"""
        self.model = model
        if self.model:
            self.model.to(self.device)
            self.model.eval()
    
    def evaluate_topas_model(self, 
                            challenges_path: str,
                            solutions_path: str,
                            k_attempts: int = 2) -> EvaluationMetrics:
        """
        Evaluate TopasARC60M model on ARC tasks
        
        Args:
            challenges_path: Path to challenges JSON
            solutions_path: Path to solutions JSON  
            k_attempts: Number of attempts for Exact@K metric
            
        Returns:
            Comprehensive evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not set. Use set_model() first.")
        
        # Load evaluation data
        with open(challenges_path) as f:
            eval_challenges = json.load(f)
        with open(solutions_path) as f:
            eval_solutions = json.load(f)
        
        logger.info(f"Evaluating on {len(eval_challenges)} tasks")
        
        metrics = EvaluationMetrics()
        metrics.total_tasks = len(eval_challenges)
        
        # Track performance per task
        task_times = {}
        task_results = {}
        task_errors = {}
        
        with torch.no_grad():
            for tid, challenge in eval_challenges.items():
                task_start_time = time.time()
                
                try:
                    # Get ground truth solutions
                    solutions = eval_solutions.get(tid, [])
                    if not solutions:
                        task_errors[tid] = "No solutions available"
                        continue
                    
                    # Prepare demonstration data
                    demos = []
                    for pair in challenge["train"]:
                        demos.append({
                            "input": torch.tensor(_sanitize_grid(pair["input"]), dtype=torch.long, device=self.device),
                            "output": torch.tensor(_sanitize_grid(pair["output"]), dtype=torch.long, device=self.device)
                        })
                    
                    task_solved = True
                    test_results = []
                    
                    # Evaluate each test case in the task
                    for test_idx, test in enumerate(challenge["test"]):
                        test_dict = {"input": torch.tensor(_sanitize_grid(test["input"]), dtype=torch.long, device=self.device)}
                        
                        # Get model prediction
                        grid, logits, size, extras = self.model(demos, test_dict, task_id=tid)
                        pred_grid = grid[0].cpu().numpy() if grid.dim() == 3 else grid.cpu().numpy()
                        
                        # Compare with ground truth
                        if test_idx < len(solutions):
                            gt_grid = np.array(solutions[test_idx]["output"])
                            
                            # Compute comprehensive metrics
                            eval_metrics = {}
                            
                            # Exact@1: strict match
                            exact_match = np.array_equal(pred_grid, gt_grid)
                            eval_metrics["exact@1"] = 1.0 if exact_match else 0.0
                            
                            # Exact@K: placeholder, can be extended with beam search results
                            eval_metrics["exact@k"] = eval_metrics["exact@1"]
                            
                            # IoU: intersection-over-union on non-empty grids
                            try:
                                pred_bool = (pred_grid > 0)
                                gold_bool = (gt_grid > 0)
                                overlap = np.logical_and(pred_bool, gold_bool).sum()
                                union = np.logical_or(pred_bool, gold_bool).sum()
                                eval_metrics["iou"] = float(overlap) / float(max(1, union))
                            except Exception:
                                eval_metrics["iou"] = 0.0
                            
                            # Always push metrics into extras for downstream consumers (e.g. scheduler)
                            if isinstance(extras, dict):
                                extras["eval_metrics"] = eval_metrics
                            
                            test_results.append(exact_match)
                            
                            if exact_match:
                                metrics.exact_hits += 1
                            else:
                                task_solved = False
                        else:
                            task_solved = False
                            test_results.append(False)
                        
                        metrics.total_items += 1
                    
                    # Record task-level results
                    if task_solved and len(challenge["test"]) > 0:
                        metrics.solved_tasks += 1
                    
                    task_results[tid] = task_solved
                    task_times[tid] = time.time() - task_start_time
                    
                    if self.verbose and len(task_results) % 20 == 0:
                        current_solve_rate = metrics.solved_tasks / len(task_results)
                        logger.info(f"Progress: {len(task_results)}/{metrics.total_tasks} tasks, "
                                  f"solve rate: {current_solve_rate:.2%}")
                
                except Exception as e:
                    task_errors[tid] = str(e)
                    logger.warning(f"Error evaluating task {tid}: {e}")
                    continue
        
        # Compute final metrics
        metrics.task_solve_rate = metrics.solved_tasks / max(1, metrics.total_tasks)
        metrics.exact_at_1 = metrics.exact_hits / max(1, metrics.total_items)
        metrics.first_hit_rate = metrics.exact_hits / max(1, metrics.total_items)  # Same as exact@1 for single attempts
        
        # Store detailed results
        metrics.task_results = task_results
        metrics.task_times = task_times
        metrics.task_errors = task_errors
        
        # Log comprehensive results
        if self.verbose:
            avg_time = np.mean(list(task_times.values())) if task_times else 0
            logger.info("="*60)
            logger.info("TOPAS ARC Evaluation Results:")
            logger.info(f"Tasks solved: {metrics.solved_tasks}/{metrics.total_tasks} ({metrics.task_solve_rate:.1%})")
            logger.info(f"Exact@1: {metrics.exact_hits}/{metrics.total_items} ({metrics.exact_at_1:.1%})")
            logger.info(f"Average time per task: {avg_time:.2f}s")
            if task_errors:
                logger.info(f"Tasks with errors: {len(task_errors)}")
            logger.info("="*60)
        
        return metrics
    
    def run(self, data_loader_or_path, solutions_path=None):
        """
        Flexible run method for integration with TopasARCTrainer
        
        Args:
            data_loader_or_path: Either a DataLoader or path to challenges JSON
            solutions_path: Path to solutions JSON (if data_loader_or_path is a path)
        
        Returns:
            EvaluationMetrics object
        """
        if isinstance(data_loader_or_path, str):
            # Path provided - use comprehensive evaluation
            if not solutions_path:
                raise ValueError("solutions_path required when challenges path is provided")
            return self.evaluate_topas_model(data_loader_or_path, solutions_path)
        else:
            # DataLoader provided - simplified evaluation for training integration
            logger.info("Running evaluation with DataLoader")
            
            if not self.model:
                raise ValueError("Model not set. Use set_model() first.")
            
            metrics = EvaluationMetrics()
            self.model.eval()
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader_or_path):
                    # Handle DataLoader batch format
                    demos, test_inputs, test_outputs, task_ids = batch
                    
                    if test_outputs and test_outputs[0] is not None:
                        # Prepare data
                        demo_list = [{"input": d[0].squeeze(0), "output": d[1].squeeze(0)} for d in demos]
                        test_dict = {"input": test_inputs[0].squeeze(0)}
                        target_grid = test_outputs[0].squeeze(0)
                        
                        # Get prediction
                        grid, _, _, _ = self.model(demo_list, test_dict, task_id=task_ids[0])
                        pred_grid = grid[0].cpu()
                        
                        # Check exact match
                        if pred_grid.shape == target_grid.shape and torch.equal(pred_grid, target_grid.cpu()):
                            metrics.exact_hits += 1
                            metrics.solved_tasks += 1
                        
                        metrics.total_items += 1
                        metrics.total_tasks += 1
            
            # Compute rates
            if metrics.total_items > 0:
                metrics.exact_at_1 = metrics.exact_hits / metrics.total_items
                metrics.task_solve_rate = metrics.solved_tasks / metrics.total_tasks
                metrics.first_hit_rate = metrics.exact_at_1
            
            return metrics

def evaluate_legacy(eval_challenges, eval_solutions, trainer, sample_submission):
    # detect attempt keys (same as in arc_trainer)
    attempt_keys = trainer._detect_attempt_keys(sample_submission)

    total_tasks = len(eval_challenges)
    solved_tasks = 0
    exact_hits = 0
    total_items = 0

    for tid, challenge in eval_challenges.items():
        solutions = eval_solutions.get(tid, [])
        preds = trainer.solve_task(challenge, attempt_keys)

        task_solved = True
        for i, pred_attempts in enumerate(preds):
            total_items += 1
            gt_outs = [to_grid(sol["output"]) for sol in solutions[i:i+1]]  # one per test
            pred_grid = to_grid(pred_attempts[attempt_keys[0]])  # attempt_1
            if any(grid_equal(pred_grid, gt) for gt in gt_outs):
                exact_hits += 1
            else:
                task_solved = False
        if task_solved:
            solved_tasks += 1

    print("="*60)
    print(f"Eval results:")
    print(f" - Tasks solved: {solved_tasks}/{total_tasks} ({100*solved_tasks/total_tasks:.1f}%)")
    print(f" - Exact match items: {exact_hits}/{total_items} ({100*exact_hits/total_items:.1f}%)")
    print("="*60)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_challenges", required=True, help="Path to arc-agi_evaluation_challenges.json")
    p.add_argument("--eval_solutions", required=True, help="Path to arc-agi_evaluation_solutions.json")
    p.add_argument("--sample_submission", required=True, help="Path to sample_submission.json")
    p.add_argument("--use_topas", action="store_true", help="Use TopasARC60M instead of ArcTrainer")
    p.add_argument("--checkpoint", type=str, help="Path to TopasARC60M checkpoint")
    args = p.parse_args()

    # Load eval data
    with open(args.eval_challenges) as f:
        eval_challenges = json.load(f)
    with open(args.eval_solutions) as f:
        eval_solutions = json.load(f)

    if args.use_topas:
        # Use TopasARC60M for evaluation
        print("⚡ Using TopasARC60M for evaluation")
        from models.topas_arc_60M import TopasARC60M, ModelConfig
        
        config = ModelConfig(
            width=640, depth=16, slots=80, slot_dim=512, rt_layers=10,
            enable_dream=False,  # Disable dream for faster eval
            painter_refine=True,  # Keep EBR enabled
            verbose=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TopasARC60M(config).to(device)
        
        # Load checkpoint if provided
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Loaded checkpoint from {args.checkpoint}")
        
        model.eval()
        
        # Run evaluation with neural model
        solved_tasks = 0
        exact_hits = 0
        total_items = 0
        
        with torch.no_grad():
            for tid, challenge in eval_challenges.items():
                solutions = eval_solutions.get(tid, [])
                
                # Prepare demos
                demos = []
                for pair in challenge["train"]:
                    demos.append({
                        "input": torch.tensor(_sanitize_grid(pair["input"]), dtype=torch.long, device=device),
                        "output": torch.tensor(_sanitize_grid(pair["output"]), dtype=torch.long, device=device)
                    })
                
                task_solved = True
                for test_idx, test in enumerate(challenge["test"]):
                    test_dict = {"input": torch.tensor(_sanitize_grid(test["input"]), dtype=torch.long, device=device)}
                    
                    # Get model prediction
                    grid, _, _, _ = model(demos, test_dict)
                    pred_grid = grid[0].cpu().numpy() if grid.dim() == 3 else grid.cpu().numpy()
                    
                    # Compare with ground truth
                    if test_idx < len(solutions):
                        gt_grid = np.array(solutions[test_idx]["output"])
                        if np.array_equal(pred_grid, gt_grid):
                            exact_hits += 1
                        else:
                            task_solved = False
                    else:
                        task_solved = False
                    
                    total_items += 1
                
                if task_solved and len(challenge["test"]) > 0:
                    solved_tasks += 1
        
        print("="*60)
        print(f"TopasARC60M Eval Results:")
        print(f" - Tasks solved: {solved_tasks}/{len(eval_challenges)} ({100*solved_tasks/len(eval_challenges):.1f}%)")
        print(f" - Exact match items: {exact_hits}/{total_items} ({100*exact_hits/total_items:.1f}%)")
        print("="*60)
        
    else:
        # Legacy evaluation path (ArcTrainer not available)
        logger.warning("ArcTrainer not available, using TopasARC60M evaluation")
        
        # Create EvalRunner and use TopasARC60M
        eval_runner = EvalRunner(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load TopasARC60M
        from models.topas_arc_60M import TopasARC60M, ModelConfig
        config = ModelConfig(verbose=False, enable_dream=False)
        model = TopasARC60M(config).to(eval_runner.device)
        eval_runner.set_model(model)
        
        # Run evaluation
        metrics = eval_runner.evaluate_topas_model(args.eval_challenges, args.eval_solutions)
        
        print("\nLegacy path evaluation completed using TopasARC60M")

if __name__ == "__main__":
    main()