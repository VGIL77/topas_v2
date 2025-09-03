"""
Meta-Learning Implementation for TOPAS ARC Solver
MAML/Reptile-style meta-learning for few-shot adaptation on ARC tasks

Core objective: Enable model to adapt to new ARC tasks in 1-3 gradient updates
North Star: Few-shot generalization on unseen task compositions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from collections import OrderedDict
import numpy as np
from dataclasses import dataclass

# Import TOPAS model and related components
from models.topas_arc_60M import TopasARC60M, ModelConfig
from models.policy_nets import OpPolicyNet, PolicyPrediction
from arc_dataset_loader import ARCDataset

@dataclass
class Episode:
    """Single meta-learning episode with support and query sets"""
    task_id: str
    support_set: List[Dict]  # Few demos for adaptation
    query_set: List[Dict]    # Held-out examples for evaluation
    difficulty: float        # Estimated task difficulty 0.0-1.0
    composition_type: str    # e.g., "rotation", "color_map", "composite"

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning system"""
    # MAML parameters
    inner_lr: float = 0.01          # Learning rate for inner loop adaptation
    outer_lr: float = 0.001         # Learning rate for meta-optimization
    inner_steps: int = 2            # Number of adaptation steps
    first_order: bool = True        # Use first-order MAML (more efficient)
    
    # Reptile parameters  
    reptile_epsilon: float = 0.1    # Reptile interpolation rate
    reptile_inner_steps: int = 5    # Steps for Reptile inner training
    
    # Episode configuration
    support_size: int = 3           # Number of support examples
    query_size: int = 2             # Number of query examples
    meta_batch_size: int = 16       # Number of tasks per meta-batch
    
    # Training configuration
    max_meta_epochs: int = 1000     # Maximum meta-training epochs
    evaluation_freq: int = 100      # How often to evaluate few-shot performance
    
    # Curriculum learning
    start_difficulty: float = 0.2   # Start with easier tasks
    end_difficulty: float = 1.0     # Progress to harder tasks
    difficulty_schedule: str = "linear"  # "linear", "exponential", "cosine"
    
    # Policy network integration
    adapt_policy_net: bool = True   # Include policy network in adaptation
    policy_adaptation_weight: float = 0.5  # Weight for policy loss in adaptation
    
    # Device and precision
    device: str = "cuda"
    mixed_precision: bool = True    # Use automatic mixed precision

class MetaLearner(nn.Module):
    """
    MAML/Reptile implementation for ARC few-shot learning
    
    Enables rapid adaptation to new ARC tasks through meta-learning.
    Supports both first-order MAML and Reptile algorithms.
    """
    
    def __init__(self, base_model: TopasARC60M, config: MetaLearningConfig = None):
        super().__init__()
        
        self.config = config or MetaLearningConfig()
        self.base_model = base_model
        
        # Create meta-optimizable copy of the model
        self.meta_model = copy.deepcopy(base_model)
        
        # Meta-optimizer for outer loop updates
        self.meta_optimizer = torch.optim.Adam(
            self.meta_model.parameters(), 
            lr=self.config.outer_lr
        )
        
        # Track meta-learning statistics
        self.meta_stats = {
            "episodes_trained": 0,
            "adaptation_success_rate": 0.0,
            "avg_adaptation_loss": 0.0,
            "few_shot_accuracies": {"1-shot": [], "3-shot": [], "5-shot": []},
            "curriculum_progress": 0.0
        }
        
        # Automatic Mixed Precision scaler
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"[MetaLearner] Initialized with config: inner_lr={config.inner_lr}, "
              f"outer_lr={config.outer_lr}, inner_steps={config.inner_steps}")
    
    def inner_loop(self, task_demos: List[Dict], task_tests: List[Dict], 
                   num_steps: int = None) -> Tuple[torch.nn.Module, Dict[str, float]]:
        """
        Fast adaptation on task demonstrations (inner loop of MAML)
        
        Args:
            task_demos: Support set demonstrations for adaptation
            task_tests: Query set for evaluation during adaptation
            num_steps: Number of adaptation steps (uses config default if None)
            
        Returns:
            Tuple of (adapted_model, adaptation_metrics)
        """
        num_steps = num_steps or self.config.inner_steps
        
        # Clone model for adaptation (creates computational graph for second-order gradients)
        if self.config.first_order:
            # First-order MAML: detach to avoid second-order gradients
            adapted_model = copy.deepcopy(self.meta_model)
            for param in adapted_model.parameters():
                param.detach_()
        else:
            # Full MAML: maintain computational graph
            adapted_model = self._clone_model_with_grad(self.meta_model)
        
        # Create inner optimizer
        inner_optimizer = torch.optim.SGD(
            adapted_model.parameters(), 
            lr=self.config.inner_lr
        )
        
        adaptation_losses = []
        
        # Adaptation steps
        for step in range(num_steps):
            inner_optimizer.zero_grad()
            
            # Compute adaptation loss on support set
            total_loss = 0.0
            valid_demos = 0
            
            for demo in task_demos:
                try:
                    # Forward pass through adapted model
                    if self.config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            pred_grid, pred_logits, pred_size, extras = adapted_model(
                                [demo], {"input": demo["input"]}
                            )
                            loss = self._compute_task_loss(
                                pred_grid, pred_logits, demo["output"]
                            )
                    else:
                        pred_grid, pred_logits, pred_size, extras = adapted_model(
                            [demo], {"input": demo["input"]}
                        )
                        loss = self._compute_task_loss(
                            pred_grid, pred_logits, demo["output"]
                        )
                    
                    total_loss += loss
                    valid_demos += 1
                    
                except Exception as e:
                    print(f"[MetaLearner] Inner loop demo failed: {e}")
                    continue
            
            if valid_demos == 0:
                print("[MetaLearner] Warning: No valid demos in inner loop")
                break
                
            avg_loss = total_loss / valid_demos
            adaptation_losses.append(float(avg_loss))
            
            # Backward pass and update
            if self.config.mixed_precision:
                self.scaler.scale(avg_loss).backward()
                self.scaler.step(inner_optimizer)
                self.scaler.update()
            else:
                avg_loss.backward()
                inner_optimizer.step()
        
        # Compute adaptation metrics
        adaptation_metrics = {
            "adaptation_loss_initial": adaptation_losses[0] if adaptation_losses else float('inf'),
            "adaptation_loss_final": adaptation_losses[-1] if adaptation_losses else float('inf'),
            "adaptation_improvement": (adaptation_losses[0] - adaptation_losses[-1]) if len(adaptation_losses) > 1 else 0.0,
            "valid_adaptation_steps": len(adaptation_losses)
        }
        
        return adapted_model, adaptation_metrics
    
    def outer_loop(self, task_batch: List[Episode]) -> Dict[str, float]:
        """
        Meta-optimization step (outer loop of MAML)
        
        Args:
            task_batch: Batch of episodes for meta-learning
            
        Returns:
            Dictionary of meta-learning metrics
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        valid_tasks = 0
        batch_metrics = {
            "meta_loss": 0.0,
            "adaptation_success_rate": 0.0,
            "avg_query_accuracy": 0.0,
            "policy_adaptation_loss": 0.0
        }
        
        for episode in task_batch:
            try:
                # Inner loop: adapt to task
                adapted_model, adapt_metrics = self.inner_loop(
                    episode.support_set, episode.query_set
                )
                
                # Evaluate adapted model on query set
                query_loss = 0.0
                query_correct = 0
                query_total = 0
                
                for query_demo in episode.query_set:
                    if self.config.mixed_precision:
                        with torch.cuda.amp.autocast():
                            pred_grid, pred_logits, pred_size, extras = adapted_model(
                                episode.support_set[:1],  # Use first support as context
                                {"input": query_demo["input"]}
                            )
                            loss = self._compute_task_loss(
                                pred_grid, pred_logits, query_demo["output"]
                            )
                    else:
                        pred_grid, pred_logits, pred_size, extras = adapted_model(
                            episode.support_set[:1],
                            {"input": query_demo["input"]}
                        )
                        loss = self._compute_task_loss(
                            pred_grid, pred_logits, query_demo["output"]
                        )
                    
                    query_loss += loss
                    
                    # Compute accuracy
                    accuracy = self._compute_accuracy(pred_grid, query_demo["output"])
                    query_correct += accuracy
                    query_total += 1
                
                if query_total > 0:
                    avg_query_loss = query_loss / query_total
                    avg_query_accuracy = query_correct / query_total
                    
                    meta_loss += avg_query_loss
                    valid_tasks += 1
                    
                    # Update batch metrics
                    batch_metrics["avg_query_accuracy"] += avg_query_accuracy
                    
                    # Track adaptation success (improvement > threshold)
                    if adapt_metrics["adaptation_improvement"] > 0.1:
                        batch_metrics["adaptation_success_rate"] += 1.0
                
            except Exception as e:
                print(f"[MetaLearner] Outer loop episode failed: {e}")
                continue
        
        if valid_tasks == 0:
            print("[MetaLearner] Warning: No valid tasks in meta-batch")
            return batch_metrics
        
        # Average meta-loss across tasks
        meta_loss = meta_loss / valid_tasks
        batch_metrics["meta_loss"] = float(meta_loss)
        batch_metrics["adaptation_success_rate"] /= len(task_batch)
        batch_metrics["avg_query_accuracy"] /= valid_tasks
        
        # Meta-gradient step
        if self.config.mixed_precision:
            self.scaler.scale(meta_loss).backward()
            self.scaler.step(self.meta_optimizer)
            self.scaler.update()
        else:
            meta_loss.backward()
            self.meta_optimizer.step()
        
        # Update meta-statistics
        self.meta_stats["episodes_trained"] += len(task_batch)
        self.meta_stats["avg_adaptation_loss"] = 0.9 * self.meta_stats["avg_adaptation_loss"] + 0.1 * float(meta_loss)
        self.meta_stats["adaptation_success_rate"] = 0.9 * self.meta_stats["adaptation_success_rate"] + 0.1 * batch_metrics["adaptation_success_rate"]
        
        return batch_metrics
    
    def adapt(self, support_demos: List[Dict], num_steps: int = None) -> nn.Module:
        """
        Adapt to new task using support demonstrations
        
        Args:
            support_demos: Few-shot demonstrations for adaptation
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model ready for inference
        """
        num_steps = num_steps or self.config.inner_steps
        
        # Use inner loop for adaptation (without query evaluation)
        adapted_model, _ = self.inner_loop(support_demos, [], num_steps)
        
        return adapted_model
    
    def evaluate_few_shot(self, test_tasks: List[Episode], 
                         shot_counts: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        Evaluate few-shot performance on test tasks
        
        Args:
            test_tasks: List of test episodes
            shot_counts: Number of shots to evaluate [1, 3, 5]
            
        Returns:
            Dictionary with few-shot accuracies
        """
        results = {f'{k}-shot': [] for k in shot_counts}
        
        self.meta_model.eval()
        
        with torch.no_grad():
            for episode in test_tasks:
                for k in shot_counts:
                    if len(episode.support_set) >= k:
                        # Use k support examples for adaptation
                        support_subset = episode.support_set[:k]
                        
                        # Adapt model
                        adapted_model = self.adapt(support_subset, num_steps=3)
                        adapted_model.eval()
                        
                        # Evaluate on query set
                        correct = 0
                        total = 0
                        
                        for query in episode.query_set:
                            try:
                                pred_grid, _, _, _ = adapted_model(
                                    support_subset,
                                    {"input": query["input"]}
                                )
                                accuracy = self._compute_accuracy(pred_grid, query["output"])
                                correct += accuracy
                                total += 1
                            except Exception as e:
                                print(f"[MetaLearner] Evaluation failed: {e}")
                                continue
                        
                        if total > 0:
                            results[f'{k}-shot'].append(correct / total)
        
        # Compute averages
        avg_results = {}
        for k in shot_counts:
            if results[f'{k}-shot']:
                avg_results[f'{k}-shot'] = float(np.mean(results[f'{k}-shot']))
                self.meta_stats["few_shot_accuracies"][f'{k}-shot'].append(avg_results[f'{k}-shot'])
            else:
                avg_results[f'{k}-shot'] = 0.0
        
        return avg_results
    
    def _clone_model_with_grad(self, model: nn.Module) -> nn.Module:
        """Clone model while preserving gradient computation graph"""
        cloned_model = copy.deepcopy(model)
        
        # Ensure parameters require gradients and maintain computation graph
        for (name, param), (_, cloned_param) in zip(model.named_parameters(), cloned_model.named_parameters()):
            if param.requires_grad:
                cloned_param.requires_grad_(True)
                # For non-first-order MAML, we need to maintain the computational graph
                if not self.config.first_order and param.grad is not None:
                    cloned_param.grad = param.grad.clone()
        
        return cloned_model
    
    def _compute_task_loss(self, pred_grid: torch.Tensor, pred_logits: torch.Tensor, 
                          target_grid: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss for adaptation"""
        # Ensure target has batch dimension
        if target_grid.dim() == 2:
            target_grid = target_grid.unsqueeze(0)
        
        # Handle size mismatches by cropping/padding
        pred_h, pred_w = pred_grid.shape[-2:]
        target_h, target_w = target_grid.shape[-2:]
        
        if pred_h != target_h or pred_w != target_w:
            # Crop both to smaller size for fair comparison
            min_h = min(pred_h, target_h)
            min_w = min(pred_w, target_w)
            pred_grid = pred_grid[:, :min_h, :min_w]
            target_grid = target_grid[:, :min_h, :min_w]
            
            # Adjust logits accordingly
            pred_logits = pred_logits[:, :min_h*min_w, :]
        
        # Cross-entropy loss on flattened grids
        target_flat = target_grid.reshape(-1).long()
        logits_flat = pred_logits.reshape(-1, pred_logits.size(-1))
        
        loss = F.cross_entropy(logits_flat, target_flat, ignore_index=-1)
        
        return loss
    
    def _compute_accuracy(self, pred_grid: torch.Tensor, target_grid: torch.Tensor) -> float:
        """Compute pixel-level accuracy"""
        if target_grid.dim() == 2:
            target_grid = target_grid.unsqueeze(0)
            
        # Handle size mismatches
        pred_h, pred_w = pred_grid.shape[-2:]
        target_h, target_w = target_grid.shape[-2:]
        
        if pred_h != target_h or pred_w != target_w:
            min_h = min(pred_h, target_h)
            min_w = min(pred_w, target_w)
            pred_grid = pred_grid[:, :min_h, :min_w]
            target_grid = target_grid[:, :min_h, :min_w]
        
        correct = (pred_grid == target_grid).float().mean()
        return float(correct)
    
    def get_meta_stats(self) -> Dict[str, Any]:
        """Get current meta-learning statistics"""
        stats = dict(self.meta_stats)
        
        # Compute recent performance trends
        for shot_count in ["1-shot", "3-shot", "5-shot"]:
            recent_scores = self.meta_stats["few_shot_accuracies"][shot_count][-10:]
            if recent_scores:
                stats[f"recent_{shot_count}_mean"] = float(np.mean(recent_scores))
                stats[f"recent_{shot_count}_std"] = float(np.std(recent_scores))
            else:
                stats[f"recent_{shot_count}_mean"] = 0.0
                stats[f"recent_{shot_count}_std"] = 0.0
        
        return stats
    
    def save_checkpoint(self, filepath: str):
        """Save meta-learning checkpoint"""
        checkpoint = {
            'meta_model_state_dict': self.meta_model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config,
            'meta_stats': self.meta_stats,
            'episodes_trained': self.meta_stats["episodes_trained"]
        }
        
        if self.config.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, filepath)
        print(f"[MetaLearner] Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load meta-learning checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.meta_model.load_state_dict(checkpoint['meta_model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_stats = checkpoint.get('meta_stats', self.meta_stats)
        
        if self.config.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"[MetaLearner] Loaded checkpoint from {filepath}")
        print(f"[MetaLearner] Episodes trained: {self.meta_stats['episodes_trained']}")


def create_meta_learner(base_model: TopasARC60M, config: MetaLearningConfig = None) -> MetaLearner:
    """
    Factory function to create a MetaLearner with proper initialization
    
    Args:
        base_model: Pre-trained TOPAS model to meta-learn from
        config: Meta-learning configuration
        
    Returns:
        Initialized MetaLearner ready for training
    """
    if config is None:
        config = MetaLearningConfig()
    
    meta_learner = MetaLearner(base_model, config)
    
    print(f"[MetaLearner] Created with {sum(p.numel() for p in meta_learner.meta_model.parameters()) / 1e6:.1f}M parameters")
    print(f"[MetaLearner] Inner LR: {config.inner_lr}, Outer LR: {config.outer_lr}")
    print(f"[MetaLearner] Target: 1-shot ≥40%, 3-shot ≥65%, 5-shot ≥80%")
    
    return meta_learner


if __name__ == "__main__":
    # Quick test of MetaLearner
    print("="*60)
    print("TOPAS MetaLearner - Quick Test")
    print("="*60)
    
    # Create base model
    from models.topas_arc_60M import create_model, ModelConfig
    base_model = create_model(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Create meta-learner
    config = MetaLearningConfig(
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=2,
        meta_batch_size=4  # Small for testing
    )
    
    meta_learner = create_meta_learner(base_model, config)
    
    # Create dummy episode for testing
    dummy_demo = {
        'input': torch.randint(0, 10, (8, 8)),
        'output': torch.randint(0, 10, (8, 8))
    }
    
    episode = Episode(
        task_id="test_task",
        support_set=[dummy_demo, dummy_demo],  # 2 support examples
        query_set=[dummy_demo],  # 1 query example
        difficulty=0.5,
        composition_type="test"
    )
    
    # Test inner loop adaptation
    print("\n[TEST] Testing inner loop adaptation...")
    adapted_model, metrics = meta_learner.inner_loop(episode.support_set, episode.query_set)
    print(f"Adaptation metrics: {metrics}")
    
    # Test outer loop meta-update
    print("\n[TEST] Testing outer loop meta-update...")
    meta_metrics = meta_learner.outer_loop([episode])
    print(f"Meta-learning metrics: {meta_metrics}")
    
    print("\n[TEST] Meta-learning system operational!")
    print("="*60)