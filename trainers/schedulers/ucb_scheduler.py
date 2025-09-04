"""
UCB Task Scheduler - Consolidated Version
Includes both vanilla UCB1 and enhanced version with empowerment, Kuramoto sync, etc.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import math
from dataclasses import dataclass, asdict

# Import sophisticated difficulty estimator
from trainers.schedulers.difficulty_estimator import DifficultyEstimator

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for UCB scheduler parameters"""
    exploration_weight: float = 2.0
    empowerment_weight: float = 0.3
    sync_weight: float = 0.2
    retry_budget: float = 0.15
    verbose: bool = False
    numerical_stability_epsilon: float = 1e-6
    empowerment_normalization_scale: float = 0.1
    kuramoto_coupling_success: float = 0.5
    kuramoto_coupling_failure: float = -0.2
    kuramoto_dt: float = 0.1


@dataclass
class TaskStatistics:
    """Statistics for a single task"""
    task_id: str
    difficulty: float
    num_attempts: int = 0
    num_successes: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    ucb_score: float = float('inf')
    last_attempt_epoch: int = -1
    meta_strategy: str = "exploration"
    empowerment: float = 0.0
    kuramoto_sync: float = 0.0
    # New fields for difficulty estimator integration
    last_confidence_score: Optional[float] = None
    last_eval_metrics: Dict[str, Any] = None
    last_latents: List[torch.Tensor] = None
    # Feature-based difficulty components
    size_difficulty: Optional[float] = None
    symmetry_difficulty: Optional[float] = None
    transform_difficulty: Optional[float] = None
    
    def __post_init__(self):
        if self.last_eval_metrics is None:
            self.last_eval_metrics = {}
        if self.last_latents is None:
            self.last_latents = []


class UCBTaskScheduler:
    """
    Vanilla UCB1 Task Scheduler
    
    Implements the classic Upper Confidence Bound algorithm for task selection.
    Balances exploration and exploitation using the UCB1 formula.
    
    Args:
        config: SchedulerConfig instance with parameters
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.task_stats: Dict[str, TaskStatistics] = {}
        self.total_attempts: int = 0
        
        if self.config.verbose:
            logger.info(f"Initialized UCB scheduler with exploration_weight={self.config.exploration_weight}")
        
    def select_task(self, available_tasks: List[str]) -> str:
        """
        Select next task using UCB1 algorithm
        
        Args:
            available_tasks: List of available task IDs
            
        Returns:
            Selected task ID
            
        Raises:
            ValueError: If no tasks are available
        """
        if not available_tasks:
            raise ValueError("No tasks available for selection")
            
        # Initialize stats for new tasks with safe defaults
        for task_id in available_tasks:
            if task_id not in self.task_stats:
                self.task_stats[task_id] = TaskStatistics(
                    task_id=task_id,
                    difficulty=0.5  # Neutral default difficulty
                )
        
        # Find unvisited tasks first (infinite UCB score)
        unvisited = [t for t in available_tasks if self.task_stats[t].num_attempts == 0]
        if unvisited:
            selected = unvisited[0]
            if self.config.verbose:
                logger.debug(f"Selected unvisited task: {selected}")
            return selected
        
        # Select task with highest UCB score
        best_task = None
        best_score = -float('inf')
        
        for task_id in available_tasks:
            stats = self.task_stats[task_id]
            
            # UCB1 formula: exploitation + exploration
            exploitation = stats.avg_reward
            exploration = self.config.exploration_weight * math.sqrt(
                math.log(max(self.total_attempts, 1)) / max(stats.num_attempts, 1)
            )
            stats.ucb_score = exploitation + exploration
            
            if stats.ucb_score > best_score:
                best_score = stats.ucb_score
                best_task = task_id
        
        if self.config.verbose:
            logger.debug(f"Selected task: {best_task} with UCB score: {best_score:.4f}")
        
        return best_task if best_task else available_tasks[0]
    
    def update_task_stats(self, task_id: str, reward: float, success: bool) -> None:
        """
        Update statistics after attempting a task
        
        Args:
            task_id: ID of the attempted task
            reward: Reward received (0.0 to 1.0)
            success: Whether the task was solved successfully
        """
        # Initialize if new task or if corrupted (dict instead of TaskStatistics)
        if task_id not in self.task_stats or isinstance(self.task_stats[task_id], dict):
            self.task_stats[task_id] = TaskStatistics(
                task_id=task_id,
                difficulty=0.5
            )
        
        stats = self.task_stats[task_id]
        stats.num_attempts += 1
        stats.total_reward += reward
        stats.avg_reward = stats.total_reward / stats.num_attempts
        
        if success:
            stats.num_successes += 1
        
        self.total_attempts += 1
        
        if self.config.verbose:
            success_rate = stats.num_successes / stats.num_attempts
            logger.debug(f"Updated {task_id}: attempts={stats.num_attempts}, "
                        f"success_rate={success_rate:.3f}, avg_reward={stats.avg_reward:.3f}")


class EnhancedUCBTaskScheduler:
    """
    Enhanced UCB Task Scheduler with advanced features
    
    Extends the basic UCB algorithm with:
    - Empowerment metrics (log determinant of latent covariance)
    - Kuramoto synchronization for task coupling
    - Meta-strategy selection (exploration/exploitation/diversity/consolidation)
    - Retry allocation for failed tasks
    - Adaptive difficulty estimation
    
    Args:
        config: SchedulerConfig instance with parameters
        meta_strategies: List of available meta-strategies
    """
    
    def __init__(self, 
                 config: Optional[SchedulerConfig] = None,
                 meta_strategies: Optional[List[str]] = None):
        
        self.config = config or SchedulerConfig()
        self.meta_strategies = meta_strategies or [
            "exploration", "exploitation", "diversity", "consolidation"
        ]
        
        # Core state
        self.task_stats: Dict[str, TaskStatistics] = {}
        self.total_attempts: int = 0
        self.current_epoch: int = 0
        
        # Initialize sophisticated difficulty estimator
        self.difficulty_estimator = DifficultyEstimator(
            target_difficulty=getattr(config, "target_difficulty", 0.6),
            tau_history=getattr(config, "tau_history", 20.0),
            ucb_alpha=getattr(config, "ucb_alpha", 2.0),
            temperature=getattr(config, "confidence_temperature", 1.5),
            verbose=config.verbose
        )
        
        # Kuramoto oscillator state
        self.global_phase: float = 0.0
        self.task_phases: Dict[str, float] = {}
        
        # Empowerment tracking
        self.latent_covariance: Optional[torch.Tensor] = None
        self.empowerment_history: List[float] = []
        
        # Meta-strategy state
        self.current_strategy: str = "exploration"
        self.strategy_performance: Dict[str, float] = {s: 0.0 for s in self.meta_strategies}
        
        if self.config.verbose:
            logger.info(f"Initialized Enhanced UCB scheduler with strategies: {self.meta_strategies}")
        
    def compute_empowerment(self, latent_vectors: Optional[torch.Tensor] = None) -> float:
        """
        Compute empowerment as log determinant of latent covariance matrix
        
        Empowerment E = log det(Cov(z)) measures the diversity of latent representations.
        Higher empowerment indicates more diverse and potentially more informative states.
        
        Args:
            latent_vectors: Tensor of latent vectors [N, D] or None
            
        Returns:
            Empowerment value normalized to [0, 1] range
        """
        # Safe fallback for missing data
        if latent_vectors is None or len(latent_vectors) < 2:
            if self.config.verbose:
                logger.debug("Insufficient latent vectors for empowerment computation, using fallback")
            return 0.0
        
        try:
            # Prepare tensor
            z = latent_vectors.detach().cpu()
            if z.dim() == 1:
                z = z.unsqueeze(0)
            
            if z.shape[0] < 2:
                return 0.0
                
            # Compute covariance matrix
            cov = torch.cov(z.T)
            self.latent_covariance = cov
            
            # Add numerical stability
            cov_stable = cov + torch.eye(cov.shape[0]) * self.config.numerical_stability_epsilon
            
            # Use slogdet for more stable empowerment computation
            sign, logabsdet = torch.slogdet(cov_stable)
            if sign <= 0:
                logger.warning("[Empowerment] Non-positive determinant, returning 0.0")
                empowerment = 0.0
            else:
                empowerment = logabsdet.item()
            
            # Normalize to [0, 1] range using sigmoid
            empowerment_normalized = torch.sigmoid(
                torch.tensor(empowerment * self.config.empowerment_normalization_scale)
            ).item()
            
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"Empowerment computation failed: {e}, using fallback")
            empowerment_normalized = 0.0
        
        self.empowerment_history.append(empowerment_normalized)
        
        # Keep history bounded
        if len(self.empowerment_history) > 1000:
            self.empowerment_history = self.empowerment_history[-1000:]
        
        return empowerment_normalized
    
    def update_kuramoto_sync(self, task_id: str, success: bool) -> float:
        """
        Update Kuramoto oscillator synchronization
        
        Models tasks as coupled oscillators where success promotes synchronization
        and failures cause desynchronization. This helps balance task selection.
        
        Args:
            task_id: Task identifier
            success: Whether the task attempt was successful
            
        Returns:
            Current synchronization order parameter [0, 1]
        """
        # Initialize phase for new tasks (use torch for reproducibility if seeded)
        if task_id not in self.task_phases:
            if torch.initial_seed() is not None:
                self.task_phases[task_id] = torch.rand(1).item() * 2 * np.pi
            else:
                self.task_phases[task_id] = np.random.uniform(0, 2 * np.pi)
        
        # Natural frequency based on task difficulty (safer fallback)
        stats = self.task_stats.get(task_id)
        omega = 1.0 - stats.difficulty if stats else 0.5
        
        # Coupling strength based on success/failure
        K = (self.config.kuramoto_coupling_success if success 
             else self.config.kuramoto_coupling_failure)
        
        # Update phase using Kuramoto dynamics
        phase_diff = self.global_phase - self.task_phases[task_id]
        self.task_phases[task_id] += self.config.kuramoto_dt * (omega + K * np.sin(phase_diff))
        self.task_phases[task_id] = self.task_phases[task_id] % (2 * np.pi)
        
        # Update global phase as circular mean
        if self.task_phases:
            phases = np.array(list(self.task_phases.values()))
            complex_phases = np.exp(1j * phases)
            self.global_phase = np.angle(np.mean(complex_phases))
        
        # Compute synchronization order parameter
        if len(self.task_phases) > 1:
            phases = np.array(list(self.task_phases.values()))
            sync = np.abs(np.mean(np.exp(1j * phases)))
            
            if self.config.verbose:
                logger.debug(f"Kuramoto sync for {task_id}: {sync:.3f}")
            
            return sync
        return 0.0
    
    def select_meta_strategy(self) -> str:
        """
        Select meta-strategy based on recent performance and context
        
        Chooses between exploration, exploitation, diversity, and consolidation
        based on current performance metrics and training progress.
        
        Returns:
            Selected meta-strategy name
        """
        scores = {}
        
        for strategy in self.meta_strategies:
            base_score = self.strategy_performance.get(strategy, 0.0)
            
            # Context-specific bonuses with safe fallbacks
            if strategy == "exploration" and self.total_attempts < 100:
                base_score += 0.3
            elif strategy == "exploitation" and self.total_attempts > 200:
                base_score += 0.2
            elif strategy == "diversity":
                # Bonus for low empowerment (need more diversity)
                if self.empowerment_history:
                    recent_emp = np.mean(self.empowerment_history[-10:])
                    if recent_emp < 0.3:
                        base_score += 0.4
            elif strategy == "consolidation":
                # Bonus when many tasks are partially solved
                if self.task_stats:
                    partial_solved = sum(1 for s in self.task_stats.values() 
                                       if 0.3 < s.avg_reward < 0.7)
                    if partial_solved > 5:
                        base_score += 0.3
            
            scores[strategy] = base_score
        
        # Select strategy using softmax probabilities
        if scores:
            strategies = list(scores.keys())
            values = np.array(list(scores.values()))
            
            # Softmax with optional temperature for smoother strategy selection
            temp = getattr(self.config, "meta_temp", 1.0)
            scaled = (values - np.max(values)) / max(temp, 1e-6)
            exp_values = np.exp(scaled)
            probs = exp_values / (np.sum(exp_values) + 1e-10)
            
            self.current_strategy = np.random.choice(strategies, p=probs)
        else:
            self.current_strategy = "exploration"  # Safe fallback
        
        if self.config.verbose:
            logger.debug(f"Selected meta-strategy: {self.current_strategy}")
        
        return self.current_strategy
    
    def estimate_difficulty(self, task_id: str, task_features: Optional[Dict[str, Any]] = None) -> float:
        """
        Estimate task difficulty from historical performance and features
        
        Combines historical success rate with feature-based estimates for robust
        difficulty assessment with safe fallbacks.
        
        Args:
            task_id: Task identifier
            task_features: Optional dictionary of task features
            
        Returns:
            Difficulty estimate in [0, 1] range
        """
        # Historical estimate with safe fallbacks
        if task_id in self.task_stats:
            stats = self.task_stats[task_id]
            if stats.num_attempts > 0:
                success_rate = stats.num_successes / stats.num_attempts
                historical_difficulty = 1.0 - success_rate
            else:
                historical_difficulty = 0.5  # Neutral default
        else:
            historical_difficulty = 0.5  # Safe default
        
        # Feature-based estimate with safe fallbacks
        if task_features:
            feature_difficulty = 0.5  # Base difficulty
            
            try:
                # Adjust based on task properties
                if 'num_objects' in task_features:
                    obj_count = min(task_features.get('num_objects', 0), 10)  # Cap at 10
                    feature_difficulty += 0.05 * obj_count
                
                if 'grid_size' in task_features:
                    size = task_features.get('grid_size', 8)
                    feature_difficulty += 0.02 * max(0, size - 8)  # Penalty for large grids
                
                if 'transformation_type' in task_features:
                    hard_transforms = {'rotation', 'reflection', 'complex', 'composition'}
                    if task_features.get('transformation_type') in hard_transforms:
                        feature_difficulty += 0.2
                
                # Weighted combination
                difficulty = 0.7 * historical_difficulty + 0.3 * feature_difficulty
                
            except Exception as e:
                if self.config.verbose:
                    logger.warning(f"Feature-based difficulty estimation failed: {e}")
                difficulty = historical_difficulty
        else:
            difficulty = historical_difficulty
        
        return np.clip(difficulty, 0.0, 1.0)
    
    def select_task(self, 
                   available_tasks: List[str], 
                   latent_vectors: Optional[torch.Tensor] = None,
                   task_features: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """
        Select next task using enhanced UCB with all features
        
        Combines UCB exploration/exploitation with empowerment, Kuramoto sync,
        meta-strategy selection, and retry allocation for robust task selection.
        
        Args:
            available_tasks: List of available task IDs
            latent_vectors: Optional latent representations for empowerment
            task_features: Optional feature dictionary for tasks
            
        Returns:
            Selected task ID
            
        Raises:
            ValueError: If no tasks are available
        """
        if not available_tasks:
            raise ValueError("No tasks available for selection")
        
        # Update state
        self.current_epoch += 1
        
        # Compute empowerment with safe fallback
        empowerment = self.compute_empowerment(latent_vectors)
        
        # Select meta-strategy
        strategy = self.select_meta_strategy()
        
        # Initialize stats for new tasks with safe defaults
        for task_id in available_tasks:
            if task_id not in self.task_stats:
                features = (task_features or {}).get(task_id)
                difficulty = self.estimate_difficulty(task_id, features)
                
                self.task_stats[task_id] = TaskStatistics(
                    task_id=task_id,
                    difficulty=difficulty,
                    meta_strategy=strategy,
                    empowerment=empowerment
                )
        
        # Check retry budget for previously failed tasks
        retry_candidates = []
        for task_id in available_tasks:
            stats = self.task_stats[task_id]
            if (stats.num_attempts > 0 and 
                stats.avg_reward < 0.5 and
                self.current_epoch - stats.last_attempt_epoch > 10):
                retry_candidates.append(task_id)
        
        # Retry allocation with probability
        if retry_candidates and np.random.random() < self.config.retry_budget:
            selected = np.random.choice(retry_candidates)
            if self.config.verbose:
                logger.debug(f"Selected retry task: {selected}")
            return selected
        
        # Find best task using enhanced UCB
        best_task = None
        best_score = -float('inf')
        
        for task_id in available_tasks:
            stats = self.task_stats[task_id]
            
            # Gather feature difficulty samples for this task
            feat_samples = []
            if stats.size_difficulty is not None:
                feat_samples.append(stats.size_difficulty)
            if stats.symmetry_difficulty is not None:
                feat_samples.append(stats.symmetry_difficulty)
            if stats.transform_difficulty is not None:
                feat_samples.append(stats.transform_difficulty)
            
            # Extract from task_features if available
            if task_features and task_id in task_features:
                tf = task_features[task_id]
                for key in ["size_difficulty", "symmetry_difficulty", "transform_difficulty"]:
                    if key in tf and tf[key] is not None:
                        feat_samples.append(tf[key])
            
            # Use difficulty estimator for sophisticated scoring
            scores = self.difficulty_estimator.difficulty(
                task_id,
                feature_difficulty_samples=feat_samples,
                heuristic_confidence=stats.last_confidence_score
            )
            
            # Update stats with computed difficulty
            stats.difficulty = scores["difficulty"]
            stats.empowerment = empowerment
            stats.kuramoto_sync = self.task_phases.get(task_id, 0.0)
            
            # Use difficulty estimator's score as base
            score = scores["score"]
            
            # Add strategy-specific adjustments
            if strategy == "exploitation":
                # Boost nearly-mastered tasks
                if scores["mastery_p"] > 0.7:
                    score *= 1.2
            elif strategy == "diversity":
                # Extra exploration bonus
                score += 0.2 * scores["ucb_bonus"]
            elif strategy == "consolidation":
                # Kuramoto sync bonus
                task_phase = self.task_phases.get(task_id, 0.0)
                sync_bonus = self.config.sync_weight * np.cos(task_phase - self.global_phase)
                score += sync_bonus
                
                # Prefer partially solved tasks
                if 0.3 < scores["mastery_p"] < 0.7:
                    score *= 1.1
            
            # Store for debugging
            stats.ucb_score = score
            
            if score > best_score:
                best_score = score
                best_task = task_id
        
        selected = best_task if best_task else available_tasks[0]
        
        if self.config.verbose:
            logger.debug(f"Selected task: {selected} with enhanced UCB score: {best_score:.4f} "
                        f"(strategy: {strategy})")
        
        return selected
    
    def update_task_stats(self, 
                         task_id: str, 
                         reward: float, 
                         success: bool,
                         latent_vector: Optional[torch.Tensor] = None,
                         extras: Optional[Dict[str, Any]] = None) -> None:
        """
        Update comprehensive statistics after attempting a task
        
        Updates UCB statistics, difficulty estimates, Kuramoto sync,
        and meta-strategy performance with safe fallbacks.
        
        Args:
            task_id: Canonical task/bucket id
            reward: Scalar reward used by bandit layer
            success: Exact@1 hit for this attempt (strict)
            latent_vector: Optional embedding for empowerment/diversity bookkeeping
            extras: Optional dict carrying:
                - eval_metrics: {"exact@1": float, "exact@k": float, "iou": float}
                - confidence_score: float (heuristic signal, e.g., DSL/EBR)
                - any other diagnostics (ebr_deltas, timings, etc.)
        """
        # Initialize if new task or if corrupted (dict instead of TaskStatistics)
        if task_id not in self.task_stats or isinstance(self.task_stats[task_id], dict):
            self.task_stats[task_id] = TaskStatistics(
                task_id=task_id,
                difficulty=0.5
            )
        
        stats = self.task_stats[task_id]
        stats.num_attempts += 1
        stats.total_reward += reward
        stats.avg_reward = stats.total_reward / stats.num_attempts
        stats.last_attempt_epoch = self.current_epoch
        
        if success:
            stats.num_successes += 1
        
        # Keep a small FIFO of latents for empowerment/novelty
        if latent_vector is not None:
            try:
                stats.last_latents.append(latent_vector.detach().cpu()[:64])
                if len(stats.last_latents) > 16:
                    stats.last_latents.pop(0)
            except Exception:
                pass  # bookkeeping only
        
        # --- Honest metrics integration (separate accuracy vs. heuristics) ---
        ex = extras or {}
        eval_metrics = ex.get("eval_metrics") or {}
        exact1 = eval_metrics.get("exact@1", None)
        
        # Update the Bayesian mastery tracker (Bernoulli) used by the scheduler
        self.difficulty_estimator.update_from_eval(task_id, exact1)
        
        # Stash for visibility / later selection (do NOT treat as accuracy)
        stats.last_confidence_score = ex.get("confidence_score", None)
        stats.last_eval_metrics = eval_metrics
        
        # Update adaptive difficulty estimate (keep legacy for compatibility)
        if stats.num_attempts > 0:
            success_rate = stats.num_successes / stats.num_attempts
            # Exponential moving average for stability
            stats.difficulty = 0.7 * stats.difficulty + 0.3 * (1.0 - success_rate)
        
        # Update Kuramoto synchronization
        try:
            sync = self.update_kuramoto_sync(task_id, success)
            stats.kuramoto_sync = sync
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"Kuramoto sync update failed: {e}")
            stats.kuramoto_sync = 0.0
        
        # Update meta-strategy performance with safe fallback
        current_strategy = getattr(self, 'current_strategy', 'exploration')
        if current_strategy in self.strategy_performance:
            # Exponential moving average
            self.strategy_performance[current_strategy] *= 0.9
            self.strategy_performance[current_strategy] += 0.1 * reward
        
        self.total_attempts += 1
        
        if self.config.verbose:
            success_rate = stats.num_successes / stats.num_attempts
            logger.debug(f"Updated {task_id}: attempts={stats.num_attempts}, "
                        f"success_rate={success_rate:.3f}, difficulty={stats.difficulty:.3f}, "
                        f"avg_reward={stats.avg_reward:.3f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive scheduler statistics
        
        Returns:
            Dictionary with all scheduler state and statistics
        """
        recent_empowerment = (np.mean(self.empowerment_history[-10:]) 
                            if self.empowerment_history else 0.0)
        
        stats = {
            "total_attempts": self.total_attempts,
            "current_epoch": self.current_epoch,
            "current_strategy": self.current_strategy,
            "global_phase": float(self.global_phase),
            "recent_empowerment": float(recent_empowerment),
            "empowerment_history_length": len(self.empowerment_history),
            "num_tracked_tasks": len(self.task_stats),
            "task_stats": {k: asdict(v) for k, v in self.task_stats.items()},
            "strategy_performance": dict(self.strategy_performance),
            "config": asdict(self.config)
        }
        return stats
    
    def save_state(self, filepath: str) -> None:
        """
        Save scheduler state to JSON file
        
        Args:
            filepath: Path to save state file
        """
        try:
            state = self.get_statistics()
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            if self.config.verbose:
                logger.info(f"Saved scheduler state to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")
    
    def load_state(self, filepath: str) -> None:
        """
        Load scheduler state from JSON file
        
        Args:
            filepath: Path to state file
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore basic state
            self.total_attempts = state.get("total_attempts", 0)
            self.current_epoch = state.get("current_epoch", 0)
            self.current_strategy = state.get("current_strategy", "exploration")
            self.global_phase = state.get("global_phase", 0.0)
            self.strategy_performance = state.get("strategy_performance", 
                                                  {s: 0.0 for s in self.meta_strategies})
            
            # Reconstruct task statistics
            self.task_stats = {}
            task_stats_data = state.get("task_stats", {})
            for task_id, stats_dict in task_stats_data.items():
                try:
                    self.task_stats[task_id] = TaskStatistics(**stats_dict)
                except Exception as e:
                    if self.config.verbose:
                        logger.warning(f"Failed to restore stats for task {task_id}: {e}")
            
            if self.config.verbose:
                logger.info(f"Loaded scheduler state from {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")


# Export the enhanced version as the default
UCBScheduler = EnhancedUCBTaskScheduler

__all__ = [
    "UCBTaskScheduler",
    "EnhancedUCBTaskScheduler", 
    "UCBScheduler",
    "TaskStatistics",
    "SchedulerConfig"
]