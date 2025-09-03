# schedulers/difficulty_estimator.py
"""
Sophisticated difficulty estimation for ARC Prize curriculum learning.
Combines Bayesian mastery tracking, calibrated confidence, robust statistics,
and exploration bonuses to achieve optimal task scheduling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

EPS = 1e-8

def isfinite(x: float) -> bool:
    """Check if value is finite (not NaN/inf)"""
    try:
        return np.isfinite(x)
    except Exception:
        return False

def safe_float(x: Optional[float], default: float = 0.5) -> float:
    """Convert to float with NaN safety"""
    if x is None: 
        return default
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else default
    except Exception:
        return default

def robust_mean_nan(values: List[float], default: float = 0.5) -> float:
    """Median-of-means style robust mean with NaN safety."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return default
    
    # Number of blocks ~ sqrt(n) for optimal robustness
    k = int(max(1, round(math.sqrt(arr.size))))
    try:
        blocks = np.array_split(np.random.permutation(arr), k)
        means = np.array([b.mean() for b in blocks if b.size > 0], dtype=float)
        if means.size == 0:
            return default
        return float(np.median(means))
    except Exception as e:
        logger.warning(f"Robust mean failed: {e}, using simple mean")
        return float(np.nan_to_num(arr.mean(), nan=default))

class DebiasedEMA:
    """Adam-style bias-corrected EMA, NaN-safe, for progress signals."""
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.t = 0
        self.m = 0.0
    
    def update(self, x: Optional[float]) -> float:
        """Update EMA with new value, skipping NaNs"""
        if x is None or not isfinite(x):  # skip NaNs
            return self.value()
        self.t += 1
        self.m = self.beta * self.m + (1 - self.beta) * x
        # bias correction
        denom = 1 - (self.beta ** self.t)
        return self.m / max(EPS, denom)
    
    def value(self) -> float:
        """Get current debiased value"""
        if self.t == 0: 
            return 0.0
        return self.m / max(EPS, 1 - (self.beta ** self.t))

@dataclass
class BetaPosterior:
    """Bernoulli success posterior for Exact@1."""
    alpha: float = 1.0  # Prior successes + 1
    beta: float = 1.0   # Prior failures + 1
    trials: int = 0
    successes: int = 0
    
    def update(self, success: Optional[bool]) -> None:
        """Update posterior with new trial result"""
        if success is None:
            return
        self.trials += 1
        if success:
            self.successes += 1
            self.alpha += 1.0
        else:
            self.beta += 1.0
    
    @property
    def mean(self) -> float:
        """Expected success probability"""
        return self.alpha / max(EPS, (self.alpha + self.beta))
    
    @property
    def var(self) -> float:
        """Posterior variance"""
        a, b = self.alpha, self.beta
        denom = (a + b)**2 * (a + b + 1)
        return (a * b) / max(EPS, denom)
    
    @property
    def std(self) -> float:
        """Posterior standard deviation"""
        return math.sqrt(self.var)

class TemperatureScaler:
    """Binary temperature scaling for probabilities; NaN-safe."""
    def __init__(self, T: float = 1.0):
        self.T = max(EPS, T)
    
    @staticmethod
    def _logit(p: float) -> float:
        """Safe logit transform"""
        p = min(1.0 - EPS, max(EPS, p))
        return math.log(p / (1 - p))
    
    def calibrate(self, p: Optional[float]) -> float:
        """Apply temperature calibration to probability"""
        p = safe_float(p, 0.5)
        z = self._logit(p) / self.T
        # sigmoid(z)
        return 1.0 / (1.0 + math.exp(-z))

@dataclass
class TaskDifficultyState:
    """Per-task state for difficulty estimation"""
    beta_post: BetaPosterior = field(default_factory=BetaPosterior)
    acc_ema: DebiasedEMA = field(default_factory=lambda: DebiasedEMA(beta=0.9))
    lp_ema: DebiasedEMA = field(default_factory=lambda: DebiasedEMA(beta=0.8))
    last_acc: float = 0.0
    # diagnostics
    nan_hits: int = 0
    total_updates: int = 0

class DifficultyEstimator:
    """
    Combines Bayesian mastery (Exact@1), calibrated confidence, robust feature priors,
    and UCB-style exploration. NaN-safe end-to-end.
    
    Key principles:
    1. Curriculum learning: Target "Goldilocks" difficulty zone
    2. Calibrated confidence: Temperature-scaled heuristics
    3. Bayesian tracking: Beta posterior for success probability
    4. Robust aggregation: Median-of-means for outlier resistance
    5. Exploration bonus: UCB with posterior variance
    """
    
    def __init__(self, 
                 target_difficulty: float = 0.6,
                 tau_history: float = 20.0,
                 ucb_alpha: float = 2.0,
                 temperature: float = 1.5,
                 verbose: bool = False):
        """
        Args:
            target_difficulty: Optimal difficulty for curriculum (0.6 = learnable edge)
            tau_history: How quickly history dominates priors (higher = slower)
            ucb_alpha: Exploration pressure (higher = more exploration)
            temperature: Confidence calibration strength (higher = flatter)
            verbose: Enable debug logging
        """
        self.target_difficulty = target_difficulty
        self.tau_history = tau_history
        self.ucb_alpha = ucb_alpha
        self.verbose = verbose
        self.scaler = TemperatureScaler(T=temperature)
        self.state: Dict[str, TaskDifficultyState] = {}
        self.global_trials = 0

    def _get_state(self, task_id: str) -> TaskDifficultyState:
        """Get or create state for task"""
        if task_id not in self.state:
            self.state[task_id] = TaskDifficultyState()
        return self.state[task_id]

    def update_from_eval(self, task_id: str, exact1: Optional[float]) -> None:
        """
        Update mastery from ground-truth Exact@1 evaluation.
        This is the ONLY place where real accuracy enters the system.
        """
        st = self._get_state(task_id)
        st.total_updates += 1
        
        if exact1 is None or not isfinite(exact1):
            st.nan_hits += 1
            if self.verbose:
                logger.warning(f"[DE] {task_id}: NaN in exact@1 (hit {st.nan_hits}/{st.total_updates})")
            return
        
        # Threshold for "success" (can be tuned)
        success = (exact1 >= 0.9999)
        st.beta_post.update(success)
        
        # Update EMAs for learning progress
        acc_now = st.beta_post.mean
        delta_acc = acc_now - st.last_acc
        st.lp_ema.update(delta_acc)
        st.acc_ema.update(acc_now)
        st.last_acc = acc_now
        
        self.global_trials += 1
        
        if self.verbose:
            logger.info(f"[DE] {task_id}: exact@1={exact1:.3f} success={success} "
                       f"mastery={acc_now:.3f} lp={st.lp_ema.value():.3f}")

    def difficulty(self, 
                   task_id: str,
                   feature_difficulty_samples: List[float],
                   heuristic_confidence: Optional[float] = None) -> Dict[str, float]:
        """
        Compute difficulty and scheduling score for a task.
        
        Args:
            task_id: Task identifier
            feature_difficulty_samples: List of feature-based difficulty estimates
            heuristic_confidence: Optional confidence from DSL/EBR/Painter (NOT accuracy!)
            
        Returns:
            Dict with keys:
                difficulty: ∈ [0,1] — robust, NaN-safe difficulty estimate
                mastery_p: ∈ [0,1] — expected success probability
                ucb_bonus: ≥ 0 — exploration bonus
                score: — scheduler index (higher = schedule sooner)
        """
        st = self._get_state(task_id)

        # 1) Bayesian mastery (from Beta posterior)
        p_mean = st.beta_post.mean
        diff_from_mastery = 1.0 - p_mean  # difficulty = 1 - E[p]

        # 2) Robust feature prior (median-of-means)
        feat_diff = robust_mean_nan(feature_difficulty_samples, default=0.5)

        # 3) Calibrated heuristic confidence (if available)
        conf_cal = None
        if heuristic_confidence is not None:
            # Convert confidence to difficulty (1 - confidence)
            conf_cal = 1.0 - self.scaler.calibrate(heuristic_confidence)

        # 4) Reliability-aware weighting
        # History weight grows with trials; features/conf fill gaps early
        n = max(0, st.beta_post.trials)
        w_hist = n / (n + self.tau_history)
        w_feat = (1.0 - w_hist) * 0.7
        w_conf = (1.0 - w_hist) * 0.3 if conf_cal is not None else 0.0

        # Build weighted combination
        parts, weights = [], []
        parts.append(diff_from_mastery)
        weights.append(w_hist)
        parts.append(feat_diff)
        weights.append(w_feat)
        if conf_cal is not None:
            parts.append(conf_cal)
            weights.append(w_conf)

        # NaN-safe weighted combination
        vals = [safe_float(v, 0.5) for v in parts]
        ws = [safe_float(w, 0.0) for w in weights]
        
        if sum(ws) < EPS:
            difficulty = 0.5
        else:
            difficulty = sum(v * w for v, w in zip(vals, ws)) / sum(ws)

        # 5) Exploration bonus (UCB-V using Beta variance)
        t_total = max(1, self.global_trials)
        var = st.beta_post.var
        ucb_bonus = math.sqrt(
            max(0.0, self.ucb_alpha * math.log(t_total + 1.0) * max(var, EPS) / max(1.0, n))
        )

        # 6) Curriculum targeting: favor "Goldilocks" difficulty
        goldilocks = 1.0 - abs(difficulty - self.target_difficulty)  # ∈ [0,1]

        # 7) Learning progress (EMA of delta in mastery)
        lp = st.lp_ema.value()  # can be negative
        lp = max(0.0, lp)  # clamp to non-negative

        # 8) Final scheduling score (weights are tunable)
        score = 0.45 * goldilocks + 0.35 * ucb_bonus + 0.20 * lp

        if self.verbose:
            logger.info(f"[DE] {task_id}: diff={difficulty:.3f} gold={goldilocks:.3f} "
                       f"ucb={ucb_bonus:.3f} lp={lp:.3f} score={score:.3f}")

        return {
            "difficulty": float(np.clip(difficulty, 0.0, 1.0)),
            "mastery_p": float(np.clip(p_mean, 0.0, 1.0)),
            "ucb_bonus": float(max(0.0, ucb_bonus)),
            "score": float(score),
            "goldilocks": float(goldilocks),
            "learning_progress": float(lp),
            "trials": n,
            "nan_rate": st.nan_hits / max(1, st.total_updates)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get global statistics for monitoring"""
        stats = {
            "total_tasks": len(self.state),
            "global_trials": self.global_trials,
            "avg_mastery": np.mean([s.beta_post.mean for s in self.state.values()]) if self.state else 0.0,
            "avg_trials": np.mean([s.beta_post.trials for s in self.state.values()]) if self.state else 0.0,
            "total_nan_hits": sum(s.nan_hits for s in self.state.values()),
        }
        return stats