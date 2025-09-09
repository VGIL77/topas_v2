#!/usr/bin/env python3
"""
DreamEngine: Unified dream system with FSHO oscillator, CIO meta-learning,
NMDA gating, GCCRF curiosity, theme synthesis, and wormhole mining.
"""
import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from nmda_dreaming import NMDAGatedDreaming
from gccrf_curiosity import GCCRFCuriosity
from emergent_theme_synthesis import EmergentThemeSynthesis
from wormhole_offline import WormholeTemplateMiner
from phi_metrics_neuro import phi_synergy_features, kappa_floor, cge_boost
from ripple_substrate import RippleSubstrate, RippleConfig

@dataclass
class DreamMotif:
    """TTL Dream Motif with usage tracking and entropy measures"""
    pattern: torch.Tensor  # The pattern/template
    ttl: int  # Time to live (decreases each cycle)
    usage_count: int = 0
    success_rate: float = 0.0
    entropy_reduction: float = 0.0
    created_time: float = 0.0
    last_used: float = 0.0
    
    def tick(self) -> bool:
        """Decrease TTL and return True if should be kept"""
        self.ttl -= 1
        # Keep if TTL > 0 and success rate is reasonable OR very recent
        keep_condition = (self.ttl > 0 and self.success_rate > 0.3) or \
                        (time.time() - self.created_time < 60.0)  # Keep new motifs for 1 minute
        return keep_condition
        
    def update_success(self, success: bool, entropy_delta: float = 0.0):
        """Update success tracking"""
        self.usage_count += 1
        self.last_used = time.time()
        
        # Exponential moving average for success rate
        alpha = 0.2  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * float(success)
        
        # Track entropy reduction
        if entropy_delta < 0:  # Negative delta means entropy reduced (good)
            self.entropy_reduction = (1 - alpha) * self.entropy_reduction + alpha * abs(entropy_delta)
            
class MetaLearner:
    """Meta-learning for dream strategy selection"""
    def __init__(self):
        self.strategy_success = {}  # strategy → success_rate list
        self.task_features = {}  # task → feature vector
        self.feature_dim = 20  # Fixed feature dimension
        
    def update(self, task_id: str, strategy: str, success: bool, features: torch.Tensor = None):
        """Update strategy success rates and task features"""
        # Update strategy success rates
        if strategy not in self.strategy_success:
            self.strategy_success[strategy] = []
        self.strategy_success[strategy].append(success)
        
        # Keep only recent history
        if len(self.strategy_success[strategy]) > 100:
            self.strategy_success[strategy] = self.strategy_success[strategy][-100:]
            
        # Store task features if provided
        if features is not None and features.numel() > 0:
            # Pad or truncate to fixed dimension
            if features.numel() >= self.feature_dim:
                self.task_features[task_id] = features.flatten()[:self.feature_dim]
            else:
                padded = torch.zeros(self.feature_dim, device=features.device, dtype=features.dtype)
                padded[:features.numel()] = features.flatten()
                self.task_features[task_id] = padded
                
    def recommend_strategy(self, task_features: torch.Tensor = None) -> str:
        """Recommend best strategy based on similarity to successful tasks"""
        if not self.strategy_success:
            return 'default'
            
        best_strategy = 'default'
        best_score = -1
        
        for strategy, successes in self.strategy_success.items():
            if not successes:
                continue
                
            base_success = np.mean(successes)
            
            # Add task similarity bonus if we have features
            similarity_bonus = 0.0
            if task_features is not None and self.task_features:
                # Find most similar successful task
                max_similarity = 0.0
                for task_id, stored_features in self.task_features.items():
                    if stored_features.device != task_features.device:
                        stored_features = stored_features.to(task_features.device)
                    similarity = F.cosine_similarity(
                        task_features.flatten()[:self.feature_dim],
                        stored_features.flatten()[:self.feature_dim],
                        dim=0
                    )
                    max_similarity = max(max_similarity, similarity.item())
                similarity_bonus = max_similarity * 0.1
                
            total_score = base_success + similarity_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_strategy = strategy
                
        return best_strategy

@dataclass
class DreamConfig:
    state_dim: int
    action_dim: int = 4
    device: str = "cpu"  # Explicit device specification with fallback
    # Determinism control
    deterministic: bool = False
    cio_seed: int = 1337
    # Logging control
    verbose: bool = False
    # FSHO params
    fsho_K: float = 0.2
    fsho_eta: float = 0.1
    fsho_alpha: float = 1.6  # Levy alpha (1<α<=2), 2 => Gaussian
    fsho_H: float = 0.7      # Hurst memory (0<H<1)
    fsho_fgn_scale: float = 0.1   # FGN noise scale
    fsho_levy_scale: float = 0.05  # Levy noise scale
    # CIO meta
    cio_lr: float = 0.02
    cio_hist: int = 128
    # NMDA gate
    valence_default: float = 0.7
    arousal_default: float = 0.5
    # Budgets
    micro_ticks: int = 1    # micro-dream steps per forward
    offline_iters: int = 50 # deep-dream steps per cycle
    # Ripple substrate params
    ripple_rate_hz: float = 0.8
    stdp_gain: float = 3.0
    micro_dt_ms: float = 5.0  # Time step in milliseconds

class DreamEngine:
    """
    Unified dream engine: FSHO + CIO + NMDA + GCCRF + Themes + Wormhole.
    Enhanced with TTL motifs, selective updates, and meta-learning.
    """
    def __init__(self, cfg: DreamConfig):
        self.cfg = cfg
        # Robust device handling with validation
        self.device = self._validate_device(cfg.device)
        
        # Update config with validated device
        self.cfg.device = str(self.device)
        
        # safety: ensure small internal MLPs are created if missing
        # (we'll lazily create minimal trainable heads used by train_step())
        if not hasattr(self, "_dream_color_head"):
            import torch.nn as nn
            self._dream_color_head = nn.Sequential(
                nn.Linear(getattr(cfg, "state_dim", 64), 64),
                nn.ReLU(),
                nn.Linear(64, 10)  # predict 10 colors
            )
            self._dream_opbias_head = nn.Sequential(
                nn.Linear(getattr(cfg, "state_dim", 64), 64),
                nn.ReLU(),
                nn.Linear(64, getattr(cfg, "action_dim", 41))
            )
            # place on device
            self._dream_color_head.to(self.device)
            self._dream_opbias_head.to(self.device)
        
        # Set up deterministic behavior if requested
        if cfg.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(cfg.cio_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(cfg.cio_seed)
                torch.cuda.manual_seed_all(cfg.cio_seed)
        
        # Create seeded generator for reproducible random numbers
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(cfg.cio_seed)
        
        # Log device and deterministic configuration
        if cfg.verbose:
            pass  # Device and CIO configuration initialized
            
            pass  # DreamEngine config initialized
        
        # Optional attached relational memory (set by model)
        self._relmem = None

    def attach_relmem(self, relmem):
        """Attach a relational memory module for dream-gated plasticity."""
        self._relmem = relmem
        
        self.nmda = NMDAGatedDreaming(
            state_dim=self.cfg.state_dim, action_dim=self.cfg.action_dim, device=self.device
        )
        self.curiosity = GCCRFCuriosity(state_dim=self.cfg.state_dim)
        self.theme = EmergentThemeSynthesis()
        self.wormhole = WormholeTemplateMiner()
        # FSHO oscillator state (complex z = x + i y)
        self.z = torch.randn(2, device=self.device, generator=self._rng) * 0.1  # [Re, Im]
        # CIO Meta-Learner memory buffers
        self._cio_X = []  # Feature vectors
        self._cio_y = []  # Retention gains
        self._cio_max_hist = 512  # Maximum history size
        self._ridge_lambda = 1e-2  # Ridge regression regularization
        
        # TTL Dream Motifs storage
        self.dream_motifs = []  # List[DreamMotif]
        self.max_motifs = 50  # Maximum number of motifs to keep
        self.motif_entropy_threshold = 0.85  # Only keep motifs that reduce entropy by ≥15%
        
        # Meta-learning for strategy selection
        self.meta_learner = MetaLearner()
        
        # Selective update tracking
        self.beam_entropy_history = []
        self.template_performance = {}  # template_id → performance history
        
        # Initialize ripple substrate
        # Ripple substrate requires center_freq_hz in range [120.0, 250.0] Hz
        # Need dt_ms small enough so that Nyquist > 120Hz: fs = 1000/dt_ms > 240 => dt_ms < 4.17ms
        # Use dt_ms = 2.0ms to be safe, which gives fs = 500Hz, Nyquist = 225Hz (with 0.45 factor)
        ripple_dt_ms = 2.0  # Override dream engine dt for ripple substrate
        
        ripple_config = RippleConfig(
            event_rate_hz=self.cfg.ripple_rate_hz,
            stdp_gain=self.cfg.stdp_gain,
            center_freq_hz=170.0,  # Use default 170Hz which is in valid range
            phase_lock=True,
            dt_ms=ripple_dt_ms  # Use faster sampling for ripple substrate
        )
        self.ripple = RippleSubstrate(ripple_config)
        
        # Ripple configuration initialized
        pass
    
    def _validate_device(self, device_str: str) -> torch.device:
        """
        Validate and convert device string to torch.device with proper fallbacks.
        
        Args:
            device_str: Device specification (e.g., "cuda", "cpu", "cuda:0")
            
        Returns:
            torch.device: Validated device object
            
        Fallback chain:
        1. Try to create torch.device from input string
        2. If CUDA requested but not available, fall back to CPU
        3. If any error, fall back to CPU
        """
        try:
            device = torch.device(device_str)
            
            # Special handling for CUDA devices
            if device.type == 'cuda':
                if not torch.cuda.is_available():
                    print(f"[DreamEngine] CUDA requested ({device_str}) but not available, falling back to CPU")
                    return torch.device('cpu')
                elif device.index is not None and device.index >= torch.cuda.device_count():
                    print(f"[DreamEngine] CUDA device {device.index} not available, falling back to cuda:0")
                    return torch.device('cuda:0')
            
            return device
            
        except Exception as e:
            print(f"[DreamEngine] Invalid device '{device_str}': {e}, falling back to CPU")
            return torch.device('cpu')

    # --------- FSHO dynamics (fractional + Levy-ish noise, no external deps) ----------
    def _stable_noise(self, alpha: float, size: Tuple[int, ...]) -> torch.Tensor:
        """Chambers-Mallows-Stuck sampling for symmetric alpha-stable (β=0)"""
        # For α=2 => Gaussian
        U = (torch.rand(size, device=self.device, generator=self._rng) - 0.5) * math.pi
        W = -torch.log(torch.rand(size, device=self.device, generator=self._rng).clamp_min(1e-10))
        if abs(alpha - 2.0) < 1e-5:
            return torch.randn(size, device=self.device, generator=self._rng)
        const = math.tan(math.pi * alpha / 2.0)
        X = (torch.sin(alpha * U) / (torch.cos(U) ** (1.0 / alpha))) * \
            ((torch.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha))
        return X

    def _fgn_davies_harte(self, L: int, H: float) -> torch.Tensor:
        """
        Generate Fractional Gaussian Noise using corrected Davies-Harte FFT method.
        Target PSD slope: β = 1 - 2H
        
        Args:
            L: Length of FGN sequence to generate
            H: Hurst parameter (0 < H < 1)
            
        Returns:
            torch.Tensor: FGN sequence of length L with correct spectral characteristics
        """
        # Find next power of 2 for efficient FFT
        n = 1
        while n < 2*L:
            n <<= 1
        
        # Compute autocovariance for FBM increments (FGN)
        # γ(k) = σ²/2 * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))
        gamma = torch.zeros(n, dtype=torch.float64, device=self.device)
        
        # For k=0: variance = 1
        gamma[0] = 1.0
        
        # For k > 0: use correct FGN autocovariance formula
        for k in range(1, L):
            gamma[k] = 0.5 * ((k+1)**(2*H) - 2*(k**(2*H)) + abs(k-1)**(2*H))
        
        # Create circulant embedding by mirroring (excluding gamma[0] from mirror)
        if L > 1:
            gamma[n-L+1:n] = gamma[1:L].flip(0)
        
        # FFT to get eigenvalues
        eigenvalues = torch.fft.fft(gamma).real
        
        # Check for negative eigenvalues and fix if needed
        min_eig = eigenvalues.min().item()
        if min_eig < 0:
            # Add small correction to ensure positive definiteness
            eigenvalues = eigenvalues - min_eig + 1e-10
        
        # Generate complex Gaussian noise with correct variance
        # Each complex component should be N(0, 0.5) for total variance 1
        Z_real = torch.randn(n, dtype=torch.float64, device=self.device, generator=self._rng) * math.sqrt(0.5)
        Z_imag = torch.randn(n, dtype=torch.float64, device=self.device, generator=self._rng) * math.sqrt(0.5)
        Z = torch.complex(Z_real, Z_imag)
        
        # Apply square root of eigenvalues
        Y = torch.sqrt(eigenvalues + 1e-10) * Z
        
        # IFFT and extract first L samples
        y = torch.fft.ifft(Y).real
        fgn = y[:L]
        
        # Normalize to unit variance
        fgn = (fgn - fgn.mean()) / (fgn.std() + 1e-10)
        
        return fgn.float()  # Return as float32

    def _fractional_innovation(self, H: float, size: Tuple[int, ...]) -> torch.Tensor:
        """Cheap proxy: AR(1)-like persistence to mimic H>0.5 vs anti-persistence for H<0.5"""
        rho = (H - 0.5) * 1.6  # map H∈(0,1) to rho∈(-0.8,0.8)
        eps = torch.randn(size, device=self.device, generator=self._rng)
        out = torch.zeros_like(eps)
        for t in range(size[0]):
            out[t] = (rho * out[t-1] if t > 0 else 0.0) + eps[t]
        return out

    def fsho_step(self, steps: int = 1):
        """Stuart-Landau oscillator with fractional Gaussian noise + stable innovations"""
        K = self.cfg.fsho_K
        eta = self.cfg.fsho_eta
        alpha = self.cfg.fsho_alpha
        H = self.cfg.fsho_H
        fgn_scale = self.cfg.fsho_fgn_scale
        levy_scale = self.cfg.fsho_levy_scale
        
        for _ in range(steps):
            # z = x + i y
            x, y = self.z[0], self.z[1]
            r2 = x*x + y*y
            # Hopf-like drift
            dx = (1 - r2) * x - K * y
            dy = (1 - r2) * y + K * x
            
            # True Fractional Gaussian Noise using Davies-Harte
            fgn_sequence = self._fgn_davies_harte(L=4, H=H)  # Generate small sequence
            f = fgn_sequence.mean().item()  # Use mean as scalar innovation
            
            # Levy-stable jumps (independent for x, y components)
            Lx = self._stable_noise(alpha, ()).item()
            Ly = self._stable_noise(alpha, ()).item()
            
            # Update with scaled noise contributions
            x = x + eta * (dx + fgn_scale * f + levy_scale * Lx)
            y = y + eta * (dy + fgn_scale * f + levy_scale * Ly)
            
            # Update oscillator state
            self.z = torch.stack([x, y])
            
            # Stability check: ensure ||z|| stays reasonable
            z_norm = torch.norm(self.z).item()
            if z_norm > 5.0 or torch.isnan(self.z).any():
                # Reset if unstable
                self.z = torch.randn(2, device=self.device, generator=self._rng) * 0.1
                print(f"FSHO reset due to instability: ||z||={z_norm:.3f}")

    # -------------------- CIO meta-learning feature engineering ----------------------
    def _feat_vec(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Centralized feature extraction in pure Torch on model device.
        Returns tensor ready for conversion to numpy only when needed.
        """
        # Control knobs (first 4 features for gradient updates)
        K = torch.tensor(self.cfg.fsho_K, device=self.device, dtype=torch.float32)
        eta = torch.tensor(self.cfg.fsho_eta, device=self.device, dtype=torch.float32)
        alpha = torch.tensor(self.cfg.fsho_alpha, device=self.device, dtype=torch.float32)
        H = torch.tensor(self.cfg.fsho_H, device=self.device, dtype=torch.float32)
        
        # Ripple knobs
        ripple_rate = torch.tensor(self.cfg.ripple_rate_hz, device=self.device, dtype=torch.float32)
        stdp_gain = torch.tensor(self.cfg.stdp_gain, device=self.device, dtype=torch.float32)
        
        # Oscillator state
        z_re = self.z[0]
        z_im = self.z[1]
        z_norm = torch.norm(self.z)
        
        # Token statistics
        B, T, D = tokens.shape
        tokens_flat = tokens.reshape(-1, D)
        token_mean = tokens_flat.mean()
        token_std = tokens_flat.std()
        
        # Token entropy (approximate)
        token_prob = torch.softmax(tokens_flat.mean(0), dim=0)
        token_entropy = (-token_prob * torch.log(token_prob + 1e-10)).sum()
        
        # Curiosity metrics
        R_i, info_dict = self.curiosity.compute_reward(tokens_flat.detach())
        R_i_scalar = R_i.mean() if R_i.numel() > 1 else R_i
        I_alpha_scalar = info_dict.get('prediction_error', torch.tensor(0.0, device=self.device))
        
        # Derived oscillator metrics
        eta_cur = eta * (1.0 + 0.1 * z_norm)  # Current effective eta
        delta_eta = eta_cur - eta  # Deviation from baseline
        
        # Empowerment proxy (token variance scaled by curiosity)
        empowerment = tokens.var() * R_i_scalar
        
        # Alignment (oscillator-token coherence proxy)
        alignment = z_norm * token_std
        
        # Ripple metrics (update time first)
        current_time = time.time()
        self.ripple.update(current_time)
        
        ripple_coherence = torch.tensor(self.ripple.get_coherence_metrics()['current_coherence'], 
                                       device=self.device, dtype=torch.float32)
        ripple_phase = torch.tensor(self.ripple.get_phase_info()['phase_normalized'], 
                                   device=self.device, dtype=torch.float32)
        ripple_stdp = torch.tensor(self.ripple.get_stdp_gain(), 
                                  device=self.device, dtype=torch.float32)
        ripple_active = torch.tensor(1.0 if self.ripple.is_ripple_active() else 0.0, 
                                    device=self.device, dtype=torch.float32)
        
        features = torch.stack([
            K, eta, alpha, H,  # Control knobs (first 4)
            ripple_rate, stdp_gain,  # Ripple knobs (indices 4-5)
            z_re, z_im, z_norm,  # Oscillator state  
            token_mean, token_std, token_entropy,  # Token stats
            eta_cur, delta_eta, I_alpha_scalar,  # Derived metrics
            empowerment, alignment,  # High-level features
            ripple_coherence, ripple_phase, ripple_stdp, ripple_active  # Ripple metrics
        ])
        
        return features
    
    def _extract_cio_features(self, tokens: torch.Tensor) -> np.ndarray:
        """
        Extract comprehensive feature vector for Ridge regression.
        Uses centralized _feat_vec() and converts to CPU numpy only at the end.
        """
        # Get features in Torch on device
        features_torch = self._feat_vec(tokens)
        
        # Convert to CPU numpy only once for Ridge regression
        features_numpy = features_torch.detach().cpu().numpy().astype(np.float32)
        
        return features_numpy
        
    def compute_beam_entropy(self) -> float:
        """Compute current beam entropy for selective updates"""
        if not hasattr(self, '_last_tokens') or self._last_tokens is None:
            return 1.0  # High entropy = uncertain
            
        try:
            # Use cached tokens from last computation
            tokens = self._last_tokens
            
            # Compute entropy across feature dimensions
            # H = -sum(p * log(p)) where p = softmax(features)
            B, T, D = tokens.shape
            
            # Average across batch and time, compute entropy over features
            mean_features = tokens.mean(dim=(0, 1))  # [D]
            probs = F.softmax(mean_features, dim=0) + 1e-10
            entropy = -(probs * torch.log(probs)).sum().item()
            
            # Normalize to [0, 1] range (log(D) is max entropy for D dimensions)
            normalized_entropy = entropy / math.log(D) if D > 1 else 0.0
            
            return float(normalized_entropy)
            
        except Exception:
            return 1.0
            
    def compute_beam_entropy_with_template(self, template: torch.Tensor) -> float:
        """Compute beam entropy if we were to add this template"""
        if not hasattr(self, '_last_tokens') or self._last_tokens is None:
            return 1.0
            
        try:
            tokens = self._last_tokens
            
            # Simulate adding template by modifying features
            # Simple approach: add template as guidance to features
            B, T, D = tokens.shape
            
            # Broadcast template to match token dimensions
            if template.dim() == 1:
                template_expanded = template.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            else:
                template_expanded = template
                
            # Ensure compatible dimensions
            if template_expanded.shape[-1] != D:
                # Pad or truncate template to match feature dimension
                if template_expanded.shape[-1] > D:
                    template_expanded = template_expanded[..., :D]
                else:
                    padding = D - template_expanded.shape[-1]
                    template_expanded = F.pad(template_expanded, (0, padding))
                    
            # Mix template with tokens (small influence)
            mixed_tokens = 0.95 * tokens + 0.05 * template_expanded
            
            # Compute entropy of mixed features
            mean_features = mixed_tokens.mean(dim=(0, 1))
            probs = F.softmax(mean_features, dim=0) + 1e-10
            entropy = -(probs * torch.log(probs)).sum().item()
            
            normalized_entropy = entropy / math.log(D) if D > 1 else 0.0
            return float(normalized_entropy)
            
        except Exception:
            return 1.0
            
    def add_dream_motif(self, pattern: torch.Tensor, ttl: int = 10) -> bool:
        """Add a new dream motif with TTL and entropy checking"""
        try:
            # Test if motif reduces entropy
            initial_entropy = self.compute_beam_entropy()
            test_entropy = self.compute_beam_entropy_with_template(pattern)
            
            entropy_reduction = (initial_entropy - test_entropy) / initial_entropy if initial_entropy > 0 else 0
            
            # Only add if entropy reduction meets threshold
            if entropy_reduction >= (1.0 - self.motif_entropy_threshold):  # 15% reduction
                motif = DreamMotif(
                    pattern=pattern.detach().clone(),
                    ttl=ttl,
                    entropy_reduction=entropy_reduction,
                    created_time=time.time()
                )
                
                self.dream_motifs.append(motif)
                
                # Limit number of motifs
                if len(self.dream_motifs) > self.max_motifs:
                    # Remove worst performing motifs
                    self.dream_motifs.sort(key=lambda m: m.success_rate + m.entropy_reduction, reverse=True)
                    self.dream_motifs = self.dream_motifs[:self.max_motifs]
                    
                if self.cfg.verbose:
                    print(f"[DreamMotif] Added motif with {entropy_reduction:.3f} entropy reduction")
                    
                return True
            else:
                if self.cfg.verbose:
                    print(f"[DreamMotif] Rejected motif: {entropy_reduction:.3f} < {1.0 - self.motif_entropy_threshold:.3f} entropy reduction")
                return False
                
        except Exception as e:
            if self.cfg.verbose:
                print(f"[DreamMotif] Error adding motif: {e}")
            return False
            
    def tick_motifs(self):
        """Update TTL for all motifs and remove expired ones"""
        before_count = len(self.dream_motifs)
        self.dream_motifs = [motif for motif in self.dream_motifs if motif.tick()]
        removed_count = before_count - len(self.dream_motifs)
        
        if removed_count > 0 and self.cfg.verbose:
            print(f"[DreamMotif] Removed {removed_count} expired motifs, {len(self.dream_motifs)} remaining")
            
    def extract_selective_templates(self, tokens: torch.Tensor, demos_programs=None) -> List[torch.Tensor]:
        """Extract templates only if they provide entropy reduction"""
        # Cache tokens for entropy computation
        self._last_tokens = tokens.detach()
        
        initial_entropy = self.compute_beam_entropy()
        
        # Extract candidate templates (simplified)
        templates = []
        
        try:
            B, T, D = tokens.shape
            
            # Extract patterns from token statistics
            # Pattern 1: High-variance features (indicating structure)
            feature_var = tokens.var(dim=(0, 1))  # [D]
            high_var_mask = feature_var > feature_var.mean() + feature_var.std()
            if high_var_mask.any():
                pattern = torch.zeros(D, device=tokens.device)
                pattern[high_var_mask] = tokens.mean(dim=(0, 1))[high_var_mask]
                templates.append(pattern)
                
            # Pattern 2: Dominant activation patterns
            mean_activation = tokens.mean(dim=(0, 1))  # [D]
            activation_threshold = mean_activation.mean() + mean_activation.std()
            dominant_mask = mean_activation > activation_threshold
            if dominant_mask.any():
                pattern = torch.zeros(D, device=tokens.device)
                pattern[dominant_mask] = mean_activation[dominant_mask]
                templates.append(pattern)
                
            # Pattern 3: Temporal consistency patterns
            if T > 1:
                temporal_std = tokens.std(dim=1).mean(dim=0)  # [D] - low std = consistent
                consistent_mask = temporal_std < temporal_std.median()
                if consistent_mask.any():
                    pattern = torch.zeros(D, device=tokens.device)
                    pattern[consistent_mask] = tokens.mean(dim=(0, 1))[consistent_mask]
                    templates.append(pattern)
                    
        except Exception as e:
            if self.cfg.verbose:
                print(f"[DreamEngine] Template extraction failed: {e}")
                
        # Filter templates by entropy reduction
        good_templates = []
        for template in templates:
            test_entropy = self.compute_beam_entropy_with_template(template)
            entropy_reduction = (initial_entropy - test_entropy) / initial_entropy if initial_entropy > 0 else 0
            
            if entropy_reduction >= (1.0 - self.motif_entropy_threshold):
                good_templates.append(template)
                if self.cfg.verbose:
                    print(f"[DreamEngine] Accepted template with {entropy_reduction:.3f} entropy reduction")
            
        return good_templates

    # -------------------- CIO meta-learning (perturb-and-learn) -----------------------
    def cio_perturb_and_learn(self, tokens: torch.Tensor, prior_retention: float) -> float:
        """
        CIO Meta-Learner with Ridge regression instead of heuristic updates.
        Collects (features, gain) pairs and uses Ridge regression to predict optimal knob gradients.
        """
        # Extract current feature vector
        current_features = self._extract_cio_features(tokens)
        
        # Compute current retention gain (same calculation as before)
        B, T, D = tokens.shape
        tokens_flat = tokens.reshape(B*T, D)
        R_i, _ = self.curiosity.compute_reward(tokens_flat.detach())
        R_i_scalar = R_i.mean() if R_i.numel() > 1 else R_i
        gain = float((R_i_scalar + tokens.var()).item())
        
        # Store (features, gain) pair in memory buffer
        self._cio_X.append(current_features.copy())
        self._cio_y.append(gain)
        
        # Maintain maximum history size
        if len(self._cio_X) > self._cio_max_hist:
            self._cio_X.pop(0)
            self._cio_y.pop(0)
            
        # Ridge regression learning (only if we have sufficient data)
        if len(self._cio_X) >= 32:
            try:
                # Prepare training data
                X = np.array(self._cio_X)  # [N, 15] features
                y = np.array(self._cio_y)  # [N,] gains
                
                # Ridge regression: (X^T X + λI)^{-1} X^T y
                XtX = X.T @ X
                lambda_I = self._ridge_lambda * np.eye(X.shape[1])
                try:
                    ridge_coeff = np.linalg.solve(XtX + lambda_I, X.T @ y)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    ridge_coeff = np.linalg.pinv(XtX + lambda_I) @ (X.T @ y)
                
                # Gradient ascent on control knobs (first 6 coefficients)
                grad_K = ridge_coeff[0]
                grad_eta = ridge_coeff[1] 
                grad_alpha = ridge_coeff[2]
                grad_H = ridge_coeff[3]
                grad_ripple_rate = ridge_coeff[4]
                grad_stdp_gain = ridge_coeff[5]
                
                # Update control knobs with gradient ascent and clamping
                step_size = self.cfg.cio_lr
                self.cfg.fsho_K = np.clip(
                    self.cfg.fsho_K + step_size * grad_K, 0.05, 0.8
                )
                self.cfg.fsho_eta = np.clip(
                    self.cfg.fsho_eta + step_size * grad_eta, 0.02, 0.5
                )
                self.cfg.fsho_alpha = np.clip(
                    self.cfg.fsho_alpha + step_size * grad_alpha, 1.1, 2.0
                )
                self.cfg.fsho_H = np.clip(
                    self.cfg.fsho_H + step_size * grad_H, 0.1, 0.95
                )
                # Update ripple knobs
                self.cfg.ripple_rate_hz = np.clip(
                    self.cfg.ripple_rate_hz + step_size * grad_ripple_rate, 0.1, 5.0
                )
                self.cfg.stdp_gain = np.clip(
                    self.cfg.stdp_gain + step_size * grad_stdp_gain, 1.0, 10.0
                )
                
                # Update ripple substrate configuration
                self.ripple.config.event_rate_hz = self.cfg.ripple_rate_hz
                self.ripple.config.stdp_gain = self.cfg.stdp_gain
                
                # Log configuration changes for debugging
                if self.cfg.verbose:
                    print(f"CIO Ridge update: K={self.cfg.fsho_K:.4f}, eta={self.cfg.fsho_eta:.4f}, "
                          f"alpha={self.cfg.fsho_alpha:.4f}, H={self.cfg.fsho_H:.4f}, "
                          f"ripple_rate={self.cfg.ripple_rate_hz:.4f}, stdp_gain={self.cfg.stdp_gain:.4f}, "
                          f"gain={gain:.4f}")
                      
            except Exception as e:
                # Error handling without silent failure
                if self.cfg.verbose:
                    print(f"CIO Ridge regression failed: {e}")
                # Fall back to random perturbation for this step only
                self._fallback_perturbation(gain, prior_retention)
                
        else:
            # Fallback: use simple heuristic until we have enough data points
            if self.cfg.verbose:
                print(f"CIO: Using fallback (only {len(self._cio_X)}/32 samples)")
            self._fallback_perturbation(gain, prior_retention)
            
        return gain
    
    def _fallback_perturbation(self, gain: float, prior_retention: float):
        """Fallback perturbation method using seeded generator when Ridge regression is not applicable."""
        if gain > prior_retention:
            # Small random improvements to all knobs using seeded generator
            perturb_K = torch.rand(1, generator=self._rng).item() * 0.04 - 0.02  # [-0.02, 0.02]
            perturb_eta = torch.rand(1, generator=self._rng).item() * 0.02 - 0.01  # [-0.01, 0.01]
            perturb_alpha = torch.rand(1, generator=self._rng).item() * 0.1 - 0.05  # [-0.05, 0.05]
            perturb_H = torch.rand(1, generator=self._rng).item() * 0.04 - 0.02  # [-0.02, 0.02]
            perturb_ripple_rate = torch.rand(1, generator=self._rng).item() * 0.2 - 0.1  # [-0.1, 0.1]
            perturb_stdp = torch.rand(1, generator=self._rng).item() * 0.4 - 0.2  # [-0.2, 0.2]
            
            self.cfg.fsho_K = np.clip(
                self.cfg.fsho_K + perturb_K, 0.05, 0.8
            )
            self.cfg.fsho_eta = np.clip(
                self.cfg.fsho_eta + perturb_eta, 0.02, 0.5
            )
            self.cfg.fsho_alpha = np.clip(
                self.cfg.fsho_alpha + perturb_alpha, 1.1, 2.0
            )
            self.cfg.fsho_H = np.clip(
                self.cfg.fsho_H + perturb_H, 0.1, 0.95
            )
            # Also perturb ripple knobs in fallback
            self.cfg.ripple_rate_hz = np.clip(
                self.cfg.ripple_rate_hz + perturb_ripple_rate, 0.1, 5.0
            )
            self.cfg.stdp_gain = np.clip(
                self.cfg.stdp_gain + perturb_stdp, 1.0, 10.0
            )

    # -------------------- Public API used by model -----------------------------------
    @torch.no_grad()
    def compute_priors(self, tokens: torch.Tensor, H: int, W: int) -> Dict[str, torch.Tensor]:
        """
        Compute neuromorphic priors for EBR: phi, kappa, cge (hodge optional).
        tokens: [B, T, D]  (T = H*W if possible)
        """
        device = tokens.device
        try:
            phi = phi_synergy_features(tokens)  # scalar
        except Exception:
            phi = torch.tensor(0.0, device=device)
        try:
            kappa = kappa_floor(tokens, H, W)
        except Exception:
            kappa = torch.tensor(0.0, device=device)
        try:
            cge = cge_boost(tokens)
        except Exception:
            cge = torch.tensor(0.0, device=device)
        return dict(phi=phi, kappa=kappa, cge=cge)

    def record_experience(self, latent_state: torch.Tensor, next_latent: Optional[torch.Tensor] = None,
                          action: int = 0, reward: float = 0.0, valence: Optional[float] = None,
                          arousal: Optional[float] = None):
        """Push one experience into NMDA buffer (latent space)."""
        v = self.cfg.valence_default if valence is None else float(valence)
        a = self.cfg.arousal_default if arousal is None else float(arousal)
        if next_latent is None:
            next_latent = latent_state
        self.nmda.store_dream_memory(latent_state, action, reward, next_latent)
        # NMDA gate checked during consolidation call

    def step_micro(self, valence: Optional[float], arousal: Optional[float]):
        """Tiny consolidation step (online-safe)."""
        v = self.cfg.valence_default if valence is None else float(valence)
        a = self.cfg.arousal_default if arousal is None else float(arousal)
        return self.nmda.dream_consolidation(v, a)  # returns 0.0 if gate closed / not enough buffer

    def cycle_offline(self, tokens: torch.Tensor, demos_programs: Optional[List[List[Tuple[str, dict]]]] = None,
                      valence: float = 0.7, arousal: float = 0.3) -> Dict[str, float]:
        """
        Deep dream consolidation: oscillator rollouts, CIO perturb-learn, theme synthesis,
        wormhole motif mining, NMDA consolidation sweeps with ripple substrate integration.
        """
        import time
        
        # Begin ripple cycle with energy scalar based on tokens
        B, T, D = tokens.shape
        energy_scalar = 1.0 + float(tokens.var().item()) * 0.01
        
        # Initialize ripple cycle with proper number of steps
        iters = self.cfg.offline_iters
        self.ripple.begin_cycle(iters, energy_scalar=energy_scalar)
        current_time = 0.0  # Use relative time for cycle
        
        # 0) FSHO roll with ripple integration
        for i in range(iters):
            self.fsho_step()
            
            # Update ripple substrate with proper time step
            step_time = i * (self.cfg.micro_dt_ms / 1000.0)  # Convert ms to seconds
            self.ripple.update(step_time)
            
            # Get active ripple context
            ripple_ctx = self.ripple.get_current_context() if self.ripple.is_ripple_active() else None
            
            # Perform NMDA consolidation with ripple context every few steps
            if i % 10 == 0:
                loss = self.nmda.dream_consolidation(valence, arousal, ripple_ctx)

        # 1) CIO meta-learn a small step using tokens
        prior = 0.0
        gain = self.cio_perturb_and_learn(tokens, prior)
        
        # Tick motifs to update TTL
        self.tick_motifs()

        # 2) Theme synthesis from tokens (mock labels as cluster ids)
        labels = torch.arange(T, device=tokens.device) % max(2, T//4)
        themes = self.theme.process_dream_themes(tokens.mean(1), labels)
        self.theme.synthesize_emergent_themes(themes)

        # 3) Extract selective templates and add as motifs
        selective_templates = self.extract_selective_templates(tokens, demos_programs)
        for template in selective_templates:
            self.add_dream_motif(template, ttl=15)  # Longer TTL for good templates
            
        # 4) Wormhole motif mining (if programs recorded)
        mined = []
        if demos_programs:
            mined = self.wormhole.mine_from_programs(demos_programs, top_k=5)

        # 5) Several NMDA consolidation passes with ripple context
        losses = []
        for i in range(3):
            step_time = current_time + (self.cfg.offline_iters + i) * 0.001
            self.ripple.update(step_time)
            ripple_ctx = self.ripple.get_current_context() if self.ripple.is_ripple_active() else None
            losses.append(self.nmda.dream_consolidation(valence, arousal, ripple_ctx) or 0.0)

        # --- Dream-gated RelMem plasticity ---
        # Gate on valence/arousal (biologically-inspired) and scale by CIO/ripple knobs
        try:
            if self._relmem is not None:
                # Determine thresholds (balanced defaults; tune as needed)
                v_hebb, a_hebb = 0.50, 0.30
                v_wta,  a_wta  = 0.70, 0.50
                # CIO/ripple knobs for "how much" plasticity this window gets
                scale = float(self.cfg.stdp_gain) * float(self.cfg.ripple_rate_hz)
                # Apply Hebbian when pleasant & engaged
                if valence > v_hebb and arousal > a_hebb:
                    self._relmem.apply_hebbian()
                # Apply WTA when very pleasant & strongly engaged
                if valence > v_wta and arousal > a_wta:
                    # Optional: call multiple times to reflect scale (bounded)
                    reps = int(min(3, max(1, round(scale))))
                    for _ in range(reps):
                        self._relmem.apply_wta()
        except Exception:
            pass

        # Get ripple metrics with reset for next cycle
        ripple_metrics = self.ripple.metrics(reset=True)
        
        # Update meta-learner with cycle results
        strategy_success = gain > prior * 1.1  # 10% improvement threshold
        self.meta_learner.update('dream_cycle', 'cio_perturb', strategy_success, tokens.mean(dim=(0, 1)))
        
        return {
            "cio_gain": gain, 
            "nmda_loss_mean": float(sum(losses)/max(1, len(losses))), 
            "motifs": float(len(mined)),
            "dream_motifs": len(self.dream_motifs),
            "active_motifs": len([m for m in self.dream_motifs if m.success_rate > 0.5]),
            "avg_motif_entropy_reduction": np.mean([m.entropy_reduction for m in self.dream_motifs]) if self.dream_motifs else 0.0,
            "ripple_events": ripple_metrics.get('ripple_events', 0),
            "ripple_active_steps": ripple_metrics.get('ripple_active_steps', 0),
            "ripple_phase_coherence": ripple_metrics.get('ripple_phase_coherence', 0.0)
        }
    def train_step(self, slot_vecs, target=None):
        """Minimal training step for Dream/ETS pretraining (lightweight but real)."""
        import torch.nn.functional as F
        loss = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True)
        # If slots present, do a color prediction head (mean pool slots => predict colors)
        if slot_vecs is not None:
            if slot_vecs.dim() == 3:  # [B, T, D]
                pooled = slot_vecs.mean(dim=1)  # [B, D]
            else:
                pooled = slot_vecs.view(slot_vecs.size(0), -1)
            logits = self._dream_color_head(pooled)
            # if target provided (color labels), compute CE; else create a pseudo target via argmax on pooled proj
            if target is not None and target.dim() == 2:
                # collapse to color labels for simplicity (training-only)
                tgt = target.flatten().long()
                ce = F.cross_entropy(logits, tgt[: logits.size(0)])
            else:
                # pseudo supervision: encourage logits entropy to shrink (low-entropy)
                probs = torch.softmax(logits, dim=-1)
                ce = (probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = loss + 1.0 * ce
        # If op-bias training is desired (weak KL toward some prior), include small KL term
        try:
            op_logits = self._dream_opbias_head(slot_vecs.mean(dim=1) if slot_vecs is not None else torch.randn(1, getattr(self.cfg,'state_dim',64), device=self.device))
            op_probs = torch.softmax(op_logits, dim=-1)
            # encourage non-degenerate distribution (entropy regularizer)
            ent = - (op_probs * torch.log(op_probs + 1e-8)).sum(dim=-1).mean()
            loss = loss + 0.01 * (-ent)  # encourage peaky distributions slightly
        except Exception:
            pass
        return loss

    def parameters(self):
        """Yield parameters from internal nn modules and discovered submodules."""
        seen = set()
        def walk(obj):
            if id(obj) in seen: return
            seen.add(id(obj))
            import torch.nn as nn
            if isinstance(obj, nn.Module):
                for p in obj.parameters():
                    yield p
                for name, sub in vars(obj).items():
                    for q in walk(sub):
                        yield q
            else:
                for name, sub in getattr(obj, "__dict__", {}).items():
                    if id(sub) in seen: continue
                    if hasattr(sub, "parameters") and callable(sub.parameters):
                        try:
                            for p in sub.parameters():
                                yield p
                        except Exception:
                            for q in walk(sub):
                                yield q
                    else:
                        for q in walk(sub):
                            yield q
        for p in walk(self):
            yield p

    def save_state(self, path: str):
        """Save minimal state for pretrain (heads + themes if present)."""
        state = {}
        try:
            state["color_head"] = {k: v.cpu() for k, v in self._dream_color_head.state_dict().items()}
            state["opbias_head"] = {k: v.cpu() for k, v in self._dream_opbias_head.state_dict().items()}
        except Exception:
            pass
        try:
            if hasattr(self, "theme") and hasattr(self.theme, "state_dict"):
                state["themes"] = {k: v.cpu() for k, v in self.theme.state_dict().items()}
        except Exception:
            pass
        import torch
        torch.save(state, path)

    def load_state(self, path: str):
        import os, torch
        if not os.path.exists(path): return
        state = torch.load(path, map_location=self.device)
        if "color_head" in state:
            try:
                self._dream_color_head.load_state_dict(state["color_head"])
            except Exception:
                pass
        if "opbias_head" in state:
            try:
                self._dream_opbias_head.load_state_dict(state["opbias_head"])
            except Exception:
                pass
