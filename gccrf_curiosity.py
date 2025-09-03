#!/usr/bin/env python3
"""
Fixed GCCRFCuriosity with stable rewards and gradient flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

class GCCRFCuriosity(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: Optional[int] = None, 
                 alpha_start: float = -0.5, alpha_end: float = 0.0, 
                 anneal_steps: int = 1000, learning_rate: float = 1e-3,
                 enable_kde: bool = True, bandwidth: float = 0.5, 
                 reservoir_size: int = 2048):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim or (state_dim * 2)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.anneal_steps = anneal_steps
        self.current_step = 0
        
        # KDE parameters
        self.enable_kde = enable_kde
        self.bandwidth = bandwidth
        self.reservoir_size = reservoir_size
        self.register_buffer('reservoir', torch.zeros(self.reservoir_size, self.state_dim))
        self.reservoir_ptr = 0
        self.strategic_targets = None  # [M, state_dim]

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, state_dim)
        )
        
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        # Running statistics - graph-connected
        self.register_buffer('running_error', torch.zeros(1))
        self.register_buffer('error_momentum', torch.ones(1) * 0.95)

    def set_strategic_targets(self, targets: torch.Tensor):
        """Set strategic targets for alignment computation"""
        assert targets.dim() == 2 and targets.size(-1) == self.state_dim, "bad targets"
        self.strategic_targets = targets.detach()
    
    def _kde_density(self, x: torch.Tensor) -> torch.Tensor:
        """Compute KDE density estimate for novelty"""
        # x: [B, D]
        n = min(self.reservoir_ptr, self.reservoir_size)
        if n < 16:
            return torch.ones(x.size(0), device=x.device) * 0.5
        bank = self.reservoir[:n].to(x.device)  # [n, D]
        # RBF kernel density
        # dist2: [B, n]
        dist2 = torch.cdist(x, bank, p=2.0).pow(2)
        k = torch.exp(-dist2 / (2 * (self.bandwidth ** 2)))
        p = k.mean(dim=1).clamp_min(1e-6)
        return p  # pseudo-density in (0, 1]
    
    def _update_reservoir(self, states: torch.Tensor):
        """Update reservoir with new states"""
        # states: [B, D]
        b = states.size(0)
        take = min(b, self.reservoir_size)
        idxs = torch.arange(take, device=states.device)
        pos = (self.reservoir_ptr + idxs) % self.reservoir_size
        self.reservoir[pos] = states[:take].detach()
        self.reservoir_ptr = int((self.reservoir_ptr + take) % self.reservoir_size)
    
    def get_alpha(self):
        """Get annealed alpha value"""
        frac = min(self.current_step / self.anneal_steps, 1.0)
        alpha = self.alpha_start + frac * (self.alpha_end - self.alpha_start)
        self.current_step += 1
        return alpha

    def compute_reward(self, states: torch.Tensor, actions: Optional[torch.Tensor] = None, 
                      update_predictor: bool = True, next_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute curiosity reward with stable gradients
        All rewards scaled to [0, 0.05] to prevent explosion
        """
        device = states.device
        
        # Prediction error (differentiable)
        pred = self.predictor(states)
        eta = F.mse_loss(pred, states, reduction='none').mean(dim=-1)
        
        # Update running error (detached to avoid graph retention)
        with torch.no_grad():
            self.running_error = self.error_momentum * self.running_error + (1 - self.error_momentum) * eta.mean().detach()
        delta_eta = eta - self.running_error
        
        # Novelty with KDE or fallback to Tsallis entropy
        alpha = self.get_alpha()
        if self.enable_kde and states.dim() == 2:
            density = self._kde_density(states)  # [B]
            I_alpha = ((density + 1e-6).pow(-(alpha + 1.0) / 2.0) - 1.0).clamp(-1.0, 1.0)
        else:
            # Fallback to existing softmax entropy-based novelty
            probs = F.softmax(states, dim=-1)
            probs_stable = probs.clamp(1e-8, 1.0)
            I_alpha = (probs_stable.pow(-(alpha + 1)/2).clamp(0.0, 10.0).mean(dim=-1) - 1).clamp(-1.0, 1.0)
        
        # Empowerment (stabilized logdet)
        cov = torch.cov(states.t())
        # Add epsilon * I for numerical stability
        eps = 1e-3
        cov_stable = cov + eps * torch.eye(self.state_dim, device=device)
        
        # Clamp eigenvalues for stable logdet
        try:
            eigvals = torch.linalg.eigvalsh(cov_stable)
            eigvals = eigvals.clamp(eps, 1.0)
            logdet = eigvals.log().sum()
            E = logdet.clamp(-10.0, 10.0)  # Bounded logdet
        except:
            E = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Modulator (stable)
        mu_t = torch.sigmoid(eta.std().clamp(0.0, 10.0)) if eta.numel() > 1 else torch.sigmoid(states.std().clamp(0.0, 10.0))
        
        # Strategic Alignment S if provided
        S = torch.tensor(0.0, device=states.device)
        if self.strategic_targets is not None and states.dim() == 2:
            tgt = self.strategic_targets.to(states.device)  # [M, D]
            # max alignment per state then mean
            S = torch.max(F.cosine_similarity(states.unsqueeze(1), tgt.unsqueeze(0), dim=-1), dim=1).values.mean()
        else:
            # Fallback to original alignment (stable cosine similarity)
            S = F.cosine_similarity(states, states.mean(0, keepdim=True), dim=-1).mean()
        
        # Combine rewards with SMALL weights (0.01-0.05 scale)
        R_i = (0.01 * eta.mean() + 
               0.01 * delta_eta.clamp(-1.0, 1.0) + 
               0.01 * I_alpha.mean() if not torch.is_tensor(I_alpha) else 0.01 * I_alpha + 
               0.01 * (E * mu_t) + 
               0.01 * S).clamp(0.0, 0.05)  # Final scale: [0, 0.05]
        
        # Update reservoir with new states
        if self.enable_kde and states.dim() == 2:
            self._update_reservoir(states)
        
        # Update predictor if requested
        if update_predictor and next_states is not None:
            loss = self.update_predictor(states, next_states)
        
        return R_i, {
            'prediction_error': torch.nan_to_num(eta.mean(), nan=0.0, posinf=1.0, neginf=0.0),
            'learning_progress': torch.nan_to_num(delta_eta.mean(), nan=0.0, posinf=1.0, neginf=-1.0),
            'novelty': torch.nan_to_num(I_alpha.mean(), nan=0.0, posinf=1.0, neginf=-1.0),
            'empowerment': torch.nan_to_num(E, nan=0.0, posinf=10.0, neginf=-10.0),
            'modulator': torch.nan_to_num(mu_t, nan=0.5, posinf=1.0, neginf=0.0),
            'alignment': torch.nan_to_num(S, nan=0.0, posinf=1.0, neginf=-1.0)
        }

    def update_predictor(self, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Update predictor network - returns loss tensor (not item)"""
        self.predictor_optimizer.zero_grad()
        predictions = self.predictor(states)
        loss = F.mse_loss(predictions, targets)
        
        # Only backward if not in no_grad context
        if torch.is_grad_enabled():
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.predictor_optimizer.step()
        
        return loss  # Return tensor, not item()