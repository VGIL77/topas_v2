#!/usr/bin/env python3
"""
SGI Optimizer
SGI-tuned optimization schedule with advanced regularization and scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
import math
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for SGI optimizer"""
    # Base learning rate parameters
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    
    # Learning rate scheduling
    scheduler_type: str = "cosine_warm_restarts"  # "cosine_warm_restarts", "polynomial", "exponential"
    warmup_epochs: int = 100
    T_0: int = 1000  # Initial restart period for cosine annealing
    T_mult: int = 2  # Period multiplier for cosine annealing
    eta_min: float = 1e-6  # Minimum learning rate
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    grad_clip_value: Optional[float] = None
    
    # Stochastic depth
    stochastic_depth_prob: float = 0.1
    stochastic_depth_mode: str = "linear"  # "linear", "uniform"
    
    # Dropout rates
    dropout_rate: float = 0.1
    attention_dropout: float = 0.05
    
    # Label smoothing
    label_smoothing_grid: float = 0.0   # No smoothing for grid predictions
    label_smoothing_tokens: float = 0.1  # Smoothing for token predictions
    
    # Advanced regularization
    spectral_norm: bool = False
    weight_standardization: bool = True
    shake_shake_alpha: float = 0.0  # 0.0 disables shake-shake
    
    # Adaptive optimization
    adaptive_lr: bool = True
    patience: int = 50  # Epochs to wait before reducing LR
    factor: float = 0.8  # Factor to reduce LR by
    
    # SAM (Sharpness-Aware Minimization)
    use_sam: bool = False
    sam_rho: float = 0.05
    sam_adaptive: bool = True


class SGIOptimizer:
    """SGI-tuned optimization with advanced regularization"""
    
    def __init__(self, 
                 model: nn.Module, 
                 config: OptimizerConfig = None):
        self.model = model
        self.config = config or OptimizerConfig()
        
        # Initialize optimizer
        if self.config.use_sam:
            from utils.sam import SAM
            base_optimizer = optim.AdamW
            self.optimizer = SAM(
                model.parameters(), 
                base_optimizer,
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
                eps=self.config.eps,
                rho=self.config.sam_rho,
                adaptive=self.config.sam_adaptive
            )
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
                eps=self.config.eps
            )
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Adaptive LR scheduler
        if self.config.adaptive_lr:
            self.adaptive_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # We want to maximize accuracy
                factor=self.config.factor,
                patience=self.config.patience
            )
        else:
            self.adaptive_scheduler = None
        
        # Regularization components
        self._setup_regularization()
        
        # Tracking
        self.step_count = 0
        self.epoch_count = 0
        self.best_metric = 0.0
        self.lr_history = []
        self.gradient_norms = []
        
    def _create_scheduler(self):
        """Create learning rate scheduler based on config"""
        if self.config.scheduler_type == "cosine_warm_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.eta_min
            )
        elif self.config.scheduler_type == "polynomial":
            # Polynomial decay
            def lr_lambda(epoch):
                if epoch < self.config.warmup_epochs:
                    return epoch / self.config.warmup_epochs
                else:
                    return ((self.config.T_0 - epoch + self.config.warmup_epochs) / self.config.T_0) ** 0.9
            scheduler = LambdaLR(self.optimizer, lr_lambda)
        elif self.config.scheduler_type == "exponential":
            # Exponential decay with warmup
            def lr_lambda(epoch):
                if epoch < self.config.warmup_epochs:
                    return epoch / self.config.warmup_epochs
                else:
                    return 0.95 ** (epoch - self.config.warmup_epochs)
            scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            # No scheduling
            scheduler = None
        
        return scheduler
    
    def _setup_regularization(self):
        """Setup advanced regularization techniques"""
        # Apply spectral normalization if requested
        if self.config.spectral_norm:
            self._apply_spectral_norm()
        
        # Apply weight standardization if requested
        if self.config.weight_standardization:
            self._apply_weight_standardization()
        
        # Setup stochastic depth
        self._setup_stochastic_depth()
        
        # Setup dropout
        self._setup_dropout()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to linear layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.utils.spectral_norm(module)
                logger.info(f"Applied spectral norm to {name}")
    
    def _apply_weight_standardization(self):
        """Apply weight standardization wrapper"""
        def weight_standardization_hook(module, input):
            """Hook for weight standardization"""
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                mean = weight.mean(dim=(1, 2, 3) if len(weight.shape) == 4 else 1, keepdim=True)
                std = weight.std(dim=(1, 2, 3) if len(weight.shape) == 4 else 1, keepdim=True) + 1e-5
                module.weight.data = (weight - mean) / std
        
        # Apply to linear and conv layers
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_pre_hook(weight_standardization_hook)
    
    def _setup_stochastic_depth(self):
        """Setup stochastic depth for transformer layers"""
        if not hasattr(self.model, 'set_stochastic_depth'):
            logger.warning("Model doesn't support stochastic depth")
            return
        
        # Calculate drop probabilities
        if self.config.stochastic_depth_mode == "linear":
            # Linear increase in drop probability with depth
            num_layers = getattr(self.model, 'num_layers', 12)
            drop_probs = [i / num_layers * self.config.stochastic_depth_prob for i in range(num_layers)]
        else:  # uniform
            drop_probs = [self.config.stochastic_depth_prob] * getattr(self.model, 'num_layers', 12)
        
        self.model.set_stochastic_depth(drop_probs)
    
    def _setup_dropout(self):
        """Setup dropout rates throughout model"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.config.dropout_rate
            elif hasattr(module, 'attention_dropout'):
                module.attention_dropout = self.config.attention_dropout
    
    def step(self, 
             loss: torch.Tensor, 
             retain_graph: bool = False) -> Dict[str, float]:
        """
        Perform optimization step with advanced features.
        
        Args:
            loss: Loss tensor to optimize
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary of optimization metrics
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward(retain_graph=retain_graph)
        
        # Compute gradient norms before clipping
        grad_norm_before = self._compute_gradient_norm()
        self.gradient_norms.append(grad_norm_before)
        
        # Gradient clipping
        grad_norm_after = grad_norm_before
        if self.config.grad_clip_norm > 0:
            grad_norm_after = nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.grad_clip_norm
            )
        
        if self.config.grad_clip_value is not None:
            nn.utils.clip_grad_value_(
                self.model.parameters(),
                clip_value=self.config.grad_clip_value
            )
        
        # Optimizer step
        if self.config.use_sam:
            # SAM requires two forward passes
            self.optimizer.step(zero_grad=True)
        else:
            self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update counters
        self.step_count += 1
        
        # Track learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        return {
            'lr': current_lr,
            'grad_norm_before': float(grad_norm_before),
            'grad_norm_after': float(grad_norm_after),
            'grad_clip_ratio': float(grad_norm_after / max(grad_norm_before, 1e-8))
        }
    
    def epoch_step(self, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch"""
        self.epoch_count = epoch
        
        # Update adaptive scheduler
        if self.adaptive_scheduler is not None:
            # Use accuracy or other performance metric
            performance_metric = metrics.get('accuracy', metrics.get('exact_match', 0.0))
            self.adaptive_scheduler.step(performance_metric)
            
            # Track best metric
            if performance_metric > self.best_metric:
                self.best_metric = performance_metric
    
    def _compute_gradient_norm(self) -> float:
        """Compute L2 norm of gradients"""
        total_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def apply_regularization(self, epoch: int) -> Dict[str, torch.Tensor]:
        """
        Apply various regularization techniques during training.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary of regularization losses
        """
        reg_losses = {}
        
        # L1/L2 weight penalties (already in optimizer, but can add custom)
        if hasattr(self.config, 'l1_penalty') and self.config.l1_penalty > 0:
            l1_loss = 0.0
            for param in self.model.parameters():
                l1_loss += param.abs().sum()
            reg_losses['l1_penalty'] = self.config.l1_penalty * l1_loss
        
        # Orthogonal penalty for attention weights
        if hasattr(self.config, 'orthogonal_penalty') and self.config.orthogonal_penalty > 0:
            ortho_loss = self._compute_orthogonal_penalty()
            if ortho_loss is not None:
                reg_losses['orthogonal_penalty'] = self.config.orthogonal_penalty * ortho_loss
        
        # Activation regularization
        if hasattr(self.config, 'activation_penalty') and self.config.activation_penalty > 0:
            act_loss = self._compute_activation_penalty()
            if act_loss is not None:
                reg_losses['activation_penalty'] = self.config.activation_penalty * act_loss
        
        return reg_losses
    
    def _compute_orthogonal_penalty(self) -> Optional[torch.Tensor]:
        """Compute orthogonality penalty for attention matrices"""
        penalty = 0.0
        count = 0
        
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'weight'):
                W = module.weight
                if W.dim() >= 2:
                    # Reshape to 2D if needed
                    W_2d = W.view(W.size(0), -1)
                    
                    # Compute WW^T - I
                    WWT = torch.mm(W_2d, W_2d.t())
                    I = torch.eye(WWT.size(0), device=W.device)
                    penalty += (WWT - I).pow(2).sum()
                    count += 1
        
        return penalty / count if count > 0 else None
    
    def _compute_activation_penalty(self) -> Optional[torch.Tensor]:
        """Compute penalty on large activations"""
        # This would typically be computed during forward pass
        # and stored in the model. For now, return None.
        return None
    
    def get_label_smoothing_loss(self, 
                                logits: torch.Tensor,
                                targets: torch.Tensor,
                                loss_type: str = "grid") -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model predictions [B, ..., C]
            targets: Ground truth labels [B, ...]
            loss_type: "grid" or "tokens"
            
        Returns:
            Smoothed cross-entropy loss
        """
        smoothing = (self.config.label_smoothing_grid if loss_type == "grid" 
                    else self.config.label_smoothing_tokens)
        
        if smoothing == 0.0:
            return nn.functional.cross_entropy(logits, targets)
        
        # Reshape for processing
        original_shape = logits.shape
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Compute smooth targets
        num_classes = logits.size(-1)
        confidence = 1.0 - smoothing
        smooth_positive = confidence
        smooth_negative = smoothing / (num_classes - 1)
        
        # Create smooth target distribution
        smooth_targets = torch.full_like(logits_flat, smooth_negative)
        smooth_targets.scatter_(1, targets_flat.unsqueeze(1), smooth_positive)
        
        # Compute KL divergence loss
        log_probs = nn.functional.log_softmax(logits_flat, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        stats = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'best_metric': self.best_metric,
            'current_lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Learning rate statistics
        if self.lr_history:
            stats['lr_mean'] = np.mean(self.lr_history[-100:])  # Last 100 steps
            stats['lr_std'] = np.std(self.lr_history[-100:])
            stats['lr_min'] = min(self.lr_history)
            stats['lr_max'] = max(self.lr_history)
        
        # Gradient statistics
        if self.gradient_norms:
            recent_grads = self.gradient_norms[-100:]  # Last 100 steps
            stats['grad_norm_mean'] = np.mean(recent_grads)
            stats['grad_norm_std'] = np.std(recent_grads)
            stats['grad_norm_max'] = max(recent_grads)
            stats['grad_explosion_rate'] = sum(1 for g in recent_grads if g > 10.0) / len(recent_grads)
        
        # Optimizer state
        stats['optimizer_type'] = type(self.optimizer).__name__
        stats['scheduler_type'] = self.config.scheduler_type
        
        return stats
    
    def save_state(self, path: str):
        """Save optimizer state"""
        state = {
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'adaptive_scheduler_state': (self.adaptive_scheduler.state_dict() 
                                       if self.adaptive_scheduler else None),
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'best_metric': self.best_metric,
            'config': self.config,
            'lr_history': self.lr_history,
            'gradient_norms': self.gradient_norms
        }
        
        torch.save(state, path)
        logger.info(f"Saved optimizer state to {path}")
    
    def load_state(self, path: str):
        """Load optimizer state"""
        state = torch.load(path, map_location='cpu')
        
        self.optimizer.load_state_dict(state['optimizer_state'])
        
        if self.scheduler and state['scheduler_state']:
            self.scheduler.load_state_dict(state['scheduler_state'])
        
        if self.adaptive_scheduler and state['adaptive_scheduler_state']:
            self.adaptive_scheduler.load_state_dict(state['adaptive_scheduler_state'])
        
        self.step_count = state['step_count']
        self.epoch_count = state['epoch_count']
        self.best_metric = state['best_metric']
        self.lr_history = state.get('lr_history', [])
        self.gradient_norms = state.get('gradient_norms', [])
        
        logger.info(f"Loaded optimizer state from {path}")


class AdvancedRegularizer:
    """Additional regularization techniques for TOPAS"""
    
    @staticmethod
    def mixup_loss(logits1: torch.Tensor, 
                   logits2: torch.Tensor,
                   targets1: torch.Tensor,
                   targets2: torch.Tensor,
                   lam: float) -> torch.Tensor:
        """
        Mixup loss for improved generalization.
        
        Args:
            logits1, logits2: Model predictions for mixed samples
            targets1, targets2: Ground truth for original samples
            lam: Mixing parameter
            
        Returns:
            Mixed loss
        """
        loss1 = nn.functional.cross_entropy(logits1, targets1)
        loss2 = nn.functional.cross_entropy(logits2, targets2)
        return lam * loss1 + (1 - lam) * loss2
    
    @staticmethod
    def cutmix_loss(logits: torch.Tensor,
                    targets_a: torch.Tensor,
                    targets_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
        """
        CutMix loss for spatial augmentation.
        
        Args:
            logits: Model predictions
            targets_a, targets_b: Ground truth for mixed samples
            lam: Area ratio
            
        Returns:
            CutMix loss
        """
        loss_a = nn.functional.cross_entropy(logits, targets_a)
        loss_b = nn.functional.cross_entropy(logits, targets_b)
        return lam * loss_a + (1 - lam) * loss_b
    
    @staticmethod
    def consistency_loss(logits1: torch.Tensor, 
                        logits2: torch.Tensor,
                        temperature: float = 1.0) -> torch.Tensor:
        """
        Consistency loss between different augmented views.
        
        Args:
            logits1, logits2: Predictions for different augmentations
            temperature: Temperature for softmax
            
        Returns:
            KL divergence consistency loss
        """
        p1 = nn.functional.softmax(logits1 / temperature, dim=-1)
        p2 = nn.functional.softmax(logits2 / temperature, dim=-1)
        
        # Symmetric KL divergence
        kl1 = nn.functional.kl_div(p1.log(), p2, reduction='batchmean')
        kl2 = nn.functional.kl_div(p2.log(), p1, reduction='batchmean')
        
        return (kl1 + kl2) / 2