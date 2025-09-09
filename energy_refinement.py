
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict
from phi_metrics_neuro import phi_synergy_features, kappa_floor, cge_boost, clamp_neuromorphic_terms

class EnergyRefiner(nn.Module):
    # Iterative "System-2" refinement head with temperature annealing and early stopping.
    # Minimizes E = L_fit + λ_viol * L_violation + λ_prior * (Φ + κ + CGE + Hodge).
    def __init__(self, min_steps:int=3, max_steps:int=7, step_size:float=0.25, noise:float=0.0, 
                 lambda_violation:float=1.0, lambda_prior:float=1e-3, lambda_size:float=0.0,
                 w_phi:float=0.1, w_kappa:float=0.1, w_cge:float=0.1, 
                 temp_schedule=None, early_stop_threshold:float=1e-4, verbose:bool=False):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.step_size = step_size
        self.noise = noise
        self.lambda_violation = lambda_violation
        self.lambda_prior = lambda_prior
        self.lambda_size = lambda_size
        self.w_phi = w_phi
        self.w_kappa = w_kappa
        self.w_cge = w_cge
        self.early_stop_threshold = early_stop_threshold
        self.verbose = verbose
        
        # Temperature annealing schedule
        if temp_schedule is None:
            self.temp_schedule = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
        else:
            self.temp_schedule = temp_schedule

    def forward(self, logits: torch.Tensor, constraint_obj, prior_tensors: dict, 
                prior_scales: Optional[Dict[str, float]] = None, 
                extras: Optional[Dict] = None):
        # logits: Float tensor, e.g., [B, C, H, W]; must be differentiable
        # constraint_obj must expose: fit_loss(logits), violation_loss(logits)
        # prior_tensors: dict of optional scalar tensors/floats: {'phi','kappa','cge','hodge'}
        # make a leaf tensor that can get grads, regardless of upstream context
        x = logits.detach().clone().requires_grad_(True)

        # Apply prior scaling with safe defaults
        if prior_scales is None:
            prior_scales = {"phi": 1.0, "kappa": 1.0, "cge": 1.0}
            
        # ensure autograd is ON inside refinement (eval often uses no_grad)
        with torch.enable_grad():
            for step in range(self.max_steps):
                # Anneal softmax temperature for better convergence
                temp_idx = min(step, len(self.temp_schedule) - 1)
                temp = self.temp_schedule[temp_idx]
                
                # Apply temperature to logits
                x_temp = x / temp
                
                fit = constraint_obj.fit_loss(x_temp)
                viol = constraint_obj.violation_loss(x_temp)
                
                # Early stopping if violation is low enough and we've done minimum steps
                if step >= self.min_steps and viol < self.early_stop_threshold:
                    if self.verbose:
                        import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[EBR] Early stopping at step {step}, violation={viol:.6f}")
                    break
                
                # Derive "soft tokens" from x_temp: [B, C, H, W] -> [B, H*W, C]
                B, C, H, W = x_temp.shape
                soft = torch.softmax(x_temp, dim=1).permute(0,2,3,1).reshape(B, H*W, C)
                
                # Compute differentiable φ/κ/CGE penalties w.r.t. x
                phi = phi_synergy_features(soft, parts=2)
                kappa = kappa_floor(soft, H, W)
                cge = cge_boost(soft)
                
                # Clamp neuromorphic terms to prevent divergence
                phi, kappa, cge = clamp_neuromorphic_terms(phi, kappa, cge)
                
                # Include hodge term from prior_tensors if provided
                hodge = prior_tensors.get("hodge", 0.0)
                hodge_term = hodge.mean() if torch.is_tensor(hodge) else float(hodge)
                
                # Scale internal priors by prior_scales
                w_phi_scaled = self.w_phi * prior_scales.get("phi", 1.0)
                w_kappa_scaled = self.w_kappa * prior_scales.get("kappa", 1.0)
                w_cge_scaled = self.w_cge * prior_scales.get("cge", 1.0)
                
                # Form weighted prior term: pri = wφ*φ + wκ*κ + wCGE*CGE + hodge
                pri = w_phi_scaled * phi + w_kappa_scaled * kappa + w_cge_scaled * cge + hodge_term
                
                # Add size constraint loss
                size_constraints = prior_tensors.get("size_constraints", None)
                size_loss = torch.tensor(0.0, device=x.device)
                if size_constraints and self.lambda_size > 0:
                    size_loss = self._compute_size_loss(x_temp, size_constraints)
                
                E = fit + self.lambda_violation * viol + self.lambda_prior * pri + self.lambda_size * size_loss
                
                # Debug print for first step to verify priors influence
                # Debug logging if verbose
                if step == 0 and self.verbose:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"[EBR] step={step}, temp={temp:.2f}, fit={fit:.4f}, viol={viol:.4f}, size={size_loss:.4f}")
                    logger.debug(f"[EBR] phi={phi:.4f}, kappa={kappa:.4f}, cge={cge:.4f}, pri={pri:.4f}")
                
                # PROPER FIDELITY FIX: Prevent double-backward graph errors
                try:
                    E.backward(retain_graph=True)
                except RuntimeError as e:
                    if "backward through the graph a second time" in str(e):
                        # Graph already freed, skip this iteration
                        if self.verbose:
                            print(f"[EBR] Graph freed, stopping at step {step}")
                        break
                    else:
                        raise e
                with torch.no_grad():
                    # Use temperature-adjusted step size for better convergence
                    adjusted_step_size = self.step_size * temp
                    x -= adjusted_step_size * x.grad
                    
                    if self.noise > 0:
                        x += self.noise * torch.randn_like(x)
                    x.grad.zero_()

        return x.detach()
    
    def _compute_size_loss(self, logits: torch.Tensor, size_constraints: dict) -> torch.Tensor:
        """Compute size constraint loss based on predicted vs actual output size"""
        if not size_constraints or 'predicted_size' not in size_constraints:
            return torch.tensor(0.0, device=logits.device)
        
        predicted_size = size_constraints['predicted_size']
        confidence = size_constraints.get('confidence', 1.0)
        
        # Current logits size
        B, C, H, W = logits.shape
        actual_size = (H, W)
        
        # Size mismatch penalty
        if actual_size != predicted_size:
            # L1 distance between sizes, weighted by confidence
            size_diff = abs(actual_size[0] - predicted_size[0]) + abs(actual_size[1] - predicted_size[1])
            size_penalty = confidence * size_diff * 0.1
            
            if self.verbose:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"[EBR] Size mismatch: actual={actual_size}, predicted={predicted_size}, penalty={size_penalty:.4f}")
            
            return torch.tensor(size_penalty, device=logits.device)
        
        # Size histogram matching if predicted size matches
        if 'demos' in size_constraints:
            try:
                hist_loss = self._compute_histogram_loss(logits, size_constraints['demos'])
                return hist_loss * confidence * 0.05  # Small weight for histogram matching
            except Exception as e:
                if self.verbose:
                    print(f"[EBR] Histogram loss failed: {e}")
        
        return torch.tensor(0.0, device=logits.device)
    
    def _compute_histogram_loss(self, logits: torch.Tensor, demos) -> torch.Tensor:
        """Compute histogram matching loss for size-correct outputs"""
        if not demos:
            return torch.tensor(0.0, device=logits.device)
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        pred_hist = probs.mean(dim=(2, 3))    # [B, C] - average color distribution
        
        # Get target histogram from demo outputs
        target_hists = []
        for demo in demos:
            if isinstance(demo, (tuple, list)) and len(demo) >= 2:
                output = demo[1]
                if hasattr(output, 'cpu') and hasattr(output, 'numpy'):
                    output = output.cpu().numpy()
                if isinstance(output, np.ndarray):
                    output = torch.from_numpy(output)
                elif not isinstance(output, torch.Tensor):
                    continue
                
                # Compute histogram of output
                output_flat = output.flatten()
                hist = torch.bincount(output_flat.long(), minlength=10).float()
                hist = hist / hist.sum().clamp(min=1e-8)  # Normalize
                target_hists.append(hist[:logits.size(1)])  # Match channel count
        
        if not target_hists:
            return torch.tensor(0.0, device=logits.device)
        
        # Average target histogram
        target_hist = torch.stack(target_hists).mean(dim=0).to(logits.device)
        target_hist = target_hist.unsqueeze(0).expand(pred_hist.size(0), -1)  # [B, C]
        
        # L2 loss between predicted and target histograms
        hist_loss = F.mse_loss(pred_hist, target_hist)
        return hist_loss
