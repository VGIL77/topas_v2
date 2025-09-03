
import torch
import torch.nn.functional as F
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

def softmax_logits(logits):
    return F.softmax(logits, dim=1)

class SudokuConstraints:
    # Works with logits [B, C, 9, 9]; C can be 10 (blank + 1..9) or 9 (1..9).
    def __init__(self, labels=None, known_mask=None):
        self.labels = labels
        self.known_mask = known_mask

    def fit_loss(self, logits):
        if self.labels is None or self.known_mask is None:
            probs = softmax_logits(logits)
            ent = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1).mean()
            return ent * 0.1
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0,2,3,1).reshape(-1, C)
        labels_flat = self.labels.reshape(-1)
        mask_flat   = self.known_mask.reshape(-1)
        if mask_flat.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        ce = F.cross_entropy(logits_flat[mask_flat], labels_flat[mask_flat])
        return ce

    def violation_loss(self, logits):
        probs = softmax_logits(logits)
        B, C, H, W = probs.shape
        assert H == 9 and W == 9, "SudokuConstraints expects 9x9 grid"

        row_counts = probs.sum(dim=3)
        row_pen = F.relu(row_counts - 1.0).mean()

        col_counts = probs.sum(dim=2)
        col_pen = F.relu(col_counts - 1.0).mean()

        box_pen = 0.0
        for br in range(3):
            for bc in range(3):
                box = probs[:,:, br*3:(br+1)*3, bc*3:(bc+1)*3].sum(dim=(2,3))
                box_pen = box_pen + F.relu(box - 1.0).mean()
        box_pen = box_pen / 9.0

        return row_pen + col_pen + box_pen


@dataclass
class ARCGridConstraints:
    """
    Generic ARC invariants for logits [B, C, H, W] enriched with κ/CGE priors and size constraints.
    
    Args:
        expect_symmetry: Expected symmetry type ('h', 'v', 'rot90', 'rot180', 'rot270')
        color_hist: Target color histogram
        sparsity: Target sparsity level
        kappa: κ (kappa) tensor from phi_metrics - assembly depth proxy
        cge: CGE tensor from phi_metrics - compositional generalization energy
        lam_kappa: Weight for kappa penalty (default: 1e-3)
        lam_cge: Weight for CGE penalty (default: 1e-3)
        expected_size: (H, W) tuple for expected output size
        size_confidence: Confidence in size prediction (0.0-1.0)
        lam_size: Weight for size constraint penalty (default: 1.0)
    """
    expect_symmetry: Optional[str] = None
    color_hist: Optional[torch.Tensor] = None
    sparsity: Optional[float] = None
    kappa: Optional[torch.Tensor] = None
    cge: Optional[torch.Tensor] = None
    lam_kappa: float = 1e-3
    lam_cge: float = 1e-3
    expected_size: Optional[tuple] = None  # Size constraint
    size_confidence: float = 1.0  # Size prediction confidence
    lam_size: float = 1.0  # Size constraint weight

    def fit_loss(self, logits):
        probs = softmax_logits(logits)
        ent = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1).mean()
        return ent * 0.05

    def violation_loss(self, logits):
        probs = softmax_logits(logits)
        B, C, H, W = probs.shape
        viol = torch.tensor(0.0, device=logits.device)

        # Original constraints
        if self.expect_symmetry is not None:
            if self.expect_symmetry == 'h':
                probs_flip = probs.flip(dims=[3])
            elif self.expect_symmetry == 'v':
                probs_flip = probs.flip(dims=[2])
            elif self.expect_symmetry == 'rot90':
                probs_flip = probs.transpose(2,3).flip(3)
            elif self.expect_symmetry == 'rot180':
                probs_flip = probs.flip(dims=[2,3])
            elif self.expect_symmetry == 'rot270':
                probs_flip = probs.transpose(2,3).flip(2)
            else:
                probs_flip = probs
            viol = viol + F.mse_loss(probs, probs_flip)

        if self.color_hist is not None:
            target = self.color_hist.to(logits.device)
            if target.sum() > 1.5:
                target = target / target.sum()
            pred_hist = probs.mean(dim=(2,3))
            target = target.view(1, -1).expand_as(pred_hist)
            viol = viol + F.mse_loss(pred_hist, target)

        if self.sparsity is not None:
            background_prob = probs[:,0].mean()
            viol = viol + torch.clamp(self.sparsity - background_prob, min=0.0)

        # κ/CGE priors - higher κ/CGE reduces violation (acts as reward)
        kappa_reward = torch.tensor(0.0, device=logits.device)
        cge_reward = torch.tensor(0.0, device=logits.device)
        
        if self.kappa is not None:
            # κ reward: -lam_kappa * kappa.clamp(0,1)
            # High κ → high reward (negative penalty), low κ → low reward
            kappa_clamped = self.kappa.clamp(0.0, 1.0)
            kappa_reward = -self.lam_kappa * kappa_clamped
            viol = viol + kappa_reward  # Adding negative value reduces total violation
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"κ prior active: κ={self.kappa.item():.4f}, reward={-kappa_reward.item():.6f}")

        if self.cge is not None:
            # CGE reward: -lam_cge * torch.sigmoid(cge)
            # High CGE → high reward (negative penalty), low CGE → low reward
            cge_sigmoid = torch.sigmoid(self.cge)
            cge_reward = -self.lam_cge * cge_sigmoid
            viol = viol + cge_reward  # Adding negative value reduces total violation
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"CGE prior active: CGE={self.cge.item():.4f}, reward={-cge_reward.item():.6f}")

        # Size constraint violation
        if self.expected_size is not None:
            B, C, H, W = probs.shape
            actual_size = (H, W)
            
            if actual_size != self.expected_size:
                # Size mismatch penalty
                size_diff = abs(actual_size[0] - self.expected_size[0]) + abs(actual_size[1] - self.expected_size[1])
                size_penalty = self.size_confidence * size_diff * self.lam_size * 0.1
                viol = viol + size_penalty
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Size constraint violation: actual={actual_size}, expected={self.expected_size}, penalty={size_penalty:.4f}")
        
        return viol
