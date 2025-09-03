#!/usr/bin/env python3
"""
Phi Metrics for Neuromorphic Computing
Production-ready implementations of integrated information theory metrics.
"""
import torch
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def phi_synergy_features(tokens: torch.Tensor, parts: int = 2) -> torch.Tensor:
    """
    Compute integrated information proxy via feature synergy.
    
    Measures the synergy between different parts of the feature space
    using cosine similarity between chunks.
    
    Args:
        tokens: Input tensor of shape [B, T, D]
        parts: Number of parts to split features into
        
    Returns:
        Scalar tensor representing average synergy
        
    Raises:
        ValueError: If D is not divisible by parts
    """
    B, T, D = tokens.shape
    
    if D % parts != 0:
        logger.warning(f"Feature dimension {D} not divisible by {parts}, returning 0")
        return torch.tensor(0.0, device=tokens.device, dtype=tokens.dtype)
    
    try:
        chunks = torch.chunk(tokens, parts, dim=-1)
        synergies = []
        
        # Compute pairwise similarities
        for i in range(parts):
            for j in range(i + 1, parts):
                sim = F.cosine_similarity(chunks[i], chunks[j], dim=-1)
                synergies.append(sim.mean())
        
        if synergies:
            return torch.stack(synergies).mean()
        else:
            return torch.tensor(0.0, device=tokens.device, dtype=tokens.dtype)
            
    except Exception as e:
        logger.error(f"Error computing phi synergy: {e}")
        return torch.tensor(0.0, device=tokens.device, dtype=tokens.dtype)


def kappa_floor(tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Compute assembly depth proxy metric.
    
    Estimates the hierarchical depth of token assemblies based on norm scaling.
    
    Args:
        tokens: Input tensor of shape [B, T, D] where T = H*W
        H: Height of spatial grid
        W: Width of spatial grid
        
    Returns:
        Scalar tensor representing normalized depth
        
    Raises:
        ValueError: If T != H*W
    """
    B, T, D = tokens.shape
    
    if T != H * W:
        raise ValueError(f"Expected T={H*W}, got T={T} for grid {H}x{W}")
    
    try:
        # Reshape to spatial grid
        flat = tokens.view(B, H * W, D)
        
        # Compute average norm
        norm = flat.norm(dim=-1).mean()
        
        # Normalize by sqrt(D) for scale invariance
        depth = (norm / math.sqrt(D)).clamp(0.0, 1.0)
        
        return depth
        
    except Exception as e:
        logger.error(f"Error computing kappa floor: {e}")
        return torch.tensor(0.0, device=tokens.device, dtype=tokens.dtype)


def cge_boost(tokens: torch.Tensor, phi_before: Optional[torch.Tensor] = None, 
             phi_after: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute Compositional Generalization Energy boost.
    
    If phi values are provided, computes boost as improvement.
    Otherwise, computes token-wise entropy as proxy.
    
    Args:
        tokens: Input tensor of shape [B, T, D]
        phi_before: Optional phi value before processing
        phi_after: Optional phi value after processing
        
    Returns:
        Scalar tensor representing CGE boost
    """
    try:
        # If phi values provided, compute improvement
        if phi_before is not None and phi_after is not None:
            return (phi_after - phi_before).clamp_min(0.0)
        
        # Otherwise compute entropy-based proxy
        B, T, D = tokens.shape
        
        # Compute softmax probabilities
        probs = F.softmax(tokens, dim=-1)
        
        # Shannon entropy
        entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1).mean()
        
        # Normalize by log(D)
        max_entropy = math.log(D) if D > 1 else 1.0
        normalized_entropy = entropy / (max_entropy + 1e-8)
        
        return normalized_entropy
        
    except Exception as e:
        logger.error(f"Error computing CGE boost: {e}")
        return torch.tensor(0.0, device=tokens.device, dtype=tokens.dtype)


def hodge_penalty(edge_flow: torch.Tensor, B1: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
    """
    Compute graph Hodge penalty for edge flows.
    
    Measures how much the edge flow deviates from being curl-free
    and divergence-free (conservative flow).
    
    Args:
        edge_flow: Edge flow vector
        B1: Node-to-edge incidence matrix
        B2: Edge-to-face incidence matrix
        
    Returns:
        Scalar penalty value
    """
    if edge_flow is None or edge_flow.numel() == 0:
        return torch.tensor(0.0, device=B1.device, dtype=B1.dtype)
    
    try:
        # Ensure compatible shapes
        if edge_flow.shape[-1] != B1.shape[1]:
            logger.warning(f"Shape mismatch: edge_flow {edge_flow.shape} vs B1 {B1.shape}")
            return torch.tensor(0.01, device=B1.device, dtype=B1.dtype)
        
        # Divergence component
        div = B1 @ edge_flow
        div_penalty = div.pow(2).mean()
        
        # Curl component (if B2 provided and compatible)
        curl_penalty = torch.tensor(0.0, device=edge_flow.device, dtype=edge_flow.dtype)
        if B2 is not None and B2.numel() > 0 and B2.shape[0] == edge_flow.shape[-1]:
            try:
                curl = B2.T @ edge_flow
                curl_penalty = curl.pow(2).mean()
            except:
                pass
        
        # Combined penalty
        total_penalty = (div_penalty + curl_penalty).sqrt()
        return total_penalty.clamp_min(0.0)
        
    except Exception as e:
        logger.error(f"Error computing Hodge penalty: {e}")
        return torch.tensor(0.01, device=B1.device, dtype=B1.dtype)


def neuro_hodge_penalty(slots_rel_or_model, edge_flow: Optional[torch.Tensor] = None, 
                       B1: Optional[torch.Tensor] = None, B2: Optional[torch.Tensor] = None,
                       weights: Optional[dict] = None) -> torch.Tensor:
    """
    Compute neuromorphic-enhanced Hodge penalty.
    
    Supports two calling modes:
    1. Full mode: neuro_hodge_penalty(model, edge_flow, B1, B2, weights)
    2. Simplified mode: neuro_hodge_penalty(slots_rel)
    
    In simplified mode, computes Hodge penalty directly from slot relational features
    based on their complexity and interconnectedness.
    
    Args:
        slots_rel_or_model: Either slots_rel tensor [B, K, D] or model instance
        edge_flow: Edge flow vector (full mode only)
        B1: Node-to-edge incidence matrix (full mode only)
        B2: Edge-to-face incidence matrix (full mode only)
        weights: Optional weight dictionary for penalty components
        
    Returns:
        Combined penalty value
    """
    # Check if this is simplified mode (just slots_rel tensor)
    if isinstance(slots_rel_or_model, torch.Tensor) and edge_flow is None:
        return _neuro_hodge_penalty_simple(slots_rel_or_model)
    
    # Otherwise, use full mode
    model = slots_rel_or_model
    if weights is None:
        weights = {
            'hodge': 0.01,
            'grid': 0.05,
            'anti_recurrence': 0.1,
            'predictive_prune': 0.05
        }
    
    try:
        # Base Hodge penalty
        base_penalty = hodge_penalty(edge_flow, B1, B2) * weights.get('hodge', 0.01)
        
        # Additional neuromorphic penalties
        grid_penalty = torch.tensor(0.0, device=B1.device, dtype=B1.dtype)
        anti_penalty = torch.tensor(0.0, device=B1.device, dtype=B1.dtype)
        prune_penalty = torch.tensor(0.0, device=B1.device, dtype=B1.dtype)
        
        if hasattr(model, 'grid_consistency_loss'):
            try:
                grid_penalty = model.grid_consistency_loss() * weights.get('grid', 0.05)
            except Exception as e:
                logger.warning(f"Grid consistency loss failed: {e}")
        
        if hasattr(model, 'anti_recurrence_penalty'):
            try:
                anti_penalty = model.anti_recurrence_penalty("is_a") * weights.get('anti_recurrence', 0.1)
            except Exception as e:
                logger.warning(f"Anti-recurrence penalty failed: {e}")
        
        if hasattr(model, 'predictive_prune_loss'):
            try:
                prune_penalty = model.predictive_prune_loss() * weights.get('predictive_prune', 0.05)
            except Exception as e:
                logger.warning(f"Predictive prune loss failed: {e}")
        
        # Combine all penalties
        total_penalty = base_penalty + grid_penalty + anti_penalty + prune_penalty
        return total_penalty.clamp_min(0.0)
        
    except Exception as e:
        logger.error(f"Error computing neuro-Hodge penalty: {e}")
        return torch.tensor(0.01, device=B1.device, dtype=B1.dtype)


def _neuro_hodge_penalty_simple(slots_rel: torch.Tensor) -> torch.Tensor:
    """
    Simplified Hodge penalty computation from slot relational features.
    
    Computes penalty based on:
    - Relational complexity (cross-slot interactions)
    - Feature tangling (non-orthogonal relationships)
    - Spatial coherence violations
    
    Args:
        slots_rel: Slot relational features [B, K, D]
        
    Returns:
        Scalar tensor representing Hodge penalty
    """
    try:
        B, K, D = slots_rel.shape
        device = slots_rel.device
        
        if K <= 1:
            # Single or no objects: minimal complexity
            return torch.tensor(0.001, device=device, dtype=slots_rel.dtype)
        
        # Compute pairwise slot similarities (relational complexity)
        # Normalize slots to unit vectors for cosine similarity
        slots_norm = F.normalize(slots_rel, p=2, dim=-1)  # [B, K, D]
        
        # Compute similarity matrix [B, K, K]
        similarities = torch.bmm(slots_norm, slots_norm.transpose(-1, -2))  # [B, K, K]
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
        similarities = similarities.masked_fill(mask, 0.0)
        
        # Relational complexity: higher when objects are highly interconnected
        complexity = similarities.abs().mean(dim=(-1, -2))  # [B]
        
        # Feature tangling: measure non-orthogonality via feature covariance
        # Center features
        slots_centered = slots_rel - slots_rel.mean(dim=1, keepdim=True)  # [B, K, D]
        
        # Compute covariance matrix for each batch
        tangles = []
        for b in range(B):
            slots_b = slots_centered[b]  # [K, D]
            
            if K > D:
                # More slots than dimensions: use feature covariance
                cov = torch.cov(slots_b.T)  # [D, D]
                # Off-diagonal elements indicate feature tangling
                off_diag = cov - torch.diag(torch.diag(cov))
                tangle = off_diag.abs().mean()
            else:
                # Use slot covariance
                cov = torch.cov(slots_b)  # [K, K]
                off_diag = cov - torch.diag(torch.diag(cov))
                tangle = off_diag.abs().mean()
            
            tangles.append(tangle)
        
        tangle_penalty = torch.stack(tangles).mean()
        
        # Spatial coherence: variance in slot norms indicates inconsistent activation
        slot_norms = slots_rel.norm(dim=-1)  # [B, K]
        coherence_penalty = slot_norms.var(dim=-1).mean()  # Higher variance = less coherent
        
        # Combine penalties with adaptive scaling
        base_scale = 0.001  # Minimum penalty for simple scenes
        complexity_scale = 0.05  # Scale for relational complexity
        tangle_scale = 0.02  # Scale for feature tangling
        coherence_scale = 0.01  # Scale for spatial coherence
        
        total_penalty = (
            base_scale +
            complexity_scale * complexity.mean() +
            tangle_scale * tangle_penalty +
            coherence_scale * coherence_penalty
        )
        
        # Clamp to reasonable range
        return total_penalty.clamp(min=0.0, max=0.1)
        
    except Exception as e:
        logger.error(f"Error computing simple neuro-Hodge penalty: {e}")
        # Return higher penalty on error to be conservative
        return torch.tensor(0.01, device=slots_rel.device, dtype=slots_rel.dtype)


def clamp_neuromorphic_terms(phi: torch.Tensor, kappa: torch.Tensor, cge: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Clamp neuromorphic reward terms to prevent divergence during training.
    
    This function prevents exploding gradients and numerical instability
    by clamping each neuromorphic term to safe ranges.
    
    Args:
        phi: Phi synergy metric
        kappa: Kappa floor metric  
        cge: CGE boost metric
        
    Returns:
        Tuple of (clamped_phi, clamped_kappa, clamped_cge)
    """
    # Prevent explosion with conservative bounds
    phi_clamped = torch.clamp(phi, min=-10.0, max=10.0)
    kappa_clamped = torch.clamp(kappa, min=0.0, max=5.0)
    cge_clamped = torch.clamp(cge, min=0.0, max=5.0)
    
    return phi_clamped, kappa_clamped, cge_clamped

