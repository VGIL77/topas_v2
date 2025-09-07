"""
HRM-TOPAS Bridge Module
Implements bidirectional communication between HRM's hierarchical reasoning and TOPAS execution.

Key Integration Points:
- HRM H-level (slow, abstract planning) guides high-level DSL operation selection
- HRM L-level (rapid computation) interfaces with TOPAS EnergyRefiner
- HRM adaptive halting determines DSL search depth
- Puzzle embeddings from HRM incorporated into TOPAS encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class HRMTOPASIntegrationConfig:
    """Configuration for HRM-TOPAS integration."""
    hrm_hidden_size: int = 512
    topas_width: int = 256
    num_attention_heads: int = 8
    cross_attention_dropout: float = 0.1
    puzzle_emb_dim: int = 128
    dsl_ops_count: int = 25  # Number of DSL operations
    adaptive_halting_threshold: float = 0.5
    max_planning_steps: int = 6


class CrossAttentionLayer(nn.Module):
    """Cross-attention between HRM reasoning states and TOPAS grid features."""
    
    def __init__(self, config: HRMTOPASIntegrationConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.topas_width // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(config.topas_width, config.topas_width)
        self.k_proj = nn.Linear(config.hrm_hidden_size, config.topas_width)
        self.v_proj = nn.Linear(config.hrm_hidden_size, config.topas_width)
        self.o_proj = nn.Linear(config.topas_width, config.topas_width)
        
        self.dropout = nn.Dropout(config.cross_attention_dropout)
        self.layer_norm = nn.LayerNorm(config.topas_width)
        
    def forward(self, topas_features: torch.Tensor, hrm_states: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention between TOPAS features and HRM states.
        
        Args:
            topas_features: [B, H*W, topas_width] - TOPAS grid features (flattened)
            hrm_states: [B, seq_len, hrm_hidden_size] - HRM reasoning states
            
        Returns:
            attended_features: [B, H*W, topas_width] - Attended TOPAS features
        """
        batch_size, seq_len, _ = topas_features.shape
        
        # Compute Q, K, V
        Q = self.q_proj(topas_features)  # [B, H*W, topas_width]
        K = self.k_proj(hrm_states)     # [B, seq_len, topas_width]
        V = self.v_proj(hrm_states)     # [B, seq_len, topas_width]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.config.topas_width
        )
        
        # Output projection
        attended_features = self.o_proj(attn_output)
        
        # Residual connection and layer norm
        return self.layer_norm(topas_features + attended_features)


class HRMGuidedDSLSelector(nn.Module):
    """HRM-guided DSL operation selector with hierarchical reasoning."""
    
    def __init__(self, config: HRMTOPASIntegrationConfig):
        super().__init__()
        self.config = config
        
        # H-level to DSL operation mapping
        self.h_to_dsl = nn.Linear(config.hrm_hidden_size, config.dsl_ops_count)
        
        # L-level refinement layer
        self.l_refinement = nn.Linear(config.hrm_hidden_size, config.hrm_hidden_size)
        
        # Combined reasoning layer
        self.combined_projection = nn.Linear(
            config.hrm_hidden_size * 2, config.dsl_ops_count
        )
        
        # Adaptive halting controller
        self.halting_controller = nn.Linear(config.hrm_hidden_size, 2)  # halt/continue
        
    def forward(self, z_H: torch.Tensor, z_L: torch.Tensor, 
                current_depth: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate DSL operation biases and halting decisions.
        
        Args:
            z_H: [B, hrm_hidden_size] - HRM H-level (abstract planning)
            z_L: [B, hrm_hidden_size] - HRM L-level (rapid computation)
            current_depth: Current search depth
            
        Returns:
            dsl_op_logits: [B, dsl_ops_count] - DSL operation selection biases
            control_signals: Dict with halting and refinement signals
        """
        batch_size = z_H.size(0)
        
        # H-level provides high-level operation preferences
        h_op_logits = self.h_to_dsl(z_H)
        
        # L-level provides refinement
        l_refined = torch.relu(self.l_refinement(z_L))
        
        # Combine H and L levels
        combined = torch.cat([z_H, l_refined], dim=-1)
        combined_logits = self.combined_projection(combined)
        
        # Final DSL operation logits (weighted combination)
        alpha = torch.sigmoid(torch.mean(z_H, dim=-1, keepdim=True))  # Adaptive weighting
        dsl_op_logits = alpha * h_op_logits + (1 - alpha) * combined_logits
        
        # Halting decision based on H-level state
        halt_logits = self.halting_controller(z_H)
        q_halt, q_continue = halt_logits[:, 0], halt_logits[:, 1]
        
        # Adaptive depth control
        confidence = torch.max(F.softmax(dsl_op_logits, dim=-1), dim=-1)[0]
        should_halt = (q_halt > q_continue) | (confidence > self.config.adaptive_halting_threshold)
        
        control_signals = {
            'q_halt_logits': q_halt,
            'q_continue_logits': q_continue,
            'should_halt': should_halt,
            'confidence': confidence,
            'adaptive_depth': torch.where(should_halt, 
                                        torch.clamp(torch.tensor(current_depth - 1), min=1),
                                        torch.tensor(current_depth + 1))
        }
        
        return dsl_op_logits, control_signals


class PuzzleEmbeddingIntegrator(nn.Module):
    """Integrates HRM puzzle embeddings into TOPAS grid processing."""
    
    def __init__(self, config: HRMTOPASIntegrationConfig):
        super().__init__()
        self.config = config
        
        # Projection from puzzle embedding to TOPAS width
        self.puzzle_proj = nn.Linear(config.puzzle_emb_dim, config.topas_width)
        
        # Spatial broadcast mechanism
        self.spatial_proj = nn.Conv2d(config.topas_width, config.topas_width, 1)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(config.topas_width * 2, config.topas_width, 3, padding=1),
            nn.BatchNorm2d(config.topas_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.topas_width, config.topas_width, 1)
        )
        
    def forward(self, grid_features: torch.Tensor, 
                puzzle_embedding: torch.Tensor) -> torch.Tensor:
        """
        Integrate puzzle embedding into grid features.
        
        Args:
            grid_features: [B, topas_width, H, W] - TOPAS grid features
            puzzle_embedding: [B, puzzle_emb_dim] - HRM puzzle embedding
            
        Returns:
            enhanced_features: [B, topas_width, H, W] - Enhanced grid features
        """
        B, C, H, W = grid_features.shape
        
        # Project puzzle embedding to TOPAS width
        puzzle_proj = self.puzzle_proj(puzzle_embedding)  # [B, topas_width]
        
        # Spatially broadcast puzzle information
        puzzle_spatial = puzzle_proj.view(B, -1, 1, 1).expand(B, -1, H, W)
        puzzle_spatial = self.spatial_proj(puzzle_spatial)
        
        # Concatenate and fuse
        combined = torch.cat([grid_features, puzzle_spatial], dim=1)
        enhanced_features = self.fusion(combined)
        
        # Residual connection
        return grid_features + enhanced_features


class HRMTOPASBridge(nn.Module):
    """
    Main bridge module implementing bidirectional communication between HRM and TOPAS.
    
    This module:
    1. Processes HRM reasoning states (z_H, z_L) to guide TOPAS execution
    2. Integrates HRM puzzle embeddings into TOPAS grid encoding
    3. Implements cross-attention between HRM and TOPAS representations
    4. Controls DSL search depth via HRM adaptive halting
    """
    
    def __init__(self, config: HRMTOPASIntegrationConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.cross_attention = CrossAttentionLayer(config)
        self.dsl_selector = HRMGuidedDSLSelector(config)
        self.puzzle_integrator = PuzzleEmbeddingIntegrator(config)
        
        # Bidirectional communication layers
        self.hrm_to_topas = nn.Linear(config.hrm_hidden_size, config.topas_width)
        self.topas_to_hrm = nn.Linear(config.topas_width, config.hrm_hidden_size)
        
        # Feature alignment layer
        self.feature_aligner = nn.Sequential(
            nn.Linear(config.topas_width + config.hrm_hidden_size, config.topas_width),
            nn.ReLU(),
            nn.Linear(config.topas_width, config.topas_width)
        )
        
    def forward(self, 
                grid_features: torch.Tensor,
                hrm_outputs: Dict[str, torch.Tensor],
                puzzle_embedding: Optional[torch.Tensor] = None,
                current_search_depth: int = 1) -> Dict[str, torch.Tensor]:
        """
        Forward pass implementing HRM-TOPAS integration.
        
        Args:
            grid_features: [B, topas_width, H, W] - TOPAS grid features
            hrm_outputs: Dict containing HRM states (z_H, z_L, etc.)
            puzzle_embedding: [B, puzzle_emb_dim] - Optional puzzle embedding
            current_search_depth: Current DSL search depth
            
        Returns:
            integration_outputs: Dict containing:
                - enhanced_features: Enhanced grid features
                - dsl_op_biases: DSL operation selection biases
                - control_signals: Halting and depth control signals
                - attention_features: Cross-attention enhanced features
        """
        B, C, H, W = grid_features.shape
        
        # Extract HRM states
        z_H = hrm_outputs.get('z_H')  # [B, hrm_hidden_size]
        z_L = hrm_outputs.get('z_L')  # [B, hrm_hidden_size] 
        
        if z_H is None or z_L is None:
            raise ValueError("HRM outputs must contain z_H and z_L states")
            
        # 1. Integrate puzzle embeddings if available
        if puzzle_embedding is not None:
            grid_features = self.puzzle_integrator(grid_features, puzzle_embedding)
        
        # 2. Reshape grid features for cross-attention
        grid_flat = grid_features.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 3. Prepare HRM states for cross-attention (add sequence dimension)
        hrm_combined = torch.stack([z_H, z_L], dim=1)  # [B, 2, hrm_hidden_size]
        
        # 4. Apply cross-attention
        attended_features = self.cross_attention(grid_flat, hrm_combined)  # [B, H*W, topas_width]
        
        # 5. Reshape back to spatial format
        attended_features = attended_features.transpose(1, 2).view(B, C, H, W)
        
        # 6. Generate DSL operation biases and control signals
        dsl_op_logits, control_signals = self.dsl_selector(z_H, z_L, current_search_depth)
        
        # 7. Bidirectional feature alignment
        # HRM -> TOPAS direction
        hrm_to_topas_signal = self.hrm_to_topas(z_H).view(B, -1, 1, 1).expand(B, -1, H, W)
        
        # TOPAS -> HRM direction (global pooling)
        topas_global = torch.mean(attended_features, dim=(2, 3))  # [B, topas_width]
        topas_to_hrm_signal = self.topas_to_hrm(topas_global)  # [B, hrm_hidden_size]
        
        # 8. Final feature fusion
        combined_signal = torch.cat([topas_global, z_H], dim=-1)  # [B, topas_width + hrm_hidden_size]
        alignment_weights = torch.sigmoid(self.feature_aligner(combined_signal))  # [B, topas_width]
        
        # Apply alignment weights spatially
        alignment_weights = alignment_weights.view(B, -1, 1, 1).expand(B, -1, H, W)
        enhanced_features = attended_features * alignment_weights + hrm_to_topas_signal * (1 - alignment_weights)
        
        # 9. Compile outputs
        integration_outputs = {
            'enhanced_features': enhanced_features,
            'dsl_op_biases': F.softmax(dsl_op_logits, dim=-1),
            'dsl_op_logits': dsl_op_logits,
            'control_signals': control_signals,
            'attention_features': attended_features,
            'bidirectional_signals': {
                'hrm_to_topas': hrm_to_topas_signal,
                'topas_to_hrm': topas_to_hrm_signal
            },
            'feature_alignment_weights': alignment_weights
        }
        
        return integration_outputs
    
    def extract_dsl_operation_dict(self, dsl_op_biases: torch.Tensor, 
                                 dsl_ops_list: List[str]) -> Dict[str, float]:
        """
        Convert DSL operation biases tensor to dictionary format.
        
        Args:
            dsl_op_biases: [B, dsl_ops_count] - DSL operation biases
            dsl_ops_list: List of DSL operation names
            
        Returns:
            op_bias_dict: Dict mapping operation names to bias values
        """
        if len(dsl_ops_list) != self.config.dsl_ops_count:
            raise ValueError(f"DSL ops list length {len(dsl_ops_list)} != config count {self.config.dsl_ops_count}")
            
        # Take first batch element and convert to dict
        biases = dsl_op_biases[0].detach().cpu().numpy()
        return {op: float(bias) for op, bias in zip(dsl_ops_list, biases)}
    
    def compute_adaptive_search_params(self, control_signals: Dict[str, torch.Tensor],
                                     base_depth: int, base_beam_width: int) -> Tuple[int, int]:
        """
        Compute adaptive search parameters based on HRM control signals.
        
        Args:
            control_signals: Dict containing HRM control signals
            base_depth: Base DSL search depth
            base_beam_width: Base DSL beam width
            
        Returns:
            adapted_depth: Adapted search depth
            adapted_beam_width: Adapted beam width
        """
        should_halt = control_signals.get('should_halt', torch.tensor(False))
        confidence = control_signals.get('confidence', torch.tensor(0.5))
        adaptive_depth = control_signals.get('adaptive_depth', torch.tensor(base_depth))
        
        # Extract scalar values
        if torch.is_tensor(should_halt):
            should_halt = should_halt.item() if should_halt.numel() == 1 else should_halt.any().item()
        if torch.is_tensor(confidence):
            confidence = confidence.mean().item() if confidence.numel() > 1 else confidence.item()
        if torch.is_tensor(adaptive_depth):
            adaptive_depth = int(adaptive_depth.mean().item() if adaptive_depth.numel() > 1 else adaptive_depth.item())
            
        # Adaptive depth control
        if should_halt:
            adapted_depth = max(1, min(base_depth, adaptive_depth))
        else:
            adapted_depth = min(base_depth + 1, adaptive_depth)
            
        # Adaptive beam width based on confidence
        if confidence > 0.8:  # High confidence - reduce beam width
            adapted_beam_width = max(1, base_beam_width // 2)
        elif confidence < 0.3:  # Low confidence - increase beam width  
            adapted_beam_width = min(base_beam_width * 2, 32)  # Cap at reasonable max
        else:
            adapted_beam_width = base_beam_width
            
        return adapted_depth, adapted_beam_width