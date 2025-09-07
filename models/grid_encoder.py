import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

class ResidualBlock(nn.Module):
    """True residual block with skip connections and proper initialization"""
    def __init__(self, channels, num_convs=2):
        super().__init__()
        self.num_convs = num_convs
        
        # Main convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_convs):
            conv = nn.Conv2d(channels, channels, 3, padding=1)
            # He initialization for ReLU networks
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm2d(channels))
        
        # Zero-initialize the last BN in each residual block,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if self.norms:
            nn.init.constant_(self.norms[-1].weight, 0)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass with residual connection"""
        identity = x
        
        out = x
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.norms[i](out)
            if i < self.num_convs - 1:  # Apply ReLU except on last layer
                out = self.relu(out)
        
        # Add residual connection
        out = out + identity
        out = self.relu(out)  # Final activation after residual addition
        
        return out

class ProjectedResidualBlock(nn.Module):
    """Residual block with projection for dimension changes and proper initialization"""
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.num_convs = num_convs
        
        # Main convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First conv changes dimensions
        first_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        nn.init.kaiming_normal_(first_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(first_conv.bias, 0)
        self.convs.append(first_conv)
        self.norms.append(nn.BatchNorm2d(out_channels))
        
        # Remaining convs maintain dimensions
        for i in range(1, num_convs):
            conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm2d(out_channels))
        
        # Zero-initialize the last BN
        if self.norms:
            nn.init.constant_(self.norms[-1].weight, 0)
        
        # Projection layer for skip connection
        self.projection = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.projection.bias, 0)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass with projected residual connection"""
        identity = self.projection(x)
        
        out = x
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.norms[i](out)
            if i < self.num_convs - 1:  # Apply ReLU except on last layer
                out = self.relu(out)
        
        # Add residual connection
        out = out + identity
        out = self.relu(out)  # Final activation after residual addition
        
        return out

class HRMAwareGridEncoder(nn.Module):
    """
    HRM-aware grid encoder that can incorporate HRM puzzle embeddings and reasoning states.
    Maintains compatibility with original GridEncoder interface.
    """
    def __init__(self, width=256, depth=10, hrm_integration=False, hrm_hidden_size=512, puzzle_emb_dim=128):
        super().__init__()
        self.width = width
        self.depth = depth
        self.hrm_integration = hrm_integration
        
        # Basic grid embedding
        self.embed = nn.Embedding(10, width)  # 10 colors in ARC
        
        # HRM integration components (optional)
        if hrm_integration:
            self.hrm_proj = nn.Linear(hrm_hidden_size, width)
            self.puzzle_emb_proj = nn.Linear(puzzle_emb_dim, width) if puzzle_emb_dim > 0 else None
            self.hrm_fusion = nn.Conv2d(width * 2, width, 1)  # Fuse HRM signals with grid
        
        # Create residual block structure based on depth parameter
        self.blocks = nn.ModuleList()
        
        # Calculate number of residual blocks (2-4 convs per block)
        convs_per_block = min(4, max(2, depth // 4))  # Adaptive convs per block
        num_blocks = max(1, depth // convs_per_block)
        
        print(f"[GridEncoder] Creating {num_blocks} residual blocks with {convs_per_block} convs each (target depth: {depth})")
        if hrm_integration:
            print(f"[GridEncoder] HRM integration enabled with hidden_size={hrm_hidden_size}, puzzle_emb_dim={puzzle_emb_dim}")
        
        # Create residual blocks (all maintain same width for simplicity)
        for i in range(num_blocks):
            self.blocks.append(ResidualBlock(width, convs_per_block))
        
        # Add final conv layers to reach exact target depth if needed
        remaining_depth = depth - (num_blocks * convs_per_block)
        if remaining_depth > 0:
            print(f"[GridEncoder] Adding {remaining_depth} additional conv layers")
            self.final_convs = nn.ModuleList()
            for i in range(remaining_depth):
                self.final_convs.append(nn.Conv2d(width, width, 3, padding=1))
                self.final_convs.append(nn.ReLU(inplace=True))
        else:
            self.final_convs = None


class GridEncoder(HRMAwareGridEncoder):
    """
    Original GridEncoder interface maintained for compatibility.
    Enhanced with optional HRM integration capabilities.
    """
    def __init__(self, width=256, depth=10, hrm_integration=False, hrm_hidden_size=512, puzzle_emb_dim=128):
        super().__init__(width, depth, hrm_integration, hrm_hidden_size, puzzle_emb_dim)
        
    def forward(self, x: torch.Tensor, hrm_context: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode grid with optional HRM integration.
        
        Args:
            x: [B, H, W] integer grid
            hrm_context: Optional dict containing HRM states and embeddings:
                - 'z_H': [B, hrm_hidden_size] HRM H-level state
                - 'z_L': [B, hrm_hidden_size] HRM L-level state  
                - 'puzzle_emb': [B, puzzle_emb_dim] puzzle embedding
                
        Returns:
            features: [B, width, H, W] encoded features
            global_feat: [B, width] global pooled features
        """
        # Handle Long tensor input - convert to embedding
        if x.dtype in [torch.long, torch.int32, torch.int64]:
            # Ensure values are in valid range [0, 9]
            x = x.clamp(0, 9)
            # Get embeddings [B, H, W, width]
            feat = self.embed(x)
            # Permute to [B, width, H, W] for conv layers
            feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            # Already float, expand to width channels
            # Handle extra dimensions by squeezing to [B, H, W] first
            while x.dim() > 3:
                x = x.squeeze(1)  # Remove extra dims until we get [B, H, W]
            
            if x.dim() == 3:
                x = x.unsqueeze(1)  # Add channel dim → [B, 1, H, W]
            
            # Now safe to expand to [B, width, H, W]
            feat = x.expand(-1, self.width, -1, -1).float()
        
        # HRM integration (if enabled and context provided)
        if self.hrm_integration and hrm_context is not None:
            B, C, H, W = feat.shape
            
            # Extract HRM states
            z_H = hrm_context.get('z_H')
            z_L = hrm_context.get('z_L') 
            puzzle_emb = hrm_context.get('puzzle_emb')
            
            # Combine HRM reasoning states
            hrm_combined = None
            if z_H is not None and z_L is not None:
                # Combine H and L level states with learned weighting
                hrm_combined = (z_H + z_L) / 2  # Simple average, could be learned
            elif z_H is not None:
                hrm_combined = z_H
            elif z_L is not None:
                hrm_combined = z_L
                
            # Project HRM state to grid feature space
            if hrm_combined is not None:
                hrm_proj = self.hrm_proj(hrm_combined)  # [B, width]
                hrm_spatial = hrm_proj.view(B, -1, 1, 1).expand(B, -1, H, W)  # Broadcast spatially
                
                # Fuse with grid features
                fused = torch.cat([feat, hrm_spatial], dim=1)  # [B, width*2, H, W]
                feat = self.hrm_fusion(fused)  # [B, width, H, W]
            
            # Integrate puzzle embedding if available
            if puzzle_emb is not None and self.puzzle_emb_proj is not None:
                puzzle_proj = self.puzzle_emb_proj(puzzle_emb)  # [B, width]
                puzzle_spatial = puzzle_proj.view(B, -1, 1, 1).expand(B, -1, H, W)
                # Additive integration (could be more sophisticated)
                feat = feat + 0.1 * puzzle_spatial  # Scale factor to prevent overwhelming
        
        # Apply residual blocks
        for block in self.blocks:
            feat = block(feat)
        
        # Apply any remaining conv layers
        if self.final_convs is not None:
            for layer in self.final_convs:
                feat = layer(feat)
        
        # Global pooling
        glob = feat.mean(dim=(2, 3))
        
        return feat, glob