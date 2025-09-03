import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlock(nn.Module):
    """Basic UNet block with conv + norm + relu"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        # Use GroupNorm instead of BatchNorm to handle small spatial sizes and batch_size=1
        num_groups = min(32, out_ch)  # Ensure groups don't exceed channels
        if out_ch % num_groups != 0:
            num_groups = 1  # Fallback to LayerNorm equivalent
        self.norm = nn.GroupNorm(num_groups, out_ch)
        
    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class DownBlock(nn.Module):
    """Encoder block: conv + norm + relu + maxpool"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = UNetBlock(in_ch, out_ch)
        self.conv2 = UNetBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv1(x)
        skip = self.conv2(x)  # Skip connection output
        pooled = self.pool(skip)
        return pooled, skip


class UpBlock(nn.Module):
    """Decoder block: upsample + concat + conv + norm + relu"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv1 = UNetBlock(in_ch // 2 + skip_ch, out_ch)
        self.conv2 = UNetBlock(out_ch, out_ch)
        
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class NeuralPainter(nn.Module):
    """
    UNet-lite painter for ARC grid reconstruction and denoising.
    
    Architecture:
    - Encoder: 2-3 downsampling blocks (conv + norm + relu + pool)
    - Bottleneck: convolutional processing
    - Decoder: 2-3 upsampling blocks (upsample + conv + norm + relu)
    - Skip connections from encoder to decoder
    - Final conv to produce logits [B,C,H,W]
    
    Sacred signature: (grid, logits, size)
    """
    
    def __init__(self, width=192, num_colors=10):
        super().__init__()
        self.width = width
        self.num_colors = num_colors
        
        # Encoder (downsampling path)
        self.down1 = DownBlock(width, width // 2)      # width -> width//2
        self.down2 = DownBlock(width // 2, width // 4)  # width//2 -> width//4
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            UNetBlock(width // 4, width // 2),
            UNetBlock(width // 2, width // 2)
        )
        
        # Decoder (upsampling path)
        self.up1 = UpBlock(width // 2, width // 4, width // 4)  # bottleneck + skip2
        self.up2 = UpBlock(width // 4, width // 2, width // 2)  # up1 + skip1
        
        # Final output layer to produce logits
        self.final_conv = nn.Conv2d(width // 2, num_colors, 1)
        
        # Direct projection for small grids (bypasses encoder/decoder)
        self.small_grid_conv = nn.Conv2d(width, num_colors, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feat):
        """
        Forward pass enforcing sacred signature validation.
        
        Args:
            feat: Input feature tensor [B, C, H, W]
            
        Returns:
            grid: [B, H, W] integer grid (argmax of logits)
            logits: [B, H*W, C] flat logits for loss computation  
            size: [B, 2] output dimensions [H, W]
            
        Raises:
            RuntimeError: On any signature violation
        """
        # INPUT VALIDATION
        if not isinstance(feat, torch.Tensor):
            raise RuntimeError(f"[PAINTER] INPUT VIOLATION: feat must be torch.Tensor, got {type(feat)}")
        
        if feat.dim() != 4:
            raise RuntimeError(f"[PAINTER] INPUT VIOLATION: feat must be [B,C,H,W], got shape {feat.shape}")
        
        B, C, H, W = feat.shape
        
        if B <= 0 or C <= 0 or H <= 0 or W <= 0:
            raise RuntimeError(f"[PAINTER] INPUT VIOLATION: all dimensions must be positive, got [{B},{C},{H},{W}]")
        
        # Minimum size guard - painter needs at least 8x8 to avoid pooling issues
        if H < 8 or W < 8:
            # For very small grids, use direct conv projection
            logits_raw = self.small_grid_conv(feat)  # [B, 10, H, W]
            
            # VALIDATE SMALL GRID PATH OUTPUT
            if logits_raw.shape != (B, self.num_colors, H, W):
                raise RuntimeError(f"[PAINTER-SMALL] SIGNATURE VIOLATION: logits_raw expected ({B},{self.num_colors},{H},{W}), got {logits_raw.shape}")
            
            grid = logits_raw.argmax(dim=1)    # [B, H, W]
            logits = logits_raw.permute(0, 2, 3, 1).reshape(B, H*W, self.num_colors)  # [B, H*W, 10]
            size = torch.tensor([H, W], device=feat.device).unsqueeze(0).expand(B, -1).long()  # [B, 2]
            
            # Validate small grid signature
            assert isinstance(grid, torch.Tensor), f"[PAINTER-SMALL] grid must be tensor, got {type(grid)}"
            assert grid.shape == (B, H, W), f"[PAINTER-SMALL] grid shape violation: expected ({B},{H},{W}), got {grid.shape}"
            assert grid.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long], f"[PAINTER-SMALL] grid must be integer, got {grid.dtype}"
            assert grid.min() >= 0 and grid.max() < self.num_colors, f"[PAINTER-SMALL] grid range violation: [{grid.min()},{grid.max()}] not in [0,{self.num_colors})"
            
            assert isinstance(logits, torch.Tensor), f"[PAINTER-SMALL] logits must be tensor, got {type(logits)}"
            assert logits.shape == (B, H*W, self.num_colors), f"[PAINTER-SMALL] logits shape violation: expected ({B},{H*W},{self.num_colors}), got {logits.shape}"
            assert logits.dtype.is_floating_point, f"[PAINTER-SMALL] logits must be float, got {logits.dtype}"
            
            assert isinstance(size, torch.Tensor), f"[PAINTER-SMALL] size must be tensor, got {type(size)}"
            assert size.shape == (B, 2), f"[PAINTER-SMALL] size shape violation: expected ({B},2), got {size.shape}"
            
            return grid, logits, size
        
        # Encoder path with skip connections
        x1, skip1 = self.down1(feat)      # x1: [B, width//2, H//2, W//2]
        x2, skip2 = self.down2(x1)        # x2: [B, width//4, H//4, W//4]
        
        # VALIDATE ENCODER OUTPUTS
        if x1.shape[0] != B:
            raise RuntimeError(f"[PAINTER] ENCODER VIOLATION: x1 batch mismatch, expected {B}, got {x1.shape[0]}")
        if x2.shape[0] != B:
            raise RuntimeError(f"[PAINTER] ENCODER VIOLATION: x2 batch mismatch, expected {B}, got {x2.shape[0]}")
        
        # Bottleneck
        bottleneck = self.bottleneck(x2)  # [B, width//2, H//4, W//4]
        
        # VALIDATE BOTTLENECK OUTPUT
        if bottleneck.shape[0] != B:
            raise RuntimeError(f"[PAINTER] BOTTLENECK VIOLATION: batch mismatch, expected {B}, got {bottleneck.shape[0]}")
        
        # Decoder path with skip connections
        up1 = self.up1(bottleneck, skip2) # [B, width//4, H//2, W//2]
        up2 = self.up2(up1, skip1)        # [B, width//2, H, W]
        
        # VALIDATE DECODER OUTPUTS
        if up1.shape[0] != B:
            raise RuntimeError(f"[PAINTER] DECODER VIOLATION: up1 batch mismatch, expected {B}, got {up1.shape[0]}")
        if up2.shape[0] != B:
            raise RuntimeError(f"[PAINTER] DECODER VIOLATION: up2 batch mismatch, expected {B}, got {up2.shape[0]}")
        if up2.shape[-2:] != (H, W):
            raise RuntimeError(f"[PAINTER] DECODER VIOLATION: up2 spatial mismatch, expected ({H},{W}), got {up2.shape[-2:]}")
        
        # Final logits [B, num_colors, H, W]
        logits_raw = self.final_conv(up2)  # [B, 10, H, W]
        
        # VALIDATE RAW LOGITS
        if logits_raw.shape != (B, self.num_colors, H, W):
            raise RuntimeError(f"[PAINTER] LOGITS VIOLATION: expected ({B},{self.num_colors},{H},{W}), got {logits_raw.shape}")
        
        # Sacred signature outputs
        grid = logits_raw.argmax(dim=1)    # [B, H, W]
        logits = logits_raw.permute(0, 2, 3, 1).reshape(B, H*W, self.num_colors)  # [B, H*W, 10]
        size = torch.tensor([H, W], device=feat.device).unsqueeze(0).expand(B, -1).long()  # [B, 2]
        
        # Final signature validation
        assert isinstance(grid, torch.Tensor), f"[PAINTER] grid must be tensor, got {type(grid)}"
        assert grid.shape == (B, H, W), f"[PAINTER] grid shape violation: expected ({B},{H},{W}), got {grid.shape}"
        assert grid.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long], f"[PAINTER] grid must be integer, got {grid.dtype}"
        assert grid.min() >= 0 and grid.max() < self.num_colors, f"[PAINTER] grid range violation: [{grid.min()},{grid.max()}] not in [0,{self.num_colors})"
        
        assert isinstance(logits, torch.Tensor), f"[PAINTER] logits must be tensor, got {type(logits)}"
        assert logits.shape == (B, H*W, self.num_colors), f"[PAINTER] logits shape violation: expected ({B},{H*W},{self.num_colors}), got {logits.shape}"
        assert logits.dtype.is_floating_point, f"[PAINTER] logits must be float, got {logits.dtype}"
        
        assert isinstance(size, torch.Tensor), f"[PAINTER] size must be tensor, got {type(size)}"
        assert size.shape == (B, 2), f"[PAINTER] size shape violation: expected ({B},2), got {size.shape}"
        
        # VERIFY SIZE TENSOR VALUES
        expected_size = torch.tensor([H, W], device=feat.device).unsqueeze(0).expand(B, -1).long()
        if not torch.equal(size, expected_size):
            raise RuntimeError(f"[PAINTER] size tensor value violation: expected {expected_size[0].tolist()}, got {size[0].tolist()}")
        
        return grid, logits, size
