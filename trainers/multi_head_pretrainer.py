"""
Multi-Head Pretrainer - World Grammar Pretraining
5-head supervised pretraining system for TOPAS ARC solver
Learns factorized representations of ARC concept algebra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import TOPAS model
from models.topas_arc_60M import TopasARC60M, ModelConfig

@dataclass 
class PretrainConfig:
    """Configuration for multi-head pretraining"""
    # Loss weights for each head
    lambda_grid: float = 1.0          # Final grid prediction
    lambda_program: float = 0.8       # DSL program tokens  
    lambda_size: float = 0.6          # Size classification
    lambda_symmetry: float = 0.4      # Symmetry classification
    lambda_histogram: float = 0.3     # Color histogram changes
    
    # Head architectures
    program_vocab_size: int = 128     # DSL operation vocabulary
    max_program_length: int = 8       # Maximum program sequence length
    size_classes: int = 16            # Number of size transformation classes
    symmetry_classes: int = 8         # Number of symmetry types
    histogram_dim: int = 10           # Color histogram dimension (0-9)
    
    # Training
    use_teacher_forcing: bool = True  # Teacher forcing for program head
    program_dropout: float = 0.1      # Dropout in program head
    head_dropout: float = 0.2         # General head dropout
    
    # Validation
    validate_sacred_signature: bool = True  # Enforce sacred signature compliance

class MultiHeadPretrainer(nn.Module):
    """
    Multi-head supervised pretraining for world grammar learning
    
    Five specialized heads:
    (a) final_grid_head - Predict output grid directly  
    (b) program_tokens_head - Teacher-forced DSL sequence prediction
    (c) size_class_head - Classify size transformations (tile/scale/pad/crop)
    (d) symmetry_class_head - Classify symmetry operations (rotation/reflection/translation)
    (e) color_histogram_head - Predict color distribution changes
    """
    
    def __init__(self, base_model: TopasARC60M, config: PretrainConfig):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        
        # Get model dimensions from TOPAS
        self.width = base_model.config.width  # Encoder width
        self.slot_dim = base_model.config.slot_dim  # Slot dimension
        
        # Control dimension = encoder width + pooled slots
        self.ctrl_dim = self.width + base_model.slots.out_dim
        
        print(f"[MultiHead] Control dimension: {self.ctrl_dim} (width={self.width} + slots={base_model.slots.out_dim})")
        
        # === HEAD (A): FINAL GRID HEAD ===
        # Uses existing painter from base model - no additional parameters needed
        self.final_grid_head = base_model.painter
        
        # === HEAD (B): PROGRAM TOKENS HEAD ===
        # Predict sequence of DSL operation tokens with teacher forcing
        self.program_embed = nn.Embedding(config.program_vocab_size, 128)
        self.program_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=512,
                dropout=config.program_dropout,
                batch_first=True
            ),
            num_layers=3
        )
        self.program_head = nn.Linear(self.ctrl_dim, config.program_vocab_size * config.max_program_length)
        
        # === HEAD (C): SIZE CLASS HEAD ===
        # Classify type of size transformation
        self.size_class_head = nn.Sequential(
            nn.Linear(self.ctrl_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(config.head_dropout),
            nn.Linear(128, config.size_classes)
        )
        
        # === HEAD (D): SYMMETRY CLASS HEAD ===
        # Classify type of symmetry operation
        self.symmetry_class_head = nn.Sequential(
            nn.Linear(self.ctrl_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(128, config.symmetry_classes)
        )
        
        # === HEAD (E): COLOR HISTOGRAM HEAD ===
        # Predict change in color histogram distribution
        self.color_histogram_head = nn.Sequential(
            nn.Linear(self.ctrl_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(128, config.histogram_dim * 2)  # Before and after histograms
        )
        
        # Operation vocabulary for program head
        self.op_vocab = self._build_operation_vocabulary()
        self.vocab_size = len(self.op_vocab)
        
        print(f"[MultiHead] Initialized with {self.vocab_size} operation vocabulary")
        print(f"[MultiHead] Heads: grid({self.final_grid_head.__class__.__name__}), "
              f"program({config.max_program_length}), size({config.size_classes}), "
              f"symmetry({config.symmetry_classes}), histogram({config.histogram_dim})")
    
    def _build_operation_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary mapping from DSL operations to token IDs"""
        # Core DSL operations from DSLHead
        operations = [
            '<PAD>', '<START>', '<END>', '<UNK>',  # Special tokens
            
            # Geometric transformations
            'rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v', 
            'translate', 'scale', 'resize_nn', 'center_pad_to',
            
            # Color operations  
            'color_map', 'extract_color', 'mask_color', 'flood_fill',
            
            # Spatial operations
            'crop_bbox', 'crop_nonzero', 'paste', 'tile', 'tile_pattern',
            
            # Pattern operations
            'outline', 'symmetry', 'boundary_extract',
            
            # Counting and arithmetic
            'count_objects', 'count_colors', 'arithmetic_op',
            
            # Pattern matching
            'find_pattern', 'extract_pattern', 'match_template',
            
            # Conditional logic
            'apply_rule', 'conditional_map',
            
            # Grid algebra
            'grid_union', 'grid_intersection', 'grid_xor', 'grid_difference',
            
            # Advanced selection
            'flood_select', 'select_by_property',
            
            # Per-object operations
            'for_each_object', 'for_each_object_translate', 'for_each_object_recolor',
            'for_each_object_rotate', 'for_each_object_scale', 'for_each_object_flip',
            
            # Composite operations
            'repeat_n', 'if_else', 'while_condition',
            
            # Identity and utility
            'identity', 'no_op'
        ]
        
        # Build vocabulary dictionary
        vocab = {op: idx for idx, op in enumerate(operations)}
        
        # Ensure vocabulary fits in config
        if len(vocab) > self.config.program_vocab_size:
            print(f"[WARN] Operation vocabulary ({len(vocab)}) exceeds config size ({self.config.program_vocab_size})")
            # Truncate to fit
            vocab = {op: idx for op, idx in vocab.items() if idx < self.config.program_vocab_size}
        
        return vocab
    
    def encode_program(self, operations: List[str], params: List[Dict] = None) -> torch.Tensor:
        """
        Encode DSL program as token sequence for program head training
        
        Args:
            operations: List of operation names
            params: List of parameter dictionaries (optional)
            
        Returns:
            Token sequence tensor [max_program_length]
        """
        tokens = [self.op_vocab['<START>']]  # Start token
        
        for op in operations[:self.config.max_program_length - 2]:  # Leave room for START/END
            if op in self.op_vocab:
                tokens.append(self.op_vocab[op])
            else:
                tokens.append(self.op_vocab['<UNK>'])  # Unknown operation
        
        tokens.append(self.op_vocab['<END>'])  # End token
        
        # Pad to max length
        while len(tokens) < self.config.max_program_length:
            tokens.append(self.op_vocab['<PAD>'])
        
        return torch.tensor(tokens[:self.config.max_program_length])
    
    def classify_size_transformation(self, input_size: Tuple[int, int], output_size: Tuple[int, int]) -> int:
        """
        Classify the type of size transformation between input and output
        
        Returns class ID for size transformation type
        """
        h_in, w_in = input_size
        h_out, w_out = output_size
        
        # Calculate size ratios
        h_ratio = h_out / max(h_in, 1)
        w_ratio = w_out / max(w_in, 1)
        
        # Classify transformation type
        if h_ratio == 1.0 and w_ratio == 1.0:
            return 0  # No size change
        elif h_ratio > 1.0 and w_ratio > 1.0:
            if abs(h_ratio - w_ratio) < 0.1:
                return 1  # Uniform scaling up
            else:
                return 2  # Non-uniform scaling up
        elif h_ratio < 1.0 and w_ratio < 1.0:
            if abs(h_ratio - w_ratio) < 0.1:
                return 3  # Uniform scaling down  
            else:
                return 4  # Non-uniform scaling down
        elif h_ratio > 1.0 or w_ratio > 1.0:
            return 5  # Padding/extension
        else:
            return 6  # Cropping/reduction
    
    def classify_symmetry_operation(self, operations: List[str]) -> int:
        """
        Classify the primary symmetry operation in the program
        
        Returns class ID for symmetry type
        """
        # Priority order for classification
        symmetry_ops = {
            'rotate90': 1,
            'rotate180': 2, 
            'rotate270': 3,
            'flip_h': 4,
            'flip_v': 5,
            'translate': 6,
            'identity': 7
        }
        
        # Find highest priority symmetry operation
        for op in operations:
            if op in symmetry_ops:
                return symmetry_ops[op]
        
        return 0  # No symmetry operation
    
    def compute_color_histogram(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized color histogram for grid
        
        Args:
            grid: Input grid tensor [H, W]
            
        Returns:
            Histogram tensor [10] for colors 0-9
        """
        if grid.numel() == 0:
            return torch.zeros(10, device=grid.device)
        
        # Clamp to valid color range
        grid_clamped = torch.clamp(grid.long(), 0, 9)
        
        # Compute histogram
        hist = torch.zeros(10, device=grid.device, dtype=torch.float32)
        unique, counts = torch.unique(grid_clamped, return_counts=True)
        hist[unique] = counts.float()
        
        # Normalize by total pixels
        hist = hist / max(grid.numel(), 1)
        
        return hist
    
    def forward(self, demos: List[Dict], test: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all heads for pretraining
        
        Args:
            demos: List of demonstration input/output pairs
            test: Test input (for consistency with base model)
            
        Returns:
            Dictionary of predictions from each head
        """
        # Get neural features from base model encoder
        test_grid = test['input']
        if test_grid.dim() == 2:
            test_grid = test_grid.unsqueeze(0)  # Add batch dim
        
        # Normalize for encoder
        enc_input = test_grid.float() / 9.0  # Scale to [0, 1]
        
        # Extract features using base model components
        feat, glob = self.base_model.encoder(enc_input)
        # slots returns 3 values: slot_vecs, attention, extras
        slot_vecs, _, _ = self.base_model.slots(feat)
        slots_rel = self.base_model.reln(slot_vecs)
        pooled = slots_rel.mean(dim=1)
        
        # Control vector (same as base model)
        brain = torch.cat([glob, pooled], dim=-1)  # [B, ctrl_dim]
        
        # === HEAD PREDICTIONS ===
        predictions = {}
        
        # (A) Final Grid Head - Use painter from base model
        grid_pred, _, _ = self.final_grid_head(feat)
        predictions['grid'] = grid_pred
        
        # (B) Program Tokens Head - Predict operation sequence
        program_logits = self.program_head(brain)  # [B, vocab_size * max_length]
        program_logits = program_logits.view(-1, self.config.max_program_length, self.config.program_vocab_size)
        predictions['program_tokens'] = program_logits
        
        # (C) Size Class Head - Classify size transformation
        size_logits = self.size_class_head(brain)  # [B, size_classes]
        predictions['size_class'] = size_logits
        
        # (D) Symmetry Class Head - Classify symmetry operations
        symmetry_logits = self.symmetry_class_head(brain)  # [B, symmetry_classes]
        predictions['symmetry_class'] = symmetry_logits
        
        # (E) Color Histogram Head - Predict histogram changes
        histogram_pred = self.color_histogram_head(brain)  # [B, histogram_dim * 2]
        histogram_pred = histogram_pred.view(-1, 2, self.config.histogram_dim)  # [B, 2, 10]
        predictions['color_histogram'] = histogram_pred
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-head loss for pretraining
        
        Args:
            predictions: Dictionary of predictions from each head
            targets: Dictionary of target values for each head
            
        Returns:
            total_loss: Combined weighted loss
            loss_components: Individual loss for each head
        """
        device = predictions['grid'].device
        loss_components = {}
        
        # (A) Grid Head Loss - Pixel-wise cross-entropy
        if 'grid' in targets:
            grid_pred = predictions['grid']  # [B, H, W]
            grid_target = targets['grid']    # [B, H, W]
            
            # Convert to logits format for cross-entropy
            B, H, W = grid_pred.shape
            grid_pred_flat = grid_pred.view(B, H * W)  # [B, H*W]
            grid_target_flat = grid_target.view(B, H * W)  # [B, H*W]
            
            # Create one-hot logits (assuming grid_pred contains class indices)
            grid_logits = F.one_hot(torch.clamp(grid_pred_flat.long(), 0, 9), num_classes=10).float()
            grid_logits = grid_logits.view(B, H * W, 10)  # [B, H*W, 10]
            
            grid_loss = F.cross_entropy(
                grid_logits.transpose(1, 2),  # [B, 10, H*W]
                grid_target_flat.long().clamp(0, 9)  # [B, H*W]
            )
            loss_components['grid'] = grid_loss
        else:
            loss_components['grid'] = torch.tensor(0.0, device=device)
        
        # (B) Program Tokens Loss - Sequence cross-entropy with teacher forcing
        if 'program_tokens' in targets:
            program_pred = predictions['program_tokens']  # [B, max_length, vocab_size]
            program_target = targets['program_tokens']    # [B, max_length]
            
            # Flatten for cross-entropy
            B, L, V = program_pred.shape
            program_loss = F.cross_entropy(
                program_pred.view(B * L, V),
                program_target.view(B * L).long().clamp(0, V - 1)
            )
            loss_components['program'] = program_loss
        else:
            loss_components['program'] = torch.tensor(0.0, device=device)
        
        # (C) Size Class Loss - Classification cross-entropy
        if 'size_class' in targets:
            size_pred = predictions['size_class']    # [B, size_classes]
            size_target = targets['size_class']      # [B]
            
            size_loss = F.cross_entropy(
                size_pred, 
                size_target.long().clamp(0, self.config.size_classes - 1)
            )
            loss_components['size'] = size_loss
        else:
            loss_components['size'] = torch.tensor(0.0, device=device)
        
        # (D) Symmetry Class Loss - Classification cross-entropy
        if 'symmetry_class' in targets:
            symmetry_pred = predictions['symmetry_class']  # [B, symmetry_classes]
            symmetry_target = targets['symmetry_class']    # [B]
            
            symmetry_loss = F.cross_entropy(
                symmetry_pred,
                symmetry_target.long().clamp(0, self.config.symmetry_classes - 1)
            )
            loss_components['symmetry'] = symmetry_loss
        else:
            loss_components['symmetry'] = torch.tensor(0.0, device=device)
        
        # (E) Color Histogram Loss - L1 regression
        if 'color_histogram' in targets:
            hist_pred = predictions['color_histogram']  # [B, 2, 10]
            hist_target = targets['color_histogram']    # [B, 2, 10]
            
            histogram_loss = F.l1_loss(hist_pred, hist_target)
            loss_components['histogram'] = histogram_loss
        else:
            loss_components['histogram'] = torch.tensor(0.0, device=device)
        
        # Combined weighted loss
        total_loss = (
            self.config.lambda_grid * loss_components['grid'] +
            self.config.lambda_program * loss_components['program'] +
            self.config.lambda_size * loss_components['size'] +
            self.config.lambda_symmetry * loss_components['symmetry'] +
            self.config.lambda_histogram * loss_components['histogram']
        )
        
        return total_loss, loss_components
    
    def prepare_targets(self, synthetic_task) -> Dict[str, torch.Tensor]:
        """
        Prepare target values for all heads from synthetic task
        
        Args:
            synthetic_task: SyntheticTask object with ground truth
            
        Returns:
            Dictionary of target tensors for each head
        """
        targets = {}
        
        # Grid target - output grid
        targets['grid'] = synthetic_task.test_output.unsqueeze(0)  # Add batch dim
        
        # Program tokens target - encode DSL operations
        program_tokens = self.encode_program(synthetic_task.program, synthetic_task.params)
        targets['program_tokens'] = program_tokens.unsqueeze(0)  # Add batch dim
        
        # Size class target - classify size transformation
        input_size = synthetic_task.test_input.shape
        output_size = synthetic_task.test_output.shape
        size_class = self.classify_size_transformation(input_size, output_size)
        targets['size_class'] = torch.tensor([size_class])
        
        # Symmetry class target - classify symmetry operations
        symmetry_class = self.classify_symmetry_operation(synthetic_task.program)
        targets['symmetry_class'] = torch.tensor([symmetry_class])
        
        # Color histogram target - input and output histograms
        input_hist = self.compute_color_histogram(synthetic_task.test_input)
        output_hist = self.compute_color_histogram(synthetic_task.test_output)
        histogram_target = torch.stack([input_hist, output_hist], dim=0)  # [2, 10]
        targets['color_histogram'] = histogram_target.unsqueeze(0)  # Add batch dim
        
        return targets
    
    def get_pretraining_mode(self) -> bool:
        """Check if model is in pretraining mode"""
        return hasattr(self, '_pretraining_mode') and self._pretraining_mode
    
    def set_pretraining_mode(self, enabled: bool = True):
        """Set pretraining mode flag"""
        self._pretraining_mode = enabled
        if enabled:
            print("[MultiHead] Pretraining mode enabled")
        else:
            print("[MultiHead] Pretraining mode disabled")


# Quick test if run directly
if __name__ == "__main__":
    if not TOPAS_AVAILABLE:
        print("TOPAS model not available - cannot test MultiHeadPretrainer")
        exit(1)
    
    print("Testing Multi-Head Pretrainer...")
    
    # Create base model
    config = ModelConfig(width=320, depth=8, slots=40)  # Smaller for testing
    base_model = TopasARC60M(config)
    
    # Create pretrainer
    pretrain_config = PretrainConfig()
    pretrainer = MultiHeadPretrainer(base_model, pretrain_config)
    pretrainer.set_pretraining_mode(True)
    
    # Test forward pass
    test_input = torch.randint(0, 10, (8, 8))
    demos = [{'input': test_input, 'output': torch.randint(0, 10, (8, 8))}]
    test = {'input': test_input}
    
    # Forward pass
    with torch.no_grad():
        predictions = pretrainer(demos, test)
    
    print(f"\nPredictions:")
    for head_name, pred in predictions.items():
        print(f"  {head_name}: {pred.shape}")
    
    # Test target preparation (requires synthetic task)
    print(f"\nOperation vocabulary size: {len(pretrainer.op_vocab)}")
    print(f"Sample operations: {list(pretrainer.op_vocab.keys())[:10]}")
    
    print("\n✅ Multi-Head Pretrainer test completed!")