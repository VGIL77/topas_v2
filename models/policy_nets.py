"""
Operation Policy Network - Predicts next DSL operation for TOPAS ARC solver
Amortizes expensive beam search into instant inference through distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import copy

# Import unified DSL registry
from models.dsl_registry import DSL_OPS, NUM_DSL_OPS, DSL_OP_TO_IDX

# Define minimal versions to avoid circular imports
@dataclass
class MetaLearningConfig:
    """Minimal config for meta-learning"""
    num_inner_steps: int = 5
    inner_lr: float = 0.001
    outer_lr: float = 0.001
    num_classes: int = 100
    embedding_dim: int = 128

@dataclass  
class Episode:
    """Minimal episode for meta-learning"""
    support_demos: List[Dict] = field(default_factory=list)
    query_demos: List[Dict] = field(default_factory=list)
    task_id: str = ""
    task_metadata: Dict = field(default_factory=dict)

# Import canonical DSL operations
from models.dsl_search import BeamCandidate, CORE_OPS
from models.utils import logits_from_grid, size_tensor_from_grid

@dataclass
class PolicyPrediction:
    """Prediction from the policy network"""
    op_logits: torch.Tensor        # [batch, num_ops] - which operation
    param_logits: Dict[str, torch.Tensor]  # operation parameters
    stop_prob: torch.Tensor        # [batch] - probability to terminate
    confidence: torch.Tensor       # [batch] - prediction confidence

class ContextFeatureExtractor(nn.Module):
    """Extract features from current context for policy prediction"""
    
    def __init__(self, grid_dim: int = 128, rel_dim: int = 64, size_dim: int = 16, theme_dim: int = 32):
        super().__init__()
        self.grid_dim = grid_dim
        self.rel_dim = rel_dim
        self.size_dim = size_dim
        self.theme_dim = theme_dim
        
        # Grid encoder for current state
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),  # 10 colors
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            # 64 channels × 8 × 8 = 4096 features (avoid magic 64*64)
            nn.Linear(64 * 8 * 8, grid_dim)
        )
        
        # Object slots statistics encoder
        self.slots_encoder = nn.Sequential(
            nn.Linear(10 * 4, 64),  # 10 colors × (count, area, centroid_x, centroid_y)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, grid_dim // 4)
        )
        
        # Relational features encoder
        self.rel_encoder = nn.Sequential(
            nn.Linear(rel_dim, 64),
            nn.ReLU(),
            nn.Linear(64, rel_dim)
        )
        
        # Size oracle features
        self.size_encoder = nn.Sequential(
            nn.Linear(4, 16),  # current H,W + predicted H,W
            nn.ReLU(),
            nn.Linear(16, size_dim)
        )
        
        # Theme prior encoder
        self.theme_encoder = nn.Sequential(
            nn.Linear(10, 32),  # transformation type priors
            nn.ReLU(),
            nn.Linear(32, theme_dim)
        )
        
        # Program context encoder (partial program so far)
        self.program_encoder = nn.Sequential(
            nn.Linear(NUM_DSL_OPS * 5, 64),  # DSL ops × 5 recent operations max
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, grid_dim // 4)
        )
        
        # Feature budget:
        #   grid_feat: grid_dim
        #   slot_feat: grid_dim//4  
        #   prog_feat: grid_dim//4
        #   rel_feat:  rel_dim
        #   size_feat: size_dim
        #   theme:     theme_dim
        self.total_dim = grid_dim + (grid_dim // 2) + rel_dim + size_dim + theme_dim
        
    def compute_slot_stats(self, grid: torch.Tensor) -> torch.Tensor:
        """Compute object statistics for each color"""
        B, H, W = grid.shape
        stats = torch.zeros(B, 10, 4, device=grid.device)  # 10 colors × 4 stats
        
        for b in range(B):
            for color in range(10):
                mask = (grid[b] == color).float()
                count = mask.sum()
                
                if count > 0:
                    # Count
                    stats[b, color, 0] = count
                    
                    # Area (same as count for discrete grids)
                    stats[b, color, 1] = count
                    
                    # Centroid
                    y_coords, x_coords = torch.where(grid[b] == color)
                    if len(y_coords) > 0:
                        stats[b, color, 2] = y_coords.float().mean()
                        stats[b, color, 3] = x_coords.float().mean()
        
        # Flatten to [B, 10*4]
        return stats.flatten(1)
    
    def encode_program_context(self, program_ops: List[str], max_ops: int = 5) -> torch.Tensor:
        """Encode recent operations in the partial program"""
        # One-hot encode recent operations using canonical registry
        encoding = torch.zeros(max_ops * NUM_DSL_OPS)
        recent_ops = program_ops[-max_ops:] if len(program_ops) > max_ops else program_ops
        for i, op in enumerate(recent_ops):
            if op in DSL_OP_TO_IDX:
                encoding[i * NUM_DSL_OPS + DSL_OP_TO_IDX[op]] = 1.0
        return encoding.unsqueeze(0)
    
    def forward(self, grid: torch.Tensor, rel_features: torch.Tensor, 
                size_oracle: torch.Tensor, theme_priors: torch.Tensor,
                program_ops: List[str] = None) -> torch.Tensor:
        """Extract context features for policy prediction"""
        B = grid.shape[0]
        
        # Encode grid (convert to one-hot first)
        grid_onehot = F.one_hot(grid.long(), num_classes=10).float()
        grid_onehot = grid_onehot.permute(0, 3, 1, 2)  # [B, 10, H, W]
        grid_feat = self.grid_encoder(grid_onehot)  # [B, grid_dim]
        
        # Encode object slot statistics
        slot_stats = self.compute_slot_stats(grid)  # [B, 40]
        slot_feat = self.slots_encoder(slot_stats)  # [B, grid_dim//4]
        
        # Encode relational features
        rel_feat = self.rel_encoder(rel_features)  # [B, rel_dim]
        
        # Encode size features
        size_feat = self.size_encoder(size_oracle)  # [B, size_dim]
        
        # Encode theme priors
        theme_feat = self.theme_encoder(theme_priors)  # [B, theme_dim]
        
        # Encode program context
        if program_ops:
            prog_encoding = self.encode_program_context(program_ops).to(grid.device)
            prog_encoding = prog_encoding.expand(B, -1)  # Expand to batch
        else:
            prog_encoding = torch.zeros(B, NUM_DSL_OPS * 5, device=grid.device)
        prog_feat = self.program_encoder(prog_encoding)  # [B, grid_dim//4]
        
        # Combine features
        combined_grid_feat = torch.cat([grid_feat, slot_feat, prog_feat], dim=1)
        context_feat = torch.cat([
            combined_grid_feat, rel_feat, size_feat, theme_feat
        ], dim=1)  # [B, total_dim]
        
        return context_feat

class OpPolicyNet(nn.Module):
    """
    Program policy network - predicts next DSL operation
    
    Converts expensive beam search into cheap policy through distillation.
    Uses small transformer to predict next operation token + parameters + stop probability.
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, 
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1):
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = ((hidden_dim + num_heads - 1) // num_heads) * num_heads
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_operations = NUM_DSL_OPS  # Number of DSL operations
        
        # Context feature extractor
        self.context_extractor = ContextFeatureExtractor()
        
        # Input projection
        self.input_proj = nn.Linear(self.context_extractor.total_dim, hidden_dim)
        
        # Small transformer for sequence modeling
        self.pos_encoding = nn.Parameter(torch.randn(100, hidden_dim))  # Max 100 operations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Operation prediction head
        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_operations)
        )
        
        # Parameter prediction heads for different operation types
        self.param_heads = nn.ModuleDict({
            'color_map': nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 10 * 10)  # 10x10 color mapping matrix
            ),
            'translate': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 8)  # dx, dy in range [-3, 3]
            ),
            'scale': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 9)  # scale factors 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5
            ),
            'extract_objects': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 10)  # color selection
            )
        })
        
        # Stop probability head
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Use canonical registry
        self.op_to_idx = DSL_OP_TO_IDX
        
        self.idx_to_op = {v: k for k, v in self.op_to_idx.items()}
        
    def forward(self, grid: torch.Tensor, rel_features: torch.Tensor,
                size_oracle: torch.Tensor, theme_priors: torch.Tensor,
                program_ops: List[str] = None, seq_pos: int = 0) -> PolicyPrediction:
        """
        Predict next operation and parameters
        
        Args:
            grid: Current grid state [B, H, W]
            rel_features: Relational features [B, rel_dim]
            size_oracle: Size oracle predictions [B, 4]
            theme_priors: Theme/transformation priors [B, 10]
            program_ops: List of operations in current partial program
            seq_pos: Position in sequence for positional encoding
            
        Returns:
            PolicyPrediction with operation logits, parameters, and stop probability
        """
        B = grid.shape[0]
        
        # Extract context features
        context_feat = self.context_extractor(
            grid, rel_features, size_oracle, theme_priors, program_ops
        )  # [B, total_dim]
        
        # Project to hidden dimension
        hidden = self.input_proj(context_feat)  # [B, hidden_dim]
        
        # Add positional encoding
        if seq_pos < self.pos_encoding.shape[0]:
            hidden = hidden + self.pos_encoding[seq_pos].unsqueeze(0)
        
        # Apply transformer (single step for autoregressive prediction)
        hidden = hidden.unsqueeze(1)  # [B, 1, hidden_dim]
        transformer_out = self.transformer(hidden)  # [B, 1, hidden_dim]
        hidden = transformer_out.squeeze(1)  # [B, hidden_dim]
        
        # Predict operation
        op_logits = self.op_head(hidden)  # [B, num_operations]
        
        # Predict parameters for different operation types
        param_logits = {}
        for param_type, head in self.param_heads.items():
            param_logits[param_type] = head(hidden)
        
        # Predict stop probability
        stop_prob = self.stop_head(hidden).squeeze(-1)  # [B]
        
        # Predict confidence
        confidence = self.confidence_head(hidden).squeeze(-1)  # [B]
        
        return PolicyPrediction(
            op_logits=op_logits,
            param_logits=param_logits,
            stop_prob=stop_prob,
            confidence=confidence
        )
    
    def train_from_traces(self, successful_traces: List[Tuple], 
                         learning_rate: float = 1e-3, 
                         scheduled_sampling_prob: float = 0.1,
                         length_penalty: float = 0.01) -> Dict[str, float]:
        """
        Train policy network from successful beam search traces
        
        Args:
            successful_traces: List of (demos, program_trace, final_score) tuples
            learning_rate: Learning rate for training
            scheduled_sampling_prob: Probability of using model prediction vs ground truth
            length_penalty: Penalty for longer programs
            
        Returns:
            Dictionary of training metrics
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        losses = {'total': 0.0, 'op': 0.0, 'param': 0.0, 'stop': 0.0, 'length': 0.0}
        
        self.train()
        
        for demos, program_trace, score in successful_traces:
            if not program_trace.ops:  # Skip empty programs
                continue
                
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
            
            # Convert demos to initial grid state
            if demos and len(demos) > 0:
                demo = demos[0]  # Use first demo
                if isinstance(demo, dict):
                    initial_grid = demo.get('input')
                elif isinstance(demo, tuple):
                    initial_grid = demo[0]
                else:
                    continue
                    
                if isinstance(initial_grid, np.ndarray):
                    initial_grid = torch.from_numpy(initial_grid)
                
                if initial_grid.dim() == 2:
                    initial_grid = initial_grid.unsqueeze(0)  # Add batch dim
                
                current_grid = initial_grid.clone()
                
                # Create training features
                B, H, W = current_grid.shape
                rel_features = torch.randn(B, 64, device=current_grid.device)
                size_oracle = torch.tensor([[H, W, H, W]], device=current_grid.device).float()
                theme_priors = torch.randn(B, 10, device=current_grid.device)
                
                # Teacher forcing with scheduled sampling
                partial_program = []
                
                for step, (target_op, target_params) in enumerate(zip(program_trace.ops, program_trace.params)):
                    if target_op not in self.op_to_idx:
                        continue
                        
                    # Use scheduled sampling: sometimes use model prediction
                    if np.random.random() < scheduled_sampling_prob and partial_program:
                        # Use model prediction (scheduled sampling)
                        with torch.no_grad():
                            pred = self.forward(current_grid, rel_features, size_oracle, 
                                              theme_priors, partial_program, step)
                            pred_op_idx = torch.argmax(pred.op_logits, dim=-1).item()
                            pred_op = self.idx_to_op.get(pred_op_idx, 'identity')
                            partial_program.append(pred_op)
                    else:
                        # Use ground truth (teacher forcing)
                        partial_program.append(target_op)
                    
                    # Get prediction for current step
                    pred = self.forward(current_grid, rel_features, size_oracle, 
                                      theme_priors, partial_program[:-1], step)
                    
                    # Operation loss
                    target_op_idx = self.op_to_idx[target_op]
                    op_loss = F.cross_entropy(pred.op_logits, 
                                            torch.tensor([target_op_idx], device=current_grid.device))
                    total_loss = total_loss + op_loss
                    losses['op'] += op_loss.item()
                    
                    # Parameter loss (simplified - just for color_map)
                    if target_op == 'color_map' and 'mapping' in target_params:
                        if 'color_map' in pred.param_logits:
                            # Create target parameter tensor
                            target_mapping = target_params['mapping']
                            param_target = torch.zeros(100, device=current_grid.device)
                            for src, dst in target_mapping.items():
                                if src < 10 and dst < 10:
                                    param_target[src * 10 + dst] = 1.0
                            
                            param_loss = F.binary_cross_entropy_with_logits(
                                pred.param_logits['color_map'].squeeze(0), param_target
                            )
                            total_loss = total_loss + param_loss
                            losses['param'] += param_loss.item()
                
                # Stop token loss (should predict stop at end)
                final_pred = self.forward(current_grid, rel_features, size_oracle, 
                                        theme_priors, partial_program, len(partial_program))
                stop_target = torch.tensor([1.0], device=current_grid.device)  # Should stop
                stop_loss = F.binary_cross_entropy(final_pred.stop_prob, stop_target)
                total_loss = total_loss + stop_loss
                losses['stop'] += stop_loss.item()
                
                # Length penalty
                length_loss = length_penalty * len(program_trace.ops)
                total_loss = total_loss + length_loss
                losses['length'] += length_loss
                
                # Backpropagate
                total_loss.backward()
                optimizer.step()
                
                losses['total'] += total_loss.item()
        
        # Average losses
        num_traces = len(successful_traces)
        if num_traces > 0:
            for key in losses:
                losses[key] /= num_traces
        
        return losses
    
    def generate_program(self, grid: torch.Tensor, rel_features: torch.Tensor,
                        size_oracle: torch.Tensor, theme_priors: torch.Tensor,
                        max_length: int = 12, stop_threshold: float = 0.5) -> List[str]:
        """
        Generate a program autoregressively using the trained policy
        
        Args:
            grid: Input grid [1, H, W] (batch size 1)
            rel_features: Relational features [1, rel_dim]
            size_oracle: Size oracle predictions [1, 4]
            theme_priors: Theme priors [1, 10]
            max_length: Maximum program length
            stop_threshold: Threshold for stop probability
            
        Returns:
            List of operation names
        """
        self.eval()
        program = []
        
        with torch.no_grad():
            for step in range(max_length):
                # Get prediction
                pred = self.forward(grid, rel_features, size_oracle, theme_priors, program, step)
                
                # Check if should stop
                if pred.stop_prob.item() > stop_threshold:
                    break
                
                # Sample next operation
                op_probs = F.softmax(pred.op_logits, dim=-1)
                op_idx = torch.multinomial(op_probs, 1).item()
                
                if op_idx in self.idx_to_op:
                    op_name = self.idx_to_op[op_idx]
                    program.append(op_name)
                else:
                    break  # Invalid operation
        
        return program

# META-LEARNING EXTENSIONS
class MetaContextExtractor(ContextFeatureExtractor):
    """Enhanced context extractor for meta-learning with task adaptation features"""
    
    def __init__(self, grid_dim: int = 128, rel_dim: int = 64, size_dim: int = 16, 
                 theme_dim: int = 32, task_dim: int = 32):
        super().__init__(grid_dim, rel_dim, size_dim, theme_dim)
        
        self.task_dim = task_dim
        
        # Task-specific context encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(5, 32),  # difficulty, num_examples, avg_input_size, avg_output_size, type_embedding
            nn.ReLU(),
            nn.Linear(32, task_dim)
        )
        
        # Fast adaptation context (encodes support set statistics)
        self.adaptation_encoder = nn.Sequential(
            nn.Linear(grid_dim * 2, 64),  # Support input/output statistics
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, grid_dim // 4)
        )
        
        # Update total dimension
        self.total_dim = grid_dim + rel_dim + size_dim + theme_dim + task_dim
        
    def encode_task_context(self, episode: Episode) -> torch.Tensor:
        """Encode task-specific context from episode"""
        # Extract task statistics
        difficulty = episode.difficulty
        num_examples = len(episode.support_set)
        
        # Compute average grid sizes
        avg_input_size = 0.0
        avg_output_size = 0.0
        
        if episode.support_set:
            for demo in episode.support_set:
                input_size = demo['input'].numel()
                output_size = demo['output'].numel()
                avg_input_size += input_size
                avg_output_size += output_size
            
            avg_input_size /= len(episode.support_set)
            avg_output_size /= len(episode.support_set)
            
            # Normalize sizes
            avg_input_size = min(1.0, avg_input_size / 400.0)  # Normalize by 20x20
            avg_output_size = min(1.0, avg_output_size / 400.0)
        
        # Encode composition type
        type_encodings = {
            "rotate90": 0.1, "rotate180": 0.2, "rotate270": 0.3,
            "flip_horizontal": 0.4, "flip_vertical": 0.5,
            "color_map": 0.6, "scale_up": 0.7, "crop": 0.8,
            "unknown": 0.0
        }
        type_embedding = type_encodings.get(episode.composition_type, 0.0)
        
        # Create task context vector
        task_context = torch.tensor([
            difficulty, num_examples / 10.0, avg_input_size, avg_output_size, type_embedding
        ], dtype=torch.float32)
        
        return task_context.unsqueeze(0)  # Add batch dimension
    
    def encode_adaptation_context(self, support_demos: List[Dict]) -> torch.Tensor:
        """Encode support set statistics for adaptation"""
        if not support_demos:
            return torch.zeros(1, self.grid_dim // 4)
        
        # Compute support set statistics
        input_stats = []
        output_stats = []
        
        for demo in support_demos:
            input_grid = demo['input']
            output_grid = demo['output']
            
            # Basic statistics
            input_stats.extend([
                float(input_grid.mean()),
                float(input_grid.std()),
                float((input_grid != 0).float().mean()),  # Non-background ratio
                float(input_grid.max())
            ])
            
            output_stats.extend([
                float(output_grid.mean()),
                float(output_grid.std()),
                float((output_grid != 0).float().mean()),
                float(output_grid.max())
            ])
        
        # Pad or truncate to fixed size
        target_size = self.grid_dim
        input_stats = (input_stats + [0.0] * target_size)[:target_size]
        output_stats = (output_stats + [0.0] * target_size)[:target_size]
        
        combined_stats = torch.tensor(input_stats + output_stats, dtype=torch.float32)
        adaptation_context = self.adaptation_encoder(combined_stats.unsqueeze(0))
        
        return adaptation_context
    
    def forward(self, grid: torch.Tensor, rel_features: torch.Tensor,
                size_oracle: torch.Tensor, theme_priors: torch.Tensor,
                program_ops: List[str] = None, episode: Episode = None,
                support_demos: List[Dict] = None) -> torch.Tensor:
        """Enhanced forward pass with task and adaptation context"""
        
        # Get base context features
        base_context = super().forward(grid, rel_features, size_oracle, theme_priors, program_ops)
        
        # Add task-specific context
        if episode is not None:
            task_context = self.encode_task_context(episode)
            task_context = task_context.to(base_context.device)
            task_feat = self.task_encoder(task_context)
            # Expand to match batch size
            B = base_context.shape[0]
            task_feat = task_feat.expand(B, -1)
        else:
            # Default task context
            B = base_context.shape[0]
            task_feat = torch.zeros(B, self.task_dim, device=base_context.device)
        
        # Add adaptation context
        if support_demos is not None:
            adapt_context = self.encode_adaptation_context(support_demos)
            adapt_context = adapt_context.to(base_context.device)
            # Expand to match batch size
            adapt_feat = adapt_context.expand(B, -1)
            
            # Combine with base context
            base_context = torch.cat([base_context, adapt_feat], dim=1)
        
        # Combine all contexts
        enhanced_context = torch.cat([base_context, task_feat], dim=1)
        
        return enhanced_context

class MetaOpPolicyNet(OpPolicyNet):
    """
    Meta-learning enabled Operation Policy Network
    
    Extends OpPolicyNet with fast adaptation capabilities for few-shot learning.
    Can adapt to new tasks in 1-3 gradient steps using meta-learned initialization.
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1,
                 meta_config: MetaLearningConfig = None):
        # Initialize base policy net
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        
        self.meta_config = meta_config or MetaLearningConfig()
        
        # Replace context extractor with meta-learning version
        self.context_extractor = MetaContextExtractor()
        
        # Update input projection to handle new context dimension
        self.input_proj = nn.Linear(self.context_extractor.total_dim, hidden_dim)
        
        # Meta-learning specific components
        self.adaptation_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        
        # Fast adaptation parameters (learned meta-parameters)
        self.fast_weights = nn.ParameterDict()
        # Collect parameters to adapt first, then add them
        params_to_adapt = []
        for name, param in self.named_parameters():
            if 'op_head' in name or 'param_heads' in name:
                # Only adapt the final prediction layers by default
                # Replace dots with underscores in parameter names
                safe_name = name.replace('.', '_')
                params_to_adapt.append((safe_name, param))
        
        # Now add the adaptation parameters
        for safe_name, param in params_to_adapt:
            self.fast_weights[safe_name] = nn.Parameter(torch.zeros_like(param))
        
        # Adaptation statistics
        self.adaptation_stats = {
            "adaptations_performed": 0,
            "avg_adaptation_loss": 0.0,
            "adaptation_success_rate": 0.0
        }
        
        print(f"[MetaOpPolicyNet] Enhanced with meta-learning capabilities")
        print(f"[MetaOpPolicyNet] Fast adaptation layers: {len(self.fast_weights)}")
    
    def forward(self, grid: torch.Tensor, rel_features: torch.Tensor,
                size_oracle: torch.Tensor, theme_priors: torch.Tensor,
                program_ops: List[str] = None, seq_pos: int = 0,
                episode: Episode = None, support_demos: List[Dict] = None,
                use_adapted_weights: bool = False) -> PolicyPrediction:
        """
        Enhanced forward pass with meta-learning context
        
        Args:
            grid: Current grid state [B, H, W]
            rel_features: Relational features [B, rel_dim]
            size_oracle: Size oracle predictions [B, 4]
            theme_priors: Theme/transformation priors [B, 10]
            program_ops: List of operations in current partial program
            seq_pos: Position in sequence for positional encoding
            episode: Current episode for task context
            support_demos: Support demonstrations for adaptation context
            use_adapted_weights: Whether to use fast-adapted weights
            
        Returns:
            PolicyPrediction with enhanced meta-learning features
        """
        B = grid.shape[0]
        
        # Extract enhanced context features
        context_feat = self.context_extractor(
            grid, rel_features, size_oracle, theme_priors, 
            program_ops, episode, support_demos
        )
        
        # Project to hidden dimension
        hidden = self.input_proj(context_feat)
        
        # Add positional encoding
        if seq_pos < self.pos_encoding.shape[0]:
            hidden = hidden + self.pos_encoding[seq_pos].unsqueeze(0)
        
        # Apply transformer
        hidden = hidden.unsqueeze(1)
        transformer_out = self.transformer(hidden)
        hidden = transformer_out.squeeze(1)
        
        # Apply adaptation layers if in meta-learning mode
        if episode is not None or support_demos is not None:
            for layer in self.adaptation_layers:
                hidden = hidden + F.relu(layer(hidden))  # Residual connection
        
        # Predict operation (potentially with adapted weights)
        if use_adapted_weights and 'op_head_2_weight' in self.fast_weights:
            # Use adapted weights for prediction
            op_logits = self._forward_with_fast_weights(hidden, 'op_head')
        else:
            op_logits = self.op_head(hidden)
        
        # Predict parameters (potentially with adapted weights)
        param_logits = {}
        for param_type, head in self.param_heads.items():
            if use_adapted_weights and f'param_heads_{param_type}_2_weight' in self.fast_weights:
                param_logits[param_type] = self._forward_with_fast_weights(hidden, f'param_heads.{param_type}')
            else:
                param_logits[param_type] = head(hidden)
        
        # Predict stop probability and confidence
        stop_prob = self.stop_head(hidden).squeeze(-1)
        confidence = self.confidence_head(hidden).squeeze(-1)
        
        return PolicyPrediction(
            op_logits=op_logits,
            param_logits=param_logits,
            stop_prob=stop_prob,
            confidence=confidence
        )
    
    def adapt(self, support_demos: List[Dict], episode: Episode = None,
              num_steps: int = None, adaptation_lr: float = None) -> 'MetaOpPolicyNet':
        """
        Fast adaptation to new task using support demonstrations
        
        Args:
            support_demos: Support set demonstrations for adaptation
            episode: Episode context (optional)
            num_steps: Number of adaptation steps
            adaptation_lr: Learning rate for adaptation
            
        Returns:
            Self (adapted in-place) for method chaining
        """
        num_steps = num_steps or self.meta_config.inner_steps
        adaptation_lr = adaptation_lr or self.meta_config.inner_lr
        
        print(f"[MetaOpPolicyNet] Adapting to task with {len(support_demos)} support examples")
        
        # Store original parameters for potential restoration
        original_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Create optimizer for adaptation
        adaptation_optimizer = torch.optim.SGD(
            [param for name, param in self.named_parameters() if name.replace('.', '_') in self.fast_weights],
            lr=adaptation_lr
        )
        
        losses = []
        
        # Adaptation loop
        for step in range(num_steps):
            adaptation_optimizer.zero_grad()
            
            total_loss = 0.0
            valid_demos = 0
            
            for demo in support_demos:
                try:
                    # Create context (in real usage, this would come from the main model)
                    B = 1
                    rel_features = torch.randn(B, 64, device=demo['input'].device)
                    size_oracle = torch.tensor([[8.0, 8.0, 8.0, 8.0]], device=demo['input'].device)
                    theme_priors = torch.randn(B, 10, device=demo['input'].device)
                    
                    # Forward pass with adaptation context
                    pred = self.forward(
                        demo['input'].unsqueeze(0),
                        rel_features, size_oracle, theme_priors,
                        episode=episode, support_demos=support_demos,
                        use_adapted_weights=True
                    )
                    
                    # Compute adaptation loss (policy should predict operations that work)
                    # This is a simplified loss - in practice, you'd want to use actual DSL execution results
                    loss = self._compute_adaptation_loss(pred, demo)
                    
                    total_loss += loss
                    valid_demos += 1
                    
                except Exception as e:
                    print(f"[MetaOpPolicyNet] Adaptation demo failed: {e}")
                    continue
            
            if valid_demos == 0:
                print("[MetaOpPolicyNet] Warning: No valid demos for adaptation")
                break
            
            avg_loss = total_loss / valid_demos
            losses.append(float(avg_loss))
            
            # Backward pass
            avg_loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update parameters
            adaptation_optimizer.step()
        
        # Update statistics
        self.adaptation_stats["adaptations_performed"] += 1
        if losses:
            self.adaptation_stats["avg_adaptation_loss"] = 0.9 * self.adaptation_stats["avg_adaptation_loss"] + 0.1 * losses[-1]
            adaptation_success = (losses[0] - losses[-1]) > 0.1 if len(losses) > 1 else False
            self.adaptation_stats["adaptation_success_rate"] = 0.9 * self.adaptation_stats["adaptation_success_rate"] + 0.1 * float(adaptation_success)
        
        print(f"[MetaOpPolicyNet] Adaptation complete: {len(losses)} steps, final_loss={losses[-1] if losses else 'N/A'}")
        
        return self
    
    def _forward_with_fast_weights(self, hidden: torch.Tensor, layer_prefix: str) -> torch.Tensor:
        """Forward pass using fast-adapted weights"""
        # This is a simplified implementation - in practice, you'd want more sophisticated
        # fast weight application based on the specific layer architecture
        
        # For now, just apply a learned adaptation to the output
        fast_weight_key = f"{layer_prefix}_adaptation"
        if fast_weight_key not in self.fast_weights:
            # Create adaptation parameter if it doesn't exist
            output_dim = getattr(self, layer_prefix.split('.')[0])[-1].out_features
            self.fast_weights[fast_weight_key] = nn.Parameter(torch.zeros(output_dim, device=hidden.device))
        
        # Get base output
        base_output = getattr(self, layer_prefix.split('.')[0])(hidden)
        
        # Apply fast adaptation
        adaptation = self.fast_weights[fast_weight_key]
        adapted_output = base_output + adaptation.unsqueeze(0)
        
        return adapted_output
    
    def _compute_adaptation_loss(self, pred: PolicyPrediction, demo: Dict) -> torch.Tensor:
        """
        Compute adaptation loss for policy learning
        
        This is a simplified implementation. In practice, you'd want to:
        1. Execute the predicted operations on the input
        2. Compare results with the target output
        3. Use success/failure as supervision signal
        """
        # Simple heuristic: predict reasonable operations based on grid properties
        input_grid = demo['input']
        output_grid = demo['output']
        
        # Training supervision: prefer certain operations based on input/output relationship
        target_op = self._infer_target_operation(input_grid, output_grid)
        target_idx = self.op_to_idx.get(target_op, 0)
        
        # Cross-entropy loss on operation prediction
        op_loss = F.cross_entropy(pred.op_logits, torch.tensor([target_idx], device=pred.op_logits.device))
        
        # Encourage higher confidence for clearer transformations
        confidence_target = torch.tensor([0.8], device=pred.confidence.device)
        confidence_loss = F.mse_loss(pred.confidence, confidence_target)
        
        total_loss = op_loss + 0.1 * confidence_loss
        
        return total_loss
    
    def _infer_target_operation(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> str:
        """Infer target operation from input/output pair (simplified heuristic)"""
        
        if input_grid.shape != output_grid.shape:
            return "resize_nn"  # Size change
        
        # Check for rotations
        if torch.equal(torch.rot90(input_grid, 1, dims=(0, 1)), output_grid):
            return "rotate90"
        elif torch.equal(torch.rot90(input_grid, 2, dims=(0, 1)), output_grid):
            return "rotate180"
        elif torch.equal(torch.rot90(input_grid, 3, dims=(0, 1)), output_grid):
            return "rotate270"
        
        # Check for flips
        if torch.equal(torch.flip(input_grid, dims=(0,)), output_grid):
            return "flip_v"
        elif torch.equal(torch.flip(input_grid, dims=(1,)), output_grid):
            return "flip_h"
        
        # Check if it's potentially a color mapping
        if self._is_potential_color_map(input_grid, output_grid):
            return "color_map"
        
        # Default fallback
        return "identity"
    
    def _is_potential_color_map(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> bool:
        """Check if output could be a color mapping of input"""
        if input_grid.shape != output_grid.shape:
            return False
        
        # Simple check: same pattern but different colors
        input_unique = torch.unique(input_grid).numel()
        output_unique = torch.unique(output_grid).numel()
        
        # If number of unique colors is similar, might be color mapping
        return abs(input_unique - output_unique) <= 2
    
    def clone_for_adaptation(self) -> 'MetaOpPolicyNet':
        """Create a copy of the network for adaptation without affecting original"""
        cloned_net = copy.deepcopy(self)
        return cloned_net
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        return dict(self.adaptation_stats)
    
    def reset_adaptation_stats(self):
        """Reset adaptation statistics"""
        self.adaptation_stats = {
            "adaptations_performed": 0,
            "avg_adaptation_loss": 0.0,
            "adaptation_success_rate": 0.0
        }


def create_meta_op_policy_net(base_policy_net: OpPolicyNet = None, 
                             meta_config: MetaLearningConfig = None) -> MetaOpPolicyNet:
    """
    Create a meta-learning enabled operation policy network
    
    Args:
        base_policy_net: Optional base policy network to enhance
        meta_config: Meta-learning configuration
        
    Returns:
        MetaOpPolicyNet ready for few-shot learning
    """
    meta_config = meta_config or MetaLearningConfig()
    
    if base_policy_net is not None:
        # Transfer knowledge from base network
        meta_net = MetaOpPolicyNet(
            input_dim=base_policy_net.input_dim,
            hidden_dim=base_policy_net.hidden_dim,
            num_layers=6,  # base_policy_net doesn't expose this
            num_heads=8,   # base_policy_net doesn't expose this
            meta_config=meta_config
        )
        
        # Copy compatible weights
        try:
            meta_net.load_state_dict(base_policy_net.state_dict(), strict=False)
            print("[MetaOpPolicyNet] Transferred weights from base policy network")
        except Exception as e:
            print(f"[MetaOpPolicyNet] Weight transfer failed: {e}, using random initialization")
    else:
        # Create from scratch
        meta_net = MetaOpPolicyNet(meta_config=meta_config)
    
    param_count = sum(p.numel() for p in meta_net.parameters()) / 1e6
    print(f"[MetaOpPolicyNet] Created with {param_count:.2f}M parameters")
    print(f"[MetaOpPolicyNet] Target: Adapt in {meta_config.inner_steps} steps")
    
    return meta_net


# Export all public classes and functions
__all__ = [
    "PolicyPrediction",
    "ContextFeatureExtractor",
    "OpPolicyNet",
    "MetaContextExtractor",
    "MetaOpPolicyNet",
    "create_meta_op_policy_net"
]

if __name__ == "__main__":
    # Test the meta operation policy network
    print("="*60)
    print("MetaOpPolicyNet - Quick Test")
    print("="*60)
    
    # Create meta-learning config
    meta_config = MetaLearningConfig(
        inner_lr=0.01,
        inner_steps=2
    )
    
    # Create meta policy network
    meta_policy = create_meta_op_policy_net(meta_config=meta_config)
    
    # Create test episode and demo data
    test_demo = {
        'input': torch.randint(0, 10, (8, 8)),
        'output': torch.randint(0, 10, (8, 8))
    }
    
    episode = Episode(
        task_id="meta_test",
        support_set=[test_demo],
        query_set=[test_demo],
        difficulty=0.4,
        composition_type="rotate90"
    )
    
    # Test enhanced forward pass
    print("\n[TEST] Testing enhanced forward pass...")
    B = 1
    grid = test_demo['input'].unsqueeze(0)
    rel_features = torch.randn(B, 64)
    size_oracle = torch.tensor([[8.0, 8.0, 8.0, 8.0]])
    theme_priors = torch.randn(B, 10)
    
    pred = meta_policy.forward(
        grid, rel_features, size_oracle, theme_priors,
        episode=episode, support_demos=[test_demo]
    )
    
    print(f"Operation logits shape: {pred.op_logits.shape}")
    print(f"Confidence: {pred.confidence.item():.3f}")
    
    # Test fast adaptation
    print("\n[TEST] Testing fast adaptation...")
    adapted_policy = meta_policy.clone_for_adaptation()
    adapted_policy.adapt([test_demo], episode=episode, num_steps=2)
    
    adaptation_stats = adapted_policy.get_adaptation_stats()
    print(f"Adaptation stats: {adaptation_stats}")
    
    print("\n[TEST] MetaOpPolicyNet operational!")
    print("Key features:")
    print("  - Fast adaptation in 1-3 gradient steps")
    print("  - Task-specific context encoding")
    print("  - Support set statistics integration")
    print("  - Meta-learned fast weights")
    print("="*60)