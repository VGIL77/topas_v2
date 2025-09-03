"""
Value Network - Predicts solvability and expected EBR steps for TOPAS ARC solver
Used for smart beam pruning and candidate ranking in distilled search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import context feature extractor from policy network
from models.policy_nets import ContextFeatureExtractor

# Import unified DSL registry
from models.dsl_registry import NUM_DSL_OPS

@dataclass
class ValuePrediction:
    """Prediction from the value network"""
    solvability: torch.Tensor      # [batch] - probability task is solvable
    expected_ebr_steps: torch.Tensor  # [batch] - expected EBR refinement steps
    difficulty: torch.Tensor       # [batch] - estimated task difficulty
    confidence: torch.Tensor       # [batch] - prediction confidence

class ProgramStateEncoder(nn.Module):
    """Encode partial program state for value estimation"""
    
    def __init__(self, hidden_dim: int = 128, max_depth: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        
        # Operation embedding
        self.op_embedding = nn.Embedding(NUM_DSL_OPS, hidden_dim // 2)  # DSL operations
        
        # Depth embedding
        self.depth_embedding = nn.Embedding(max_depth + 1, hidden_dim // 4)
        
        # Program structure encoder (LSTM for sequence)
        self.program_lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Operation to index mapping
        self.op_to_idx = {
            'rotate90': 0, 'rotate180': 1, 'rotate270': 2, 'flip_h': 3, 'flip_v': 4,
            'color_map': 5, 'crop_bbox': 6, 'flood_fill': 7, 'outline': 8, 'symmetry': 9,
            'translate': 10, 'scale': 11, 'tile': 12, 'paste': 13, 'tile_pattern': 14,
            'crop_nonzero': 15, 'extract_color': 16, 'resize_nn': 17, 'center_pad_to': 18,
            'identity': 19, 'count_objects': 20, 'count_colors': 21, 'arithmetic_op': 22,
            'find_pattern': 23, 'extract_pattern': 24, 'match_template': 25, 'apply_rule': 26,
            'conditional_map': 27, 'grid_union': 28, 'grid_intersection': 29, 'grid_xor': 30,
            'grid_difference': 31, 'flood_select': 32, 'select_by_property': 33, 'boundary_extract': 34,
            'for_each_object': 35, 'for_each_object_translate': 36, 'for_each_object_recolor': 37,
            'for_each_object_rotate': 38, 'for_each_object_scale': 39, 'for_each_object_flip': 40
        }
        
    def forward(self, program_ops: List[str], program_depth: int) -> torch.Tensor:
        """Encode partial program state"""
        if not program_ops:
            # Empty program
            return torch.zeros(1, self.hidden_dim, device=next(self.parameters()).device)
        
        # Convert operations to indices
        op_indices = []
        for op in program_ops:
            op_indices.append(self.op_to_idx.get(op, 0))  # Default to 'rotate90' if unknown
        
        # Create tensor
        device = next(self.parameters()).device
        op_tensor = torch.tensor(op_indices, device=device).unsqueeze(0)  # [1, seq_len]
        
        # Embed operations
        op_embeds = self.op_embedding(op_tensor)  # [1, seq_len, hidden_dim//2]
        
        # Encode with LSTM
        lstm_out, (hidden, _) = self.program_lstm(op_embeds)  # [1, seq_len, hidden_dim//2]
        
        # Use final hidden state
        program_feat = hidden[-1]  # [1, hidden_dim//2]
        
        # Add depth embedding
        depth_tensor = torch.tensor([min(program_depth, self.max_depth)], device=device)
        depth_feat = self.depth_embedding(depth_tensor)  # [1, hidden_dim//4]
        
        # Combine features
        combined_feat = torch.cat([
            program_feat.squeeze(0), 
            depth_feat.squeeze(0),
            torch.zeros(self.hidden_dim // 4, device=device)  # Padding to hidden_dim
        ], dim=0)
        
        return combined_feat.unsqueeze(0)  # [1, hidden_dim]

class ValueNet(nn.Module):
    """
    Predicts solvability and expected EBR steps for task-program pairs
    
    Used to:
    1. Prune unpromising beam search candidates early
    2. Rank candidates from OpPolicyNet by expected success
    3. Estimate computation budget needed for EBR refinement
    """
    
    def __init__(self, context_dim: int = 1024, program_dim: int = 128, 
                 hidden_dim: int = 512, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.context_dim = context_dim
        self.program_dim = program_dim
        self.hidden_dim = hidden_dim
        
        # Context feature extractor (reuse from OpPolicyNet)
        self.context_extractor = ContextFeatureExtractor()
        
        # Program state encoder
        self.program_encoder = ProgramStateEncoder(hidden_dim=program_dim)
        
        # Input projection
        self.input_proj = nn.Linear(
            self.context_extractor.total_dim + program_dim, 
            hidden_dim
        )
        
        # Multi-layer value network
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
        self.value_net = nn.Sequential(*layers)
        
        # Output heads
        self.solvability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Probability of solvability
        )
        
        self.ebr_steps_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 7),  # 1-7 EBR steps
            nn.Softmax(dim=-1)
        )
        
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Difficulty score [0, 1]
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Confidence in predictions
        )
        
    def forward(self, grid: torch.Tensor, rel_features: torch.Tensor,
                size_oracle: torch.Tensor, theme_priors: torch.Tensor,
                program_ops: List[str] = None, program_depth: int = 0) -> ValuePrediction:
        """
        Predict value metrics for current state and partial program
        
        Args:
            grid: Current grid state [B, H, W]
            rel_features: Relational features [B, rel_dim]
            size_oracle: Size oracle predictions [B, 4]
            theme_priors: Theme/transformation priors [B, 10]
            program_ops: List of operations in partial program
            program_depth: Current depth in search tree
            
        Returns:
            ValuePrediction with solvability, EBR steps, difficulty, confidence
        """
        B = grid.shape[0]
        
        # Extract context features
        context_feat = self.context_extractor(
            grid, rel_features, size_oracle, theme_priors, program_ops
        )  # [B, context_total_dim]
        
        # Encode program state
        if program_ops:
            prog_feat = self.program_encoder(program_ops, program_depth)  # [1, program_dim]
            prog_feat = prog_feat.expand(B, -1)  # Expand to batch size
        else:
            prog_feat = torch.zeros(B, self.program_dim, device=grid.device)
        
        # Combine features
        combined_feat = torch.cat([context_feat, prog_feat], dim=1)
        hidden = self.input_proj(combined_feat)  # [B, hidden_dim]
        
        # Apply value network
        value_feat = self.value_net(hidden)  # [B, hidden_dim]
        
        # Predict outputs
        solvability = self.solvability_head(value_feat).squeeze(-1)  # [B]
        ebr_steps_dist = self.ebr_steps_head(value_feat)  # [B, 7]
        difficulty = self.difficulty_head(value_feat).squeeze(-1)  # [B]
        confidence = self.confidence_head(value_feat).squeeze(-1)  # [B]
        
        # Convert EBR steps distribution to expected value
        steps_values = torch.arange(1, 8, device=grid.device).float()  # [1, 2, ..., 7]
        expected_ebr_steps = (ebr_steps_dist * steps_values.unsqueeze(0)).sum(dim=-1)  # [B]
        
        return ValuePrediction(
            solvability=solvability,
            expected_ebr_steps=expected_ebr_steps,
            difficulty=difficulty,
            confidence=confidence
        )
    
    def train_from_search_results(self, search_results: List[Tuple], 
                                 learning_rate: float = 1e-3,
                                 weight_decay: float = 1e-5) -> Dict[str, float]:
        """
        Train value network from beam search results
        
        Args:
            search_results: List of (demos, grid, program_ops, depth, solved, ebr_steps) tuples
            learning_rate: Learning rate for training
            weight_decay: Weight decay for regularization
            
        Returns:
            Dictionary of training metrics
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        losses = {'total': 0.0, 'solvability': 0.0, 'ebr_steps': 0.0, 'difficulty': 0.0}
        
        self.train()
        
        for demos, grid, program_ops, depth, solved, actual_ebr_steps in search_results:
            if demos and len(demos) > 0:
                optimizer.zero_grad()
                
                # Convert demo to grid
                demo = demos[0]
                if isinstance(demo, dict):
                    input_grid = demo.get('input')
                elif isinstance(demo, tuple):
                    input_grid = demo[0]
                else:
                    continue
                
                if isinstance(input_grid, np.ndarray):
                    input_grid = torch.from_numpy(input_grid)
                
                if input_grid.dim() == 2:
                    input_grid = input_grid.unsqueeze(0)  # Add batch dim
                
                # Create dummy features
                B, H, W = input_grid.shape
                rel_features = torch.randn(B, 64, device=input_grid.device)
                size_oracle = torch.tensor([[H, W, H, W]], device=input_grid.device).float()
                theme_priors = torch.randn(B, 10, device=input_grid.device)
                
                # Get prediction
                pred = self.forward(input_grid, rel_features, size_oracle, theme_priors, 
                                  program_ops, depth)
                
                total_loss = torch.tensor(0.0, device=input_grid.device)
                
                # Solvability loss
                solvability_target = torch.tensor([1.0 if solved else 0.0], device=input_grid.device)
                solvability_loss = F.binary_cross_entropy(pred.solvability, solvability_target)
                total_loss += solvability_loss
                losses['solvability'] += solvability_loss.item()
                
                # EBR steps loss (only if solved)
                if solved and actual_ebr_steps > 0:
                    ebr_target = torch.tensor([actual_ebr_steps], device=input_grid.device).float()
                    ebr_loss = F.mse_loss(pred.expected_ebr_steps, ebr_target)
                    total_loss += ebr_loss
                    losses['ebr_steps'] += ebr_loss.item()
                
                # Difficulty loss (heuristic based on depth and success)
                difficulty_target = torch.tensor([min(depth / 12.0, 1.0)], device=input_grid.device)
                if not solved:
                    difficulty_target += 0.3  # Increase difficulty if unsolved
                difficulty_target = torch.clamp(difficulty_target, 0.0, 1.0)
                
                difficulty_loss = F.mse_loss(pred.difficulty, difficulty_target)
                total_loss += difficulty_loss * 0.5  # Lower weight
                losses['difficulty'] += difficulty_loss.item()
                
                # Backpropagate
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                losses['total'] += total_loss.item()
        
        # Average losses
        num_results = len(search_results)
        if num_results > 0:
            for key in losses:
                losses[key] /= num_results
        
        return losses
    
    def should_prune(self, grid: torch.Tensor, rel_features: torch.Tensor,
                    size_oracle: torch.Tensor, theme_priors: torch.Tensor,
                    program_ops: List[str], program_depth: int,
                    solvability_threshold: float = 0.1,
                    max_ebr_steps: int = 6) -> bool:
        """
        Determine if a candidate should be pruned from beam search
        
        Args:
            grid: Current grid state
            rel_features: Relational features
            size_oracle: Size oracle predictions
            theme_priors: Theme priors
            program_ops: Partial program operations
            program_depth: Current search depth
            solvability_threshold: Minimum solvability to keep candidate
            max_ebr_steps: Maximum acceptable EBR steps
            
        Returns:
            True if candidate should be pruned
        """
        self.eval()
        
        with torch.no_grad():
            pred = self.forward(grid, rel_features, size_oracle, theme_priors, 
                              program_ops, program_depth)
            
            # Prune if solvability is too low
            if pred.solvability.item() < solvability_threshold:
                return True
            
            # Prune if expected EBR steps are too high
            if pred.expected_ebr_steps.item() > max_ebr_steps:
                return True
            
            # Prune if confidence is very low and solvability is marginal
            if pred.confidence.item() < 0.3 and pred.solvability.item() < 0.3:
                return True
        
        return False
    
    def rank_candidates(self, candidates: List[Tuple]) -> List[Tuple]:
        """
        Rank candidates by expected value
        
        Args:
            candidates: List of (grid, rel_feat, size_oracle, theme_priors, program_ops, depth) tuples
            
        Returns:
            Sorted list of candidates (best first)
        """
        self.eval()
        
        candidate_scores = []
        
        with torch.no_grad():
            for i, (grid, rel_feat, size_oracle, theme_priors, program_ops, depth) in enumerate(candidates):
                pred = self.forward(grid, rel_feat, size_oracle, theme_priors, program_ops, depth)
                
                # Composite score: solvability - difficulty - expected_ebr_steps/10
                score = (pred.solvability.item() - 
                        pred.difficulty.item() * 0.3 - 
                        pred.expected_ebr_steps.item() / 10.0)
                
                # Bonus for high confidence
                score += pred.confidence.item() * 0.1
                
                candidate_scores.append((score, i, candidates[i]))
        
        # Sort by score (descending)
        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [candidate for _, _, candidate in candidate_scores]

# Export all public classes
__all__ = [
    "ValuePrediction",
    "ProgramStateEncoder",
    "ValueNet"
]