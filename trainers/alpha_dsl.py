"""
Alpha-DSL: Monte-Carlo Tree Search over DSL programs (like AlphaGo for ARC)

This implements MCTS for program search, discovering deeper solutions (length 6-10)
and distilling them back into the policy network for improved performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from copy import deepcopy
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

# Import existing TOPAS components
from models.dsl_search import DSLProgram, apply_program, CORE_OPS, DSLProgram
from models.dsl_search import BeamCandidate, score_candidate, verify_candidate
from models.policy_nets import OpPolicyNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProgramState:
    """Represents a partial DSL program state"""
    ops: List[str]
    params: List[Dict[str, Any]]
    depth: int
    
    def __post_init__(self):
        if len(self.ops) != len(self.params):
            raise ValueError("ops and params must have same length")
    
    def add_op(self, op_name: str, op_params: Dict[str, Any] = None) -> 'ProgramState':
        """Add an operation to create new state"""
        if op_params is None:
            op_params = {}
        return ProgramState(
            ops=self.ops + [op_name],
            params=self.params + [op_params],
            depth=self.depth + 1
        )
    
    def to_dsl_program(self) -> DSLProgram:
        """Convert to DSLProgram for execution"""
        return DSLProgram(ops=self.ops, params=self.params)
    
    def is_terminal(self, max_depth: int = 10) -> bool:
        """Check if this is a terminal state"""
        return self.depth >= max_depth

class AlphaDSLNode:
    """MCTS Node for program search (like AlphaGo tree nodes)"""
    
    def __init__(self, state: ProgramState, parent: 'AlphaDSLNode' = None, 
                 prior: float = 0.0, action: str = None):
        self.state = state  # Partial program
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children = {}  # Dict[action_str, AlphaDSLNode]
        
        # MCTS statistics
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior  # From policy network
        self.solved = False  # Whether this program solves the task
        
        # Thread safety
        self.lock = threading.Lock()
    
    @property 
    def value(self) -> float:
        """Average value"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def uct_score(self, c_puct: float = 1.0, parent_visits: int = None) -> float:
        """Upper Confidence Tree score for selection"""
        if self.visits == 0:
            return float('inf')
        
        if parent_visits is None:
            parent_visits = self.parent.visits if self.parent else 1
        
        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        
        return exploitation + exploration
    
    def select_child(self, c_puct: float = 1.0) -> 'AlphaDSLNode':
        """Select best child using UCT"""
        if not self.children:
            return self
        
        best_child = max(
            self.children.values(),
            key=lambda child: child.uct_score(c_puct, self.visits)
        )
        return best_child
    
    def expand(self, action_priors: Dict[str, float]) -> Dict[str, 'AlphaDSLNode']:
        """Expand node with children based on action priors"""
        with self.lock:
            if self.children:  # Already expanded
                return self.children
            
            for action, prior in action_priors.items():
                # Parse action into operation and parameters
                op_name, op_params = self._parse_action(action)
                child_state = self.state.add_op(op_name, op_params)
                
                child = AlphaDSLNode(
                    state=child_state,
                    parent=self,
                    prior=prior,
                    action=action
                )
                self.children[action] = child
            
            return self.children
    
    def backup(self, value: float):
        """Backup value through the tree"""
        node = self
        while node is not None:
            with node.lock:
                node.visits += 1
                node.value_sum += value
            node = node.parent
    
    def _parse_action(self, action: str) -> Tuple[str, Dict[str, Any]]:
        """Parse action string into operation name and parameters"""
        if ':' in action:
            parts = action.split(':', 1)
            op_name = parts[0]
            
            # Parse parameters (simplified - extend for more complex params)
            try:
                if parts[1].startswith('{') and parts[1].endswith('}'):
                    # Dictionary-like parameters
                    param_str = parts[1][1:-1]  # Remove braces
                    params = {}
                    if param_str:
                        for pair in param_str.split(','):
                            if '=' in pair:
                                k, v = pair.split('=', 1)
                                try:
                                    params[k.strip()] = int(v.strip())
                                except ValueError:
                                    params[k.strip()] = v.strip()
                    return op_name, params
                else:
                    # Simple parameter
                    return op_name, {'param': parts[1]}
            except:
                return op_name, {}
        else:
            return action, {}
    
    def get_action_stats(self) -> Dict[str, Tuple[int, float]]:
        """Get visit counts and values for all actions"""
        stats = {}
        total_visits = sum(child.visits for child in self.children.values())
        
        for action, child in self.children.items():
            stats[action] = (child.visits, child.value)
        
        return stats

class ValueNet(nn.Module):
    """Value network to evaluate program states"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.program_encoder = nn.Sequential(
            nn.Linear(41 * 10, hidden_dim),  # 41 ops × max 10 ops in program
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, hidden_dim // 2)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Value in [-1, 1]
        )
        
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
    
    def forward(self, program: DSLProgram, grid: torch.Tensor) -> torch.Tensor:
        """Evaluate the value of a partial program on current grid state"""
        # Encode program as one-hot
        program_encoding = torch.zeros(41 * 10)
        for i, op in enumerate(program.ops[:10]):  # Max 10 operations
            if op in self.op_to_idx:
                program_encoding[i * 41 + self.op_to_idx[op]] = 1.0
        
        program_encoding = program_encoding.unsqueeze(0)  # Add batch dim
        if grid.is_cuda:
            program_encoding = program_encoding.cuda()
        
        program_feat = self.program_encoder(program_encoding)
        
        # Encode grid
        if grid.dim() == 2:
            grid = grid.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif grid.dim() == 3:
            grid = grid.unsqueeze(0)  # Add batch dim
        
        # Convert to one-hot
        if grid.shape[1] != 10:  # Not already one-hot
            grid_onehot = F.one_hot(grid.long().squeeze(1), num_classes=10).float()
            grid_onehot = grid_onehot.permute(0, 3, 1, 2)  # [B, 10, H, W]
        else:
            grid_onehot = grid
            
        grid_feat = self.grid_encoder(grid_onehot)
        
        # Combine features
        combined = torch.cat([program_feat, grid_feat], dim=1)
        value = self.value_head(combined)
        
        return value.squeeze(-1)  # Remove last dim

class AlphaDSL:
    """MCTS for DSL program search (like AlphaGo for DSL)"""
    
    def __init__(self, policy_net: OpPolicyNet, value_net: ValueNet, dsl_head: DSLHead):
        self.policy_net = policy_net
        self.value_net = value_net 
        self.dsl_head = dsl_head
        self.c_puct = 1.0
        
        # Available operations for expansion
        self.available_ops = [
            'rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v',
            'color_map', 'crop_bbox', 'flood_fill', 'outline', 'symmetry',
            'translate', 'scale', 'tile', 'paste', 'tile_pattern',
            'crop_nonzero', 'extract_color', 'resize_nn', 'center_pad_to',
            'identity', 'count_objects', 'count_colors', 'arithmetic_op',
            'find_pattern', 'extract_pattern', 'match_template', 'apply_rule',
            'conditional_map', 'grid_union', 'grid_intersection', 'grid_xor',
            'grid_difference', 'flood_select', 'select_by_property', 'boundary_extract',
            'for_each_object', 'for_each_object_translate', 'for_each_object_recolor',
            'for_each_object_rotate', 'for_each_object_scale', 'for_each_object_flip'
        ]
    
    def search(self, task_demos: List[Tuple], num_simulations: int = 800, 
               max_depth: int = 10, verbose: bool = False) -> Optional[DSLProgram]:
        """
        Run MCTS to find best program for solving the task
        
        Args:
            task_demos: List of (input, output) demonstration pairs
            num_simulations: Number of MCTS simulations to run
            max_depth: Maximum program depth to explore
            verbose: Enable debug logging
            
        Returns:
            Best DSL program found, or None if no solution
        """
        if not task_demos:
            return None
        
        # Initialize root with empty program
        root_state = ProgramState(ops=[], params=[], depth=0)
        root = AlphaDSLNode(state=root_state)
        
        start_time = time.time()
        
        for sim in range(num_simulations):
            if verbose and sim % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"MCTS simulation {sim}/{num_simulations} ({elapsed:.2f}s)")
            
            # Selection: traverse tree using UCT
            node = self.select(root)
            
            # Expansion and Evaluation
            if not node.state.is_terminal(max_depth):
                # Expand node if not terminal
                node = self.expand(node, task_demos)
            
            # Evaluation: get value estimate
            value = self.evaluate(node, task_demos)
            
            # Backup: propagate value up tree
            node.backup(value)
            
            # Early termination if perfect solution found
            if node.solved and node.visits > 10:
                if verbose:
                    logger.info(f"Perfect solution found at simulation {sim}")
                break
        
        # Return best program from root
        return self.get_best_program(root, verbose=verbose)
    
    def select(self, root: AlphaDSLNode) -> AlphaDSLNode:
        """Select leaf node with highest UCT score"""
        node = root
        
        while node.children:
            node = node.select_child(self.c_puct)
        
        return node
    
    def expand(self, node: AlphaDSLNode, task_demos: List[Tuple]) -> AlphaDSLNode:
        """Expand node using policy network to get action priors"""
        # Get current grid state by applying partial program
        demo_input = task_demos[0][0]  # Use first demo input
        if isinstance(demo_input, np.ndarray):
            demo_input = torch.from_numpy(demo_input)
        
        current_grid = demo_input.clone()
        
        # Apply partial program to get current state
        try:
            if node.state.ops:
                program = node.state.to_dsl_program()
                current_grid = self.dsl_head.apply_program(current_grid, program)
        except Exception as e:
            logger.debug(f"Failed to apply partial program: {e}")
        
        # Get policy predictions for next actions
        action_priors = self.get_action_priors(current_grid, node.state)
        
        # Expand with top-k actions
        k = min(8, len(action_priors))  # Limit branching factor
        top_actions = dict(sorted(action_priors.items(), key=lambda x: x[1], reverse=True)[:k])
        
        children = node.expand(top_actions)
        
        # Return first child for immediate evaluation
        if children:
            return next(iter(children.values()))
        return node
    
    def get_action_priors(self, grid: torch.Tensor, state: ProgramState) -> Dict[str, float]:
        """Get action priors from policy network"""
        try:
            # Create dummy features for policy network
            B = 1
            H, W = grid.shape[-2:]
            
            if grid.dim() == 2:
                grid = grid.unsqueeze(0)  # Add batch dim
            
            rel_features = torch.randn(B, 64, device=grid.device)
            size_oracle = torch.tensor([[H, W, H, W]], device=grid.device).float()
            theme_priors = torch.randn(B, 10, device=grid.device)
            
            # Get policy prediction
            with torch.no_grad():
                self.policy_net.eval()
                pred = self.policy_net.forward(
                    grid, rel_features, size_oracle, theme_priors,
                    program_ops=state.ops, seq_pos=len(state.ops)
                )
            
            # Convert operation logits to action priors
            op_probs = F.softmax(pred.op_logits, dim=-1).cpu().numpy()[0]
            
            action_priors = {}
            for i, op_name in enumerate(self.available_ops):
                if i < len(op_probs):
                    # Simple action without parameters for now
                    action_priors[op_name] = float(op_probs[i])
            
            # Add some basic parameterized actions
            if 'color_map' in self.available_ops:
                idx = self.available_ops.index('color_map')
                if idx < len(op_probs):
                    base_prob = float(op_probs[idx])
                    # Add common color mappings
                    for mapping in ['{0=1,1=0}', '{0=2,2=0}', '{1=2,2=1}']:
                        action_priors[f'color_map:{mapping}'] = base_prob * 0.3
            
            if 'translate' in self.available_ops:
                idx = self.available_ops.index('translate')
                if idx < len(op_probs):
                    base_prob = float(op_probs[idx])
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        action_priors[f'translate:{{dx={dx},dy={dy}}}'] = base_prob * 0.25
            
            return action_priors
            
        except Exception as e:
            logger.debug(f"Failed to get action priors: {e}")
            # Fallback: uniform priors
            uniform_prob = 1.0 / len(self.available_ops)
            return {op: uniform_prob for op in self.available_ops[:10]}  # Top 10
    
    def evaluate(self, node: AlphaDSLNode, task_demos: List[Tuple]) -> float:
        """Evaluate node using value network and rollout"""
        # Quick check: does current program solve the task?
        try:
            program = node.state.to_dsl_program()
            if self.verify_solution(program, task_demos):
                node.solved = True
                return 1.0  # Perfect solution
        except Exception:
            pass
        
        # Value network evaluation
        try:
            demo_input = task_demos[0][0]
            if isinstance(demo_input, np.ndarray):
                demo_input = torch.from_numpy(demo_input)
            
            with torch.no_grad():
                self.value_net.eval()
                value = self.value_net(node.state.to_dsl_program(), demo_input)
                base_value = float(value.item())
        except Exception as e:
            logger.debug(f"Value network evaluation failed: {e}")
            base_value = 0.0
        
        # Add rollout evaluation for more accuracy
        rollout_value = self.rollout_evaluation(node, task_demos)
        
        # Combine evaluations
        final_value = 0.7 * base_value + 0.3 * rollout_value
        
        return final_value
    
    def rollout_evaluation(self, node: AlphaDSLNode, task_demos: List[Tuple], 
                          rollout_depth: int = 3) -> float:
        """Fast rollout evaluation using policy network"""
        if node.state.is_terminal() or rollout_depth == 0:
            return 0.0
        
        try:
            # Apply current program to get state
            demo_input = task_demos[0][0]
            if isinstance(demo_input, np.ndarray):
                demo_input = torch.from_numpy(demo_input)
            
            current_grid = demo_input.clone()
            if node.state.ops:
                program = node.state.to_dsl_program()
                current_grid = self.dsl_head.apply_program(current_grid, program)
            
            # Sample a few operations using policy
            rollout_state = deepcopy(node.state)
            
            for _ in range(rollout_depth):
                action_priors = self.get_action_priors(current_grid, rollout_state)
                if not action_priors:
                    break
                
                # Sample action based on priors
                actions, probs = zip(*action_priors.items())
                probs = np.array(probs)
                probs = probs / probs.sum()  # Normalize
                
                action = np.random.choice(actions, p=probs)
                op_name, op_params = node._parse_action(action)
                rollout_state = rollout_state.add_op(op_name, op_params)
                
                # Apply operation
                try:
                    current_grid = self.dsl_head.ops[op_name](current_grid, **op_params)
                except Exception:
                    break  # Invalid operation
            
            # Check if rollout program solves task
            rollout_program = rollout_state.to_dsl_program()
            if self.verify_solution(rollout_program, task_demos):
                return 0.8  # Good rollout value
            
            # Partial credit for similarity
            return self.compute_similarity(current_grid, task_demos[0][1]) * 0.5
            
        except Exception as e:
            logger.debug(f"Rollout evaluation failed: {e}")
            return 0.0
    
    def verify_solution(self, program: DSLProgram, task_demos: List[Tuple]) -> bool:
        """Check if program solves all demonstrations"""
        try:
            for demo_input, demo_output in task_demos:
                if isinstance(demo_input, np.ndarray):
                    demo_input = torch.from_numpy(demo_input)
                if isinstance(demo_output, np.ndarray):
                    demo_output = torch.from_numpy(demo_output)
                
                predicted = self.dsl_head.apply_program(demo_input, program)
                
                if predicted.shape != demo_output.shape:
                    return False
                
                if not torch.equal(predicted, demo_output):
                    return False
            
            return True
        except Exception:
            return False
    
    def compute_similarity(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Compute similarity between predicted and target grids"""
        try:
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)
            
            if predicted.shape != target.shape:
                return 0.0
            
            # Pixel-wise accuracy
            accuracy = (predicted == target).float().mean().item()
            return accuracy
        except Exception:
            return 0.0
    
    def get_best_program(self, root: AlphaDSLNode, verbose: bool = False) -> Optional[DSLProgram]:
        """Get best program from MCTS tree"""
        if not root.children:
            return None
        
        # Find child with highest visit count (most promising)
        best_child = max(root.children.values(), key=lambda child: child.visits)
        
        if verbose:
            logger.info(f"Best program visits: {best_child.visits}, value: {best_child.value:.3f}")
            logger.info(f"Program: {best_child.state.ops}")
        
        # If best child is a solution, return it
        if best_child.solved:
            return best_child.state.to_dsl_program()
        
        # Otherwise, recursively get best path
        if best_child.children:
            return self.get_best_program(best_child, verbose=verbose)
        
        return best_child.state.to_dsl_program()
    
    def get_visit_distribution(self, root: AlphaDSLNode) -> Dict[str, int]:
        """Get visit count distribution for distillation"""
        if not root.children:
            return {}
        
        total_visits = sum(child.visits for child in root.children.values())
        distribution = {}
        
        for action, child in root.children.items():
            distribution[action] = child.visits / max(total_visits, 1)
        
        return distribution
    
    def collect_training_data(self, root: AlphaDSLNode) -> List[Tuple]:
        """Collect (state, action_distribution, value) tuples for training"""
        training_data = []
        
        def traverse(node: AlphaDSLNode):
            if node.children and node.visits > 10:  # Only collect from well-visited nodes
                # Get action distribution
                total_visits = sum(child.visits for child in node.children.values())
                action_dist = {}
                
                for action, child in node.children.items():
                    action_dist[action] = child.visits / max(total_visits, 1)
                
                # Add training tuple
                training_data.append((
                    deepcopy(node.state),
                    action_dist,
                    node.value
                ))
                
                # Recurse to children
                for child in node.children.values():
                    traverse(child)
        
        traverse(root)
        return training_data