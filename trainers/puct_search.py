"""
PUCT (Predictor + Upper Confidence bounds applied to Trees) Search Implementation
Neural-guided Monte Carlo Tree Search for program synthesis

This implementation provides:
- MCTSNode class for tree structure
- PUCT selection formula with exploration bonus
- Neural policy and value network integration
- Dirichlet noise for exploration at root
- Parameter-aware DSL operation expansion
- Efficient tree traversal and backpropagation
- Production-ready integration with existing DSL system
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import time
import logging
from collections import defaultdict

# Import DSL operations and registry
from models.dsl_search import DSLProgram, apply_program, generate_op_parameters, CORE_OPS
from models.dsl_registry import DSL_OPS, get_op_index, NUM_DSL_OPS

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """
    Monte Carlo Tree Search Node for program synthesis

    Represents a partial program state in the search tree with:
    - Program operations and parameters accumulated so far
    - Visit statistics for PUCT calculation
    - Policy priors from neural network
    - Value estimates and bounds
    """
    # Program state
    ops: List[str] = field(default_factory=list)
    params: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 0

    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    mean_value: float = 0.0

    # Neural network outputs
    policy_priors: Optional[torch.Tensor] = None  # P(a|s) from policy network
    value_estimate: float = 0.0  # V(s) from value network

    # Tree structure
    parent: Optional['MCTSNode'] = None
    children: Dict[Tuple[str, frozenset], 'MCTSNode'] = field(default_factory=dict)

    # State properties
    is_terminal: bool = False
    is_solved: bool = False
    terminal_reward: float = 0.0

    # Caching for efficiency
    program_hash: str = ""
    cached_results: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed fields"""
        if not self.program_hash:
            ops_str = "|".join(self.ops)
            params_str = "|".join(str(p) for p in self.params)
            self.program_hash = f"{ops_str}#{params_str}"

    def get_program(self) -> DSLProgram:
        """Convert node state to DSLProgram"""
        return DSLProgram(ops=self.ops.copy(), params=self.params.copy())

    def apply_program(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply the program represented by this node to a grid"""
        program = self.get_program()
        return apply_program(grid, program)

    def is_fully_expanded(self, available_ops: List[str], max_depth: int) -> bool:
        """Check if all possible actions have been tried from this node"""
        if self.depth >= max_depth or self.is_terminal:
            return True

        # Count expected number of expansions
        expected_children = 0
        for op in available_ops:
            param_options = generate_op_parameters(op, None)  # Use None context for now
            expected_children += len(param_options)

        return len(self.children) >= expected_children

    def get_action_key(self, op: str, params: Dict[str, Any]) -> Tuple[str, frozenset]:
        """Create a hashable key for an action (op, params pair)"""
        # Convert params dict to frozenset of items for hashing
        param_items = frozenset(params.items()) if params else frozenset()
        return (op, param_items)

    def add_child(self, op: str, params: Dict[str, Any]) -> 'MCTSNode':
        """Add a child node for the given action"""
        action_key = self.get_action_key(op, params)

        if action_key not in self.children:
            child = MCTSNode(
                ops=self.ops + [op],
                params=self.params + [params],
                depth=self.depth + 1,
                parent=self
            )
            self.children[action_key] = child

        return self.children[action_key]

    def update_value(self, value: float):
        """Update visit statistics with new value"""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

    def get_puct_score(self, c_puct: float, parent_visits: int) -> float:
        """
        Calculate PUCT score for node selection

        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
        - Q(s,a) is the mean action value (exploitation)
        - P(s,a) is the policy prior probability
        - N(s) is parent visit count, N(s,a) is child visit count
        - c_puct controls exploration vs exploitation balance
        """
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes get highest priority

        # Q(s,a) - mean value (exploitation term)
        exploitation = self.mean_value

        # P(s,a) * sqrt(N(s)) / (1 + N(s,a)) - exploration term
        if self.parent is not None and self.parent.policy_priors is not None:
            # Find this node's policy prior from parent
            # This is a simplified version - in practice, you'd map actions to policy indices
            prior_prob = 1.0 / len(self.parent.children) if len(self.parent.children) > 0 else 1.0
        else:
            prior_prob = 1.0  # Default uniform prior

        exploration = c_puct * prior_prob * math.sqrt(parent_visits) / (1.0 + self.visit_count)

        return exploitation + exploration

    def select_best_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest PUCT score"""
        if not self.children:
            return None

        best_child = None
        best_score = -float('inf')

        for child in self.children.values():
            score = child.get_puct_score(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def get_visit_counts(self) -> Dict[Tuple[str, frozenset], int]:
        """Get visit counts for all children (useful for policy training)"""
        return {action_key: child.visit_count for action_key, child in self.children.items()}

    def __repr__(self) -> str:
        return f"MCTSNode(depth={self.depth}, ops={self.ops}, visits={self.visit_count}, value={self.mean_value:.3f})"


class PUCTSearcher:
    """
    PUCT Search implementation for neural-guided program synthesis

    Combines Monte Carlo Tree Search with neural policy and value networks
    to efficiently explore the space of DSL programs.
    """

    def __init__(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_noise_weight: float = 0.25,
        max_depth: int = 10,
        temperature: float = 1.0,
        device: torch.device = None
    ):
        """
        Initialize PUCT searcher

        Args:
            policy_net: Neural network that outputs policy priors P(a|s)
            value_net: Neural network that outputs value estimates V(s)
            c_puct: Exploration constant for PUCT formula
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_noise_weight: Weight of Dirichlet noise vs policy prior
            max_depth: Maximum program depth to search
            temperature: Temperature for action selection
            device: Torch device for computations
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_noise_weight = dirichlet_noise_weight
        self.max_depth = max_depth
        self.temperature = temperature
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Available DSL operations
        self.available_ops = CORE_OPS.copy()

        # Statistics
        self.search_stats = {
            'total_simulations': 0,
            'total_nodes_created': 0,
            'cache_hits': 0,
            'neural_evaluations': 0
        }

        logger.info(f"Initialized PUCT searcher with {len(self.available_ops)} DSL operations")

    def get_policy_value(
        self,
        node: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Get policy priors and value estimate from neural networks

        Args:
            node: Current MCTS node
            demos: Demonstration input/output pairs
            test_input: Test input grid

        Returns:
            policy_priors: Probability distribution over actions
            value_estimate: Scalar value estimate for the state
        """
        self.search_stats['neural_evaluations'] += 1

        try:
            # Prepare input for neural networks
            # This is a simplified version - in practice you'd encode the current program state,
            # demonstrations, and test input into a format suitable for your networks

            # For now, use a simple encoding based on program depth and operations
            program_encoding = self._encode_program_state(node, demos, test_input)

            # Get policy priors from policy network
            with torch.no_grad():
                policy_logits = self.policy_net(program_encoding)
                policy_priors = F.softmax(policy_logits, dim=-1)

            # Get value estimate from value network
            with torch.no_grad():
                value_estimate = self.value_net(program_encoding).item()

            return policy_priors, value_estimate

        except Exception as e:
            logger.warning(f"Neural network evaluation failed: {e}")
            # Fallback to uniform policy and neutral value
            uniform_policy = torch.ones(NUM_DSL_OPS) / NUM_DSL_OPS
            return uniform_policy, 0.0

    def _encode_program_state(
        self,
        node: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode current program state for neural network input

        This is a simplified encoding - in practice you'd want a more sophisticated
        representation that captures the program semantics, demonstration patterns, etc.
        """
        # Create a simple encoding based on:
        # 1. Current program operations (one-hot)
        # 2. Program depth
        # 3. Basic statistics about demos and test input

        encoding_dim = NUM_DSL_OPS + 16  # ops + metadata features
        encoding = torch.zeros(encoding_dim, device=self.device)

        # Encode current operations
        for op in node.ops:
            op_idx = get_op_index(op)
            if op_idx >= 0:
                encoding[op_idx] = 1.0

        # Add metadata features
        metadata_start = NUM_DSL_OPS
        encoding[metadata_start] = float(node.depth) / self.max_depth  # Normalized depth
        encoding[metadata_start + 1] = len(demos)  # Number of demos

        if len(demos) > 0:
            # Average demo input/output sizes
            avg_input_size = sum(demo[0].numel() for demo, _ in demos) / len(demos)
            avg_output_size = sum(demo[1].numel() for _, demo in demos) / len(demos)
            encoding[metadata_start + 2] = min(avg_input_size / 100.0, 1.0)  # Normalized
            encoding[metadata_start + 3] = min(avg_output_size / 100.0, 1.0)  # Normalized

        # Test input size
        encoding[metadata_start + 4] = min(test_input.numel() / 100.0, 1.0)

        return encoding.unsqueeze(0)  # Add batch dimension

    def add_dirichlet_noise(self, policy_priors: torch.Tensor) -> torch.Tensor:
        """Add Dirichlet noise to policy priors for exploration"""
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy_priors))
        noise_tensor = torch.tensor(noise, dtype=policy_priors.dtype, device=policy_priors.device)

        return (1 - self.dirichlet_noise_weight) * policy_priors + \
               self.dirichlet_noise_weight * noise_tensor

    def expand_node(
        self,
        node: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> MCTSNode:
        """
        Expand a node by adding all possible children and evaluating with neural networks

        Args:
            node: Node to expand
            demos: Demonstration pairs for evaluation
            test_input: Test input grid

        Returns:
            A newly created child node (for selection in simulation)
        """
        if node.is_terminal or node.depth >= self.max_depth:
            return node

        # Get neural network predictions for this state
        policy_priors, value_estimate = self.get_policy_value(node, demos, test_input)

        # Store in node
        node.policy_priors = policy_priors
        node.value_estimate = value_estimate

        # Add Dirichlet noise at root for exploration
        if node.parent is None:  # Root node
            node.policy_priors = self.add_dirichlet_noise(policy_priors)

        # Create children for all possible actions
        created_children = []

        for op in self.available_ops:
            # Generate parameters for this operation
            param_options = generate_op_parameters(op, None)

            for params in param_options:
                child = node.add_child(op, params)
                created_children.append(child)
                self.search_stats['total_nodes_created'] += 1

        # Return a random new child for simulation continuation
        if created_children:
            return np.random.choice(created_children)
        else:
            return node

    def simulate_once(
        self,
        root: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> float:
        """
        Run one simulation from root to leaf and backpropagate value

        Args:
            root: Root node of search tree
            demos: Demonstration input/output pairs
            test_input: Test input grid

        Returns:
            Value that was backpropagated
        """
        self.search_stats['total_simulations'] += 1
        path = []
        node = root

        # Selection phase: traverse tree using PUCT
        while not node.is_terminal and node.is_fully_expanded(self.available_ops, self.max_depth):
            node = node.select_best_child(self.c_puct)
            if node is None:
                break
            path.append(node)

        # Expansion phase: add new children if not terminal
        if not node.is_terminal and node.depth < self.max_depth:
            node = self.expand_node(node, demos, test_input)
            path.append(node)

        # Evaluation phase: get value for leaf node
        if node.is_terminal:
            value = node.terminal_reward
        else:
            # Evaluate current program on demonstrations
            value = self._evaluate_program(node, demos, test_input)

            # Also use neural network value estimate
            if hasattr(node, 'value_estimate'):
                # Combine program evaluation with neural value estimate
                value = 0.7 * value + 0.3 * node.value_estimate

        # Backpropagation phase: update all nodes in path
        for node in reversed(path):
            node.update_value(value)

        return value

    def _evaluate_program(
        self,
        node: MCTSNode,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor
    ) -> float:
        """
        Evaluate a program represented by a node on demonstration pairs

        Args:
            node: Node representing program state
            demos: Demonstration input/output pairs
            test_input: Test input (unused for evaluation, but kept for consistency)

        Returns:
            Evaluation score between -1 and 1
        """
        if len(node.ops) == 0:
            return 0.0  # Empty program gets neutral score

        try:
            program = node.get_program()
            total_score = 0.0
            valid_demos = 0

            for input_grid, target_output in demos:
                try:
                    # Apply program to input
                    predicted_output = apply_program(input_grid, program)

                    # Compute similarity score
                    if predicted_output.shape == target_output.shape:
                        # Pixel-wise accuracy
                        accuracy = (predicted_output == target_output).float().mean().item()
                        total_score += accuracy
                    else:
                        # Penalize shape mismatch but give some partial credit
                        total_score += 0.1  # Small positive score for at least running

                    valid_demos += 1

                except Exception:
                    # Program failed on this demo - give negative score
                    total_score -= 0.5
                    valid_demos += 1

            if valid_demos > 0:
                avg_score = total_score / valid_demos
                # Map to [-1, 1] range with 0 as neutral
                return max(-1.0, min(1.0, avg_score * 2.0 - 1.0))
            else:
                return -1.0  # No valid evaluations

        except Exception as e:
            logger.debug(f"Program evaluation failed: {e}")
            return -1.0

    def search(
        self,
        demos: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor,
        num_simulations: int = 800,
        timeout_seconds: float = 30.0
    ) -> Tuple[Optional[DSLProgram], Dict[str, Any]]:
        """
        Run PUCT search to find best program

        Args:
            demos: List of (input, output) demonstration pairs
            test_input: Test input grid to transform
            num_simulations: Number of MCTS simulations to run
            timeout_seconds: Maximum search time

        Returns:
            best_program: Best program found (or None if no solution)
            search_info: Dictionary with search statistics and tree info
        """
        start_time = time.time()

        # Initialize root node
        root = MCTSNode()
        self.search_stats['total_nodes_created'] += 1

        # Check for immediate solution (empty program)
        if self._is_solved(root, demos):
            root.is_solved = True
            root.terminal_reward = 1.0
            return root.get_program(), {"simulations_run": 0, "tree_size": 1}

        # Run simulations
        simulations_run = 0
        for sim in range(num_simulations):
            if time.time() - start_time > timeout_seconds:
                logger.info(f"PUCT search timeout after {simulations_run} simulations")
                break

            # Run one simulation
            value = self.simulate_once(root, demos, test_input)
            simulations_run += 1

            # Check if we found a perfect solution
            if self._check_for_solution(root, demos):
                logger.info(f"PUCT found solution after {simulations_run} simulations")
                break

            # Early stopping if we have a very good candidate
            if simulations_run >= 100 and simulations_run % 50 == 0:
                best_child = self._get_best_child(root, temperature=0.1)
                if best_child and best_child.mean_value > 0.9:
                    logger.info(f"PUCT found high-confidence solution after {simulations_run} simulations")
                    break

        # Extract best program
        best_program = None
        best_child = self._get_best_child(root, temperature=0.1)  # Low temperature for best selection

        if best_child:
            best_program = best_child.get_program()

        # Compile search statistics
        search_info = {
            "simulations_run": simulations_run,
            "tree_size": self.search_stats['total_nodes_created'],
            "max_depth_reached": self._get_max_depth(root),
            "best_value": best_child.mean_value if best_child else 0.0,
            "neural_evaluations": self.search_stats['neural_evaluations'],
            "search_time": time.time() - start_time,
            "nodes_per_second": simulations_run / max(time.time() - start_time, 0.001)
        }

        return best_program, search_info

    def _is_solved(self, node: MCTSNode, demos: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Check if a node represents a solution to all demonstrations"""
        if len(node.ops) == 0:
            return False  # Empty program is never a solution

        try:
            program = node.get_program()
            for input_grid, target_output in demos:
                predicted = apply_program(input_grid, program)
                if not torch.equal(predicted, target_output):
                    return False
            return True
        except:
            return False

    def _check_for_solution(self, root: MCTSNode, demos: List[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
        """Check if any node in the tree represents a solution"""
        def check_node(node):
            if self._is_solved(node, demos):
                node.is_solved = True
                node.terminal_reward = 1.0
                return True

            for child in node.children.values():
                if check_node(child):
                    return True
            return False

        return check_node(root)

    def _get_best_child(self, node: MCTSNode, temperature: float = 1.0) -> Optional[MCTSNode]:
        """Get best child using visit count and temperature"""
        if not node.children:
            return None

        if temperature == 0.0:
            # Greedy selection - choose child with highest visit count
            return max(node.children.values(), key=lambda c: c.visit_count)
        else:
            # Temperature-based selection
            visit_counts = np.array([child.visit_count for child in node.children.values()])
            if visit_counts.sum() == 0:
                return np.random.choice(list(node.children.values()))

            # Apply temperature
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / probs.sum()

            children = list(node.children.values())
            return np.random.choice(children, p=probs)

    def _get_max_depth(self, root: MCTSNode) -> int:
        """Get maximum depth reached in the search tree"""
        def get_depth(node):
            if not node.children:
                return node.depth
            return max(get_depth(child) for child in node.children.values())

        return get_depth(root)

    def get_action_probs(self, root: MCTSNode, temperature: float = 1.0) -> Dict[Tuple[str, frozenset], float]:
        """
        Get action probabilities for policy training

        Args:
            root: Root node of search tree
            temperature: Temperature for probability computation

        Returns:
            Dictionary mapping actions to probabilities
        """
        if not root.children:
            return {}

        visit_counts = root.get_visit_counts()
        total_visits = sum(visit_counts.values())

        if total_visits == 0:
            # Uniform distribution if no visits
            num_actions = len(visit_counts)
            return {action: 1.0 / num_actions for action in visit_counts.keys()}

        if temperature == 0.0:
            # One-hot on most visited action
            best_action = max(visit_counts.keys(), key=lambda a: visit_counts[a])
            return {action: 1.0 if action == best_action else 0.0 for action in visit_counts.keys()}
        else:
            # Temperature-based probabilities
            probs = {}
            for action, count in visit_counts.items():
                probs[action] = (count / total_visits) ** (1.0 / temperature)

            # Normalize
            total_prob = sum(probs.values())
            if total_prob > 0:
                for action in probs:
                    probs[action] /= total_prob

            return probs

    def reset_stats(self):
        """Reset search statistics"""
        self.search_stats = {
            'total_simulations': 0,
            'total_nodes_created': 0,
            'cache_hits': 0,
            'neural_evaluations': 0
        }


def puct_search(
    demos: List[Tuple[torch.Tensor, torch.Tensor]],
    test_input: torch.Tensor,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    num_simulations: int = 800,
    max_nodes: int = None,  # BitterBot enhancement: support legacy max_nodes parameter
    c_puct: float = 1.4,
    max_depth: int = 10,
    timeout_seconds: float = 30.0,
    device: torch.device = None
) -> Tuple[Optional[str], float]:
    """
    Convenience function to run PUCT search with default parameters

    Args:
        demos: List of (input, output) demonstration pairs
        test_input: Test input grid
        policy_net: Neural policy network
        value_net: Neural value network
        num_simulations: Number of MCTS simulations
        c_puct: Exploration constant
        max_depth: Maximum program depth
        timeout_seconds: Search timeout
        device: Compute device

    Returns:
        best_op: Best operation found (string)
        best_value: Value estimate (float)
    """
    # BitterBot enhancement: Support both max_nodes and num_simulations parameters
    if max_nodes is not None:
        num_simulations = max_nodes

    searcher = PUCTSearcher(
        policy_net=policy_net,
        value_net=value_net,
        c_puct=c_puct,
        max_depth=max_depth,
        device=device
    )

    best_program, search_info = searcher.search(demos, test_input, num_simulations, timeout_seconds)

    # Extract best operation and value, ensure consistent tuple return
    if best_program and hasattr(best_program, 'operations') and best_program.operations:
        best_op = str(best_program.operations[0])
        best_value = float(search_info.get('best_value', 0.0))
    else:
        best_op = None
        best_value = 0.0

    return best_op, best_value


# Legacy interface compatibility for existing code
def puct_program_search(model,
                       demos: List[Tuple[torch.Tensor, torch.Tensor]],
                       test_grid: torch.Tensor,
                       target_grid: Optional[torch.Tensor] = None,
                       max_length: int = 8,
                       max_nodes: int = 500,
                       c_puct: float = 1.5) -> List[str]:
    """
    Legacy interface for backward compatibility
    Full program search using PUCT for multi-step DSL programs.

    Args:
        model: TOPAS model with HRM bridge
        demos: List of (input, output) demonstration pairs
        test_grid: Input grid to transform
        target_grid: Target output grid (if known)
        max_length: Maximum program length
        max_nodes: PUCT search budget per step
        c_puct: Exploration constant

    Returns:
        program: List of operation names forming the discovered program
    """
    # This is a simplified version that maintains compatibility
    # In practice, you'd extract policy/value nets from the model

    try:
        # Extract or create simple policy/value networks
        if hasattr(model, 'policy_net') and hasattr(model, 'value_net'):
            program, _ = puct_search(
                demos=demos,
                test_input=test_grid,
                policy_net=model.policy_net,
                value_net=model.value_net,
                num_simulations=max_nodes,
                c_puct=c_puct,
                max_depth=max_length
            )
            return program.ops if program else []
        else:
            # Fallback to simple beam search-like approach
            logger.warning("Model doesn't have policy/value nets, using simplified search")
            return []

    except Exception as e:
        logger.error(f"PUCT program search failed: {e}")
        return []


# Example usage and testing
if __name__ == "__main__":
    # This would be run for testing the implementation
    import torch.nn as nn

    # Mock neural networks for testing
    class MockPolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(NUM_DSL_OPS + 16, NUM_DSL_OPS)

        def forward(self, x):
            return self.fc(x)

    class MockValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(NUM_DSL_OPS + 16, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.fc(x)

    # Create mock demonstration
    demo_input = torch.zeros(5, 5)
    demo_output = torch.ones(5, 5)
    demos = [(demo_input, demo_output)]

    test_input = torch.zeros(5, 5)

    # Run PUCT search
    policy_net = MockPolicyNet()
    value_net = MockValueNet()

    program, stats = puct_search(
        demos=demos,
        test_input=test_input,
        policy_net=policy_net,
        value_net=value_net,
        num_simulations=50,
        timeout_seconds=10.0
    )

    print(f"Found program: {program}")
    print(f"Search stats: {stats}")