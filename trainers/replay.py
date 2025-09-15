"""
Prioritized Replay Buffer for Program Traces

Stores and samples diverse program traces from multiple sources:
- Self-play: Short programs from automated generation
- Deep mining: Complex programs from exhaustive search
- Near-miss repairs: Fixed programs from failed attempts

Uses priority-based sampling with configurable weighting and automatic eviction.
"""

import torch
import numpy as np
import random
import heapq
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProgramTrace:
    """
    Represents a complete program trace with metadata for prioritized sampling.

    Attributes:
        task_id: Unique identifier for the source task
        program: List of DSL operation names
        params: List of parameter dictionaries for each operation
        score: Success/accuracy score (0.0 to 1.0)
        novelty: Novelty metric (0.0 to 1.0, higher = more novel)
        depth: Program length/complexity
        source: Trace source ('self_play', 'deep_mining', 'near_miss_repair')
        input_grid: Input grid tensor
        output_grid: Output grid tensor
        target_grid: Expected output grid tensor (if available)
        timestamp: When trace was added to buffer
        metadata: Additional source-specific metadata
    """
    task_id: str
    program: List[str]
    params: List[Dict[str, Any]]
    score: float
    novelty: float
    depth: int
    source: str
    input_grid: torch.Tensor
    output_grid: torch.Tensor
    target_grid: Optional[torch.Tensor] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trace data on creation"""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0,1], got {self.score}")
        if not 0.0 <= self.novelty <= 1.0:
            raise ValueError(f"Novelty must be in [0,1], got {self.novelty}")
        if len(self.program) != len(self.params):
            raise ValueError("Program ops and params must have same length")
        if self.source not in ['self_play', 'deep_mining', 'near_miss_repair']:
            logger.warning(f"Unknown trace source: {self.source}")


class PrioritizedReplay:
    """
    Prioritized replay buffer for storing and sampling diverse program traces.

    Uses priority-based sampling with configurable weighting:
    - Priority = score^alpha * (1 + novelty)
    - Supports automatic eviction of old/low-priority traces
    - Tracks statistics and source distribution
    """

    def __init__(self,
                 capacity: int = 10000,
                 alpha: float = 0.6,
                 novelty_weight: float = 0.5,
                 eviction_strategy: str = 'oldest_low_priority',
                 min_score_threshold: float = 0.1):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of traces to store
            alpha: Priority exponent for score weighting (higher = more selective)
            novelty_weight: Weight for novelty component in priority calculation
            eviction_strategy: Strategy for removing traces when full
                - 'oldest_low_priority': Remove oldest traces with lowest priority
                - 'random_low_priority': Remove random traces with priority < median
                - 'fifo': Simple first-in-first-out
            min_score_threshold: Minimum score required to add trace
        """
        self.capacity = capacity
        self.alpha = alpha
        self.novelty_weight = novelty_weight
        self.eviction_strategy = eviction_strategy
        self.min_score_threshold = min_score_threshold

        # Storage
        self.traces: List[ProgramTrace] = []
        self.priorities: List[float] = []
        self._next_id = 0

        # Statistics
        self.source_counts = defaultdict(int)
        self.total_pushes = 0
        self.total_samples = 0
        self.eviction_count = 0

        # Novelty tracking for scoring
        self.program_signatures: Dict[str, int] = defaultdict(int)
        self.operation_counts: Dict[str, int] = defaultdict(int)

    def _compute_priority(self, trace: ProgramTrace) -> float:
        """
        Compute sampling priority for a trace.

        Priority = score^alpha * (1 + novelty_weight * novelty)
        """
        base_priority = (trace.score ** self.alpha) if trace.score > 0 else 0.001
        novelty_bonus = 1 + (self.novelty_weight * trace.novelty)
        return base_priority * novelty_bonus

    def _compute_novelty(self, trace: ProgramTrace) -> float:
        """
        Compute novelty score based on program signature and operation frequency.

        Returns value in [0, 1] where 1 = completely novel
        """
        # Program signature based on operation sequence
        program_sig = '->'.join(trace.program)
        sig_frequency = self.program_signatures.get(program_sig, 0)

        # Operation rarity score
        op_rarities = []
        for op in trace.program:
            op_count = self.operation_counts.get(op, 0)
            total_ops = sum(self.operation_counts.values()) or 1
            rarity = 1.0 - (op_count / total_ops)
            op_rarities.append(rarity)

        avg_op_rarity = np.mean(op_rarities) if op_rarities else 1.0

        # Combine signature novelty and operation rarity
        sig_novelty = 1.0 / (1.0 + sig_frequency)
        combined_novelty = 0.7 * sig_novelty + 0.3 * avg_op_rarity

        return min(1.0, combined_novelty)

    def _update_novelty_tracking(self, trace: ProgramTrace):
        """Update internal tracking for novelty computation"""
        program_sig = '->'.join(trace.program)
        self.program_signatures[program_sig] += 1

        for op in trace.program:
            self.operation_counts[op] += 1

    def _evict_traces(self, num_to_evict: int):
        """
        Remove traces according to eviction strategy.
        """
        if len(self.traces) <= num_to_evict:
            return

        if self.eviction_strategy == 'oldest_low_priority':
            # Sort by (priority, -timestamp) to get lowest priority oldest first
            indexed_traces = [(i, self.priorities[i], -self.traces[i].timestamp)
                            for i in range(len(self.traces))]
            indexed_traces.sort(key=lambda x: (x[1], x[2]))
            indices_to_remove = [idx for idx, _, _ in indexed_traces[:num_to_evict]]

        elif self.eviction_strategy == 'random_low_priority':
            # Remove random traces with below-median priority
            median_priority = np.median(self.priorities) if self.priorities else 0
            low_priority_indices = [i for i, p in enumerate(self.priorities)
                                  if p < median_priority]
            if len(low_priority_indices) >= num_to_evict:
                indices_to_remove = random.sample(low_priority_indices, num_to_evict)
            else:
                # Fall back to random if not enough low priority traces
                indices_to_remove = random.sample(range(len(self.traces)), num_to_evict)

        elif self.eviction_strategy == 'fifo':
            # Remove oldest traces
            indices_to_remove = list(range(num_to_evict))
        else:
            # Default to oldest_low_priority
            indices_to_remove = list(range(num_to_evict))

        # Remove traces in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            removed_trace = self.traces.pop(idx)
            self.priorities.pop(idx)
            self.source_counts[removed_trace.source] -= 1
            self.eviction_count += 1

        logger.debug(f"Evicted {num_to_evict} traces using {self.eviction_strategy}")

    def push(self,
             task_id: str,
             program: List[str],
             params: List[Dict[str, Any]],
             score: float,
             input_grid: torch.Tensor,
             output_grid: torch.Tensor,
             target_grid: Optional[torch.Tensor] = None,
             source: str = 'unknown',
             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a program trace to the buffer.

        Args:
            task_id: Unique identifier for source task
            program: List of DSL operation names
            params: List of parameter dictionaries
            score: Success/accuracy score (0.0 to 1.0)
            input_grid: Input grid tensor
            output_grid: Output grid tensor
            target_grid: Expected output (for scoring)
            source: Trace source identifier
            metadata: Additional metadata dictionary

        Returns:
            True if trace was added, False if rejected
        """
        if score < self.min_score_threshold:
            logger.debug(f"Rejecting trace with score {score} < {self.min_score_threshold}")
            return False

        try:
            # Create trace object
            trace = ProgramTrace(
                task_id=task_id,
                program=program.copy(),
                params=[p.copy() for p in params],
                score=score,
                novelty=0.0,  # Will be computed below
                depth=len(program),
                source=source,
                input_grid=input_grid.clone(),
                output_grid=output_grid.clone(),
                target_grid=target_grid.clone() if target_grid is not None else None,
                metadata=metadata.copy() if metadata else {}
            )

            # Compute novelty based on current buffer state
            trace.novelty = self._compute_novelty(trace)

            # Check if buffer is full and evict if needed
            if len(self.traces) >= self.capacity:
                self._evict_traces(1)

            # Add trace and compute priority
            self.traces.append(trace)
            priority = self._compute_priority(trace)
            self.priorities.append(priority)

            # Update tracking
            self._update_novelty_tracking(trace)
            self.source_counts[source] += 1
            self.total_pushes += 1

            logger.debug(f"Added trace: {task_id} ({source}) "
                       f"score={score:.3f} novelty={trace.novelty:.3f} priority={priority:.3f}")
            return True

        except Exception as e:
            logger.error(f"Failed to add trace {task_id}: {e}")
            return False

    def sample(self,
               n: int,
               temperature: float = 1.0,
               source_filter: Optional[List[str]] = None) -> List[ProgramTrace]:
        """
        Sample n traces using priority-weighted sampling.

        Args:
            n: Number of traces to sample
            temperature: Sampling temperature (lower = more selective)
            source_filter: Only sample from these sources (None = all sources)

        Returns:
            List of sampled ProgramTrace objects
        """
        if not self.traces or n <= 0:
            return []

        # Filter by source if specified
        valid_indices = list(range(len(self.traces)))
        if source_filter:
            valid_indices = [i for i in valid_indices
                           if self.traces[i].source in source_filter]

        if not valid_indices:
            logger.warning(f"No traces match source filter: {source_filter}")
            return []

        # Get priorities for valid traces
        valid_priorities = [self.priorities[i] for i in valid_indices]

        # Apply temperature scaling
        if temperature != 1.0:
            valid_priorities = [p ** (1.0 / temperature) for p in valid_priorities]

        # Handle edge cases
        if all(p == 0 for p in valid_priorities):
            # All priorities are zero, sample uniformly
            sampled_indices = random.choices(valid_indices, k=min(n, len(valid_indices)))
        else:
            # Priority-weighted sampling
            sampled_indices = random.choices(
                valid_indices,
                weights=valid_priorities,
                k=min(n, len(valid_indices))
            )

        sampled_traces = [self.traces[i] for i in sampled_indices]
        self.total_samples += len(sampled_traces)

        logger.debug(f"Sampled {len(sampled_traces)} traces with temperature={temperature}")
        return sampled_traces

    def sample_balanced(self,
                       n: int,
                       source_ratios: Optional[Dict[str, float]] = None) -> List[ProgramTrace]:
        """
        Sample traces with balanced representation across sources.

        Args:
            n: Total number of traces to sample
            source_ratios: Desired ratio for each source (None = equal split)

        Returns:
            List of sampled traces
        """
        if not self.traces or n <= 0:
            return []

        # Default to equal ratios if not specified
        available_sources = list(self.source_counts.keys())
        if not source_ratios:
            source_ratios = {source: 1.0/len(available_sources)
                           for source in available_sources}

        # Normalize ratios
        total_ratio = sum(source_ratios.values())
        if total_ratio > 0:
            source_ratios = {k: v/total_ratio for k, v in source_ratios.items()}

        # Sample from each source
        all_samples = []
        for source, ratio in source_ratios.items():
            source_n = int(n * ratio)
            if source_n > 0:
                source_samples = self.sample(source_n, source_filter=[source])
                all_samples.extend(source_samples)

        # Fill remaining slots if needed
        remaining = n - len(all_samples)
        if remaining > 0:
            extra_samples = self.sample(remaining)
            all_samples.extend(extra_samples)

        return all_samples[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics and metrics.

        Returns:
            Dictionary with buffer statistics
        """
        if not self.traces:
            return {
                'size': 0,
                'capacity': self.capacity,
                'total_pushes': self.total_pushes,
                'total_samples': self.total_samples,
                'eviction_count': self.eviction_count,
                'source_distribution': {},
                'score_stats': {},
                'novelty_stats': {},
                'depth_stats': {},
                'priority_stats': {}
            }

        scores = [t.score for t in self.traces]
        novelties = [t.novelty for t in self.traces]
        depths = [t.depth for t in self.traces]

        return {
            'size': len(self.traces),
            'capacity': self.capacity,
            'total_pushes': self.total_pushes,
            'total_samples': self.total_samples,
            'eviction_count': self.eviction_count,
            'source_distribution': dict(self.source_counts),
            'score_stats': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            },
            'novelty_stats': {
                'mean': np.mean(novelties),
                'std': np.std(novelties),
                'min': np.min(novelties),
                'max': np.max(novelties),
                'median': np.median(novelties)
            },
            'depth_stats': {
                'mean': np.mean(depths),
                'std': np.std(depths),
                'min': np.min(depths),
                'max': np.max(depths),
                'median': np.median(depths)
            },
            'priority_stats': {
                'mean': np.mean(self.priorities),
                'std': np.std(self.priorities),
                'min': np.min(self.priorities),
                'max': np.max(self.priorities),
                'median': np.median(self.priorities)
            }
        }

    def clear(self):
        """Clear all traces and reset statistics"""
        self.traces.clear()
        self.priorities.clear()
        self.source_counts.clear()
        self.program_signatures.clear()
        self.operation_counts.clear()
        self.total_pushes = 0
        self.total_samples = 0
        self.eviction_count = 0
        logger.info("Replay buffer cleared")

    def resize(self, new_capacity: int):
        """
        Resize buffer capacity, evicting traces if necessary.

        Args:
            new_capacity: New maximum capacity
        """
        old_capacity = self.capacity
        self.capacity = new_capacity

        if len(self.traces) > new_capacity:
            excess = len(self.traces) - new_capacity
            self._evict_traces(excess)

        logger.info(f"Buffer capacity changed: {old_capacity} -> {new_capacity}")

    def get_top_traces(self, n: int = 10) -> List[Tuple[ProgramTrace, float]]:
        """
        Get the n highest priority traces.

        Args:
            n: Number of top traces to return

        Returns:
            List of (trace, priority) tuples sorted by priority (descending)
        """
        if not self.traces:
            return []

        trace_priority_pairs = list(zip(self.traces, self.priorities))
        trace_priority_pairs.sort(key=lambda x: x[1], reverse=True)

        return trace_priority_pairs[:n]

    def export_traces(self,
                     filepath: str,
                     source_filter: Optional[List[str]] = None,
                     min_score: Optional[float] = None):
        """
        Export traces to file for analysis or backup.

        Args:
            filepath: Output file path
            source_filter: Only export traces from these sources
            min_score: Minimum score threshold for export
        """
        try:
            import pickle

            # Filter traces based on criteria
            filtered_traces = []
            for trace in self.traces:
                if source_filter and trace.source not in source_filter:
                    continue
                if min_score and trace.score < min_score:
                    continue
                filtered_traces.append(trace)

            # Export to pickle file
            export_data = {
                'traces': filtered_traces,
                'metadata': {
                    'export_time': time.time(),
                    'total_traces': len(self.traces),
                    'filtered_traces': len(filtered_traces),
                    'buffer_stats': self.get_statistics()
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(export_data, f)

            logger.info(f"Exported {len(filtered_traces)} traces to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export traces: {e}")


def create_replay_buffer_from_config(config: Dict[str, Any]) -> PrioritizedReplay:
    """
    Create replay buffer from configuration dictionary.

    Args:
        config: Configuration with keys like 'capacity', 'alpha', etc.

    Returns:
        Configured PrioritizedReplay instance
    """
    return PrioritizedReplay(
        capacity=config.get('capacity', 10000),
        alpha=config.get('alpha', 0.6),
        novelty_weight=config.get('novelty_weight', 0.5),
        eviction_strategy=config.get('eviction_strategy', 'oldest_low_priority'),
        min_score_threshold=config.get('min_score_threshold', 0.1)
    )


# Convenience functions for integration with existing systems

def add_self_play_traces(replay_buffer: PrioritizedReplay,
                        self_play_data: List[Dict[str, Any]],
                        base_score: float = 0.6) -> int:
    """
    Add self-play generated traces to replay buffer.

    Args:
        replay_buffer: Target replay buffer
        self_play_data: List of self-play trace dictionaries
        base_score: Base score for self-play traces

    Returns:
        Number of traces successfully added
    """
    added_count = 0

    for data in self_play_data:
        try:
            success = replay_buffer.push(
                task_id=data.get('task_id', f'selfplay_{int(time.time())}'),
                program=data.get('program', []),
                params=data.get('params', []),
                score=data.get('score', base_score),
                input_grid=data['input_grid'],
                output_grid=data['output_grid'],
                target_grid=data.get('target_grid'),
                source='self_play',
                metadata=data.get('metadata', {})
            )
            if success:
                added_count += 1

        except Exception as e:
            logger.warning(f"Failed to add self-play trace: {e}")

    return added_count


def add_near_miss_repairs(replay_buffer: PrioritizedReplay,
                         repair_macros: List[Dict[str, Any]]) -> int:
    """
    Add near-miss repair traces to replay buffer.

    Args:
        replay_buffer: Target replay buffer
        repair_macros: List of repair macro dictionaries

    Returns:
        Number of traces successfully added
    """
    added_count = 0

    for macro in repair_macros:
        try:
            success = replay_buffer.push(
                task_id=macro['task_id'],
                program=macro['repair_ops'],
                params=[{} for _ in macro['repair_ops']],  # Simple params for repairs
                score=macro['improvement'],
                input_grid=macro['original_pred'],
                output_grid=macro['repaired_grid'],
                target_grid=macro['target_grid'],
                source='near_miss_repair',
                metadata={
                    'initial_distance': macro['initial_distance'],
                    'final_distance': macro['final_distance']
                }
            )
            if success:
                added_count += 1

        except Exception as e:
            logger.warning(f"Failed to add repair trace: {e}")

    return added_count


def add_deep_mining_programs(replay_buffer: PrioritizedReplay,
                           mining_results: List[Dict[str, Any]]) -> int:
    """
    Add deep mining program traces to replay buffer.

    Args:
        replay_buffer: Target replay buffer
        mining_results: List of deep mining result dictionaries

    Returns:
        Number of traces successfully added
    """
    added_count = 0

    for result in mining_results:
        try:
            success = replay_buffer.push(
                task_id=result.get('task_id', f'mining_{int(time.time())}'),
                program=result['operations'],
                params=result.get('parameters', []),
                score=result['score'],
                input_grid=result['input_grid'],
                output_grid=result['output_grid'],
                target_grid=result.get('target_grid'),
                source='deep_mining',
                metadata={
                    'depth': len(result['operations']),
                    'mining_method': result.get('method', 'unknown')
                }
            )
            if success:
                added_count += 1

        except Exception as e:
            logger.warning(f"Failed to add deep mining trace: {e}")

    return added_count