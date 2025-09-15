"""
Example integration of PrioritizedReplay with existing TOPAS training systems.

Shows how to integrate the replay buffer with:
- Self-play generation
- Near-miss repair mining
- Deep program discovery
- Training loop integration

This serves as a reference for how to use the replay buffer in production.
"""

import torch
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict, Any, Optional
from trainers.replay import (
    PrioritizedReplay,
    add_self_play_traces,
    add_near_miss_repairs,
    add_deep_mining_programs
)

logger = logging.getLogger(__name__)


class ReplayIntegratedTrainer:
    """
    Example trainer that integrates prioritized replay with existing systems.

    Demonstrates the full pipeline:
    1. Generate traces from multiple sources
    2. Store in prioritized buffer
    3. Sample diverse traces for training
    4. Update priorities based on performance
    """

    def __init__(self,
                 model,
                 replay_config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer with replay buffer integration.

        Args:
            model: The TOPAS model to train
            replay_config: Configuration for replay buffer
        """
        self.model = model

        # Initialize replay buffer with default or custom config
        default_config = {
            'capacity': 20000,
            'alpha': 0.7,  # Slightly higher selectivity
            'novelty_weight': 0.4,
            'eviction_strategy': 'oldest_low_priority',
            'min_score_threshold': 0.15
        }

        if replay_config:
            default_config.update(replay_config)

        self.replay_buffer = PrioritizedReplay(**default_config)

        # Track training statistics
        self.training_step = 0
        self.replay_usage_stats = {
            'self_play_used': 0,
            'near_miss_used': 0,
            'deep_mining_used': 0,
            'total_replays': 0
        }

    def collect_self_play_traces(self,
                               demos: List[Dict[str, Any]],
                               wormhole_miner=None,
                               num_traces: int = 50) -> int:
        """
        Collect traces from self-play generation and add to replay buffer.

        Args:
            demos: Training demonstration data
            wormhole_miner: Optional wormhole template miner
            num_traces: Number of traces to generate

        Returns:
            Number of traces successfully added
        """
        try:
            # Use existing self-play system
            from trainers.self_play import SelfPlayBuffer

            self_play_buffer = SelfPlayBuffer(maxlen=num_traces * 2)
            generated_puzzles = self_play_buffer.generate_batch(
                demos, wormhole=wormhole_miner, top_k=num_traces
            )

            # Convert to replay buffer format
            self_play_data = []
            for i, (input_grid, output_grid) in enumerate(generated_puzzles):
                # Simulate program generation (in practice, this would come from the generator)
                program = ['rotate90'] if i % 3 == 0 else ['flip_h', 'color_map']
                params = [{}] if len(program) == 1 else [{}, {'mapping': {1: 2}}]

                # Score based on grid complexity and transformation success
                score = min(0.9, 0.4 + (torch.unique(output_grid).numel() / 10.0))

                self_play_data.append({
                    'task_id': f'selfplay_{self.training_step}_{i}',
                    'program': program,
                    'params': params,
                    'score': score,
                    'input_grid': input_grid,
                    'output_grid': output_grid,
                    'metadata': {
                        'generation_step': self.training_step,
                        'wormhole_used': wormhole_miner is not None
                    }
                })

            # Add to replay buffer
            added_count = add_self_play_traces(self.replay_buffer, self_play_data)
            self.replay_usage_stats['self_play_used'] += added_count

            logger.info(f"[Replay] Added {added_count} self-play traces")
            return added_count

        except Exception as e:
            logger.error(f"[Replay] Self-play trace collection failed: {e}")
            return 0

    def collect_near_miss_traces(self,
                               failed_predictions: List[torch.Tensor],
                               target_outputs: List[torch.Tensor],
                               task_ids: List[str]) -> int:
        """
        Collect traces from near-miss repair and add to replay buffer.

        Args:
            failed_predictions: Model predictions that didn't match targets
            target_outputs: Ground truth outputs
            task_ids: Task identifiers

        Returns:
            Number of repair traces successfully added
        """
        try:
            # Use existing near-miss mining
            from trainers.near_miss import integrate_near_miss_learning

            # This returns the number of repairs found
            repair_count = integrate_near_miss_learning(
                self.model, failed_predictions, target_outputs, task_ids, self.replay_buffer
            )

            self.replay_usage_stats['near_miss_used'] += repair_count
            logger.info(f"[Replay] Added {repair_count} near-miss repair traces")
            return repair_count

        except Exception as e:
            logger.error(f"[Replay] Near-miss trace collection failed: {e}")
            return 0

    def collect_deep_mining_traces(self,
                                 tasks: List[Dict[str, Any]],
                                 max_depth: int = 8,
                                 num_tasks: int = 10) -> int:
        """
        Collect traces from deep program mining and add to replay buffer.

        Args:
            tasks: Task definitions with input/output examples
            max_depth: Maximum program depth to mine
            num_tasks: Number of tasks to mine

        Returns:
            Number of mining traces successfully added
        """
        try:
            # Use existing deep mining
            from trainers.augmentation.deep_program_discoverer import mine_deep_programs

            mining_results = []
            for task in tasks[:num_tasks]:
                deep_programs = mine_deep_programs(task, max_depth=max_depth)
                mining_results.extend(deep_programs)

            # Add to replay buffer
            added_count = add_deep_mining_programs(self.replay_buffer, mining_results)
            self.replay_usage_stats['deep_mining_used'] += added_count

            logger.info(f"[Replay] Added {added_count} deep mining traces")
            return added_count

        except Exception as e:
            logger.error(f"[Replay] Deep mining trace collection failed: {e}")
            return 0

    def sample_training_batch(self,
                            batch_size: int,
                            source_ratios: Optional[Dict[str, float]] = None,
                            temperature: float = 1.0) -> List[Dict[str, Any]]:
        """
        Sample a diverse training batch from the replay buffer.

        Args:
            batch_size: Number of traces to sample
            source_ratios: Desired ratio of different trace sources
            temperature: Sampling temperature (lower = more selective)

        Returns:
            List of training examples in format expected by model
        """
        if source_ratios:
            # Use balanced sampling
            sampled_traces = self.replay_buffer.sample_balanced(batch_size, source_ratios)
        else:
            # Use priority-weighted sampling
            sampled_traces = self.replay_buffer.sample(batch_size, temperature=temperature)

        # Convert traces to training format
        training_batch = []
        for trace in sampled_traces:
            training_example = {
                'task_id': trace.task_id,
                'input': trace.input_grid,
                'output': trace.output_grid,
                'target': trace.target_grid,
                'program': trace.program,
                'params': trace.params,
                'source': trace.source,
                'score': trace.score,
                'metadata': trace.metadata
            }
            training_batch.append(training_example)

        self.replay_usage_stats['total_replays'] += len(training_batch)
        return training_batch

    def update_trace_priorities(self,
                              task_ids: List[str],
                              new_scores: List[float]):
        """
        Update priorities of traces based on new performance scores.

        Args:
            task_ids: Identifiers of traces to update
            new_scores: New performance scores for these traces
        """
        # Find and update matching traces
        updated_count = 0
        for task_id, new_score in zip(task_ids, new_scores):
            for i, trace in enumerate(self.replay_buffer.traces):
                if trace.task_id == task_id:
                    # Update score and recompute priority
                    trace.score = max(trace.score, new_score)  # Keep best score
                    new_priority = self.replay_buffer._compute_priority(trace)
                    self.replay_buffer.priorities[i] = new_priority
                    updated_count += 1
                    break

        logger.debug(f"[Replay] Updated priorities for {updated_count} traces")

    def training_step_with_replay(self,
                                demos: List[Dict[str, Any]],
                                failed_preds: Optional[List[torch.Tensor]] = None,
                                target_outputs: Optional[List[torch.Tensor]] = None,
                                task_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform one training step with replay buffer integration.

        Args:
            demos: Current training demonstrations
            failed_preds: Recent failed predictions for near-miss mining
            target_outputs: Ground truth outputs for failed predictions
            task_ids: Task IDs for failed predictions

        Returns:
            Training step results and statistics
        """
        step_stats = {
            'replay_added': 0,
            'replay_sampled': 0,
            'buffer_size': len(self.replay_buffer.traces)
        }

        # 1. Collect new traces from various sources
        if self.training_step % 10 == 0:  # Collect self-play traces every 10 steps
            added = self.collect_self_play_traces(demos, num_traces=20)
            step_stats['replay_added'] += added

        if failed_preds and target_outputs and task_ids:
            # Collect near-miss repairs from failed predictions
            added = self.collect_near_miss_traces(failed_preds, target_outputs, task_ids)
            step_stats['replay_added'] += added

        if self.training_step % 25 == 0:  # Deep mining less frequently
            added = self.collect_deep_mining_traces(demos, num_tasks=5)
            step_stats['replay_added'] += added

        # 2. Sample diverse training batch
        replay_batch_size = min(16, len(self.replay_buffer.traces))
        if replay_batch_size > 0:
            # Use balanced sampling across sources
            source_ratios = {'self_play': 0.5, 'near_miss_repair': 0.3, 'deep_mining': 0.2}
            replay_batch = self.sample_training_batch(
                replay_batch_size,
                source_ratios=source_ratios,
                temperature=0.8  # Slightly selective
            )
            step_stats['replay_sampled'] = len(replay_batch)

            # TODO: Integrate replay batch with main training here
            # This would involve:
            # - Combining replay batch with regular training data
            # - Training model on the combined batch
            # - Computing loss and updating model parameters

        # 3. Update training step counter
        self.training_step += 1

        return step_stats

    def get_replay_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about replay buffer usage."""
        buffer_stats = self.replay_buffer.get_statistics()

        return {
            'buffer': buffer_stats,
            'usage': self.replay_usage_stats,
            'training_step': self.training_step,
            'efficiency': {
                'replay_ratio': (self.replay_usage_stats['total_replays'] /
                               max(1, self.training_step)),
                'source_diversity': len(buffer_stats['source_distribution'])
            }
        }

    def save_replay_buffer(self, filepath: str):
        """Save replay buffer for resuming training."""
        self.replay_buffer.export_traces(filepath)
        logger.info(f"[Replay] Saved buffer to {filepath}")

    def adaptive_sampling_strategy(self, epoch: int) -> Dict[str, Any]:
        """
        Adaptive strategy that changes sampling based on training progress.

        Early training: More diverse, higher temperature
        Late training: More selective, focus on high-quality traces
        """
        if epoch < 10:
            # Early training: explore diversity
            return {
                'temperature': 1.5,
                'source_ratios': {'self_play': 0.6, 'near_miss_repair': 0.2, 'deep_mining': 0.2},
                'min_score_override': 0.1
            }
        elif epoch < 50:
            # Mid training: balanced approach
            return {
                'temperature': 1.0,
                'source_ratios': {'self_play': 0.4, 'near_miss_repair': 0.4, 'deep_mining': 0.2},
                'min_score_override': 0.2
            }
        else:
            # Late training: focus on quality
            return {
                'temperature': 0.6,
                'source_ratios': {'self_play': 0.2, 'near_miss_repair': 0.5, 'deep_mining': 0.3},
                'min_score_override': 0.3
            }


def example_usage():
    """
    Example of how to use the replay-integrated trainer.
    """

    # This would be your actual TOPAS model
    mock_model = None

    # Initialize trainer with custom replay config
    replay_config = {
        'capacity': 15000,
        'alpha': 0.8,
        'novelty_weight': 0.3,
        'min_score_threshold': 0.2
    }

    trainer = ReplayIntegratedTrainer(mock_model, replay_config)

    # Simulate training loop
    for epoch in range(5):
        print(f"\n--- Epoch {epoch} ---")

        # Get adaptive strategy for this epoch
        strategy = trainer.adaptive_sampling_strategy(epoch)
        print(f"Strategy: temp={strategy['temperature']}, ratios={strategy['source_ratios']}")

        # Simulate training steps
        for step in range(3):
            # Mock training data
            demos = [{'input': torch.randint(0, 10, (3, 3)),
                     'output': torch.randint(0, 10, (3, 3))} for _ in range(5)]

            # Perform training step with replay
            step_stats = trainer.training_step_with_replay(demos)
            print(f"  Step {step}: added={step_stats['replay_added']}, "
                  f"sampled={step_stats['replay_sampled']}, "
                  f"buffer_size={step_stats['buffer_size']}")

    # Final statistics
    final_stats = trainer.get_replay_statistics()
    print(f"\nFinal buffer size: {final_stats['buffer']['size']}")
    print(f"Source distribution: {final_stats['buffer']['source_distribution']}")
    print(f"Usage stats: {final_stats['usage']}")


if __name__ == "__main__":
    example_usage()