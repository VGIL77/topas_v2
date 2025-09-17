"""
Hippocampal Sharp-Wave Ripple Substrate Implementation

This module implements hippocampal sharp-wave ripples (150-200Hz oscillations) that occur
during offline consolidation. During ripple bursts, STDP gain is multiplied and 
phase-locked replay can occur.

Key Features:
- Poisson-distributed ripple event scheduling
- Phase calculation with optional QREF metric warp
- Coherence tracking and metrics
- STDP gain modulation during ripple events
- Phase-locked replay alignment
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RippleConfig:
    """Configuration for hippocampal ripple substrate"""
    
    # Core ripple parameters
    event_rate_hz: float = 0.8                 # Ripples per second (physiological range)
    center_freq_hz: float = 170.0              # Carrier frequency (150-200Hz range)
    burst_duration_ms: Tuple[float, float] = (50.0, 150.0)  # Min/max burst duration
    dt_ms: float = 2.0                         # Time step in milliseconds (for sampling rate)
    
    # STDP modulation
    stdp_gain: float = 3.0                     # Multiplier during ripple events
    phase_lock: bool = True                    # Align replay to ripple phase
    
    # Phase and coherence
    phase_bins: int = 8                        # Number of phase bins for analysis
    coherence_window_ms: float = 200.0         # Window for coherence calculation
    
    # Optional QREF-style metric warp
    metric_kappa: Optional[float] = None       # Energy warp parameter
    qref_threshold: float = 0.5                # Threshold for metric warping
    
    # Performance parameters
    buffer_size: int = 1000                    # Size of event buffer
    coherence_history: int = 50                # Number of coherence samples to keep
    
    # Unified clocking parameters
    dt_ms: float = 2.0                         # Sampling time step in milliseconds
    
    # Validation bounds
    min_event_rate: float = 0.1                # Minimum event rate (Hz)
    max_event_rate: float = 5.0                # Maximum event rate (Hz)
    min_center_freq: float = 120.0             # Minimum center frequency (Hz)
    max_center_freq: float = 250.0             # Maximum center frequency (Hz)
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate timing parameters
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        
        # Compute sampling frequency and validate Nyquist condition
        fs_hz = 1000.0 / self.dt_ms
        nyquist_freq = 0.45 * fs_hz  # 0.45 to leave margin below 0.5*fs
        
        if self.center_freq_hz > nyquist_freq:
            raise ValueError(f"center_freq_hz {self.center_freq_hz} Hz exceeds Nyquist limit "
                           f"{nyquist_freq:.1f} Hz for dt_ms={self.dt_ms} ms (fs={fs_hz:.1f} Hz)")
        
        if not (self.min_event_rate <= self.event_rate_hz <= self.max_event_rate):
            raise ValueError(f"event_rate_hz {self.event_rate_hz} outside valid range "
                           f"[{self.min_event_rate}, {self.max_event_rate}]")
        
        if not (self.min_center_freq <= self.center_freq_hz <= self.max_center_freq):
            raise ValueError(f"center_freq_hz {self.center_freq_hz} outside valid range "
                           f"[{self.min_center_freq}, {self.max_center_freq}]")
        
        if self.burst_duration_ms[0] >= self.burst_duration_ms[1]:
            raise ValueError("burst_duration_ms min must be less than max")
        
        if self.stdp_gain <= 0:
            raise ValueError("stdp_gain must be positive")
        
        if self.phase_bins <= 0:
            raise ValueError("phase_bins must be positive")
        
        if self.coherence_window_ms <= 0:
            raise ValueError("coherence_window_ms must be positive")
        
        if self.metric_kappa is not None and self.metric_kappa <= 0:
            raise ValueError("metric_kappa must be positive if specified")


class RippleContext:
    """Efficient context for ripple state using __slots__"""
    
    __slots__ = [
        'is_active', 'phase', 'amplitude', 'frequency', 'burst_start_time',
        'burst_duration', 'stdp_multiplier', 'coherence', 'phase_bin',
        'qref_warp_factor', 'replay_alignment', 'event_id'
    ]
    
    def __init__(self):
        self.is_active: bool = False
        self.phase: float = 0.0
        self.amplitude: float = 0.0
        self.frequency: float = 0.0
        self.burst_start_time: float = 0.0
        self.burst_duration: float = 0.0
        self.stdp_multiplier: float = 1.0
        self.coherence: float = 0.0
        self.phase_bin: int = 0
        self.qref_warp_factor: float = 1.0
        self.replay_alignment: float = 0.0
        self.event_id: int = 0
    
    def reset(self):
        """Reset context to inactive state"""
        self.is_active = False
        self.phase = 0.0
        self.amplitude = 0.0
        self.frequency = 0.0
        self.burst_start_time = 0.0
        self.burst_duration = 0.0
        self.stdp_multiplier = 1.0
        self.coherence = 0.0
        self.phase_bin = 0
        self.qref_warp_factor = 1.0
        self.replay_alignment = 0.0
        self.event_id = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging/analysis"""
        return {
            'is_active': self.is_active,
            'phase': self.phase,
            'amplitude': self.amplitude,
            'frequency': self.frequency,
            'burst_start_time': self.burst_start_time,
            'burst_duration': self.burst_duration,
            'stdp_multiplier': self.stdp_multiplier,
            'coherence': self.coherence,
            'phase_bin': self.phase_bin,
            'qref_warp_factor': self.qref_warp_factor,
            'replay_alignment': self.replay_alignment,
            'event_id': self.event_id
        }


class RippleSubstrate:
    """
    Hippocampal sharp-wave ripple substrate implementation
    
    Implements physiological ripple oscillations (150-200Hz) that occur during
    offline consolidation periods. During ripple bursts:
    - STDP gain is multiplied by config.stdp_gain
    - Phase-locked replay can be aligned to ripple phase
    - Coherence metrics track oscillation quality
    """
    
    def __init__(self, config: RippleConfig):
        self.config = config
        self.context = RippleContext()
        
        # Compute sampling frequency and validate Nyquist
        self._fs_hz = 1000.0 / config.dt_ms
        self._validate_nyquist()
        
        # Poisson process state
        self._next_event_time = 0.0
        self._current_time = 0.0
        self._event_counter = 0
        
        # Phase tracking
        self._phase_accumulator = 0.0
        self._last_phase_time = 0.0
        
        # Coherence tracking
        self._coherence_buffer = np.zeros(config.coherence_history)
        self._coherence_index = 0
        self._phase_history = []
        self._amplitude_history = []
        
        # Event buffer for analysis
        self._event_buffer = []
        self._max_buffer_size = config.buffer_size
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'total_duration': 0.0,
            'avg_coherence': 0.0,
            'phase_distribution': np.zeros(config.phase_bins),
            'stdp_activations': 0
        }
        
        # Coherence tracking counters (reset each cycle)
        self._coherence_samples = 0
        self._coherence_sum = 0.0
        self._active_steps = 0  # Counter for active ripple steps
        self._events = []  # Track events in current cycle
        
        # Initialize first event
        self._schedule_next_event()
        
        # Validate and log frequency parameters
        fs_hz = 1000.0 / config.dt_ms
        nyquist_freq = 0.45 * fs_hz
        
        logger.info(f"[Ripple] fs={fs_hz:.1f}Hz center={config.center_freq_hz:.1f}Hz OK")
        logger.info(f"RippleSubstrate initialized with rate {config.event_rate_hz} Hz, "
                   f"center freq {config.center_freq_hz} Hz, dt_ms={config.dt_ms} ms")
        logger.debug(f"[Ripple] fs={self._fs_hz:.1f}Hz center={config.center_freq_hz}Hz OK")
    
    def _validate_nyquist(self) -> None:
        """Validate that center frequency satisfies Nyquist criterion"""
        nyquist_freq = 0.5 * self._fs_hz
        safe_freq = 0.45 * self._fs_hz  # Use 0.45 for safety margin
        
        if self.config.center_freq_hz > safe_freq:
            raise ValueError(
                f"Ripple center frequency {self.config.center_freq_hz}Hz exceeds "
                f"safe Nyquist limit {safe_freq:.1f}Hz (fs={self._fs_hz:.1f}Hz). "
                f"Either increase sampling rate (decrease dt_ms) or lower center_freq_hz."
            )
    
    def begin_cycle(self, num_steps: int, energy_scalar: float = 1.0) -> None:
        """
        Begin a new dream cycle and reset counters
        
        Args:
            num_steps: Number of steps in this cycle
            energy_scalar: Energy scalar for optional QREF warping
        """
        # Validate Nyquist again in case config changed
        self._validate_nyquist()
        
        # Reset all counters
        self._coherence_sum = 0.0
        self._coherence_samples = 0
        self._active_steps = 0
        self._events = []
        
        # Store energy scalar for QREF
        self._energy_scalar = energy_scalar
        
        # Schedule first event
        self._schedule_next_event()
        
        logger.debug(f"[Ripple] Beginning cycle with {num_steps} steps, energy={energy_scalar:.2f}")
    
    def update(self, current_time: float) -> None:
        """
        Update ripple substrate state
        
        Args:
            current_time: Current simulation time in seconds
        """
        if current_time < 0:
            raise ValueError("current_time must be non-negative")
        
        self._current_time = current_time
        
        # Check for new ripple events
        if current_time >= self._next_event_time and not self.context.is_active:
            self._trigger_ripple_event()
        
        # Update active ripple
        if self.context.is_active:
            self._update_active_ripple()
    
    def _schedule_next_event(self) -> None:
        """Schedule next ripple event using Poisson process"""
        # Generate inter-event interval from exponential distribution
        # Rate parameter is events per second
        interval = np.random.exponential(1.0 / self.config.event_rate_hz)
        self._next_event_time = self._current_time + interval
        
        logger.debug(f"Next ripple event scheduled at time {self._next_event_time:.3f}")
    
    def _trigger_ripple_event(self) -> None:
        """Trigger a new ripple burst event"""
        self._event_counter += 1
        
        # Generate burst duration
        min_dur, max_dur = self.config.burst_duration_ms
        burst_duration = np.random.uniform(min_dur, max_dur) / 1000.0  # Convert to seconds
        
        # Initialize context
        self.context.is_active = True
        self.context.burst_start_time = self._current_time
        self.context.burst_duration = burst_duration
        self.context.event_id = self._event_counter
        self.context.frequency = self.config.center_freq_hz + np.random.normal(0, 5.0)  # Small freq jitter
        self.context.stdp_multiplier = self.config.stdp_gain
        
        # Reset phase tracking
        self._phase_accumulator = 0.0
        self._last_phase_time = self._current_time
        
        # Initialize amplitude with some variability
        self.context.amplitude = 1.0 + 0.2 * np.random.randn()
        self.context.amplitude = max(0.1, self.context.amplitude)  # Ensure positive
        
        # Log event
        self.stats['total_events'] += 1
        self.stats['stdp_activations'] += 1
        
        logger.info(f"Ripple event {self._event_counter} triggered at time {self._current_time:.3f}, "
                   f"duration {burst_duration*1000:.1f}ms, freq {self.context.frequency:.1f}Hz")
    
    def _update_active_ripple(self) -> None:
        """Update state of active ripple burst"""
        elapsed = self._current_time - self.context.burst_start_time
        
        # Check if burst is complete
        if elapsed >= self.context.burst_duration:
            self._end_ripple_burst()
            return
        
        # Update phase
        self._update_phase()
        
        # Update amplitude (envelope)
        self._update_amplitude(elapsed)
        
        # Update coherence
        self._update_coherence()
        
        # Update coherence tracking counters
        self._coherence_samples += 1
        self._coherence_sum += self.context.coherence
        
        # Update QREF warp if enabled
        if self.config.metric_kappa is not None:
            self._update_qref_warp()
        
        # Update replay alignment if enabled
        if self.config.phase_lock:
            self._update_replay_alignment()
    
    def _update_phase(self) -> None:
        """Update ripple phase based on frequency"""
        dt = self._current_time - self._last_phase_time
        if dt <= 0:
            return
        
        # Accumulate phase based on instantaneous frequency
        phase_increment = 2 * np.pi * self.context.frequency * dt
        self._phase_accumulator += phase_increment
        
        # Wrap phase to [0, 2π)
        self.context.phase = self._phase_accumulator % (2 * np.pi)
        
        # Update phase bin
        self.context.phase_bin = int(self.context.phase / (2 * np.pi) * self.config.phase_bins)
        self.context.phase_bin = min(self.context.phase_bin, self.config.phase_bins - 1)
        
        # Update statistics
        self.stats['phase_distribution'][self.context.phase_bin] += 1
        
        self._last_phase_time = self._current_time
    
    def _update_amplitude(self, elapsed: float) -> None:
        """Update ripple amplitude with realistic envelope"""
        # Use Gaussian envelope centered at burst midpoint
        midpoint = self.context.burst_duration / 2.0
        sigma = self.context.burst_duration / 4.0  # Standard deviation
        
        # Gaussian envelope
        envelope = np.exp(-0.5 * ((elapsed - midpoint) / sigma) ** 2)
        
        # Apply some high-frequency amplitude modulation for realism
        hf_mod = 1.0 + 0.1 * np.sin(2 * np.pi * 40 * elapsed)  # 40Hz modulation
        
        # Base amplitude with envelope and modulation
        base_amp = 1.0 + 0.2 * np.random.randn()  # Some variability
        self.context.amplitude = max(0.1, base_amp * envelope * hf_mod)
    
    def _update_coherence(self) -> None:
        """Update coherence metric based on phase consistency"""
        # Store phase history for coherence calculation
        self._phase_history.append(self.context.phase)
        self._amplitude_history.append(self.context.amplitude)
        
        # Limit history size
        window_samples = int(self.config.coherence_window_ms / 1000.0 * 1000)  # Assume 1kHz sampling
        if len(self._phase_history) > window_samples:
            self._phase_history.pop(0)
            self._amplitude_history.pop(0)
        
        # Calculate coherence if we have enough samples
        if len(self._phase_history) >= 10:
            # Simple coherence: consistency of phase progression
            phases = np.array(self._phase_history)
            amplitudes = np.array(self._amplitude_history)
            
            # Phase coherence: how well phases follow expected progression
            if len(phases) > 1:
                phase_diffs = np.diff(phases)
                # Handle phase wrapping
                phase_diffs = np.angle(np.exp(1j * phase_diffs))
                expected_diff = 2 * np.pi * self.context.frequency / 1000.0  # Assume 1kHz
                
                # Coherence based on consistency of phase differences
                phase_consistency = np.exp(-np.var(phase_diffs - expected_diff))
                
                # Amplitude stability contribution
                amp_stability = np.exp(-np.var(amplitudes) / np.mean(amplitudes)**2)
                
                # Combined coherence
                coherence = 0.7 * phase_consistency + 0.3 * amp_stability
                self.context.coherence = np.clip(coherence, 0.0, 1.0)
            else:
                self.context.coherence = 1.0  # Perfect for single sample
        
        # Update coherence buffer for statistics
        self._coherence_buffer[self._coherence_index] = self.context.coherence
        self._coherence_index = (self._coherence_index + 1) % self.config.coherence_history
        
        # Update average coherence
        valid_samples = min(self.stats['total_events'], self.config.coherence_history)
        if valid_samples > 0:
            self.stats['avg_coherence'] = np.mean(self._coherence_buffer[:valid_samples])
    
    def _update_qref_warp(self) -> None:
        """Update QREF-style metric warp factor"""
        if self.config.metric_kappa is None:
            self.context.qref_warp_factor = 1.0
            return
        
        # QREF warp based on coherence and phase
        # Higher coherence and specific phases get amplified
        coherence_factor = self.context.coherence
        
        # Phase preference (e.g., prefer certain phases for replay)
        phase_preference = 0.5 * (1 + np.cos(self.context.phase - np.pi))  # Prefer phase π
        
        # Combined metric
        metric_value = 0.6 * coherence_factor + 0.4 * phase_preference
        
        # Apply QREF-style exponential warp
        if metric_value > self.config.qref_threshold:
            excess = metric_value - self.config.qref_threshold
            warp = 1.0 + self.config.metric_kappa * excess**2
            self.context.qref_warp_factor = warp
        else:
            self.context.qref_warp_factor = 1.0
    
    def _update_replay_alignment(self) -> None:
        """Update replay alignment factor based on phase"""
        if not self.config.phase_lock:
            self.context.replay_alignment = 0.0
            return
        
        # Alignment strength based on phase
        # Peak alignment at phase 0 (start of cycle)
        alignment_strength = np.cos(self.context.phase)  # [-1, 1]
        
        # Convert to positive alignment factor [0, 1]
        self.context.replay_alignment = 0.5 * (1 + alignment_strength)
    
    def _end_ripple_burst(self) -> None:
        """End current ripple burst and schedule next event"""
        duration = self._current_time - self.context.burst_start_time
        self.stats['total_duration'] += duration
        
        # Store event in buffer
        event_record = {
            'event_id': self.context.event_id,
            'start_time': self.context.burst_start_time,
            'duration': duration,
            'frequency': self.context.frequency,
            'max_amplitude': self.context.amplitude,
            'coherence': self.context.coherence,
            'qref_warp': self.context.qref_warp_factor
        }
        
        self._event_buffer.append(event_record)
        if len(self._event_buffer) > self._max_buffer_size:
            self._event_buffer.pop(0)
        
        logger.info(f"Ripple event {self.context.event_id} ended, duration {duration*1000:.1f}ms, "
                   f"coherence {self.context.coherence:.3f}")
        
        # Reset context
        self.context.reset()
        
        # Clear phase history
        self._phase_history.clear()
        self._amplitude_history.clear()
        
        # Schedule next event
        self._schedule_next_event()
    
    def get_stdp_gain(self) -> float:
        """Get current STDP gain multiplier"""
        return self.context.stdp_multiplier if self.context.is_active else 1.0
    
    def get_phase_info(self) -> Dict[str, float]:
        """Get current phase information"""
        return {
            'phase_rad': self.context.phase,
            'phase_deg': np.degrees(self.context.phase),
            'phase_bin': self.context.phase_bin,
            'phase_normalized': self.context.phase / (2 * np.pi)
        }
    
    def get_coherence_metrics(self) -> Dict[str, float]:
        """Get coherence metrics"""
        return {
            'current_coherence': self.context.coherence,
            'avg_coherence': self.stats['avg_coherence'],
            'coherence_std': np.std(self._coherence_buffer) if self.stats['total_events'] > 1 else 0.0
        }
    
    def get_replay_alignment(self) -> float:
        """Get current replay alignment factor [0, 1]"""
        return self.context.replay_alignment
    
    def get_qref_warp(self) -> float:
        """Get current QREF warp factor"""
        return self.context.qref_warp_factor
    
    def is_ripple_active(self) -> bool:
        """Check if ripple is currently active"""
        return self.context.is_active
    
    def get_current_context(self) -> RippleContext:
        """Get current ripple context"""
        return self.context

    def replay_weight(self) -> float:
        """
        Compute replay weighting factor based on coherence × phase alignment.
        High coherence and early-phase bursts = strong replay weighting.
        """
        coh = self.context.coherence
        align = self.get_replay_alignment()
        return float(coh * align)
    
    def metrics(self, reset: bool = False) -> Dict[str, Any]:
        """Get ripple metrics with optional reset
        
        Args:
            reset: If True, reset counters after returning metrics
            
        Returns:
            Dictionary with ripple metrics
        """
        # Compute coherence
        coh = (self._coherence_sum / max(1, self._coherence_samples)) if self._coherence_samples else 0.0
        
        # Prepare output
        out = {
            "ripple_events": len(getattr(self, '_events', [])),
            "ripple_active_steps": int(self._active_steps),
            "ripple_phase_coherence": float(coh)
        }
        
        # Reset if requested
        if reset:
            self._coherence_sum = 0.0
            self._coherence_samples = 0
            self._active_steps = 0
            if hasattr(self, '_events'):
                self._events.clear()
        
        return out
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = self.stats.copy()
        
        # Add computed statistics
        if stats['total_events'] > 0:
            stats['avg_event_duration'] = stats['total_duration'] / stats['total_events']
            stats['duty_cycle'] = stats['total_duration'] / max(self._current_time, 1e-6)
        else:
            stats['avg_event_duration'] = 0.0
            stats['duty_cycle'] = 0.0
        
        # Add phase distribution statistics
        if np.sum(stats['phase_distribution']) > 0:
            stats['phase_entropy'] = self._calculate_phase_entropy()
        else:
            stats['phase_entropy'] = 0.0
        
        # Add recent events
        stats['recent_events'] = self._event_buffer[-10:]  # Last 10 events
        
        return stats
    
    def _calculate_phase_entropy(self) -> float:
        """Calculate entropy of phase distribution"""
        phase_dist = self.stats['phase_distribution']
        total = np.sum(phase_dist)
        
        if total == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = phase_dist / total
        
        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def reset_statistics(self) -> None:
        """Reset all statistics"""
        self.stats = {
            'total_events': 0,
            'total_duration': 0.0,
            'avg_coherence': 0.0,
            'phase_distribution': np.zeros(self.config.phase_bins),
            'stdp_activations': 0
        }
        
        self._coherence_buffer.fill(0.0)
        self._coherence_index = 0
        self._event_buffer.clear()
        
        # Reset coherence tracking counters
        self._coherence_samples = 0
        self._coherence_sum = 0.0
        
        logger.info("RippleSubstrate statistics reset")
    
    def force_ripple_event(self, duration_ms: Optional[float] = None) -> None:
        """
        Force trigger a ripple event (for testing/debugging)
        
        Args:
            duration_ms: Optional duration override in milliseconds
        """
        if self.context.is_active:
            logger.warning("Cannot force ripple event while one is already active")
            return
        
        # Override next event time to trigger immediately
        self._next_event_time = self._current_time
        
        # Override duration if specified
        if duration_ms is not None:
            original_duration = self.config.burst_duration_ms
            self.config.burst_duration_ms = (duration_ms, duration_ms)
            
            # Trigger event
            self.update(self._current_time)
            
            # Restore original duration
            self.config.burst_duration_ms = original_duration
        else:
            self.update(self._current_time)
        
        logger.info(f"Forced ripple event triggered at time {self._current_time:.3f}")
    
    def __str__(self) -> str:
        """String representation"""
        status = "ACTIVE" if self.context.is_active else "INACTIVE"
        return (f"RippleSubstrate(rate={self.config.event_rate_hz:.1f}Hz, "
               f"freq={self.config.center_freq_hz:.0f}Hz, status={status}, "
               f"events={self.stats['total_events']})")
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"RippleSubstrate(config={self.config}, "
               f"context={self.context.to_dict()}, "
               f"stats={self.get_statistics()})")


# Utility functions for common use cases

def create_default_ripple_substrate() -> RippleSubstrate:
    """Create ripple substrate with default physiological parameters"""
    config = RippleConfig()
    return RippleSubstrate(config)


def create_enhanced_ripple_substrate(
    event_rate: float = 1.2,
    stdp_gain: float = 4.0,
    metric_kappa: float = 2.0
) -> RippleSubstrate:
    """Create ripple substrate with enhanced parameters for learning"""
    config = RippleConfig(
        event_rate_hz=event_rate,
        stdp_gain=stdp_gain,
        metric_kappa=metric_kappa,
        qref_threshold=0.6
    )
    return RippleSubstrate(config)


def create_test_ripple_substrate(fast_rate: bool = True) -> RippleSubstrate:
    """Create ripple substrate optimized for testing"""
    config = RippleConfig(
        event_rate_hz=4.0 if fast_rate else 2.0,  # Within valid range [0.1, 5.0]
        burst_duration_ms=(20.0, 50.0),  # Shorter bursts for testing
        coherence_history=20,
        buffer_size=100
    )
    return RippleSubstrate(config)


if __name__ == "__main__":
    # Example usage and validation
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
    
    # Create substrate
    substrate = create_default_ripple_substrate()
    
    # Simulate for 10 seconds
    dt = 0.001  # 1ms timesteps
    times = np.arange(0, 10.0, dt)
    
    phases = []
    amplitudes = []
    coherences = []
    stdp_gains = []
    active_states = []
    
    logger.info("Running ripple substrate simulation...")
    
    for t in times:
        substrate.update(t)
        
        phases.append(substrate.context.phase)
        amplitudes.append(substrate.context.amplitude)
        coherences.append(substrate.context.coherence)
        stdp_gains.append(substrate.get_stdp_gain())
        active_states.append(1.0 if substrate.is_ripple_active() else 0.0)
    
    # Print final statistics
    stats = substrate.get_statistics()
    logger.info("Simulation completed:")
    logger.info(f"Total ripple events: {stats['total_events']}")
    logger.info(f"Average coherence: {stats['avg_coherence']:.3f}")
    logger.info(f"Average event duration: {stats['avg_event_duration']*1000:.1f}ms")
    logger.info(f"Duty cycle: {stats['duty_cycle']*100:.1f}%")
    logger.info(f"Phase entropy: {stats['phase_entropy']:.3f}")
    
    # Create visualization if matplotlib available
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Ripple activity and STDP gain
        axes[0,0].plot(times, active_states, 'r-', alpha=0.7, label='Ripple Active')
        axes[0,0].plot(times, np.array(stdp_gains)/max(stdp_gains), 'b-', alpha=0.7, label='STDP Gain (norm)')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Activity / Gain')
        axes[0,0].set_title('Ripple Activity and STDP Modulation')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Phase evolution
        axes[0,1].plot(times, phases, 'g-', alpha=0.8)
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Phase (rad)')
        axes[0,1].set_title('Ripple Phase Evolution')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Amplitude and coherence
        axes[1,0].plot(times, amplitudes, 'orange', alpha=0.7, label='Amplitude')
        axes[1,0].plot(times, coherences, 'purple', alpha=0.7, label='Coherence')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Amplitude / Coherence')
        axes[1,0].set_title('Amplitude and Coherence')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Phase distribution
        axes[1,1].bar(range(len(stats['phase_distribution'])), 
                     stats['phase_distribution'], 
                     alpha=0.7, color='cyan')
        axes[1,1].set_xlabel('Phase Bin')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Phase Distribution')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ripple_substrate_validation.png', dpi=150, bbox_inches='tight')
        logger.info("Validation plot saved as 'ripple_substrate_validation.png'")
    else:
        logger.info("Matplotlib not available, skipping visualization")
    
    logger.info("Ripple substrate validation completed successfully!")
    logger.debug(f"Final substrate state: {substrate}")