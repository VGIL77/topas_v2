#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math
from dataclasses import dataclass

@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool = False
    valence: float = 0.7
    arousal: float = 0.5
    timestamp: int = 0
    phase: float = 0.0

class NMDAGatedDreaming(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, g_max: float = 1.0, mg_conc: float = 1.0, valence_thresh: float = 0.5, learning_rate: float = 0.001, gamma: float = 0.99, tau: float = 0.005, buffer_size: int = 10000, batch_size: int = 64, device: str = "cpu", phase_lock: bool = True):
        super().__init__()
        self.device = torch.device(device)
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.g_max = g_max
        self.mg_conc = mg_conc
        self.valence_thresh = valence_thresh
        self.phase_lock = phase_lock

    def compute_conductance(self, voltage: float, synaptic_act: float = 1.0) -> float:
        B = 1 / (1 + self.mg_conc * math.exp(-0.062 * voltage) / 3.57)
        return self.g_max * B * synaptic_act

    def compute_gate(self, valence: float, arousal: float) -> float:
        voltage = arousal * 40 - 20
        conductance = self.compute_conductance(voltage)
        gate = 1 / (1 + math.exp(-10 * (valence - self.valence_thresh)))
        return gate * conductance
    
    def wrap_dist(self, phi: float, peak: float = 0.0) -> float:
        """
        Compute circular distance between two phase values.
        
        Args:
            phi: Phase value to compare
            peak: Peak phase value (default 0.0)
            
        Returns:
            float: Wrap-around aware distance on circle
        """
        d = abs((phi - peak + math.pi) % (2*math.pi) - math.pi)
        return d

    def store_dream_memory(self, state, action, reward, next_state, phase=0.0):
        state = state.detach().to(self.device)
        next_state = next_state.detach().to(self.device)
        self.buffer.append(Experience(state, action, reward, next_state, phase=phase))

    def dream_consolidation(self, valence: float, arousal: float, ripple_ctx=None):
        """
        NMDA-gated dream consolidation with ripple-aware amplification.
        
        Args:
            valence: Emotional valence (0-1)
            arousal: Arousal level (0-1) 
            ripple_ctx: Optional ripple context with gain multiplier
            
        Returns:
            float: Loss value after consolidation
        """
        if len(self.buffer) < self.batch_size: 
            return 0.0
            
        # Apply phase-locked replay ordering if ripple context provided
        batch = self.sample_batch(ripple_ctx) if ripple_ctx else random.sample(self.buffer, self.batch_size)
        
        states, actions, rewards, next_states = zip(*[(e.state, e.action, e.reward, e.next_state) for e in batch])
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)

        # Compute NMDA gate
        gate = self.compute_gate(valence, arousal)
        if gate < 0.1: 
            return 0.0

        # Forward pass
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q
        loss = F.mse_loss(q_values, target_q)
        
        # Apply ripple-aware gain with biological consistency
        # Both NMDA gate and ripple must be active for maximum effect
        if ripple_ctx and hasattr(ripple_ctx, 'gain') and ripple_ctx.gain > 1.0:
            # Combine ripple gain with NMDA gating
            # mult = 1.0 + (ripple_gain - 1.0) * nmda_gate
            # This ensures ripple only amplifies when NMDA allows plasticity
            ripple_mult = 1.0 + (ripple_ctx.gain - 1.0) * gate
            loss = loss * gate * ripple_mult
        else:
            # Standard NMDA gating without ripple amplification
            loss = loss * gate

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.item()
        
    def sample_batch(self, ripple_ctx=None):
        """
        Sample a batch of experiences with optional phase-locked ordering.
        
        During ripple bursts, order batch by phase distance to peak for
        biologically plausible replay sequences.
        
        Args:
            ripple_ctx: Optional ripple context with active flag and phase_peak
            
        Returns:
            list: Batch of experiences, potentially ordered by phase
        """
        if len(self.buffer) < self.batch_size:
            return list(self.buffer)
            
        # Random sampling by default
        batch = random.sample(self.buffer, self.batch_size)
        
        # Apply phase-locked ordering during active ripples
        if (ripple_ctx and 
            hasattr(ripple_ctx, 'active') and ripple_ctx.active and 
            self.phase_lock and
            hasattr(ripple_ctx, 'phase_peak')):
            
            # Quick check if phases are diverse enough to benefit from sorting
            phases = [getattr(e, 'phase', 0.0) for e in batch]
            phase_std = torch.std(torch.tensor(phases))
            
            # Only sort if there's meaningful phase diversity (> 0.5 radians std)
            if phase_std > 0.5:
                phase_peak = getattr(ripple_ctx, 'phase_peak', 0.0)
                batch.sort(key=lambda e: self.wrap_dist(getattr(e, 'phase', 0.0), phase_peak))
                
                # Log only occasionally to avoid spam
                if random.random() < 0.1:  # Log 10% of the time
                    logger.debug(f"Phase-locked replay: batch reordered (N={len(batch)})")
            
        return batch
    
    def to(self, device):
        """Move the module and buffer contents to the specified device."""
        # Call parent class to() to move nn.Module parameters
        super().to(device)
        self.device = torch.device(device)
        
        # Move buffer experiences to the new device
        new_buffer = deque(maxlen=self.buffer.maxlen)
        for exp in self.buffer:
            # Create new Experience with tensors on the new device
            new_exp = Experience(
                state=exp.state.to(device) if torch.is_tensor(exp.state) else exp.state,
                action=exp.action,
                reward=exp.reward,
                next_state=exp.next_state.to(device) if torch.is_tensor(exp.next_state) else exp.next_state,
                done=exp.done,
                valence=exp.valence,
                arousal=exp.arousal,
                timestamp=exp.timestamp,
                phase=exp.phase
            )
            new_buffer.append(new_exp)
        self.buffer = new_buffer
        
        return self