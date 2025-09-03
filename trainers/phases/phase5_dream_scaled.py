"""
Phase 5 – Dream Engine Scaled Integration
Attach DreamEngine mid-epoch scaled replay + RelMem gating.
"""

import torch
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.train_logger import TrainLogger

def run(config, state):
    device = torch.device(config.get("device", "cuda"))

    model = state["model"]
    dream = state.get("dream")
    logger = state.get("logger") or TrainLogger()

    epochs = config.get("epochs", 5)
    steps_per_epoch = config.get("steps_per_epoch", 200)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # Normal training loop is assumed ongoing upstream;
            # here we inject scaled-back DreamEngine replay.
            if step % config.get("dream_interval", 50) == 0 and dream is not None:
                valence, arousal = 0.6, 0.4  # placeholder from scheduler or state
                dream.cycle_offline(
                    tokens=torch.randn(1, 16, 128).to(device),
                    valence=valence,
                    arousal=arousal
                )
                logger.log_batch(step, {"total": 0.0}, {}, {
                    "ripple": True,
                    "valence": valence,
                    "arousal": arousal
                })

            # (In practice, wrap around the real optimizer loop as in Phase 0)

    state.update({"dream": dream, "logger": logger})
    return state