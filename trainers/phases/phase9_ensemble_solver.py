"""
Phase 8 – Ensemble Solver
Truth-conditioned solving with self-consistency + mixture of experts.
"""

import torch
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.train_logger import TrainLogger
from trainers.ensemble_solver import EnsembleSolver

def run(config, state):
    device = torch.device(config.get("device", "cuda"))
    model = state["model"]
    logger = state.get("logger") or TrainLogger()

    # Initialize ensemble solver if not already
    ensemble = state.get("ensemble_solver") or EnsembleSolver(model, config)
    state["ensemble_solver"] = ensemble

    tasks = state.get("eval_tasks", [])
    steps = config.get("steps", 50)

    for step, task in enumerate(tasks[:steps]):
        demos, test = state["sample_fn"]()
        demos = [{k: v.to(device) for k, v in demo.items()} for demo in demos]
        test = {k: v.to(device) for k, v in test.items()}

        # Forward to extract real features (consistent with Phase 6)
        try:
            _, _, _, extras = model(demos, test, training_mode=False)
            if "latent" in extras and extras["latent"] is not None:
                task_features = extras["latent"].flatten().unsqueeze(0)
            elif "slots_rel" in extras:
                task_features = extras["slots_rel"].mean(dim=1)
            else:
                task_features = torch.randn(1, 128, device=device)
        except Exception as e:
            print(f"[Phase 9] Feature extraction failed: {e}")
            task_features = torch.randn(1, 128, device=device)

        candidate = ensemble.ensemble_solve(task, task_features)
        if candidate:
            logger.log_batch(step, {
                "total": 0.0,
                "ensemble_confidence": getattr(candidate, "confidence", 0.0),
                "verification": getattr(candidate, "verification_score", 0.0)
            })

    # Save ensemble stats into state
    state.update({"ensemble_solver": ensemble, "logger": logger})
    return state