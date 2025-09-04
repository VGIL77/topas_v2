"""
Phase 5 – Dream Engine Scaled Integration
Attach DreamEngine mid-epoch scaled replay + RelMem gating.
Full neuroscientific dream system with GCCRF curiosity, NMDA gating, 
emergent themes, ripple substrate, and wormhole mining.
"""

import torch
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.train_logger import TrainLogger
from gccrf_curiosity import GCCRFCuriosity
from nmda_dreaming import NMDAGatedDreaming
from emergent_theme_synthesis import EmergentThemeSynthesis
from ripple_substrate import create_default_ripple_substrate
from wormhole_offline import WormholeTemplateMiner

def run(config, state):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = state["model"]
    dataset = state.get("dataset")
    logger = state.get("logger") or TrainLogger()
    
    # === Initialize Dream subsystems if missing ===
    if "dream" not in state:
        from dream_engine import DreamEngine
        state["dream"] = DreamEngine(device=device)

    if "gccrf" not in state:
        state["gccrf"] = GCCRFCuriosity(state_dim=model.config.slot_dim).to(device)

    if "nmda" not in state:
        state["nmda"] = NMDAGatedDreaming(state_dim=model.config.slot_dim,
                                          action_dim=model.config.dsl_vocab_size,
                                          device=device).to(device)

    if "themes" not in state:
        state["themes"] = EmergentThemeSynthesis(embedding_dim=model.config.slot_dim)

    if "ripple" not in state:
        state["ripple"] = create_default_ripple_substrate()

    if "wormhole" not in state:
        state["wormhole"] = WormholeTemplateMiner()

    dream, gccrf, nmda, themes, ripple, wormhole = (
        state["dream"], state["gccrf"], state["nmda"], state["themes"], state["ripple"], state["wormhole"]
    )

    epochs = config.get("epochs", 5)
    steps_per_epoch = config.get("steps_per_epoch", 400)
    
    def default_sample_fn(dataset, device):
        """Sample from dataset"""
        if dataset is None:
            return [], {}
        idx = torch.randint(0, len(dataset), (1,)).item()
        data = dataset[idx]
        demos = data.get("demos", [])
        test = data.get("test", {})
        return demos, test
    
    def safe_model_forward(model, demos, test, device, training_mode=True):
        """Safe model forward with error handling"""
        try:
            grid, logits, size, extras = model(demos, test, training_mode=training_mode)
            return grid, logits, extras
        except Exception as e:
            print(f"[Dream] Model forward failed: {e}")
            return None, None, {}

    # === Main Dream Cycle ===
    global_step = 0
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            demos, test = default_sample_fn(dataset, device)
            grid, logits, extras = safe_model_forward(model, demos, test, device, training_mode=True)

            latents = extras.get("latent") if extras else None
            if latents is None:
                continue

            # Curiosity-driven valence/arousal
            R_i, curiosity_info = gccrf.compute_reward(latents.flatten(1))
            valence = curiosity_info["prediction_error"].item()
            arousal = curiosity_info["novelty"].item()

            # Run ripple update
            ripple.update(step * config.get("dt", 0.01))

            # Dream replay
            replay_stats = dream.cycle_offline(latents, demos, valence=valence, arousal=arousal)

            # NMDA consolidation
            nmda_loss = nmda.dream_consolidation(valence, arousal, ripple_ctx=ripple.get_current_context())

            # Theme synthesis
            if "output" in test:
                motifs = themes.process_dream_themes(latents.unsqueeze(0), test["output"].view(-1))
                themes.synthesize_emergent_themes(motifs)

            # Wormhole mining every N steps
            if step % config.get("wormhole_interval", 50) == 0:
                wormhole.mined = wormhole.mine_from_programs(demos, top_k=5)

            # Logging
            if step % config.get("log_interval", 25) == 0:
                logger.log_batch(global_step, {}, dream_info={
                    "dream_valence": valence,
                    "dream_arousal": arousal,
                    "nmda_loss": float(nmda_loss) if isinstance(nmda_loss, torch.Tensor) else nmda_loss,
                    "num_themes": len(themes.themes) if hasattr(themes, 'themes') else 0,
                    "curiosity_pred_error": curiosity_info["prediction_error"].item(),
                    "curiosity_novelty": curiosity_info["novelty"].item(),
                    "ripple_phase": ripple.get_current_context() if hasattr(ripple, 'get_current_context') else 0.0
                })
            
            global_step += 1
        
        print(f"[Phase 5] Epoch {epoch+1}/{epochs} complete - Dream cycles: {global_step}")

    state.update({
        "dream": dream, 
        "gccrf": gccrf,
        "nmda": nmda,
        "themes": themes,
        "ripple": ripple,
        "wormhole": wormhole,
        "logger": logger
    })
    
    print(f"Phase 5: Full neuroscientific dream system activated - {global_step} dream cycles complete.")
    return state