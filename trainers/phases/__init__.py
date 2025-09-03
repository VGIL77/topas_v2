"""
TOPAS Phase Training Modules
"""

# Import all phase modules for easy access
import trainers.phases.phase0_world_grammar as phase0_world_grammar
import trainers.phases.phase1_policy_distill as phase1_policy_distill
import trainers.phases.phase2_meta_learning as phase2_meta_learning
import trainers.phases.phase3_self_critique as phase3_self_critique
import trainers.phases.phase4_mcts_alpha as phase4_mcts_alpha
import trainers.phases.phase5_dream_scaled as phase5_dream_scaled
import trainers.phases.phase6_neuro_priors as phase6_neuro_priors
import trainers.phases.phase7_relmem as phase7_relmem
import trainers.phases.phase8_sgi_optimizer as phase8_sgi_optimizer
import trainers.phases.phase9_ensemble_solver as phase9_ensemble_solver
import trainers.phases.phase10_production as phase10_production

__all__ = [
    'phase0_world_grammar',
    'phase1_policy_distill', 
    'phase2_meta_learning',
    'phase3_self_critique',
    'phase4_mcts_alpha',
    'phase5_dream_scaled',
    'phase6_neuro_priors',
    'phase7_relmem',
    'phase8_sgi_optimizer',
    'phase9_ensemble_solver',
    'phase10_production'
]