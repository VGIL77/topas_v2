#!/usr/bin/env python3
"""
Unified Training Entrypoint for TOPAS ARC Prize

This script sequentially runs Phases 0–10:
 P0 Grammar → P1 Policy → P2 Meta → P3 Self-Critique → P4 MCTS
 → P5 Dream Replay → P6 Neuro-Priors + Template Curriculum
 → P7 RelMem Integration → P8 SGI Opt → P9 Ensemble → P10 Production

This is the **only canonical training entrypoint**.
"""

import torch
import os
import json
import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all phases
from trainers.phases import (
    phase0_world_grammar,
    phase1_policy_distill,
    phase2_meta_learning,
    phase3_self_critique,
    phase4_mcts_alpha,
    phase5_dream_scaled,
    phase6_neuro_priors,
    phase7_relmem,
    phase8_sgi_optimizer,
    phase9_ensemble_solver,
    phase10_production,
)
from trainers.train_logger import TrainLogger
from trainers.arc_dataset_loader import ARCDataset

def create_sample_function(dataset):
    """Create a sample function for phases that need it"""
    def sample_fn():
        idx = torch.randint(0, len(dataset), (1,)).item()
        data = dataset[idx]
        demos = data.get("demos", [])
        test = data.get("test", {})
        return demos, test
    return sample_fn

def run_full_pipeline(config):
    """Run the full 0–10 training pipeline with shared state dict."""
    
    print("🚀 Starting TOPAS ARC Prize Training Pipeline (Phases 0-10)")
    print("="*70)
    
    # Get device
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"[Config] Strict mode: {config.get('strict', False)}")
    
    # Initialize shared state
    state = {
        "logger": TrainLogger(config.get("log_dir", "./logs/arc_prize")),
        "global_step": 0
    }
    
    # === Dataset Selection ===
    from trainers.arc_dataset_loader import SyntheticGrammarDataset, ARCDataset
    
    # HRM integration: Use ARC data throughout when enabled
    if config.get("hrm_integration_enabled", False):
        dataset = ARCDataset(
            challenge_file=config.get("train_challenges", "/mnt/d/Bitterbot/research/topas_v2/ARC-AGI/data/training"),
            solution_file=config.get("train_solutions", "/mnt/d/Bitterbot/research/topas_v2/ARC-AGI/data/training"),
            device=str(device),
            max_grid_size=config.get("max_grid_size", 30)
        )
        print(f"[Dataset] HRM Integration Mode: Using ARCDataset throughout, length={len(dataset)}")
    elif config.get("phase0_mode", True):
        # Always synthetic for Phase 0 (legacy mode)
        dataset = SyntheticGrammarDataset(
            num_samples=config.get("dataset_size", 2000),
            max_grid_size=config.get("max_grid_size", 30),
            device=str(device)
        )
        print(f"[Dataset] Phase 0: Using SyntheticGrammarDataset, length={len(dataset)}")
    else:
        dataset = ARCDataset(
            challenge_file=config.get("train_challenges", "ARC/arc-agi_training_challenges.json"),
            solution_file=config.get("train_solutions", "ARC/arc-agi_training_solutions.json"),
            device=str(device),
            max_grid_size=config.get("max_grid_size", 30)
        )
        print(f"[Dataset] Phase >=1: Using ARCDataset, length={len(dataset)}")

    state["dataset"] = dataset
    state["sample_fn"] = create_sample_function(dataset)
    
    # Initialize task list for scheduler phases
    train_tasks = []
    for i in range(min(100, len(dataset))):  # Limit for efficiency
        # Dataset returns (demos, test_inputs, test_outputs, task_id)
        demos, test_inputs, test_outputs, task_id = dataset[i]
        train_tasks.append({
            "id": task_id,
            "demos": demos,
            "test": {"input": test_inputs[0] if test_inputs else None,
                    "output": test_outputs[0] if test_outputs else None}
        })
    state["train_tasks"] = train_tasks
    
    # Run phases sequentially
    phases = [
        ("Phase 0: World Grammar", phase0_world_grammar, "phase0_world_grammar"),
        ("Phase 1: Policy Distill", phase1_policy_distill, "phase1_policy_distill"),
        ("Phase 2: Meta Learning", phase2_meta_learning, "phase2_meta_learning"),
        ("Phase 3: Self Critique", phase3_self_critique, "phase3_self_critique"),
        ("Phase 4: MCTS Alpha", phase4_mcts_alpha, "phase4_mcts_alpha"),
        ("Phase 5: Dream Scaled", phase5_dream_scaled, "phase5_dream_scaled"),
        ("Phase 6: Neuro Priors + Templates", phase6_neuro_priors, "phase6_neuro_priors"),
        ("Phase 7: RelMem", phase7_relmem, "phase7_relmem"),
        ("Phase 8: SGI Optimizer", phase8_sgi_optimizer, "phase8_sgi_optimizer"),
        ("Phase 9: Ensemble Solver", phase9_ensemble_solver, "phase9_ensemble_solver"),
        ("Phase 10: Production", phase10_production, "phase10_production"),
    ]
    
    for phase_name, phase_module, config_key in phases:
        print(f"\n{'='*20} {phase_name} {'='*20}")
        
        # Switch to ARC dataset for Phase 1 onward (unless HRM integration already using ARC)
        if config_key == "phase1_policy_distill" and not config.get("hrm_integration_enabled", False):
            # Switch to ARC dataset for all later phases
            dataset = ARCDataset(
                challenge_file=config.get("train_challenges"),
                solution_file=config.get("train_solutions"),
                device=str(device),
                max_grid_size=config.get("max_grid_size", 30)
            )
            print(f"[Dataset] Switching to ARCDataset for {phase_name}, length={len(dataset)}")
            state["dataset"] = dataset
            state["sample_fn"] = create_sample_function(dataset)
        
        phase_config = config.get(config_key, {})
        # Add global config overrides
        phase_config.update({
            "device": config.get("device", "cuda"),
            "log_dir": config.get("log_dir", "./logs/arc_prize")
        })
        
        try:
            state = phase_module.run(phase_config, state)
            print(f"✅ {phase_name} completed successfully")
            
            # HRM Integration: Curriculum progression and specialized checkpoints
            if config.get("hrm_integration_enabled", False):
                # Save HRM-specific checkpoints
                if config.get("save_hrm_checkpoints", True):
                    hrm_checkpoint = {
                        "phase": config_key,
                        "model_state_dict": state["model"].state_dict() if "model" in state else None,
                        "puzzle_embedder": state.get("puzzle_embedder"),
                        "curriculum_level": state.get("curriculum_level", 0),
                        "hrm_metrics": {
                            "arc_tasks_processed": state.get("arc_tasks_processed", 0),
                            "hrm_meta_learning_enabled": state.get("hrm_meta_learning_enabled", False),
                            "hrm_embedding_enabled": state.get("hrm_embedding_enabled", False)
                        }
                    }
                    checkpoint_path = os.path.join(
                        config.get("checkpoint_dir", "./checkpoints"), 
                        f"hrm_{config_key}_checkpoint.pt"
                    )
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(hrm_checkpoint, checkpoint_path)
                    print(f"💾 HRM checkpoint saved: {checkpoint_path}")
                
                # Curriculum progression for Phase 0 completion
                if config_key == "phase0_world_grammar" and config.get("enable_curriculum_learning", True):
                    curriculum_config = config.get("curriculum_config", {})
                    current_level = state.get("curriculum_level", 0)
                    progression_epochs = curriculum_config.get("progression_epochs", [5, 12, 25, 45])
                    
                    # Check if we should progress to next curriculum level
                    global_step = state.get("global_step", 0)
                    epochs_completed = state.get("phase0_epochs", 0)
                    
                    if curriculum_config.get("enable_progression", True) and epochs_completed > 0:
                        next_level = min(current_level + 1, curriculum_config.get("final_level", 4))
                        if next_level > current_level and epochs_completed >= progression_epochs[min(current_level, len(progression_epochs)-1)]:
                            print(f"📈 Curriculum progression: Level {current_level} → {next_level}")
                            state["curriculum_level"] = next_level
            
            # Log phase completion + check acceptance gates (migrated from master_trainer.py)
            metrics = {
                "phase": config_key,
                "model_parameters": sum(p.numel() for p in state["model"].parameters()) if "model" in state else 0,
                "global_step": state.get("global_step", 0)
            }
            
            # Phase-specific acceptance gates
            if config_key == "phase6_neuro_priors":
                # Phase 6 gates: calibrated confidence + discrete grids
                calibrated = state.get("overconfidence", 0.0) < 0.1
                discrete = state.get("ebr_temperature", 1.0) < 0.5
                metrics.update({"calibrated_confidence": calibrated, "discrete_grids": discrete})
                if calibrated and discrete:
                    print("🎯 Phase 6 gating criteria met: calibrated confidence + discrete grids")
                else:
                    print("⚠️ Phase 6 gating criteria not met")
                    
            elif config_key == "phase7_relmem":
                # Phase 7 gates: template hit rate ≥ 40%
                hit_rate = state.get("template_hit_rate", 0.0)
                metrics.update({"template_hit_rate": hit_rate})
                if hit_rate >= 0.40:
                    print("🎯 Phase 7 gating criteria met: template hit rate ≥ 40%")
                else:
                    print(f"⚠️ Phase 7 gating criteria not met: hit rate {hit_rate:.1%} < 40%")
                    
            elif config_key == "phase8_sgi_optimizer":
                # Phase 8 gates: balanced curriculum sampling
                balanced = state.get("balanced_sampling", True)
                metrics.update({"balanced_sampling": balanced})
                if balanced:
                    print("🎯 Phase 8 gating criteria met: balanced curriculum sampling")
                else:
                    print("⚠️ Phase 8 gating criteria not met: sampling imbalanced")
                    
            elif config_key == "phase9_ensemble_solver":
                # Phase 9 gates: stable training + no gradient explosion
                stable = state.get("stable_training", True)
                no_explosion = state.get("no_gradient_explosion", True)
                metrics.update({"stable_training": stable, "no_gradient_explosion": no_explosion})
                if stable and no_explosion:
                    print("🎯 Phase 9 gating criteria met: optimizer stable")
                else:
                    print("⚠️ Phase 9 gating criteria not met: training instability")
                    
            elif config_key == "phase10_production":
                # Phase 10 gates: ensemble improvement ≥ 5%
                improvement = state.get("ensemble_improvement", 0.0)
                metrics.update({"ensemble_improvement": improvement})
                if improvement >= 0.05:
                    print("🎯 Phase 10 gating criteria met: ensemble improved ≥ 5% over single model")
                else:
                    print(f"⚠️ Phase 10 gating criteria not met: improvement {improvement:.1%} < 5%")

            state["logger"].log_milestone(f"{phase_name} Complete", metrics)
            
        except Exception as e:
            print(f"❌ {phase_name} failed: {e}")
            state["logger"].log_error(f"{phase_name} failed", {"error": str(e)})
            if config.get("strict", True):
                raise   # 🚨 fail fast instead of skipping
            continue
    
    # Final statistics
    print(f"\n{'='*70}")
    print("🎯 TOPAS ARC Prize Training Pipeline Complete!")
    
    if "model" in state:
        total_params = sum(p.numel() for p in state["model"].parameters())
        print(f"📊 Final Model: {total_params:,} parameters")
    
    if "logger" in state:
        state["logger"].print_summary()
        state["logger"].close()
    
    # Save final state
    final_checkpoint = {
        "model_state_dict": state["model"].state_dict() if "model" in state else None,
        "training_complete": True,
        "phases_completed": [key for _, _, key in phases],
        "final_metrics": state.get("final_metrics", {}),
        "hrm_integration_enabled": config.get("hrm_integration_enabled", False)
    }
    
    # Add HRM-specific final state
    if config.get("hrm_integration_enabled", False):
        final_checkpoint.update({
            "puzzle_embedder_state": state.get("puzzle_embedder").state_dict() if state.get("puzzle_embedder") else None,
            "final_curriculum_level": state.get("curriculum_level", 0),
            "total_arc_tasks_processed": state.get("arc_tasks_processed", 0),
            "hrm_final_metrics": {
                "meta_learning_enabled": state.get("hrm_meta_learning_enabled", False),
                "embedding_enabled": state.get("hrm_embedding_enabled", False),
                "q_learning_integration": True,
                "fast_adaptation_enabled": True
            }
        })
    
    checkpoint_path = os.path.join(config.get("checkpoint_dir", "./checkpoints"), "topas_arc_prize_final.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(final_checkpoint, checkpoint_path)
    print(f"💾 Final checkpoint saved: {checkpoint_path}")
    
    return state


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="TOPAS ARC Prize Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/full_run.json",
                       help="Path to configuration JSON file")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Training device (cuda/cpu)")
    parser.add_argument("--log-dir", type=str, default="./logs/arc_prize",
                       help="Directory for training logs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Directory for model checkpoints")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        print(f"⚠️ Config file {args.config} not found, using defaults")
        config = {}
    
    # Override with command line arguments
    config.update({
        "device": args.device,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir
    })
    
    print(f"📋 Configuration:")
    print(f"  Device: {config['device']}")
    print(f"  Log Dir: {config['log_dir']}")
    print(f"  Checkpoint Dir: {config['checkpoint_dir']}")
    print(f"  Config File: {args.config}")
    
    # Run the full pipeline
    final_state = run_full_pipeline(config)
    
    print("\n🏆 Training pipeline completed successfully!")
    return final_state


if __name__ == "__main__":
    main()