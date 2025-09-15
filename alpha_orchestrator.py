#!/usr/bin/env python3
"""
Alpha Orchestrator - Neural-Guided Search 2.0 Main Training System

This is the production entry point that orchestrates complete training "seasons"
with all phases integrated into a coherent Neural-Guided Search 2.0 system.

Phases executed in sequence:
- Phase 2: Meta Learning (MAML/Reptile + HRM fast adaptation)
- Phase 3: Self Critique (STaR methodology + consistency enforcement)
- Phase 4: MCTS Alpha (AlphaZero-style tree search with neural guidance)
- Phase 6: Neuro Priors (neural network guided DSL operation priors)
- Phase 7: RelMem (relational memory integration + biological learning)

Features:
- Shared state management across phases
- Checkpoint save/load with recovery
- Comprehensive logging and monitoring
- Error handling and phase recovery
- Both full season and individual phase execution
- Clean integration with existing phase files (no modifications needed)

Usage:
    python alpha_orchestrator.py --config configs/neural_guided_search_v2.json
    python alpha_orchestrator.py --config configs/ngs2.json --phases 2,3,4
    python alpha_orchestrator.py --resume checkpoints/season_checkpoint.pt
"""

import torch
import torch.nn as nn
import json
import os
import sys
import argparse
import traceback
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque
import logging
import shutil
import psutil

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Phase imports
from trainers.phases import (
    phase2_meta_learning,
    phase3_self_critique,
    phase4_mcts_alpha,
    phase6_neuro_priors,
    phase7_relmem
)

# Core imports
from trainers.train_logger import TrainLogger
from trainers.arc_dataset_loader import ARCDataset
from models.topas_arc_60M import TopasARC60M, ModelConfig
from relational_memory_neuro import RelationalMemoryNeuro

# Optional HRM integration
try:
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    HRM_AVAILABLE = True
except ImportError:
    HRM_AVAILABLE = False


class SeasonCheckpoint:
    """Manages comprehensive season checkpoints."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_season_checkpoint(self, orchestrator: 'AlphaOrchestrator',
                             phase_name: str, phase_state: Dict[str, Any]) -> str:
        """Save comprehensive season checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"season_{phase_name}_{timestamp}.pt"

        # Extract model state safely
        model_state = None
        if "model" in phase_state and phase_state["model"] is not None:
            try:
                model_state = phase_state["model"].state_dict()
            except Exception as e:
                orchestrator.logger.warning(f"Failed to extract model state: {e}")

        # Extract optimizer state safely
        optimizer_state = None
        if "optimizer" in phase_state and phase_state["optimizer"] is not None:
            try:
                optimizer_state = phase_state["optimizer"].state_dict()
            except Exception as e:
                orchestrator.logger.warning(f"Failed to extract optimizer state: {e}")

        checkpoint_data = {
            # Core orchestrator state
            "orchestrator_version": "1.0.0",
            "timestamp": timestamp,
            "current_phase": phase_name,
            "completed_phases": list(orchestrator.completed_phases),
            "season_config": orchestrator.config,
            "global_step": phase_state.get("global_step", 0),

            # Model and training state
            "model_state_dict": model_state,
            "model_config": orchestrator.model_config.__dict__ if orchestrator.model_config else None,
            "optimizer_state_dict": optimizer_state,

            # Phase-specific state
            "phase_state": self._serialize_phase_state(phase_state),

            # Training metrics
            "training_metrics": orchestrator.season_metrics,
            "phase_metrics": orchestrator.phase_metrics,

            # System info
            "system_info": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(orchestrator.device),
                "memory_usage": psutil.virtual_memory().percent if psutil else None,
                "python_version": sys.version
            }
        }

        torch.save(checkpoint_data, checkpoint_path)

        # Also save as latest checkpoint
        latest_path = self.checkpoint_dir / "season_latest.pt"
        torch.save(checkpoint_data, latest_path)

        orchestrator.logger.info(f"Season checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_season_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load season checkpoint with validation."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            # Try latest checkpoint
            latest_path = self.checkpoint_dir / "season_latest.pt"
            if latest_path.exists():
                checkpoint_path = latest_path
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        # Validate checkpoint structure
        required_keys = ["orchestrator_version", "current_phase", "season_config"]
        missing_keys = [k for k in required_keys if k not in checkpoint_data]
        if missing_keys:
            raise ValueError(f"Invalid checkpoint, missing keys: {missing_keys}")

        return checkpoint_data

    def _serialize_phase_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize phase state for checkpointing (excluding non-serializable objects)."""
        serializable_state = {}

        for key, value in state.items():
            if key in ["model", "optimizer", "logger"]:
                # These are handled separately or recreated
                continue
            elif key == "dataset":
                # Store dataset configuration instead of object
                if hasattr(value, "__class__"):
                    serializable_state[f"{key}_class"] = value.__class__.__name__
                    if hasattr(value, "challenge_file"):
                        serializable_state[f"{key}_config"] = {
                            "challenge_file": getattr(value, "challenge_file", None),
                            "solution_file": getattr(value, "solution_file", None),
                            "max_grid_size": getattr(value, "max_grid_size", 30)
                        }
                continue
            elif torch.is_tensor(value):
                serializable_state[key] = value.detach().cpu()
            elif isinstance(value, (int, float, str, bool, list, tuple, dict)):
                serializable_state[key] = value
            elif hasattr(value, "state_dict"):
                # Neural network modules
                try:
                    serializable_state[f"{key}_state_dict"] = value.state_dict()
                    if hasattr(value, "__class__"):
                        serializable_state[f"{key}_class"] = value.__class__.__name__
                except Exception:
                    pass
            else:
                # Skip non-serializable objects but log them
                continue

        return serializable_state


class AlphaOrchestrator:
    """Main orchestrator for Neural-Guided Search 2.0 training seasons."""

    # Define phase execution order for Neural-Guided Search 2.0
    SEASON_PHASES = [
        ("phase2_meta_learning", phase2_meta_learning, "Meta Learning + HRM Fast Adaptation"),
        ("phase3_self_critique", phase3_self_critique, "Self Critique + STaR Bootstrapping"),
        ("phase4_mcts_alpha", phase4_mcts_alpha, "MCTS Alpha + Neural Guidance"),
        ("phase6_neuro_priors", phase6_neuro_priors, "Neural DSL Operation Priors"),
        ("phase7_relmem", phase7_relmem, "Relational Memory + Biological Learning")
    ]

    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        """Initialize the Alpha Orchestrator."""
        self.config = config
        self.device = torch.device(device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # State management (initialize first)
        self.shared_state = {}
        self.completed_phases = set()
        self.current_phase = None
        self.model = None
        self.model_config = None
        self.dataset = None

        # Initialize logging
        self.setup_logging()

        # Checkpoint management
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.checkpoint_manager = SeasonCheckpoint(checkpoint_dir)

        # Metrics tracking
        self.season_metrics = {
            "start_time": None,
            "end_time": None,
            "total_duration": None,
            "phases_completed": 0,
            "total_training_steps": 0,
            "peak_memory_usage": 0,
            "total_loss": 0.0,
            "best_em_score": 0.0,
            "breakthrough_count": 0
        }

        self.phase_metrics = {}

        # Error handling
        self.max_retries = config.get("max_retries", 3)
        self.recovery_enabled = config.get("enable_recovery", True)

        self.logger.info("Alpha Orchestrator initialized for Neural-Guided Search 2.0")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"HRM Available: {HRM_AVAILABLE}")

    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create orchestrator-specific logger
        self.logger = logging.getLogger("AlphaOrchestrator")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"alpha_orchestrator_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Also initialize TrainLogger for compatibility
        self.train_logger = TrainLogger(str(log_dir))
        self.shared_state["logger"] = self.train_logger

    def initialize_shared_state(self):
        """Initialize shared state components used across all phases."""
        self.logger.info("Initializing shared state components...")

        # Initialize model
        if self.model is None:
            self.logger.info("Creating TOPAS model...")
            model_config = self.config.get("model", {})

            try:
                # Filter config for ModelConfig
                from trainers.trainer_utils import filter_config
                model_cfg_dict = filter_config(model_config, ModelConfig.__annotations__)
                self.model_config = ModelConfig(**model_cfg_dict)
                self.model_config.validate()
            except Exception as e:
                self.logger.warning(f"Error creating model config: {e}. Using defaults.")
                self.model_config = ModelConfig()

            self.model = TopasARC60M(self.model_config).to(self.device)
            self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")

        # Initialize dataset
        if self.dataset is None:
            self.logger.info("Loading ARC dataset...")
            dataset_config = self.config.get("dataset", {})

            try:
                self.dataset = ARCDataset(
                    challenge_file=dataset_config.get("train_challenges", "arc-agi_training_challenges.json"),
                    solution_file=dataset_config.get("train_solutions", "arc-agi_training_solutions.json"),
                    device=str(self.device),
                    max_grid_size=dataset_config.get("max_grid_size", 30)
                )
                self.logger.info(f"Dataset loaded with {len(self.dataset)} tasks")
            except Exception as e:
                self.logger.error(f"Failed to load ARC dataset: {e}")
                raise

        # Initialize RelMem if not exists
        if "relmem" not in self.shared_state:
            self.logger.info("Initializing RelationalMemoryNeuro...")
            self.shared_state["relmem"] = RelationalMemoryNeuro(
                hidden_dim=self.model_config.slot_dim,
                max_concepts=4096,
                device=self.device
            ).to(self.device)

        # Initialize HRM components if available
        if HRM_AVAILABLE and self.config.get("hrm_integration_enabled", True):
            self.logger.info("Initializing HRM components...")
            self.shared_state["hrm_integration_enabled"] = True
            self.shared_state["hrm_embedding_enabled"] = True
            self.shared_state["curriculum_level"] = 0
        else:
            self.logger.info("HRM integration disabled or unavailable")
            self.shared_state["hrm_integration_enabled"] = False

        # Populate shared state
        self.shared_state.update({
            "model": self.model,
            "dataset": self.dataset,
            "device": self.device,
            "global_step": 0,
            "orchestrator": self,
            "train_logger": self.train_logger
        })

        self.logger.info("Shared state initialization complete")

    def execute_phase(self, phase_name: str, phase_module, phase_description: str,
                     retry_count: int = 0) -> Tuple[bool, Dict[str, Any]]:
        """Execute a single training phase with error handling and recovery."""
        self.logger.info(f"Starting {phase_description} (attempt {retry_count + 1})")

        # Track phase start time
        phase_start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            # Get phase configuration
            phase_config = self.config.get(phase_name, {}).copy()
            phase_config.update({
                "device": str(self.device),
                "log_dir": self.config.get("log_dir", "logs"),
                "checkpoint_dir": self.config.get("checkpoint_dir", "checkpoints"),
                "season_orchestrator": True,  # Flag to indicate orchestrated execution
                "retry_count": retry_count
            })

            # Execute phase
            self.current_phase = phase_name
            updated_state = phase_module.run(phase_config, self.shared_state.copy())

            # Update shared state with phase results
            if isinstance(updated_state, dict):
                self.shared_state.update(updated_state)

                # Update global step tracking
                if "global_step" in updated_state:
                    self.season_metrics["total_training_steps"] = updated_state["global_step"]

            # Track phase completion metrics
            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time
            final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_delta = final_memory - initial_memory

            phase_metrics = {
                "duration": phase_duration,
                "memory_usage_delta": memory_delta,
                "start_time": phase_start_time,
                "end_time": phase_end_time,
                "success": True,
                "retry_count": retry_count,
                "final_loss": updated_state.get(f"{phase_name}_final_loss", 0.0),
                "epochs_completed": updated_state.get(f"{phase_name}_epochs", 0)
            }

            # Check for breakthroughs
            em_score = updated_state.get("best_em_score", 0.0)
            if em_score > self.season_metrics["best_em_score"]:
                self.season_metrics["best_em_score"] = em_score
                if em_score >= 33.0:  # Breakthrough threshold
                    self.season_metrics["breakthrough_count"] += 1
                    self.logger.info(f"🚀 BREAKTHROUGH DETECTED: {em_score}% EM!")

            self.phase_metrics[phase_name] = phase_metrics
            self.completed_phases.add(phase_name)

            self.logger.info(f"✅ {phase_description} completed successfully in {phase_duration:.1f}s")

            # Save checkpoint after successful phase completion
            checkpoint_path = self.checkpoint_manager.save_season_checkpoint(
                self, phase_name, updated_state
            )

            return True, updated_state

        except Exception as e:
            self.logger.error(f"❌ {phase_description} failed: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Track failure metrics
            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time

            self.phase_metrics[phase_name] = {
                "duration": phase_duration,
                "start_time": phase_start_time,
                "end_time": phase_end_time,
                "success": False,
                "retry_count": retry_count,
                "error": str(e),
                "error_type": type(e).__name__
            }

            # Attempt recovery if enabled and retries available
            if self.recovery_enabled and retry_count < self.max_retries:
                self.logger.info(f"Attempting phase recovery (retry {retry_count + 1}/{self.max_retries})")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self.execute_phase(phase_name, phase_module, phase_description, retry_count + 1)

            return False, self.shared_state

    def run_season(self, phases_to_run: Optional[List[str]] = None,
                  resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a complete training season with all phases.

        Args:
            phases_to_run: Optional list of phase names to execute. If None, runs all phases.
            resume_from: Optional checkpoint path to resume from.

        Returns:
            Season results dictionary with metrics and outcomes.
        """
        self.logger.info("🚀 Starting Neural-Guided Search 2.0 Training Season")
        self.season_metrics["start_time"] = time.time()

        try:
            # Resume from checkpoint if specified
            if resume_from:
                self.logger.info(f"Resuming from checkpoint: {resume_from}")
                checkpoint_data = self.checkpoint_manager.load_season_checkpoint(resume_from)
                self._restore_from_checkpoint(checkpoint_data)
            else:
                # Initialize fresh shared state
                self.initialize_shared_state()

            # Determine phases to execute
            if phases_to_run is None:
                phases_to_execute = self.SEASON_PHASES
            else:
                phases_to_execute = [
                    (name, module, desc) for name, module, desc in self.SEASON_PHASES
                    if name in phases_to_run
                ]

            self.logger.info(f"Executing phases: {[desc for _, _, desc in phases_to_execute]}")

            # Execute phases in sequence
            season_success = True
            for phase_name, phase_module, phase_description in phases_to_execute:

                # Skip already completed phases (for resume scenarios)
                if phase_name in self.completed_phases:
                    self.logger.info(f"Skipping already completed phase: {phase_description}")
                    continue

                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"PHASE: {phase_description}")
                self.logger.info(f"{'='*60}")

                # Execute phase with error handling
                success, updated_state = self.execute_phase(phase_name, phase_module, phase_description)

                if not success:
                    self.logger.error(f"Season failed at phase: {phase_description}")
                    season_success = False

                    if not self.config.get("continue_on_failure", False):
                        break

                # Update metrics
                self.season_metrics["phases_completed"] += 1

                # Memory management
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated()
                    if current_memory > self.season_metrics["peak_memory_usage"]:
                        self.season_metrics["peak_memory_usage"] = current_memory

                    # Clear cache periodically
                    torch.cuda.empty_cache()

            # Finalize season
            self.season_metrics["end_time"] = time.time()
            self.season_metrics["total_duration"] = (
                self.season_metrics["end_time"] - self.season_metrics["start_time"]
            )

            # Generate final report
            season_results = self._generate_season_report(season_success)

            if season_success:
                self.logger.info("🎉 Neural-Guided Search 2.0 Training Season COMPLETED SUCCESSFULLY!")
            else:
                self.logger.info("⚠️ Neural-Guided Search 2.0 Training Season COMPLETED WITH ERRORS")

            return season_results

        except Exception as e:
            self.logger.error(f"Fatal error during season execution: {e}")
            self.logger.error(traceback.format_exc())

            # Generate error report
            season_results = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "completed_phases": list(self.completed_phases),
                "metrics": self.season_metrics,
                "phase_metrics": self.phase_metrics
            }

            return season_results
        finally:
            # Cleanup
            if hasattr(self, 'train_logger'):
                self.train_logger.close()

    def _restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Restore orchestrator state from checkpoint."""
        self.logger.info("Restoring from checkpoint...")

        # Restore basic state
        self.completed_phases = set(checkpoint_data.get("completed_phases", []))
        self.current_phase = checkpoint_data.get("current_phase")
        self.season_metrics.update(checkpoint_data.get("training_metrics", {}))
        self.phase_metrics.update(checkpoint_data.get("phase_metrics", {}))

        # Restore model if available
        if checkpoint_data.get("model_state_dict") and checkpoint_data.get("model_config"):
            try:
                self.model_config = ModelConfig(**checkpoint_data["model_config"])
                self.model = TopasARC60M(self.model_config).to(self.device)
                self.model.load_state_dict(checkpoint_data["model_state_dict"])
                self.logger.info("Model restored from checkpoint")
            except Exception as e:
                self.logger.warning(f"Failed to restore model: {e}")

        # Initialize remaining components
        self.initialize_shared_state()

        # Restore phase state
        phase_state = checkpoint_data.get("phase_state", {})
        self.shared_state.update(phase_state)

        self.logger.info(f"Restored from checkpoint. Completed phases: {list(self.completed_phases)}")

    def _generate_season_report(self, success: bool) -> Dict[str, Any]:
        """Generate comprehensive season report."""
        total_duration = self.season_metrics.get("total_duration", 0)

        report = {
            "success": success,
            "season_type": "Neural-Guided Search 2.0",
            "total_duration": total_duration,
            "total_duration_formatted": str(timedelta(seconds=int(total_duration))),
            "phases_completed": len(self.completed_phases),
            "phases_attempted": len(self.phase_metrics),
            "completed_phases": list(self.completed_phases),

            # Performance metrics
            "best_em_score": self.season_metrics.get("best_em_score", 0.0),
            "breakthrough_count": self.season_metrics.get("breakthrough_count", 0),
            "total_training_steps": self.season_metrics.get("total_training_steps", 0),
            "peak_memory_usage_mb": self.season_metrics.get("peak_memory_usage", 0) / (1024**2),

            # Phase breakdown
            "phase_metrics": self.phase_metrics,
            "phase_success_rate": len([p for p in self.phase_metrics.values() if p.get("success", False)]) / max(1, len(self.phase_metrics)),

            # System info
            "device": str(self.device),
            "torch_version": torch.__version__,
            "model_parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "dataset_size": len(self.dataset) if self.dataset else 0,

            # Timing breakdown
            "phase_durations": {name: metrics.get("duration", 0) for name, metrics in self.phase_metrics.items()},
            "average_phase_duration": sum(metrics.get("duration", 0) for metrics in self.phase_metrics.values()) / max(1, len(self.phase_metrics))
        }

        # Log summary
        self.logger.info("\n" + "="*80)
        self.logger.info("NEURAL-GUIDED SEARCH 2.0 SEASON REPORT")
        self.logger.info("="*80)
        self.logger.info(f"Status: {'SUCCESS' if success else 'FAILED'}")
        self.logger.info(f"Duration: {report['total_duration_formatted']}")
        self.logger.info(f"Phases: {report['phases_completed']}/{len(self.SEASON_PHASES)} completed")
        self.logger.info(f"Best EM Score: {report['best_em_score']:.1f}%")
        self.logger.info(f"Breakthroughs: {report['breakthrough_count']}")
        self.logger.info(f"Training Steps: {report['total_training_steps']:,}")
        self.logger.info(f"Peak Memory: {report['peak_memory_usage_mb']:.1f} MB")
        self.logger.info("="*80)

        return report

    def run_single_phase(self, phase_name: str) -> Dict[str, Any]:
        """Execute a single phase (useful for debugging and testing)."""
        self.logger.info(f"Running single phase: {phase_name}")

        # Find phase module
        phase_info = None
        for name, module, description in self.SEASON_PHASES:
            if name == phase_name:
                phase_info = (name, module, description)
                break

        if phase_info is None:
            raise ValueError(f"Unknown phase: {phase_name}")

        # Initialize shared state if needed
        if not self.shared_state:
            self.initialize_shared_state()

        # Execute phase
        success, updated_state = self.execute_phase(*phase_info)

        return {
            "success": success,
            "phase_name": phase_name,
            "metrics": self.phase_metrics.get(phase_name, {}),
            "state": updated_state
        }


def create_default_config() -> Dict[str, Any]:
    """Create default Neural-Guided Search 2.0 configuration."""
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": "logs/ngs2",
        "checkpoint_dir": "checkpoints/ngs2",
        "max_retries": 3,
        "enable_recovery": True,
        "continue_on_failure": False,
        "hrm_integration_enabled": HRM_AVAILABLE,

        # Model configuration
        "model": {
            "model_width": 640,
            "model_slots": 64,
            "vocab_size": 10,
            "max_seq_len": 2048,
            "num_heads": 8,
            "num_layers": 12
        },

        # Dataset configuration
        "dataset": {
            "train_challenges": "arc-agi_training_challenges.json",
            "train_solutions": "arc-agi_training_solutions.json",
            "max_grid_size": 30
        },

        # Phase 2: Meta Learning
        "phase2_meta_learning": {
            "num_epochs": 5,
            "meta_lr": 1e-3,
            "inner_lr": 1e-2,
            "inner_steps": 5,
            "tasks_per_meta_batch": 4,
            "use_hrm_embeddings": True,
            "use_hrm_fast_adapt": True,
            "hrm_inner_steps": 2,
            "hrm_inner_lr": 5e-3
        },

        # Phase 3: Self Critique
        "phase3_self_critique": {
            "epochs": 3,
            "steps_per_epoch": 200,
            "learning_rate": 5e-5,
            "counterexample_rate": 0.3,
            "trace_analysis_steps": 100,
            "consistency_weight": 0.1,
            "star_bootstrap_rate": 0.5
        },

        # Phase 4: MCTS Alpha
        "phase4_mcts_alpha": {
            "epochs": 4,
            "steps_per_epoch": 150,
            "mcts_simulations": 100,
            "exploration_constant": 1.4,
            "neural_guidance_weight": 0.7,
            "alpha_zero_style": True,
            "tree_search_depth": 10
        },

        # Phase 6: Neuro Priors
        "phase6_neuro_priors": {
            "epochs": 3,
            "steps_per_epoch": 250,
            "learning_rate": 1e-4,
            "prior_network_layers": [256, 128, 64],
            "dsl_operation_count": 41,
            "prior_regularization": 0.01,
            "democratic_bias_weight": 0.5
        },

        # Phase 7: RelMem + Biological Learning
        "phase7_relmem": {
            "epochs": 150,
            "steps_per_epoch": 400,
            "learning_rate": 1e-4,
            "dream_pretrain_epochs": 3,
            "nightmare_alpha": 0.08,
            "dopamine_capture_rate": 0.1,
            "biological_learning": True,
            "breakthrough_threshold": 33.0,
            "conscious_ai_enabled": True,
            "monologue_interval": 200,
            "hemispheric_competition": True
        }
    }


def main():
    """Main entry point for Alpha Orchestrator."""
    parser = argparse.ArgumentParser(
        description="Alpha Orchestrator - Neural-Guided Search 2.0 Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alpha_orchestrator.py --config configs/ngs2.json
  python alpha_orchestrator.py --create-config --output configs/my_ngs2.json
  python alpha_orchestrator.py --config configs/ngs2.json --phases phase2_meta_learning,phase3_self_critique
  python alpha_orchestrator.py --resume checkpoints/season_latest.pt
  python alpha_orchestrator.py --single-phase phase2_meta_learning --config configs/ngs2.json
        """)

    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")
    parser.add_argument("--output", type=str, default="neural_guided_search_v2.json", help="Output path for created config")
    parser.add_argument("--phases", type=str, help="Comma-separated list of phases to run")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--single-phase", type=str, help="Run single phase only")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=getattr(logging, args.log_level))

    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        config_path = Path(args.output)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, sort_keys=True)

        print(f"✅ Default Neural-Guided Search 2.0 config created: {config_path}")
        return 0

    # Load configuration
    if not args.config:
        print("❌ Configuration file required. Use --config or --create-config")
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Override device if specified
    if args.device:
        config["device"] = args.device

    try:
        # Initialize orchestrator
        orchestrator = AlphaOrchestrator(config, device=config.get("device"))

        # Parse phases if specified
        phases_to_run = None
        if args.phases:
            phases_to_run = [phase.strip() for phase in args.phases.split(",")]

        # Execute training
        if args.single_phase:
            # Run single phase
            results = orchestrator.run_single_phase(args.single_phase)
            success = results["success"]
        else:
            # Run full season or specified phases
            results = orchestrator.run_season(
                phases_to_run=phases_to_run,
                resume_from=args.resume
            )
            success = results["success"]

        # Save final results
        results_file = Path(config.get("log_dir", "logs")) / "season_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📊 Results saved to: {results_file}")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())