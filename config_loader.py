"""
Configuration loader for Alpha-ARC X Neural-Guided Search 2.0
Maps JSON config files to CLI argument namespace for train_simple_hrm_topas.py
"""

import json
import argparse
from typing import Dict, Any

def load_config_to_args(config_path: str, base_args=None) -> argparse.Namespace:
    """
    Load JSON config and convert to argument namespace compatible with train_simple_hrm_topas.py

    Args:
        config_path: Path to JSON config file
        base_args: Optional base arguments to extend (CLI args take precedence)

    Returns:
        argparse.Namespace with all configuration values
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Start with base args if provided, otherwise create new namespace
    if base_args is not None:
        args = base_args
    else:
        args = argparse.Namespace()

    # Training parameters
    training = config.get("training", {})
    if not hasattr(args, 'epochs') or args.epochs is None:
        args.epochs = training.get("num_epochs", 150)
    if not hasattr(args, 'lr') or args.lr is None:
        args.lr = training.get("learning_rate", 2e-4)
    if not hasattr(args, 'grad_clip') or args.grad_clip is None:
        args.grad_clip = training.get("grad_clip", 0.8)

    # Search parameters
    search = config.get("search", {})
    if not hasattr(args, 'search_alg') or args.search_alg is None:
        args.search_alg = search.get("search_alg", "beam")
    if not hasattr(args, 'puct_nodes') or args.puct_nodes is None:
        args.puct_nodes = search.get("puct_nodes", 1500)
    if not hasattr(args, 'puct_depth') or args.puct_depth is None:
        args.puct_depth = search.get("puct_depth", 6)
    if not hasattr(args, 'c_puct') or args.c_puct is None:
        args.c_puct = search.get("c_puct", 1.25)
    if not hasattr(args, 'root_dirichlet_alpha') or args.root_dirichlet_alpha is None:
        args.root_dirichlet_alpha = search.get("root_dirichlet_alpha", 0.3)
    if not hasattr(args, 'root_dirichlet_eps') or args.root_dirichlet_eps is None:
        args.root_dirichlet_eps = search.get("root_dirichlet_eps", 0.25)
    if not hasattr(args, 'sc_star') or args.sc_star is None:
        args.sc_star = search.get("sc_star", 1)

    # Replay parameters
    replay = config.get("replay", {})
    if not hasattr(args, 'replay_cap') or args.replay_cap is None:
        args.replay_cap = replay.get("replay_cap", 100000)
    if not hasattr(args, 'near_miss_hamming_pct') or args.near_miss_hamming_pct is None:
        args.near_miss_hamming_pct = replay.get("near_miss_hamming_pct", 5.0)

    # Dopamine parameters
    dopamine = config.get("dopamine", {})
    if not hasattr(args, 'breakthrough_threshold') or args.breakthrough_threshold is None:
        args.breakthrough_threshold = dopamine.get("breakthrough_threshold", 0.33)
    if not hasattr(args, 'nightmare_alpha') or args.nightmare_alpha is None:
        args.nightmare_alpha = dopamine.get("nightmare_alpha", 0.08)
    if not hasattr(args, 'nightmare_min_interval') or args.nightmare_min_interval is None:
        args.nightmare_min_interval = dopamine.get("nightmare_min_interval", 200)
    if not hasattr(args, 'nightmare_max_interval') or args.nightmare_max_interval is None:
        args.nightmare_max_interval = dopamine.get("nightmare_max_interval", 1000)

    # RelMem parameters
    relmem = config.get("relmem", {})
    if not hasattr(args, 'relmem_reg_alpha') or args.relmem_reg_alpha is None:
        args.relmem_reg_alpha = relmem.get("relmem_reg_alpha", 1e-3)
    if not hasattr(args, 'relmem_reg_beta') or args.relmem_reg_beta is None:
        args.relmem_reg_beta = relmem.get("relmem_reg_beta", 5e-4)
    if not hasattr(args, 'relmem_bind_iou') or args.relmem_bind_iou is None:
        args.relmem_bind_iou = relmem.get("relmem_bind_iou", 0.25)
    if not hasattr(args, 'relmem_bias_ramp_start') or args.relmem_bias_ramp_start is None:
        args.relmem_bias_ramp_start = relmem.get("relmem_bias_ramp_start", 10)
    if not hasattr(args, 'relmem_bias_max') or args.relmem_bias_max is None:
        args.relmem_bias_max = relmem.get("relmem_bias_max", 0.5)

    # Dream parameters
    dream = config.get("dream", {})
    if not hasattr(args, 'enable_dream') or args.enable_dream is None:
        args.enable_dream = dream.get("enable_dream", True)
    if not hasattr(args, 'dream_micro_ticks') or args.dream_micro_ticks is None:
        args.dream_micro_ticks = dream.get("dream_micro_ticks", 1)
    if not hasattr(args, 'dream_full_every') or args.dream_full_every is None:
        args.dream_full_every = dream.get("dream_full_every", 10)
    if not hasattr(args, 'dream_pretrain_epochs') or args.dream_pretrain_epochs is None:
        args.dream_pretrain_epochs = dream.get("dream_pretrain_epochs", 3)
    if not hasattr(args, 'dream_pretrain_freeze_model') or args.dream_pretrain_freeze_model is None:
        args.dream_pretrain_freeze_model = dream.get("dream_pretrain_freeze_model", True)

    # Self-play parameters
    selfplay = config.get("selfplay", {})
    if not hasattr(args, 'selfplay_enable') or args.selfplay_enable is None:
        args.selfplay_enable = selfplay.get("selfplay_enable", True)
    if not hasattr(args, 'selfplay_interval') or args.selfplay_interval is None:
        args.selfplay_interval = selfplay.get("selfplay_interval", 250)
    if not hasattr(args, 'selfplay_weight') or args.selfplay_weight is None:
        args.selfplay_weight = selfplay.get("selfplay_weight", 0.1)
    if not hasattr(args, 'selfplay_topk') or args.selfplay_topk is None:
        args.selfplay_topk = selfplay.get("selfplay_topk", 3)
    if not hasattr(args, 'selfplay_buffer_size') or args.selfplay_buffer_size is None:
        args.selfplay_buffer_size = selfplay.get("selfplay_buffer_size", 200)

    # Monologue parameters
    monologue = config.get("monologue", {})
    if not hasattr(args, 'monologue_interval') or args.monologue_interval is None:
        args.monologue_interval = monologue.get("monologue_interval", 200)
    if not hasattr(args, 'monologue_consistency_target') or args.monologue_consistency_target is None:
        args.monologue_consistency_target = monologue.get("monologue_consistency_target", 0.85)
    if not hasattr(args, 'monologue_selfplay_bonus') or args.monologue_selfplay_bonus is None:
        args.monologue_selfplay_bonus = monologue.get("monologue_selfplay_bonus", 0.05)

    # Evaluation parameters
    eval_config = config.get("eval", {})
    if not hasattr(args, 'eval_interval') or args.eval_interval is None:
        args.eval_interval = eval_config.get("eval_interval", 5)
    if not hasattr(args, 'dataset') or args.dataset is None:
        args.dataset = eval_config.get("dataset", "arc2")

    # Set default device if not specified
    if not hasattr(args, 'device') or args.device is None:
        args.device = "cuda"

    return args

def config_to_cli_args(config_path: str) -> str:
    """
    Convert JSON config to CLI argument string for easy command generation

    Args:
        config_path: Path to JSON config file

    Returns:
        String of CLI arguments
    """
    args = load_config_to_args(config_path)

    # Build argument string
    arg_parts = []

    # Core arguments
    arg_parts.append(f"--device {args.device}")
    arg_parts.append(f"--dataset {args.dataset}")
    arg_parts.append(f"--epochs {args.epochs}")
    arg_parts.append(f"--lr {args.lr}")
    arg_parts.append(f"--grad-clip {args.grad_clip}")
    arg_parts.append(f"--eval-interval {args.eval_interval}")

    # Dream arguments
    if args.enable_dream:
        arg_parts.append("--enable-dream")
        arg_parts.append(f"--dream-micro-ticks {args.dream_micro_ticks}")
        arg_parts.append(f"--dream-full-every {args.dream_full_every}")
        arg_parts.append(f"--dream-pretrain-epochs {args.dream_pretrain_epochs}")
        if args.dream_pretrain_freeze_model:
            arg_parts.append("--dream-pretrain-freeze-model")

    # Search arguments
    arg_parts.append(f"--search-alg {args.search_alg}")
    if args.search_alg == "puct":
        arg_parts.append(f"--puct-nodes {args.puct_nodes}")
        arg_parts.append(f"--puct-depth {args.puct_depth}")
        arg_parts.append(f"--c-puct {args.c_puct}")
        arg_parts.append(f"--root-dirichlet-alpha {args.root_dirichlet_alpha}")
        arg_parts.append(f"--root-dirichlet-eps {args.root_dirichlet_eps}")
    if args.sc_star > 1:
        arg_parts.append("--sc-star")

    # Replay arguments
    arg_parts.append(f"--replay-cap {args.replay_cap}")
    arg_parts.append(f"--near-miss-hamming-pct {args.near_miss_hamming_pct}")

    # Dopamine arguments
    arg_parts.append(f"--breakthrough-threshold {args.breakthrough_threshold}")
    arg_parts.append(f"--nightmare-alpha {args.nightmare_alpha}")
    arg_parts.append(f"--nightmare-min-interval {args.nightmare_min_interval}")
    arg_parts.append(f"--nightmare-max-interval {args.nightmare_max_interval}")

    # RelMem arguments
    arg_parts.append(f"--relmem-reg-alpha {args.relmem_reg_alpha}")
    arg_parts.append(f"--relmem-reg-beta {args.relmem_reg_beta}")
    arg_parts.append(f"--relmem-bind-iou {args.relmem_bind_iou}")
    arg_parts.append(f"--relmem-bias-ramp-start {args.relmem_bias_ramp_start}")
    arg_parts.append(f"--relmem-bias-max {args.relmem_bias_max}")

    # Self-play arguments
    if args.selfplay_enable:
        arg_parts.append("--selfplay-enable")
        arg_parts.append(f"--selfplay-interval {args.selfplay_interval}")
        arg_parts.append(f"--selfplay-weight {args.selfplay_weight}")
        arg_parts.append(f"--selfplay-topk {args.selfplay_topk}")
        arg_parts.append(f"--selfplay-buffer-size {args.selfplay_buffer_size}")

    # Monologue arguments
    arg_parts.append(f"--monologue-interval {args.monologue_interval}")
    arg_parts.append(f"--monologue-consistency-target {args.monologue_consistency_target}")
    arg_parts.append(f"--monologue-selfplay-bonus {args.monologue_selfplay_bonus}")

    return " \\\n  ".join(arg_parts)

if __name__ == "__main__":
    # Demo usage
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        cli_args = config_to_cli_args(config_path)
        print("# Alpha-ARC X Command:")
        print(f"venv/bin/python train_simple_hrm_topas.py \\")
        print(f"  {cli_args}")
    else:
        print("Usage: python config_loader.py configs/alpha_win.json")