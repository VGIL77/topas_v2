#!/usr/bin/env python3
"""
ARC-II Dataset Loader (ARC-AGI format, Prize 2025)
Fully compatible with arc-agi_training/arc-agi_evaluation/arc-agi_test JSON files
"""

import os, json, torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _pad_or_crop(grid: List[List[int]], max_size: int = 30) -> List[List[int]]:
    if not grid or not grid[0]:
        return [[0]]
    h, w = len(grid), len(grid[0])
    if h > max_size: grid = grid[:max_size]
    if w > max_size: grid = [row[:max_size] for row in grid]
    return grid


def _sanitize_grid(grid: List[List[int]], num_colors: int = 10, max_size: int = 30) -> List[List[int]]:
    grid = _pad_or_crop(grid, max_size)
    return [[min(max(int(c), 0), num_colors - 1) for c in row] for row in grid]


class ARC2Dataset(Dataset):
    """
    Wraps ARC-II JSON files into a PyTorch Dataset.
    Returns (demos, test_inputs, test_outputs, task_id).
    """

    def __init__(self, challenge_file: str, solution_file: Optional[str] = None,
                 device: str = "cpu", max_grid_size: int = 30, min_test_cases: int = 0):
        self.device = device
        self.max_grid_size = max_grid_size
        self.min_test_cases = min_test_cases

        if not os.path.exists(challenge_file):
            raise FileNotFoundError(f"Challenge file not found: {challenge_file}")

        with open(challenge_file, "r") as f:
            self.challenges = json.load(f)

        if solution_file and os.path.exists(solution_file):
            with open(solution_file, "r") as f:
                self.solutions = json.load(f)
        else:
            self.solutions = None

        # Hard-mode curriculum shaping: only load puzzles with >= min_test_cases
        self.task_ids = []
        for tid, task in self.challenges.items():
            test_cases = task.get("test", [])
            if len(test_cases) >= self.min_test_cases:
                self.task_ids.append(tid)

        if self.min_test_cases > 0:
            logger.info(f"[Curriculum] Loaded {len(self.task_ids)}/{len(self.challenges)} puzzles with ≥{self.min_test_cases} test cases")

    def __len__(self) -> int:
        return len(self.task_ids)

    def __getitem__(self, idx: int):
        tid = self.task_ids[idx]
        task = self.challenges[tid]

        # Curriculum Warmup (epochs 0-4): skip trivial puzzles
        # Guard so Dream pretrain (before epochs start) never breaks
        current_epoch = globals().get("CURRENT_EPOCH", 9999)
        if current_epoch < 5:
            test_inputs_raw = task.get("test", [])
            if test_inputs_raw:
                first_test = test_inputs_raw[0].get("input", [])
                H = len(first_test) if first_test else 0
                W = len(first_test[0]) if first_test and first_test[0] else 0
                if (H < 8 or W < 8) or (len(test_inputs_raw) < 2):
                    # Strict skip (no resample)
                    raise StopIteration("Skipped trivial puzzle due to curriculum gating")

        # Demos (train pairs)
        demos = []
        for pair in task["train"]:
            inp = torch.tensor(_sanitize_grid(pair["input"], max_size=self.max_grid_size),
                               dtype=torch.long, device=self.device).unsqueeze(0)
            out = torch.tensor(_sanitize_grid(pair["output"], max_size=self.max_grid_size),
                               dtype=torch.long, device=self.device).unsqueeze(0)
            demos.append((inp, out))

        # Test inputs
        test_inputs = []
        for t in task["test"]:
            inp = torch.tensor(_sanitize_grid(t["input"], max_size=self.max_grid_size),
                               dtype=torch.long, device=self.device).unsqueeze(0)
            test_inputs.append(inp)

        # Test outputs if solutions exist
        test_outputs = None
        if self.solutions and tid in self.solutions:
            sols = self.solutions[tid]
            test_outputs = []
            for s in sols:
                if isinstance(s, dict) and "output" in s:
                    grid = s["output"]
                else:
                    grid = s
                out = torch.tensor(_sanitize_grid(grid, max_size=self.max_grid_size),
                                   dtype=torch.long, device=self.device).unsqueeze(0)
                test_outputs.append(out)

        return demos, test_inputs, test_outputs, tid


class ARC2DataLoader:
    """
    Convenience class: wraps ARC-II training/eval/test sets
    """

    def __init__(self, data_dir: str = ".", device: str = "cpu"):
        self.data_dir = data_dir
        self.device = device

        self.train_ds = ARC2Dataset(
            os.path.join(data_dir, "arc-agi_training_challenges.json"),
            os.path.join(data_dir, "arc-agi_training_solutions.json"),
            device=device
        )
        self.eval_ds = ARC2Dataset(
            os.path.join(data_dir, "arc-agi_evaluation_challenges.json"),
            os.path.join(data_dir, "arc-agi_evaluation_solutions.json"),
            device=device
        )
        self.test_ds = ARC2Dataset(
            os.path.join(data_dir, "arc-agi_test_challenges.json"),
            None,
            device=device
        )

    def get_train_loader(self, batch_size=1, shuffle=True):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=shuffle)

    def get_eval_loader(self, batch_size=1, shuffle=False):
        return DataLoader(self.eval_ds, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(self, batch_size=1, shuffle=False):
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=shuffle)