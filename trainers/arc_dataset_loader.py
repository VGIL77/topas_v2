#!/usr/bin/env python3
"""
ARC Dataset Loader for ARC Prize 2025
Loads actual ARC-AGI JSON files in the competition format
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
import logging

# Module logger
logger = logging.getLogger(__name__)


def load_arc_json(challenge_file: str, solution_file: Optional[str] = None):
    """
    Load ARC challenge and solution JSON files
    
    Args:
        challenge_file: Path to challenges JSON or directory containing JSON files
        solution_file: Optional path to solutions JSON
    
    Returns:
        Tuple of (challenges dict, solutions dict or None)
    """
    challenges = {}
    solutions = {}
    
    # Check if it's a directory (individual task files) or a single JSON file
    if os.path.isdir(challenge_file):
        # Load individual JSON files from directory
        json_files = [f for f in os.listdir(challenge_file) if f.endswith('.json')]
        for json_file in json_files:
            task_id = json_file.replace('.json', '')
            file_path = os.path.join(challenge_file, json_file)
            with open(file_path, 'r') as f:
                task_data = json.load(f)
                challenges[task_id] = task_data
        
        # For directory format, solutions are in the same file
        solutions = None
    else:
        # Original format: single JSON file
        with open(challenge_file, 'r') as f:
            challenges = json.load(f)
        
        if solution_file and os.path.exists(solution_file):
            with open(solution_file, 'r') as f:
                solutions = json.load(f)
        else:
            solutions = None
    
    return challenges, solutions


def _pad_or_crop(grid: List[List[int]], max_size: int = 30) -> List[List[int]]:
    """
    Pad or crop a 2D grid to ensure it doesn't exceed max_size x max_size.
    
    Args:
        grid: 2D grid as list of lists
        max_size: Maximum allowed dimension
        
    Returns:
        Grid guaranteed to be <= max_size x max_size
    """
    if not grid or not grid[0]:
        # Handle empty grid case
        return [[0]]
    
    height = len(grid)
    width = len(grid[0])
    
    # Crop if too large
    if height > max_size:
        grid = grid[:max_size]
        height = max_size
        logger.debug(f"Cropped grid height from {len(grid)} to {height}")
    
    if width > max_size:
        grid = [row[:max_size] for row in grid]
        width = max_size
        logger.debug(f"Cropped grid width to {width}")
    
    # Pad if too small (optional - ensures consistent size if needed)
    # For ARC, we typically don't pad to max_size but just ensure we don't exceed it
    # The model should handle variable sizes within the max constraint
    
    return grid

def _sanitize_grid(grid: List[List[int]], num_colors: int = 10, max_size: int = 30, log_clamps: bool = False) -> List[List[int]]:
    """
    Sanitize ARC grid: crop/pad first, then clamp all values into [0, num_colors-1]
    
    Args:
        grid: 2D grid as list of lists
        num_colors: Maximum color value + 1 (typically 10 for ARC)
        max_size: Maximum grid dimension
        log_clamps: Whether to log clamped pixel counts
        
    Returns:
        Sanitized grid with all values in [0, num_colors-1]
    """
    # Crop/pad first
    grid = _pad_or_crop(grid, max_size)
    
    # Track clamping statistics if requested
    clamp_count = 0
    negative_clamps = 0
    positive_clamps = 0
    
    # Clamp all values into [0, num_colors-1]
    sanitized_grid = []
    for row in grid:
        sanitized_row = []
        for cell in row:
            original_value = int(cell)
            clamped_value = min(max(original_value, 0), num_colors - 1)
            
            # Count clamps if logging enabled
            if log_clamps and clamped_value != original_value:
                clamp_count += 1
                if original_value < 0:
                    negative_clamps += 1
                elif original_value >= num_colors:
                    positive_clamps += 1
            
            sanitized_row.append(clamped_value)
        sanitized_grid.append(sanitized_row)
    
    # Log clamping statistics if requested
    if log_clamps and clamp_count > 0:
        logger.debug(f"Sanitization clamped {clamp_count} pixels: {negative_clamps} negative → 0, {positive_clamps} over-range → {num_colors-1}")
    
    return sanitized_grid


class ARCDataset(Dataset):
    """
    Wraps ARC Prize 2025 JSON files into a PyTorch Dataset.
    Returns (demos, test_inputs, test_outputs, task_id).
    
    Compatible with the official ARC-AGI JSON format:
    - arc-agi_training-challenges.json / arc-agi_training-solutions.json
    - arc-agi_evaluation-challenges.json / arc-agi_evaluation-solutions.json  
    - arc-agi_test-challenges.json (no solutions)
    """
    
    def __init__(self, challenge_file: str, solution_file: Optional[str] = None, 
                 device: str = "cpu", max_grid_size: int = 30, deterministic: bool = False):
        """
        Args:
            challenge_file: Path to challenges JSON file
            solution_file: Optional path to solutions JSON file
            device: Device to place tensors on
            max_grid_size: Maximum grid size (for validation and cropping)
            deterministic: Flag for reproducible evaluation
        """
        self.device = device
        self.max_grid_size = max_grid_size
        self.deterministic = deterministic
        
        # Check if files exist
        if not os.path.exists(challenge_file):
            raise FileNotFoundError(f"Challenge file not found: {challenge_file}")
        
        # Load JSON files
        self.challenges, self.solutions = load_arc_json(challenge_file, solution_file)
        self.task_ids = list(self.challenges.keys())
        
        # Validate and get statistics
        self._validate_data()
        
        logger.info(f"Loaded {len(self.task_ids)} tasks from {os.path.basename(challenge_file)}")
        if self.solutions:
            logger.info(f"Solutions available from {os.path.basename(solution_file)}")
        else:
            logger.info("No solutions available (test mode)")
    
    def _validate_data(self):
        """Validate data and collect statistics"""
        self.stats = {
            'max_grid_h': 0,
            'max_grid_w': 0,
            'max_demos': 0,
            'max_tests': 0,
            'total_demos': 0,
            'total_tests': 0
        }
        
        for tid in self.task_ids:
            task = self.challenges[tid]
            
            # Check training pairs
            self.stats['max_demos'] = max(self.stats['max_demos'], len(task['train']))
            self.stats['total_demos'] += len(task['train'])
            
            for pair in task['train']:
                inp = pair['input']
                out = pair['output']
                self.stats['max_grid_h'] = max(self.stats['max_grid_h'], len(inp), len(out))
                self.stats['max_grid_w'] = max(self.stats['max_grid_w'], 
                                              max(len(row) for row in inp),
                                              max(len(row) for row in out))
            
            # Check test pairs
            self.stats['max_tests'] = max(self.stats['max_tests'], len(task['test']))
            self.stats['total_tests'] += len(task['test'])
            
            for test in task['test']:
                inp = test['input']
                self.stats['max_grid_h'] = max(self.stats['max_grid_h'], len(inp))
                self.stats['max_grid_w'] = max(self.stats['max_grid_w'], 
                                              max(len(row) for row in inp))
    
    def __len__(self) -> int:
        return len(self.task_ids)
    
    def __getitem__(self, idx: int):
        """
        Returns:
            demos: List of (input, output) tensor pairs from training
            test_inputs: List of test input tensors
            test_outputs: List of test output tensors (or None if no solutions)
            task_id: String ID of the task
        """
        tid = self.task_ids[idx]
        task = self.challenges[tid]
        
        # Process training demonstrations
        demos = []
        for pair in task["train"]:
            try:
                # Apply pad/crop before tensorizing to ensure shape constraints
                inp_grid = _sanitize_grid(pair["input"], num_colors=10, max_size=self.max_grid_size)
                out_grid = _sanitize_grid(pair["output"], num_colors=10, max_size=self.max_grid_size)
                
                inp = torch.tensor(inp_grid, dtype=torch.long, device=self.device)
                out = torch.tensor(out_grid, dtype=torch.long, device=self.device)
                
                # Add batch dimension if needed
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                if out.dim() == 2:
                    out = out.unsqueeze(0)
                
                demos.append((inp, out))
            except Exception as e:
                logger.error(f"Error processing training pair for task {tid}: {e}")
                # Create fallback minimal tensor to prevent crashes
                fallback_tensor = torch.zeros((1, 1, 1), dtype=torch.long, device=self.device)
                demos.append((fallback_tensor, fallback_tensor))
        
        # Process test inputs
        test_inputs = []
        for test in task["test"]:
            try:
                # Apply pad/crop before tensorizing to ensure shape constraints
                inp_grid = _sanitize_grid(test["input"], num_colors=10, max_size=self.max_grid_size)
                inp = torch.tensor(inp_grid, dtype=torch.long, device=self.device)
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                test_inputs.append(inp)
            except Exception as e:
                logger.error(f"Error processing test input for task {tid}: {e}")
                # Create fallback minimal tensor to prevent crashes
                fallback_tensor = torch.zeros((1, 1, 1), dtype=torch.long, device=self.device)
                test_inputs.append(fallback_tensor)
        
        # Process test outputs (if solutions available)
        if self.solutions and tid in self.solutions:
            test_outputs = []
            for sol in self.solutions[tid]:
                try:
                    # Handle different solution formats
                    if isinstance(sol, dict) and 'output' in sol:
                        out_grid = _sanitize_grid(sol['output'], num_colors=10, max_size=self.max_grid_size)
                        out = torch.tensor(out_grid, dtype=torch.long, device=self.device)
                    elif isinstance(sol, list):
                        out_grid = _sanitize_grid(sol, num_colors=10, max_size=self.max_grid_size)
                        out = torch.tensor(out_grid, dtype=torch.long, device=self.device)
                    else:
                        # Try direct tensor conversion with pad/crop
                        out_grid = _sanitize_grid(sol, num_colors=10, max_size=self.max_grid_size)
                        out = torch.tensor(out_grid, dtype=torch.long, device=self.device)
                    
                    if out.dim() == 2:
                        out = out.unsqueeze(0)
                    test_outputs.append(out)
                except Exception as e:
                    logger.error(f"Error processing test output for task {tid}: {e}")
                    # Create fallback minimal tensor to prevent crashes
                    fallback_tensor = torch.zeros((1, 1, 1), dtype=torch.long, device=self.device)
                    test_outputs.append(fallback_tensor)
        else:
            test_outputs = None
        
        return demos, test_inputs, test_outputs, tid
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return self.stats


class ARCDataLoader:
    """
    Convenience class for managing ARC dataloaders
    """
    
    def __init__(self, data_dir: str = ".", device: str = "cpu", deterministic: bool = False):
        """
        Args:
            data_dir: Directory containing ARC JSON files
            device: Device for tensors
            deterministic: Flag for reproducible evaluation
        """
        self.data_dir = data_dir
        self.device = device
        self.deterministic = deterministic
        
        # File paths
        self.train_challenges = os.path.join(data_dir, "arc-agi_training-challenges.json")
        self.train_solutions = os.path.join(data_dir, "arc-agi_training-solutions.json")
        self.eval_challenges = os.path.join(data_dir, "arc-agi_evaluation-challenges.json")
        self.eval_solutions = os.path.join(data_dir, "arc-agi_evaluation-solutions.json")
        self.test_challenges = os.path.join(data_dir, "arc-agi_test-challenges.json")
        
        # Load datasets
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        
        if os.path.exists(self.train_challenges):
            self.train_dataset = ARCDataset(
                self.train_challenges, 
                self.train_solutions if os.path.exists(self.train_solutions) else None,
                device=device,
                deterministic=deterministic
            )
            logger.info(f"Training set: {len(self.train_dataset)} tasks")
        
        if os.path.exists(self.eval_challenges):
            self.eval_dataset = ARCDataset(
                self.eval_challenges,
                self.eval_solutions if os.path.exists(self.eval_solutions) else None,
                device=device,
                deterministic=deterministic
            )
            logger.info(f"Evaluation set: {len(self.eval_dataset)} tasks")
        
        if os.path.exists(self.test_challenges):
            self.test_dataset = ARCDataset(
                self.test_challenges,
                None,  # No solutions for test set
                device=device,
                deterministic=deterministic
            )
            logger.info(f"Test set: {len(self.test_dataset)} tasks")
    
    def get_train_loader(self, batch_size: int = 1, shuffle: bool = True):
        if self.train_dataset is None:
            raise ValueError("Training dataset not available")
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_eval_loader(self, batch_size: int = 1, shuffle: bool = False):
        if self.eval_dataset is None:
            raise ValueError("Evaluation dataset not available")
        return DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_test_loader(self, batch_size: int = 1, shuffle: bool = False):
        if self.test_dataset is None:
            raise ValueError("Test dataset not available")
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)


def test_arc_loader():
    """Test function to demonstrate usage"""
    logger.info("="*70)
    logger.info("TESTING ARC DATASET LOADER")
    logger.info("="*70)
    
    # Test with real files if they exist
    try:
        # Try loading real data
        train_ds = ARCDataset(
            "arc-agi_training-challenges.json",
            "arc-agi_training-solutions.json"
        )
        
        logger.info(f"\nTrain tasks: {len(train_ds)}")
        
        # Load a sample
        demos, test_inputs, test_outputs, tid = train_ds[0]
        logger.info(f"\nTask ID: {tid}")
        logger.info(f"Number of demos: {len(demos)}")
        logger.info(f"Train demo input shape: {demos[0][0].shape}")
        logger.info(f"Train demo output shape: {demos[0][1].shape}")
        logger.info(f"Test input shape: {test_inputs[0].shape}")
        if test_outputs:
            logger.info(f"Ground truth shape: {test_outputs[0].shape}")
        
        # Show stats
        stats = train_ds.get_stats()
        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Max grid height: {stats['max_grid_h']}")
        logger.info(f"  Max grid width: {stats['max_grid_w']}")
        logger.info(f"  Max demos per task: {stats['max_demos']}")
        logger.info(f"  Max tests per task: {stats['max_tests']}")
        
        # Test the pad/crop functionality with various grid sizes
        logger.info("\nTesting pad/crop functionality:")
        test_grids = [
            [[1, 2, 3], [4, 5, 6]],  # Normal grid
            [[i for i in range(35)] for _ in range(35)],  # Oversized grid
            [],  # Empty grid
            [[1]],  # Single cell
        ]
        
        for i, grid in enumerate(test_grids):
            try:
                result = _pad_or_crop(grid, 30)
                height = len(result) if result else 0
                width = len(result[0]) if result and result[0] else 0
                logger.info(f"  Test grid {i+1}: {height}x{width} (within limits: {height <= 30 and width <= 30})")
            except Exception as e:
                logger.error(f"  Test grid {i+1} failed: {e}")
        
    except FileNotFoundError as e:
        logger.info(f"Real ARC files not found: {e}")
        logger.info("\nTo use this loader, download the ARC-AGI dataset:")
        logger.info("  1. Get arc-agi_training-challenges.json")
        logger.info("  2. Get arc-agi_training-solutions.json")
        logger.info("  3. Get arc-agi_evaluation-challenges.json")
        logger.info("  4. Get arc-agi_evaluation-solutions.json")
        logger.info("  5. Get arc-agi_test-challenges.json")
        logger.info("\nFrom: https://github.com/fchollet/ARC-AGI")
    
    logger.info("\n✅ ARC Dataset Loader ready!")
    logger.info("OK: Loader pads/crops any >30×30 grid and never crashes on shape.")


def create_submission(model, test_dataset: ARCDataset, output_file: str = "submission.json"):
    """
    Create a submission file for ARC Prize 2025
    
    Args:
        model: Trained NPARC model
        test_dataset: Test dataset
        output_file: Path to save submission JSON
    """
    import json
    
    submission = {}
    
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            demos, test_inputs, _, tid = test_dataset[i]
            
            task_predictions = []
            for test_input in test_inputs:
                # Run model inference with correct API
                grid, logits, size = model(demos, {"input": test_input})
                
                # Convert to list format (grid only)
                pred_list = grid.squeeze(0).cpu().numpy().tolist()
                task_predictions.append(pred_list)
            
            submission[tid] = task_predictions
    
    # Save submission
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)
    
    logger.info(f"Submission saved to {output_file}")
    logger.info(f"Total tasks: {len(submission)}")


class SyntheticGrammarDataset(Dataset):
    """
    Synthetic dataset for World Grammar pretraining.
    Generates random grids with simple patterns for training the encoder/painter rails.
    """
    
    def __init__(self, num_samples: int = 1000, max_grid_size: int = 30, 
                 num_colors: int = 10, device: str = "cpu", seed: int = 42):
        """
        Args:
            num_samples: Number of synthetic tasks to generate
            max_grid_size: Maximum grid dimension
            num_colors: Number of colors (0-9 typically for ARC)
            device: Device to place tensors on
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.device = device
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # MASTER'S LAZY-LOADING WISDOM: Only store metadata, generate on demand
        # No more CPU-burning pre-generation!
        
        logger.info(f"Lazy-loading dataset ready: {num_samples} synthetic grammar samples")
    
    def _generate_sample(self):
        """Generate a single synthetic sample with input/output grids"""
        # Random grid size
        h = np.random.randint(3, min(15, self.max_grid_size))
        w = np.random.randint(3, min(15, self.max_grid_size))
        
        # Choose a random transformation type
        transform_type = np.random.choice([
            'identity', 'rotate', 'flip', 'color_map', 
            'pattern_fill', 'border', 'symmetry'
        ])
        
        # Generate input grid
        if transform_type in ['identity', 'rotate', 'flip']:
            # Simple random grid
            input_grid = np.random.randint(0, self.num_colors, (h, w))
        elif transform_type == 'color_map':
            # Grid with fewer colors for mapping
            input_grid = np.random.randint(0, min(3, self.num_colors), (h, w))
        elif transform_type == 'pattern_fill':
            # Grid with a clear pattern
            input_grid = np.zeros((h, w), dtype=np.int32)
            pattern = np.random.randint(1, self.num_colors)
            input_grid[::2, ::2] = pattern
        elif transform_type == 'border':
            # Grid with border
            input_grid = np.zeros((h, w), dtype=np.int32)
            border_color = np.random.randint(1, self.num_colors)
            input_grid[0, :] = border_color
            input_grid[-1, :] = border_color
            input_grid[:, 0] = border_color
            input_grid[:, -1] = border_color
        else:  # symmetry
            # Half random, will be mirrored
            half_w = w // 2
            left_half = np.random.randint(0, self.num_colors, (h, half_w))
            input_grid = np.zeros((h, w), dtype=np.int32)
            input_grid[:, :half_w] = left_half
        
        # Apply transformation to get output
        if transform_type == 'identity':
            output_grid = input_grid.copy()
        elif transform_type == 'rotate':
            output_grid = np.rot90(input_grid)
        elif transform_type == 'flip':
            output_grid = np.flip(input_grid, axis=np.random.randint(0, 2))
        elif transform_type == 'color_map':
            output_grid = input_grid.copy()
            # Simple color remapping
            for old_color in range(min(3, self.num_colors)):
                new_color = (old_color + 1) % self.num_colors
                output_grid[input_grid == old_color] = new_color
        elif transform_type == 'pattern_fill':
            output_grid = input_grid.copy()
            # Fill empty spaces with another color
            fill_color = np.random.randint(1, self.num_colors)
            output_grid[output_grid == 0] = fill_color
        elif transform_type == 'border':
            output_grid = input_grid.copy()
            # Fill interior
            interior_color = np.random.randint(1, self.num_colors)
            output_grid[1:-1, 1:-1] = interior_color
        else:  # symmetry
            output_grid = input_grid.copy()
            # Mirror left half to right (handle odd widths)
            half_w = w // 2
            if w % 2 == 0:
                output_grid[:, half_w:] = np.flip(input_grid[:, :half_w], axis=1)
            else:
                output_grid[:, half_w+1:] = np.flip(input_grid[:, :half_w], axis=1)
        
        # Convert to tensors (use .copy() to avoid negative stride issues)
        input_tensor = torch.tensor(input_grid.copy(), dtype=torch.long, device=self.device)
        output_tensor = torch.tensor(output_grid.copy(), dtype=torch.long, device=self.device)
        
        # Add batch dimension
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)
        if output_tensor.dim() == 2:
            output_tensor = output_tensor.unsqueeze(0)
        
        return input_tensor, output_tensor, transform_type
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            demos: List with single (input, output) pair
            test_inputs: List with single input (same as demo input for simplicity)
            test_outputs: List with single output (same as demo output)
            task_id: String ID for the synthetic task
        """
        # MASTER'S LAZY-LOADING: Generate sample on the fly
        input_tensor, output_tensor, transform_type = self._generate_sample()
        
        # Format like ARCDataset for compatibility
        demos = [(input_tensor, output_tensor)]
        test_inputs = [input_tensor.clone()]
        test_outputs = [output_tensor.clone()]
        task_id = f"synthetic_{transform_type}_{idx}"
        
        return demos, test_inputs, test_outputs, task_id


if __name__ == "__main__":
    test_arc_loader()