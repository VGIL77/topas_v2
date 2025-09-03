"""
Lightweight deep DSL program miner for Phase 4 augmentation
"""

import torch
import random
import itertools
from typing import List, Dict, Any
from models.dsl_search import DSLProgram, apply_program, CORE_OPS

def mine_deep_programs(task: Dict[str, Any], max_depth: int = 10) -> List[Dict[str, Any]]:
    """
    Lightweight deep DSL program miner.
    Returns rare but valid programs (6–10 ops).
    
    Args:
        task: Task with demos and test data
        max_depth: Maximum program depth to explore
        
    Returns:
        List of program dictionaries with operations and scores
    """
    programs = []
    
    try:
        input_grid = task.get("test", {}).get("input")
        target_grid = task.get("test", {}).get("output")
        
        if input_grid is None or target_grid is None:
            return programs
        
        # Generate deep programs through guided search
        target_attempts = 5  # Limit attempts for performance
        
        for attempt in range(target_attempts):
            # Generate program of 6-10 operations
            program_length = random.randint(6, min(max_depth, 10))
            
            ops = []
            params = []
            
            # Build program with some structure
            for step in range(program_length):
                # Use weighted selection favoring transformations
                if step < 2:
                    # Start with basic transforms
                    op = random.choice(["rotate90", "rotate180", "rotate270", "flip_h", "flip_v"])
                elif step < program_length - 2:
                    # Middle: more complex operations
                    op = random.choice(CORE_OPS)
                else:
                    # End: refinement operations
                    op = random.choice(["color_map", "crop_bbox", "outline"])
                
                ops.append(op)
                
                # Generate parameters
                if op == "color_map":
                    # More sophisticated color mapping
                    unique_colors = torch.unique(input_grid).tolist()
                    if len(unique_colors) >= 2:
                        # Map one color to another
                        old_color = random.choice(unique_colors)
                        new_color = random.choice([c for c in range(10) if c not in unique_colors])
                        params.append({"mapping": {old_color: new_color}})
                    else:
                        params.append({"mapping": {}})
                elif op == "translate":
                    # Bounded translation
                    max_shift = min(3, input_grid.shape[0] // 4)
                    params.append({
                        "dx": random.randint(-max_shift, max_shift),
                        "dy": random.randint(-max_shift, max_shift)
                    })
                elif op == "scale":
                    # Conservative scaling
                    params.append({
                        "fx": random.choice([1, 2, 3]),
                        "fy": random.choice([1, 2, 3])
                    })
                elif op == "flood_fill":
                    # Random flood fill parameters
                    H, W = input_grid.shape
                    params.append({
                        "start_pos": (random.randint(0, H-1), random.randint(0, W-1)),
                        "target_color": random.choice(torch.unique(input_grid).tolist()),
                        "fill_color": random.randint(1, 9)
                    })
                else:
                    params.append({})
            
            # Create and test program
            program = DSLProgram(ops=ops, params=params)
            
            try:
                result = apply_program(input_grid, program)
                
                # Score program based on target similarity
                if result.shape == target_grid.shape:
                    # Exact match bonus
                    exact_match = (result == target_grid).all().item()
                    if exact_match:
                        score = 1.0
                    else:
                        # Partial match scoring
                        intersection = (result == target_grid).sum().item()
                        union = result.numel()
                        score = intersection / union if union > 0 else 0.0
                else:
                    # Shape mismatch penalty but not zero
                    score = 0.01  # Small score for valid execution
                
                # Bonus for program complexity (rare deep programs)
                complexity_bonus = (program_length - 5) * 0.05  # Bonus for 6+ ops
                score += complexity_bonus
                
                # Only keep programs with reasonable success
                if score > 0.15:  # Higher threshold for deep programs
                    programs.append({
                        "program": ops,
                        "params": params,
                        "score": score,
                        "depth": program_length,
                        "result": result,
                        "exact_match": exact_match if 'exact_match' in locals() else False
                    })
                    
            except Exception:
                # Skip failed programs
                continue
    
    except Exception:
        # Return empty if any error
        pass
    
    # Sort by score and return best deep programs
    programs.sort(key=lambda x: x["score"], reverse=True)
    
    # Filter for truly deep programs (6+ ops) and return top 3
    deep_programs = [p for p in programs if p["depth"] >= 6]
    return deep_programs[:3]