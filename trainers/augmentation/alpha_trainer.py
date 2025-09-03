"""
Lightweight Alpha self-play generator for Phase 1 augmentation
"""

import torch
import random
from typing import List, Dict, Any
from models.dsl_search import DSLProgram, apply_program, CORE_OPS

def generate_self_play_traces(task: Dict[str, Any], n_games: int = 2, depth: int = 4) -> List[Dict[str, Any]]:
    """
    Lightweight Alpha self-play generator.
    Returns a list of synthetic traces for distillation.
    
    Args:
        task: Task with demos and test data
        n_games: Number of self-play games to generate
        depth: Maximum depth for program search
        
    Returns:
        List of trace dictionaries with program and score
    """
    traces = []
    
    try:
        input_grid = task.get("test", {}).get("input")
        target_grid = task.get("test", {}).get("output")
        
        if input_grid is None or target_grid is None:
            return traces
        
        # Run lightweight self-play games
        for game in range(n_games):
            # Generate random program of given depth
            ops = []
            params = []
            
            for step in range(depth):
                # Random operation selection
                op = random.choice(CORE_OPS[:8])  # Use basic ops only
                ops.append(op)
                
                # Generate random parameters based on operation
                if op == "color_map":
                    # Random color mapping
                    colors = torch.unique(input_grid).tolist()
                    if len(colors) > 1:
                        old_color = random.choice(colors)
                        new_color = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                        params.append({"mapping": {old_color: new_color}})
                    else:
                        params.append({"mapping": {}})
                elif op == "translate":
                    params.append({"dx": random.randint(-2, 2), "dy": random.randint(-2, 2)})
                elif op == "scale":
                    params.append({"fx": random.choice([1, 2]), "fy": random.choice([1, 2])})
                else:
                    params.append({})
            
            # Create program and apply
            program = DSLProgram(ops=ops, params=params)
            
            try:
                result = apply_program(input_grid, program)
                
                # Score based on similarity to target (simple IoU)
                if result.shape == target_grid.shape:
                    intersection = (result == target_grid).sum().item()
                    union = result.numel()
                    score = intersection / union if union > 0 else 0.0
                else:
                    score = 0.0
                
                # Only keep traces with some success
                if score > 0.1:
                    traces.append({
                        "program": ops,
                        "params": params,
                        "score": score,
                        "result": result
                    })
                    
            except Exception:
                # Skip failed programs
                continue
    
    except Exception:
        # Return empty if any error
        pass
    
    # Sort by score and return best traces
    traces.sort(key=lambda x: x["score"], reverse=True)
    return traces[:n_games]  # Return top n_games traces