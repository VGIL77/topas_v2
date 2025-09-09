"""
Self-Play Buffer for surgical template-guided puzzle generation.
Uses WormholeTemplateMiner + ETS to create derived ARC puzzles.
"""

import torch
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import random

class SelfPlayBuffer:
    """Buffer for storing and sampling self-play generated puzzles"""
    
    def __init__(self, maxlen: int = 200):
        self.buffer = deque(maxlen=maxlen)
        self.generation_count = 0
        self.sample_count = 0
        
    def generate_from_wormhole(self, demos: List[Tuple[torch.Tensor, torch.Tensor]], 
                              wormhole, themes=None, top_k: int = 3) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate template-guided puzzles from existing demos using wormhole mining.
        
        Args:
            demos: List of (input, target) pairs from training data
            wormhole: WormholeTemplateMiner instance
            themes: EmergentThemeSynthesis instance (optional)
            top_k: Maximum number of puzzles to generate
            
        Returns:
            List of (input, target) transformed puzzle pairs
        """
        generated = []
        
        if not demos or not wormhole:
            return generated
            
        # Sample demos for transformation
        sampled_demos = random.sample(demos, min(len(demos), top_k))
        
        for input_grid, target_grid in sampled_demos:
            try:
                # Extract template from the demo pair
                if hasattr(wormhole, 'extract_template'):
                    template = wormhole.extract_template(input_grid, target_grid)
                else:
                    # Fallback: simple transformations
                    template = {'transform': 'rotate', 'params': {}}
                
                # Apply transformations based on template
                if template.get('transform') == 'rotate':
                    # 90-degree rotation
                    transformed_input = torch.rot90(input_grid, k=1, dims=[-2, -1])
                    transformed_target = torch.rot90(target_grid, k=1, dims=[-2, -1])
                elif template.get('transform') == 'flip':
                    # Horizontal flip
                    transformed_input = torch.flip(input_grid, dims=[-1])
                    transformed_target = torch.flip(target_grid, dims=[-1])
                else:
                    # Color shift (simple transformation)
                    shift = random.randint(1, 9)
                    transformed_input = (input_grid + shift) % 10
                    transformed_target = (target_grid + shift) % 10
                
                generated.append((transformed_input, transformed_target))
                self.generation_count += 1
                
                if len(generated) >= top_k:
                    break
                    
            except Exception as e:
                continue  # Skip failed transformations
                
        # Add to buffer
        self.buffer.extend(generated)
        
        return generated
    
    def sample_batch(self, n: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample n items from the buffer"""
        if len(self.buffer) == 0:
            return []
            
        n = min(n, len(self.buffer))
        samples = random.sample(list(self.buffer), n)
        self.sample_count += n
        
        return samples