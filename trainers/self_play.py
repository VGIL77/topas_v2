"""
Self-Play Buffer for surgical template-guided puzzle generation.
Uses WormholeTemplateMiner + ETS to create derived ARC puzzles.
"""

import torch
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import random
import logging

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
    
    def generate_batch(self, demos, wormhole=None, top_k=4):
        gen = []
        try:
            gen = self.generate_from_wormhole(demos, wormhole, themes=None, top_k=top_k)
        except Exception as e:
            logging.getLogger(__name__).warning("[SelfPlay] wormhole generation failed: %s", e)
            gen = []
        if not gen:
            # Try wormhole library templates if available
            if wormhole and hasattr(wormhole, "get_best_templates"):
                try:
                    templates = wormhole.get_best_templates(count=top_k)
                    for tpl in templates:
                        try:
                            # Apply template to the first demo input
                            demo = demos[0] if demos else None
                            if demo is not None:
                                inp, out_grid = demo if isinstance(demo, tuple) else (demo['input'], demo['output'])
                                grid = tpl.apply_to_grid(inp, dsl_ops=wormhole.library.templates)
                                gen.append((inp, grid))
                        except Exception:
                            continue
                except Exception as e:
                    logging.getLogger(__name__).warning("[SelfPlay] wormhole template fallback failed: %s", e)

        if not gen:
            # deterministic geometric/color fallback
            try:
                gen = self.fallback_from_demos(demos, top_k=top_k)
            except Exception as e:
                logging.getLogger(__name__).exception("[SelfPlay] fallback generation failed: %s", e)
                gen = []
        logging.getLogger(__name__).info("[SelfPlay] generate_batch produced=%d (wormhole=%s, templates=%s)",
                                         len(gen),
                                         'yes' if wormhole is not None else 'no',
                                         'yes' if wormhole and hasattr(wormhole, 'get_best_templates') else 'no')
        return gen[:top_k] if gen else []
    
    def sample_batch(self, n: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Sample n items from the buffer"""
        if len(self.buffer) == 0:
            return []
            
        n = min(n, len(self.buffer))
        samples = random.sample(list(self.buffer), n)
        self.sample_count += n
        
        return samples
    
    def fallback_from_demos(self, demos, top_k=4):
        """Simple deterministic transforms: rotate/flip color mapping to create new puzzles from demos."""
        out = []
        for demo in demos:
            try:
                inp, out_grid = demo if isinstance(demo, tuple) else (demo['input'], demo['output'])
                # create 2 transforms: rotate90 and flipud (ensure Kaggle-safe)
                t1_in = torch.rot90(inp, 1, [1,2])
                t1_out = torch.rot90(out_grid, 1, [1,2])
                out.append((t1_in, t1_out))
                t2_in = torch.flip(inp, dims=[1])
                t2_out = torch.flip(out_grid, dims=[1])
                out.append((t2_in, t2_out))
                if len(out) >= top_k:
                    break
            except Exception:
                continue
        return out[:top_k]