#!/usr/bin/env python3
"""
Quick smoke test for stabilized HRM-TOPAS system
"""

import torch
import logging
from models.topas_arc_60M import TopasARC60M, ModelConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

def test_stabilized():
    print("🧪 Testing stabilized HRM-TOPAS system...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create models
    topas_config = ModelConfig()
    topas_config.pretraining_mode = True
    topas_model = TopasARC60M(topas_config).to(device)
    
    hrm_config = {
        "batch_size": 1, "seq_len": 900, "vocab_size": 10,
        "num_puzzle_identifiers": 1000, "puzzle_emb_ndim": 128,
        "H_cycles": 3, "L_cycles": 4, "H_layers": 4, "L_layers": 4,
        "hidden_size": 512, "expansion": 3.0, "num_heads": 8,
        "pos_encodings": "rope", "halt_max_steps": 6,
        "halt_exploration_prob": 0.1, "forward_dtype": "bfloat16",
    }
    hrm_model = HierarchicalReasoningModel_ACTV1(hrm_config).to(device)
    
    print(f"✅ Models created: TOPAS {sum(p.numel() for p in topas_model.parameters()):,}, HRM {sum(p.numel() for p in hrm_model.parameters()):,}")
    
    # Test with various grid sizes
    test_grids = [
        torch.randint(0, 10, (1, 3, 3), device=device),   # Small
        torch.randint(0, 10, (1, 9, 9), device=device),   # Medium  
        torch.randint(0, 10, (1, 30, 30), device=device), # Large
    ]
    
    for i, test_grid in enumerate(test_grids):
        print(f"\n📏 Test {i+1}: Grid shape {test_grid.shape}")
        
        try:
            # Test without HRM
            outputs = topas_model.forward_pretraining(test_grid)
            if outputs.get('logits') is not None:
                logits = outputs['logits']
                print(f"  ✅ No HRM: logits {logits.shape}, finite={torch.isfinite(logits).all()}")
            else:
                print(f"  ⚠️  No HRM: logits=None (model detected issues)")
            
            # Test with HRM latents
            hrm_latents = torch.randn(1, 10, 512, device=device)  # [B, T, D]
            outputs = topas_model.forward_pretraining(test_grid, hrm_latents=hrm_latents)
            if outputs.get('logits') is not None:
                logits = outputs['logits']
                print(f"  ✅ With HRM: logits {logits.shape}, finite={torch.isfinite(logits).all()}")
            else:
                print(f"  ⚠️  With HRM: logits=None (model detected issues)")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n🎉 Stabilization test complete!")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_stabilized()