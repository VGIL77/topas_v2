#!/usr/bin/env python3
"""
Smoke test for EBR/Dream/SelfPlay integration
"""

import logging
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    try:
        from models.topas_arc_60M import TopasARC60M, ModelConfig
        from dream_engine import DreamEngine, DreamConfig
        from trainers.self_play import SelfPlayBuffer
        
        # Test 1: DreamEngine trainability
        print("Testing DreamEngine parameters...")
        dcfg = DreamConfig(state_dim=64, action_dim=41, device='cpu')
        dream = DreamEngine(dcfg)
        dream.attach_relmem(None)
        
        params = list(dream.parameters())
        assert len(params) > 0, f"DreamEngine has no parameters: {len(params)}"
        print(f"✅ DreamEngine has {len(params)} trainable parameters")
        
        # Test train_step
        slot_vecs = torch.randn(2, 32, 64, requires_grad=True)
        loss = dream.train_step(slot_vecs)
        assert loss.requires_grad, "train_step loss has no gradients"
        assert abs(loss.item()) > 1e-6, "train_step loss is zero"
        print(f"✅ DreamEngine train_step works: loss={loss.item():.4f}")
        
        # Test 2: Model with EBR
        print("Testing EBR integration...")
        config = ModelConfig()
        config.enable_dream = True
        config.use_ebr = True
        config.verbose = False
        config.width = 64
        config.depth = 2
        config.slots = 4
        config.slot_dim = 64
        
        model = TopasARC60M(config)
        model.eval()
        
        # Create dummy batch
        demo_input = torch.randint(0, 10, (3, 3))
        demo_output = torch.randint(0, 10, (3, 3))
        demos = [{'input': demo_input, 'output': demo_output}]
        test = {'input': torch.randint(0, 10, (1, 3, 3))}
        
        with torch.no_grad():
            _ = model(demos, test)
        print("✅ EBR integration works - no NameError")
        
        # Test 3: Self-play
        print("Testing self-play...")
        buffer = SelfPlayBuffer()
        result = buffer.generate_batch([(demo_input, demo_output)], None, top_k=2)
        print(f"✅ SelfPlay generate_batch returned {len(result)} items")
        
        print("SMOKE_OK")
        return 0
        
    except Exception as e:
        print("SMOKE_FAIL", e)
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())