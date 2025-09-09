#!/usr/bin/env python3
"""Test the complete DSL+EBR integration"""

import torch
import logging
from models.topas_arc_60M import TopasARC60M, ModelConfig

def test_dsl_ebr():
    print("🧪 Testing complete DSL+EBR integration...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    config = ModelConfig()
    config.pretraining_mode = True
    config.use_dsl_loss = True
    config.use_ebr = True
    config.lambda_dsl = 0.05
    config.verbose = False
    
    model = TopasARC60M(config).to(device)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test data
    test_grid = torch.randint(0, 10, (1, 5, 5), device=device)
    target_grid = torch.randint(0, 10, (1, 5, 5), device=device)
    
    print("🔧 Testing forward_pretraining with DSL+EBR...")
    
    try:
        # Test without HRM
        outputs = model.forward_pretraining(test_grid, target_shape=(5, 5))
        
        if outputs.get('logits') is not None:
            print(f"✅ Forward pass: logits {outputs['logits'].shape}")
            
            if 'losses' in outputs and outputs['losses']:
                for loss_name, loss_val in outputs['losses'].items():
                    print(f"✅ {loss_name}: {loss_val:.4f}")
            else:
                print("⚠️  No DSL losses generated")
        else:
            print("❌ Forward pass returned None logits")
            
        # Test EBR evaluation
        print("🔧 Testing EBR evaluation...")
        eval_outputs = model.evaluate_with_ebr(test_grid, target_grid)
        
        if 'exact_match_refined' in eval_outputs:
            print(f"✅ EBR evaluation: EM_refined={eval_outputs['exact_match_refined']:.2%}")
        else:
            print("❌ EBR evaluation failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("🎉 DSL+EBR integration test complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    test_dsl_ebr()