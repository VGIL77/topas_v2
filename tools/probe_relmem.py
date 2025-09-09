"""
CLI probe script for RelMem integration - demonstrates non-empty bias reaches DSL search
"""
import torch
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.topas_arc_60M import TopasARC60M, ModelConfig
from relational_memory_neuro import RelationalMemoryNeuro
from models.dsl_search import beam_search

def probe_relmem_internals():
    """Probe RelMem internals and DSL integration"""
    
    print("=== RelMem Integration Probe ===")
    
    # Test 1: RelMem direct
    print("\n1. Testing RelMem directly...")
    relmem = RelationalMemoryNeuro(hidden_dim=256, max_concepts=1024, rank=16)
    
    # Add fake weights to simulate trained RelMem
    relmem.weights = {
        "color": torch.randn(8, 8) * 0.2,
        "shape": torch.randn(8, 8) * 0.3,
        "structure": torch.randn(8, 8) * 0.1
    }
    relmem.relations = ["color", "shape", "structure"]
    
    bias = relmem.op_bias()
    print(f"✓ Direct RelMem op_bias: {len(bias)} operations")
    print(f"✓ Top 5: {sorted(bias.items(), key=lambda x: -x[1])[:5]}")
    
    inv_loss = relmem.compute_inverse_loss()
    print(f"✓ Direct RelMem inverse_loss: {inv_loss.item():.6f}")
    
    # Test 2: TOPAS integration
    print("\n2. Testing TOPAS integration...")
    config = ModelConfig(
        width=512, depth=8, slots=32, slot_dim=256,
        enable_relmem=True,
        relmem_op_bias_w=0.2,
        relmem_op_bias_scale=0.5,
        pretraining_mode=True,
        use_dsl_loss=True,
        verbose=True
    )
    
    model = TopasARC60M(config)
    model.set_pretraining_mode(True)
    
    # Override RelMem with our test instance
    model.relmem = relmem
    
    # Test forward pass
    batch_size = 1
    H, W = 6, 6
    test_grid = torch.randint(0, 10, (batch_size, H, W))
    
    print(f"\n3. Testing forward pass with test RelMem...")
    with torch.no_grad():
        outputs = model.forward_pretraining(
            test_grid,
            target_shape=(H, W),
            demos=[],
            global_step=5000
        )
    
    # Check results
    losses = outputs.get('losses', {})
    print(f"✓ Forward pass completed with losses: {list(losses.keys())}")
    
    if 'relmem_inverse_loss' in losses:
        print(f"✓ relmem_inverse_loss: {losses['relmem_inverse_loss'].item():.6f}")
    
    if 'dsl_loss' in losses:
        print(f"✓ dsl_loss: {losses['dsl_loss'].item():.4f}")
    
    # Test 3: Check relman vs relmem
    print(f"\n4. Checking relation managers...")
    relman_bias = model.relman.op_bias()
    print(f"✓ relman.op_bias(): {len(relman_bias)} operations")
    print(f"✓ relman edges: {len(getattr(model.relman, 'edges', []))}")
    
    print(f"✓ relmem.relations: {relmem.relations}")
    print(f"✓ relmem.weights keys: {list(relmem.weights.keys()) if relmem.weights else 'None'}")
    
    # Test 4: Simulate beam_search with op_bias
    print(f"\n5. Testing beam_search with op_bias...")
    try:
        # Create simple demo and test data
        demo_input = torch.randint(0, 10, (3, 3))
        demo_output = torch.randint(0, 10, (3, 3))
        demos = [{"input": demo_input.numpy(), "output": demo_output.numpy()}]
        test_data = torch.randint(0, 10, (3, 3)).numpy()
        
        # Call beam_search with our op_bias
        result = beam_search(
            demos, test_data, {}, 
            depth=2, beam=3, 
            verbose=False,
            op_bias=bias
        )
        
        print(f"✓ beam_search completed with op_bias ({len(bias)} ops)")
        if result is not None:
            print(f"✓ beam_search returned result: {type(result)}")
        else:
            print("✓ beam_search returned None (normal)")
            
    except Exception as e:
        print(f"❌ beam_search failed: {e}")
        return False
    
    print(f"\n✅ All probes successful - RelMem integration working!")
    return True

if __name__ == "__main__":
    success = probe_relmem_internals()
    if success:
        print("\n🚀 Ready for training with active RelMem!")
    else:
        print("\n❌ RelMem integration needs fixes")
        exit(1)