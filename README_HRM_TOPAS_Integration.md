# HRM-TOPAS Integrated System

A neurosymbolic AI system for solving Abstraction and Reasoning Corpus (ARC) tasks that combines Hierarchical Reasoning Memory (HRM) with TOPAS (Transformation Operations Pattern Analysis System) to achieve the ARC Challenge goal of 85% accuracy.

## Overview

This integrated system leverages:
- **HRM**: Hierarchical reasoning with adaptive halting and puzzle embeddings
- **TOPAS**: Grid transformation operations and symbolic reasoning
- **Neurosymbolic Bridge**: Bidirectional communication between neural and symbolic components
- **Multi-Phase Training**: Progressive curriculum learning across 11 specialized phases

## Key Features

- **Bidirectional Integration**: HRM's hierarchical reasoning guides TOPAS DSL operations
- **Adaptive Search Control**: HRM halting mechanisms optimize DSL search depth
- **Puzzle Embedding Integration**: HRM embeddings enhance TOPAS grid encoding
- **Cross-Attention Mechanisms**: Neural attention between HRM states and TOPAS features
- **Robust Training Pipeline**: 11-phase progressive training with error recovery
- **Comprehensive Monitoring**: Real-time metrics, performance tracking, and visualization
- **CUDA Optimization**: Mixed precision training and GPU memory management

## Performance Targets

- **Minimum Baseline**: 50% accuracy on ARC evaluation set
- **Target Goal**: 85% accuracy (ARC Challenge winning threshold)
- **Training Efficiency**: <48 hours on single A100 GPU
- **Memory Requirements**: <40GB GPU memory

## Quick Start

### 1. Environment Setup

```bash
# Create Python virtual environment
python -m venv venv_cuda
source venv_cuda/bin/activate  # Linux/Mac
# venv_cuda\Scripts\activate  # Windows

# Install PyTorch with CUDA support
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### 2. Download ARC Dataset

```bash
# Download ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI.git
# Or place your ARC dataset in ARC-AGI/data/
```

### 3. Setup HRM Models

```bash
# Clone HRM repository (if not already included)
cd docs/
git clone https://github.com/institution/HRM-main.git
cd ..
```

### 4. Configuration

The system uses `configs/hrm_integrated_training.json` for configuration. Key settings:

- **Device**: Set to "cuda" for GPU training, "cpu" for CPU
- **Batch Size**: Adjust based on available GPU memory (default: 8)
- **Data Paths**: Update paths to your ARC dataset
- **HRM Settings**: Configure HRM model parameters
- **Phase Settings**: Customize training phases

### 5. Training

#### Full Training Pipeline

```bash
python train_hrm_topas.py --config configs/hrm_integrated_training.json
```

#### Configuration Validation

```bash
python train_hrm_topas.py --config configs/hrm_integrated_training.json --validate-config
```

#### Forward Pass Testing

```bash
python train_hrm_topas.py --config configs/hrm_integrated_training.json --test-forward
```

#### Resume from Checkpoint

```bash
python train_hrm_topas.py --config configs/hrm_integrated_training.json --resume
```

## System Architecture

### Core Components

1. **HRM Model** (`docs/HRM-main/`)
   - Hierarchical reasoning with H-level (slow, abstract) and L-level (fast, computation)
   - Adaptive halting mechanism
   - Puzzle embeddings for task-specific reasoning

2. **TOPAS Model** (`models/topas_arc_60M.py`)
   - Grid encoder for spatial patterns
   - DSL operation generator
   - Transformation predictor

3. **HRM-TOPAS Bridge** (`models/hrm_topas_bridge.py`)
   - Cross-attention layers
   - DSL operation selector guided by HRM
   - Puzzle embedding integrator
   - Bidirectional feature alignment

### Training Phases

The system uses an 11-phase progressive training approach:

0. **World Grammar** - Basic grid understanding and painter operations
1. **Policy Distillation** - Learn from expert demonstrations with HRM guidance
2. **Meta Learning** - MAML-based fast adaptation with HRM embeddings
3. **Self Critique** - Generate and learn from counterexamples
4. **MCTS Alpha** - Tree search with HRM-guided exploration
5. **Dream Scaled** - Large-scale synthetic data generation
6. **Neuro Priors** - Neuromorphic priors integration
7. **RelMem** - Relational memory with Hebbian learning
8. **SGI Optimizer** - Sharpness-aware optimization
9. **Ensemble Solver** - Multi-strategy solution combination
10. **Production** - Final fine-tuning for deployment

## Testing and Validation

### Forward Pass Testing

```bash
python test_hrm_forward_pass.py --config configs/hrm_integrated_training.json
```

Tests:
- HRM model forward pass with ARC data
- TOPAS grid encoding and DSL generation
- HRM-TOPAS bridge integration
- Tensor shape compatibility
- CUDA memory usage

### Phase Transition Testing

```bash
python test_phase_transitions.py --config configs/hrm_integrated_training.json
```

Tests:
- Smooth transitions between training phases
- Model state preservation
- Checkpoint loading/saving
- Curriculum progression
- Error recovery

### ARC Dataset Validation

```bash
python validate_arc.py --config configs/hrm_integrated_training.json
```

Tests:
- ARC dataset loading and parsing
- Grid size and format compatibility
- Task demonstration validity
- Performance against baseline targets

## Monitoring and Evaluation

### Real-time Training Monitoring

```bash
python monitor_training.py --log-dir logs/hrm_integrated_v1
```

Features:
- Live metrics dashboard
- HRM-specific metrics tracking
- Resource usage monitoring
- Phase transition visualization
- Error detection and alerting

### Checkpoint Evaluation

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/hrm_integrated_v1/latest_checkpoint.pt
```

Features:
- Load and evaluate saved checkpoints
- Performance comparison across phases
- ARC evaluation set testing
- Detailed accuracy reports

## Directory Structure

```
topas_v2/
├── configs/
│   ├── hrm_integrated_training.json    # Main training configuration
│   └── *.json                          # Other configurations
├── models/
│   ├── hrm_topas_bridge.py            # HRM-TOPAS integration
│   ├── topas_arc_60M.py               # Main TOPAS model
│   └── *.py                           # Other model components
├── trainers/
│   ├── phases/                        # Training phase implementations
│   ├── train_arc_prize.py             # Main training pipeline
│   └── *.py                           # Training utilities
├── docs/
│   └── HRM-main/                      # HRM model code
├── ARC-AGI/
│   └── data/                          # ARC dataset
├── checkpoints/                       # Model checkpoints
├── logs/                             # Training logs
├── train_hrm_topas.py                # Main training launcher
├── test_hrm_forward_pass.py          # Forward pass testing
├── test_phase_transitions.py         # Phase transition testing
├── validate_arc.py                   # ARC dataset validation
├── monitor_training.py               # Training monitoring
├── evaluate_checkpoint.py            # Checkpoint evaluation
└── requirements.txt                  # Python dependencies
```

## Configuration Options

### Global Settings

```json
{
  "global": {
    "device": "cuda",                    # Training device
    "log_dir": "logs/hrm_integrated_v1/",
    "checkpoint_dir": "checkpoints/hrm_integrated_v1/",
    "epochs": 120,                       # Total training epochs
    "batch_size": 8,                     # Batch size
    "use_amp": true,                     # Mixed precision training
    "eval_interval": 2,                  # Evaluation frequency
    "save_checkpoint_interval": 5        # Checkpoint save frequency
  }
}
```

### HRM Configuration

```json
{
  "hrm_config": {
    "batch_size": 1,
    "seq_len": 400,                      # Sequence length
    "vocab_size": 10,                    # Vocabulary size
    "hidden_size": 512,                  # Hidden dimension
    "H_cycles": 3,                       # H-level cycles
    "L_cycles": 4,                       # L-level cycles
    "halt_max_steps": 6,                 # Maximum halting steps
    "puzzle_emb_ndim": 128,              # Puzzle embedding dimension
    "learning_rate": 1e-4,
    "weight_decay": 1.0
  }
}
```

### Model Configuration

```json
{
  "model_config": {
    "slot_dim": 128,                     # TOPAS slot dimension
    "dsl_vocab_size": 64,                # DSL vocabulary size
    "use_dsl": true,                     # Enable DSL operations
    "use_ebr": true,                     # Enable EBR reasoning
    "use_relations": true                # Enable relational reasoning
  }
}
```

## Performance Optimization

### CUDA Settings

- **Mixed Precision**: Enabled by default for A100/V100 GPUs
- **Memory Management**: Dynamic GPU memory allocation
- **Batch Size**: Auto-adjusted based on available GPU memory
- **Gradient Accumulation**: Supported for larger effective batch sizes

### Memory Requirements

| Component | Memory Usage |
|-----------|--------------|
| TOPAS Model | ~4GB |
| HRM Model | ~8GB |
| HRM-TOPAS Bridge | ~2GB |
| Training Overhead | ~10GB |
| **Total Recommended** | **32GB+ GPU memory** |

### Performance Tuning

1. **Batch Size**: Start with 8, reduce if OOM
2. **Sequence Length**: Reduce HRM seq_len if memory constrained
3. **Mixed Precision**: Keep enabled for modern GPUs
4. **Checkpoint Interval**: Balance between safety and performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   # Set use_amp to true
   # Reduce sequence length in HRM config
   ```

2. **Import Errors**
   ```bash
   python fix_integration_issues.py --check-only
   python fix_integration_issues.py --all-fixes
   ```

3. **HRM Models Not Found**
   ```bash
   # Ensure docs/HRM-main/ directory exists
   # Check HRM model files are present
   ```

4. **ARC Dataset Issues**
   ```bash
   python validate_arc.py --config configs/hrm_integrated_training.json
   ```

### Performance Issues

1. **Slow Training**
   - Check CUDA is properly installed
   - Verify GPU utilization with `nvidia-smi`
   - Enable mixed precision training

2. **Low Accuracy**
   - Ensure all 11 phases complete successfully
   - Check curriculum progression is enabled
   - Validate HRM integration is active

3. **Memory Leaks**
   - Monitor memory usage with training monitor
   - Reduce batch size if memory keeps growing
   - Check for gradient accumulation issues

### Debugging

1. **Enable Debug Logging**
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   python train_hrm_topas.py --config configs/hrm_integrated_training.json
   ```

2. **Check Integration**
   ```bash
   python test_hrm_forward_pass.py --config configs/hrm_integrated_training.json
   ```

3. **Validate Configuration**
   ```bash
   python train_hrm_topas.py --config configs/hrm_integrated_training.json --validate-config
   ```

## Expected Results and Milestones

### Training Progress Milestones

| Phase | Expected Accuracy | Key Metrics |
|-------|------------------|-------------|
| Phase 0-2 | 15-25% | Basic pattern recognition |
| Phase 3-5 | 30-45% | Reasoning and planning |
| Phase 6-8 | 45-65% | Advanced pattern synthesis |
| Phase 9-10 | 65-85% | Ensemble and production |

### Performance Monitoring

- **Training Speed**: 10-50 samples/sec (A100 GPU)
- **Memory Usage**: Should stabilize around 30GB
- **Phase Transitions**: Should be smooth without accuracy drops
- **Convergence**: Each phase should show decreasing loss

### Final Validation

Upon completion, the system should achieve:
- **Minimum**: 50% accuracy on ARC evaluation set
- **Target**: 85% accuracy (ARC Challenge goal)
- **Consistency**: <5% variance across multiple runs
- **Efficiency**: <48 hours total training time

## Development and Contribution

### Code Structure

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Document all new components
- Include unit tests for new features

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python test_hrm_forward_pass.py
python test_phase_transitions.py
python validate_arc.py
```

### Adding New Features

1. Create new modules in appropriate directories
2. Update configuration files as needed
3. Add tests for new functionality
4. Update documentation

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hrm_topas_2024,
  title={HRM-TOPAS: Neurosymbolic Integration for ARC Challenge},
  author={Research Team},
  year={2024},
  url={https://github.com/yourrepo/topas_v2}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support and Contact

For issues, questions, or contributions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review existing issues for similar problems

---

**Note**: This system represents cutting-edge research in neurosymbolic AI. Performance may vary based on hardware, data, and configuration. For production use, thorough testing and validation is recommended.