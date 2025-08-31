# Agent Development Guidelines - Dynamic Astrocyte Neural Network

## Project Context
- **Type:** Biologically-inspired neural network research
- **Language:** Pure Python (NumPy-based implementation)
- **Focus:** Astrocyte-modulated synaptic plasticity in artificial neural networks
- **Performance:** 91.5% accuracy vs 80.8% control (10.7 point improvement)
- **Data Scale:** 70K samples total (42K training, 28K test, 400 validation)

## Environment Setup

### Dependencies
```bash
# Core requirements (no virtual environment currently)
pip install numpy matplotlib jupyter

# For type checking and linting
pip install mypy flake8

# Optional for enhanced development
pip install ipython pandas scikit-learn
```

### Development Tools
- **Editor:** Compatible with Jupyter notebooks (`my_network.ipynb`)
- **Python Version:** 3.7+ (currently using system Python)
- **Memory:** Large datasets (123MB total) - monitor RAM usage
- **Session:** Supports long development sessions (20+ hours)

## Code Architecture

### Core Files
1. **`network.py`** - Base feedforward neural network class
2. **`ann.py`** - Single-threshold astrocyte implementation (primary)
3. **`dynamic_ann.py`** - Multi-threshold astrocyte variant (experimental)
4. **`my_network.ipynb`** - Jupyter notebook with experiments and visualizations

### Class Hierarchy
```python
Network (base class)
├── AstrocyteNetwork (ann.py) - Single threshold per astrocyte
└── AstrocyteNetwork (dynamic_ann.py) - Multiple thresholds per astrocyte
```

### Key Implementation Details
- **Inheritance:** Astrocyte classes extend base `Network` class
- **Method Overrides:** `forward_prop`, `backward_prop`, `update_params` modified for astrocyte dynamics
- **Parameter Management:** Astrocyte thresholds and effects are trainable via gradient descent
- **Activation Logic:** Astrocytes activate based on layer-wise mean activation patterns

## Development Standards

### Code Style
- **PEP 8 Compliance:** 88-character line limit preferred
- **Type Hints:** Add gradually for better maintainability
- **Docstrings:** Document astrocyte-specific methods thoroughly
- **Variable Naming:** Clear biological terminology (astrocyte_active, threshold, effect)

### Scientific Computing Practices
- **NumPy Convention:** Use vectorized operations, avoid loops where possible
- **Memory Efficiency:** Handle large datasets (74MB train.csv) efficiently
- **Numerical Stability:** Careful with softmax, ReLU implementations
- **Reproducibility:** Set random seeds for consistent results

### Neural Network Specifics
- **Xavier/He Initialization:** Continue using for weight initialization
- **Gradient Clipping:** Consider adding for training stability
- **Learning Rates:** Document optimal learning rates for astrocyte parameters
- **Batch Processing:** Current implementation handles single examples and batches

## Research Development Guidelines

### Experimental Workflow
1. **Hypothesis Formation:** Document biological motivation for changes
2. **Implementation:** Start with `ann.py` (simpler single-threshold)
3. **Testing:** Use `planar_flower.csv` for quick validation
4. **Scaling:** Test on full datasets (`train.csv`, `test.csv`)
5. **Analysis:** Record results in `results.txt`, visualize in notebook

### Biological Accuracy
- **Astrocyte Density:** Realistic ranges (0.1-1.0, typically 0.3-0.7 in brain)
- **Threshold Values:** Biologically plausible activation thresholds
- **Temporal Dynamics:** Future work - add time-dependent responses
- **Calcium Signaling:** Current mean-activation approximates calcium waves

### Performance Metrics
- **Primary:** Classification accuracy, precision, recall, F1-score
- **Astrocyte-Specific:** Track threshold evolution, activation patterns
- **Efficiency:** Monitor training time, memory usage with large datasets
- **Comparison:** Always compare against control network (base Network class)

## Common Development Tasks

### Adding New Astrocyte Mechanisms
```python
# Template for new astrocyte behaviors
def new_astrocyte_function(self, activation, parameters):
    """
    Implement new astrocyte dynamics
    
    Args:
        activation: Layer activation patterns
        parameters: Trainable astrocyte parameters
    
    Returns:
        Modified activation or weights
    """
    pass
```

### Training Loop Template
```python
# Standard training pattern
for epoch in range(epochs):
    # Forward pass
    predictions = network.forward_prop(X_batch)
    
    # Backward pass (returns gradients for all parameters)
    dW, dB, dT, dE = network.backward_prop(X_batch, Y_batch)
    
    # Update parameters (including astrocyte parameters)
    network.update_params(learning_rate, dW, dB, dT, dE)
    
    # Log astrocyte activity if needed
    if epoch % 100 == 0:
        log_astrocyte_stats(network)
```

### Debugging Astrocyte Behavior
- **Activation Patterns:** Print `astrocyte_active` arrays to verify triggering
- **Weight Modulation:** Check weight change magnitudes (should be modest)
- **Gradient Flow:** Ensure astrocyte parameters receive gradients
- **Threshold Evolution:** Plot threshold changes during training

## Dataset Guidelines

### Data Handling
- **Loading:** Use NumPy's `loadtxt` or `genfromtxt` for CSV files
- **Preprocessing:** Normalize features for stable training
- **Splitting:** Current split: train (42K), test (28K), validation (400)
- **Memory:** Monitor usage with large files - consider batch loading if needed

### Data Format
```python
# Expected CSV format: y,x1,x2
# y: class label (0 or 1)
# x1, x2: input features (continuous values)
```

## Testing and Validation

### Unit Testing
```bash
# Run basic network tests
python -c "from ann import AstrocyteNetwork; net = AstrocyteNetwork([2,3,2]); print('OK')"

# Check gradient computation
python -c "from network import Network; import numpy as np; net = Network([2,3,2], True); 
X = np.random.randn(2,10); Y = np.random.randint(0,2,(1,10)); 
dW, dB = net.backward_prop(X,Y); print('Gradients OK')"
```

### Performance Validation
- **Baseline:** Always compare against base `Network` class performance
- **Convergence:** Monitor loss curves, astrocyte parameter evolution
- **Generalization:** Test on unseen data, avoid overfitting to astrocyte dynamics
- **Statistical Significance:** Multiple runs with different seeds

## Known Issues and Solutions

### Current Type Errors (from project diagnostics)
- **Import Errors:** NumPy/Matplotlib not found by type checker (install or configure path)
- **Method Override Mismatches:** Astrocyte classes return additional gradients (4-tuple vs 2-tuple)
- **Parameter Count Mismatch:** `update_params` has extra parameters for astrocyte updates

### Fixes Needed
```python
# Type hint fixes for method overrides
def backward_prop(self, X, Y) -> tuple[list, list, list, list]:
    # Returns (dW, dB, dT, dE) instead of just (dW, dB)
    
def update_params(self, learning_rate, dW, dB, dT, dE, momentum=0.9):
    # Additional parameters for astrocyte gradients
```

## Performance Optimization

### Memory Usage
- **Large Datasets:** Current 123MB total - fits in RAM but monitor usage
- **Batch Processing:** Implement mini-batch training for memory efficiency
- **Gradient Accumulation:** For very large datasets

### Computational Efficiency
- **Vectorization:** Ensure all operations use NumPy vectorized functions
- **Caching:** Store intermediate calculations during forward/backward passes
- **Profile:** Use `cProfile` to identify bottlenecks

## Future Research Directions

### Biological Enhancements
- **Temporal Dynamics:** Add time-dependent astrocyte responses
- **Spatial Organization:** Implement astrocyte-neuron topological relationships
- **Gliotransmitter Release:** Model chemical signaling mechanisms
- **Network Oscillations:** Explore astrocyte contributions to neural rhythms

### Machine Learning Applications
- **Multi-Task Learning:** Apply astrocyte dynamics to different domains
- **Transfer Learning:** Investigate astrocyte parameter transfer
- **Hyperparameter Optimization:** Systematic search of astrocyte parameters
- **Comparison Studies:** Benchmark against other bio-inspired approaches

Remember: This is research code - prioritize experimentation and biological plausibility while maintaining scientific rigor.