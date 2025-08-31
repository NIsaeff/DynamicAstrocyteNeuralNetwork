# Dynamic Astrocyte Neural Network

A biologically-inspired neural network implementation that incorporates astrocyte-like dynamics to enhance learning performance. This project explores how astrocyte cells' modulatory effects on synaptic plasticity can be simulated to improve artificial neural network capabilities.

## Overview

Traditional artificial neural networks lack the dynamic regulatory mechanisms found in biological brains. Astrocytes, the most abundant glial cells in the brain, play crucial roles in synaptic modulation, neural plasticity, and information processing. This implementation integrates astrocyte-inspired dynamics into feedforward neural networks.

### Key Features

- **Astrocyte-Modulated Weights**: Dynamic weight modification based on layer activation patterns
- **Adaptive Thresholds**: Trainable astrocyte activation thresholds for each network layer
- **Multi-Threshold Architecture**: Support for multiple threshold levels per astrocyte (research variant)
- **Biological Inspiration**: Mimics astrocyte calcium signaling and synaptic modulation
- **Performance Enhancement**: Demonstrated ~13% accuracy improvement over control networks

## Architecture

### Core Components

1. **Base Network** (`network.py`): Standard feedforward network with ReLU activation and softmax output
2. **Astrocyte Network** (`ann.py`): Single-threshold astrocyte implementation 
3. **Dynamic Astrocyte Network** (`dynamic_ann.py`): Multi-threshold research variant

### Astrocyte Mechanism

```python
# Astrocyte activation based on layer average activation
astrocyte_active = (mean_layer_activation > threshold).astype(float)

# Dynamic weight modulation
modified_weights = weights * (1 + astrocyte_active * effect_strength)
```

The astrocyte system monitors layer-wise activation patterns and dynamically modulates connection weights when activation exceeds learned thresholds.

## Results

Performance comparison on classification tasks:

| Network Type | Accuracy | Precision | Recall | F1 Score |
|-------------|----------|-----------|---------|----------|
| Control Network | 80.80% | 80.78% | 80.83% | 80.71% |
| Astrocyte Network | **91.50%** | **91.73%** | **91.60%** | **91.53%** |

**Improvement**: ~10.7 percentage point accuracy increase with astrocyte dynamics.

## Datasets

### Available Datasets
- **planar_flower.csv** (400 samples): Small 2D classification dataset for initial testing
- **train.csv** (42K samples, 74MB): Large-scale training data
- **test.csv** (28K samples, 49MB): Evaluation dataset

### Data Format
```csv
y,x1,x2
0,-1.035531873,-0.912076121
1,1.44089627,-0.801500713
```
- `y`: Class label (0 or 1 for binary classification)
- `x1, x2`: Input features

## Usage

### Basic Training
```python
from ann import AstrocyteNetwork
import numpy as np

# Initialize network
network = AstrocyteNetwork(
    sizes=[2, 10, 10, 2],           # Architecture: 2 inputs, 2 hidden layers, 2 outputs
    astrocyte_density=1.0,          # Astrocyte coverage (0-1)
    initial_threshold=0.5,          # Starting activation threshold
    initial_effect=0.1              # Initial modulation strength
)

# Training loop
for epoch in range(epochs):
    predictions = network.forward_prop(X_train)
    network.backward_prop(X_train, Y_train)
    network.update_parameters(learning_rate)
```

### Multi-Threshold Variant
```python
from dynamic_ann import AstrocyteNetwork

# Advanced astrocyte with multiple activation thresholds
network = AstrocyteNetwork(
    sizes=[2, 10, 10, 2],
    num_thresholds=3,               # Multiple threshold levels
    initial_threshold=0.5,
    initial_effect=0.1
)
```

## Implementation Details

### Network Architecture
- **Input Layer**: Flexible input dimensions
- **Hidden Layers**: ReLU activation with astrocyte modulation
- **Output Layer**: Softmax for multi-class classification
- **Weight Initialization**: Xavier/He initialization for stable training

### Astrocyte Parameters
- **Density**: Controls astrocyte coverage per layer (0-1 range)
- **Thresholds**: Learnable activation thresholds (trainable via backpropagation)
- **Effects**: Modulation strength parameters (trainable)
- **Activation**: Based on layer-wise mean activation levels

### Training Features
- **Gradient Descent**: Standard backpropagation with astrocyte parameter updates
- **Loss Function**: Cross-entropy for classification tasks
- **Metrics**: Accuracy, precision, recall, F1-score evaluation

## File Structure

```
DynamicAstrocyteNeuralNetwork/
├── README.md              # This file
├── AGENTS.md             # Development guidelines
├── network.py            # Base neural network implementation
├── ann.py                # Single-threshold astrocyte network
├── dynamic_ann.py        # Multi-threshold astrocyte network (research)
├── my_network.ipynb      # Jupyter notebook with experiments
├── planar_flower.csv     # Small test dataset (400 samples)
├── train.csv            # Training data (42K samples, 74MB)
├── test.csv             # Test data (28K samples, 49MB)
├── results.txt          # Performance comparison results
└── __pycache__/         # Python bytecode cache
```

## Research Background

This implementation is inspired by recent neuroscience research on astrocyte functions:

1. **Synaptic Modulation**: Astrocytes release gliotransmitters that modulate synaptic strength
2. **Calcium Signaling**: Astrocyte calcium waves correspond to network activation patterns  
3. **Plasticity Enhancement**: Astrocytes facilitate long-term potentiation and synaptic plasticity
4. **Network Synchronization**: Astrocytes help coordinate neural network dynamics

The artificial astrocyte mechanism captures these biological principles through:
- Activity-dependent activation (calcium signaling analog)
- Dynamic weight modulation (synaptic strength changes)
- Adaptive thresholds (plasticity mechanisms)

## Future Directions

- **Temporal Dynamics**: Implement time-dependent astrocyte responses
- **Network Topology**: Explore astrocyte-neuron connectivity patterns
- **Multi-Modal Learning**: Apply to different ML tasks (regression, unsupervised)
- **Biological Validation**: Compare with experimental astrocyte data
- **Scalability**: Test on larger networks and datasets

## Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- Jupyter (for notebook experiments)

## License

Open source research project. Feel free to use and modify for research purposes.