# Examples

- This module contains 4 files: `__init__.py`, `binary_classification.py`, `multiclass_classification.py`, `regression.py`
- `__init__.py`: used define `examples` module

### Regression Example Architecture
```
layers: [
    Input(10),
    Dense(16),
    ReLU(),
    Dense(8),
    ReLU(),
    Dense(1)
]

loss: MSE

optimizer: Nesterov Accelerated Gradient Descent
lr: 0.05
beta: 0.1
batch_size: 8
epochs: 50
```

- Dataset: scikit-learn diabetes dataset
- Evaluation: MSE and $R^2$
- Run instructions: `python -m examples.regression` (from project root)

### Binary Classification Example Architecture
```
layers: [
    Input(30),
    Dense(16),
    ReLU(),
    Dense(8),
    ReLU(),
    Dense(1),
    Sigmoid()
]

loss: BCE

optimizer: RMSProp
learning rate: 0.05
beta: 0.2
batch size: 16
epochs: 30
```
- Dataset: scikit-learn Breast Cancer Wisconsin dataset
- Evaluation: Accuracy and classification report
- Run instructions: `python -m examples.binary_classification` (from project root)

### Multi-class Classification Example Architecture
```
layers: [
    Input(784),
    Dense(64),
    ReLU(),
    Dense(10),
    Softmax()
]

loss: CCE

optimizer: Adam
learning rate: 0.001
beta1: 0.9
beta2: 0.999
batch size: 16
epochs: 50
```

- Dataset: MNIST (5k samples sub-set)
- Evaluation: Accuracy
- Run instructions: `python -m examples.multiclass_classification` (from project root)