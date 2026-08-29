# Rekas

`rekas` is the keras equivalent of FlensorTow

It implements the core components required to construct and train simple
feed-forward neural networks using only Python and NumPy.


## Components

### Templates
- `Node`
- `Layer`
- `Loss`
- `Activation`
- `Optimizer`

### Layers

Currently supported:

- `Input`
- `Dense`

Each layer implements both a forward pass and its derivative for
backpropagation.
---

### Activation Functions

- `ReLU`
- `Sigmoid`
- `Softmax`

Each activation implements both a forward pass and its derivative for
backpropagation.

---

### Loss Functions

- `MSE`:  Mean Squared Error
- `BCE`: Binary Cross Entropy
- `CCE`: Categorical Cross Entropy

The loss function provides both the loss value and the gradient passed into
the backward pass.


## Optimizers

The following optimization algorithms have been implemented from their
underlying update equations:

- `SGD`
- `MiniBatch`
- `Nesterov`
- `RMSProp`
- `Adam`

## Sequential API

Models are constructed using the `Sequential` class.

```python
model = Sequential([
    Input(784),
    Dense(64),
    ReLU(),
    ...
    Dense(10),
    Softmax()
])

model.compile(...)
model.fit(...)
```

## Future scope:
- Implement layers to support CNNs, RNNs, Dropouts
- Autograd
- Callbacks for training loops (LRScheduler, EarlyStopping)
- GPU support