import numpy as np
from rekas.templates import Layer
from rekas.layers import Input

class Sequential:
  def __init__(self, layers):
    if (not isinstance(layers[0], Input)):
      raise TypeError("First layer needs to be Input")
    self.layers = layers
    self.weights = []
    self.biases = []
    self.nonacts = []
    for i in layers:
      if isinstance(i, Layer):
        self.nonacts.append(i)
    for i in range(len(self.nonacts)-1):
      w_i = np.random.randn(self.nonacts[i+1].units, self.nonacts[i].units) * np.sqrt(2.0 / self.nonacts[i].units) # He initialization of weights in ReLU networks
      b_i = np.zeros((self.nonacts[i+1].units, 1))
      self.weights.append(w_i)
      self.biases.append(b_i)

  def compile(self, loss, optimizer):
    self.loss = loss
    self.optimizer = optimizer

  def fit(self, X, y, epochs, verbose=False):
    for epoch in range(1, epochs+1):
      self.weights, self.biases, msg = self.optimizer.update(X, y, self.weights, self.biases, self.loss, self.layers, self.nonacts)
      if verbose:
        print(f"Epoch {epoch}\n{msg}")

  def predict(self, X):
    predictions = []
    for xi in X:
      xi = np.asarray(xi).reshape(-1, 1)
      z = xi
      l = 0
      for i in range(1, len(self.layers)):
        if isinstance(self.layers[i], Layer):
          z = self.layers[i].forward(z, self.weights[l], self.biases[l])
          l += 1
        else:
          z = self.layers[i].forward(z)

      predictions.append(z.squeeze())
    return np.array(predictions)