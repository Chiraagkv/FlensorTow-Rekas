import numpy as np
from rekas.templates import Layer

class Input(Layer):
  pass

class Dense(Layer):
  def forward(self, x, weights, bias):
    if(x.shape[0] == weights.shape[1] and self.units == weights.shape[0]):
      self.x = x
      self.fnodes = weights@x + bias
      self.bnodes = weights.T
      return self.fnodes
    else:
      raise ValueError("Shape mismatch")

  def backward(self, prev):
    self.grad_weights = prev @ (self.x.T)
    self.grad_bias = prev
    return self.bnodes @ prev

