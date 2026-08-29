import numpy as np
from rekas.templates import Activation

class ReLU(Activation):
  def forward(self, inputs):
    self.nodes = inputs > 0
    return np.multiply(inputs, self.nodes)

  def backward(self, prev):
    return np.multiply(prev, self.nodes)

class Sigmoid(Activation):
  def forward(self, inputs):
    self.nodes = 1/ (1 + np.exp(-np.clip(inputs, -500, 500)))
    return self.nodes

  def backward(self, prev):
    self.bnodes = np.multiply(self.nodes , 1-self.nodes)
    return np.multiply(prev, self.bnodes)

class Softmax(Activation):
  def forward(self, inputs):
    shifted = inputs - np.max(inputs)
    exps = np.exp(shifted)
    self.nodes = exps / np.sum(exps)
    return self.nodes

  def backward(self, prev):
    return prev # NOTE!!!! I am not doing anything here because when used along with CCE, the derivative is easier to code
