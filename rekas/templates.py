class Node:
  def forward(self):
    pass
  def backward(self):
    pass

class Layer(Node):
  def __init__(self, units):
    self.units = units
    self.fnodes = None
    self.bnodes = None
    self.x = None

class Activation(Node):
  pass

class Loss(Node):
  pass

class Optimizer:
  def __init__(self, lr):
    self.lr = lr

  def update(self, X, y, weights, biases, loss_fn, layers, nonacts):
    pass