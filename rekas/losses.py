import numpy as np
from rekas.templates import Loss

class MSE(Loss):
  def forward(self, y, ypred):
    self.y = y
    self.preds = ypred
    return np.mean(0.5*(y-ypred)**2)

  def backward(self):
    return self.preds - self.y

class BCE(Loss):
  def forward(self, y, preds):
    self.y = y
    self.preds = np.clip(preds, 1e-15, 1-(1e-15))
    return np.mean(-(y * np.log(self.preds) + (1 - y) * np.log(1 - self.preds)))

  def backward(self):
    return (self.preds - self.y) / (self.preds * (1 - self.preds) + 1e-15)

class CCE(Loss):
  def forward(self, y, preds):
    self.y = y
    self.preds = np.clip(preds, 1e-15, 1 - 1e-15)
    return -np.sum(self.y * np.log(self.preds))

  def backward(self):
    return self.preds - self.y