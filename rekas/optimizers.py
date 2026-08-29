import numpy as np
from rekas.templates import Optimizer
from rekas.templates import Layer

class SGD(Optimizer):
  def update(self, X, y, weights, biases, loss_fn, layers, nonacts):
    total_loss = 0

    gradweights = [None] * len(weights)
    gradbiases = [None] * len(biases)

    for xi, yi in zip(X, y):
      xi = np.asarray(xi).reshape(-1, 1)
      yi = np.asarray(yi).reshape(-1, 1)

      layers[0].fnodes = layers[0].x = xi

      z = xi
      l = 0
      for i in range(1, len(layers)):
        if(layers[i] in nonacts):
          z = layers[i].forward(z, weights[l], biases[l]);
          l += 1
        else:
          z = layers[i].forward(z)

      loss = loss_fn.forward(yi, z)
      total_loss += np.sum(loss)

      prev = loss_fn.backward()

      l = len(weights)-1
      for i in range(len(layers)-1, 0, -1):
        if(isinstance(layers[i], Layer)):
          prev = layers[i].backward(prev)

          gradweights[l] = layers[i].grad_weights
          gradbiases[l] = layers[i].grad_bias
          l -= 1
        else:
          prev = layers[i].backward(prev)

      for i in range(len(weights)):
        weights[i] -= self.lr * gradweights[i]
        biases[i] -= self.lr * gradbiases[i]

    n = len(X)
    msg = f"Loss: {total_loss/n}"

    return weights, biases, msg

class MiniBatch(Optimizer):
  def __init__(self, lr, batch_size):
    super().__init__(lr)
    self.batch_size = batch_size

  def update(self, X, y, weights, biases, loss_fn, layers, nonacts):
    total_loss = 0
    gradweights = [np.zeros_like(w) for w in weights]
    gradbiases = [np.zeros_like(b) for b in biases]

    c = 0
    n = len(X)
    for xi, yi in zip(X, y):
      c += 1

      xi = np.asarray(xi).reshape(-1, 1)
      yi = np.asarray(yi).reshape(-1, 1)

      layers[0].fnodes = layers[0].x = xi

      z = xi
      l = 0
      for i in range(1, len(layers)):
        if(layers[i] in nonacts):
          z = layers[i].forward(z, weights[l], biases[l]);
          l += 1
        else:
          z = layers[i].forward(z)

      loss = loss_fn.forward(yi, z)
      total_loss += np.sum(loss)

      prev = loss_fn.backward()

      l = len(weights)-1
      for i in range(len(layers)-1, 0, -1):
        if(isinstance(layers[i], Layer)):
          prev = layers[i].backward(prev)

          gradweights[l] += layers[i].grad_weights
          gradbiases[l] += layers[i].grad_bias
          l -= 1
        else:
          prev = layers[i].backward(prev)

      if c % self.batch_size == 0 or c == n:
        current_batch_size = self.batch_size if c % self.batch_size == 0 else (n % self.batch_size)

        for i in range(len(weights)):
          weights[i] -= self.lr * (gradweights[i] / current_batch_size)
          biases[i] -= self.lr * (gradbiases[i] / current_batch_size)

        gradweights = [np.zeros_like(w) for w in weights]
        gradbiases = [np.zeros_like(b) for b in biases]

    msg = f"Loss: {total_loss/n}"

    return weights, biases, msg

class Nesterov(Optimizer):
  def __init__(self, lr, beta, batch_size=None):
    super().__init__(lr)
    self.beta = beta
    self.v_weights = None
    self.v_biases = None
    self.batch_size = 1 if not batch_size else batch_size

  def update(self, X, y, weights, biases, loss_fn, layers, nonacts):
    if self.v_weights is None or self.v_biases is None:
      self.v_weights = [np.zeros_like(w) for w in weights]
      self.v_biases = [np.zeros_like(b) for b in biases]

    future_weights = [w - self.beta * vw for w, vw in zip(weights, self.v_weights)]
    future_biases = [b - self.beta * vb for b, vb in zip(biases, self.v_biases)]

    total_loss = 0
    gradweights = [np.zeros_like(w) for w in weights]
    gradbiases = [np.zeros_like(b) for b in biases]
    n = len(X)

    c = 0
    for xi, yi in zip(X, y):
      c += 1

      xi = np.asarray(xi).reshape(-1, 1)
      yi = np.asarray(yi).reshape(-1, 1)

      layers[0].fnodes = layers[0].x = xi

      z = xi
      l = 0
      for i in range(1, len(layers)):
        if(layers[i] in nonacts):
          z = layers[i].forward(z, future_weights[l], future_biases[l]);
          l += 1
        else:
          z = layers[i].forward(z)

      loss = loss_fn.forward(yi, z)
      total_loss += np.sum(loss)

      prev = loss_fn.backward()

      l = len(weights)-1
      for i in range(len(layers)-1, 0, -1):
        if(isinstance(layers[i], Layer)):
          prev = layers[i].backward(prev)

          gradweights[l] += layers[i].grad_weights
          gradbiases[l] += layers[i].grad_bias
          l -= 1
        else:
          prev = layers[i].backward(prev)

      if c % self.batch_size == 0 or c == n:
        current_batch_size = self.batch_size if c % self.batch_size == 0 else (n % self.batch_size)

        for i in range(len(weights)):
          avg_gw = gradweights[i] / current_batch_size
          avg_gb = gradbiases[i] / current_batch_size

          self.v_weights[i] = self.beta * self.v_weights[i] + self.lr * avg_gw
          self.v_biases[i] = self.beta * self.v_biases[i] + self.lr * avg_gb

          weights[i] -= self.v_weights[i]
          biases[i] -= self.v_biases[i]

        gradweights = [np.zeros_like(w) for w in weights]
        gradbiases = [np.zeros_like(b) for b in biases]

        future_weights = [w - self.beta * vw for w, vw in zip(weights, self.v_weights)]
        future_biases = [b - self.beta * vb for b, vb in zip(biases, self.v_biases)]

    msg = f"Loss: {total_loss/n}"

    return weights, biases, msg

class RMSProp(Optimizer):
  def __init__(self, lr, beta, batch_size=None):
    super().__init__(lr)
    self.beta = beta
    self.batch_size = 1 if not batch_size else batch_size
    self.v_weights = None
    self.v_biases = None
    self.epsilon = 1e-8


  def update(self, X, y, weights, biases, loss_fn, layers, nonacts):
    if self.v_weights is None or self.v_biases is None:
      self.v_weights = [np.zeros_like(w) for w in weights]
      self.v_biases = [np.zeros_like(b) for b in biases]

    total_loss = 0
    gradweights = [np.zeros_like(w) for w in weights]
    gradbiases = [np.zeros_like(b) for b in biases]

    c = 0
    n = len(X)
    for xi, yi in zip(X, y):
      c += 1

      xi = np.asarray(xi).reshape(-1, 1)
      yi = np.asarray(yi).reshape(-1, 1)

      layers[0].fnodes = layers[0].x = xi

      z = xi
      l = 0
      for i in range(1, len(layers)):
        if(layers[i] in nonacts):
          z = layers[i].forward(z, weights[l], biases[l]);
          l += 1
        else:
          z = layers[i].forward(z)

      loss = loss_fn.forward(yi, z)
      total_loss += np.sum(loss)

      prev = loss_fn.backward()

      l = len(weights)-1
      for i in range(len(layers)-1, 0, -1):
        if(isinstance(layers[i], Layer)):
          prev = layers[i].backward(prev)

          gradweights[l] += layers[i].grad_weights
          gradbiases[l] += layers[i].grad_bias
          l -= 1
        else:
          prev = layers[i].backward(prev)

      if c % self.batch_size == 0 or c == n:
        current_batch_size = self.batch_size if c % self.batch_size == 0 else (n % self.batch_size)

        for i in range(len(weights)):
          gw = gradweights[i] / current_batch_size
          gb = gradbiases[i] / current_batch_size

          self.v_weights[i] = self.beta*self.v_weights[i] + (1-self.beta)*(gw**2)
          self.v_biases[i] = self.beta*self.v_biases[i] + (1-self.beta)*(gb**2)
          weights[i] -= self.lr * (gw / (np.sqrt(self.v_weights[i]) + self.epsilon))
          biases[i] -= self.lr * (gb / (np.sqrt(self.v_biases[i]) + self.epsilon))

        gradweights = [np.zeros_like(w) for w in weights]
        gradbiases = [np.zeros_like(b) for b in biases]

    msg = f"Loss: {total_loss/n}"

    return weights, biases, msg

class Adam(Optimizer):
  def __init__(self, lr, beta1, beta2, batch_size=None):
    super().__init__(lr)
    self.beta1 = beta1
    self.beta2 = beta2
    self.batch_size = 1 if not batch_size else batch_size
    self.epsilon = 1e-8

    self.v_weights = None
    self.v_biases = None
    self.m_weights = None
    self.m_biases = None
    self.t = 0

  def update(self, X, y, weights, biases, loss_fn, layers, nonacts):
    if self.v_weights is None or self.v_biases is None:
      self.v_weights = [np.zeros_like(w) for w in weights]
      self.v_biases = [np.zeros_like(b) for b in biases]

    if self.m_weights is None or self.m_biases is None:
      self.m_weights = [np.zeros_like(w) for w in weights]
      self.m_biases = [np.zeros_like(b) for b in biases]

    total_loss = 0
    gradweights = [np.zeros_like(w) for w in weights]
    gradbiases = [np.zeros_like(b) for b in biases]

    c = 0
    n = len(X)
    for xi, yi in zip(X, y):
      c += 1

      xi = np.asarray(xi).reshape(-1, 1)
      yi = np.asarray(yi).reshape(-1, 1)

      layers[0].fnodes = layers[0].x = xi

      z = xi
      l = 0
      for i in range(1, len(layers)):
        if(layers[i] in nonacts):
          z = layers[i].forward(z, weights[l], biases[l]);
          l += 1
        else:
          z = layers[i].forward(z)

      loss = loss_fn.forward(yi, z)
      total_loss += np.sum(loss)

      prev = loss_fn.backward()

      l = len(weights)-1
      for i in range(len(layers)-1, 0, -1):
        if(isinstance(layers[i], Layer)):
          prev = layers[i].backward(prev)

          gradweights[l] += layers[i].grad_weights
          gradbiases[l] += layers[i].grad_bias
          l -= 1
        else:
          prev = layers[i].backward(prev)

      if c % self.batch_size == 0 or c == n:
        self.t += 1
        current_batch_size = self.batch_size if c % self.batch_size == 0 else (n % self.batch_size)

        for i in range(len(weights)):
          gw = gradweights[i] / current_batch_size
          gb = gradbiases[i] / current_batch_size

          self.v_weights[i] = self.beta2*self.v_weights[i] + (1-self.beta2)*(gw**2)
          self.v_biases[i] = self.beta2*self.v_biases[i] + (1-self.beta2)*(gb**2)

          self.m_weights[i] = self.beta1*self.m_weights[i] + (1-self.beta1)*(gw)
          self.m_biases[i] = self.beta1*self.m_biases[i] + (1-self.beta1)*(gb)

          m_weights_cap = self.m_weights[i] / (1- self.beta1**self.t)
          m_biases_cap = self.m_biases[i] / (1-self.beta1**self.t)

          v_weights_cap = self.v_weights[i] / (1- self.beta2**self.t)
          v_biases_cap = self.v_biases[i] / (1-self.beta2**self.t)

          weights[i] -= self.lr * (m_weights_cap / (np.sqrt(v_weights_cap) + self.epsilon))
          biases[i] -= self.lr * (m_biases_cap / (np.sqrt(v_biases_cap) + self.epsilon))

        gradweights = [np.zeros_like(w) for w in weights]
        gradbiases = [np.zeros_like(b) for b in biases]

    msg = f"Loss: {total_loss/n}"

    return weights, biases, msg
