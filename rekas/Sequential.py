import numpy as np
from rekas.templates import Layer
from rekas.layers import Input
import pickle 

class Sequential:
  count = 0
  def __init__(self, layers, name=f'Sequential {count+1}'):
    count += 1
    self.name = name
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
  
  def summary(self):
      print(f'Model: "{self.name}"')
      print("=" * 65)

      print(f'{"Layer":<25} {"Output Shape":<20} {"Parameters":>15}')
      print("-" * 65)

      total_params = 0
      trainable_params = 0

      for i, layer in enumerate(self.layers):
          layer_name = layer.__class__.__name__

          if isinstance(layer, Input):
              output_shape = f'({layer.units},)'
              params = 0

          elif isinstance(layer, Layer):
              output_shape = f'({layer.units},)'

              weight_idx = self.nonacts.index(layer) - 1

              if weight_idx >= 0:
                  params = (
                      self.weights[weight_idx].size +
                      self.biases[weight_idx].size
                  )
              else:
                  params = 0

              trainable_params += params

          else:
              output_shape = f'({layer.units},)' if hasattr(layer, 'units') else '-'
              params = 0

          total_params += params

          print(f'{layer_name:<25} {output_shape:<20} {params:>15,}')

      print("=" * 65)
      print(f'Total parameters:     {total_params:,}')
      print(f'Trainable parameters: {trainable_params:,}')
  
  def save(self, path):
     data = {
        "name": self.name,
        "layers": self.layers,
        "weights": self.weights,
        "biases": self.biases
      }
     
     with open(path, 'wb') as f:
        pickle.dump(data, f)
  
  @classmethod
  def load(cls, path):
     with open(path, 'rb') as f:
        data = pickle.load(f)
     model = cls(data['layers'], data['name']+'_loaded')
     model.weights = data['weights']
     model.biases = data['biases']

     return model