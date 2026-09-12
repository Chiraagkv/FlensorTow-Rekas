import numpy as np
from rekas.templates import Initializer

class RandomNormal(Initializer):
    def initialize(self, shape):
        return np.random.randn(*shape) * 0.01

class XavierNormal(Initializer):
    def initialize(self, shape):
        fan_out, fan_in = shape
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_out, fan_in) * std

class HeNormal(Initializer):
    def initialize(self, shape):
        fan_out, fan_in = shape
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_out, fan_in) * std

