import math
import random


random.seed(42)

class value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, value) else value(other)
    out = value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, value) else value(other)
    out = value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = value(self.data ** other, (self,), f'**{other}')

    def _backward():
      self.grad += (other * (self.data ** (other - 1))) * out.grad
    out._backward = _backward
    return out

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    return self * (other ** -1)

  def tanh(self):
    x = self.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = value(t, (self,), 'tanh')

    def _backward():
      self.grad += (1 - t ** 2) * out.grad
    out._backward = _backward
    return out

  def exp(self):
    x = self.data
    out = value(math.exp(x), (self,), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

class Neuron:
    def __init__(self, nin):
      """Number of inputs come to the neuron."""
      self.w = [value(random.uniform(-1, 1)) for _ in range(nin)]     # Weights for each input
      self.b = value(random.uniform(-1, 1))       # Bias controls the threshold of the neuron.

    def __call__(self, x):
      act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
      out = act.tanh()
      return out

    def parameters(self):
      return self.w + [self.b]           # [w1, w2, ..., wn, b] list of parameters for the neuron.

class Layer:
    def __init__(self, nin, nout):    # How many inputs neurons and how many outputs neurons in the layer.
      self.neurons = [Neuron(nin) for _ in range(nout)]   # List of neurons in the layer

    def __call__(self, x):
      outs = [n(x) for n in self.neurons]
      return outs[0] if len(outs) == 1 else outs

    def parameters(self):
      return [p for neuron in self.neurons for p in neuron.parameters()]     # Flatten the list of parameters from all neurons in the layer.

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]         # Flatten the list of parameters from all layers in the MLP.

# --- Dataset & Training Loop ---
x = [2.0, 3.0, -1.0]         # Sample input features
n = MLP(3, [4, 4, 1])        # 3 inputs, 2 hidden layers of 4 neurons each, and 1 output neuron

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.2, -1.0]
]
ys = [0.5, -1.0, -1.0, 0.7]  # Scaled targets for tanh output limits

for k in range(1000):
  # Forward pass
  ypred = [n(x) for x in xs]
  loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)], value(0))   # Compute mean squared error loss.

  # Backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  # Update weights
  for p in n.parameters():
    p.data += -0.06 * p.grad    # Tiny nudge in the direction of the gradient to minimize loss.
ypred