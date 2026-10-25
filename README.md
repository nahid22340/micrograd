# Micrograd 

A tiny scalar-valued autograd engine and neural network library built from scratch, inspired by Andrej Karpathy's `micrograd`. 

This project implements a backpropagation engine from the ground up, capable of building multi-layer perceptrons (MLPs) and training them on small datasets.

---

# Features

* **Scalar Autograd Engine:** Implements automatic differentiation (backward pass) tracking operations (`+`, `-`, `*`, `/`, `**`, `tanh`).
* **Neural Network Modules:** Built-in `Neuron`, `Layer`, and `MLP` classes supporting custom architectures and activation toggles.
* **Zero External Dependencies:** Core engine relies entirely on Python's standard library (`math`, `random`).

---

# Structure

```text
micrograd-from-scratch/
│
├── engine.py          # Core autograd engine, Neuron, Layer, and MLP classes
├── test_engine.py     # Test suite and training loop verification
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
