from engine import value, MLP
import random

# Fix random seed for reproducible tests
random.seed(42)

# Instantiate a neural network
model = MLP(3, [4, 4, 1])

# Input samples & targets (small dataset)
xs = [[2.0, 3.0, -1.0]]
ys = [1.0]

# Get initial loss before training
initial_loss = (model(xs[0]) - ys[0]) ** 2

# Run a quick training loop (e.g., 20 steps)
for step in range(20):
    # Forward pass
    ypred = model(xs[0])
    loss = (ypred - ys[0]) ** 2
    
    # Zero gradients & Backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    
    # Update parameters
    for p in model.parameters():
        p.data += -0.05 * p.grad

final_loss = loss.data
print(f"Initial Loss: {initial_loss.data:.4f}")
print(f"Final Loss (after 20 steps): {final_loss:.4f}")

# CHECK
assert final_loss < initial_loss.data, "Test Failed: Loss did not decrease!"
print("Test Passed: The network successfully learned and minimized the loss!")