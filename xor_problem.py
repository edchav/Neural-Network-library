# XOR Problem

import numpy as np
import neural_network_library as nnl

np.random.seed(42)

# Define the xor dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype = np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0],
], dtype= np.float32)

# Define the model
model_sigmoid = nnl.Sequential(
    layers=[
        nnl.Linear(input_size = 2, output_size=2),
        nnl.Sigmoid(),
        nnl.Linear(input_size = 2, output_size = 1),
        nnl.Sigmoid()
    ]
)

# Define the loss function
loss_fn = nnl.BceLoss()
learning_rate = 0.1
n_epochs = 100000

# Train the model
for epoch in range(n_epochs):
    # Forward pass
    y_pred = model_sigmoid.forward(X)
    loss = loss_fn.forward(y_pred, y)
    
    # Backward pass
    grad = loss_fn.backward()
    model_sigmoid.backward(grad)
        
    # Update the weights
    for layer in model_sigmoid.layers:
        if isinstance(layer, nnl.Linear):
            layer.weights -= learning_rate * layer.grad_weights
            layer.bias -= learning_rate * layer.grad_bias

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, loss: {loss}")

print("\nPredictions after training:")
print(np.round(model_sigmoid.forward(X), 1))
model_sigmoid.save("xor_model_sigmoid.npz")
# for i in range(X.shape[0]):
#     x = X[i].reshape(1, -1)
#     target = y[i].reshape(1, -1)
#     output = model_sigmoid.forward(x)
#     print(f"XOR({x}) = {output}")

