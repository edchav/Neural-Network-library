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

model_tanh = nnl.Sequential(
    layers=[
        nnl.Linear(input_size = 2, output_size=2),
        nnl.Tanh(),
        nnl.Linear(input_size = 2, output_size = 1),
        nnl.Sigmoid()
    ]
)

# Define the loss function
loss_sig_fn = nnl.BceLoss()
loss_tanh_fn = nnl.BceLoss()
learning_rate_sigmoid = 1
learning_rate_tanh = 0.5
n_epochs = 100000

# Train the model_sigmoid
for epoch in range(n_epochs):
    # Forward pass
    y_pred_sig = model_sigmoid.forward(X)
    loss_sig = loss_sig_fn.forward(y_pred_sig, y)
    
    # Backward pass
    grad_sig = loss_sig_fn.backward()
    model_sigmoid.backward(grad_sig)

    # Update the weights
    for layer in model_sigmoid.layers:
        if isinstance(layer, nnl.Linear):
            layer.weights -= learning_rate_sigmoid * layer.grad_weights
            layer.bias -= learning_rate_sigmoid * layer.grad_bias
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Sigmoid loss: {loss_sig}")

# Train the model_tanh
for epoch in range(n_epochs):
    # Forward pass
    y_pred_tanh = model_tanh.forward(X)
    loss_tanh = loss_tanh_fn.forward(y_pred_tanh, y)
    
    # Backward pass
    grad_tanh = loss_tanh_fn.backward()
    model_tanh.backward(grad_tanh)

    # Update the weights
    for layer in model_tanh.layers:
        if isinstance(layer, nnl.Linear):
            layer.weights -= learning_rate_tanh * layer.grad_weights
            layer.bias -= learning_rate_tanh * layer.grad_bias
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Tanh loss: {loss_tanh}")

print("\nPredictions after training for sigmoid:")
print(np.round(model_sigmoid.forward(X), 3))
model_sigmoid.save(r"Models\xor_model_sigmoid.npz")
print('weights sigmoid:', model_sigmoid.layers[0].weights)

print("\nPredictions after training for tanh:")
print(np.round(model_tanh.forward(X), 3))
model_tanh.save(r"Models\xor_model_tanh.npz")
print('weights tanh:', model_tanh.layers[0].weights)

print("\nGround truth:")
print(y)
