import numpy as np

np.random.seed(42)
class Layer():
    """
    Base class for all layers

    """
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grad_z):
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size):
        """
        Initializes a linear layer.

        Parameters:
        ----------
        input_size: int
            The number of input features (the dimension of the input)
        output_size: int
            The number of output features (the dimension of the output)
        """
        super().__init__()

        # Initialize weights and bias
        self.weights = np.random.randn(output_size, input_size) #* 0.01
        self.bias = np.zeros(output_size) ##np.random.randn(output_size)
        self.x = None

        ## debug print
        ##print('weights should be init:', self.weights)
        ##print('shape of weights:', self.weights.shape)

        ##print('bias should be init:', self.bias)
        ##print('shape of bias:', self.bias.shape)

        # set gradients to zero
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        ## debug print
        ##print('grad_weights should be zeros:', self.grad_weights)
        ##print('shape of grad_weights:', self.grad_weights.shape)

        ##print('grad_bias should be zeros:', self.grad_bias)
        ##print('shape of grad_bias:', self.grad_bias.shape)
        

    def forward(self, x):
        """
        Computes the forward pass of the linear layer.
        
        Parameters:
        ----------
        x: np.ndarray
            The input data. The shape of x is (batch_size, input_size)
        
        """
        self.x = x

        ## debug print
        ##print('x:', x)
        ##print('shape of x: ', x.shape)

        z = np.dot(x, self.weights.T) + self.bias

        ## debug print
        ##print('z:', z)
        ##print('shape of z:', z.shape)

        return z
    
    def backward(self, grad_z):
        """
        Computes the backward pass of the linear layer.
        
        Parameters:
        ----------
        grad_z: np.ndarray
            The gradient of the loss with respect to the output of the linear layer. The shape of grad_z is (batch_size, output_size)
        """

        # Compute the gradients via the chain rule 
        # derivative of loss with respect to weights = derivative of loss with respect to z * derivative of z with respect to weights
        # derivative of loss with respect to bias = derivative of loss with respect to z * derivative of z with respect to bias
        # derivative of loss with respect to x = derivative of loss with respect to z * derivative of z with respect to x
        self.grad_weights = np.dot(grad_z.T, self.x)

        ## debug print
        ##print('grad_weights:', self.grad_weights)
        ##print('shape of grad_weights:', self.grad_weights.shape)

        self.grad_bias = np.sum(grad_z, axis=0)
        
        ## debug print
        ##print('grad_bias:', self.grad_bias)
        ##print('shape of grad_bias:', self.grad_bias.shape)
        
        grad_x = np.dot(grad_z, self.weights)

        ## debug print
        ##print('grad_x:', grad_x)
        ##print('shape of grad_x:', grad_x.shape)

        return grad_x        

class Sigmoid(Layer):
    def __init__(self):
        """
        Initializes a sigmoid layer.
        """
        super().__init__()
        self.sigmoid = None

    def forward(self, x):
        """
        Forward pass of the Sigmoid layer.

        Parameters:
        ----------
        x: np.ndarray
            The input data. The shape of x is (batch_size, input_size)
        """
        self.sigmoid = 1 / (1 + np.exp(-x))
        return self.sigmoid

    def backward(self, grad):
        """
        Backward pass of the Sigmoid layer.

        Parameters:
        ----------
        grad: np.ndarray
            The gradient of the loss with respect to the output of the sigmoid layer. The shape of x is (batch_size, input_size)
        """
        grad_x = grad * self.sigmoid * (1 - self.sigmoid)
        return grad_x

class ReLU(Layer):
    def __init__(self):
        """
        Initializes a ReLU layer.
        """
        super().__init__()
        self.x = None

    def forward(self, x):
        """
        Forward pass of the ReLU layer.
        
        Parameters:
        ----------
        x: np.ndarray
            The input data. The shape of x is (batch_size, input_size)
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        """
        Backward pass of the ReLU layer.
        
        Parameters:
        ----------
        grad: np.ndarray
            The gradient of the loss with respect to the output of the ReLU layer. The shape of x is (batch_size, input_size)
        """
        grad_x = grad * (self.x > 0)
        return grad_x



class BceLoss(Layer):
    def __init__(self, epsilon=1e-8):
        """
        Initializes a Binary Cross-Entropy loss layer.
        
        Parameters:
        ----------
        epsilon: float
            A small value to avoid log(0) and division by zero.
        """
        super().__init__()
        self.epsilon = epsilon
        self.y = None
        self.y_pred_clip = None
        self.batch_size = None

    def forward(self, y_pred, y):
        """
        Forward pass of the Binary Cross-Entropy loss layer.
        
        Parameters:
        ----------
        y_pred: np.ndarray
            The predicted values. The shape of y_pred is (batch_size, 1)
        y: np.ndarray
            The ground truth values. The shape of y is (batch_size, 1)
        """
        self.y = y
        self.batch_size = y_pred.shape[0]

        # clip the predictions to avoid log(0)
        self.y_pred_clip = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        loss_per_sample = -(self.y * np.log(self.y_pred_clip) + (1 - self.y) * np.log(1 - self.y_pred_clip))
        loss = np.mean(loss_per_sample)
        return loss

    def backward(self, grad=1.0):
        """
        Backward pass of the Binary Cross-Entropy loss layer.
        
        Parameters:
        ----------
        grad: float
            The gradient of the loss with respect to the output of the BCE loss layer.
        """
        grad_y_pred = (self.y_pred_clip - self.y) / self.batch_size # b/c if sigmoid is used in the last layer, the derivative of BCE loss w.r.t. y_pred is (y_pred - y)
        grad_y_pred = grad_y_pred * grad
        return grad_y_pred
    
class Sequential(Layer):
    def __init__(self, layers=None):
        """
        Initializes a sequential model.
        """
        super().__init__()
        if layers is None:
            self.layers = []
        else:
            self.layers = layers
    
    def add(self, layer):
        """
        Adds a layer to the model.
        
        Parameters:
        ----------
        layer: Layer
            A layer object.
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass of the sequential model.
        
        Parameters:
        ----------
        x: np.ndarray
            The input data. The shape of x is (batch_size, input_size)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """
        Backward pass of the sequential model.

        Parameters:
        ----------
        grad: np.ndarray
            The gradient of the loss with respect to the output of the sequential model.
        """

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def save(self, filepath):
        """
        Saves model weights to a file. 
        
        Parameters:
        ----------
        filepath: str
            The file path where the weights will be saved. 
        """
        weights_data = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                weights_data[f"layer_{idx}_weights"] = layer.weights
                weights_data[f"layer_{idx}_bias"] = layer.bias
        np.savez(filepath, **weights_data)
        print(f"Weights saved to {filepath}")
    
    def load(self, filepath):
        """
        Loads model weights from a file. 
        
        Parameters:
        ----------
        filepath: str
            The file path from where the weights will be loaded.
        """
        loaded = np.load(filepath)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.weights = loaded[f'layer_{idx}_weights']
                layer.bias = loaded[f'layer_{idx}_bias']
        print(f"Weights loaded from {filepath}")
    
# # Test Linear Layer
# linear_layer = Linear(input_size=3, output_size=2)
# x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# output = linear_layer.forward(x)
# grad_z = np.array([[1.0, 2.0], [3.0, 4.0]])
# grad_x = linear_layer.backward(grad_z)

# print("Forward Output:", output)
# print("Input Gradient:", grad_x)
# print("Weights Gradient:", linear_layer.grad_weights)
# print("Bias Gradient:", linear_layer.grad_bias)

# # Test Sigmoid Layer
# sigmoid = Sigmoid()
# x_sigmoid = np.array([[0.5, -1.0], [2.0, 0.0]])
# output_sigmoid = sigmoid.forward(x_sigmoid)
# grad_sigmoid = np.array([[0.1, 0.2], [0.3, 0.4]])
# dx_sigmoid = sigmoid.backward(grad_sigmoid)

# print("\nSigmoid Output:", output_sigmoid)
# print("Sigmoid Input Gradient:", dx_sigmoid)

# # Test ReLU Layer
# relu = ReLU()
# x_relu = np.array([[2.0, -1.0], [0.0, 3.0]])
# output_relu = relu.forward(x_relu)
# print("\nReLU Output:", output_relu)

# grad_relu = np.array([[1.0, 2.0], [3.0, 4.0]])
# dx_relu = relu.backward(grad_relu)
# print("ReLU Input Gradient:", dx_relu)


# Testing xor with neural network
np.random.seed(42)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# model_sigmoid = Sequential()
# model_sigmoid.add(Linear(input_size=2, output_size=2))
# model_sigmoid.add(Sigmoid())
# model_sigmoid.add(Linear(input_size=2, output_size=1))
# model_sigmoid.add(Sigmoid())
model_sigmoid = Sequential(
    layers=[
        Linear(input_size=2, output_size=2),
        Sigmoid(),
        Linear(input_size=2, output_size=1),
        Sigmoid()
    ]
)
loss_fn = BceLoss()
learning_rate = 1.0
n_epochs = 10000

for epoch in range(n_epochs):
    # Forward pass
    y_pred = model_sigmoid.forward(X)
    loss = loss_fn.forward(y_pred, y)

    # Backward pass
    grad = loss_fn.backward()
    model_sigmoid.backward(grad)

    # Update weights
    for layer in model_sigmoid.layers:
        if isinstance(layer, Linear):
            layer.weights -= learning_rate * layer.grad_weights
            layer.bias -= learning_rate * layer.grad_bias

    if epoch % 1000 == 0:
        print(f"sigmoid loss at epoch {epoch}: {loss:.6f}")

print("\nPredictions after training:")
print(np.round(model_sigmoid.forward(X), 3))

print('ground truth:')
print(y)

model_sigmoid.save("XOR_solved_model_sigmoid.npz")