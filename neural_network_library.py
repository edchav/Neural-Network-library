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
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros(output_size) ##np.random.randn(output_size)
        self.x = None

        ## debug print
        print('weights should be init:', self.weights)
        print('shape of weights:', self.weights.shape)

        print('bias should be init:', self.bias)
        print('shape of bias:', self.bias.shape)

        # set gradients to zero
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        ## debug print
        print('grad_weights should be zeros:', self.grad_weights)
        print('shape of grad_weights:', self.grad_weights.shape)

        print('grad_bias should be zeros:', self.grad_bias)
        print('shape of grad_bias:', self.grad_bias.shape)
        

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
        print('x:', x)
        print('shape of x: ', x.shape)

        z = np.dot(x, self.weights.T) + self.bias

        ## debug print
        print('z:', z)
        print('shape of z:', z.shape)

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
        print('grad_weights:', self.grad_weights)
        print('shape of grad_weights:', self.grad_weights.shape)

        self.grad_bias = np.sum(grad_z, axis=0)
        
        ## debug print
        print('grad_bias:', self.grad_bias)
        print('shape of grad_bias:', self.grad_bias.shape)
        
        grad_x = np.dot(grad_z, self.weights)

        ## debug print
        print('grad_x:', grad_x)
        print('shape of grad_x:', grad_x.shape)

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
        
# Test Linear Layer
linear_layer = Linear(input_size=3, output_size=2)
x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
output = linear_layer.forward(x)
grad_z = np.array([[1.0, 2.0], [3.0, 4.0]])
grad_x = linear_layer.backward(grad_z)

print("Forward Output:", output)
print("Input Gradient:", grad_x)
print("Weights Gradient:", linear_layer.grad_weights)
print("Bias Gradient:", linear_layer.grad_bias)

# Test Sigmoid Layer
sigmoid = Sigmoid()
x_sigmoid = np.array([[0.5, -1.0], [2.0, 0.0]])
output_sigmoid = sigmoid.forward(x_sigmoid)
grad_sigmoid = np.array([[0.1, 0.2], [0.3, 0.4]])
dx_sigmoid = sigmoid.backward(grad_sigmoid)

print("\nSigmoid Output:", output_sigmoid)
print("Sigmoid Input Gradient:", dx_sigmoid)

# class Relu(Layer):
#     def __init__(self, ) -> None:
#         pass

#     def forward(self, x):
#         pass

#     def backward(self, x):
#         pass

# class BceLoss(Layer):
#     def __init__(self, ) -> None:
#         pass

#     def forward(self, x):
#         pass

#     def backward(self, x):
#         pass

# class Sequential():
#     def __init__(self, ) -> None:
#         pass

#     def forward(self, x):
#         pass

#     def backward(self, x):
#         pass