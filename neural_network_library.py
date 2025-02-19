import numpy as np

np.random.seed(42)
class Layer():
    """
    Base class for all layers. 

    """
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, grad):
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
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.zeros(output_size) ##np.random.randn(output_size)
        self.x = None

        # set gradients to zero
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x):
        """
        Computes the forward pass of the linear layer.
        
        Parameters:
        ----------
        x: np.ndarray
            The input data.
        
        """
    
        self.x = x # cache the input data for backpropagation

        return np.dot(x, self.weights.T) + self.bias
    
    def backward(self, grad):
        """
        Computes the backward pass of the linear layer.
        
        Parameters:
        ----------
        grad: np.ndarray
            The gradient of the loss with respect to the output of the linear layer.
        """

        # Compute the gradients for weights and bias
        self.grad_weights = np.dot(grad.T, self.x)
        self.grad_bias = np.sum(grad, axis=0)
    
        return np.dot(grad, self.weights)
        
class Sigmoid(Layer):
    def __init__(self):
        """
        Initializes a sigmoid layer.
        """
        super().__init__()
        self.x = None 

    def forward(self, x):
        """
        Forward pass of the Sigmoid layer.

        Parameters:
        ----------
        x: np.ndarray
            The input data. 
        """
        self.x = 1 / (1 + np.exp(-x))
        return self.x

    def backward(self, grad):
        """
        Backward pass of the Sigmoid layer.

        Parameters:
        ----------
        grad: np.ndarray
            The gradient of the loss with respect to the output of the sigmoid layer.
        """
        return grad * self.x * (1 - self.x)

class Tanh(Layer):
    def __init__(self):
        """
        Initializes a tanh layer.
        """
        super().__init__()
        self.x = None
    
    def forward(self, x):
        """
        Forward pass of the tanh layer.
        
        Parameters:
        ----------
        x: np.ndarray
            The input data. 
        """
        self.x = np.tanh(x)
        return self.x

    def backward(self, grad):
        """
        Backward pass of the tanh layer.
        
        Parameters:
        ----------
        grad: np.ndarray
            The gradient of the loss with respect to the output of the tanh layer.
        """
        return grad * (1 - self.x ** 2)

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
            The input data.
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        """
        Backward pass of the ReLU layer.
        
        Parameters:
        ----------
        grad: np.ndarray
            The gradient of the loss with respect to the output of the ReLU layer.
        """

        return grad * (self.x > 0)

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
            The predicted values.
        y: np.ndarray
            The ground truth values.
        """
        self.y = y
        self.batch_size = y_pred.shape[0]

        # clip the predictions numerical stability
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

class MseLoss(Layer):
    def __init__(self):
        """
        Initializes a Mean Squared Error loss layer.
        """

        super().__init__()
        self.y = None
        self.y_pred = None
        self.batch_size = None
    
    def forward(self, y_pred, y):
        """
        Forward pass of the Mean Squared Error loss layer.
        
        Parameters:
        ----------
        y_pred: np.ndarray
            The predicted values.
        y: np.ndarray
            The ground truth values.
        """

        self.y = y
        self.y_pred = y_pred
        self.batch_size = y_pred.shape[0]
        loss = np.mean((y_pred - y) ** 2)
        return loss

    def backward(self, grad=1.0):
        """
        Backward pass of the Mean Squared Error loss layer.
        
        Parameters:
        ----------
        grad: float
            The gradient of the loss with respect to the output of the MSE loss layer.
        """
        grad_y_pred = 2 * (self.y_pred - self.y) / self.batch_size
        return grad_y_pred * grad

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
            The input data.
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


