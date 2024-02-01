import numpy as np
from initializers import GlorotNormal, Zero, Initializer


class Layer:
    """Base class for layers.

    Layers can be anything. They have to override forward and backward methods.
    Inside "backward" is a derivative of whatever operation happens inside "forward".
    If weights and biases are present inside the layer, also apply their gradient
    inside the "backward" method.
    Additionally, they have to override the "reset" method if there are any weights and
    biases in the layer which are modified during backprop.
    """
    def __init__(self):
        pass

    def forward(self, layer_input: np.ndarray) -> np.ndarray:
        """Return output that is equal to the input.

        Forward method of the base layer class. Merely outputs the input it gets.

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :return: Output matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """
        layer_output = layer_input
        return layer_output

    def backward(self, layer_input: np.ndarray, layer_output_grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """Propagate the gradient without changing it

        Backward method of the base layer class. Propagates the gradient matrix without changing it.

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :param layer_output_grad: Gradient of the output matrix of shape [batch,output_size]
        :type layer_output_grad: np.ndarray
        :param learning_rate: Learning rate during backpropagation
        :type learning_rate: float
        :return: Gradient of the input matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """
        layer_input_grad = layer_output_grad
        return layer_input_grad

    def reset(self):
        """Reset layer weight and biases if any are present."""
        pass


class Sigmoid(Layer):
    """Seems to cause exploding gradients. Layer containing the Sigmoid function."""

    def __init__(self):
        super().__init__()

    def forward(self, layer_input: np.ndarray) -> np.ndarray:
        """Return the output of the sigmoid function for the given input.

        The sigmoid function is defined by the formula:
        sig(x) = 1 / (1 + e^(-x))

        The output of the function is in the range [0,1]

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :return: Output matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """
        out = 1 / (1 + np.exp(-layer_input))
        return out

    def backward(self, layer_input: np.ndarray, layer_output_grad: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        """Apply the derivative of the sigmoid function and propagate the gradient.

        The derivative of the sigmoid function is defined by the formula:
        d_sig(x) = sig(x) * (1 - sig(x))

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :param layer_output_grad: Gradient of the output matrix of shape [batch,output_size]
        :type layer_output_grad: np.ndarray
        :param learning_rate: Learning rate during backpropagation
        :type learning_rate: float
        :return: Gradient of the input matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """

        sig_x = self.forward(layer_output_grad)
        out = sig_x * (1 - sig_x)
        return out


class ReLU(Layer):
    """Layer containing the ReLU function."""

    def __init__(self):
        super().__init__()

    def forward(self, layer_input: np.ndarray) -> np.ndarray:
        """Return the output of the ReLU function for the given input.

        The ReLU function is defined by the formula:
        relu(x) = maximum(0, x)

        In other words, the output is:
        if x =< 0  ->   out = 0
        if x > 0   ->   out = x

        The output of the function is in the range [0,x]

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :return: Output matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """

        relu_forward = np.maximum(0, layer_input)
        return relu_forward

    def backward(self, layer_input: np.ndarray, layer_output_grad: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        """Apply the derivative of the ReLU function and propagate the gradient.

        The derivative of the ReLu function is defined by the formula:
        d_relu(x) = 1 (x>0)

        Or simpler:
        if x =< 0  ->  out = 0
        if x > 0   ->  out = 1

        So the ReLU derivative is just a step function.

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :param layer_output_grad: Gradient of the output matrix of shape [batch,output_size]
        :type layer_output_grad: np.ndarray
        :param learning_rate: Learning rate during backpropagation
        :type learning_rate: float
        :return: Gradient of the input matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """

        layer_output_grad[layer_input < 0] = 0
        return layer_output_grad


class Dropout(Layer):
    """Dropout layer which randomly sets certain activations in the layer to 0."""

    def reset(self):
        """Reset the dropout values."""
        super().reset()
        self.dropout = np.zeros(shape=1)

    def __init__(self, ratio: float):
        """Dropout layer constructor

        :param ratio: ratio of layers that are not dropped out
        :type ratio: float
        """
        super().__init__()
        self.ratio = ratio
        self.dropout = np.zeros(shape=1)

    def forward(self, layer_input: np.ndarray) -> np.ndarray:
        """Forward pass of the dropout layer

        Generates a mask of 1's and 0's with a distribution dictated by the ratio.
        Ratio is the probability of 1 appearing and (1-ratio) is the probability of the 0
        appearing.

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :return: Output matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """
        self.dropout = np.random.choice([0, 1], size=layer_input.shape, p=[1 - self.ratio, self.ratio])
        return layer_input * self.dropout

    def backward(self, layer_input: np.ndarray, layer_output_grad: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        """Derivative of the dropout layer.

        Simply, the nodes which were ignored during the forward pass should also be zeroed out
        during backward pass.

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :param layer_output_grad: Gradient of the output matrix of shape [batch,output_size]
        :type layer_output_grad: np.ndarray
        :param learning_rate: Learning rate during backpropagation
        :type learning_rate: float
        :return: Gradient of the input matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """
        out = self.dropout * layer_output_grad
        return out


class Dense(Layer):
    """Dense layer with weights and biases."""

    def reset(self):
        """Reset weights and biases using the given initializer."""
        super().reset()
        self.weights = self.weight_init.generate(self.input_size, self.output_size)
        self.biases = self.bias_init.generate(1, self.output_size)

    def __init__(self, input_size: int, output_size: int,
                 weight_init: Initializer = GlorotNormal(), bias_init: Initializer = Zero()):
        """Dense layer constructor

        :param input_size: Layer's input size
        :type input_size: int
        :param output_size: Layer's output size
        :type output_size: int
        :param weight_init: Specific initializer for the weights
        :type weight_init: Initializer
        :param bias_init: Specific initializer for the biases
        :type bias_init: Initializer
        """
        super().__init__()
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.input_size = input_size
        self.output_size = output_size
        self.weights = self.weight_init.generate(input_size, output_size)
        self.biases = self.bias_init.generate(1, output_size)

    def forward(self, layer_input: np.ndarray) -> np.ndarray:
        """Forward pass of the dense layer.

        Output is calculated using the following formula:
        output = weights * input + bias

        or:   y = w*x + b

        As this is a linear equation, Dense layer is also called a Linear layer.

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :return: Output matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """

        layer_output = np.dot(layer_input, self.weights) + self.biases
        return layer_output

    def backward(self, layer_input: np.ndarray, layer_output_grad: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        """Perform the backward pass for the dense layer and apply SGD.

        :param layer_input: Input matrix of shape [batch,input_size]
        :type layer_input: np.ndarray
        :param layer_output_grad: Gradient of the output matrix of shape [batch,output_size]
        :type layer_output_grad: np.ndarray
        :param learning_rate: Learning rate during backpropagation
        :type learning_rate: float
        :return: Gradient of the input matrix of shape [batch,output_size]
        :rtype: np.ndarray
        """

        """
        Good read:
        https://stats.stackexchange.com/questions/316029/deep-nns-backpropagation-and-error-calculation        
        
        If during forward pass:
        
        output = weights * input + bias
        
        or 
        
        a^l = w^l * a^(l-1) + b^l
        
        derivative of this w.r.t input is:
        
        d_a^l = w^l
        
        keeping the chain rule in mind, as this layer is a part of the "chain", multiply this
        by (d loss/d_a^l), which is contained in the "layer_output_grad" parameter. In the 
        end the formula is:
        
        delta^l = delta^(l+1) ⋅ (weights^(l+1))^T 
        """

        layer_input_grad = np.dot(layer_output_grad, self.weights.T)

        """
        a^l = w^l * a^(l-1) + b^l
        
        derivative of this w.r.t bias is:
        
        d_a^l = 1
        
        And as before, multiply this 1 with the (d loss/d_a^l), or "layer_output_grad".
        Gradient for the biases is simply the gradient of the layer output.
        However we have to add up the values between batches together
        """

        bias_grad = np.sum(layer_output_grad, axis=0)

        """
        a^l = w^l * a^(l-1) + b^l
        
        derivative of this w.r.t weights is:
        
        d_a^l = a^(l-1)
        
        So multiply a^(l-1) with (d loss/d_a^l), or "layer_output_grad".
        """

        weight_grad = np.dot(layer_input.T, layer_output_grad)

        # Stochastic gradient descent
        self.biases -= learning_rate * bias_grad
        self.weights -= learning_rate * weight_grad

        return layer_input_grad
