import numpy as np

from . import activations
from . import initializers
from . import regularizers


class Layer():
    """Abstract class
    """
    def add_to_loss(self):
        return 0

class Dense(Layer):
    """The fully conncected layer

    Arguments:
        output_dim: integer, the number of output units from this layer.
        input_dim: integer, the number of input units, only needed for
            the first layer in the network.
        activation: str, the name of the activation function.
        weights_initializer: str, the name of the weights initializer.
        weights_regularizer: str or Regularizer instance, the weights regularizer.
        bias_initializer: str, the bias initializer.
        bias_regularizer: str or Regularizer instance, the bias regularizer.

    """
    def __init__(
        self,
        output_dim,
        input_dim=None,
        activation=None,
        weights_initializer='glorot_uniform',
        weights_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None
    ):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activations.get(activation)
        self.weights_initializer = initializers.get(weights_initializer)
        self.weights_regularizer = regularizers.get(weights_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # if this layer has trainable parameters
        self.trainable = True

    def build(self, input_dim):
        """Initialize the layer.

        Arguments:
            input_dim: integer, the shape of inputs, excluding batch_size.
        """
        # Initialize weights, bias and their gradients.
        w = self.weights_initializer(shape=(input_dim, self.output_dim))
        b = self.bias_initializer(shape=(self.output_dim,))
        grad_w = np.zeros_like(w)
        grad_b = np.zeros_like(b)

        self.params = [w, b]
        self.grads = [grad_w, grad_b]

    def forward(self, inputs, **kwargs):
        """Forward propagation step.

        Arguments:
            inputs: `numpy.ndarray`, with a shape of (N, input_dim).

        Returns:
            a `numpy.arange` with a shape of (N, output_dim).
        """
        w, b = self.params
        # linear transformation.
        output = np.dot(inputs, w) + b
        self._h = output.copy()
        # non linear transformation if activation is set.
        if self.activation is not None:
            output = self.activation(output)
        self._inputs = inputs
        return output

    def backward(self, delta, output_layer=False):
        """Backpropagation step.

        Arguments:
            delta: `numpy.ndarray`, the derivative of loss w.r.t the
                output of current layer.
            output_layer: boolean, as the derivatives returned from
                the loss function are w.r.t the hiiden layer, will skip
                the activation derivative if set to True.

        Returns:
            a `numpy.array` of derivatives w.r.t the input of current layer.
        """
        _delta = np.atleast_2d(delta)

        # If this is not the last layer, multiply the delta with the
        # derivatives of activation function w.r.t the hidden layer.
        if not output_layer and self.activation is not None:
            _delta = _delta * self.activation.deriv(self._h)

        # Update the weights gradients
        self.grads[0] = (
            np.atleast_2d(self._inputs).T.dot(_delta) / self._inputs.shape[0]
        )
        # Add the regularization gradients.
        if self.weights_regularizer is not None:
            self.grads[0] += self.weights_regularizer.deriv(self.params[0])

        # Update bias gradients.
        self.grads[1] = _delta.mean(axis=0)
        # Add the regularization gradients.
        if self.bias_regularizer is not None:
            self.grads[1] += self.bias_regularizer.deriv(self.params[1])

        # Calculate the derivatives w.r.t the input.
        _delta = _delta.dot(self.params[0].T)

        self._h = None
        self._inputs = None

        return _delta

    def add_to_loss(self):
        """Add loss from parts other than the loss function.

        Returns:
            an integer of additional loss
        """
        loss = 0
        # Additional loss from weights regularization.
        if self.weights_regularizer is not None:
            loss += self.weights_regularizer(self.params[0])
        # Additional loss from bias regularization.
        if self.bias_regularizer is not None:
            loss += self.bias_regularizer(self.params[1])
        return loss


class Activation(Layer):
    """The activation layer

    Arguments:
        activation: str, the name of the activation function.
    """
    def __init__(self, activation):
        self.activation = activations.get(activation)

        self.trainable = False

    def build(self, input_dim):
        """Initialize the layer.

        Arguments:
            input_dim: integer, the shape of inputs, excluding batch_size.
        """
        self.output_dim = input_dim

    def forward(self, inputs, **kwargs):
        """Forward propagation step.

        Arguments:
            inputs: `numpy.ndarray`, with a shape of (N, input_dim).

        Returns:
            a `numpy.arange` with a shape of (N, output_dim).
        """
        self._inputs = inputs
        return self.activation(inputs)

    def backward(self, delta):
        """Backpropagation step.

        Arguments:
            delta: `numpy.ndarray`, the derivative of loss w.r.t the
                output of current layer.
            output_layer: boolean, as the derivatives returned from
                the loss function are w.r.t the hiiden layer, will skip
                the activation derivative if set to True.

        Returns:
            a `numpy.array` of derivatives w.r.t the input of current layer.
        """
        _delta = np.atleast_2d(delta)
        return _delta * self.activation.deriv(self._inputs)


class Dropout(Layer):
    """The dropout layer

    Arguments:
        rate: float, between 0 and 1. Fraction of the input units to keep.
    """
    def __init__(self, rate):
        self.rate = min(1., max(0., rate))

        self.trainable = False

    def build(self, input_dim):
        """Initialize the layer.

        Arguments:
            input_dim: integer or a tuple of intergers, the shape of inputs,
                excluding batch_size.
        """
        # Set the output shape which is the same as the input shape
        self.output_dim = input_dim

    def forward(self, inputs, training):
        """Forward propagation step.

        During training step, randomly change input units to 0 based
        on the `rate`.

        Arguments:
            inputs: `numpy.ndarray`, with a shape of (N, input_dim).
            training: boolean, specify if it's training step.

        Returns:
            a `numpy.arange` with a shape of (N, output_dim).
        """
        if training:
            # The dropout mask is resacled by dividing `rate`.
            self._noise =  (
                np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
            )

            output = inputs * self._noise
            return output
        else:
            return inputs

    def backward(self, delta):
        """Backpropagation step.

        Arguments:
            delta: `numpy.ndarray`, the derivative of loss w.r.t the
                output of current layer.

        Returns:
            a `numpy.array` of derivatives w.r.t the input of current layer.
        """
        _delta = delta * self._noise
        self._noise = None
        return _delta


class BatchNormalization(Layer):
    """The batch normalization layer

    Arguments:
        momentum: float, for the moving mean and variance.
        epsilon: float, small number added to avoid dividing by zero.
        beta_initializer: str, beta weights initializer.
        gamma_initializer: gamma weights initializer.
        moving_mean_initializer: moving mean initializer.
        moving_variance_initializer: moving variance initializer.
    """
    def __init__(
        self,
        momentum=0.99,
        epsilon=1e-3,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones'
    ):
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer
        )

        self.trainable = True

    def build(self, input_dim):
        """Initialize the layer.

        Arguments:
            input_dim: integer, the shape of inputs, excluding batch_size.
        """
        # Initialize weights, bias and their gradients.
        self.output_dim = input_dim

        # Initialize weights and gradients.
        gamma = self.gamma_initializer((input_dim,))
        beta = self.beta_initializer((input_dim,))
        grad_gamma = np.zeros_like(gamma)
        grad_beta = np.zeros_like(beta)
        self.params = [gamma, beta]
        self.grads = [grad_gamma, grad_beta]

        # Initialize moving mean and moving variance.
        self._moving_mean = self.moving_mean_initializer((input_dim,))
        self._moving_variance = self.moving_variance_initializer((input_dim,))

    def forward(self, inputs, training):
        """Forward propagation step.

        During training step, normalizing with batch mean and variance,
        and updating the moving mean and moving variance. During inference,
        using the moving mean and moving variance to normalize the iputs.

        Arguments:
            inputs: `numpy.ndarray`, with a shape of (N, input_dim).
            training: boolean, specify if it's training step.

        Returns:
            a `numpy.arange` with a shape of (N, output_dim).
        """
        gamma, beta = self.params
        if training:
            mu = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)

            std = np.sqrt(var) + self.epsilon
            # Normalize the inputs.
            xhat = (inputs - mu) / std
            # Linear transformation.
            output =  gamma * xhat + beta

            # Update moving mean and variance.
            self._moving_mean = (
                self.momentum * self._moving_mean + (1. - self.momentum) * mu
            )
            self._moving_variance = (
                self.momentum * self._moving_variance
                + (1. - self.momentum) * var
            )

            self._std = std
            self._xhat = xhat
        else:
            output = (
                (inputs - self._moving_mean)
                / (np.sqrt(self._moving_variance) + self.epsilon)
            )
            output =  gamma * output + beta

        return output

    def backward(self, delta):
        """Backpropagation step.

        Arguments:
            delta: `numpy.ndarray`, the derivative of loss w.r.t the
                output of current layer.

        Returns:
            a `numpy.array` of derivatives w.r.t the input of current layer.
        """
        _delta = np.atleast_2d(delta)

        # Update the gradients of gamma and beta.
        self.grads[0] = np.mean(self._xhat * _delta, axis=0)
        self.grads[1] = _delta.mean(axis=0)

        # Update the derivatives w.r.t the inputs.
        _delta = _delta * self.params[0]
        _delta = (
            _delta / self._std
            - np.mean(_delta, axis=0) / self._std
            - self._xhat * np.mean(_delta * self._xhat, axis=0)
        )

        self._std = None
        self._xhat = None

        return _delta


class Flatten(Layer):
    """The flatten layer.

    Collapse the inputs into one dimension.
    """
    def __init__(self):
        self.trainable = False

    def build(self, input_dim):
        self.input_dim = tuple(input_dim)
        # Calculate the output shape.
        self.output_dim = np.prod(input_dim)

    def forward(self, inputs, **kwargs):
        return np.reshape(inputs, (len(inputs), -1))

    def backward(self, delta):
        return np.reshape(delta, (len(delta),) + self.input_dim)
