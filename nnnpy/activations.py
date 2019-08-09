import numpy as np


class Activation():
    def __init__(self, func, func_deriv):
        self._func = func
        self._func_deriv = func_deriv

    def __call__(self, x, **kwargs):
        return self._func(x, **kwargs)

    def deriv(self, x, **kwargs):
        return self._func_deriv(x, **kwargs)


def softmax(x, axis=-1): 
    """Stable softmax activation function.

    Arguments:
        x: input tensor.
        axis: integer, axis along which the softmax is applied.

    Returns:
        a `numpy.array` of softmax output.
    """
    y = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return y / np.sum(y, axis=axis, keepdims=True)


def softmax_deriv(x, axis=-1):
    raise NotImplementedError


def relu(x, derivative=False):
    """Relu activation function.
    
    Arguments:
        x: `numpy.array`, input tensor.
        derivative: bool, returns derivatives instead if true.

    Returns:
        a `numpy.array` of relu output.
    """
    return x * (x >= 0)


def relu_deriv(x):
    """Relu activation function derivatives.
    
    Arguments:
        x: `numpy.array`, input tensor.

    Returns:
        a `numpy.array` of derivatives.
    """
    return 1. * (x >= 0)


def tanh(x, derivative=False):
    """Tanh activation function.
    
    Arguments:
        x: `numpy.array`, input tensor.

    Returns:
        a `numpy.array` of tanh output.
    """
    return np.tanh(x)


def tanh_deriv(x):
    """Tanh activation function derivatives.
    
    Arguments:
        x: `numpy.array`, input tensor.

    Returns:
        a `numpy.array` of derivatives.
    """
    return 1. - np.tanh(x)**2


def linear(x, derivative=False):
    """Linear (i.e. identity) activation function.

    Arguments:
        x: input tensor.

    Returns:
        input tensor.
    """
    return x


def linear_deriv(x):
    """Linear activation function derivatives.
    
    Arguments:
        x: `numpy.array`, input tensor.

    Returns:
        a `numpy.array` of derivatives.
    """
    return np.ones_like(x)


def get(name):
    """Get the activation function by name.

    Arguments:
        name: str, name of the function.

    Returns:
        The activation function.

    Raises:
        ValueError if unknown name.
    """
    if name is None:
        return
    elif name in globals():
        return Activation(
            func=globals()[name],
            func_deriv=globals().get('{}_deriv'.format(name))
        )
    else:
        raise ValueError('Unknown activation function: {}'.format(name))
