import numpy as np


class Optimizer():
    """Abstract class
    """
    pass


class SGD(Optimizer):
    """Stochastic gradient descent.

    Arguments:
        lr: float, learning rate.
        momentum: float, accumulated gradient decay rate.
        decay: float, leanring rate decay hyperparameter.
        nesterov: boolean, whether to use standard momentum or
            Nesterov accelerated gradient.
    """
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov

    def build(self, layer):
        """Initialize the gradient accumulator in the given layer.

        Arguments:
            layer: `layer` instance, layer with trainable parameters.
        """
        layer.velocities = [np.zeros(p.shape) for p in layer.params]

    def update(self, layer, iterations):
        """Update the parameters in the given layer.

        Arguments:
            layer: `layer` instance, layer with trainable parameters.
            iterations: integer, current epoch
        """
        lr = self.lr
        # if learning rate decay is enabled.
        if self.decay > 0:
            lr = lr * (1. / (1. + self.decay * iterations))

        iterable = zip(layer.params, layer.grads, layer.velocities)
        for i, (p, g, v) in enumerate(iterable):
            # Calculate new step using the accumulated velocity
            # and the current gradients.
            v = self.momentum * v - lr * g

            # Update parameters with standard momentum or Nesterov
            if self.nesterov:
                p = p + self.momentum * v - lr * g
            else:
                p = p + v

            layer.params[i] = p
            if self.momentum > 0:
                layer.velocities[i] = v


class RMSprop(Optimizer):
    """RMSprop optimizer.

    Arguments:
        lr: float, learning rate.
        rho: float, accumulated squared gradient decay rate.
        epsilon: float, small number added to avoid dividing by zero.
        decay: float, leanring rate decay hyperparameter.
    """
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-7, decay=0.):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay

    def build(self, layer):
        """Initialize the squared gradient accumulator in the given layer.

        Arguments:
            layer: `layer` instance, layer with trainable parameters.
        """
        layer.accumulators = [np.zeros(p.shape) for p in layer.params]

    def update(self, layer, iterations):
        """Update the parameters in the given layer.

        Arguments:
            layer: `layer` instance, layer with trainable parameters.
            iterations: integer, current epoch
        """
        lr = self.lr
        if self.decay > 0:
            lr = lr * (1. / (1. + self.decay * iterations))

        iterable = zip(layer.params, layer.grads, layer.accumulators)
        for i, (p, g, a) in enumerate(iterable):
            # Update the accumulator
            a = self.rho * a + (1. - self.rho) * np.square(g)

            # Update the parameters
            p = p - lr * g / (np.sqrt(a) + self.epsilon)

            layer.params[i] = p
            layer.accumulators[i] = a


class Adam(Optimizer):
    """Adam optimizer.

    Arguments:
        lr: float, learning rate.
        beta_1: float, accumulated gradient decay rate.
        beta_2: float, accumulated squared gradient decay rate.
        epsilon: float, small number added to avoid dividing by zero.
        decay: float, leanring rate decay hyperparameter.
        amsgrad: boolean, whether to use AMSGrad or Adam.
    """
    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        decay=0.,
        amsgrad=False
    ):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon,
        self.decay = decay
        self.amsgrad = amsgrad

    def build(self, layer):
        """Initialize the accumulators in the given layer.

        Arguments:
            layer: `layer` instance, layer with trainable parameters.
        """
        layer.ms = [np.zeros(p.shape) for p in layer.params]
        layer.vs = [np.zeros(p.shape) for p in layer.params]
        if self.amsgrad:
            layer.vhats = [np.zeros(p.shape) for p in layer.params]
        else:
            layer.vhats = [np.zeros(1) for _ in layer.params]

    def update(self, layer, iterations):
        """Update the parameters in the given layer.

        Arguments:
            layer: `layer` instance, layer with trainable parameters.
            iterations: integer, current epoch
        """
        lr = self.lr
        if self.decay > 0:
            lr = lr * (1. / (1. + self.decay * iterations))

        #bias correction
        t = iterations + 1
        lr = (
            lr
            * np.sqrt(1. - np.power(self.beta_2, t))
            / (1 - np.power(self.beta_1, t))
        )

        iterable = zip(layer.params, layer.grads, layer.ms, layer.vs, layer.vhats)
        for i, (p, g, m, v, vhat) in enumerate(iterable):
            # Update accumulators.
            m = self.beta_1 * m + (1. - self.beta_1) * g
            v = self.beta_2 * v + (1. - self.beta_2) * np.square(g)

            # Update parameters.
            if self.amsgrad:
                vhat = np.maximum(vhat, v)
                p = p - lr * m / (np.sqrt(vhat) + self.epsilon)
            else:
                p = p - lr * m / (np.sqrt(v) + self.epsilon)

            layer.params[i] = p
            layer.ms[i] = m
            layer.vs[i] = v
            if self.amsgrad:
                layer.vhats[i] = vhat


def get(name):
    """Get the optimizer by name.

    Arguments:
        name: one of
            - str, name of the optimizer
            - Optimizer instance, will be returned unchanged

    Returns:
        An Optimizer instance.

    Raises:
        ValueError if unknown name.
    """
    all_classes = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adam': Adam
    }
    if isinstance(name, Optimizer):
        return name
    elif isinstance(name, str) and name.lower() in all_classes:
        return all_classes[name.lower()]()
    else:
        raise ValueError('Unknown optimizer: {}'.format(name))
