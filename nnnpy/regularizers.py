import numpy as np


class Regularizer():
    """Abstract class
    """
    pass


class L1(Regularizer):
    """L1 regularizer

    Arguments:
        a: float, regularization factor.
    """
    def __init__(self, a=0.01):
        self.a = a

    def __call__(self, x):
        return np.sum(self.a * np.abs(x))

    def deriv(self, x):
        # the derivatives will be 1 * regularization factor
        return np.full(x.shape, self.a)


class L2(Regularizer):
    """L2 regularizer

    Arguments:
        a: float, regularization factor.
    """
    def __init__(self, a=0.01):
        self.a = a

    def __call__(self, x):
        return np.sum(self.a * np.square(x))

    def deriv(self, x):
        return 2 * self.a * x


def get(name):
    """Get the regularizer by name.

    Arguments:
        name: one of
            - str, name of the regularizer
            - Regularizer instance, will be returned unchanged

    Returns:
        An Regularizer instance.

    Raises:
        ValueError if unknown name.
    """
    all_classes = {
        'l1': L1,
        'l2': L2
    }
    if name is None:
        return
    if isinstance(name, Regularizer):
        return name
    elif isinstance(name, str) and name.lower() in all_classes:
        return all_classes[name.lower()]()
    else:
        raise ValueError('Unknown regularizer: {}'.format(name))
