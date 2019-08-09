import numpy as np


def to_categorical(y, num_classes=None):
    """ Converts a list of class integers to one hot encoded labels.

    Arguments
        y: list or `numpy.array`, a list of class integers,
            from 0 to num_classes.
        num_classes: integer, total number of classes.

    Returns
        A `numpy.array` of one hot encoded labels.
    """
    y = np.array(y, dtype='int')
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
