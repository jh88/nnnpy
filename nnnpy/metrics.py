import numpy as np


def categorical_accuracy(y_true, y_pred, axis=-1):
    """Calculate the accuracy
    
    Number of correct predictions / Total predictions

    Arguments:
        y_true: `numpy.array`, one hot encoded ground truth label.
        y_pred: `numpy.array`, softmax output.

    Returns:
        Accuracy
    """
    return (np.argmax(y_true, axis=axis) == np.argmax(y_pred, axis=axis)).mean()


def get(name):
    """Get the mertic by name.

    Arguments:
        name: str, name of the mertic.

    Returns:
        The mertic.

    Raises:
        ValueError if unknown name.
    """
    if name in globals():
        return globals()[name]
    else:
        raise ValueError('Unknown mertic: {}'.format(name))
