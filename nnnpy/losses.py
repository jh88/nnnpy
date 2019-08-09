import numpy as np

from .activations import softmax


class Loss():
    def __init__(self, func, func_delta):
        self._func = func
        self._func_delta = func_delta

    def __call__(self, target, output, **kwargs):
        return self._func(target, output, **kwargs)

    def delta(self, target, output, **kwargs):
        return self._func_delta(target, output, **kwargs)


def _preprocess_output(output, from_logits=False, axis=-1):
    """Preprocess the predicted output.
    
    If the predict output is still logits, do the softmax
    transformation. Otherwise make sure the output sums to one.

    Arguments:
        output: `numpy.array`, logits or softmat output.
        from_logits: boolean, whether the output is logits.
        axis: integer, axis along which the softmax is applied.

    Returns:
        a `numpy.array` of softmax outputs.

    """
    if from_logits:
        output = softmax(output, axis=axis)
    else:
        output /= output.sum(axis=axis, keepdims=True)
    # To aviod `np.log(0)`
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return output


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    Arguments
        target: `numpy.array`, ground truth label.
        output: `numpy.array`, predicted softmax output or logits.
        from_logits: boolean, whether the output is logits.
        axis: integer, axis along which the softmax is applied.

    Returns
        a `numpy.array` of losses.
    """
    output = _preprocess_output(output, from_logits, axis)
    return -np.sum(target * np.log(output), axis=axis, keepdims=False)


def categorical_crossentropy_delta(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy loss w.r.t the logits.

    Arguments
        target: `numpy.array`, ground truth label.
        output: `numpy.array`, predicted softmax output or lgoits.
        from_logits: boolean, whether the output is logits.
        axis: integer, axis along which the softmax is applied.

    Returns
        a `numpy.array` of losses.
    """
    output = _preprocess_output(output, from_logits, axis)
    return output - target


def get(name):
    """Get the loss function by name.

    Arguments:
        name: str, name of the function.

    Returns:
        The loss function.

    Raises:
        ValueError if unknown name.
    """
    if name in globals():
        return Loss(
            func=globals()[name],
            func_delta=globals().get('{}_delta'.format(name))
        )
    else:
        raise ValueError('Unknown loss function: {}'.format(name))
