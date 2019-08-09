import numpy as np
import scipy.stats


def _compute_fans(shape):
    """Get the numbers of input units and output units.

    Arguments:
        shape: a tuple of integers, the shape of the output tensor.

    Returns:
        A tuple of input units and output units counts.
    """
    if len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) in {3,4,5}:
        # Assuming channel first format
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in = fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def zeros(shape):
    """Generate a tensor of zeros.

    Arguments:
        shape: a tuple of integers, the shape of the output tensor.

    Returns:
        a `numpy.array` of zeros.
    """
    return np.zeros(shape)


def ones(shape):
    """Generate a tensor of ones.

    Arguments:
        shape: a tuple of integers, the shape of the output tensor.

    Returns:
        a `numpy.array` of ones.
    """
    return np.ones(shape)


def truncated_normal(shape, mean=0., stddev=1., seed=None):
    """Generate random values from the truncated normal distribution.

    The values are in the range of 2 standard deviations from the mean.

    Arguments:
        shape: a tuple of integers, the shape of the output tensor.
        mean: mean of the values.
        stddev: standard deviation of the values.
        seed: integer, random seed.

    Returns:
        a `numpy.array` of values from the truncated normal distribution.
    """
    rv = scipy.stats.truncnorm(-2, 2, loc=mean, scale=stddev)
    return rv.rvs(np.prod(shape), random_state=seed).reshape(shape)


def uniform(shape, lower, upper, seed=None):
    """Generate random values from the uniform distribution.

    The values are in the range of [lower, upper].

    Arguments:
        shape: a tuple of integers, the shape of the output tensor.
        lower: float, lower bound.
        upper: float, upper bound.
        seed: integer, random seed.

    Returns:
        a `numpy.array` of values from the uniform distribution.
    """
    rv = scipy.stats.uniform(lower, upper - lower)
    return rv.rvs(np.prod(shape), random_state=seed).reshape(shape)


def glorot_normal(shape, seed=None):
    """Glorot normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    Arguments:
        shape: a tuple of integers, the shape of the output tensor.
        seed: integer, random seed.

    Returns:
        a `numpy.array` of values from the truncated normal distribution.
    """
    fan_in, fan_out = _compute_fans(shape)
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return truncated_normal(shape, stddev=stddev, seed=seed)


def glorot_uniform(shape, seed=None):
    """Glorot uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    Arguments:
        shape: a tuple of integers, the shape of the output tensor.
        seed: integer, random seed.

    Returns:
        a `numpy.array` of values from the uniform distribution.
    """
    fan_in, fan_out = _compute_fans(shape)
    upper = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, -upper, upper, seed)


def he_normal(shape, seed=None):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    Arguments:
        shape: a tuple of integers, the shape of the output tensor.
        seed: integer, random seed.

    Returns:
        a `numpy.array` of values from the truncated normal distribution.
    """
    fan_in, _ = _compute_fans(shape)
    stddev = np.sqrt(2 / fan_in)
    return truncated_normal(shape, stddev=stddev, seed=seed)


def he_uniform(shape, seed=None):
    """Glorot uniform initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    Arguments:
        seed: integer, random seed.

    Returns:
        a `numpy.array` of values from the uniform distribution.
    """
    fan_in, _ = _compute_fans(shape)
    upper = np.sqrt(6. / fan_in)
    return uniform(shape, -upper, upper, seed)


def get(name):
    """Get the initializer by name.

    Arguments:
        name: str, name of the initializer.

    Returns:
        The initializer.

    Raises:
        ValueError if unknown name.
    """
    if name in globals():
        return globals()[name]
    else:
        raise ValueError('Unknown initializer: {}'.format(name))
