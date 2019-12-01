import numpy
from utils import _ni_support
from utils import _nd_image


def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None):
    """
    Exact euclidean distance transform.
    """

    if (not return_distances) and (not return_indices):
        msg = 'at least one of distances/indices must be specified'
        raise RuntimeError(msg)

    ft_inplace = isinstance(indices, numpy.ndarray)
    dt_inplace = isinstance(distances, numpy.ndarray)
    # calculate the feature transform
    input = numpy.atleast_1d(numpy.where(input, 1, 0).astype(numpy.int8))
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, input.ndim)
        sampling = numpy.asarray(sampling, dtype=numpy.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()

    if ft_inplace:
        ft = indices
        if ft.shape != (input.ndim,) + input.shape:
            raise RuntimeError('indices has wrong shape')
        if ft.dtype.type != numpy.int32:
            raise RuntimeError('indices must be of int32 type')
    else:
        ft = numpy.zeros((input.ndim,) + input.shape, dtype=numpy.int32)

    _nd_image.euclidean_feature_transform(input, sampling, ft)
    # if requested, calculate the distance transform
    if return_distances:
        dt = ft - numpy.indices(input.shape, dtype=ft.dtype)
        dt = dt.astype(numpy.float16)
        if sampling is not None:
            for ii in range(len(sampling)):
                dt[ii, ...] *= sampling[ii]
        numpy.multiply(dt, dt, dt)
        if dt_inplace:
            dt = numpy.add.reduce(dt, axis=0)
            if distances.shape != dt.shape:
                raise RuntimeError('indices has wrong shape')
            if distances.dtype.type != numpy.float64:
                raise RuntimeError('indices must be of float64 type')
            numpy.sqrt(dt, distances)
        else:
            dt = numpy.add.reduce(dt, axis=0)
            dt = numpy.sqrt(dt)

    # construct and return the result
    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None

