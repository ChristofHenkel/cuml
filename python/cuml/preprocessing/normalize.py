import cupy as cp


def _handle_zeros_in_scale(scale, copy=True):
    """ Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features."""

    # if we are fitting on 1D arrays, scale might be a scalar
    if cp.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, cp.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def normalize(X, norm='l2', axis=1, return_norm=False):
    """
    Normalizes array by givven norm.

    Parameters
    ----------
    X : 2d array
    norm : one of 'l1', 'l2', 'max'
    axis : either 0 or 1
    return_norm : wether to return calculated norms together with normed array

    Returns
    -------
    X : normalized array
    norms (optional) : norms of X
    """

    # TODO add suppport for sparse X

    if isinstance(X, list):
        X = cp.asarray(X)

    if norm == 'l1':
        norms = cp.abs(X).sum(axis=1)
    elif norm == 'l2':
        norms = cp.sqrt(cp.einsum('ij,ij->i', X, X))
    elif norm == 'max':
        norms = cp.max(X, axis=1)
    norms = _handle_zeros_in_scale(norms, copy=False)
    X = cp.divide(X, norms[:, cp.newaxis])

    if axis == 0:
        X = X.T

    if return_norm:
        return X, norms
    else:
        return X
