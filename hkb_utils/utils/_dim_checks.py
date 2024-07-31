# Adapted from dynamiqs: https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/_checks.py

from numpy import issubdtype, integer
from qutip import Qobj
from .quantum_utils import Array, ArrayLike

####################################################################################################
# Check shape
_is_perfect_square = lambda n: int(n**0.5) ** 2 == n
_cases = {
    '(..., n, 1)'   : lambda ndim, shape: ndim >= 2      and shape[-1] == 1,
    '(..., 1, n)'   : lambda ndim, shape: ndim >= 2      and shape[-2] == 1,
    '(..., n, n)'   : lambda ndim, shape: ndim >= 2      and shape[-2] == shape[-1],
    '(N, ..., n, n)': lambda ndim, shape: ndim >= 3      and shape[-2] == shape[-1],
    '(..., m, n)'   : lambda ndim, shape: ndim >= 2,
    '(..., n)'      : lambda ndim, shape: ndim >= 1,
    '(n, 1)'        : lambda ndim, shape: ndim == 2      and shape[-1] == 1,
    '(1, n)'        : lambda ndim, shape: ndim == 2      and shape[-2] == 1,
    '(n, n)'        : lambda ndim, shape: ndim == 2      and shape[-2] == shape[-1],
    '(n,)'          : lambda ndim, shape: ndim == 1,
    '(N, n, 1)'     : lambda ndim, shape: ndim == 3      and shape[-1] == 1,
    '(N, n, n)'     : lambda ndim, shape: ndim == 3      and shape[-2] == shape[-1],
    '(?, n, 1)'     : lambda ndim, shape: 2 <= ndim <= 3 and shape[-1] == 1,
    '(?, n, n)'     : lambda ndim, shape: 2 <= ndim <= 3 and shape[-2] == shape[-1],
    '(..., n^2, 1)' : lambda ndim, shape: ndim >= 2
                                          and _is_perfect_square(shape[-2])
                                          and shape[-1] == 1,
}

def has_shape(x: ArrayLike, shape: str) -> bool:
    if isinstance(x, Qobj):
        ndim_of_x = x.data.ndim
        shape_of_x = x.shape
    else:
        # handle list[...], np.array[...] where ... = Number, Array, Qobj
        # TODO: current approach is super inefficient, I could try and avoid recreating full matrices
        if isinstance(x, list):
            x = Array(x)

        ndim_of_x = x.ndim
        shape_of_x = x.shape

    if shape in _cases:
        return _cases[shape](ndim_of_x, shape_of_x)
    else:
        raise ValueError(f'Unknown shape specification `{shape}`.')

def check_shape(x: ArrayLike, argname: str, *shapes: str, subs: dict[str, str] | None = None):
    # subs is used to replace symbols in the error message
    # e.g. specify a shape (?, n, n) but print an error message with (nH?, n, n), by passing
    # subs={'?': 'nH?'} to replace the '?' by 'nH?' in the shape specification

    for shape in shapes:
        if has_shape(x, shape):
            return

    if len(shapes) == 1:
        shapes_str = shapes[0]
    else:
        shapes_str = ', '.join(shapes[:-1]) + ' or ' + shapes[-1]

    if subs is not None:
        for k, v in subs.items():
            shapes_str = shapes_str.replace(k, v)

    raise ValueError(f'Argument `{argname}` must have shape {shapes_str}, but has shape' 
                     f' {argname}.shape={x.shape}.')

####################################################################################################
# Check times
def check_times(x: ArrayLike, argname: str, allow_empty: bool = False) -> Array:
    # check that an array of time is valid (it must be a 1D array sorted in strictly
    # ascending order)

    # this function should be used as e.g. `x = check_times(x, 'x')`, and the returned
    # value should be used, otherwise the final check will be removed as part of dead
    # code elimination, see https://docs.kidger.site/equinox/api/errors/ for more
    # details
    x = Array(x)

    if x.ndim != 1:
        raise ValueError(f'Argument {argname} must be a 1D array, but is a {x.ndim}D array.')

    if not allow_empty and len(x) == 0:
        raise ValueError(f'Argument {argname} must contain at least one element.')

    # TODO[dynamiqs]: make JIT compatible (see their code)
    if not any(x[1:] > x[:-1]):
        raise ValueError(f'Argument {argname} must be sorted in strictly ascending order.')

####################################################################################################
# Check if array of ints
def check_type_int(x: Array, argname: str):
    if not issubdtype(x.dtype, integer):
        raise ValueError(f'Argument {argname} must be of type integer, but is of type'
                         f' {argname}.dtype={x.dtype}.')