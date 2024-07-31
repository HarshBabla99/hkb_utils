# Adapted from dynamiqs: https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/utils/utils/general.py

from numpy import allclose, ndindex
from qutip import Qobj
from .quantum_utils import Array, ArrayLike

def isket(x: ArrayLike) -> bool:
    r"""Returns True if the array is in the format of a ket.

    Args:
        x _(array_like of shape (...))_: Array.

    Returns:
        True if the last dimension of `x` is 1, False otherwise.
    """
    if isinstance(x, list):
        x = Array(x)
    return x.shape[-1] == 1

def isbra(x: ArrayLike) -> bool:
    r"""Returns True if the array is in the format of a bra.

    Args:
        x _(array_like of shape (...))_: Array.

    Returns:
        True if the second to last dimension of `x` is 1, False otherwise.
    """
    if isinstance(x, list):
        x = Array(x)
    return x.shape[-2] == 1

def isop(x: ArrayLike) -> bool:
    r"""Returns True if the array is in the format of an operator.

    Args:
        x _(array_like of shape (...))_: Array.

    Returns:
        True if the last two dimensions of `x` are equal, False otherwise.
    """
    if isinstance(x, list):
        x = Array(x)
    return x.shape[-1] == x.shape[-2]


def isherm(x: ArrayLike, rtol: float = 1e-5, atol: float = 1e-8) -> bool | list[bool]:
    r"""Returns True if the array is Hermitian. If a list of matrices/Qobjs is provided, returns a list of bools.

    Args:
        x _(array_like of shape (..., n, n))_: Array.
        rtol: Relative tolerance of the check. (TODO: unused)
        atol: Absolute tolerance of the check. (TODO: unused)

    Returns:
        True if `x` is Hermitian, False otherwise.
    """
    if isinstance(x, Qobj):
        return x.isherm

    if check_shape(x, 'x', '(n,n)'):
        return Qobj(x).isherm

    if check_shape(x, 'x', '(..., n, n)'):
        isherm_list = []
        leading_shape = x.shape[:-2]

        for index in numpy.ndindex(leading_shape):
            qobj = Qobj(x[index])
            isherm_list.append(qobj.isherm)

        return isherm_list

def isdm(x:ArrayLike):
    r"""Returns True if the array is in the format of a density matrix. (i.e. square and Hermitian)

    Args:
        x _(array_like of shape (...))_: Array.

    Returns:
        True if the last two dimensions of `x` are equal, False otherwise.
    """
    return isop(x) and isherm(x)