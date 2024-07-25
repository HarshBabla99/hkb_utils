from __future__ import annotations

import numpy as np
import qutip as qp
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from ..utils._dim_checks import check_shape, check_times
from ..utils._quantum_checks import isdm, isket

from .utils import (
    add_colorbar,
    colors,
    integer_ticks,
    ket_ticks,
    optional_ax,
    sample_cmap,
)

__all__ = ['plot_fock', 'plot_fock_evolution']


ArrayLike = list | np.ndarray | qp.Qobj


def _populations(x: ArrayLike) -> Array:
    if isket(x):
        x = np.array(x)
        return np.abs(x.squeeze(-1)) ** 2
    elif isdm(x):
        x = np.array(x)
        return np.diag(x).real
    else:
        raise TypeError


@optional_ax
def plot_fock(
    state: ArrayLike,
    *,
    ax: Axes | None = None,
    allxticks: bool = False,
    ymax: float | None = 1.0,
    color: str = colors['blue'],
    alpha: float = 1.0,
    label: str = '',
):
    """Plot the photon number population of a state.
    """
    state = jnp.asarray(state)
    check_shape(state, 'state', '(n, 1)', '(n, n)')

    n = state.shape[0]
    x = range(n)
    y = _populations(state)

    # plot
    ax.bar(x, y, color=color, alpha=alpha, label=label)
    if ymax is not None:
        ax.set_ylim(ymax=ymax)
    ax.set(xlim=(0 - 0.5, n - 0.5))

    # set x ticks
    integer_ticks(ax.xaxis, n, all=allxticks)
    ket_ticks(ax.xaxis)

    # turn legend on
    if label != '':
        ax.legend()


@optional_ax
def plot_fock_evolution(
    states: ArrayLike,
    *,
    ax: Axes | None = None,
    times: ArrayLike | None = None,
    cmap: str = 'Blues',
    logscale: bool = False,
    logvmin: float = 1e-4,
    colorbar: bool = True,
    allyticks: bool = False,
):
    """Plot the photon number population of state as a function of time.
    """
    states = jnp.asarray(states)
    times = jnp.asarray(times) if times is not None else None
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')
    if times is not None:
        times = check_times(times, 'times')

    x = jnp.arange(len(states)) if times is None else times
    n = states[0].shape[0]
    y = range(n)
    z = _populations(states).T

    # set norm and colormap
    if logscale:
        norm = LogNorm(vmin=logvmin, vmax=1.0, clip=True)
        # stepped cmap
        ncolors = jnp.round(jnp.log10(1 / logvmin)).astype(int)
        clist = sample_cmap(cmap, ncolors + 2)[1:-1]  # remove extremal colors
        cmap = ListedColormap(clist)
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)

    # plot
    ax.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    ax.grid(False)

    # set y ticks
    integer_ticks(ax.yaxis, n, all=allyticks)
    ket_ticks(ax.yaxis)

    if colorbar:
        add_colorbar(ax, cmap, norm, size='2%', pad='2%')
