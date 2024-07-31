# Adapted from dynamiqs: https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/plots/plots_hinton.py

from itertools import product

import matplotlib as mpl
import numpy as np

from ..utils import Array, ArrayLike

from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

from ..utils._dim_checks import check_shape
from .utils import add_colorbar, bra_ticks, integer_ticks, ket_ticks, optional_ax


__all__ = ['plot_hinton']


def _normalize(x: ArrayLike, vmin: float, vmax: float) -> Array:
    # linearly normalizes data into the [0.0, 1.0] interval
    # values outside the range [vmin, vmax] are also transformed linearly, resulting in
    # values outside [0, 1]
    x = np.array(x)
    return (x - vmin) / (vmax - vmin)


def _plot_squares(ax      : Axes,
                  areas   : ArrayLike,
                  colors  : ArrayLike,
                  offsets : ArrayLike,
                  ecolor  : str = 'white',
                  ewidth  : float = 0.5):
    # areas: 1D array (n) with real values in [0, 1]
    # colors: 2D array (n, 4) with RGBA values
    # offsets: 2D array (n, 2) with real values in R

    # we convert all inputs to numpy arrays
    # warning: do not use jax arrays here
    areas   = np.array(areas)
    colors  = np.array(colors)
    offsets = np.array(offsets)

    # compute squares side length
    sides = np.sqrt(areas)

    # for efficiency we only keep squares with non-negligible side length >= 0.01 (side
    # length is in [0, 1])
    mask = sides >= 0.01
    sides, colors, offsets = sides[mask], colors[mask], offsets[mask]

    # compute squares corner coordinates
    corners = offsets - sides[..., None] / 2

    patch_list = [patches.Rectangle(xy, side, side, facecolor=color)
                  for xy, side, color in zip(corners, sides, colors)]

    squares = PatchCollection(patch_list, match_original=True, edgecolor=ecolor, linewidth=ewidth)

    ax.add_collection(squares)


@optional_ax
def _plot_hinton(areas       : ArrayLike,
                 colors      : ArrayLike,
                 colors_vmin : float,
                 colors_vmax : float,
                 cmap        : str,
                 *,
                 ax          : Axes | None = None,
                 colorbar    : bool        = True,
                 allticks    : bool        = True,
                 ecolor      : str         = 'white',
                 ewidth      : float       = 0.5):
    # areas: 2D array (n, n) with real values in [0, 1]
    # colors: 2D array (n, n) with real values in [0, 1]
    areas = np.array(areas)
    colors = np.array(colors)

    areas = areas.clip(0.0, 1.0)
    colors = colors.clip(0.0, 1.0)

    # === set axes
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='both', direction='out')
    n = areas.shape[0]
    ax.set(xlim=(-0.5, n - 1 + 0.5), ylim=(-0.5, n - 1 + 0.5))
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    integer_ticks(ax.xaxis, n, all=allticks)
    integer_ticks(ax.yaxis, n, all=allticks)

    # === plot squares
    # squares coordinates (cartesian product of all indices)
    offsets = np.array(list(product(*(range(s) for s in areas.shape))))
    # squares areas
    areas = areas.T.flatten()
    # squares colors
    cmap = mpl.colormaps[cmap]
    colors = cmap(colors.T).reshape(-1, 4)
    _plot_squares(ax, areas, colors, offsets, ecolor=ecolor, ewidth=ewidth)

    # === colorbar
    if colorbar:
        norm = Normalize(colors_vmin, colors_vmax)
        cax = add_colorbar(ax, cmap, norm, size='4%', pad='4%')
        if colors_vmin == -np.pi and colors_vmax == np.pi:
            cax.set_yticks([-np.pi, 0.0, np.pi], labels=[r'$-\pi$', r'$0$', r'$\pi$'])


@optional_ax
def plot_hinton(x          : ArrayLike,
                *,
                ax         : Axes | None      = None,
                cmap       : str | None       = None,
                vmin       : float | None     = None,
                vmax       : float | None     = None,
                colorbar   : bool             = True,
                allticks   : bool             = False,
                tickslabel : list[str] | None = None,
                ecolor     : str              = 'white',
                ewidth     : float            = 0.5,
                clear      : bool             = False):
    """
    Plot a Hinton diagram.
    """
    x = np.asarray(x)
    check_shape(x, 'x', '(n, n)')

    # set different defaults, areas and colors for real matrix, positive real matrix
    # and complex matrix
    if np.isrealobj(x):
        # x: 2D array with real data in [vmin, vmax]

        all_positive = np.all(x >= 0)
        if cmap is None:
            # sequential colormap for positive data, diverging colormap otherwise
            cmap = 'Blues' if all_positive else 'dq_bwr'
        if vmin is None:
            vmin = 0.0 if all_positive else np.min(x)

        vmax = np.max(x) if vmax is None else vmax

        # areas: absolute value of x
        area_max = max(abs(vmin), abs(vmax))
        areas = _normalize(np.abs(x), 0.0, area_max)

        # colors: value of x
        colors = _normalize(x, vmin, vmax)
        colors_vmin, colors_vmax = vmin, vmax

    elif np.iscomplexobj(x):
        # x: 2D array with complex data

        # cyclic colormap for the phase
        cmap = 'cmr_copper' if cmap is None else cmap

        # areas: magnitude of x
        magnitude = np.abs(x)
        areas_max = np.max(magnitude) if vmax is None else vmax
        areas = _normalize(magnitude, 0.0, areas_max)

        # colors: phase of x
        phase = np.angle(x)
        colors = _normalize(phase, -np.pi, np.pi)
        colors_vmin, colors_vmax = -np.pi, np.pi

    if clear:
        colorbar = False

    if tickslabel is not None:
        allticks = True

    _plot_hinton(areas,
                 colors,
                 colors_vmin,
                 colors_vmax,
                 cmap,
                 ax       = ax,
                 colorbar = colorbar,
                 allticks = allticks,
                 ecolor   = ecolor,
                 ewidth   = ewidth)

    # set ticks label format
    if tickslabel is not None:
        ax.xaxis.set_ticklabels(tickslabel)
        ax.yaxis.set_ticklabels(tickslabel)

    ket_ticks(ax.xaxis)
    bra_ticks(ax.yaxis)

    if clear:
        ax.axis(False)
