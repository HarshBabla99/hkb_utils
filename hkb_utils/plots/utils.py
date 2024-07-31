# Adapted from dynamiqs: https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/plots/utils.py

import numpy as np

from collections.abc import Iterable
from functools import wraps
from math import ceil
from typing import TypeVar, Callable, Sequence

from ..utils import ArrayLike

import pathlib
import shutil
from functools import wraps

import imageio as iio
import IPython.display as ipy
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import colormaps as mpl_cmaps
from matplotlib.cm import ScalarMappable
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm

__all__ = ['gif_it', 'gridplot', 'mplstyle']
# __all__ = [
#     'linmap',
#     'figax',
#     'optional_ax',
#     'gridplot',
#     'mplstyle',
#     'integer_ticks',
#     'sample_cmap',
#     'minorticks_off',
#     'ket_ticks',
#     'bra_ticks',
#     'add_colorbar',
# ]


def linmap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Map $x$ linearly from $[a,b]$ to $[c,d]$."""
    return (x - a) / (b - a) * (d - c) + c


def figax(w: float = 7.0, h: float | None = None, **kwargs) -> tuple[Figure, Axes]:
    """Return a figure with specified width and length."""
    if h is None:
        h = w / 1.6
    return plt.subplots(1, 1, figsize=(w, h), constrained_layout=True, **kwargs)


def optional_ax(func: callable) -> callable:
    """Decorator to build an `Axes` object to pass as an argument to a plot function if it wasn't passed by the user.

    Examples:
        Replace
        ```
        def myplot(ax=None):
            if ax is None:
                _, ax = plt.subplots(1, 1)

            # ...
        ```
        by
        ```
        @optax
        def myplot(ax=None):
            # ...
        ```
    """

    @wraps(func)
    def wrapper(*args, ax: Axes | None = None, w: float = 5.0, h: float | None = None, **kwargs):
        if ax is None:
            _, ax = figax(w=w, h=h)
        return func(*args, ax=ax, **kwargs)

    return wrapper


def gridplot(n: int,
             ncols: int = 1,
             *,
             w: float = 3.0,
             h: float | None = None,
             sharexy: bool = False,
             **kwargs,
            ) -> tuple[Figure, [Axes]]:
    """Returns a figure and a list of subplots organised in a grid.

    Note:
        This method is a shortcut to Matplotlib
        [`plt.subplots()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots).

    Examples:
        For example, to plot six different curves:

        >>> x = np.linspace(0, 1, 101)
        >>> ys = [np.sin(f * 2 * np.pi * x) for f in range(6)]  # (6, 101)

        Replace the usual Matplotlib code

        >>> fig, axs = plt.subplots(
        ...     2, 3, figsize=(3 * 3.0, 2 * 3.0), sharex=True, sharey=True
        ... )
        >>> for i, y in enumerate(ys):
        ...     axs[i // 3][i % 3].plot(x, y)
        [...]
        >>> fig.tight_layout()

        by

        >>> _, axs = gridplot(6, 2, sharexy=True)  # 6 subplots, 2 rows
        >>> for i,y in enumerate(ys):
        ...     axs[i].plot(x, y)
        [...]
    """
    h = w if h is None else h
    nrows = ceil(n / ncols)
    figsize = (w * ncols, h * nrows)

    if sharexy:
        kwargs['sharex'] = True
        kwargs['sharey'] = True

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, **kwargs)

    return fig, axs.flatten()


colors = {'blue'      : '#3a55a4', # dynamiqs blue: 0c5dA5',
          'orange'    : '#ff7722',
          'red'       : '#ff6b6b', # my red (a little too pinkish): #e63e62
          'turquoise' : '#2ec4b6',
          'yellow'    : '#ffc463', # my yellow (okayish): #f69320
          'green'     : '#50c878',
          'grey'      : '#9e9e9e',
          'purple'    : '#845b97',
          'brown'     : '#c0675c',
          'darkgreen' : '#20817d',
          'darkgrey'  : '#666666',
          }


def mplstyle(*, usetex: bool = True):
    r"""Set custom Matplotlib style.

    Examples:
        >>> x = np.linspace(0, 2 * np.pi, 101)
        >>> ys = [np.sin(x), np.sin(2 * x), np.sin(3 * x)]
        >>> mpl_style()
    """
    plt.rcParams.update(
        {
            # xtick
            'xtick.direction': 'in',
            'xtick.major.size': 4.5,
            'xtick.minor.size': 2.5,
            'xtick.major.width': 0.75,
            'xtick.labelsize': 14,
            'xtick.minor.visible': True,
            # ytick
            'ytick.direction': 'in',
            'ytick.major.size': 4.5,
            'ytick.minor.size': 2.5,
            'ytick.major.width': 0.75,
            'ytick.labelsize': 14,
            'ytick.minor.visible': True,
            # axes
            'axes.facecolor': 'white',
            'axes.grid': False,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'axes.linewidth': 0.75,
            'axes.prop_cycle': cycler('color', colors.values()),
            'axes.unicode_minus' : False,
            # grid
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            # legend
            'legend.frameon': False,
            'legend.fontsize': 12,
            # figure
            'figure.facecolor': 'white',
            'figure.dpi': 100,
            'figure.figsize': (5, 5 / 1.6),
            'figure.titlesize': 16,
            # fonts
            'font.family': 'serif' if usetex else 'sanserif',
            'font.size': 14,
            'text.usetex': usetex,
            'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{physics}',
            'mathtext.fontset': 'custom',
            # other
            'savefig.facecolor': 'white',
            'savefig.dpi': 100,
            'scatter.marker': 'o',
            'lines.linewidth': 1.0,
            'pdf.fonttype' : 42,
            'ps.fonttype' : 42,
        }
    )

def integer_ticks(axis: Axis, n: int, all: bool = True):  # noqa: A002
    if all:
        axis.set_ticks(range(n))
        minorticks_off(axis)
    else:
        # let maptlotlib choose major ticks location but restrict to integers
        axis.set_major_locator(MaxNLocator(integer=True))
        # fix minor ticks to integer locations only
        axis.set_minor_locator(MultipleLocator(1))

    # format major ticks as integer
    axis.set_major_formatter(lambda x, _: f'{int(x)}')

def ket_ticks(axis: Axis):
    # fix ticks location
    axis.set_major_locator(FixedLocator(axis.get_ticklocs()))

    # format ticks as ket
    new_labels = [rf'$| {label.get_text()} \rangle$' for label in axis.get_ticklabels()]
    axis.set_ticklabels(new_labels)

def bra_ticks(axis: Axis):
    # fix ticks location
    axis.set_major_locator(FixedLocator(axis.get_ticklocs()))

    # format ticks as ket
    new_labels = [rf'$\langle {label.get_text()} |$' for label in axis.get_ticklabels()]
    axis.set_ticklabels(new_labels)

def sample_cmap(name: str, n: int, alpha: float = 1.0) -> np.ndarray:
    sampled_cmap = mpl_cmaps[name](np.linspace(0, 1, n))
    sampled_cmap[:, -1] = alpha
    return sampled_cmap

def minorticks_off(axis: Axis):
    axis.set_minor_locator(NullLocator())

def add_colorbar(ax: Axes, cmap: str, norm: Normalize, *, 
                 size: str = '5%', pad: str = '5%') -> Axes:
    # append a new axes on the right with the same height
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=size, pad=pad)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable=mappable, cax=cax)
    cax.grid(False)
    return cax


T = TypeVar('T')
def gif_it(plot_function : Callable[[T, ...], None],
          *,
          gif_duration  : float = 5.0,
          fps           : int   = 10,
          filename      : str   = '.tmp/gifit.gif',
          dpi           : int   = 72,
          display       : bool  = True,
          ) -> Callable[[Sequence[T], ...], None]:
    """
    Transform a plot function into a function that creates an animated GIF.

    This function takes a plot function that normally operates on a single input and
    returns a function that creates a GIF from a sequence of inputs.

    Note:
        By default, the GIF is displayed in Jupyter notebook environments.

    Args:
        plot_function: Plot function which must take as first positional argument the
            input that will be sequenced over by the new function. It must create a
            matplotlib `Figure` object and not close it.
        gif_duration: GIF duration in seconds.
        fps: GIF frames per seconds.
        filename: Save path of the GIF file.
        dpi: GIF resolution.
        display: If `True`, the GIF is displayed in Jupyter notebook environments.

    Returns:
        A new function with the same signature as `plot_function` which accepts a
            sequence of inputs and creates a GIF by applying the original
            `plot_function` to each element in the sequence.
    """

    @wraps(plot_function)
    def wrapper(items: ArrayLike, *args, **kwargs) -> None:
        nframes = int(gif_duration * fps)

        nitems = len(items)
        if nframes >= nitems:
            indices = np.arange(nitems)
        else:
            indices = np.round(np.linspace(0, nitems - 1, nframes)).astype(int)

        try:
            # create temporary directory
            tmpdir = pathlib.Path('./.tmp/')
            tmpdir.mkdir(parents=True, exist_ok=True)

            frames = []
            for i, idx in enumerate(tqdm(indices)):
                # ensure previous plot is closed
                plt.close()

                # plot frame
                plot_function(items[idx], *args, **kwargs)

                # save frame in temporary file
                frame_filename = tmpdir / f'tmp-{i}.png'
                plt.gcf().savefig(frame_filename, bbox_inches='tight', dpi=dpi)
                plt.close()

                # read frame with imageio
                frame = iio.v3.imread(frame_filename)
                frames.append(frame)

            # duration: duration per frame in ms
            # loop=0: loop the GIF forever
            # rescale duration to account for eventual duplicate frames
            duration = int(1000 / fps * nframes / len(indices))
            iio.v3.imwrite(filename, frames, format='GIF', duration=duration, loop=0)
            if display:
                ipy.display(ipy.Image(filename))
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir, ignore_errors=True)

    return wrapper