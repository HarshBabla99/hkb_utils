# Adapted from dynamiqs: https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/plots/plots_wigner.py

import numpy as np
from ..utils import ArrayLike

import pathlib
import shutil

import imageio as iio
import IPython.display as ipy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from tqdm import tqdm

from jax.numpy import asarray
from dynamiqs import wigner

from ..utils._dim_checks import check_shape
from .utils import add_colorbar, colors, figax, gridplot, optional_ax

__all__ = ['plot_wigner', 'plot_wigner_mosaic', 'plot_wigner_gif']


@optional_ax
def plot_wigner_data(wigner        : ArrayLike,
                     xmax          : float,
                     ymax          : float,
                     *,
                     ax            : Axes | None = None,
                     vmax          : float       = 2 / np.pi,
                     cmap          : str         = 'dq_bwr',
                     interpolation : str         = 'bilinear',
                     colorbar      : bool        = True,
                     cross         : bool        = False,
                     clear         : bool        = False):

    w = np.array(wigner)
    check_shape(w, 'wigner', '(n, n)')

    # set plot norm
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # clip to avoid rounding errors
    w = w.clip(vmin, vmax)

    # plot
    ax.imshow(w,
              cmap          = cmap,
              norm          = norm,
              origin        = 'lower',
              aspect        = 'equal',
              interpolation = interpolation,
              extent        = [-xmax, xmax, -ymax, ymax])

    # axis label
    ax.set(xlabel=r'$\mathrm{Re}(\alpha)$', ylabel=r'$\mathrm{Im}(\alpha)$')

    if colorbar and not clear:
        cax = add_colorbar(ax, cmap, norm)
        if vmax == 2 / np.pi:
            cax.set_yticks([vmin, 0.0, vmax], labels=[r'$-2/\pi$', r'$0$', r'$2/\pi$'])

    if cross:
        ax.axhline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)
        ax.axvline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)

    if clear:
        ax.grid(False)
        ax.axis(False)


@optional_ax
def plot_wigner(state         : ArrayLike,
                *,
                ax            : Axes | None  = None,
                xmax          : float        = 5.0,
                ymax          : float | None = None,
                vmax          : float        = 2 / np.pi,
                npixels       : int          = 101,
                cmap          : str          = 'dq_bwr',
                interpolation : str          = 'bilinear',
                colorbar      : bool         = True,
                cross         : bool         = False,
                clear         : bool         = False):
    r"""Plot the Wigner function of a state.

    Note:
        Choose a diverging colormap `cmap` for better results.
    """
    state = np.array(state)
    check_shape(state, 'state', '(n, 1)', '(n, n)')

    ymax = xmax if ymax is None else ymax
    _, _, w = wigner(asarray(state), xmax, ymax, npixels)

    plot_wigner_data(w,
                     xmax,
                     ymax,
                     ax            = ax,
                     vmax          = vmax,
                     cmap          = cmap,
                     interpolation = interpolation,
                     colorbar      = colorbar,
                     cross         = cross,
                     clear         = clear)


def plot_wigner_mosaic(states        : ArrayLike,
                       *,
                       n             : int          = 8,
                       ncols         : int          = 3,
                       w             : float        = 3.0,
                       h             : float | None = None,
                       xmax          : float        = 5.0,
                       ymax          : float | None = None,
                       vmax          : float        = 2 / np.pi,
                       npixels       : int          = 101,
                       cmap          : str          = 'dq_bwr',
                       interpolation : str          = 'bilinear',
                       cross         : bool         = False):

    r"""Plot the Wigner function of multiple states in a mosaic arrangement.
    """
    states = np.array(states)
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')

    nstates = len(states)-1
    if nstates < n:
        n = nstates

    # create grid of plot
    _, axs = gridplot(n,
                      ncols       = ncols,
                      w           = w,
                      h           = h,
                      gridspec_kw = dict(wspace=0, hspace=0),
                      sharex      = True,
                      sharey      = True)

    ymax = xmax if ymax is None else ymax
    selected_indexes = np.linspace(0, nstates, n, dtype=int)
    _, _, wig = wigner(asarray(states[selected_indexes]), xmax, ymax, npixels)

    # plot individual wigner
    for i, ax in enumerate(axs):

        plot_wigner_data(wig[i],
                         ax            = ax,
                         xmax          = xmax,
                         ymax          = ymax,
                         vmax          = vmax,
                         cmap          = cmap,
                         interpolation = interpolation,
                         colorbar      = False,
                         cross         = cross,
                         clear         = False)

        ax.set(xlabel='', ylabel='', xticks=[], yticks=[])


def plot_wigner_gif(states        : ArrayLike,
                    *,
                    gif_duration  : float        = 5.0,
                    fps           : int          = 10,
                    w             : float        = 5.0,
                    h             : float | None = None,
                    xmax          : float        = 5.0,
                    ymax          : float | None = None,
                    vmax          : float        = 2 / np.pi,
                    npixels       : int          = 101,
                    cmap          : str          = 'dq',
                    interpolation : str          = 'bilinear',
                    cross         : bool         = False,
                    clear         : bool         = False,
                    filename      : str          = '.tmp/wigner.gif',
                    dpi           : int          = 72,
                    display       : bool         = True):
    r"""
    Plot a GIF of the Wigner function of multiple states.
    """
    states = np.asarray(states)
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')

    ymax = xmax if ymax is None else ymax
    nframes = int(gif_duration * fps)
    selected_indexes = np.linspace(0, len(states)-1, nframes, dtype=int)
    _, _, wig = wigner(asarray(states[selected_indexes]), xmax, ymax, npixels)

    try:
        # create temporary directory
        tmpdir = pathlib.Path('./.tmp/')
        tmpdir.mkdir(parents=True, exist_ok=True)

        frames = []
        for i in tqdm(range(nframes)):
            fig, ax = figax(w=w, h=h)

            plot_wigner_data(wig[i],
                             xmax          = xmax,
                             ax            = ax,
                             ymax          = ymax,
                             vmax          = vmax,
                             cmap          = cmap,
                             interpolation = interpolation,
                             colorbar      = False,
                             cross         = cross,
                             clear         = clear)

            frame_filename = tmpdir / f'tmp-{i}.png'
            fig.savefig(frame_filename, bbox_inches='tight', dpi=dpi)
            plt.close()
            frame = iio.v3.imread(frame_filename)
            frames.append(frame)

        # loop=0: loop the GIF forever
        frame_duration_ms = 1000 * 1 / fps
        iio.v3.imwrite(filename, frames, format='GIF', duration=frame_duration_ms, loop=0)
        if display:
            ipy.display(ipy.Image(filename))
    finally:
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)
