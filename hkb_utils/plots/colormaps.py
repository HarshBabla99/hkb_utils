# Adapted from dynamiqs: https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/plots/colormaps.py

import warnings

import matplotlib
from cmasher import copper
from matplotlib.colors import LinearSegmentedColormap

__all__ = []

# diverging (very french! blue white red)
cmap_colors = [
    (0.0,   '#05527B'),
    (0.225, '#639DC1'),
    (0.5,   '#FFFFFF'),
    (0.775, '#E27777'),
    (1.0,   '#BF0C0C'),
]
cmap = LinearSegmentedColormap.from_list('dq_bwr', cmap_colors)
matplotlib.colormaps.register(cmap)

# cyclic colormap (I don't like this)
#cmap_colors = [
#    (0.0,   '#07689D'),
#    (0.25,  '#AC98AB'),
#    (0.5,   '#C62525'),
#    (0.75,  '#5E1A5B'),
#    (1.0,   '#07689D'),
#]
#cmap = LinearSegmentedColormap.from_list('dq_cyclic', cmap_colors)
#matplotlib.colormaps.register(cmap)

# cyclic copper colormap
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='cmasher')

#cmap = copper
#cmap.name = 'cmr_copper'
#matplotlib.colormaps.register(cmap)
