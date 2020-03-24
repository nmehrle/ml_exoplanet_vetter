#
# Copyright (C) 2017 - Massachusetts Institute of Technology (MIT)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Plotting utilities.
"""

import numpy as np

try:
    import Tkinter
    mpl_backend = 'TkAgg'
except ImportError:
    mpl_backend = 'Agg'


# each pixel is 0.015 mm
# nominal offset from boresight is 30.72 mm

GEOM = {
    'cam_min_x': 0,
    'cam_max_x': 4272,
    'cam_min_y': 0,
    'cam_max_y': 4156,
    'ccd_width': 2048,
    'ccd_height': 2058,
    'ccd_buf_x': 44,
    'ccd_buf_y': 20,
    'mm_per_pixel': 0.015,
}


def draw_ccd_outlines(ax, crop_outliers=True, geom=GEOM):
    """draw the outlines of each CCD, and a label with the CCD number"""
    from matplotlib import collections
    textalpha = 0.25
    linealpha = 0.15
    offset = 50
    # lower left
    ax.add_collection(collections.LineCollection(
        [[(geom['ccd_buf_x'] + geom['ccd_width'], 0),
          (geom['ccd_buf_x'] + geom['ccd_width'], geom['ccd_height'])], # right
         [(geom['ccd_buf_x'] + geom['ccd_width'], geom['ccd_height']),
          (geom['ccd_buf_x'], geom['ccd_height'])], # top
         [(geom['ccd_buf_x'], geom['ccd_height']),
          (geom['ccd_buf_x'], 0)]], # left
        alpha=linealpha))
    ax.text(offset, offset, 'CCD3', alpha=textalpha)
    # upper left
    ax.add_collection(collections.LineCollection(
        [[(geom['ccd_buf_x'] + geom['ccd_width'], geom['cam_max_y']),
          (geom['ccd_buf_x'] + geom['ccd_width'],
           geom['cam_max_y'] - geom['ccd_height'])],
         [(geom['ccd_buf_x'] + geom['ccd_width'],
           geom['cam_max_y'] - geom['ccd_height']),
          (geom['ccd_buf_x'], geom['cam_max_y'] - geom['ccd_height'])],
         [(geom['ccd_buf_x'], geom['cam_max_y'] - geom['ccd_height']),
          (geom['ccd_buf_x'], geom['cam_max_y'])]],
        alpha=linealpha))
    ax.text(offset, offset + geom['ccd_height'] + geom['ccd_buf_y'],
            'CCD2', alpha=textalpha)
    # upper right
    ax.add_collection(collections.LineCollection(
        [[(geom['cam_max_x'] - (geom['ccd_buf_x'] + geom['ccd_width']),
           geom['cam_max_y']),
          (geom['cam_max_x'] - (geom['ccd_buf_x'] + geom['ccd_width']),
           geom['cam_max_y'] - geom['ccd_height'])],
         [(geom['cam_max_x'] - (geom['ccd_buf_x'] + geom['ccd_width']),
           geom['cam_max_y'] - geom['ccd_height']),
          (geom['cam_max_x'] - geom['ccd_buf_x'],
           geom['cam_max_y'] - geom['ccd_height'])],
         [(geom['cam_max_x'] - geom['ccd_buf_x'],
           geom['cam_max_y'] - geom['ccd_height']),
          (geom['cam_max_x'] - geom['ccd_buf_x'], geom['cam_max_y'])]],
        alpha=linealpha))
    ax.text(offset + geom['ccd_width'] + 2 * geom['ccd_buf_x'],
            offset + geom['ccd_height'] + geom['ccd_buf_y'],
            'CCD1', alpha=textalpha)
    # lower right
    ax.add_collection(collections.LineCollection(
        [[(geom['cam_max_x'] - (geom['ccd_buf_x'] + geom['ccd_width']), 0),
          (geom['cam_max_x'] - (geom['ccd_buf_x'] + geom['ccd_width']),
           geom['ccd_height'])],
         [(geom['cam_max_x'] - (geom['ccd_buf_x'] + geom['ccd_width']),
           geom['ccd_height']),
          (geom['cam_max_x'] - geom['ccd_buf_x'], geom['ccd_height'])],
         [(geom['cam_max_x'] - geom['ccd_buf_x'], geom['ccd_height']),
          (geom['cam_max_x'] - geom['ccd_buf_x'], 0)]],
        alpha=linealpha))
    ax.text(offset + geom['ccd_width'] + 2 * geom['ccd_buf_x'], offset,
            'CCD4', alpha=textalpha)
    # limit axes to crop outliers
    if crop_outliers:
        ax.set_xlim(geom['cam_min_x'], geom['cam_max_x'])
        ax.set_ylim(geom['cam_min_y'], geom['cam_max_y'])


def focal_to_absolute_pixel(x, y, cam_id=1, geom=GEOM):
    """transform from focal plane in mm to absolute fractional pixels"""
    # for the levine1 parameterization, flip 180 degrees for cameras 1 and 2
#    if cam_id in [1, 2]:
#        x = -x
#        y = -y
    if x < 0:
        pixel_x = x / geom['mm_per_pixel'] + geom['ccd_width'] + geom['ccd_buf_x']
    else:
        pixel_x = x / geom['mm_per_pixel'] + geom['ccd_width'] + 3 * geom['ccd_buf_x']
    if y < 0:
        pixel_y = y / geom['mm_per_pixel'] + geom['ccd_height']
    else:
        pixel_y = y / geom['mm_per_pixel'] + geom['ccd_height'] + 2 * geom['ccd_buf_y']
    # FIXME: should not have to do this?
#    if x > 0:
#        pixel_x -= 50
#    else:
#        pixel_x += 50
#    if y > 0:
#        pixel_y -= 50
#    else:
#        pixel_y += 50
    return pixel_x, pixel_y


def ccd_pixel_to_ffi_pixel(x, y, ccd, geom=GEOM):
    """transform from per-ccd coordinates to absolute fractional pixels"""
    # in april 2017 the transformations do not make sense (rotated 180 degrees)
    # 1, 2, 3, 4
    # as of march 2018 the order is 3, 4, 1, 2 (no 180 degree rotation)
    pixel_x = None
    pixel_y = None
    if ccd == 3: # lower left
        pixel_x = geom['ccd_buf_x'] + x
        pixel_y = y
    elif ccd == 4: # lower right
        pixel_x = geom['ccd_width'] + 3 * geom['ccd_buf_x'] + x
        pixel_y = y
    elif ccd == 1: # upper right
        pixel_x = 2 * geom['ccd_width'] + 3 * geom['ccd_buf_x'] - x
        pixel_y = 2 * (geom['ccd_height'] + geom['ccd_buf_y']) - y
    elif ccd == 2: # upper left
        pixel_x = geom['ccd_width'] + geom['ccd_buf_x'] - x
        pixel_y = 2 * (geom['ccd_height'] + geom['ccd_buf_y']) - y
    return pixel_x, pixel_y


class PlotData(object):
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label


def plot_pixel_comparison(figtitle, title, a, b):
    """Plot the data a and b.  a and b are of the form:
    a.x - array of x values
    a.y - array of y values
    a.label - text label for the legend
    """
    import matplotlib
    matplotlib.use(mpl_backend)
    from matplotlib import pyplot as plt
    from matplotlib import gridspec

    figure = plt.figure(figtitle, figsize=(20, 16))
    plt.suptitle(title)
    gs = gridspec.GridSpec(1, 2)

    # plot the data
    ax = figure.add_subplot(gs[0, 0])
    ax.set_title("pixels")
    ax.set_aspect(1)
    ax.scatter(a.x, a.y, s=5, marker='o', alpha=0.3, label=a.label)
    if b is not None:
        ax.scatter(b.x, b.y, s=40, marker='s', alpha=0.3, label=b.label)
    draw_ccd_outlines(ax)
    plt.legend(loc='upper right')

    # plot the residuals
    scale = 2.0
    ax = figure.add_subplot(gs[0, 1])
    ax.set_title("residuals")
    ax.set_aspect(1)
    delta_x = []
    delta_y = []
    for i in range(len(a.x)):
        delta_x.append(a.x[i] - b.x[i] if b.x[i] is not None else 0)
        delta_y.append(a.y[i] - b.y[i] if b.y[i] is not None else 0)
#    delta_x = np.array(a.x) - np.array(b.x)
#    delta_y = np.array(a.y) - np.array(b.y)
    size_x = scale * abs(np.array(delta_x))
    size_y = scale * abs(np.array(delta_y))
    ax.scatter(a.x, a.y, s=size_x, marker='o', alpha=0.3, label='delta_x',
               color='red')
    ax.scatter(a.x, a.y, s=size_y, marker='o', alpha=0.3, label='delta_y')
    draw_ccd_outlines(ax)
    plt.legend(loc='upper right')

    # try to make the spacing somewhat sane
    plt.subplots_adjust(top=0.98, bottom=0.1,
                        hspace=0.15, wspace=0.2, left=0.05, right=0.98)
    plt.show()
