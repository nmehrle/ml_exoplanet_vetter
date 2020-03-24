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

import numpy as np
from .base import ImageEffect
from tsig.util import to_float


class Smear(ImageEffect):
    """
    The smear effect simulates how new pixels are exposed to light as the
    pixels are read in row by row.

    number of transfers: 2078
    9.6 micro-seconds per row
    19.95 ms

    Smear applies to the image pixels, buffer pixels, and smear pixels.

    reference: TESS CCD Readout, Rev 2, Nov 3 2015, Dwg No 37-19501.01
    """
    version = 0.4
    units = 'electron'
    default_readout_time = 0.02
    default_exptime = 2.0

    def __init__(self, readout_time=None, exptime=None):
        """
        readout_time - time to read a single row, seconds
        exptime - time to read a single image, in seconds
        """
        super(Smear, self).__init__()
        self.exptime = to_float(exptime)
        self.readout_time = to_float(readout_time)

    def apply_to_image(self, image, header={}):
        """
        Get the mean of each column, multiply each column mean by the ratio
        of readout_time to exptime, then add that to each value in the
        corresponding column.
        """
        exptime = self.get_parameter('exptime', header.get('EXPTIME'))
        readout_time = self.get_parameter('readout_time', header.get('CCDREADT'))

        header['SMEAR'] = (self.version, '[version] %s' % self.name)
        header['SMEAR_ET'] = (exptime, '[s] smear exposure time')
        header['SMEAR_RO'] = (readout_time, '[s] smear readout time')

        view = image[0: 2068, 44: 44 + 2048]
        ones = np.ones(shape=view.shape, dtype=float)
        smear = np.sum(view, 0).reshape(1, view.shape[1]) * ones
        smear *= readout_time / exptime / 2078
        view += smear
        return image
