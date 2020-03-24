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

import math
import numpy as np

from .base import ImageEffect
from tsig.spacecraft.ccd import CCD
from tsig.util import to_float


class DarkCurrent(ImageEffect):
    """
    Apply dark noise to an object.

    For each port, the average dark current is measured in electrons per
    pixel per second.

    For a 2 second exposure, a typical dark current will be around 5 electrons
    per pixel.

    This effect has units of electrons.

    reference: Al Levine 03may2017
    """
    version = 0.2
    units = 'electron'
    default_exptime = 2.0
    default_offset = 0.08

    def __init__(self, exptime=None, offset=None):
        super(DarkCurrent, self).__init__()
        """
        offset - electrons per pixel per second
        exptime - exposure time, in seconds
        """
        self.offset = to_float(offset)
        self.exptime = to_float(exptime)

    def apply_to_image(self, image, header={}):
        if image.shape[1] != 2136:
            raise ValueError("%s aborted: "
                             "number of columns must be 2136 (found %s)"
                             % (self.name, image.shape[1]))
        exptime = self.get_parameter('exptime', header.get('EXPTIME'))
        header['DARKCURR'] = (self.version, '[version] %s' % self.name)
        header['DCEXPTM'] = (exptime, '[s] dark current exposure time')
        for port in "ABCD":
            offset = self.get_parameter('offset', header.get('CCDDCO%s' % port))
            electrons = offset * exptime # assumes that offset is per second
            header['DCNOISE%s' % port] = (electrons, '[e-] dark current electrons for port %s' % port)
            image[0: 2058, 44 + CCD.PORT_MAP[port][0]: 44 + CCD.PORT_MAP[port][1]] += electrons
            # FIXME: apply reduced level of dark current to smear pixels
            #image[2058: 2068, 44 + CCD.PORT_MAP[port][0]: 44 + CCD.PORT_MAP[port][1]] += electrons
        return image
