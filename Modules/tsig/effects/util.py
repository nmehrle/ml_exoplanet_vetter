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
from tsig.spacecraft.ccd import CCD
from tsig.util import to_float


class ElectronsToADU(ImageEffect):
    """
    Convert from electron counts to ADU counts.  The conversion factor is a
    gain determined experimentally.  There is one gain per port for each CCD.

    The gain applies to every pixel, including non-science pixels.
    """
    version = 0.3
    units = ''
    default_gain = 5.0

    def __init__(self, gain=None):
        """
        gain - gain in electrons/ADU
        """
        super(ElectronsToADU, self).__init__()
        self.gain = to_float(gain)

    def apply_to_image(self, image, header={}):
        if image.shape[1] != 2136:
            raise ValueError("%s aborted: "
                             "number of columns must be 2136 (found %s)"
                             % (self.name, image.shape[1]))
        header['E2ADU'] = (self.version, '[version] %s' % self.name)
        for i, port in enumerate("ABCD"):
            gain = self.get_parameter('gain', header.get('CCDGAIN%s' % port))
            header['E2AGAIN%s' % port] = (gain, '[e-/ADU] gain applied to port %s' % port)
            image[0:, 44 + CCD.PORT_MAP[port][0]: 44 + CCD.PORT_MAP[port][1]] /= gain
            offset = i * 11
            # apply to underclock pixels
            image[0:, offset: offset + 11] /= gain
            # apply to overclock pixels
            image[0:, 44 + 2048 + offset: 44 + 2048 + offset + 11] /= gain
        return image
