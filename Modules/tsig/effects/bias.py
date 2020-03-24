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
from tsig.util import to_int, to_float


class BiasLevel(ImageEffect):
    """
    Each port has a bias, measured in ADU.

    The bias depends on the electronics temperature.

    For normal image simulation, only the bias mean is used.

    The delta is modeled as a Gaussian with standard deviation.  Delta is only
    needed for values in the virtual pixels.  In that case, the virtual pixel
    bias is the bias_mean +/- delta +/- read noise.

    reference: Deb Woods, 14 April 2017
    """
    version = 0.2
    units = 'adu'
    default_bias = 7000
    default_stddev = 0.5

    def __init__(self, bias=None, bias_stddev=None):
        """
        bias - mean bias level, in ADU
        bias_stddev - standard deviation of bias
        """
        super(BiasLevel, self).__init__()
        self.bias = to_int(bias)
        self.bias_stddev = to_float(bias_stddev)

    def apply_to_image(self, image, header={}):
        if image.shape[1] != 2136:
            raise ValueError("%s aborted: "
                             "number of columns must be 2136 (found %s)"
                             % (self.name, image.shape[1]))
        header['BIAS'] = (self.version, '[version] %s' % self.name)
        for i, port in enumerate("ABCD"):
            bias = self.get_parameter('bias', header.get('CCDBIAS%s' % port))
            header['BIAS%s' % port] = (bias, '[ADU] bias level for port %s' % port)
            # apply to science pixels
            image[0:, 44 + CCD.PORT_MAP[port][0]: 44 + CCD.PORT_MAP[port][1]] += bias
            offset = i * 11
            # apply to underclock pixels
            image[0:, offset: offset + 11] += bias
            # apply to overclock pixels
            image[0:, 44 + 2048 + offset: 44 + 2048 + offset + 11] += bias
        # FIXME: add stddev to bias?
        return image
