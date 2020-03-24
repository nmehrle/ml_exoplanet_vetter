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


class Undershoot(ImageEffect):
    """
    Add a darker pixel next to each bright pixel to simulate the artificial
    dimming of pixels when reading bright pixels read out by the CCD.

    Ports A and C read low to high, with dimming on the higher pixel.

    Ports B and D read high to low, with dimming on the lower pixel.

    Undershoot applies to image pixels, buffer pixels, smear pixels,
    virtual pixels, and the underclock and overclock pixels.

    reference: Al Levine 03may2017
    """
    version = 0.4
    units = 'adu'

    def __init__(self, amount=0.0013):
        super(Undershoot, self).__init__()
        """
        amount - the convolution factor for calculating undershoot
        """
	self.amount = float(amount)

    def apply_to_image(self, image, header={}):
        if image.shape[1] != 2136:
            raise ValueError("%s aborted: "
                             "number of columns must be 2136 (found %s)"
                             % (self.name, image.shape[1]))

        header['UNDERSHT'] = (self.version, '[version] %s' % self.name)
        header['UNDERAMT'] = (self.amount, 'undershoot amout')

        # FIXME: incorporate 11 dark pixels at either end

        def calc_undershoot(row):
            return - self.amount * np.concatenate([[0], row[:-1]])

        for port in "AC":
            view = image[0:, 44+CCD.PORT_MAP[port][0]:44+CCD.PORT_MAP[port][1]]
            ushoot = np.apply_along_axis(calc_undershoot, 1, view)
            image[0:, 44+CCD.PORT_MAP[port][0]:44+CCD.PORT_MAP[port][1]] += ushoot

        for port in "BD":
            view = image[0:, 44+CCD.PORT_MAP[port][0]:44+CCD.PORT_MAP[port][1]]
            ushoot = np.apply_along_axis(calc_undershoot, 1, view[0:, ::-1])
            image[0:, 44+CCD.PORT_MAP[port][0]:44+CCD.PORT_MAP[port][1]] += ushoot[0:, ::-1]

        return image
