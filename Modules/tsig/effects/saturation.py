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


class Saturation(ImageEffect):
    """
    Models the saturation effect of a CCD experienced when the maximum number
    of electrons is received and further electrons are no longer read by the
    CCD pixels.

    Each port has a different saturation limit, measured in electrons.

    When the saturation limit is reached, excess electrons spill over into
    neighboring pixels in the same column.

    Saturation limit is obtained experimentally.

    Saturation applies to the image pixels, buffer pixels, and smear pixels.
    """
    version = 0.3
    units = 'electron'
    default_saturation_limit = 200000

    def __init__(self, saturation_limit=None):
        """
        saturation_limit - maximum number of electrons
        """
        super(Saturation, self).__init__()
        self.saturation_limit = to_float(saturation_limit)

    def apply_to_image(self, image, header={}):
        if image.shape[1] != 2136:
            raise ValueError("%s aborted: "
                             "number of columns must be 2136 (found %s)"
                             % (self.name, image.shape[1]))
        header['SATURATE'] = (self.version, '[version] %s' % self.name)
        for port in "ABCD":
            limit = self.get_parameter('saturation_limit', header.get('CCDSAT%s' % port))
            header['SATLIM%s' % port] = (limit, '[e-] saturation limit applied to port %s' % port)
            view = image[0: 2068, 44 + CCD.PORT_MAP[port][0]: 44 + CCD.PORT_MAP[port][1]]
            saturated = np.nonzero(view > limit)
            for i in range(len(saturated[0])):
                row = saturated[0][i]
                col = saturated[1][i]
                self.bloom(view, col, row, limit)
        return image

    @staticmethod
    def bloom(view, col, row, limit):
        # distribute any excess electrons up and down the column
        excess = view[row, col] - limit
        view[row, col] = limit
        # distribute to pixels down the column
        rem = excess / 2
        idx = row - 1
        while rem > 0 and idx >= 0:
            if view[idx, col] < limit:
                view[idx, col] += rem
                if view[idx, col] > limit:
                    rem = view[idx, col] - limit
                    view[idx, col] = limit
                else:
                    rem = 0
            idx -= 1
        # distribute to pixels up the column
        rem = excess / 2
        idx = row + 1
        while rem > 0 and idx < view.shape[0]:
            if view[idx, col] < limit:
                view[idx, col] += rem
                if view[idx, col] > limit:
                    rem = view[idx, col] - limit
                    view[idx, col] = limit
                else:
                    rem = 0
            idx += 1
