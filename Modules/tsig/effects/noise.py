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
Add different noise effects to an image.
"""

import numpy as np

from .base import ImageEffect
from tsig.spacecraft.ccd import CCD
from tsig.util import to_float


class ReadoutNoise(ImageEffect):
    """
    Read noise is modeled by applying a gaussian perturbation to each pixel.
    The mean value of the gaussian is zero, and the standard deviation is
    different for each of the 4 ports in each of the CCDs.  Values are provided
    by calibration tables, for example
    
    "SN06 basic calibration at -70C"
    https://tess-web.mit.edu/ground/PNY-1/d70601/summaries/basicCal-70C

    These tables provide a value "oc noise", or overlock noise, with units of
    either ADU or electrons.  The overclock noise is the root mean square (RMS)
    of the values from the overclock pixels.

    This implementation uses the oc_noise with units of electrons.

    Readout noise applies to the image pixels, buffer pixels, smear pixels,
    and virtual rows.  It also applies to the overclock and underclock pixels.

    reference: Deb Woods 23feb2017
    """
    version = 0.2
    units = 'electron'
    default_stddev = 10.0

    def __init__(self, stddev=None):
        """
        stddev - deviation of gaussian noise distribution
        """
        super(ReadoutNoise, self).__init__()
        self.stddev = to_float(stddev)

    def apply_to_image(self, image, header={}):
        if image.shape[1] != 2136:
            raise ValueError("%s aborted: "
                             "number of columns must be 2136 (found %s)"
                             % (self.name, image.shape[1]))
        header['READNOIS'] = (self.version, '[version] %s' % self.name)
        for i, port in enumerate("ABCD"):
            stddev = self.get_parameter('stddev', header.get('CCDOCN%s' % port))
            header['RNDEV%s' % port] = (stddev, '[e-] read noise deviation applied to port %s' % port)
            noise = np.random.normal(0, stddev, (image.shape[0], CCD.PORT_WIDTH))
            image[0:, 44 + CCD.PORT_MAP[port][0]: 44 + CCD.PORT_MAP[port][1]] += noise
            offset = i * 11
            # apply to underclock pixels
            noise = np.random.normal(0, stddev, (image.shape[0], 11))
            image[0:, offset: offset + 11] += noise
            # apply to overclock pixels
            noise = np.random.normal(0, stddev, (image.shape[0], 11))
            image[0:, 44 + 2048 + offset: 44 + 2048 + offset + 11] += noise
        return image


class ShotNoise(ImageEffect):
    """
    Model the fluctuations in the number of photons due to their occurrance
    independent of each other.  This is a consequence of the discretization
    of photons.

    Also known as Photon Noise or Poisson Noise.

    Shot noise applies only to the imaging and smear pixels.

    reference: wikipedia
    """
    version = 0.3
    units = 'electron'

    def __init__(self):
        super(ShotNoise, self).__init__()

    def apply_to_image(self, image, header={}):
        header['SHOTNOIS'] = (self.version, '[version] %s' % self.name)
        nonzero = image[0: 2068, 44: 2092] > 0
        image[0: 2068, 44: 2092][nonzero] += np.random.normal(0, np.sqrt(image[0: 2068, 44:2092][nonzero]))
        return image


class FixedPatternNoise(ImageEffect):
    """
    Fixed Pattern Noise Effect
    """
    version = 0.0
    units = 'adu'

    def __init__(self):
        super(FixedPatternNoise, self).__init__()

    def apply_to_image(self, image, header={}):
        header['FIXEDPAT'] = (self.version, '[version] %s' % self.name)
        raise NotImplementedError("%s not written yet" % self.name)
