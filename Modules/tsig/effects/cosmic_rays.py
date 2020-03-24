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


class CosmicRays(ImageEffect):
    """
    Noise effect which deposits a track of charge across the pixels depending
    on the direction. Modified lamdel distribution.

    This effect requires the tsig_cosmical C code module.
    """
    version = 0.1
    units = 'electron'
    default_exptime = 2.0

    def __init__(self, exptime=None, rate=5.0, buf=100,
                 gradient=False, diffusion=False):
        """
        exptime - exposure time, in seconds
        rate - 
        buf - number of buffer pixels
        gradient - whether to apply gradient
        diffusion - whether to apply diffusion
        """
        super(CosmicRays, self).__init__()
        self.exptime = to_float(exptime)
        self.rate = float(rate)
        self.buf = int(buf)
        self.gradient = bool(gradient)
        self.diffusion = bool(diffusion)

    def apply_to_image(self, image, header={}):
        try:
            from tsig_cosmical import cosmical
        except ImportError:
            raise ImportError("The tsig_cosmical extension must be installed to use the Cosmic Ray Effect.")

        exptime = self.get_parameter('exptime', header.get('EXPTIME'))

        header['COSMIC'] = (self.version, '[version] %s' % self.name)
        header['COSEXPTM'] = (exptime, '[s] cosmic exposure time')
        header['COSRATE'] = (self.rate, 'cosmic rate')
        header['COSBUFF'] = (self.buf, 'cosmic buffer')
        header['COSGRAD'] = (self.gradient, 'cosmic gradient')
        header['COSDIFF'] = (self.diffusion, 'cosmic diffusion')

        (height, width) = image.shape
        buf = self.buf

        smallexptime = exptime * 1.5 
        bigexptime = exptime * 1.5 
        # if a gradient is set, allow the exposure times to be different
        if self.gradient:
            smallexptime = exptime
            bigexptime = exptime * 2 

        # add margin around image, because Al's code doesn't start cosmic ray
        # images off the screen
        buffered_height = height + 2 * buf
        buffered_width = width + 2 * buf

        # set the diffusion flag to be 0 if False, 1 if True (as integer type)
        intdiffusion = 0  # np.int(self.diffusion)

        # call the fancy cosmic ray code
        bg = cosmical(self.rate, smallexptime, bigexptime,
                      buffered_height, buffered_width, intdiffusion)

        # if we need to diffuse the image, use Al's kernal (from the fancy code)
        if self.diffusion:
            import scipy.signal
            kernal = np.array([[0.0034, 0.0516, 0.0034],
                               [0.0516, 0.7798, 0.0516],
                               [0.0034, 0.0516, 0.0034]])
            bg = scipy.signal.convolve2d(bg, kernal, mode='same')

        # Return the resized image on top of the original image
        image += bg[buf:-buf, buf:-buf] if buf > 0 else bg[:, :]
        return image
