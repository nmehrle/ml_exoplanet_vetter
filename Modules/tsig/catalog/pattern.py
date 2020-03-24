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
#

"""
Provide test pattern catalogs for testing the tsig machine. 
"""

import numpy as np
from .base import logger, Catalog


class TestPattern(Catalog):
    """
    Place one or more images in the sky, each non-zero pixel is a 'star'.
    """
    def __init__(self, filename=None, ra=45.0, dec=45.0, scale=0.01,
                 epoch_start=2018.25):
        super(TestPattern, self).__init__()
        """
        filename - path to the image
        ra - right ascension of the image center
        dec - declination of the image center
        scale - degrees per pixel
        """
        self.name = 'test_pattern'
        self.mappings = []
        self.epoch = epoch_start
        self.default_tmag = 19
        if filename:
            self.place_image(filename, ra, dec, scale)

    def get_info(self):
        return "pattern mappings=%s filename=%s" % (
            len(self.mappings), self.filename)

    def place_image(self, filename, ra, dec, scale=0.01):
        """
        Place an image in the sky, with lower left corner at
        (origin_ra,origin_dec) with the image scaled using the scale factor..

        filename - path to the image
        ra - right ascension of the image origin
        dec - declination of the image origin
        scale - degrees per pixel
        """
        import scipy.misc
        logger.debug("place '%s' at (%s,%s) with scale %s" %
                     (filename, ra, dec, scale))
        try:
            img = scipy.misc.imread(filename, flatten=True)
            info = dict()
            info[filename] = filename
            info['ra'] = ra
            info['dec'] = dec
            info['scale'] = scale
            info['data'] = img
            self.mappings.append(info)
        except IOError, e:
            logger.error("place image failed: %s" % e)

    def query(self,
              ra=Catalog.DEFAULT_RA,
              dec=Catalog.DEFAULT_DEC,
              radius=Catalog.DEFAULT_RADIUS):
        """
        Map the (ra,dec) to pixels in an image.
        """
        _ra = []
        _dec = []
        _tmag = []
        ra_lo = ra - radius
        ra_hi = ra + radius
        dec_lo = dec - radius
        dec_hi = dec + radius
        # for each image, scan each pixel.  if the pixel is non-zero, and it
        # maps to celestial coordinates within the specified range, then add
        # the celestial coordinates to the list.
        for m in self.mappings:
            w, h = m['data'].shape
            for i in range(w):
                for j in range(h):
                    if not m['data'][i][j]:
                        r = m['ra'] + m['scale'] * (h - j)
                        d = m['dec'] + m['scale'] * (w - i)
                        if ra_lo < r < ra_hi and dec_lo < d < dec_hi:
                            _ra.append(r)
                            _dec.append(d)
        self._ra = np.array(_ra)
        self._dec = np.array(_dec)
        self._pmra = np.zeros(np.size(_ra))
        self._pmdec = np.zeros(np.size(_ra))
        self._tmag = np.zeros(np.size(_ra)) + self.default_tmag
        self._teff = np.zeros(np.size(_ra))

        # remember the query
        self.query_ra = ra
        self.query_dec = dec
        self.query_radius = radius
