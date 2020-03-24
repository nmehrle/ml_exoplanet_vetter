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
Grid based test pattern for testing tsig.
"""

import numpy as np
from .base import logger, Catalog


class TestGrid(Catalog):
    """
    The test grid is a grid of stars.
    """
    def __init__(self, size=2000.0, spacing=75.0,
                 mag_range=(6, 16), max_nudge=0.0, max_motion=0.0,
                 randomize_magnitudes=False, epoch_start=2018.25):
        super(TestGrid, self).__init__()
        """
        size - width and height of the grid, in arcseconds
        spacing - distance between stars in the grid, in arcseconds
        mag_range - low and high magnitudes for the stars
        max_nudge - distance to nudge each star from the grid, in arcseconds
        max_motion - mas/yr
        randomize_magnitudes - magnitudes are assigned randomly if this is True
        """
        self.name = 'test_grid'
        self.size = size
        self.spacing = spacing
        self.mag_range = mag_range
        self.max_nudge = max_nudge
        self.max_motion = max_motion
        self.randomize_magnitudes = randomize_magnitudes
        self.epoch_start = epoch_start

    def get_info(self):
        return "grid size=%s spacing=%s" % (self.size, self.spacing)

    def query(self,
              ra=Catalog.DEFAULT_RA,
              dec=Catalog.DEFAULT_DEC,
              radius=Catalog.DEFAULT_RADIUS):
        """
        Create a grid of stars.  Default behavior is to place stars evenly
        across the grid, with magnitudes increasing from lowest to highest.
        """
        # FIXME: do not ignore the radius - use it to do subset of entire grid
        logger.debug("create test grid: ra=%s dec=%s size=%s spacing=%s "
                     "mag_range=%s max_nudge=%s max_motion=%s randomize=%s" %
                     (ra, dec, self.size, self.spacing,
                      self.mag_range, self.max_nudge, self.max_motion,
                      self.randomize_magnitudes))

        # the name for this catalog is based on the grid size
        self.name = Catalog.create_query_name('test_grid', ra, dec, radius)
        self.name += "_{minmag:.0f}to{maxmag:.0f}".format(
            minmag=min(self.mag_range), maxmag=max(self.mag_range))

        # figure out how many stars are needed
        pixels = max(int(self.size / self.spacing), 1)
        n = pixels * pixels

        # create an array of magnitudes that will be applied to the stars
        if self.randomize_magnitudes:
            self._tmag = np.random.uniform(
                np.min(self.mag_range), np.max(self.mag_range), n)[::-1]
        else:
            self._tmag = np.linspace(
                np.min(self.mag_range), np.max(self.mag_range), n)[::-1]

        # create a grid centered at 0
        ra_grid, dec_grid = np.meshgrid(np.arange(pixels) * self.spacing,
                                        np.arange(pixels) * self.spacing)

        # offset
        self._dec = ((dec_grid - np.mean(dec_grid)) / 3600.0 + dec).flatten()
        self._ra = (ra_grid - np.mean(ra_grid)).flatten() / \
            np.cos(self._dec * np.pi / 180.0) / 3600.0 + ra

        # randomly nudge each star
        if self.max_nudge > 0:
            offset = self.max_nudge * (np.random.rand(2, n) - 0.5) / 3600.0
            self._dec += offset[0, :]
            self._ra += offset[1, :] * np.cos(self._dec * np.pi / 180.0)

        # make up some imaginary proper motions
        if self.max_motion > 0:
            self._pmra = np.random.normal(0, self.max_motion, n)
            self._pmdec = np.random.normal(0, self.max_motion, n)
        else:
            self._pmra = np.zeros(n)
            self._pmdec = np.zeros(n)

        # assign temperature for each star
        self._teff = 5800.0 + np.zeros_like(self._ra) # FIXME

        # set the epoch starting point
        self.epoch = self.epoch_start

        # remember the query
        self.query_ra = ra
        self.query_dec = dec
        self.query_radius = radius
