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
Loads and saves PRNU data into and from a postgresql database.
"""

import os
import re
import math
import logging
logger = logging.getLogger(__name__)

import numpy as np
from astropy.io import fits

from tsig.util.db import Database

class PhotoResponse(Database):
    """Loads PSF data from the database using queries"""

    CREDENTIALS_FILENAME = '~/.config/tsig/prnu-dbinfo'
    CREDENTIALS_URL = 'http://tessellate.mit.edu/tsig/prnu-dbinfo'

    # PRNU files use the following naming convention:
    #
    # d - Indicates dectector.  c indicates camera.  PRNU is always d.
    # board - Board version, usually 7
    # cam - Camera Serial, e.g. 05
    # series - If more than one set was taken for the same settings, this increases by one
    # wave - broadband or flood waveband number
    # temp - Temperature of the exposure
    # inter - order of the legendre polynomial fit to the illumination function
    #         either 32nd (o32) or 64th (o64) order.
    #
    # for example:
    #
    # d70500_flood660_m70_prnu_o32.fits
    # d70900_flood880_m80_prnu_o32.fits
    FN_MATCH = r'd(?P<board>\d)(?P<cam>\d{2})(?P<series>\d{2})_(flood)?(?P<wave>(bband|\d+))_m(?P<temp>\d+)_prnu_(?P<inter>\w+)\.fits'

    # Each PRNU file contains information for the entire camera.  The pixel
    # data are in an array that is 4272 columns and 4156 rows.  That is 4 CCDs
    # of science pixels (2048x2048) plus non-science pixels (4*44 columns and
    # 2*30 rows).

    @property
    def table(self):
        return self.dbinfo.get('dbtable', 'prnu')

    def user(self):
        return self.dbinfo.get('dbuser', 'tsig')

    def query(self, camera, ccd, wavelength=0, temperature=70):
        """Make a query for PRNU data"""
        data = self._do_query("SELECT grid FROM " + self.table + " WHERE camera=%s AND ccd=%s AND wavelength=%s AND temperature=%s", [
            camera, ccd, wavelength, temperature,
        ])
        return [np.array(datum[0]) for datum in data]

    def add_to_db(self, **prnu):
        """Add a single PRNU to the database"""
        self._do_query("INSERT INTO " + self.table + " (ccd, board, camera, series, wavelength, temperature, interpolation, grid) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", [
           int(prnu['ccd']),
           int(prnu['board']),
           int(prnu['cam']),
           int(prnu['series']),
           int(prnu['wave']),
           int(prnu['temp']),
           prnu['inter'],
           prnu['grid'].tolist(),
         ])  

    def add_file(self, filename):
        """Insert data from a given FITS file"""
        if not os.path.isfile(filename):
            logger.error("File not found: %s" % filename)
            return False

        name = os.path.basename(filename)
        match = re.search(self.FN_MATCH, name)
        if not match:
            logger.error("Filename not in PRNU format: %s" % name)
            return False

        data = match.groupdict()
        if data['wave'] == 'bband':
            data['wave'] = 0

        try:
            logger.debug("Loading file %s" % filename)
            hdulist = fits.open(filename, memmap=True)
        except Exception as err:
            logger.error("Error loading file: %s" % str(err))
            return False

        img = hdulist[0].data
        height_a, width_a = img.shape
        width_b, height_b = (4272, 4156)
        if width_a != width_b or height_b != height_b:
            logger.error("PRNU file is the wrong size. "
                         "Found %dx%d expected %dx%d" %
                         (width_a, height_a, width_b, height_b))
            return False

        # coordinates in the camera composite (4 CCDs including dark pixels)
        left = (0, 44)
        mid_x = (2092, 2180)
        right = (4228, 4272)
        top = (0, 0)
        # science pixels 2048x2048
        mid_y = (2048, 2108)
        # science pixels 2048x2058
        #mid_y = (2058, 2098)
        bottom = (4156, 4156)

        for i, (x1, x2) in enumerate([(left[1], mid_x[0]), (mid_x[1], right[0])]):
            for j, (y1, y2) in enumerate([(top[1], mid_y[0]), (mid_y[1], bottom[0])]):
                # The CCD number is re-mapped from a1,b1,a2,b2 to
                # anti-clockwise from top-right, See CCD reference numbers.
                data['ccd'] = [2, 1, 3, 4][i + (j * 2)]
                logger.debug("CCD %(ccd)s for Camera %(cam)s" % data)
                # The image we save is just the science bits for one ccd
                logger.debug("section %d,%d %d,%d" % (x1, y1, x2, y2))
                # coordinates for CCDs 1 and 2 are transformed
                if data['ccd'] == 1 or data['ccd'] == 2:
                    data['grid'] = img[y1: y2, x1: x2][::-1, ::-1]
                else:
                    data['grid'] = img[y1: y2, x1: x2]
                logger.debug("new size %dx%d" %
                             (data['grid'].shape[1], data['grid'].shape[0]))
                self.add_to_db(**data)
        return True

    def create_table(self):
        """Create the table with PRNU schema"""
        self._do_query("CREATE TABLE " + self.table + """(
  id          SERIAL     PRIMARY KEY,
  ccd         INTEGER,
  camera      INTEGER,
  temperature INTEGER,
  wavelength  INTEGER,
  series      INTEGER,
  board       INTEGER,
  interpolation CHAR(12),
  grid        REAL[][]
);""")
        self._do_query("GRANT SELECT ON " + self.table + " TO " + self.user)
