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

from .base import ImageEffect


class Markup(ImageEffect):
    """
    Apply a grid in corner of image to mark origin and CCD identification.
    """
    version = 0.2
    units = ''

    def __init__(self):
        super(Markup, self).__init__()

    def apply_to_image(self, image, header={}):
        header['MARKUP'] = (self.version, '[version] %s' % self.name)

        # draw the origin block
        image[10:100, 44 + 10: 44 + 100] = 30000
        # figure out which CCD this is based on the header
        ccd_id = header.get('CCDNUM')
        if ccd_id:
            # draw smaller blocks in origin to indicate the CCD number
            if ccd_id >= 1:
                image[60:90, 44 + 60: 44 + 90] = 0
            if ccd_id >= 2:
                image[20:50, 44 + 60: 44 + 90] = 0
            if ccd_id >= 3:
                image[20:50, 44 + 20: 44 + 50] = 0
            if ccd_id >= 4:
                image[60:90, 44 + 20: 44 + 50] = 0
        # draw a line in the direction of increasing x
        image[10:20, 44 + 110: 44 + 160] = 20000
        # draw a border around the CCD imaging pixels
        image[0:2048, 45:46] = 30000
        image[1:2, 44:2092] = 30000
        image[2046:2047, 44:2092] = 30000
        image[0:2048, 2090:2091] = 30000
