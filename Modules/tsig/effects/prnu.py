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


class FlatField(ImageEffect):
    """
    Flat field simulates a variety of effects including brick wall (brush
    strokes), tree rings, and straps.  Brick wall and tree rings are due to
    variations in the surface of the CCD.  Straps are reflective metal bands
    behind the CCD that reflect at certain wavelengths.

    The flat field is determined experimentally by shining light on the CCD at
    a range of wavelengths.

    Also known as Pixel Response non-Uniformity (PRNU).
    """
    version = 0.0
    units = ''

    def __init__(self):
        super(FlatField, self).__init__()

    def apply_to_image(self, image, header={}):
        header['PRNU'] = (self.version, '[version] %s' % self.name)
        raise NotImplementedError("%s not written yet" % self.name)
