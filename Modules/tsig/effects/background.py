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
Simulate celestial and galactic background light.
"""

import numpy as np
from .base import ImageEffect
from tsig.util import to_float
from tsig.util.coord import equatorial_to_ecliptic, equatorial_to_galactic


class BackgroundEffect(ImageEffect):
    """
    Adds a simulated scattered background effect to the image
    """
    default_exptime = 2.0
    default_longitude = 0.0
    default_latitude = 0.0

    def __init__(self, exptime=None, longitude=None, latitude=None):
        super(BackgroundEffect, self).__init__()
        self.exptime = to_float(exptime)
        self.longitude = to_float(longitude)
        self.latitude = to_float(latitude)

    def apply_to_image(self, image, header, exptime, calc):
        effective_area = 69.1
        pixelscale = 21.1
        pixel_solid_area = pixelscale ** 2
        bg = calc * effective_area * pixel_solid_area
        image[0: 2058, 44: 2092] += bg * exptime
        return image


class CelestialBackground(BackgroundEffect):
    """
    Adds zodiacal light, treated as smooth using celestial (ecliptic) latitude.
    This implementation adds a constant value to all pixels; there is no
    modeling of the actual variation from boresight to edges of lense.
    """
    version = 0.3
    units = 'electron'

    def __init__(self, latitude=None, exptime=None):
        """
        latitude - ecliptic latitude, in degrees
        exptime - exposure time, in seconds
        """
        super(CelestialBackground, self).__init__(
            exptime=exptime, latitude=latitude)

    def apply_to_image(self, image, header={}):
        lat = None
        dec = header.get('CAMDEC')
        ra = header.get('CAMRA')
        if dec is not None and ra is not None:
            lat, _ = equatorial_to_ecliptic(ra, dec)

        exptime = self.get_parameter('exptime', header.get('EXPTIME'))
        lat = self.get_parameter('latitude', lat)

        header['BGCELEST'] = (self.version, '[version] %s' % self.name)
        header['BGCLAT'] = (lat, '[degree] celestial background latitude')
        header['BGCEXPTM'] = (exptime, '[s] celestial background exposure time')

        v_max = 23.345
        delta_v = 1.148
        b = np.abs(lat)
        v = v_max - delta_v * ((b - 90.0) / 90.0) ** 2
        assert ((b < 90).all())
        calc = 10 ** (-0.4 * (v - 22.8)) * (2.39e-3)
        return super(CelestialBackground, self).apply_to_image(
            image, header, exptime, calc)


class GalacticBackground(BackgroundEffect):
    """
    Adds unresolved stars, treated as smooth background using galactic
    latitude and longitude.
    """
    version = 0.3
    units = 'electron'

    def __init__(self, longitude=None, latitude=None, exptime=None):
        """
        longitude - camera pointing, in degrees
        latitude - camera pointing, in degrees
        exptime - exposure time, in seconds
        """
        super(GalacticBackground, self).__init__(
            exptime=exptime, latitude=latitude, longitude=longitude)

    def apply_to_image(self, image, header={}):
        lat = None
        lon = None
        dec = header.get('CAMDEC')
        ra = header.get('CAMRA')
        if dec is not None and ra is not None:
            lat, lon = equatorial_to_galactic(ra, dec)

        exptime = self.get_parameter('exptime', header.get('EXPTIME'))
        lat = self.get_parameter('latitude', lat)
        lon = self.get_parameter('longitude', lon)

        header['BGGALACT'] = (self.version, '[version] %s' % self.name)
        header['BGGLAT'] = (lat, '[degree] galactic background latitude')
        header['BGGLON'] = (lon, '[degree] galactic background longitude')
        header['BGGEXPTM'] = (exptime, '[s] galactic background exposure time')

        # from Josh and Peter's memo
        if lon > 180.0:
            lon -= 360.0

        a0 = 18.9733
        a1 = 8.833
        a2 = 4.007
        a3 = 0.805
        I_surface_brightness = a0 + a1 * (np.abs(lat) / 40.0) + a2 * (np.abs(lon) / 180.0) ** a3

        calc = 10 ** (-0.4 * I_surface_brightness) * 1.7e6
        return super(GalacticBackground, self).apply_to_image(
            image, header, exptime, calc)
