#
# Copyright (C) 2015 - Zach Berta-Thompson <zkbt@mit.edu> (MIT License)
#               2017 - Massachusetts Institute of Technology (MIT)
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

"""Transformations from one reference frame to another"""

import astropy
import math
import numpy as np


def get_unit_vector(ra_r, dec_r):
    u = np.array([math.cos(ra_r) * math.cos(dec_r),
                  math.sin(ra_r) * math.cos(dec_r),
                  math.sin(dec_r)])
    u /= magnitude(u)
    return u

def magnitude(u):
    return math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])

def normalize(xyz):
    m = magnitude(xyz)
    return [v / m for v in xyz]

def normalize_vector(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def equatorial_to_ecliptic(ra, dec):
    """
    Convert equatorial ra, dec (degrees) to ecliptic lat, lon (degrees).
    """
#    e = 23.441844 # obliquity of the ecliptic
#    ra_r = ra * math.pi / 180.0
#    dec_r = dec * math.pi / 180.0
#    lon = atan2(sin(ra_r) * cos(e) + tan(dec_r) * sin(e), cos(ra_r))
#    lat = asin(sin(dec_r) * cos(e) - cos(dec_r) * sin(e) * sin(ra_r))
#    lon *= 180.0 / math.pi
#    lat *= 180.0 / math.pi
#    return lat, lon
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    e = c.barycentrictrueecliptic
    return e.lat.value, e.lon.value

def equatorial_to_galactic(ra, dec):
    """
    Convert equatorial ra, dec (in degrees) to galactic lat, lon (in degrees).
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    g = c.galactic
    return g.b.value, g.l.value

def spherical_to_cartesian(theta, phi):
    """Transform spherical coordinates theta (ra) and phi (dec) in radians to
    unit vector cartesian coordinates.  The spherical 'r' is implicitly a unit
    vector with length 1."""
#    xyz = np.asarray([np.cos(theta), np.sin(theta), np.sin(phi)])
#    xyz[:2] *= np.cos(phi)
    xyz = np.asarray([np.cos(theta), np.sin(theta), np.cos(phi)])
    xyz[:2] *= np.sin(phi)
    return xyz

def cartesian_to_spherical(x, y, z):
    """Transform cartesian coordinates to spherical coordinates in radians,
    where the spherical 'r' is implicitly a unit vector with length 1."""
    theta_r = math.atan2(y, x)
    r = math.sqrt(x * x + y * y + z * z)
    phi_r = math.acos(z / r) if r else None
    return theta_r, phi_r
