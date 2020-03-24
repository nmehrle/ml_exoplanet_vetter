#
# Copyright (C) 2018 - Massachusetts Institute of Technology (MIT)
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

"""Utilities for ra, dec, roll coordinates"""

import math


def normalize_ra(ra):
    """Ensure that the value, in radians, is in [0, 2*pi]"""
    ra += 2.0 * math.pi
    ra %= 2.0 * math.pi
    return ra

def normalize_dec(dec):
    """Ensure that the value, in radians, is in [-pi/2, pi/2]"""
    if dec < -math.pi / 2.:
        return -math.pi / 2.
    elif dec > math.pi / 2.:
        return math.pi / 2.
    return dec

def normalize_roll(roll):
    """Ensure that the value, in radians, is in [-pi, pi]"""
    if roll < -math.pi:
        return roll + 2. * math.pi
    elif roll > math.pi:
        return roll - 2. * math.pi
    return roll


def normalize_ra_d(ra):
    """Ensure that the value, in degrees, is in [0, 360]"""
    ra += 360.0
    ra %= 360.0
    return ra

def normalize_dec_d(dec):
    """Ensure that the value, in degrees, is in [-90, 90]"""
    if dec < -90.0:
        return -90.0
    elif dec > 90.0:
        return 90.0
    return dec

def normalize_roll_d(roll):
    """Ensure that the value, in degrees, is in [-180, 180]"""
    if roll < -180.0:
        return roll + 360.0
    elif roll > 180.0:
        return roll - 360.0
    return roll
