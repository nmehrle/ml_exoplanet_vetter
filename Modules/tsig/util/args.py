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
Standard argument parsing methods. Use with argparser for best results.
"""

import os
import sys

from argparse import ArgumentTypeError

def filename(name):
    """Make sure we have a filename and return the absolute filename"""
    filename = os.path.expanduser(name)
    filename = os.path.abspath(filename)
    if not os.path.isfile(filename):
        raise ArgumentTypeError("File not found: %s" % name)
    return filename

def position(value):
    """Construct a tuple pair of x/y from the input value"""
    values = value.split(',')
    if len(values) != 2:
        raise ArgumentTypeError("Position must have two numbers as x,y")
    try:
        return map(float, values)
    except:
        raise ArgumentTypeError("Position must be numbers, not '%s'" % value)

def dimension(value):
    """Construct a tuple pair of width/height from the input value"""
    values = value.split('x')
    if len(values) != 2:
        raise ArgumentTypeError("Dimension must have two integers as WxH")
    try:
        return map(int, values)
    except:
        raise ArgumentTypeError("Dimension must be integers, not '%s'" % value)

def pointing(value):
    """Construct a tuple of ra/dec/roll from the input value"""
    values = value.split(',')
    if len(values) != 3:
        raise ArgumentTypeError("Pointing must be ra,dec,roll")
    try:
        return map(float, values)
    except:
        raise ArgumentTypeError("Pointing must be numbers, not '%s'" % value)

def ecliptic(value):
    """Construct a tuple pair of ra/dec from the input value"""
    values = value.split(',')
    if len(values) != 2:
        raise ArgumentTypeError("Ecliptic must be ra,dec")
    try:
        return map(float, values)
    except:
        raise ArgumentTypeError("Ecliptic must be numbers, not '%s'" % value)

def quaternion(value):
    """Construct a tuple of x,y,z,w from the input value"""
    values = value.split(',')
    if len(values) != 4:
        raise ArgumentTypeError("Quaternion must be x,y,z,w")
    try:
        return map(float, values)
    except:
        raise ArgumentTypeError("Quaternion must be numbers, not '%s'" % value)
