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

from collections import namedtuple


Quaternion = namedtuple('Quaternion', 'w x y z')


def to_quaternion(x):
    """Try to coerce a quaternion (4 numbers) from the given input."""
    q = None
    if isinstance(x, str) and ',' in x:
        x = x.split(',')
    try:
        if len(x) == 4:
            q = [float(y) for y in x]
    except TypeError:
        pass
    return q

def is_quaternion(x):
    """See if the specified object can be treated as a quaternion"""
    return to_quaternion(x) is not None

def to_rdr(x):
    """Try to coerce ra,dec,roll (3 numbers) from the given input."""
    rdr = None
    if isinstance(x, str) and ',' in x:
        x = x.split(',')
    try:
        if len(x) == 3:
            rdr = [float(y) for y in x]
    except TypeError:
        pass
    return rdr

def is_rdr(x):
    """See if the specified object can be treated as ra,dec,roll"""
    return to_rdr(x) is not None
