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

from .constants import *

def to_bool(x):
    try:
        if x.lower() in ['true', 'yes']:
            return True
        elif x.lower() in ['false', 'no']:
            return False
    except AttributeError:
        pass
        try:
            return bool(int(x))
        except (ValueError, TypeError):
            pass
    raise ValueError("Unknown boolean specifier: %s" % x)

def to_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except ValueError:
        return None

def to_int(x):
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        return None
