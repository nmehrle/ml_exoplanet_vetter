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
Utilities for cracking into h5 format matlab files.
"""

from h5py import File
import numpy as np

class MatLabFile(File):
    """
    Generic unpacking of PSF matlab files using h5py

    Use:

    mp = MatLabFile(filename)
    for dictionary in mp.get_iter():
        grid = dictionray['PSF']

    """
    def get_key(self):
        for key in self.keys():
            if 'stellar' in key.lower():
                return key
        raise KeyError("Stellar key into the data is required")

    def get_iter(self, key=None):
        if key is None:
            key = self.get_key()
        doc = self[key]
        ref_field = doc.keys()[0]

        for x, xd in enumerate(doc[ref_field]):
            for y, yd in enumerate(xd):
                yield dict((field, np.array(doc[doc[field][x][y]])) for field in doc.keys())

