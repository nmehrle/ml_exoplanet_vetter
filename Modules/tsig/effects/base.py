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
The effects base classes provides all the tools required to write
a photon or electron based effect.
"""

import re
import os
import sys
import time
import inspect

import numpy as np
from astropy.io import fits


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)

def feed_in(fits_in):
    """
    Read a FITS file and return the hdulist.
    """
    if isinstance(fits_in, basestring):
        if os.path.isfile(fits_in):
            # memmap options seems to have trouble with generated fits files.
            return fits.open(fits_in)
        else:
            raise IOError("Cannot open FITS file: %s" % fits_in)
    elif isinstance(fits_in, fits.HDUList):
        return fits_in
    raise ValueError("Not sure what type of data this is: %s" %
                     type(fits_in).__name__)

def feed_out(hdulist, filename, overwrite=True):
    """
    Save the hdulist to file.
    """    
    hdulist.writeto(filename, overwrite=overwrite)
    return hdulist

def apply_now(effect, hdulist):
    """
    Apply the effect on the appropriate hdulist and return the result.
    """
    idx = 0
    for i, hdu in enumerate(hdulist):
        if hdu.data is not None:
            idx = i
            break
    return effect.apply(hdulist, idx)


class Effect(object):

    def __init__(self):
        pass

    def apply(self, hdulist, index=0):
        """
        The apply function is where the work of the effect happens. Your child
        class should overwrite this and the effect will be run.

          hdulist - The fits data, passed through for convenience.

        """
        raise NotImplementedError("An effect must have an apply() function.")

    @classproperty
    def name(cls):
        """By default returns the class name as the name of the effect."""
        return cls.__name__

    @classproperty
    def title(cls):
        return re.sub("([a-z])([A-Z])","\g<1> \g<2>", cls.name)

    @classproperty
    def version(cls):
        raise NotImplementedError('Version must be provided on: %s' % cls.name)

    @classproperty
    def args(cls):
        """Return a list of arguments and their default values"""
        docs = (cls.__init__.__doc__ or "").split("\n")
        docs = (line.strip().split(" - ", 1) for line in docs)
        docs = dict((l[0].strip(), l[1].strip()) for l in docs if len(l) == 2)

        spec = inspect.getargspec(cls.__init__)
        args = spec.args[1:]
        defs = spec.defaults or ()
        defaults = ((None,) * (len(args) - len(defs))) + defs

        for x, arg in enumerate(args):
            default = defaults[x]
            if default is not None:
                yield (arg, default, type(default), docs.get(arg, None))
            else:
                yield (arg, None, str, docs.get(arg, None))

    @classproperty
    def all(cls):
	subclasses = set()
	work = [cls]
	while work:
	    parent = work.pop()
	    for child in parent.__subclasses__():
		if child not in subclasses:
		    subclasses.add(child)
		    work.append(child)
	return list(subclasses)

    @classproperty
    def doc(cls):
        if not cls.__doc__:
            return "No documentation currently available."
        return cls.__doc__.replace('\n    ', '\n').strip()

    @classmethod
    def get_effect(cls, effect_name):
        for effect in cls.all:
            if effect.name.lower() == effect_name.lower():
                return effect
        return None


class ImageEffect(Effect):

    def apply_to_image(self, image, header):
        """
        The main entry point to applying an effect to an image.

          image - image data as np.array

        Each effect operates directly on the image data.  The image data type
        is float and should not be modified by any effect.

        All images have shape defined as nrows, ncols (height, width).
        """
        raise NotImplementedError("apply_to_image() has not been implemented")

    def apply(self, hdulist, index=0):
        """
        Performs the conversion to and from images formats as needed
        """
        self.apply_to_image(hdulist[index].data, hdulist[index].header)
        return hdulist

    def get_parameter(self, label, header_value):
        """
        Get the indicated parameter.  First try a member value.  Then try
        the header value.  Fallback to default.
        """
        if hasattr(self, label) and getattr(self, label) is not None:
            return getattr(self, label)
        if header_value is not None:
            return header_value
        return getattr(self, 'default_' + label)


class TestEffect(Effect):

    def apply(self, hdulist, index=0):
        return hdulist
