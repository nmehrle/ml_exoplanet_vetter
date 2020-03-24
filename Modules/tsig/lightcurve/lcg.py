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

import logging
logger = logging.getLogger(__name__)

import numpy as np

from tsig.util import to_float
from tsig.util.configurable import ConfigurableObject
from .lightcurve import *


class LightCurveGenerator(ConfigurableObject):
    """
    The light curve generator controls how light curves are assigned to objects.
    """
    def __init__(self, types=None, fraction=0.0, min_brightness=None):
        super(LightCurveGenerator, self).__init__()
        """
        types - which types of lightcurves should be considered
        fraction - how many of the objects should get a light curve [0.0, 1.0]
        min_brightness - apply lightcurves only to this magnitude or brighter
        """
        logger.info("light curve generator: fraction=%s" % fraction)
        if types is None:
            types = ['ConstantLightCurve']
        self.types = types
        self.fraction = float(fraction)
        self.min_brightness = to_float(min_brightness)

    def get_config(self):
        return {
            'types': self.types,
            'fraction': self.fraction,
            'min_brightness': self.min_brightness,
        }

    def get_lightcurves(self, tmag):
        """
        Get lightcurves for the specified magnitudes.  Return an array of
        lightcurves corresponding to the specified magnitudes.

        tmag - array of TESS magnitudes
        """
        lightcurve = []

        nstar = len(tmag)
        if nstar == 0:
            logger.debug('lightcurves skipped: no magnitudes')
            return lightcurve

        # start with no lightcurve for each star
        lightcurve = [None] * nstar

        # generate light curves for some fraction of the stars
        if self.fraction:
            # set a value for magnitude threshold if none was specified
            faintest_star = self.min_brightness
            if faintest_star is None:
                faintest_star = np.max(tmag) + 1
            logger.info('magnitude dimmest=%s threshold=%s' %
                        (np.max(tmag), faintest_star))
            # do only stars that are bright enough
            bright_stars = (tmag <= faintest_star).nonzero()[0]
            logger.info('%s stars brighter than %s' %
                        (len(bright_stars), faintest_star))
            n = int(len(bright_stars) * self.fraction)
            logger.info('using randomly chosen curves on %d stars (%.1f%%)' %
                        (n, self.fraction * 100.0))
            for i in np.random.choice(bright_stars, n, replace=False):
                lightcurve[i] = create_random_lightcurve()

        return lightcurve
