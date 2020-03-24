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
epoch - Decimal year used to predict the positions of stars with proper
        motions (stars that are actually visibly moving across the sky).
        UCAC4 quotes stellar positions at an epoch of 2000.0, so calculating
        the position at an epoch of 2018.5 (march 2018) involves multiplying
        by 18.25 years the angular rate at which the star is moving across
        the sky.

  bjd - barycentric julian date.  Number of days that have elapsed since noon
        on 1 January 4713 BC if you measured with an atomic clock sitting at
        the solar system barycenter.  If you did not measure from the solar
        system barycenter, the times of transits of distant stars could vary
        by up to 16 minutes over the course of an earth year, as earth is
        intercepting light early or late relative to the Sun/barycenter.

        http://astroutils.astronomy.ohio-state.edu/time/bjd_explanation.html
"""

import os
import numpy as np

import tsig.lightcurve
from tsig.util.configurable import ConfigurableObject

import logging
logger = logging.getLogger(__name__)


def _to_int(x):
    try:
        return int(x)
    except TypeError:
        pass
    return None


class Catalog(ConfigurableObject):

    # default query values, in degrees
    DEFAULT_RA=0.0
    DEFAULT_DEC=0.0
    DEFAULT_RADIUS=0.2
    DEFAULT_WIDTH=0.2
    DEFAULT_HEIGHT=0.2

    """
    A catalog contains information about stars and other heavenly objects.
    """
    def __init__(self):
        self.name = 'catalog'
        self.epoch = None
        self.clear_query()
        self.clear_lightcurves()

    def __repr__(self):
        """string representation of the catalog object"""
        return self.name

    def clear_query(self):
        self.query_ra = None
        self.query_dec = None
        self.query_radius = None
        self.query_width = None
        self.query_height = None
        self._id = []
        self._ra = []
        self._dec = []
        self._tmag = []
        self._teff = []
        self._pmra = [] # rate of change for ra in mas/year
        self._pmdec = [] # rate of change for dec mas/year

    def clear_lightcurves(self):
        self._lightcurve = []

    @staticmethod
    def get_default_cache_dir():
        return os.path.join(os.environ['HOME'], '.cache/tsig/catalog')

    @staticmethod
    def create_cache_dir(dirname=None):
        if dirname is None:
            dirname = Catalog.get_default_cache_dir()
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    @staticmethod
    def create_query_name(catalog, ra, dec,
                          radius=None, width=None, height=None):
        base = "{catalog}_ra{ra:+09.4f}dec{dec:+08.4f}".format(
            catalog=catalog, ra=ra, dec=dec)
        limits = ""
        if width is not None and height is not None:
            limits = "w{width:05.2f}h{height:05.2f}".format(
                width=width, height=height)
        else:
            limits = "rad{radius:05.2f}".format(radius=radius)
        return base + limits

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    def add_lightcurves(self, faintest_star=None, stars_with_random_curves=0.0):
        """
        Add a lightcurve to each of the stars in this catalog.

          faintest_star - minimum magnitude of included stars
          starts_with_curves - fraction of stars that should have non-constant
                               light curve (between 0.0 and 1.0)
        """
        # number of stars
        nstar = len(self._tmag)
        if nstar == 0:
            logger.debug('lightcurves skipped: no magnitudes')
            return

        # start with a constant lightcurve for each star
        # FIXME: all stars should share a single instance of constant?
        c = tsig.lightcurve.ConstantLightCurve()
        self._lightcurve = np.array([c] * nstar)

        # generate random light curves for some fraction of the stars
        if stars_with_random_curves:
            # set a value for magnitude threshold if none was specified
            if faintest_star is None:
                faintest_star = np.max(self._tmag) + 1
            logger.debug('magnitude dimmest=%s threshold=%s' %
                         (np.max(self._tmag), faintest_star))
            # do only stars that are bright enough
            bright_stars = (self._tmag <= faintest_star).nonzero()[0]
            logger.debug('%s stars brighter than %s' %
                         (len(bright_stars), faintest_star))
            logger.debug('using random light curves on %.1f%% of stars' %
                (stars_with_random_curves * 100.0))
            n = int(len(bright_stars) * stars_with_random_curves)
            for i in np.random.choice(bright_stars, n, replace=False):
                self._lightcurve[i] = tsig.lightcurve.create_random_lightcurve()

    @property
    def lightcurve_codes(self):
        """return an array of codes for the lightcurves in this catalog"""
        return [lc.code for lc in self._lightcurve]

    def stars_static(self):
        """return static star parameters as arrays of positions, magnitude,
        and effective temperature"""
        return np.array(self._ra), np.array(self._dec), np.array(self._tmag), np.array(self._teff)

    def stars_snapshot(self, bjd=None, epoch=None, exptime=0.5 / 24.0,
                       roll=0.0, ra_0=0.0, dec_0=0.0):
        """return star parameters at a specific time and position, as arrays
        of positions, magnitude, and effective temperature, any of which may
        be time-varying.

        bjd - barycentric julian date
        epoch - decimal year
        exptime - exposure time, in days
        roll - roll angle in degrees
        ra_0 - roll right ascension in degrees
        dec_0 - roll declination in degrees
        """

        if bjd is not None:
            epoch = (bjd - 2451544.5) / 365.25 + 2000.0
        elif epoch is not None:
            bjd = (epoch - 2000.0) * 365.25 + 2451544.5
        else:
            raise ValueError("no time specified: bjd or epoch is required")
        logger.debug("snapshot at bjd=%s epoch=%s exptime=%s" %
                     (bjd, epoch, exptime))

        ra, dec = self.positions_at_epoch(epoch)

        # do the rotation if roll angle is non-zero
        if roll != 0.0:
            ra_rot = np.zeros(np.size(ra))
            dec_rot = np.zeros(np.size(dec))
            rot_ang = roll * np.pi / 180.0
            rot_mat = np.array([[np.cos(rot_ang), -np.sin(rot_ang)],
                                [np.sin(rot_ang), np.cos(rot_ang)]])
            for pp in range(0, np.size(ra)):
                ra_here = ra[pp]
                dec_here = dec[pp]
                if ra_here >= 180.0: # based on VizieR output, wrap at 180 deg
                    ra_here -= 360.0
                if dec_here >= 180.0:
                    dec_here -= 360.0
                ra_c = ra_here - ra_0 # center the ra values on the fov
                dec_c = dec_here - dec_0
                rot_vec = np.dot(rot_mat, np.array([[ra_c], [dec_c]]))
                rot_vec += np.array([[ra_0], [dec_0]])
                ra_rot[pp] = rot_vec[0] # assign first vector value to ra
                dec_rot[pp] = rot_vec[1] # assign second vector value to dec
                if ra_rot[pp] < 0: # unwrap the angle to give value in [0,360)
                    ra_rot[pp] += 360.0
                if dec_rot[pp] < 0:
                    dec_rot[pp] += 360.0
            ra = ra_rot
            dec = dec_rot

        # determine brightness of star
        tmag = self._tmag
        if len(tmag):
            if len(self._lightcurve) == len(tmag):
                logger.debug("calculate moment from %s lightcurves" % len(tmag))
                moment = np.array([lc.integrated(bjd, exptime)
                                   for lc in self._lightcurve]).flatten()
                tmag += moment
            else:
                logger.debug("no lightcurves in this catalog")

        # determine color of star
        teff = self._teff

        assert(ra.shape == tmag.shape)
        assert(dec.shape == tmag.shape)
        return ra, dec, tmag, teff

    def positions_at_epoch(self, epoch):
        """
        calculate the positions of the items in the catalog at the specified
        point in time.  returns two arrays: the array of ra and array of dec
        for all of the catalog items.
        """
        # how many years since the catalog's epoch?
        elapsed = epoch - self.epoch # years
        logger.debug('projecting catalog {0:.3f} years relative to {1:.0f}'.
                     format(elapsed, self.epoch))
        # calculate the dec in degree/year, assuming pmdec in mas/year
        dec_rate = self._pmdec / 60.0 / 60.0 / 1000.0
        dec = self._dec + elapsed * dec_rate
        # calculate unprojected rate of ra motion using the mean declination
        # between the catalog and the present epoch, in degrees of ra/year
        # assuming original was projected mas/year
        ra_rate = self._pmra / 60.0 / 60.0 / np.cos((self._dec + elapsed * dec_rate / 2.0) * np.pi / 180.0) / 1000.0
        ra = self._ra + elapsed * ra_rate
        return ra, dec

    def make_plot(self, epoch=None, roll=0.0, ra_0=0.0, dec_0=0.0, ax=None):
        """
        Make a plot of the stars.  If an epoch or other parameter is specified,
        the move to that point in time and/or that viewing position.  If an
        axis is specified, add the plot to an existing plot.
        """
        if epoch is not None:
            ra, dec, tmag, teff = self.stars_snapshot(
                epoch=epoch, roll=roll, ra_0=ra_0, dec_0=dec_0)
        else:
            ra, dec, tmag, teff = self.stars_static()
        logger.debug("plotting %d stars" % len(tmag))
        delta_mag = 20.0 - tmag
        size = delta_mag * delta_mag

        import matplotlib.pylab as plt
        if ax is None:
            plt.figure(self.name)
            ax = plt.subplot()
        else:
            plt.sca(ax)
        ax.scatter(ra, dec, s=size, marker='o', c=teff, cmap=plt.cm.hot,
                   alpha=0.3, edgecolors='black')
        ax.set_aspect(1)
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        title = '%s' % self.name
        if epoch:
            title += " at epoch %s" % epoch
        ax.set_title(title)        
        # set axis ranges to match the query, plus a little buffer
        if self.query_radius is not None:
            halfw = 1.1 * self.query_radius
            halfh = 1.1 * self.query_radius
        else:
            halfw = 0.55 * self.query_width
            halfh = 0.55 * self.query_height
        ax.set_xlim(self.query_ra - halfw, self.query_ra + halfw)
        ax.set_ylim(self.query_dec - halfh, self.query_dec + halfh)
        # when looking at the sky, ra increases from right to left
        ax.invert_xaxis()
        return plt.gcf()
