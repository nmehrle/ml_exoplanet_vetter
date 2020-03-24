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
UCAC Catalog access
"""

import os

import pkgutil
import astropy
import astropy.io.ascii

import numpy as np
import numpy.polynomial.polynomial as polynomial

from .base import logger, Catalog

class UCAC4(Catalog):

    def __init__(self, faint_limit=None, row_limit=-1,
                 cachedir=None, cache_queries=True):
        super(UCAC4, self).__init__()
        self.faint_limit = faint_limit
        self.row_limit = row_limit
        if cachedir is None:
            cachedir = Catalog.get_cache_dirname()
        self.cachedir = cachedir
        Catalog.create_cache_dir(cachedir)
        self.cache_queries = cache_queries

    def get_info(self):
        return "ucac4"

    def query(self,
              ra=Catalog.DEFAULT_RA,
              dec=Catalog.DEFAULT_DEC,
              radius=Catalog.DEFAULT_RADIUS):
        catalog = 'UCAC4'
        ra_tag = '_RAJ2000'
        dec_tag = '_DEJ2000'
        vcat = 'I/322A/out'
        rmag_tag = 'f.mag'
        jmag_tag = 'Jmag'
        vmag_tag = 'Vmag'
        pmra_tag = 'pmRA'
        pmdec_tag = 'pmDE'

        self.name = Catalog.create_query_name(catalog, ra, dec, radius)
        fn = self.name + '.npy'
        fullname = os.path.join(self.cachedir, fn)

        # first try to load the catalog from local cache.  if that fails,
        # make the query.
        try:
            logger.debug("load stars from %s" % fullname)
            t = np.load(fullname)
        except IOError, e:
            logger.debug("load failed: %s" % e)
            logger.info("query %s for ra=%s dec=%s radius=%s" %
                        (catalog, ra, dec, radius))
            from astroquery.vizier import Vizier
            columns = [
                ra_tag, dec_tag, pmra_tag, pmdec_tag,
                rmag_tag, jmag_tag, vmag_tag, catalog]
            v = Vizier(catalog=vcat, columns=columns)
            v.ROW_LIMIT = self._row_limit
            t = v.query_region(
                astropy.coordinates.ICRS(
                    ra=ra, dec=dec,
                    unit=(astropy.units.deg, astropy.units.deg)),
                radius='{:f}d'.format(radius), verbose=True)[0]
            if self.cache_queries:
                logger.debug("save query results to %s" % fullname)
                np.save(fullname, t)

        _ra = np.array(t[:][ra_tag])
        _dec = np.array(t[:][dec_tag])
        _pmra = np.array(t[:][pmra_tag])
        _pmdec = np.array(t[:][pmdec_tag])
        _rmag = np.array(t[:][rmag_tag])
        _jmag = np.array(t[:][jmag_tag])
        _vmag = np.array(t[:][vmag_tag])

        rbad = (np.isfinite(_rmag) == False) * (np.isfinite(_vmag))
        _rmag[rbad] = _vmag[rbad]
        rbad = (np.isfinite(_rmag) == False) * (np.isfinite(_jmag))
        _rmag[rbad] = _jmag[rbad]

        jbad = (np.isfinite(_jmag) == False) * (np.isfinite(_vmag))
        _jmag[jbad] = _vmag[jbad]
        jbad = (np.isfinite(_jmag) == False) * (np.isfinite(_rmag))
        _jmag[jbad] = _rmag[jbad]

        vbad = (np.isfinite(_vmag) == False) * (np.isfinite(_rmag))
        _vmag[vbad] = _rmag[vbad]
        vbad = (np.isfinite(_vmag) == False) * (np.isfinite(_jmag))
        _vmag[vbad] = _jmag[vbad]

        _teff = self.pickles(_rmag - _jmag)
        _imag = _rmag - self.davenport(_rmag - _jmag)

        _pmra[np.isfinite(_pmra) == False] = 0.0
        _pmdec[np.isfinite(_pmdec) == False] = 0.0

        ok = np.isfinite(_imag)
        if self.faint_limit is not None:
            ok *= _imag <= self.faint_limit

        logger.info("found %s stars with %s < V < %s" %
                    (np.sum(ok), np.min(_rmag[ok]), np.max(_rmag[ok])))
        self.ra = _ra[ok]
        self.dec = _dec[ok]
        self.pmra = _pmra[ok]
        self.pmdec = _pmdec[ok]
        self.tmag = _imag[ok]
        self.teff = _teff[ok]
        self.epoch = 2000.0

        # remember the query
        self.query_ra = ra
        self.query_dec = dec
        self.query_radius = radius

    # Supporting files for UCAC4 queries
    #   davenport.txt was davenport_table1.txt
    #   pickles.txt was pickles_table2.txt
    #
    # These files are used to convert UCAC4 magnitudes into extremely
    # approximate TESS magnitudes.  Once TIC is fully operational, these
    # conversions (and the UCAC4) should not be necessary.

    # convert colors, using the Sloan stellar locus
    def davenport(color, input='r-J', output='r-i'):
        data = astropy.io.ascii.read(
            pkgutil.get_data(__name__, 'davenport.txt'),
            names=['g-i', '#', 'u-g', 'eu-g', 'g-r',
                   'eg-r', 'r-i', 'er-i', 'i-z', 'ei-z',
                   'z-J', 'ez-J', 'J-H', 'eJ-H', 'H-K',
                   'eH-K'])
        if input == 'r-J':
            x = data['r-i'] + data['i-z'] + data['z-J']
        if output == 'r-i':
            y = data['r-i']
        # plt.figure()
        # plt.scatter(x, y)
        import scipy.interpolate
        interpolator = scipy.interpolate.interp1d(x, y, 'linear', fill_value=0,
                                                  bounds_error=False)
        new = interpolator(color)
        new[color > np.max(x)] = y[np.argmax(x)]
        new[color < np.min(x)] = y[np.argmin(x)]
        return new

    # spit out the Pickles temperature associated with an input color
    def pickles(color, input='R-J'):
        data = astropy.io.ascii.read(pkgutil.get_data(__name__, 'pickles.txt'))
        ok = data['[Fe/H]'] == 0
        if input == 'R-J':
            x = data['V-J'][ok] - data['V-R'][ok]
        y = data['logT'][ok]
        # plt.figure()
        # plt.scatter(x, 10**y)
        coeff, stats = polynomial.polyfit(x, y, 4, full=True)
        interpolator = polynomial.Polynomial(coeff)
        #import scipy.interpolate
        #interpolator = scipy.interpolate.interp1d(x,y,'linear',bounds_error=False)
        new = interpolator(color)
        new[color > np.max(x)] = y[np.argmax(x)]
        new[color < np.min(x)] = y[np.argmin(x)]
        return 10 ** new

