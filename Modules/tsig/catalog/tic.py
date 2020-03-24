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
Access a TIC database and get stars based on position.
"""

import os
import time
import urllib
import numpy as np

from tsig.util.db import Database
from tsig.util import to_float
from .base import logger, Catalog


class TIC(Catalog, Database):
    DEFAULT_TIC = 'tic_6'
    CREDENTIALS_FILENAME = '~/.config/tsig/tic-dbinfo'
    CREDENTIALS_URL = 'http://tessellate.mit.edu/tsig/tic-dbinfo'

    TAG_ID = 'id'
    TAG_RA = 'ra'
    TAG_DEC = 'dec'
    TAG_PMRA = 'pmra'
    TAG_PMDEC = 'pmdec'
    TAG_TMAG = 'tmag'
    TAG_TEFF = 'teff'
    TAG_MASS = 'mass'
    TAG_RAD = 'rad'
    COLUMNS = [TAG_ID, TAG_RA, TAG_DEC, TAG_PMRA, TAG_PMDEC,
               TAG_TMAG, TAG_TEFF, TAG_MASS, TAG_RAD]

    def __init__(self, dbinfo_loc=None, dbtable='ticentries',
                 cache_directory=None, cache_queries=True,
                 min_brightness=None, max_brightness=None, object_type=None,
                 row_limit=-1,
                 **dbinfo):
        Catalog.__init__(self)
        Database.__init__(self, dbinfo_loc=dbinfo_loc, dbtable=dbtable, **dbinfo)

        if 'dbname' not in self.dbinfo:
            self.dbinfo['dbname'] = TIC.DEFAULT_TIC
            logger.debug("no database specified, using %s" %
                         self.dbinfo['dbname'])

        self.epoch = 2000 # positions in TIC are for julian year 2000

        # query for everything, but put these limits on which objects we use
        self.min_brightness = to_float(min_brightness)
        self.max_brightness = to_float(max_brightness)
        self.object_type = object_type
        self.row_limit = row_limit

        # configure for caching
        if cache_directory is None:
            cache_directory = Catalog.get_default_cache_dir()
        Catalog.create_cache_dir(cache_directory)
        self.cache_directory = cache_directory
        self.cache_queries = cache_queries

        logger.debug("min_brightness: %s" % self.min_brightness)
        logger.debug("max_brightness: %s" % self.max_brightness)
        logger.debug("object_type: %s" % self.object_type)

    def get_info(self):
        info = "database %s at %s" % (
            self.dbinfo.get('dbname'), self.dbinfo.get('dbhost'))
        if 'port' in self.dbinfo:
            info = "%s:%s" % (info, self.dbinfo['port'])
        return info

    def get_config(self):
        return {
            'min_brightness': self.min_brightness,
            'max_brightness': self.max_brightness,
            'object_type': self.object_type,
            'cache_directory': self.cache_directory,
            'cache_queries': self.cache_queries,
            'database_name': self.dbinfo.get('dbname'),
            'database_host': self.dbinfo.get('dbhost'),
            'database_table': self.dbinfo.get('dbtable'),
        }

    @staticmethod
    def create_query_name(catalog, ra, dec,
                          radius=None, width=None, height=None,
                          min_brightness=None, max_brightness=None):
        base = "{catalog}_ra{ra:+09.4f}dec{dec:+08.4f}".format(
            catalog=catalog, ra=ra, dec=dec)
        limits = ""
        if width is not None and height is not None:
            limits = "w{width:05.2f}h{height:05.2f}".format(
                width=width, height=height)
        else:
            limits = "rad{radius:05.2f}".format(radius=radius)
        minb = ""
        if min_brightness is not None:
            minb = "minb%s" % min_brightness
        maxb = ""
        if max_brightness is not None:
            maxb = "maxb%s" % max_brightness
        return base + limits + minb + maxb

    def query(self,
              ra=Catalog.DEFAULT_RA,
              dec=Catalog.DEFAULT_DEC,
              radius=Catalog.DEFAULT_RADIUS,
              dbindex=None):
        if radius == 0:
            self.query_by_loc(ra, dec, field_list=','.join(self.COLUMNS))
        elif dbindex == 'box':
            self.query_box_btree(ra, dec, 2 * radius, 2 * radius)
        elif dbindex == 'spacial_box':
            self.query_box(ra, dec, 2 * radius, 2 * radius)
        else:
            self.query_circle(ra, dec, radius)

    def query_circle(self,
                     ra=Catalog.DEFAULT_RA,
                     dec=Catalog.DEFAULT_DEC,
                     radius=Catalog.DEFAULT_RADIUS):
        """Do circle query using spacial index"""
        self.check_dbinfo(self.dbinfo)
        self.name = TIC.create_query_name(self.dbinfo['dbname'], ra, dec,
                                          radius=radius,
                                          min_brightness=self.min_brightness,
                                          max_brightness=self.max_brightness)
        sqlcmd = "\
SELECT %s FROM %s\
 WHERE spoint(radians(ra),radians(dec)) @ scircle '< (%sd,%sd), %sd >'" % (
     ','.join(self.COLUMNS), self.dbinfo['dbtable'], ra, dec, radius)
        self._query(sqlcmd)
        # remember the query
        self.query_ra = ra
        self.query_dec = dec
        self.query_radius = radius

    def query_box(self,
                  ra=Catalog.DEFAULT_RA,
                  dec=Catalog.DEFAULT_DEC,
                  width=Catalog.DEFAULT_WIDTH,
                  height=Catalog.DEFAULT_HEIGHT):
        """Do box query using spacial index"""
        self.check_dbinfo(self.dbinfo)
        self.name = TIC.create_query_name(self.dbinfo['dbname'], ra, dec,
                                          width=width, height=height,
                                          min_brightness=self.min_brightness,
                                          max_brightness=self.max_brightness)
        sqlcmd = "\
SELECT %s FROM %s\
 WHERE spoint(radians(ra),radians(dec)) @ sbox '( (%sd,%sd),(%sd,%sd) )'" % (
     ','.join(self.COLUMNS), self.dbinfo['dbtable'],
     ra - 0.5 * width, dec - 0.5 * height, ra + 0.5 * width, dec + 0.5 * height)
        self._query(sqlcmd)
        # remember the query
        self.query_ra = ra
        self.query_dec = dec
        self.query_width = width
        self.query_height = height

    def query_box_btree(self,
                        ra=Catalog.DEFAULT_RA,
                        dec=Catalog.DEFAULT_DEC,
                        width=Catalog.DEFAULT_WIDTH,
                        height=Catalog.DEFAULT_HEIGHT):
        """Do non-spacial box query"""
        self.check_dbinfo(self.dbinfo)
        self.name = TIC.create_query_name(self.dbinfo['dbname'], ra, dec,
                                          width=width, height=height,
                                          min_brightness=self.min_brightness,
                                          max_brightness=self.max_brightness)
        sqlcmd = "\
SELECT %s FROM %s\
 WHERE (ra between %s and %s) and (dec between %s and %s)" % (
     ','.join(self.COLUMNS), self.dbinfo['dbtable'],
     ra - 0.5 * width, ra + 0.5 * width, dec - 0.5 * height, dec + 0.5 * height)
        self._query(sqlcmd)
        # remember the query
        self.query_ra = ra
        self.query_dec = dec
        self.query_width = width
        self.query_height = height

    def _query(self, sqlcmd):
        """
        first try to load the catalog from local cache.  if that fails,
        make the query.
        """
        if self.min_brightness is not None:
            sqlcmd += " and %s <= %s" % (self.TAG_TMAG, self.min_brightness)
        if self.max_brightness is not None:
            sqlcmd += " and %s >= %s" % (self.TAG_TMAG, self.max_brightness)

        fn = self.name + '.npy'
        fullname = os.path.join(self.cache_directory, fn)
        try:
            logger.debug("load stars from %s" % fullname)
            t = np.load(fullname)
        except IOError, e:
            logger.debug("load failed: %s" % e)
            self.check_dbinfo(self.dbinfo)
            logger.info("query %s: %s" % (self.dbinfo['dbname'], sqlcmd))
            if self.row_limit is not None and self.row_limit > 0:
                sqlcmd += " LIMIT %s" % self.row_limit
            result, _ = self._do_query_raw(sqlcmd)
            dtype = [(self.COLUMNS[i], float) for i in range(len(self.COLUMNS))]
            t = np.array(result, dtype=dtype)
            if self.cache_queries:
                logger.debug("save query results to %s" % fullname)
                np.save(fullname, t)

        self._id = np.array(t[:][self.TAG_ID])
        self._ra = np.array(t[:][self.TAG_RA])
        self._dec = np.array(t[:][self.TAG_DEC])
        self._pmra = np.array(t[:][self.TAG_PMRA])
        self._pmdec = np.array(t[:][self.TAG_PMDEC])
        self._tmag = np.array(t[:][self.TAG_TMAG])
        self._teff = np.array(t[:][self.TAG_TEFF])
        self._mass = np.array(t[:][self.TAG_MASS])
        self._rad = np.array(t[:][self.TAG_RAD])

        self._pmra[np.isfinite(self._pmra) == False] = 0.0
        self._pmdec[np.isfinite(self._pmdec) == False] = 0.0

    def query_by_id(self, tic_id, field_list=None):
        """Query the catalog by TIC identifier"""
        self.check_dbinfo(self.dbinfo)
        if isinstance(tic_id, int):
            tic_id = ["%d" % tic_id]
        if isinstance(tic_id, basestring):
            tic_id = [tic_id]
        if field_list is None:
            field_list = '*'
        sqlcmd = "SELECT %s FROM %s WHERE id in (%s)" % (
            field_list, self.dbinfo['dbtable'], ','.join(tic_id))
        return self._do_query_raw(sqlcmd)

    def query_by_loc(self, ra, dec, radius=None, field_list=None):
        """Query the catalog by location ra,dec"""
        self.check_dbinfo(self.dbinfo)
        if field_list is None:
            field_list = '*'
        if radius is not None:
            sqlcmd = "\
SELECT %s FROM %s\
 WHERE spoint(radians(ra),radians(dec)) @ scircle '< (%sd,%sd), %sd >'" % (
     field_list, self.dbinfo['dbtable'], ra, dec, radius)
        else:
            sqlcmd = "SELECT %s FROM %s WHERE ra=%s and dec=%s" % (
                field_list, self.dbinfo['dbtable'], ra, dec)
        return self._do_query_raw(sqlcmd)
