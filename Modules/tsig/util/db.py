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
Background and base for making database queries.
"""

import os
import time
import urllib

import logging
logger = logging.getLogger(__name__)


class Database(object):
    """
    Database mixin provides a standard process for connecting to a database
    """
    CREDENTIALS_FILENAME = '~/.config/tsig/tsig-dbinfo'
    CREDENTIALS_URL = 'http://tessellate.mit.edu/tsig/tsig-dbinfo'
    table = None
    schema = None

    def __init__(self, dbinfo_loc=None,
                 dbhost=None, dbname=None, dbport=None,
                 dbuser=None, dbpass=None, dbtable=None):
        # start with database parameters from file or url
        self.dbloc = dbinfo_loc
        self.dbinfo = self.get_dbinfo(dbinfo_loc)
        # override with credentials and other connection parameters
        if dbhost is not None:
            self.dbinfo['dbhost'] = dbhost
        if dbname is not None:
            self.dbinfo['dbname'] = dbname
        if dbport is not None:
            self.dbinfo['dbport'] = dbport
        if dbuser is not None:
            self.dbinfo['dbuser'] = dbuser
        if dbpass is not None:
            self.dbinfo['dbpass'] = dbpass
        if dbtable is not None:
            self.dbinfo['dbtable'] = dbtable
        elif self.table:
            self.dbinfo['dbtable'] = self.table

        if self.dbinfo.get('dbport') is None:
            self.dbinfo['dbport'] = 5432

        for key, value in self.dbinfo.items():
            if key == 'dbpass':
                value = "*" * 12
            logger.debug("%s: %s", key, value)

    @classmethod
    def get_dbinfo(cls, path=None):
        """
        Get the database parameters.

        First load application defaults from the application's dbinfo.  First
        try local file, fall back to URL if file does not exist.

        Then try to read from the specified location, which could be a file
        or a URL.

        If nothing is specified, try the default file location, then try the
        web server and cache result.
        """
        dbinfo = dict()
        dbinfo.update(cls._get_dbinfo(Database.CREDENTIALS_FILENAME,
                                      Database.CREDENTIALS_URL,
                                      None))
        dbinfo.update(cls._get_dbinfo(cls.CREDENTIALS_FILENAME,
                                      cls.CREDENTIALS_URL,
                                      path))
        return dbinfo

    @classmethod
    def _get_dbinfo(cls, default_filename, default_url, path=None):
        # if file specified, use it, otherwise try to download and cache
        if not path or '://' in path:
            url = path or default_url
            filename = os.path.abspath(os.path.expanduser(default_filename))
            try:
                if not os.path.isfile(filename):
                    try:
                        os.makedirs(os.path.dirname(filename))
                    except os.error:
                        pass
                    testfile = urllib.URLopener()
                    testfile.retrieve(url, filename)
                    logging.debug("Got dbinfo details from %s -> %s", url, filename)
            except Exception as err:
                logger.debug("Download of credentials failed: "
                             "%s (%s)" % (str(err), url))
        else:
            filename = os.path.abspath(os.path.expanduser(path))

        # now try to read from the requested file
        dbinfo = dict()
        try:
            with open(filename, 'r') as f:
                for line in f:
                    name, value = line.split('=')
                    dbinfo[name.strip()] = value.strip()
            logging.debug("Got dbinfo details from %s", filename)
        except IOError:
            logger.debug("No %s database credentials at %s" %
                         (cls.__name__, filename))
        return dbinfo

    @staticmethod
    def check_dbinfo(dbinfo):
        missing = []
        for k in ['dbhost', 'dbname', 'dbuser', 'dbtable']:
            if k not in dbinfo:
                missing.append(k)
        if missing:
            raise ValueError("Missing database parameters: %s" %
                             ','.join(missing))

    def create_database(self):
        """
        Attempt to create the database specified by the dbinfo details.
        """
        self.check_dbinfo(self.dbinfo)
        dbname = self.dbinfo['dbname']
        dbuser = self.dbinfo['dbuser']
        with self.postgres_connection as conn:
            with conn.cursor() as cursor:
                cursor.execute('CREATE DATABASE %s' % (dbname,))
                cursor.execute('GRANT ALL PRIVILEGES ON %s TO %s', (dbname, dbuser))

    def grant_permission(self):
        """
        Give access to the database to the configured user
        """
        dbuser = self.dbinfo['dbuser']
        dbtable = self.dbinfo['dbtable']
        if dbtable:
            with self.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('GRANT SELECT ON %s TO %s', (dbtable, dbuser))

    def select(self, *args, **where):
        """
        Basic SQL select function. fields is a list of fields to select
        if non provided, then all fields are selected.

        where is a dictionary of field names or selection conditionals
        """
        dbtable = self.dbinfo['dbtable']
        limit = where.pop('limit', '')
        if limit:
            limit = "LIMIT %s" % str(limit)

        order = where.pop('order', '')
        if order:
            order = "ORDER BY %s" % str(order)

        fields = "%s" % ", ".join(args or ['*'])
        vals = []
        whr = self._where(vals, **where)
        sql = "SELECT %s FROM %s %s %s %s" % (fields, dbtable, whr, order, limit)
        return self._do_query(sql, vals)

    def insert(self, **values):
        """Insert one row into this table"""
        dbtable = self.dbinfo['dbtable']
        fields = []
        vals = []
        valt = []

        for (field, value) in values.items():
            fields.append(field)
            if isinstance(value, tuple) and len(value) == 2:
                valt.append("POINT(%s, %s)")
                vals += list(value)
            else:
                valt.append("%s")
                vals.append(value)

        sql = "INSERT INTO %s (%s) VALUES (%s)" % \
            (dbtable, ", ".join(fields), ", ".join(valt))
        return self._do_query(sql, vals)

    def create_table(self, dbtable=None, schema=None):
        """
        Create the given table in the configured database
        """
        dbtable = dbtable or self.dbinfo['dbtable']
        schema = schema or self.schema
        fields = []
        for (name, kind) in schema:
            fields.append("%s %s" % (name, kind.upper()))
        sql = "CREATE TABLE %s (%s)" % (dbtable, ",\n".join(fields))
        return self._do_query(sql)

    def drop_table(self, dbtable=None):
        """
        Drop the given table in the configured database
        """
        dbtable = dbtable or self.dbinfo['dbtable']
        sql = "DROP TABLE %s" % (dbtable)
        return self._do_query(sql)

    def _where(self, vals, **where):
        """Generates a where clause for sql queries"""
        if not where:
            return ""
        ret = []
        for (name, value) in where.items():
            op = '='
            if '__' in name:
                (name, op) = name.rsplit('__', 1)

            if '%' in name:
                ret.append(name)
                vals.append(value)
                continue

            # Special within clause for positional selections
            special = "where_" + op
            if hasattr(self, special):
                ret.append(getattr(self, special)(name, value, vals))
                continue

            # Add more operators here as needed.
            op = {'lt': '<', 'gt': '>'}.get(op, op)

            ret.append("%s %s %%s" % (name, op))
            vals.append(value)
        return "WHERE " + " AND ".join(ret)

    @staticmethod
    def where_within(name, value, vals):
        """Select within a geometric circle radius"""
        if len(value) != 3:
            raise IOError("When selecting within a radius, three values must be provided")
        vals += list(value)
        return "%s <@ circle '((%%s,%%s), %%s)'" % name

    # A standard connection uses everything from dbinfo.
    connection = property(lambda self: self._connection(**self.dbinfo))

    @property
    def postgres_connection(self):
        """Special connection used when creating and dropping databases"""
        dbinfo = self.dbinfo.copy()
        dbinfo['dbname'] = 'postgres'
        conn = self._connection(**dbinfo)
        conn.autocommit = True
        return conn
    
    def _connection(self, dbhost, dbname, dbport, dbuser, dbpass=None,
                    dbtable=None):
        """Return the database connection."""
        try:
            import psycopg2
        except ImportError as e:
            logger.error("PostgreSQL python module psycopg2 is required")
            logger.debug("psycopg2 import failed: %s" % e)
            return None

	return psycopg2.connect(
	    host=dbhost, database=dbname, port=dbport,
	    user=dbuser, password=dbpass, connect_timeout=5)

    def _do_query(self, sqlcmd, *args, **kw):
        logger.debug("query: %s" % sqlcmd)
        t0 = time.time()

        conn = self.connection
        cur = conn.cursor()
        t1 = time.time()
        logger.debug("query start")
        cur.execute(sqlcmd, *args, **kw)
        t2 = time.time()
        logger.debug("query complete")
        if 'select' in sqlcmd.lower():
            result = QueryResults(cur)
        else:
            result = conn.commit()

        t3 = time.time()
        logger.debug("data retrieved")
        t4 = time.time()
        logger.debug("query: setup=%.3f execute=%.3f fetch=%.3f close=%.3f"
                     % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return result

    def _do_query_raw(self, sqlcmd):
        logger.debug("query: %s" % sqlcmd)
        t0 = time.time()
        conn = self.connection
        cur = conn.cursor()
        t1 = time.time()
        logger.debug("query start")
        cur.execute(sqlcmd)
        t2 = time.time()
        logger.debug("query complete")
        result = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        t3 = time.time()
        logger.debug("data retrieved")
        cur.close()
        cur = None
        conn.close()
        conn = None
        t4 = time.time()
        logger.info("query: setup=%.3f execute=%.3f fetch=%.3f close=%.3f"
                    % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return result, column_names



class QueryResults(object):
    """Returned when we make a query"""
    def __init__(self, cur):
        self.cur = cur

    def __iter__(self):
        for row in self.cur.fetchall():
            yield row
        self.close()

    def as_dict(self):
        """Return the query result as a dictionary per row (generator)"""
        cols = self.columns
        for row in self:
            yield dict(zip(cols, row))

    @property
    def columns(self):
        return [desc[0] for desc in self.cur.description]

    def close(self):
        self.cur.close()
        self.cur.connection.close()

    def __len__(self):
        return self.cur.rowcount
