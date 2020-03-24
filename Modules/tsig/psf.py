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
Point spread function (PSF) and classes to generate them.
"""

import os
import math
import logging
logger = logging.getLogger(__name__)

from io import BytesIO
import numpy as np

from tsig.util import to_int
from tsig.util.cachable import cachable, optional_cachable
from tsig.util.db import Database
from tsig.util.configurable import ConfigurableObject
(X, Y) = range(2)


class PSFSource(object):
    """Source for PSFs"""

    def get_config(self):
        return dict()

    @staticmethod
    def _get_resolution(grid, resolution=None):
        if resolution:
            res = resolution
        elif grid.shape[0] % 101 == 0 and grid.shape[1] % 101 == 0:
            # if divisible by 101, assume resolution of 101
            res = 101
        elif grid.shape[0] % 11 == 0 and grid.shape[1] % 11 == 0:
            # if divisible by 11, assume resolution of 11
            res = 11
        else:
            res = 1
        return res

    def get_info(self):
        return "PSFSource"

    @staticmethod
    def interpolate_area(a, b, c, d, position=None, angle=None):
        """
        Do a bilinear interpolation of 4 PSFs to obtain another PSF.
        Use relative areas for the weightings.

        a - PSF 0,0
        b - PSF 0,1
        c - PSF 1,0
        d - PSF 1,1

        Origin for x,y is lower left corner.
        """
        assert(a.grid.shape == b.grid.shape == c.grid.shape == d.grid.shape)
        assert(a.resolution == b.resolution == c.resolution == d.resolution)
        psf = PSF(np.zeros(shape=a.grid.shape, dtype=float),
                  resolution=a.resolution)
        if angle is not None:
            psf.angle = angle
            area_d = (d.angle[X] - angle[X]) * (d.angle[Y] - angle[Y])
            area_c = (angle[X] - c.angle[X]) * (c.angle[Y] - angle[Y])
            area_b = (b.angle[X] - angle[X]) * (angle[Y] - b.angle[Y])
            area_a = (angle[X] - a.angle[X]) * (angle[Y] - a.angle[Y])
            area = (d.angle[X] - a.angle[X]) * (d.angle[Y] - a.angle[Y])
        if position is not None:
            psf.position = position
            area_d = (d.position[X] - position[X]) * \
                     (d.position[Y] - position[Y])
            area_c = (position[X] - c.position[X]) * \
                     (c.position[Y] - position[Y])
            area_b = (b.position[X] - position[X]) * \
                     (position[Y] - b.position[Y])
            area_a = (position[X] - a.position[X]) * \
                     (position[Y] - a.position[Y])
            area = (d.position[X] - a.position[X]) * \
                   (d.position[Y] - a.position[Y])
        psf.grid = (area_d * a.grid + area_c * b.grid +
                    area_b * c.grid + area_a * d.grid) / area
        return psf

    @staticmethod
    def interpolate_dist(a, b, c, d, position=None, angle=None):
        """
        Do a bilinear interpolation of 4 PSFs to obtain another PSF.
        Use distance to determine the weightings.

        a - PSF 0,0
        b - PSF 0,1
        c - PSF 1,0
        d - PSF 1,1

        Origin for x,y is lower left corner.
        """
        assert(a.grid.shape == b.grid.shape == c.grid.shape == d.grid.shape)
        assert(a.resolution == b.resolution == c.resolution == d.resolution)
        psf = PSF(np.zeros(shape=a.grid.shape, dtype=float),
                  resolution=a.resolution)
        if angle is not None:
            x = angle[0]
            y = angle[1]
            psf.angle = (x, y)
            ax = a.angle[X] - x
            ay = a.angle[Y] - y
            bx = b.angle[X] - x
            by = b.angle[Y] - y
            cx = c.angle[X] - x
            cy = c.angle[Y] - y
            dx = d.angle[X] - x
            dy = d.angle[Y] - y
        if position is not None:
            x = position[0]
            y = position[1]
            psf.position = (x, y)
            ax = a.position[X] - x
            ay = a.position[Y] - y
            bx = b.position[X] - x
            by = b.position[Y] - y
            cx = c.position[X] - x
            cy = c.position[Y] - y
            dx = d.position[X] - x
            dy = d.position[Y] - y
        wA = math.sqrt(dx * dx + dy * dy)
        wB = math.sqrt(cx * cx + cy * cy)
        wC = math.sqrt(bx * bx + by * by)
        wD = math.sqrt(ax * ax + ay * ay)
        tot = wA + wB + wC + wD
        psf.grid = (wA * a.grid + wB * b.grid + wC * c.grid + wD * d.grid) / tot
        return psf


class MatlabSource(PSFSource):
    """
    Open a PSF MatLab file and parse it for reading and storing.
    """
    def __init__(self, filename, resolution=None, fmt=None):
        self.filename = filename
        self.fmt = fmt
        self.resolution = resolution
        logger.debug("filename=%s fmt=%s" % (filename, fmt))

        if not self.filename:
            raise ValueError("No file specified")

        if not os.path.isfile(self.filename):
            raise IOError("File not found: %s" % self.filename)

    def get_config(self):
        return {
            'type': 'matlab',
            'filename': self.filename,
            'format': self.fmt,
            'resolution': self.resolution,
        }

    def get_info(self):
        return "matlab %s" % self.filename

    def __iter__(self):
        """Loads a matlab data file and yields a dict structure"""
        if self.fmt is None:
            self.fmt = self.get_format(self.filename)
        if self.fmt == 2:
            return self.read_format_2()
        elif self.fmt == 1:
            return self.read_format_1()

    @staticmethod
    def get_format(filename):
        try:
            from scipy.io import loadmat
            mp = loadmat(filename)
            return 1
        except (IOError, NotImplementedError):
            return 2

    def read_format_1(self):
        """
        hummingbird.mat

        __version__
        __header__
        prf_bystellar
          field_angles
          stellar_type
          stellar_temp
          PSFimage
          field_position
        __globals__
        """
        logger.debug("read as format 1 from %s" % self.filename)
        try:
            from scipy.io import loadmat
        except ImportError as err:
            raise type(err)("Cannot read matlab file: %s" % str(err))

        try:
            mp = loadmat(self.filename)
        except (NotImplementedError, IOError) as err:
            raise type(err)("Error loading matlab file: %s" % str(err))

        data_key = 'prf_bystellar'
        if data_key not in mp.keys():
            raise KeyError("Matlab file does not contain key %s (%s)" %
                           (data_key, ','.join(mp.keys())))

        for x, psf_in in enumerate(mp[data_key][0]):
            psf_d = dict(zip(psf_in.dtype.names, psf_in))
            psf_dict = {}
            psf_dict['stellar_temp'] = to_int(psf_d['stellar_temp'][0][0])
            psf_dict['stellar_type'] = psf_d['stellar_type'][0][0][0][:2]
            psf_dict['angle'] = tuple(psf_d['field_angles'][0].tolist())
            psf_dict['position'] = tuple(psf_d['field_position'][0].tolist())
            psf_dict['grid'] = psf_d['PSFimage']
            psf_dict['id'] = x
            psf_dict['resolution'] = self._get_resolution(
                psf_dict['grid'], resolution=self.resolution)
            yield psf_dict

    def read_format_2(self):
        """
        Camera_SN06_CCD1_config13_ptSrc_wSi_75C_BFD3p387_PSF.mat
        15mar2017

        #refs#
        PSF_README
          camera
          focus_mm
          optics_temp
        PSF_lambda
          PSF
          field_angles
          field_position
          wavelength
        PSF_stellar
          PSF
          field_angles
          field_position
          stellar_temp
          stellar_type
          weightings


        GaussianPSF_CCD1.mat
        05apr2017

        The matlab struct contains 13x13 PSFs for field points at one degree
        intervals between 0 and 12 deg for CCD1 (upper left quadrant of one
        camera). Each PSF is 17x17 TESS pixels at 101x101 subpixel
        resolution, so the PSF images are 1717x1717 arrays. The PSFs are
        2-dimensional Gaussians with a shape and width that is a function of
        the field position. They should work for coding purposes. Please let
        me know if you have questions about the format.

        #refs#
        PSF_README
          camera
          focus_mm
          optics_temp
        psf_bystellar
          PSF
          field_angles
          field_position
          stellar_temp
          stellar_type
          weightings
        """
        logger.debug("read as format 2 from %s" % self.filename)
        try:
            from tsig.util.highfive import MatLabFile
        except ImportError as err:
            raise type(err)("Cannot read H5 matlab file: %s" % str(err))

        try:
            mp = MatLabFile(self.filename, 'r')
        except Exception as err:
            raise type(err)("Error loading matlab file: %s" % str(err))

        x = 0
        for data in mp.get_iter():
            logger.debug("processing PSF %s" % x)
            psf_dict = {}
            psf_dict['stellar_type'] = None # FIXME: decode stellar_type
            psf_dict['stellar_temp'] = data['stellar_temp'][0][0]
            psf_dict['angle'] = (data['field_angles'][0][0],
                            data['field_angles'][1][0])
            psf_dict['position'] = (data['field_position'][0][0],
                               data['field_position'][1][0])
            psf_dict['grid'] = data['PSF']
            psf_dict['id'] = x
            psf_dict['resolution'] = self._get_resolution(
                psf_dict['grid'], resolution=self.resolution)
            yield psf_dict
            x += 1
            
    def query(self, index=None, resolution=None, stellar_temp=None):
        """Return all PSFs from the file, or just one PSF if index is given."""
        if index is not None:
            for psf in self:
                if psf['id'] == index:
                    yield PSF(**psf)
                    break
        else:
            for psf in self:
                yield PSF(**psf)

    def index_query(self):
        """Returns just the meta data without the grid."""
        for psf in self:
            psf.pop('grid')
            yield psf

    def position_query(self, x, y, **ignored):
        return self.query(index=0)

    def angle_query(self, x, y, **ignored):
        return self.query(index=0)

    @optional_cachable('interpolated')
    def interpolated_position_query(self, x, y, resolution, stellar_temp=None, **kw):
        for psf in self.query(index=0):
            if psf is not None and psf.resolution != resolution:
                psf = PSF.resample(psf, resolution)
            return psf
        return None

    def get_nearest_temperature(self, teff):
        return None


class DatabaseSource(PSFSource, Database):
    """
    Query databases for PSF files
    """
    DEFAULT_PSF_DB = 'psf_4'
    CREDENTIALS_FILENAME = '~/.config/tsig/psf-dbinfo'
    CREDENTIALS_URL = 'http://tessellate.mit.edu/tsig/psf-dbinfo'

    schema = [
        ('id', 'SERIAL PRIMARY KEY'),
        ('resolution', 'INTEGER'), # subpixels per pixel
        ('camera_id', 'INTEGER'),
        ('ccd_id', 'INTEGER'),
        ('stellar_type', 'CHAR(2)'),
        ('stellar_temp', 'INTEGER'), # temperature of light source
        ('position', 'POINT'), # chief ray position on CCD, in mm
        ('angle', 'POINT'), # field angle corresponding to position
        ('grid', 'REAL[][]'),
    ]

    def __init__(self, **kwargs):
        Database.__init__(self, **kwargs)

        if 'dbname' not in self.dbinfo:
            self.dbinfo['dbname'] = DatabaseSource.DEFAULT_PSF_DB
            logger.debug("no database specified, using %s" %
                         self.dbinfo['dbname'])

    @staticmethod
    def create_database(dbinfo_loc, dbname):
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        except ImportError as e:
            logger.error("PostgreSQL python module psycopg2 is required")
            logger.debug("psycopg2 import failed: %s" % e)
            return
        dbhost = dbinfo['dbhost']
        dbadmin = dbinfo['dbadmin']
        dbpass = dbinfo['dbpass']
        dbuser = dbinfo['dbuser']
        con = psycop2g.connect(dbname='postgres', host=dbhost, user=dbadmin, password=dbpass)
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = con.cursor()
        cur.execute('CREATE DATABASE ' + dbname)
        cur.execute('GRANT ALL PRIVILEGES ON ' + dbname + ' TO ' + dbuser)
        cur.close()
        con.close()
        con = psycop2g.connect(dbname=dbname, host=dbhost, user=dbadmin, password=dbpass)
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = con.cursor()
        cur.execute('GRANT SELECT ON ' + self.table + ' TO ' + dbuser)
        cur.close()
        con.close()

    # prefix the name of each cache file with the database name
    def cache_prefix(self):
        return self.dbinfo.get('dbname', '')

    def get_config(self):
        return {
            'type': 'database',
            'database_name': self.dbinfo.get('dbname'),
            'database_host': self.dbinfo.get('dbhost'),
            'database_table': self.table,
        }

    def get_info(self):
        info = "database %s at %s" % (
            self.dbinfo.get('dbname'), self.dbinfo.get('dbhost'))
        if 'port' in self.dbinfo:
            info = "%s:%s" % (info, self.dbinfo['port'])
        return info

    @property
    def table(self):
        return self.dbinfo.get('dbtable', 'psf')

    @staticmethod
    def distance(field, x, y):
        """Return a distance field used in queries"""
        return "%s <-> '(%f,%f)'::point as distance" % (
            field, float(x), float(y))

    def index_query(self, **where):
        """Return a simple list of PSFs"""
        # Get list of cols but always ignore the grid
        cols = [name for (name, _) in self.schema if name != 'grid']
        return self.select(*cols, **where)

    def query(self, psf_id, resolution=None, stellar_temp=None):
        """Query for a single PSF by identifier"""
        if resolution is not None and stellar_temp is not None:
            query_result = self.select(
                id=psf_id, resolution=resolution, stellar_temp=stellar_temp)
        elif resolution is not None:
            query_result = self.select(id=psf_id, resolution=resolution)
        elif stellar_temp is not None:
            query_result = self.select(id=psf_id, stellar_temp=stellar_temp)
        else:
            query_result = self.select(id=psf_id)
        return self._to_psf(query_result)

    @optional_cachable('source')
    def position_query(self, x, y, resolution, stellar_temp=3600, limit=4):
        """Make a query for PSFs by field position"""
        query_result = self.select("*", self.distance('position', x, y),
                                   order="distance", stellar_temp=stellar_temp,
                                   resolution=resolution, limit=limit)
        return self._to_psf(query_result)

    def angle_query(self, x, y, resolution, stellar_temp=3600, limit=4):
        """Make a query for PSFs by field angle"""
        query_result = self.select("*", self.distance('angle', x, y),
                                   order="distance", stellar_temp=stellar_temp,
                                   resolution=resolution, limit=limit)
        return self._to_psf(query_result)

    @optional_cachable('interpolated')
    def interpolated_position_query(self, x, y, resolution, stellar_temp=3600, **kw):
        # default to stellar temperature of 3600 (M1 stars)
        # convert to mm from pixels
        x_mm = x * 0.015 # FIXME
        y_mm = y * 0.015 # FIXME
        (a, b, c, d) = self.position_query(x_mm, y_mm, resolution, **kw)
        psf = PSFSource.interpolate_dist(a, b, c, d, position=(x_mm, y_mm))
        if psf.resolution != resolution:
            psf = PSF.resample(psf, resolution)
        return psf

    @staticmethod
    def _to_psf(query_result):
        for data in query_result.as_dict():
            # This correction required because the postgres database returns
            # strings for POINT objects, (sometimes?) unsure why, but correct
            # when needed.
            for (t, k) in ((float, 'angle'), (float, 'position')):
                if isinstance(data[k], str):
                    data[k] = tuple(t(p.strip()) for p in data[k].strip('(').strip(')').split(','))
            data['grid'] = np.array(data['grid'])
            yield PSF(**data)

    def add_file(self, filename, resize=None, normalize=False,
                 camera=None, ccd=None, resolution=None):
        """
        Insert all PSFs from a given matlab file.
        
        If width and height are provided, then downsample the grid before
        storage.

        If normalize is true, then normalize the grid values before storage.
        """
        for psf in MatlabSource(filename, resolution=resolution).query():
            logger.info("Adding %s" % str(psf))
            if resize is not None:
                logger.info("Resizing to %s" % str(resize))
                psf = psf.resize(*resize)
            if normalize:
                logger.info("Normalizing")
                psf.normalize()
            self.insert(
                camera_id=camera,
                ccd_id=ccd,
                stellar_type=psf.stellar_type,
                stellar_temp=psf.stellar_temp,
                position=psf.position,
                angle=psf.angle,
                resolution=psf.resolution,
                grid=psf.grid.tolist(),
            )
        return True

    def get_nearest_temperature(self, teff):
        """
        Get the nearest temperature.  The database contains these temperatures:
        """
        known_temps = [2800, 3250, 3600, 4350, 5250, 5770, 6030, 6440, 7200]
        for i in range(len(known_temps) - 1):
            if teff < known_temps[i] + (known_temps[i + 1] - known_temps[i]) / 2:
                return known_temps[i]
        return known_temps[-1]


class GaussianSource(PSFSource):
    """
    Return a gradient PSF.  Default to a resolution of 11x11 subpixels per
    pixel that covers 21x21 CCD pixels, so a grid of 231x231.
    """
    def __init__(self, resolution=11, width=231, height=231, **kwargs):
        self.resolution = to_int(resolution)
        self.stellar_temp = 3600
        self.stellar_type = 'M1'
        self.width = int(width)
        self.height = int(height)

    def get_config(self):
        return {
            'type': 'gaussian',
            'width': self.width,
            'height': self.height,
            'resolution': self.resolution,
            'stellar_temp': self.stellar_temp,
            'stellar_type': self.stellar_type,
        }

    def get_info(self):
        return "gaussian width=%s height=%s" % (self.width, self.height)

    def index_query(self):
        yield {
            'stellar_temp': self.stellar_temp,
            'stellar_type': self.stellar_type,
            'angle': (0, 0),
            'position': (0, 0),
            'id': 0,
            'resolution': self.resolution,
        }

    def query(self, psf_id=None, resolution=None, stellar_temp=None):
        if resolution is None:
            resolution = self.resolution
        origin_x = self.width / 2
        origin_y = self.height / 2
        yield PSF.gradient(width=self.width, height=self.height,
                           resolution=resolution, origin=(origin_x, origin_y))

    def position_query(self, x, y, **ignored):
        return self.query()

    def angle_query(self, x, y, **ignored):
        return self.query()

    @optional_cachable('interpolated')
    def interpolated_position_query(self, x, y, resolution, stellar_temp=None, **kw):
        for psf in self.position_query(x, y):
            if psf.resolution != resolution:
                psf = PSF.resample(psf, resolution)
            return psf

    def get_nearest_temperature(self, teff):
        return None


class PSF(object):
    """
    The origin for a PSF is lower left corner.
    """
    def __init__(self, grid, resolution=1, position=None, angle=None, **other):
        if grid is None:
            grid = np.zeros(shape=(1, 1), dtype=float)
        self.grid = grid
        if isinstance(position, str):
            raise ValueError("position must be a tuple!")
        self.position = position
        if isinstance(angle, str):
            raise ValueError("angle must be a tuple!")
        self.angle = angle
        self.resolution = to_int(resolution)
        self.stellar_type = other.get('stellar_type', None)
        self.stellar_temp = other.get('stellar_temp', None)
        self.id = other.get('id', None)

    def __repr__(self):
        """string description of the PSF"""
        return self.get_label()

    def get_label(self):
        label = "PSF wxh=(%d,%d)" % (self.grid.shape[1], self.grid.shape[0])
        if self.position is not None:
            label += " pos=(%.2f,%.2f)" % (self.position[X], self.position[Y])
        if self.angle is not None:
            label += " ang=(%.2f,%.2f)" % (self.angle[X], self.angle[Y])
        label += " temp=%s" % self.stellar_temp
        label += " res=%d" % self.resolution
        label += " sum=%.0f" % self.grid.sum()
        return label

    def get_contents(self):
        """
        Return the PSF contents as an ASCII string.
        """
        csv = BytesIO()
        np.savetxt(csv, self.grid, fmt="%.8f", delimiter=',')
        csv.seek(0)
        return """# id=%s
# resolution=%s
# stellar_type=%s
# stellar_temp=%s
# field_position=(%.4f,%.4f)
# field_angle=(%d,%d)
# rows=%d cols=%d
%s""" % (
            str(self.id), self.resolution,
            str(self.stellar_type), str(self.stellar_temp),
            self.position[X], self.position[Y],
            self.angle[X], self.angle[Y],
            self.grid.shape[0], self.grid.shape[1], csv.read())

    def make_plot(self, show_grid=True, ax=None, cmap='jet'):
        """
        Make a plot of the PSF.

        show_grid - if true, show a grid of pixel borders
        """
        # FIXME: pixel border rendering is not correct - it does not
        #        distinguish pixel bodies from pixel edges
        import matplotlib.pylab as plt
        if ax is None:
            plt.figure(self.get_label())
            ax = plt.subplot()
        else:
            plt.sca(ax)
        ax.imshow(self.grid, cmap=cmap, interpolation='none')
        title = "(%s,%s) (%.3f,%.3f)\nres=%s sum=%.2f" % (
            self.grid.shape[1], self.grid.shape[0],
            self.position[X], self.position[Y],
            self.resolution, self.grid.sum())
        ax.set_title(title)
        ax.set_aspect(1)
        ax.invert_yaxis()
        if show_grid:
            from matplotlib import collections
            width = self.grid.shape[1]
            height = self.grid.shape[0]
            resolution = self.resolution
            edges = []
            for i in range(0, width, resolution):
                edges.append([(i, 0), (i, height)])
            for i in range(0, height, resolution):
                edges.append([(0, i), (width, i)])
            ax.add_collection(collections.LineCollection(edges, alpha=0.6))
        return plt.gcf()

    def normalize(self):
        """
        Normalize the PSF, result is that the sum of all values of the grid
        add to 1.0.
        """
        self.grid /= self.grid.sum()
        return self

    # FIXME: why does this create a new PSF?  it should return nothing and
    # just modify the self.grid, or it should be a class method an return
    # a new PSF
    def resize(self, width, height):
        """
        Return a copy of the PSF sized to the specified width and height.
        """
        # defer the import since imresize depends on PIL/Pillow
        from scipy.misc import imresize
        grid = imresize(self.grid, (height, width)).astype(float)
        return type(self)(grid, resolution=self.resolution,
                          position=self.position, angle=self.angle,
                          stellar_temp=self.stellar_temp,
                          stellar_type=self.stellar_type)

    @classmethod
    def resample(cls, psf, resolution=1):
        """
        Return a copy of the PSF with new size based on the resolution.
        """
        ratio = float(resolution) / float(psf.resolution)
        new_w = ratio * psf.grid.shape[1]
        new_h = ratio * psf.grid.shape[0]
        new_psf = psf.resize(int(new_w), int(new_h))
        new_psf.resolution = resolution
        return new_psf

    @classmethod
    def random(cls, resolution=1, width=21, height=31,
               position=(0.0, 0.0), angle=(0.0, 0.0)):
        """
        Create a PSF with random values in its grid.
        """
        return cls(np.random.random((height, width)), resolution=resolution,
                   position=position, angle=angle)

    @classmethod
    def gradient(cls, resolution=1, width=21, height=31,
                 position=(0.0, 0.0), angle=(0.0, 0.0),
                 origin=(0,0), A=100.0):
        """
        Create a PSF with gaussian gradient.
        """
        grid = np.zeros(shape=(height, width), dtype=float)
        for i in range(grid.shape[1]):
            # XXX Use np.gradient?
            for j in range(grid.shape[0]):
                grid[j, i] = PSF._gradient(i, j, origin=origin, A=A)
        return cls(grid, resolution=resolution, position=position, angle=angle)

    @staticmethod
    def _gradient(x, y, origin, A=100.0):
        """
        Two-dimensional function with highest value at origin then diminishing
        in every direction from there.  Value drops off from maxval at the
        origin to zero, using a two-dimensional gaussian.

        origin is w,h
        """
        a = c = 0.001
        b = 0.0
        return A * math.exp(-(a * (x - origin[X]) ** 2
                              - 2 * b * (x - origin[X]) * (y - origin[Y])
                              + c * (y - origin[Y]) ** 2))
