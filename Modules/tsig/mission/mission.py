#!/usr/bin/env python
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

import math
import logging
import os
import configobj
import time

logger = logging.getLogger(__name__)

from tsig.util.configurable import ConfigurableObject, Configuration
from tsig.spacecraft.quaternion import *
from tsig.util.pointing import *
from tsig.util.clock import tjd2utc, utc2tjd


# If no pointing is specified, use this one (values in degrees)
DEFAULT_POINTING = (30, 45, 0)

# If no start time is specified, use this one
DEFAULT_START_TJD = 1178 # 01mar2017


class MissionProfile(ConfigurableObject):
    """
    Contains all the spacecraft positioning for each of the times required
    for the observation.
    """
    SECTORS_FN = os.path.join(os.path.dirname(__file__), 'sectors.cfg')
    EPHEMERIS_FN = os.path.join(os.path.dirname(__file__), 'ephemeris.csv')

    def __init__(self, ephemeris_filename=None, sectors_filename=None,
                 pointing=None, start_time=None, spacecraft_state=None):
        """
        The pointing is constant for the duration of the mission.

        The spacecraft state can be a tuple of 6 values (x, y, z, vx, vy, vz)
        in km and km/s, or the string 'use_ephemeris'.

        The start time can be specified as YYYY-mm-ddTHH:MM:SS in UTC or
        as TESS Julian Date.  If using ephemeris, the time must be within
        the ephemeris data.

        If an ephemeris filename is specified, use that instead of the default
        ephemeris.

        If a sectors filename is specified, use that instead of the default
        sectors.
        """
        pointing, sector = MissionProfile.parse_pointing(
            pointing, sectors_filename)
        self.sector = sector
        self.pointing_ra = float(pointing[0])
        self.pointing_dec = float(pointing[1])
        self.pointing_roll = float(pointing[2])

        self.start_tjd = MissionProfile.parse_time(start_time)
        self.start_utc = tjd2utc(self.start_tjd)

        if spacecraft_state is None:
            spacecraft_state = (0, 0, 0, 0, 0, 0)
        elif isinstance(spacecraft_state, list):
            spacecraft_state = [float(x) for x in spacecraft_state]
        self.spacecraft_state = spacecraft_state

        self.ephemeris_filename = ephemeris_filename

        # default to no ephemeris data.  read the data only if requested.
        self.state = []
        if isinstance(spacecraft_state, str):
            self.spacecraft_state = (0, 0, 0, 0, 0, 0) # fallback state
            self.load(self.ephemeris_filename)

    def get_config(self):
        return {
            'sector': self.sector,
            'pointing': {
                'ra': self.pointing_ra,
                'dec': self.pointing_dec,
                'roll': self.pointing_roll,
            },
            'start_utc': self.start_utc,
            'start_tjd': self.start_tjd,
            'ephemeris_filename': self.ephemeris_filename,
        }

    def get_spacecraft_state(self, tjd):
        """
        Return the spacecraft position and velocity for the indicated
        time.  The specified time is a TESS julian day.

        If the time is before or after the range, use default state.
        """
        position = self.spacecraft_state[0:3]
        velocity = self.spacecraft_state[3:]
        if self.state:
            idx = None
            for i in range(len(self.state) - 1):
                if tjd >= self.state[i][0] and tjd < self.state[i + 1][0]:
                    idx = i
                    break
            if idx is not None and idx + 1 < len(self.state):
                frac = (tjd - self.state[idx][0]) / (self.state[idx + 1][0] - self.state[idx][0])
                def interp(idx, k):
                    return self.state[idx][k] + frac * (self.state[idx + 1][k] - self.state[idx][k])
                position = (interp(idx, 1), interp(idx, 2), interp(idx, 3))
                velocity = (interp(idx, 4), interp(idx, 5), interp(idx, 6))
            print tjd, idx
        return position, velocity


    def save(self, filename):
        """
        Save to file.  Each line is:

        timestamp x y z vx vy vz
        """
        try:
            with open(filename, "w") as f:
                f.write("# mission ephemeris\n")
                f.write("# time(tjd) x y z vx vy vz\n")
                for row in self.state:
                    f.write(' '.join(["%.3f" % x for x in row]))
                    f.write('\n')
        except OSError as e:
            logger.error("write failed: %s" % e)

    def load(self, filename):
        """
        Read a file for positions and velocities.  Each line is:

        timestamp x y z vx vy vz

        Lines that begin with # are ignored.

        Times are TESS julian date.
        """
        if filename is None:
            filename = MissionProfile.EPHEMERIS_FN
        if not os.path.isfile(filename):
            raise IOError("Cannot read ephemeris file: %s" % filename)
        self.state = []
        try:
            logger.info("read ephemeris file %s" % filename)
            with open(filename) as f:
                lineno = 0
                for line in f:
                    lineno += 1
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('#'):
                            continue
                        delim = ' '
                        if ',' in line:
                            delim = ','
                        values = [float(v) for v in line.split(delim)]
                        if len(values) != 7:
                            raise ValueError("wrong number of values: expected 7, found %s" % len(values))
                        self.state.append(values)
                    except ValueError as e:
                        logger.error("problem at line %s '%s': %s" %
                                     (lineno, line, e))
        except OSError as e:
            logger.error("read failed: %s" % e)

    @staticmethod
    def get_sector_pointing(sector, filename=None):
        if filename is None:
            filename = MissionProfile.SECTORS_FN
        sectors = configobj.ConfigObj(filename)
        sector = sector.lower()
        if sector in sectors:
            logger.debug("using pointing from %s" % sector)
            return float(sectors[sector][0]), float(sectors[sector][1]), float(sectors[sector][2])
        raise ValueError("No pointing for sector '%s'" % sector)

    @staticmethod
    def parse_pointing(pointing, sectors_filename=None):
        # FIXME: there must be a better way to do this...
        sector = None
        order = 'wxyz'
        if isinstance(pointing, str):
            if pointing.lower().startswith('sector'):
                sector = pointing
                logger.debug("found sector %s" % sector)
                pointing = MissionProfile.get_sector_pointing(
                    pointing, filename=sectors_filename)
            elif ',' in pointing:
                pointing = pointing.split(',')
                if pointing and len(pointing) == 3:
                    pointing = [float(x) for x in pointing]
        if pointing and len(pointing) == 5:
            order = pointing[0].lower()
            pointing = pointing[1:]
        quat = to_quaternion(pointing)
        if quat is not None:
            logger.debug("found quaternion %s" % (quat,))
            if order == 'xyzw':
                qw = quat[3]
                qx = quat[0]
                qy = quat[1]
                qz = quat[2]
            else:
                qw = quat[0]
                qx = quat[1]
                qy = quat[2]
                qz = quat[3]
            pointing = QuaternionTransform.quat_to_rdr((qw, qx, qy, qz))
        if pointing is None:
            logger.debug("no valid pointing, using default")
            pointing = DEFAULT_POINTING
        logger.debug("using pointing %s" % (pointing,))
        return pointing, sector

    @staticmethod
    def parse_time(ts):
        """Parse a time.  The input time could be TESS julian date or UTC.
        Return a TESS julian date."""
        tjd = None
        try:
            tjd = float(ts)
        except (TypeError, ValueError):
            try:
                utc = time.mktime(time.strptime(ts, '%Y-%m-%dT%H:%M:%S'))
                tjd = utc2tjd(utc)
            except (TypeError, ValueError):
                pass
        if tjd is None:
            logger.debug("cannot determine time from '%s', using %s" % (ts, tjd))
            tjd = DEFAULT_START_TJD
        return tjd

    @staticmethod
    def pointing_to_rdr(pointing, sector_file=None):
        """Given a string, return ra,dec,roll.  The string can be any of:

        W,X,Y,Z
        wxyz,W,X,Y,Z
        xyzw,X,Y,Z,W
        RA,DEC,ROLL
        sectorN
        """
        ra = dec = roll = None
        if is_quaternion(pointing):
            w, x, y, z = to_quaternion(pointing)
            ra, dec, roll = QuaternionTransform.quat_to_rdr((w, x, y, z))
        elif is_rdr(pointing):
            ra, dec, roll = to_rdr(pointing)
        elif pointing and pointing.lower().startswith('sector'):
            ra, dec, roll = MissionProfile.get_sector_pointing(
                pointing.lower(), sector_file)
        elif pointing and pointing.lower().startswith('wxyz,'):
            w, x, y, z = to_quaternion(pointing[5:])
            ra, dec, roll = QuaternionTransform.quat_to_rdr((w, x, y, z))
        elif pointing and pointing.lower().startswith('xyzw,'):
            x, y, z, w = to_quaternion(pointing[5:])
            ra, dec, roll = QuaternionTransform.quat_to_rdr((w, x, y, z))
        return ra, dec, roll
