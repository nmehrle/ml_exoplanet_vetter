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

"""
Utilities for converting between times.

  TJD - TESS Julian Date                                              

  A Free running clock that does not adjust for leap seconds.   
  Reported in Days since "2014 Dec 8 11:58:52.816738 TDB"       

  The TJD epoch is 1418039932.816738 seconds in the unix epoch, or
  2456999.999222 days in the Julian epoch.

  00:00:00 01mar2018 is:
    1519862400 unix epoch
    2458178.5 JD
    18178 truncated JD
    1178.5 TJD
"""

# the TJD epoch in decimal years
#  (365.25 - (7.183262 / 86400 + 1 / 1440 + 12.0 / 24) ) / 365.25
TJD_DECIMAL_YEAR = 2014.9986308469825969
# the TJD epoch in seconds
TJD_EPOCH_S = 1418039932.816738

def tjd2dy(tjd):
    """Convert TESS Julian Date to decimal year."""
    return tjd / 365.0 + TJD_DECIMAL_YEAR

def dy2tjd(year):
    """Convert decimal year to TESS Julian Day."""
    return (year - TJD_DECIMAL_YEAR) * 365.0

def tjd2utc(tjd):
    """Convert TESS Julian Date to UTC."""
    return tjd * 86400.0 + TJD_EPOCH_S

def utc2tjd(utc):
    """Convert time in UTC epoch to TESS Julian Date."""
    return (utc - TJD_EPOCH_S) / 86400.0

# start of gps time is 1980.01.01 00:00:00
GPS_EPOCH_S = 315964800
# number of leap seconds since gps epoch as of june 2017
GPS_LEAP_SECONDS = 18

def gps2utc(gps):
    """Convert GPS time to UTC"""
    return gps + GPS_EPOCH_S + GPS_LEAP_SECONDS

def utc2gps(utc):
    """Convert UTC time to GPS"""
    return utc - GPS_EPOCH_S - GPS_LEAP_SECONDS
