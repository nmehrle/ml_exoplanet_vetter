#!/usr/bin/env python

import astropy
import astropy.constants
import astropy.coordinates
import astropy.time
import math
import numpy

# Presumably we can get all these from astropy somehow.  Not sure
# about GMSUN, but I advise doing it this way rather than taking G
# and multiplying MSUN (the product is known more precisely than
# the individual values).

AU = astropy.constants.au.to_value("m")

GMSUN = 1.32712440041e20  # m^3 / s^2, IAU 2009, TDB compatible
LIGHT = 2.99792458e8      # m/s, definition
DAY = 86400.0

# For precise applications, set solar system ephemeris to JPL here.
# The ERFA defaults actually seem to have decent precision so I
# haven't tried it yet.

# Compute for this UTC
utc = 56470.3
time = astropy.time.Time(utc, format="mjd", scale="utc")

# Location of observer.
obs = astropy.coordinates.EarthLocation.from_geodetic(lat=114063.1/3600.0, lon=-399161.0/3600.0, height=2384.0)

# Target.  Strictly speaking, the coordinates should be BCRS at "time",
# i.e. corrected for proper motion.  I presume astropy can do this but
# couldn't figure out how so leave it up to the user.
target = astropy.coordinates.SkyCoord(269.45402263, 4.66828781, unit="deg")

# If a parallax is available...
plx = astropy.units.Quantity(548.31, unit="mas")
pr = plx.to_value("radian")

# Geocentric position and velocity vector of observer (GCRS).
gop, gov = obs.get_gcrs_posvel(time)

# Barycentric position of Geocenter and Sun.
bep = astropy.coordinates.get_body_barycentric("earth", time)
bsp = astropy.coordinates.get_body_barycentric("sun", time)

# Barycentric and Heliocentric position vectors of observer.
bop = bep + gop
hop = bop - bsp

# Convert to numpy 3-vectors in AU.
bop = bop.get_xyz().to_value("AU")
hop = hop.get_xyz().to_value("AU")

# Distance from observer to Sun, AU.
hdist = numpy.linalg.norm(hop)

# BCRS position vector of target (corrected for proper motion).
s = target.cartesian.get_xyz().to_value()

# Make sure properly normalized.
s /= numpy.linalg.norm(s)

# Romer delay for infinite source distance: light path length
# between observer and SSB in AU.
dr = s.dot(bop)

if pr > 0:
  # First order correction for wavefront curvature if source
  # parallax is known.
  vt = numpy.cross(s, bop)
  dr -= 0.5 * pr * vt.dot(vt)

  # Compute normalized observer to source vector.
  p = s - pr * bop
  p /= numpy.linalg.norm(p)

  # Scalar product in Shapiro delay.
  stmp = p.dot(hop)
else:
  stmp = s.dot(hop)

# Shapiro delay due to the Sun.  Planets are neglected.
ds = 2.0 * math.log(1.0 + stmp / hdist) * GMSUN / (LIGHT * LIGHT)

# Time delay in seconds.
delay = (dr * AU + ds) / LIGHT

print "DELAY =", delay

# BJD-TDB.
bjdtdb = time.tdb.jd + delay / DAY

print "BJDTDB = {0:.10f}".format(bjdtdb)





