#
# Copyright (C) 2015 - Zach Berta-Thompson <zkbt@mit.edu> (MIT License)
#               2017 - Massachusetts Institute of Technology (MIT)
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

"""Spacecraft geometry models.

cam_id - camera identifier, one of 1, 2, 3, or 4
ccd_id - CCD identifier, one of 1, 2, 3, or 4

These are the geometry models:

LevineModel  - Designed by Al Levine, this model uses the Levine
               parameterization.  There are two variants of the Levine
               parameterization: Levine1 and Levine2.

               Levine1 - Camera angle relative to the spacecraft is defined
               by a single, scalar value.  Two of the CCDs are rotated in
               the transformation matrix and must be flipped back for rendering

               Levine2 - Similar to Levine1, but this uses a 3-element euler
               angle instead of a single angle for specifying the relation
               between the spacecraft and each camera.

KephartModel - The Ames model as implemented by Miranda Kephart.  Uses the
               Ames parameterization.

AmesModel    - Uses the Ames parameterization.
"""

# FIXME: can we eliminate the BUFFER in focal_to_pixel?

import math
import numpy as np
from tsig.util.pointing import to_quaternion    
from .quaternion import normalize_ra, normalize_dec, KephartTransform, NguyenTransform

# FIXME: these belong in constants
SPEED_OF_LIGHT_KPS = 299792.458 # km/s
RAD_TO_ARCSEC = 360 * 3600 / (2 * math.pi)
DEG_TO_RAD = math.pi / 180.0

NUM_SCIENCE_COLS = 2048
NUM_LEADING_COLS = 44

NUM_SCIENCE_ROWS = 2048
NUM_BUFFER_ROWS = 10

def to_array(x):
    if isinstance(x, (type(None), int, float)):
        return [x]
    return x

def calculate_jitter_deltas(x_fp, y_fp, focal_length, offset_angle,
                            (q0, q1, q2, q3)):
    """Calculate change to focal plane coordinates due to jitter as specified
    in the delta quaternion components"""
    psi = 2 * q1
    theta = 2 * q2
    phi = 2 * q3
    f_eff = focal_length
    theta_i = offset_angle
    delta_x = f_eff * (-phi * math.sin(theta_i) + psi * math.cos(theta_i)) - \
               y_fp * (-phi * math.cos(theta_i) - psi * math.sin(theta_i))
    delta_y = f_eff * theta + x_fp * (- phi * math.cos(theta_i) \
                                      - psi * math.sin(theta_i))
    return (delta_x, delta_y)


class LevineModel(object):
    """
    Use the Levine geometry and focal models to convert between coordinates.

    Details in the tsig-math-1.pdf document.
    """

    @staticmethod
    def spacecraft_to_camera_pointing(ra_sc, dec_sc, roll_sc, camera):
        """
        Get the (ra,dec,roll) of a camera given the spacecraft (ra,dec,roll).
        """
        return LevineModel._sc_to_cam_pointing(ra_sc, dec_sc, roll_sc,
                                               camera.get_angle())

    @staticmethod
    def _sc_to_cam_pointing(ra_sc, dec_sc, roll_sc, cam_angle):
        # get the tranformation matrix from equatorial to camera
        m1 = LevineModel.get_eq_sc_matrix(ra_sc, dec_sc, roll_sc)
        m2 = LevineModel.get_sc_cam_matrix(cam_angle)
        m = np.dot(m2, m1)
        # unit vectors in the camera reference frame
        x_cam = m[0]
        y_cam = m[1]
        z_cam = m[2]
        # unit vector of z axis in the equatorial
        z_eq = np.array([0, 0, 1])
        # from those get the ra and dec
        ra_cam = math.atan2(z_cam[1], z_cam[0]) * 180.0 / math.pi
        dec_cam = 90.0 - math.acos(z_cam[2]) * 180.0 / math.pi
        # get the east and north unit vectors to get roll
        e_vec = np.cross(z_eq, z_cam) / np.linalg.norm(np.cross(z_eq, z_cam))
        n_vec = np.cross(z_cam, e_vec)
        x = np.dot(x_cam, e_vec)
        y = np.dot(x_cam, n_vec)
        roll_cam = math.atan2(y, x) * 180.0 / math.pi
        # FIXME: deal with case where z_eq x z_cam is zero
        return ra_cam, dec_cam, roll_cam

    @staticmethod
    def celestial_to_pixel(ra, dec, ra_sc, dec_sc, roll_sc,
                           cam_geometry, ccd_geometries, v_sc=None):
        """
        Transform celestial coordinates (ra,dec) in degrees to CCD coordinates
        (col,row) in pixels on the indicated camera with spacecraft oriented
        at (ra_sc,dec_sc,roll_sc) in degrees.

        If spacecraft velocity v_sc is specified, then apply DVA.  Spacecraft
        velocity is a tuple (x,y,z) of the velocity components, in km/s.
        """
        # transform from celestial to focal plane
        x, y = LevineModel.celestial_to_focal(
            ra, dec, ra_sc, dec_sc, roll_sc,  cam_geometry, v_sc)
        # the levine1 implementation uses the camera id to flip CCDs, whereas
        # the levine2 implementation does not.  if there is a parameter
        # 'offset_angle' or 'angle', then we are using levine1.  otherwise
        # we are using the levine2 parameterization.
        cam_id = 0
        if 'offset_angle' in cam_geometry or 'angle' in cam_geometry:
            cam_id = cam_geometry['number']
        # transform from focal plane to pixels
        col, row, ccd = LevineModel.focal_to_pixel(
            x, y, cam_id, ccd_geometries)
        return col, row, ccd

    @staticmethod
    def celestial_to_focal(ra, dec, ra_sc, dec_sc, roll_sc, cam_geometry,
                           v_sc=None):
        """
        Transform celestial coordinates (ra,dec) in degrees to focal plane
        coordinates (x,y) in mm on the indicated camera with spacecraft
        oriented at (ra_sc,dec_sc,roll_sc) in degrees.

        If spacecraft velocity v_sc is specified, then apply DVA.  Spacecraft
        velocity is a tuple (x,y,z) of the velocity components, in km/s.
        """
        # for backward compatibility, see if this is a levine1 parameterization
        # levine1 uses either 'angle' or 'offset_angle' as the scalar name.
        cam_angle = cam_geometry.get('angle')
        if cam_angle is None:
            cam_angle = cam_geometry.get('offset_angle')
        if cam_angle is None:
            # otherwise, use the levine2 parameters
            cam_angle = (cam_geometry['angle_alpha'],
                         cam_geometry['angle_beta'],
                         cam_geometry['angle_gamma'])

        # transform ra,dec to camera ra,dec
        ra_cam, dec_cam = LevineModel.celestial_to_camera(
            ra, dec, ra_sc, dec_sc, roll_sc, cam_angle, v_sc)

        # transform ra,dec to focal coordinates
        x, y = LevineModel.camera_to_focal(ra_cam, dec_cam, cam_geometry)
        return x, y
  
    @staticmethod
    def celestial_to_camera(ra, dec, ra_sc, dec_sc, roll_sc, cam_angle,
                            v_sc=None):
        """
        Transform celestial coordinates (ra,dec) in degrees to camera
        coordinates (ra,dec) in degrees given spacecraft pointing
        (ra_sc,dec_sc,roll_sc) in degrees.

        If spacecraft velocity v_sc is specified, then apply DVA.  Spacecraft
        velocity is a tuple (x,y,z) of the velocity components, in km/s.
        """
        # ensure that the inputs are arrays
        ra = to_array(ra)
        dec = to_array(dec)
        # if we got a velocity, divide by speed of light
        v = None
        if v_sc is not None:
            v = np.array(v_sc, dtype=float)
            v /= SPEED_OF_LIGHT_KPS
        # allocate ra,dec arrays for the results
        ra_cam = np.array(ra, dtype=float)
        dec_cam = np.array(dec, dtype=float)
        # allocate field angle x,y arrays
#        fa_x = np.zeros(shape=ra_cam.shape)
#        fa_y = np.zeros(shape=ra_cam.shape)
        # generate the transformation matrix
        m1 = LevineModel.get_eq_sc_matrix(ra_sc, dec_sc, roll_sc)
        m2 = LevineModel.get_sc_cam_matrix(cam_angle)
        m = np.dot(m2, m1)
        # populate the ra,dec arrays by applying the transform to each element
        for i in range(ra_cam.size):
            ra_r = ra_cam[i] * math.pi / 180.0
            dec_r = dec_cam[i] * math.pi / 180.0
            u = np.array([math.cos(ra_r) * math.cos(dec_r),
                          math.sin(ra_r) * math.cos(dec_r),
                          math.sin(dec_r)])
            # include DVA if we got a spacecraft velocity
            if v is not None:
                u += v # add the velocity component then normalize
                u /= math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
            # get the unit vector
            c = np.dot(m, u)
            if c[2] >= 0:
                ra_cam[i] = math.atan2(c[1], c[0]) * 180.0 / math.pi
                try:
                    dec_cam[i] = math.asin(c[2]) * 180.0 / math.pi
                except ValueError:
                    dec_cam[i] = 90.0
            else:
                # if the z component of the unit vector is negative, then
                # ignore so that we do not project from opposite hemisphere.
                ra_cam[i] = None
                dec_cam[i] = None
            # calculate field angles
#            if c[2]:
#                fa_x[i] = math.atan(c[0] / c[2]) * 180.0 / math.pi
#                fa_y[i] = math.atan(c[1] / c[2]) * 180.0 / math.pi
#            else:
#                fa_x[i] = 90.0
#                fa_y[i] = 90.0
        return ra_cam, dec_cam #, fa_x, fa_y

    @staticmethod
    def camera_to_focal(ra, dec, cam_geometry):
        """
        Transform a camera celestial (ra,dec) to (x,y) on the focal plane.
        Return value is in mm.

        ra - right ascension in the camera refrence frame
        dec - declination in the camera reference frame

        If a declination is beyond the view of the camera, then return None
        for that item.  This avoids the portion of the function that causes
        problems when colat is around 50 degrees.
        """
        lon = np.array(to_array(ra))
        colat = 90.0 - np.array(to_array(dec))
        t = np.tan(colat * math.pi / 180.0)
        t2 = t * t
        r = cam_geometry['focal_length'] * t * (
            cam_geometry['a_0'] +
            cam_geometry['a_2'] * t2 +
            cam_geometry['a_4'] * t2 ** 2 +
            cam_geometry['a_6'] * t2 ** 3 +
            cam_geometry['a_8'] * t2 ** 4)
        x = -r * np.cos(lon * math.pi / 180.0)
        y = -r * np.sin(lon * math.pi / 180.0)
        # eliminate values that are beyond the camera field of view
        x[colat > 25] = None
        y[colat > 25] = None
        return x, y

    @staticmethod
    def focal_to_pixel(x, y, cam_number, ccd_geometries):
        """
        Transform (x,y) on the focal plane to pixel (col,row) on the CCD.

        x, y - coordinates in the focal plane
        camera - camera number 1,2,3, or 4

        return:
        col, row - fractional pixel coordinates on the CCD
        ccd - CCD number 1,2,3, or 4

        geom - geometric properties for a specific camera and ccd
        geom['rotation'] - CCD rotation about the camera boresight
        geom['x_0'] - bottom left x coordinate of imaging area, in mm
        geom['y_0'] - bottom left y coordinate of imaging area, in mm
        geom['pixel_size_x'] - pixel dimension, in mm
        geom['pixel_size_y'] - pixel dimension, in mm

        x_fpp, y_fpp - in the fpp frame, the +x axis runs from the origin at
        the center of the array of 4 CCDs between CCDs 1 and 4, and the +y
        axis runs from the origin between CCDs 1 and 2.

        x_fpr, y_fpr - frame in which x,y axes are parallel to the CCD rows
        and columns, but rotated relative to the fpp frame
        """
        from math import cos, sin
        BUFFER = 10 # pixels
        x = to_array(x)
        y = to_array(y)
        x_fpp = np.array(x)
        y_fpp = np.array(y)
        if cam_number in [1, 2]: # this is only for levine1 parameterization
            x_fpp *= -1
            y_fpp *= -1
        # FIXME: these should default to None not 0
        col = np.zeros(shape=x_fpp.shape, dtype=float)
        row = np.zeros(shape=y_fpp.shape, dtype=float)
        ccd_id = np.zeros(shape=x_fpp.shape, dtype=int)
        for i in range(x_fpp.size):
            if x_fpp[i] > 0 and y_fpp[i] > 0:
                ccd_id[i] = 1
            elif x_fpp[i] < 0 and y_fpp[i] > 0:
                ccd_id[i] = 2
            elif x_fpp[i] < 0 and y_fpp[i] < 0:
                ccd_id[i] = 3
            elif x_fpp[i] > 0 and y_fpp[i] < 0:
                ccd_id[i] = 4
            if ccd_id[i]:
                g = ccd_geometries[ccd_id[i]]
                r = g['rotation'] * math.pi / 180.0
                x_delta = x_fpp[i] - g['x_0']
                y_delta = y_fpp[i] - g['y_0']
                x_fpr = x_delta * cos(r) + y_delta * sin(r)
                y_fpr = y_delta * cos(r) - x_delta * sin(r)
                x_p = x_fpr / g['pixel_size_x']
                y_p = y_fpr / g['pixel_size_y']
                if (x_p > -BUFFER and x_p < g['cols'] + BUFFER and
                    y_p > -BUFFER and y_p < g['rows'] + BUFFER):
                    col[i] = x_p
                    row[i] = y_p
                else:
                    ccd_id[i] = 0 # off the grid
        return col, row, ccd_id

    @staticmethod
    def pixel_to_focal(col, row, ccd_id, ccd_geometries):
        raise NotImplementedError("pixel to focal")

    @staticmethod
    def focal_to_celestial(x, y, ra_sc, dec_sc, roll_sc, cam_geometry,
                           v_sc=None):
        raise NotImplementedError("focal to celestial")

    @staticmethod
    def pixel_to_celestial(col, row, ra_sc, dec_sc, roll_sc,
                           cam_geometry, ccd_geometries, v_sc=None):
        raise NotImplementedError("pixel to celestial")
    
    @staticmethod
    def get_eq_sc_matrix(ra, dec, roll):
        """Matrix to transform from celestial equatorial to spacecraft"""
        from math import cos, sin
        ra_r, dec_r, roll_r = map(math.radians, (ra, dec, roll))
        return np.array(
            [[-cos(roll_r) * sin(dec_r) * cos(ra_r) + sin(roll_r) * sin(ra_r),
              -cos(roll_r) * sin(dec_r) * sin(ra_r) - sin(roll_r) * cos(ra_r),
              cos(roll_r) * cos(dec_r)],
             [sin(roll_r) * sin(dec_r) * cos(ra_r) + cos(roll_r) * sin(ra_r),
              sin(roll_r) * sin(dec_r) * sin(ra_r) - cos(roll_r) * cos(ra_r),
              -sin(roll_r) * cos(dec_r)],
             [cos(dec_r) * cos(ra_r),
              cos(dec_r) * sin(ra_r),
              sin(dec_r)]])

    @staticmethod
    def get_sc_cam_matrix(cam_angle):
        """
        Matrix to transform from spacecraft to camera coordinates.

        If cam_angle is a scalar, use the Levine1 model, in which the alpha
        and gamma angles are 0 and 180 degrees, respectively.

        If cam_angle is a tuple, use the Levine2 model, in which the angles
        are as specified.
        """
        if isinstance(cam_angle, tuple) or isinstance(cam_angle, list):
            from math import cos, sin
            a, b, g = map(math.radians,
                          (cam_angle[0], cam_angle[1], cam_angle[2]))
            return np.array(
                [[cos(g) * cos(b) * cos(a) - sin(g) * sin(a),
                  cos(g) * cos(b) * sin(a) + sin(g) * cos(a),
                  -cos(g) * sin(b)],
                 [-sin(g) * cos(b) * cos(a) - cos(g) * sin(a),
                  -sin(g) * cos(b) * sin(a) + cos(g) * cos(a),
                  sin(g) * sin(b)],
                 [sin(b) * cos(a), sin(b) * sin(a), cos(b)]])
        t = math.radians(cam_angle)
        return np.array([[0, -1, 0],
                         [math.cos(t), 0, -math.sin(t)],
                         [math.sin(t), 0, math.cos(t)]])


class AmesModel(object):
    """
    Use the Ames geometry and focal models to convert between coordinates.

    Details in the ra-dec-2-pix-design-note.pdf document with LNG/LAT
    applied to equation (7) correction.
    """

    @staticmethod
    def get_rotation_matrix(x, y, z):
        # 2.4.1 Tait-Bryan angles
        from math import cos, sin
        x_r, y_r, z_r = map(math.radians, (x, y, z))
        return np.array(
            [[cos(y_r) * cos(z_r),
              -cos(y_r) * sin(z_r),
              sin(y_r)],
             [cos(x_r) * sin(z_r) + sin(x_r) * sin(y_r) * cos(z_r),
              cos(x_r) * cos(z_r) - sin(x_r) * sin(y_r) * sin(z_r),
              -sin(x_r) * cos(y_r)],
             [sin(x_r) * sin(z_r) - cos(x_r) * sin(y_r) * cos(z_r),
              sin(x_r) * cos(z_r) + cos(x_r) * sin(y_r) * sin(z_r),
              cos(x_r) * cos(y_r)]])
    
    @staticmethod
    def get_rotation_matrix_inverse(x, y, z):
        # 2.4.1 Tait-Bryan angles
        # the inverse is the same as the transpose
        from math import cos, sin
        x_r, y_r, z_r = map(math.radians(x, y, z))
        return np.array(
            [[cos(y_r) * cos(z_r),
              cos(x_r) * sin(z_r) + sin(x_r) * sin(y_r) * cos(z_r),
              sin(x_r) * sin(z_r) - cos(x_r) * sin(y_r) * cos(z_r)],
             [-cos(y_r) * sin(z_r),
              cos(x_r) * cos(z_r) - sin(x_r) * sin(y_r) * sin(z_r),
              sin(x_r) * cos(z_r) + cos(x_r) * sin(y_r) * sin(z_r)],
             [sin(y_r),
              -sin(x_r) * cos(y_r),
              cos(x_r) * cos(y_r)]])

    @staticmethod
    def spacecraft_to_camera_pointing(ra_sc, dec_sc, roll_sc, camera):
        """
        Get the (ra,dec,roll) of a camera given the spacecraft (ra,dec,roll).
        """
        raise NotImplementedError("spacecraft_to_camera_pointing not impl")

    @staticmethod
    def celestial_to_pixel(ra, dec, ra_sc, dec_sc, roll_sc,
                           cam_geometry, ccd_geometries, v_sc=None):
        """
        Transform celestial coordinates (ra,dec) in degrees to CCD coordinates
        (col,row) in pixels on the indicated camera with spacecraft oriented
        at (ra_sc,dec_sc,roll_sc) in degrees.

        If spacecraft velocity v_sc is specified, then apply DVA.  Spacecraft
        velocity is a tuple (x,y,z) of the velocity components, in km/s.

        The steps in comments below correspond to section 3 of ra-dec-2-pix.
        """

        # FIXME: this implementation is not complete

        # do everything in radians
        ra_r = np.radians(ra)
        dec_r = np.radians(dec)
        # 3.1 apply velocity aberration
        if v_sc is not None:
            xyz = spherical_to_cartesian(ra_r, dec_r)
            # FIXME: normalize xyz
            xyz = AmesModel.aberrate(xyz, v_sc)
            ra_r, dec_r = cartesian_to_spherical(xyz[0], xyz[1], xyz[2])
        # 3.2 convert ra,dec to unit vector in spacecraft equatorial
        #     u = [sin(dec), -cos(dec)*sin(ra), cos(dec)*cos(ra)]
        u = np.asarray([np.sin(dec_r), np.sin(ra_r), np.cos(ra_r)])
        u[1:] *= np.cos(dec_r)
        u[1:2] *= -1
        # 3.3 apply pointing rotation matrix
        m = AmesModel.get_rotation_matrix_inverse(ra_sc, dec_sc, roll_sc)
        x_sc = m * u
        # 3.4 apply CCD rotation matrices
        ra_ccd, dec_ccd, roll_ccd = spacecraft_to_ccd(ra_r, dec_r, ccdnum)
        m = AmesModel.get_rotation_matrix_inverse(ra_ccd, dec_ccd, roll_ccd)
        x_ccd = m * x_sc
        # 3.5 convert to lat/lon on each CCD
        lat_ccd = math.asin(-x_ccd[0])
        lon_ccd = math.atan2(-x_ccd[1], x_ccd[2])
        # 3.6 convert to standard coordinates
        d = sin(0) * sin(lat_ccd) + cos(0) * cos(lat_ccd) * cos(lon_ccd)
        nu = (cos(0) * sin(lat_ccd) - sin(0) * cos(lat_ccd) * cos(lon_ccd)) / d
        eta = cos(lat_ccd) * sin(lon_ccd) / d
        # 3.7 convert to pixel units
        rho = np.sqrt(nu * nu + eta * eta)
        psi = atan(nu / eta)
        plate_scale_poly = AmesModel.get_plate_scale_poly(ccdnum)
        r = plate_scale_poly(rho)
        x = r * cos(psi)
        y = r * sin(psi)
        # 3.8 apply pixel offset and inversion
        xy_to_rowcol = AmesModel.get_xy_to_rowcol(ccdnum)
        row, col = xy_to_rowcol(x, y)
        return col, row, ccd


class KephartModel(object):

    class PSP(object):
        def __init__(self, psp_dict):
            """Create a plate scale polynomial object from a dictionary with:
            order, originx, offsetx, scalex, coeffs[], max_domain, xindex, type
            """
            self.order = int(float(psp_dict.get('order', 6)))
            self.originx = float(psp_dict.get('originx', 0.0))
            self.offsetx = float(psp_dict.get('offsetx', 0.0))
            self.scalex = float(psp_dict.get('scalex', 0.0))
            coeffs = psp_dict.get('coeffs', {'c0': 0, 'c1': 0, 'c2': 0,
                                             'c3': 0, 'c4': 0, 'c5': 0,
                                             'c6': 0})
            self.coeffs = [float(coeffs[y]) for y in sorted(coeffs)]
            self.max_domain = float(psp_dict.get('max_domain', -1.0))
            self.xindex = float(psp_dict.get('xindex', -1.0))
            self.type_ = psp_dict.get('type', 'standard')
            if self.order != len(self.coeffs) - 1:
                raise ValueError("Order mismatch in PSP definition: %s != %s" %
                                 (self.order, len(self.coeffs) - 1))

    class CCD(object):
        def __init__(self, ccdnum, x, y, z, psp):
            """x, y, z are xAngleDegrees, yAngleDegrees, and zAngleDegrees"""
            self.number = ccdnum
            self.x = math.radians(x)
            self.y = math.radians(y)
            self.z = math.radians(z)
            if isinstance(psp, KephartModel.PSP):
                self.psp = psp
            else:
                self.psp = KephartModel.PSP(psp)

    @staticmethod
    def spacecraft_to_camera_pointing(ra_sc, dec_sc, roll_sc, camera):
        """
        Get the (ra,dec,roll) of a camera given the spacecraft (ra,dec,roll).
        """
        geom = camera.get_ccd_geometries()
        ccds = []
        for i in range(1, 5):
            ccds.append(KephartModel.CCD(
                i, geom[i]['x_angle'], geom[i]['y_angle'], geom[i]['z_angle'],
                {})) # the PSP does not matter for pointing
        # unlike quaternions everywhere else, the inst_quant is x,y,z,w
        inst_quat = KephartTransform.rdr_to_quat((ra_sc, dec_sc, roll_sc))
        # get the instrument matrix from the instrument quaternion
        inst_mat, inst_quat = KephartModel.instrument_pointing(inst_quat)
#        inst_mat = NguyenTransform.rdr_to_mat((ra_sc, dec_sc, roll_sc))
        # finally, get the camera pointing
        roll, dec, ra = KephartModel.camera_pointing(ccds, inst_mat)
        # return is ra,dec,roll in decimal degrees
        return map(math.degrees, (ra, dec, roll))

    @staticmethod
    def celestial_to_pixel(ra, dec, ra_sc, dec_sc, roll_sc,
                           cam_geometry, ccd_geometries, v_sc=None):
        """
        Transform celestial coordinates (ra,dec) in degrees to CCD coordinates
        (col,row) in pixels on the indicated camera with spacecraft oriented
        at (ra_sc,dec_sc,roll_sc) in degrees.

        If spacecraft velocity v_sc is specified, then apply DVA.  Spacecraft
        velocity is a tuple (x,y,z) of the velocity components, in km/s.

        Returns fractional pixel coordinates.

        This is the Miranda Kephart implementation of the Ames model.
        """
        # ensure that the inputs are arrays
        ra = to_array(ra)
        dec = to_array(dec)
        uvec = []
        for i in range(len(ra)):
            uvec.append(KephartModel.radec2xyz((ra[i], dec[i])))
        quat = KephartTransform.rdr_to_quat((ra_sc, dec_sc, roll_sc))
        # FIXME: these should default to None not 0
        col = np.zeros(shape=(len(uvec),), dtype=float)
        row = np.zeros(shape=(len(uvec),), dtype=float)
        ccd_id = np.zeros(shape=(len(uvec),), dtype=int)
        if len(uvec) == 0:
            return col, row, ccd_id
        # the kephart implementation does the calculations assuming a specific
        # CCD geometry, then eliminates results that do not lie on that CCD.
        # so we have to do the calculations for each ra,dec 4 times instead of
        # just once.
        for ccdnum in range(1, 5):
            x_angle = ccd_geometries[ccdnum]['x_angle']
            y_angle = ccd_geometries[ccdnum]['y_angle']
            z_angle = ccd_geometries[ccdnum]['z_angle']
            psp = KephartModel.PSP(ccd_geometries[ccdnum]['plate_scale_poly'])
            ccd = KephartModel.CCD(ccdnum, x_angle, y_angle, z_angle, psp)
            xy, rowcol, ccd_uvec = KephartModel.uvec_to_rowcol(uvec, quat, ccd)
            for i in range(len(rowcol)):
                if KephartModel.in_ccd(rowcol[i][1], rowcol[i][0], ccdnum):
                    # the kephart model includes virtual pixels, so we have to
                    # remove them here - tsig does not count them at this point
                    col[i] = rowcol[i][1] - NUM_LEADING_COLS
                    row[i] = rowcol[i][0]
                    ccd_id[i] = ccdnum
        return col, row, ccd_id

    @staticmethod
    def radec2xyz(radec):
        """
        Translate RA and Dec, measured in degrees, into a unit vector.  In this
        coordinate system the x-axis points at (RA 0, Dec 0) and the z axis
        points at Dec 90.
        """
        radec = np.asarray(radec)
        theta, phi = radec * DEG_TO_RAD
        xyz = np.asarray([np.cos(theta), np.sin(theta), np.sin(phi)])
        xyz[:2] *= np.cos(phi)
        return xyz

    @staticmethod
    def get_xy_to_rowcol(ccdnum):
        """
        Returns a transformation function for each CCD that applies the pixel
        offsets for that CCD.  Convert distance from the fiducial point of the
        CCD (in the focal plane of the CCD) to pixel (row,col), as counted from
        (0, 0) of the CCD.

        Due to the definition of the x axis (different in CCDs 1 and 4 than in
        CCDs 2 and 3) and the CCD angles (CCDs 3 and 4 are upside-down in the
        imaging plane), all stars which fall on a given CCD will have negative
        x values wrt that CCD.  So to obtain the raw value, we have to add the
        negative value to the 'top' of the CCD (nominally 2058, the top buffer
        row), which produces a value between 0 and 2058.  Similarly, the y
        value for stars on CCDs 2 and 4 will always be negative.

        From radec2rowcol.py by Miranda Kephart
        """
        if ccdnum in [1, 3]:
            col_offset = NUM_SCIENCE_COLS + NUM_LEADING_COLS
        elif ccdnum in [2, 4]:
            col_offset = NUM_LEADING_COLS
        else:
            raise ValueError("bad ccdnum %s: must be 1-4, inclusive" % ccdnum)
        row_offset = NUM_SCIENCE_ROWS + NUM_BUFFER_ROWS
        def ccd_xy_to_rowcol(xy):
            rowcol = np.copy(xy)
            rowcol[:, 1] *= -1
            rowcol[:, 0] += row_offset
            rowcol[:, 1] += col_offset
            return rowcol
        return ccd_xy_to_rowcol

    @staticmethod
    def polyval(x, c):
        """
        Speed up polyval computation.  Return the polynomical with coeffecients
        c evaluated at x.

        From radec2rowcol.py by Miranda Kephart
        """
        c0 = np.full_like(x, c[-1])
        for i in xrange(2, len(c) + 1):
            c0 *= x
            c0 += c[-i]
        return c0

    @staticmethod
    def get_psp_func(psp):
        """
        Returns a function that will apply the specified plate scale
        polynomial.

        Input x values should be the rho value, in arcseconds.

        Assumes that the order is equal to len(coeffs) - 1

        From radec2rowcol.py by Miranda Kephart
        """
        order = psp.order
        coeffs = [psp.coeffs[i] for i in xrange(order + 1)]
        originx = psp.originx
        offsetx = psp.offsetx
        scalex = psp.scalex
        def poly(x):
            _x = np.copy(x)
            _x -= originx
            _x *= scalex
            _x += offsetx
            return KephartModel.polyval(_x, coeffs)
        return poly

    @staticmethod
    def aberrate(xyz, velocity):
        """
        Given an input unit vector and velocity, return the aberrated unit
        vector.  Velocity is in km/s.

        From radec2rowcol.py by Miranda Kephart
        """
        xyz = np.asarray(xyz, dtype=np.longdouble)
        velocity = np.asarray(velocity, dtype=np.longdouble) / SPEED_OF_LIGHT_KPS
        sdotv = np.dot(xyz, velocity)
        sdots = np.sum(xyz**2, axis=1)
        vel_perp = (xyz * sdotv[:, None] / sdots[:, None] - velocity) * -1
        xyz += vel_perp
        norm = np.linalg.norm(xyz, axis=1)
        return (xyz / norm[:, None]).astype(np.float64)

    @staticmethod
    def rotate_to_ccd_frame(uvec, quat, ccd):
        """
        Given an array of unit vectors to stars in the J2000 frame, return the
        unit vectors in the frame of the CCD, using the spacecraft quaternion
        and the CCD angles relative to the spacecraft.

        This rotation is equivalent to rotating the unit vectors in the
        instrument frame then into the CCD frame (since the CCD angles are
        given relative to the instrument).

        uvec is a Nx3 array
        quat is a unit quaternion as (w, x, y, z)
        """
        import spiceypy as spice
        ccdquat = spice.m2q(spice.eul2m(ccd.z, ccd.y, ccd.x, 3, 2, 1).T)
        rotmat = spice.q2m(spice.qxq(quat, ccdquat))
        return np.dot(uvec, rotmat)

    @staticmethod
    def ccd_uvec_to_xy(ccd_uvec, ccd, ccd_psp_func=None, rad2as=None):
        """
        Given an array of unit vectors to stars, calculate the corresponding
        x,y coordinates in the plane of the CCD.

        Returns an Nx2 array of the x and y pixel distances.  These are the
        distances from the fiducial point of the CCD, not the row and column.

        Based on the ra-dec-2-pix-design-note.pdf

        To obtain the gnomic projection of the stars on the CCD, we first
        convert to a latitude and longitude:

          lat = asin(x)
          lng = atan2(y, z)

        then to the gnomic projection onto the tangent plane at (lat0, lng0):

          eta = (cos(lat0)*sin(lat) - sin(lat0)*cost(lat)*cost(lng-lng0)) /
                  (sin(lat0)*sn(lat) + cos(lat0)*cos(lat)*cos(lng-lng0))
           xi = (cos(lat)*sin(lng-lng0)) /
                  (sin(lat0)*sin(lat) + cos(lat0))*cos(lat)*cos(lng-lng0))

        However, in practice we set lat0=lon0=0 and account for the offset
        later on using the plate_scale_poly transformation.  Therefore, the
        gnomic projection simplifies to:

          eta = sin(lat)/(cos(lat)*cos(lng))
              = sin(asin(x))/cos(asin(x))*cos(atan2(y/z)))
              = x/(sqrt(1-x**2)*(z/sqrt(y**2+z**2)))
              = x/z
                      since (1-x**2) = (y**2 + z**2) for a unit vector

           xi = (cos(lat)*sin(lng))/(cos(lat)*cos(lng))
              = sin(lng)/cos(lng)
              = tan(atan(y/z))
              = y/z

        Next we apply the CCD polynomial to the projected coordinates, added
        in quadrature and converted from radians to arcseconds:

          rho = math.sqrt(eta**2 + xi**2)
          pix = plate_scale_poly(rho * rad_to_arcsec)

        Finally, retrieve the x and y pixel distances:

            x = pix * eta / rho
            y = pix * xi / rho

        From radec2rowcol.py by Miranda Kephart
        """
        if rad2as is None:
            rad2as = RAD_TO_ARCSEC
        if ccd_psp_func is None:
            ccd_psp_func = KephartModel.get_psp_func(ccd.psp)
        etaxi = np.copy(ccd_uvec[:, :2])
        etaxi /= ccd_uvec[:, 2][:, None]
        rho = np.sqrt(np.sum(etaxi**2, axis=1))
        pix = np.apply_along_axis(ccd_psp_func, 0, rho * rad2as)
        etaxi *= pix[:, None]
        etaxi /= rho[:, None]
        return etaxi

    @staticmethod
    def uvec_to_rowcol(uvec, quat, ccd,
                       rotate_to_frame_func=None,
                       uvec_to_xy_func=None,
                       xy_to_rowcol_func=None):
        """
        Convert an array of unit vectors (in the J2000 frame) to a row and
        column on the CCD, using the instrument quaternion.
        """
        # this is the original kephart implementation.  in the POC version
        # this is called uvecs2rowcol.
#        ccd_uvec = KephartModel.rotate_to_ccd_frame(uvec, quat, ccd)
#        xy = KephartModel.ccd_uvec_to_xy(ccd_uvec, ccd)
#        ccd_xy_to_rowcol = KephartModel.get_xy_to_rowcol(ccd.number)
#        rowcol = ccd_xy_to_rowcol(xy)
#        return np.abs(xy), rowcol, ccd_uvec

        # this implementation is parameterized so that we can use the same
        # unit testing pattern that was used in the POC implementation, but
        # without all of the mock obfuscation.  in the POC version of the
        # kephart implementation, they use mock all over the place.
        if rotate_to_frame_func is None:
            rotate_to_frame_func = KephartModel.rotate_to_ccd_frame
        if uvec_to_xy_func is None:
            uvec_to_xy_func = KephartModel.ccd_uvec_to_xy
        if xy_to_rowcol_func is None:
            xy_to_rowcol_func = KephartModel.get_xy_to_rowcol(ccd.number)
        ccd_uvec = rotate_to_frame_func(uvec, quat, ccd)
        xy = uvec_to_xy_func(ccd_uvec, ccd)
        rowcol = xy_to_rowcol_func(xy)
        return np.abs(xy), rowcol, ccd_uvec

    @staticmethod
    def in_ccd(col, row, ccdnum):
        """See whether the indicated column,row is on the CCD"""
        if (NUM_LEADING_COLS <= col <= NUM_LEADING_COLS + NUM_SCIENCE_COLS and
            0 <= row <= NUM_SCIENCE_ROWS):
            return True
        return False

    @staticmethod
    def instrument_pointing(inst_quat):
        """
        Given an input pointing (RA, Dec, and Roll), return the rotation
        matrix and quaternion reprsenting this pointing.

        Note that the roll is defined as in the mission pointing profile,
        i.e., the rotation from the line of RA - this is not the same as an
        intrinsic rotation angle, so it must be shifted by 180 degrees.
        Similarly, the Dec must be subtracted from 90 to obtain the rotation
        around y.
        """
        # FIXME: this smells funny - comment does not match the code
        import spiceypy as spice
        return spice.invert(spice.q2m(inst_quat)), inst_quat

    @staticmethod
    def camera_pointing(ccds, inst_mat):
        """
        Given a set of CCDs, take the mean of each and use as the approximate
        camera boresite.  While we compute z, in practice what we care about
        are the RA and Dec angles; the roll is irrelevant.
        """
        import spiceypy as spice
        camx = np.mean([e.x for e in ccds])
        camy = np.mean([e.y for e in ccds])
        camz = np.mean([e.z for e in ccds])
        cam_mat = spice.eul2m(camz, camy, camx, 3, 2, 3)
        z, y, x = spice.m2eul(np.dot(cam_mat, inst_mat), 3, 2, 3)
        return [z, normalize_dec(np.pi / 2 - y), normalize_ra(x)]

    @staticmethod
    def focal_to_celestial(x, y, ra_sc, dec_sc, roll_sc, cam_geometry,
                           v_sc=None):
        raise NotImplementedError("focal to celestial")

    @staticmethod
    def pixel_to_celestial(col, row, ra_sc, dec_sc, roll_sc,
                           cam_geometry, ccd_geometries, v_sc=None):
        raise NotImplementedError("pixel to celestial")

