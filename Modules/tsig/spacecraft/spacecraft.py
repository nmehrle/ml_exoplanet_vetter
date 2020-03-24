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

import os
import logging
logger = logging.getLogger(__name__)

import configobj
import numpy as np

from tsig.util.configurable import ConfigurableObject, Configuration
from tsig.util.rdr import normalize_ra_d, normalize_dec_d, normalize_roll_d

from .geometry import LevineModel, KephartModel, AmesModel
from .quaternion import perturb_angles
from .camera import Camera


class Spacecraft(ConfigurableObject):
    """
    The spacecraft contains four cameras with pre-defined geometry.

    The spacecraft has a position and orientation.  The position (and velocity)
    matter only for differential velocity effects; they are not used in the
    coordinate transformations from starfield to CCD pixel.  The orientation
    is specified by the RA, Dec, and roll angle.

    A typical mission will capture data from a single sector for a two week
    period.  So for two weeks the RA, Dec, and roll angle will be constant
    as the spacecraft makes two orbits around earth.

    Camera and CCD geometry and properties are specified in the geometry
    configuration file.

    The spacecraft object contains parameters from multiple geometry models.
    Some of these parameters are common to all models, others are unique to
    a specific model.
    """
    MODELS = {
        'levine': LevineModel,
        'kephart': KephartModel,
        'ames': AmesModel,
    }

    LEVINE_PARAMETERS = os.path.join(
        os.path.dirname(__file__), 'levine-parameters.cfg')
    AMES_PARAMETERS = os.path.join(
        os.path.dirname(__file__), 'ames-parameters.cfg')

    def __init__(self, **kwargs):
        """
        Create each camera using default geometry, override with any
        customizations from the configuration.
        """
        super(Spacecraft, self).__init__()

        # figure out which geometry model to use
        model = kwargs.pop('geometry_model', 'levine').lower()
        if model == 'ames' or model == 'kephart':
            filename = Spacecraft.AMES_PARAMETERS
        elif model == 'levine':
            filename = Spacecraft.LEVINE_PARAMETERS
        else:
            raise ValueError("Unrecognized geometry model '%s'" % model)
        self.geometry_model = model
        default_param = Configuration(filename)

        # load any user-specified parameters from file
        user_param = None
        filename = kwargs.pop('filename', None)
        if filename is not None:
            user_param = Configuration(filename)

        # load any model identifiers
        self.data_set_id = kwargs.pop('data_set_id', None)
        self.read_noise_model_id = kwargs.pop('read_noise_model_id', None)
        self.linearity_model_id = kwargs.pop('linearity_model_id', None)
        self.ccd_layout_model_id = kwargs.pop('ccd_layout_model_id', None)
        self.gain_model_id = kwargs.pop('gain_model_id', None)
        self.ccd_electronics_model_id = kwargs.pop('ccd_electronics_model_id', None)
        self.geometry_model_id = kwargs.pop('geometry_model_id', None)

        # configure the cameras
        self.camera = []
        for i in range(1, 5):
            label = 'camera_%d' % i
            cam_dict = configobj.ConfigObj()
            cam_dict['label'] = label
            cam_dict['number'] = i
            # default to parameters appropriate to the model
            cam_dict.merge(default_param.get(label, {}))
            # override with parameters from user file
            if user_param:
                cam_dict.merge(user_param.get(label, {}))
            # override with any specified by the configuration
            cam_dict.update(kwargs.get(label, {}))
            cam = Camera(**cam_dict)
            self.camera.append(cam)

        # these are set when the spacecraft is positioned and oriented
        self.ra = 0.0
        self.dec = 0.0
        self.roll = 0.0
        self.position = None
        self.velocity = None
        self.jitter_data = None

    def __repr__(self):
        return "pointing=(%.2f,%.2f,%.2f) pos=(%.2f,%.2f,%.2f) vel=(%.2f,%.2f,%.2f)" % (
            self.ra, self.dec, self.roll,
            self.x, self.y, self.z, self.r, self.s, self.t)

    def get_model(self):
        model = Spacecraft.MODELS.get(self.geometry_model)
        if model is not None:
            return model
        raise ValueError("unsupported geometry model '%s'" %
                         self.geometry_model)

    def set_position(self, x, y, z):
        """Position the spacecraft in orbit"""
        self.position = x, y, z

    def set_velocity(self, vx, vy, vz):
        """Set the spacecraft velocity in space"""
        self.velocity = vx, vy, vz

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def set_pointing(self, ra, dec, roll=0.0):
        """
        Aim the spacecraft at a point in the sky.

          ra   - Right ascension from the vernal equinox, in degrees
          dec  - Declination away from celestial equator, in degrees
          roll - The rotational angle of the spacecraft, in degrees

        """
        logger.debug("point spacecraft to ra=%.2f dec=%.2f roll=%.2f" %
                     (ra, dec, roll))
        self.ra = normalize_ra_d(ra)
        self.dec = normalize_dec_d(dec)
        self.roll = normalize_roll_d(roll)
        model = self.get_model()

        for i in range(4):
            ra_cam, dec_cam, roll_cam = model.spacecraft_to_camera_pointing(
                ra, dec, roll, self.camera[i])
            logger.debug("point camera %s to ra=%.2f dec=%.2f roll=%.2f" %
                         ((i + 1), ra_cam, dec_cam, roll_cam))
            self.camera[i].point_camera(ra_cam, dec_cam, roll_cam)

    def get_pointing(self, jitter=None):
        ra = self.ra
        dec = self.dec
        roll = self.roll
        logger.debug("pointing: %.4f %.4f %.4f" % (ra, dec, roll))
        if jitter is not None:
            ra, dec, roll = perturb_angles(ra, dec, roll, jitter)
            logger.debug("jittered pointing: %.4f %.4f %.4f" % (ra, dec, roll))
        return ra, dec, roll

    def get_cam_geometry(self, cam_id):
        if cam_id not in [1, 2, 3, 4]:
            raise ValueError("Bad camera ID %s: must be 1,2,3, or 4" % cam_id)
        return self.camera[cam_id - 1].get_geometry()

    def get_ccd_geometries(self, cam_id):
        if cam_id not in [1, 2, 3, 4]:
            raise ValueError("Bad camera ID %s: must be 1,2,3, or 4" % cam_id)
        return self.camera[cam_id - 1].get_ccd_geometries()

    def get_jitter_quaternion(self, tjd, num_sample=10, method='random',
                              filename=None):
        """Return a quaternion of jitter deltas at the indicated time.
        The time is TESS julian date.

        Expects the jitter data to be in the format:
        t, w, x, y, z
        """
        if self.jitter_data is None:
            if filename is None:
                filename = os.path.join(
                    os.path.dirname(__file__), 'jitter.csv')
            try:
                self.jitter_data = []
                lineno = 0
                with open(filename) as f:
                    for line in f:
                        lineno += 1
                        if line.startswith('#'):
                            continue
                        parts = line.split(',')
                        if len(parts) == 5:
                            self.jitter_data.append([float(x) for x in parts])
                        else:
                            logger.debug('skip line %s of %s' % (lineno, filename))
            except IOError, e:
                logger.error("cannot read jitter data from %s: %s" %
                             (filename, e))
                self.jitter_data = None
                return (0, 0, 0, 0)

        if method == 'random':
            # grab a random displacement from the jitter simulation
            idx = int(np.random.uniform(0, len(self.jitter_data) - 1))
            w = self.jitter_data[idx][1]
            x = self.jitter_data[idx][2]
            y = self.jitter_data[idx][3]
            z = self.jitter_data[idx][4]
        elif method == 'average':
            # grab n samples from the jitter simulation then average them
            if not hasattr(self, 'jitter_index'):
                self.jitter_index = 0
            w = x = y = z = 0.0
            idx = self.jitter_index
            for i in range(num_sample):
                w += self.jitter_data[idx][1]
                x += self.jitter_data[idx][2]
                y += self.jitter_data[idx][3]
                z += self.jitter_data[idx][4]
                idx += 1
                # wrap around so we do not exceed the jitter array
                if idx >= len(self.jitter_data):
                    idx = 0
            w /= num_sample
            x /= num_sample
            y /= num_sample
            z /= num_sample
            self.jitter_index = idx
        elif method == 'interpolate':
            # get a specific sample from the simulation, interpolate between
            raise NotImplemented("jitter method 'interpolate' not implemented")
        else:
            raise ValueError("unknown jitter method '%s'" % method)
        return w, x, y, z

    def get_headers(self):
        return [
            ('SCRA', (self.ra, '[degree] spacecraft pointing right ascension')),
            ('SCDEC', (self.dec, '[degree] spacecraft pointing declination')),
            ('SCROLL', (self.roll, '[degree] spacecraft pointing roll angle')),
            ('SCPOSX', (self.position[0], 'spacecraft position x')),
            ('SCPOSY', (self.position[1], 'spacecraft position y')),
            ('SCPOSZ', (self.position[2], 'spacecraft position z')),
            ('SCVELX', (self.velocity[0], '[km/s] spacecraft velocity x')),
            ('SCVELY', (self.velocity[1], '[km/s] spacecraft velocity y')),
            ('SCVELZ', (self.velocity[2], '[km/s] spacecraft velocity z')),
        ]

    def get_config(self):
        return {
            'geometry_model': self.geometry_model,
            'data_set_id': self.data_set_id,
            'read_noise_model_id': self.read_noise_model_id,
            'linearity_model_id': self.linearity_model_id,
            'ccd_layout_model_id': self.ccd_layout_model_id,
            'gain_model_id': self.gain_model_id,
            'ccd_electronics_model_id': self.ccd_electronics_model_id,
            'geometry_model_id': self.geometry_model_id,
            'camera_1': self.camera[0].get_config(),
            'camera_2': self.camera[1].get_config(),
            'camera_3': self.camera[2].get_config(),
            'camera_4': self.camera[3].get_config(),
        }
