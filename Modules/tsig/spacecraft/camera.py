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

from tsig.util.configurable import ConfigurableObject
from tsig.util.rdr import normalize_ra_d, normalize_dec_d, normalize_roll_d
from tsig.util import to_float
from .ccd import CCD


class Camera(ConfigurableObject):
    """
    A single camera object used to point at a specific patch of sky and
    take observations.
    """
    def __init__(self, label, number=0, serial_number=0, **kwargs):
        """
        Create a camera to generate full frame or partial images of the sky.

          label  - A unique label for this camera.
        """
        super(Camera, self).__init__()
        self.label = label
        self.number = int(number)
        self.serial_number = serial_number

        # levine parameterization uses camera angles (ames uses ccd angles)
        # levine1 parameterization
        self.offset_angle = to_float(kwargs.get('offset_angle'))
        # levine2 parameterization
        self.angle_alpha = to_float(kwargs.get('angle_alpha'))
        self.angle_beta = to_float(kwargs.get('angle_beta'))
        self.angle_gamma = to_float(kwargs.get('angle_gamma'))

        self.ccd = []
        for i in range(1, 5):
            label = 'ccd_%d' % i
            ccd_dict = dict(kwargs.get(label, {}))
            ccd_dict['label'] = label
            ccd_dict['number'] = i
            ccd = CCD(**ccd_dict)
            self.ccd.append(ccd)

        # field of view is 24x24 degrees, so radius is 12.0 * sqrt(2)
        self.fov_radius = 16.970562748477143

        # focal properties
        props = kwargs.get('focal_properties', {})
        self.focal_properties = {
            'focal_length': float(props.get('focal_length', 146.985)), # mm
            'a_0': float(props.get('a_0',   1.00000140)),
            'a_2': float(props.get('a_2',   0.28174612)),
            'a_4': float(props.get('a_4',  -0.59667259)),
            'a_6': float(props.get('a_6',   9.17151267)),
            'a_8': float(props.get('a_8',  -4.36928235)),
            'b_0': float(props.get('b_0',   0.99999822)),
            'b_2': float(props.get('b_2',  -0.28159976)),
            'b_4': float(props.get('b_4',   0.82104074)),
            'b_6': float(props.get('b_6', -10.22208978)),
            'b_8': float(props.get('b_8',  22.83962334))}

        # These parts set via point_camera(...)
        self.ra = None
        self.dec = None
        self.roll = None

    def __repr__(self):
        return "%s (%s) pointing=(%.2f,%.2f,%.2f)" % (
            self.label, self.serial_number, self.ra, self.dec, self.roll)

    def point_camera(self, ra, dec, roll):
        """
        Point the camera at a certain point in the sky.

          ra   - Right ascension from the vernal equinox, in degrees
          dec  - Declination away from celestial equator, in degrees
          roll - The rotational position of the camera, in degrees

        """
        self.ra = normalize_ra_d(ra)
        self.dec = normalize_dec_d(dec)
        self.roll = normalize_roll_d(roll)

    def get_pointing(self):
        """
        Return the ra, dec, and roll (in degrees) where the camera points.
        """
        return self.ra, self.dec, self.roll

    def get_angle(self):
        if self.offset_angle is not None:
            # levine1 parameterization
            return self.offset_angle
        elif self.angle_alpha is not None:
            # levine2 parameterization
            return self.angle_alpha, self.angle_beta, self.angle_gamma
        raise Exception("no camera angle defined for this model")

    def get_geometry(self):
        g = dict()
        if self.offset_angle is not None:
            g['offset_angle'] = self.offset_angle
        else:
            g['angle_alpha'] = self.angle_alpha
            g['angle_beta'] = self.angle_beta
            g['angle_gamma'] = self.angle_gamma
        g['number'] = self.number
        g['fov_radius'] = self.fov_radius
        g.update(self.focal_properties)
        return g

    def get_ccd_geometries(self):
        g = dict()
        for i, ccd in enumerate(self.ccd):
            g[i + 1] = ccd.get_geometry()
        return g

    def get_headers(self):
        return [
            ('CAMNUM', (self.number, 'camera number (1,2,3,4)')),
            ('CAMSN', (self.serial_number, 'camera serial number')),
            ('CAMFOCUS', (self.focal_properties['focal_length'], '[mm] focal length')),
            ('CAMRA', (self.ra, '[degree] camera pointing right ascension')),
            ('CAMDEC', (self.dec, '[degree] camera pointing declination')),
            ('CAMROLL', (self.roll, '[degree] camera pointing roll angle')),
        ]

    def get_config(self):
        cfg = {
            'label': self.label,
            'serial_number': self.serial_number,
            'number': self.number,
            'fov_radius': self.fov_radius,
            'focal_properties': self.focal_properties,
            'ccd_1': self.ccd[0].get_config(),
            'ccd_2': self.ccd[1].get_config(),
            'ccd_3': self.ccd[2].get_config(),
            'ccd_4': self.ccd[3].get_config(),
        }

        if self.offset_angle is not None:
            # levine1 parameterization
            cfg['offset_angle'] = self.offset_angle
        elif self.angle_alpha is not None:
            # levine2 parameterization
            cfg['angle_alpha'] = self.angle_alpha
            cfg['angle_beta'] = self.angle_beta
            cfg['angle_gamma'] = self.angle_gamma

        return cfg
