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

import logging
logger = logging.getLogger(__name__)

from tsig.util.configurable import ConfigurableObject
from tsig.util import to_int, to_float


def get_float(x, dflt):
    x = to_float(x)
    if x is None:
        x = dflt
    return x


class CCD(ConfigurableObject):
    """
    A CCD object used to emulate what a CCD chip will do when taking a photo.
    """
    # map the port identifier with its columns
    PORT_MAP = {
        'A': (0, 512),
        'B': (512, 1024),
        'C': (1024, 1536),
        'D': (1536, 2048),
    }
    PORT_WIDTH = 512 # pixels

    DEFAULT_GAIN = 1.0
    DEFAULT_OC_NOISE = 7.0
    DEFAULT_DC_OFFSET = 0.08
    DEFAULT_FULLWELL = 215000
    DEFAULT_BIAS = 7000
    DEFAULT_BIAS_STDDEV = 0.3

    def __init__(self, label, number=0, rows=2058, cols=2048, rotation=0.0,
                 x_0=0.0, y_0=0.0, pixel_size_x=0.015, pixel_size_y=0.015,
                 gain=[DEFAULT_GAIN] * 4,
                 oc_noise=[DEFAULT_OC_NOISE] * 4,
                 dc_offset=[DEFAULT_DC_OFFSET] * 4,
                 fullwell=[DEFAULT_FULLWELL] * 4,
                 bias=[DEFAULT_BIAS] * 4,
                 bias_stddev=[DEFAULT_BIAS_STDDEV] * 4,
                 readout_time=0.02,
                 **kwargs):
        """
        Create a CCD object.

        label - A unique (within a single camera) label for this CCD.
        rows - number of pixel rows
        cols - number of pixel columns

        rotation - about the z axis
        x_0 - x origin offset, in mm
        y_0 - y origin offset, in mm
        pixel_size_x - size of each pixel, in mm
        pixel_size_y - size of each pixel, in mm

        readout_time - time to transfer to frame store, in seconds
        fullwell - saturation limit, in electrons
        gain - electrons per adu
        oc_noise - overclock noise, in electrons
        dc_offset - dark current offset, in electrons per pixel per second
        bias - prevents negative values, in ADU count
        """
        super(CCD, self).__init__()
        self.label = label
        self.number = to_int(number)
        self.rows = to_int(rows)
        self.cols = to_int(cols)

        # levine parameters
        self.rotation = to_float(rotation)
        self.x_0 = to_float(x_0)
        self.y_0 = to_float(y_0)
        self.pixel_size_x = to_float(pixel_size_x)
        self.pixel_size_y = to_float(pixel_size_y)

        # ames parameters
        self.x_angle = to_float(kwargs.get('x_angle'))
        self.y_angle = to_float(kwargs.get('y_angle'))
        self.z_angle = to_float(kwargs.get('z_angle'))
        self.psp = kwargs.get('plate_scale_poly', {})

        # generic parameters
        self.readout_time = to_float(readout_time)
        self.sat_limit_a = get_float(fullwell[0], CCD.DEFAULT_FULLWELL)
        self.sat_limit_b = get_float(fullwell[1], CCD.DEFAULT_FULLWELL)
        self.sat_limit_c = get_float(fullwell[2], CCD.DEFAULT_FULLWELL)
        self.sat_limit_d = get_float(fullwell[3], CCD.DEFAULT_FULLWELL)
        self.gain_a = get_float(gain[0], CCD.DEFAULT_GAIN)
        self.gain_b = get_float(gain[1], CCD.DEFAULT_GAIN)
        self.gain_c = get_float(gain[2], CCD.DEFAULT_GAIN)
        self.gain_d = get_float(gain[3], CCD.DEFAULT_GAIN)
        self.oc_noise_a = get_float(oc_noise[0], CCD.DEFAULT_OC_NOISE)
        self.oc_noise_b = get_float(oc_noise[1], CCD.DEFAULT_OC_NOISE)
        self.oc_noise_c = get_float(oc_noise[2], CCD.DEFAULT_OC_NOISE)
        self.oc_noise_d = get_float(oc_noise[3], CCD.DEFAULT_OC_NOISE)
        self.dc_offset_a = get_float(dc_offset[0], CCD.DEFAULT_DC_OFFSET)
        self.dc_offset_b = get_float(dc_offset[1], CCD.DEFAULT_DC_OFFSET)
        self.dc_offset_c = get_float(dc_offset[2], CCD.DEFAULT_DC_OFFSET)
        self.dc_offset_d = get_float(dc_offset[3], CCD.DEFAULT_DC_OFFSET)
        self.bias_a = get_float(bias[0], CCD.DEFAULT_BIAS)
        self.bias_b = get_float(bias[1], CCD.DEFAULT_BIAS)
        self.bias_c = get_float(bias[2], CCD.DEFAULT_BIAS)
        self.bias_d = get_float(bias[3], CCD.DEFAULT_BIAS)
        self.bias_stddev_a = get_float(bias_stddev[0], CCD.DEFAULT_BIAS_STDDEV)
        self.bias_stddev_b = get_float(bias_stddev[1], CCD.DEFAULT_BIAS_STDDEV)
        self.bias_stddev_c = get_float(bias_stddev[2], CCD.DEFAULT_BIAS_STDDEV)
        self.bias_stddev_d = get_float(bias_stddev[3], CCD.DEFAULT_BIAS_STDDEV)

    def __repr__(self):
        return self.label

    def get_geometry(self):
        return {
            'rows': self.rows,
            'cols': self.cols,
            'rotation': self.rotation,
            'x_0': self.x_0,
            'y_0': self.y_0,
            'pixel_size_x': self.pixel_size_x,
            'pixel_size_y': self.pixel_size_y,

            'x_angle': self.x_angle,
            'y_angle': self.y_angle,
            'z_angle': self.z_angle,
            'plate_scale_poly': self.psp,
        }

    def get_headers(self):
        
        # HKBTMPC1 - ccd 1 board temp
        # HKBTMPC2 - ccd 2 board temp
        # HKBTMPC3 - ccd 3 board temp
        # HKBTMPC4 - ccd 4 board temp
        # HKDRVTMP - driver temperature
        # HKINTTMP - interface temperature
        # HKPTS01,02,03,04,05,06,07,08,09,10,11,12
        # HKALCUC1,C2,C3,C4 - ACLU sensor ccd N

        return [
            ('CCDNUM',   (self.number, 'CCD number (1,2,3,4)')),
            ('CCDROWS',  (self.rows, 'CCD rows')),
            ('CCDCOLS',  (self.cols, 'CCD columns')),
            ('CCDXOFF',  (self.x_0, '[mm] x offset from camera boresight')),
            ('CCDYOFF',  (self.x_0, '[mm] y offset from camera boresight')),
            ('CCDPIXW',  (self.pixel_size_x, '[mm] pixel width')),
            ('CCDPIXH',  (self.pixel_size_y, '[mm] pixel height')),
            ('CCDROT',
             (self.rotation, '[degree] rotation around camera boresight')),
            ('CCDREADT',
             (self.readout_time, '[s] time to transfer data to frame store')),
            ('CCDSATA',  (self.sat_limit_a, '[e-] port A saturation limit')),
            ('CCDSATB',  (self.sat_limit_b, '[e-] port B saturation limit')),
            ('CCDSATC',  (self.sat_limit_c, '[e-] port C saturation limit')),
            ('CCDSATD',  (self.sat_limit_d, '[e-] port D saturation limit')),
            ('CCDGAINA', (self.gain_a, '[e-/ADU] port A effective gain')),
            ('CCDGAINB', (self.gain_b, '[e-/ADU] port B effective gain')),
            ('CCDGAINC', (self.gain_c, '[e-/ADU] port C effective gain')),
            ('CCDGAIND', (self.gain_d, '[e-/ADU] port D effective gain')),
            ('CCDOCNA',
             (self.oc_noise_a, '[e-] port A mean value of overclock noise')),
            ('CCDOCNB',
             (self.oc_noise_b, '[e-] port B mean value of overclock noise')),
            ('CCDOCNC',
             (self.oc_noise_c, '[e-] port C mean value of overclock noise')),
            ('CCDOCND',
             (self.oc_noise_d, '[e-] port D mean value of overclock noise')),
            ('CCDDCOA',
             (self.dc_offset_a, '[e-/pixel/s] port A dark current offset')),
            ('CCDDCOB',
             (self.dc_offset_b, '[e-/pixel/s] port B dark current offset')),
            ('CCDDCOC',
             (self.dc_offset_c, '[e-/pixel/s] port C dark current offset')),
            ('CCDDCOD',
             (self.dc_offset_d, '[e-/pixel/s] port D dark current offset')),
            ('CCDBIASA', (self.bias_a, '[ADU] port A bias mean')),
            ('CCDBIASB', (self.bias_b, '[ADU] port B bias mean')),
            ('CCDBIASC', (self.bias_c, '[ADU] port C bias mean')),
            ('CCDBIASD', (self.bias_d, '[ADU] port D bias mean')),
            ('CCDBDEVA',
             (self.bias_stddev_a, '[ADU] port A bias standard deviation')),
            ('CCDBDEVB',
             (self.bias_stddev_b, '[ADU] port B bias standard deviation')),
            ('CCDBDEVC',
             (self.bias_stddev_c, '[ADU] port C bias standard deviation')),
            ('CCDBDEVD',
             (self.bias_stddev_d, '[ADU] port D bias standard deviation')),
        ]

    def get_config(self):
        return {
            'label': self.label,
            'number': self.number,
            'rows': self.rows,
            'cols': self.cols,
            'rotation': self.rotation,
            'x_0': self.x_0,
            'y_0': self.y_0,
            'pixel_size_x': self.pixel_size_x,
            'pixel_size_y': self.pixel_size_y,
            'readout_time': self.readout_time,
            'sat_limit_a': self.sat_limit_a,
            'sat_limit_b': self.sat_limit_b,
            'sat_limit_c': self.sat_limit_c,
            'sat_limit_d': self.sat_limit_d,
            'gain_a': self.gain_a,
            'gain_b': self.gain_b,
            'gain_c': self.gain_c,
            'gain_d': self.gain_d,
            'oc_noise_a': self.oc_noise_a,
            'oc_noise_b': self.oc_noise_b,
            'oc_noise_c': self.oc_noise_c,
            'oc_noise_d': self.oc_noise_d,
            'dc_offset_a': self.dc_offset_a,
            'dc_offset_b': self.dc_offset_b,
            'dc_offset_c': self.dc_offset_c,
            'dc_offset_d': self.dc_offset_d,
            'bias_a': self.bias_a,
            'bias_b': self.bias_b,
            'bias_c': self.bias_c,
            'bias_d': self.bias_d,
            'bias_stddev_a': self.bias_stddev_a,
            'bias_stddev_b': self.bias_stddev_b,
            'bias_stddev_c': self.bias_stddev_c,
            'bias_stddev_d': self.bias_stddev_d}
