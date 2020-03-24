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
The observation contains the logic for combining individual exposures and
applying effects into individual images for each CCD and camera.
"""

import configobj
import glob
import os
import math
import time
import logging
logger = logging.getLogger(__name__)

from astropy.io import fits
import numpy as np

import tsig.catalog
from .util import to_bool, to_int
from .util.configurable import ConfigurableObject
from .util.cachable import cachable, PickleCacher
from .util.timer import Timer
from .util.clock import tjd2dy
from .effects import *
from .lightcurve.lcg import LightCurveGenerator
from .psf import MatlabSource, DatabaseSource, GaussianSource


# This array defines the order in which effects will be applied
# TemperatureDependence is a characteristic for many of the effects
EFFECTS = [
    GalacticBackground,
    CelestialBackground,
    EarthMoon,
    FlatField,
    DarkCurrent,
    ShotNoise,
    CosmicRays,
    Smear,
    Saturation,
    TransferEfficiency,
    ReadoutNoise,
    ElectronsToADU,
    BiasLevel,
    FixedPatternNoise,
    Undershoot,
    LineRinging,
    BadPixels,
]


class Observation(ConfigurableObject):
    """
    The observation executes a series of exposures to generate a set of
    images.  In its simplest form, it does a set of exposures for each camera
    for a single pointing of the spacecraft with no spacecraft motion.

    1 exposure set at 2m cadence requires 60 2s exposures (2*60/2)
    1 exposure set at 30m cadence requires 900 2s exposures (30*60/2)
    """
    def __init__(self, num_exposures=60, cadence=2, camera_fov_radius=0.0,
                 apply_lightcurves=True, apply_psf=True, apply_jitter=True,
                 apply_dva=True, apply_effects=True, apply_proper_motion=True,
                 subpixel_resolution=5, num_buffer_pixels=3,
                 active_cameras=(1, 2, 3, 4),
                 active_ccds_camera_1=(1, 2, 3, 4),
                 active_ccds_camera_2=(1, 2, 3, 4),
                 active_ccds_camera_3=(1, 2, 3, 4),
                 active_ccds_camera_4=(1, 2, 3, 4),
                 cache_directory="~/.cache/tsig",
                 save_combined_ccds=False, retain_2s_images=False,
                 save_raw_images=False, apply_ccd_markup=False,
                 **kw):
        super(Observation, self).__init__()
        for line in tsig.platform_info():
            logger.info(line)
        logger.info("initializing observation")

        self.spacecraft = None
        self.mission = None

        self.exp_time = 2 # length of a single exposure, in seconds
        self.num_exposures = int(num_exposures)
        # FIXME: consider making cadence a list so we could emit images at
        # both 2m and 30m cadences
        self.cadence = int(cadence) # how often to stack exposures, in minutes
        self.cadence_s = self.cadence * 60

        # which cameras and CCDs to simulate?
        self.active_cameras = [int(x) for x in active_cameras]
        logger.info("active cameras: %s" % self.active_cameras)
        self.active_ccds = [
            [int(x) for x in active_ccds_camera_1],
            [int(x) for x in active_ccds_camera_2],
            [int(x) for x in active_ccds_camera_3],
            [int(x) for x in active_ccds_camera_4],
        ]
        for i, ccds in enumerate(self.active_ccds):
            if i + 1 in self.active_cameras:
                logger.info("active CCDs on camera %s: %s" % (i + 1, ccds))

        # how many subpixels in a pixel
        self.subpixel_resolution = int(subpixel_resolution)
        # how many CCD-pixel-sized pixels around the CCD
        self.num_buffer_pixels = int(num_buffer_pixels)
        # optional override to camera field of view radius
        self.radius = float(camera_fov_radius)
        # place marks on the CCD image to verify position and reference frame
        self.apply_ccd_markup = to_bool(apply_ccd_markup)
        # save raw image before effects are applied
        self.save_raw_images = to_bool(save_raw_images)
        # optionally combine images from all 4 CCDs into a single image
        self.save_combined_ccds = to_bool(save_combined_ccds)
        # optionally save every two-second exposure
        self.retain_2s_images = to_bool(retain_2s_images)

        self.output_directory = None
        # where to cache database query results and other items
        self.cache_directory = os.path.abspath(
            os.path.expanduser(cache_directory))

        # create the cosmic ray mitigation
        cmcfg = kw.get('Stacker', {})
        self.stacker = Stacker(**cmcfg)
        if self.stacker.action != Stacker.SUBTRACT_NONE:
            # ensure that the block size is an integral divisor of stack size
            stack_size = self.cadence_s / 2
            if stack_size % self.stacker.block_size != 0:
                msg = "Cosmic mitigation block size %s is not an integral divisor of stack size %s" % (self.stacker.block_size, stack_size)
                logger.error(msg)
                raise ValueError(msg)

        # should we apply proper motion?
        self.apply_proper_motion = to_bool(apply_proper_motion)

        # create a lightcurve generator if we are supposed to apply lightcurves
        self.apply_lightcurves = to_bool(apply_lightcurves)
        self.lcg = None
        if self.apply_lightcurves:
            lccfg = kw.get('LightCurves', {})
            self.lcg = LightCurveGenerator(**lccfg)

        # see if we are supposed to apply jitter
        self.apply_jitter = to_bool(apply_jitter)
        jitter = kw.get('Jitter', {})
        self.jitter_method = jitter.get('method', 'random')
        self.jitter_filename = jitter.get('filename', None)
        self.jitter_samples = int(jitter.get('num_samples', 10))
        if self.apply_jitter:
            logger.info("jitter using %s method" % self.jitter_method)
            if self.jitter_filename:
                logger.info("jitter data from %s" % self.jitter_filename)

        # see if we are supposed to apply differential velocity aberration
        self.apply_dva = to_bool(apply_dva)

        # create the PSF source if we are supposed to apply PSFs
        self.apply_psf = to_bool(apply_psf)
        self.psfsrc = None
        self.psf_cache = {}
        self.granularity = None
        if self.apply_psf:
            psf_type = kw.get('PSF', {}).get('source', 'DatabaseSource')
            psfcfg = kw.get('PSF', {}).get(psf_type, {})
            self.granularity = to_int(kw.get('PSF', {}).get('granularity', 20))
            self.psf_cache = kw.get('PSF', {}).get('Cache', {})
            self.psf_cache['cacher'] = PickleCacher(self.cache_directory,
                subdir=self.psf_cache.get('directory', "psf"))
            try:
                self.psfsrc = getattr(tsig.psf, psf_type)(**psfcfg)
                logger.info("PSF source: %s" % self.psfsrc.get_info())
            except AttributeError:
                logger.error("PSF source type not found: %s" % psf_type)
                raise ValueError("Unknown PSF source type %s" % psf_type)

        # create a catalog, default to TIC
        cat_type = kw.get('Catalog', {}).get('source', 'TIC')
        ccfg = kw.get('Catalog', {}).get(cat_type, {})
        if 'cache_directory' not in ccfg:
            ccfg['cache_directory'] = self.cache_directory + "/catalog"
        try:
            self.catalog = getattr(tsig.catalog, cat_type)(**ccfg)
            logger.info("TIC source: %s" % self.catalog.get_info())
        except AttributeError:
            logger.error("catalog type not found: %s" % cat_type)
            raise ValueError("Unknown catalog %s" % cat_type)

        # the target list determines which stars will be rendered
        tgtcfg = kw.get('Targets', {})
        self.target_list = TargetList(**tgtcfg)

        # see which effects should be applied
        self.apply_effects = to_bool(apply_effects)
        self.effects = []
        if self.apply_effects:
            ecfg = kw.get('Effects', {})
            for effect in EFFECTS:
                name = effect.__name__
                if name in ecfg and not to_bool(ecfg[name].pop('enable', True)):
                    logger.info("effect %s: disabled" % name)
                else:
                    logger.info("effect %s: enabled" % name)
                    cfg = ecfg.get(name, {})
                    self.effects.append(effect(**cfg))
            self.effects.append(SanityCheck())
        if self.apply_ccd_markup:
            logger.info("effect Markup: enabled")
            self.effects.append(Markup())
        # report about which effects are actually functional
        for e in self.effects:
            try:
                pixels = np.zeros(shape=(2058, 2048), dtype=np.float64)
                hdulist = CCDImage.make_hdulist(pixels, [])
                hdulist = e.apply(hdulist)
            except (NotImplementedError, ImportError) as err:
                logger.info(str(err))

    def set_outdir(self, outdir):
        self.output_directory = outdir

    def set_spacecraft(self, spacecraft):
        self.spacecraft = spacecraft

    def set_mission(self, mission):
        self.mission = mission

    def observe(self):
        """
        Make an observation.  Loop through time to take multiple exposures,
        moving the spacecraft through a series of locations at each point in
        time.  Save an image for each CCD at each time step.  Periodically
        stack those exposures into the cadence (2m or 30m) images.
        """

        logger.info("cache directory is %s" % self.cache_directory)
        # use the specified output directory...
        outdir = self.output_directory
        # 2s exposures go in a subdirectory within the output directory
        outdir_e = "%s/exposures" % outdir
        # cadence images go in a subdirectory within the output directory
        outdir_s = "%s/stacked" % outdir
        # configuration goes into a subdirectory within the output directory
        outdir_c = "%s/config" % outdir
        for d in [outdir, outdir_c, outdir_s, outdir_e]:
            try:
                os.mkdir(d)
            except os.error as e:
                pass
        logger.info("output directory is %s" % outdir)
        logger.info("output directory for 2s exposures is %s" % outdir_e)
        logger.info("output directory for cadence images is %s" % outdir_s)
        logger.info("output directory for config is %s" % outdir_c)

        # Point the spacecraft for the duration of the observation.
        logger.info("spacecraft pointing ra=%s dec=%s roll=%s" %
                    (self.mission.pointing_ra,
                     self.mission.pointing_dec,
                     self.mission.pointing_roll))
        self.spacecraft.set_pointing(
            self.mission.pointing_ra,
            self.mission.pointing_dec,
            self.mission.pointing_roll)
        model = self.spacecraft.get_model()
        for i in range(4):
            logger.info("camera %s pointing ra=%s dec=%s roll=%s" %
                        (i + 1,
                         self.spacecraft.camera[i].ra,
                         self.spacecraft.camera[i].dec,
                         self.spacecraft.camera[i].roll))
                         
        # Save the configuration
        cfg = configobj.ConfigObj(self.get_config())
        cfg.filename = "%s/parsed.cfg" % outdir_c
        cfg.write()

        # Save profiling summary to separate file
        self.profile_filename = "%s/profile.log" % outdir
        timer = Timer(self.profile_filename)

        # Query for targets
        timer.info("min/max brightness: %s/%s" %
                   (self.catalog.min_brightness, self.catalog.max_brightness))
        timer.start()
        stars = self.target_list.get_targets(
            self.catalog, self.spacecraft, self.radius, self.active_cameras)
        timer.stop()
        cnt = 0
        for cam_id in stars:
            timer.info("camera %s: %s objects" %
                       (cam_id, stars[cam_id].size()))
            cnt += stars[cam_id].size()
        timer.info("queried %s objects in %s" % (cnt, timer.elapsed_fmt()))

        # Save the resolved target list
        if self.target_list.targets:
            tgt_dict = dict()
            for i, tgt in enumerate(self.target_list.targets):
                tgt_dict["%d" % i] = tgt
            tgt = configobj.ConfigObj(tgt_dict)
            tgt.filename = "%s/resolved-targets.cfg" % outdir_c
            tgt.write()

        # Apply proper motion and assign lightcurves
        start_tjd = self.mission.start_tjd
        start_year = tjd2dy(start_tjd)
        logger.info("start time: tjd=%s year=%s" % (start_tjd, start_year))
        for (cam_i, camera) in enumerate(self.spacecraft.camera):
            cam_id = cam_i + 1
            if cam_id not in self.active_cameras:
                continue
            logger.info("proper motion and lightcurves for camera %s" % cam_id)
            if self.apply_proper_motion:
                stars[cam_id].apply_proper_motion(start_year)
            if self.apply_lightcurves:
                stars[cam_id].apply_lightcurves(self.lcg)
                stars[cam_id].save_lightcurves(outdir_c, cam_id)

        # allocate a subpixel (super-resolution) buffer on which we will render
        # this assumes that all CCDs have the same geometry
        spbuffer = SubpixelBuffer(self.spacecraft.camera[0].ccd[0].cols,
                                  self.spacecraft.camera[0].ccd[0].rows,
                                  self.subpixel_resolution,
                                  self.num_buffer_pixels)

        # count of two-second exposures
        self.e_count = 1
        first_exposure = self.e_count
        exposure_start = time.time() # track wall clock time for profiling

        timer.info("active cameras: %s" % self.active_cameras)
        logger.info("simulating %s exposures stacked every %s seconds" % (
            self.num_exposures, self.cadence_s))

        while self.e_count <= self.num_exposures:
            timer.start()

            t_s = (self.e_count - 1) * self.exp_time
            t_d = t_s / (3600.0 * 24.0)
            t_tjd = start_tjd + t_d
            logger.info("exposure=%s tjd=%s" % (self.e_count, t_tjd))

            # Move spacecraft to the correct position and velocity
            (position, velocity) = self.mission.get_spacecraft_state(t_tjd)
            self.spacecraft.set_position(*position)
            self.spacecraft.set_velocity(*velocity)
            if not self.apply_dva:
                velocity = None

            # get jitter, if requested
            jitter = None
            if self.apply_jitter:
                jitter = self.spacecraft.get_jitter_quaternion(
                    t_s, method=self.jitter_method,
                    filename=self.jitter_filename)

            # do a 2-second exposure for each CCD in each camera
            for (cam_i, camera) in enumerate(self.spacecraft.camera):
                cam_id = cam_i + 1
                if cam_id not in self.active_cameras:
                    continue
                logger.info("%s" % camera.label)
                # get the star magnitudes for the current time
                tmag = stars[cam_id].get_magnitude(t_d)
                # apply jitter quaternion to the camera pointing
                sc_ra, sc_dec, sc_roll = self.spacecraft.get_pointing(jitter)
                # transform star locations to pixel coordinates
                logger.debug("calculate %s transformations" % len(tmag))
                stars_ra = stars[cam_id].ra
                stars_dec = stars[cam_id].dec
                if self.apply_proper_motion:
                    stars_ra = stars[cam_id].projected_ra
                    stars_dec = stars[cam_id].projected_dec
                col, row, ccd_n = model.celestial_to_pixel(
                    stars_ra, stars_dec, sc_ra, sc_dec, sc_roll,
                    camera.get_geometry(), camera.get_ccd_geometries(),
                    v_sc=velocity)
                # count the stars on each ccd and report
                logger.debug("apply_electrons for %s stars" % len(col))
                cnt = [0] * 5
                for idx in range(len(col)):
                    cnt[ccd_n[idx]] += 1
                for i in range(0, 5):
                    logger.debug("ccd%d: %s" % (i, cnt[i]))
                # apply the photons for each star, by ccd
                for (ccd_i, ccd) in enumerate(camera.ccd):
                    ccd_id = ccd_i + 1
                    if ccd_id not in self.active_ccds[cam_i]:
                        continue
                    logger.debug("processing %s %s" %
                                 (camera.label, ccd.label))
                    # clear the subpixel buffer
                    spbuffer.clear()
                    # apply electrons from each star
                    for idx in range(len(col)):
                        if ccd_n[idx] == ccd_id:
                            spbuffer.apply_electrons_ccd(
                                ccd_id, int(col[idx]), int(row[idx]),
                                tmag[idx], stars[cam_id].teff[idx],
                                self.psfsrc, self.granularity, self.psf_cache)
                    # downsample from sub-pixel buffer to ccd pixels
                    pixels = spbuffer.downsample_to_ccd()
                    # create the hdulist
                    hdulist = CCDImage.make_hdulist(
                        pixels, self.get_2s_headers(),
                        self.spacecraft, camera, ccd)
                    # save the raw image
                    if self.save_raw_images:
                        CCDImage.save_image(hdulist, outdir_e, cam_id, ccd_id,
                                            self.e_count, self.exp_time, 'raw')
                    # apply any effects
                    hdulist = CCDImage.apply_effects(hdulist, self.effects)
                    # convert to 16-bit integers per pixel
                    hdulist[0].data = hdulist[0].data.astype(np.uint16)
                    # save each exposure to disk so it can be stacked later
                    CCDImage.save_image(hdulist, outdir_e, cam_id, ccd_id,
                                        self.e_count, self.exp_time)

            timer.info("exposure=%s tjd=%s in %.3f seconds (%.3fs per exp)" % (
                self.e_count, t_tjd, timer.elapsed(use_stop=False),
                (time.time() - exposure_start) / self.e_count))

            t_s += self.exp_time
            if t_s % self.cadence_s == 0 and t_s != 0:
                # it is time to stack the exposures into a cadence image
                logger.debug("stack images at exposure=%s" % self.e_count)
                for (cam_i, camera) in enumerate(self.spacecraft.camera):
                    cam_id = cam_i + 1
                    if cam_id not in self.active_cameras:
                        continue
                    composite = None
                    if self.save_combined_ccds:
                        composite = dict()
                    for (ccd_i, ccd) in enumerate(camera.ccd):
                        ccd_id = ccd_i + 1
                        if ccd_id not in self.active_ccds[cam_i]:
                            continue
                        # cadence images are 32-bits per pixel
                        pixels = np.zeros(shape=(ccd.rows, ccd.cols),
                                          dtype=np.uint32)
                        hdulist = CCDImage.make_hdulist(
                            pixels, self.get_cadence_headers(),
                            self.spacecraft, camera, ccd)
                        hdulist = self.stacker.stack_exposures(
                            hdulist, outdir_e, cam_id, ccd_id,
                            first_exposure, self.e_count)
                        CCDImage.apply_effect_headers(hdulist, self.effects)
                        CCDImage.save_image(hdulist, outdir_s, cam_id, ccd_id,
                                            self.e_count, self.cadence_s)
                        if composite is not None:
                            composite[ccd_id] = hdulist[0]
                    if composite is not None:
                        hdulist = CCDImage.create_composite(composite)
                        CCDImage.save_image(hdulist, outdir_s, cam_id, None,
                                            self.e_count, self.cadence_s)

                # clear the exposures to make way for the next image
                first_exposure = self.e_count + 1
                if not self.retain_2s_images:
                    CCDImage.delete_2s_exposures(outdir_e)

            self.e_count += 1

    def get_config(self):
        return {
            'observation': {
                'exp_time': self.exp_time,
                'cadence': self.cadence,
                'num_exposures': self.num_exposures,
                'active_cameras': self.active_cameras,
                'camera_fov_radius': self.radius,
                'apply_lightcurves': self.apply_lightcurves,
                'apply_psf': self.apply_psf,
                'apply_jitter': self.apply_jitter,
                'apply_dva': self.apply_dva,
                'apply_effects': self.apply_effects,
                'subpixel_resolution': self.subpixel_resolution,
                'num_buffer_pixels': self.num_buffer_pixels,
                'granularity': self.granularity,
                'output_directory': self.output_directory,
                'cache_directory': self.cache_directory,
                'save_combined_ccds': self.save_combined_ccds,
                'retain_2s_images': self.retain_2s_images,
                'apply_ccd_markup': self.apply_ccd_markup,
                'save_raw_images': self.save_raw_images,
            },
            'spacecraft': self.spacecraft.get_config(),
            'mission': self.mission.get_config(),
            'catalog': self.catalog.get_config(),
            'targets' : self.target_list.get_config(),
            'psf_source': self.psfsrc.get_config() if self.psfsrc else None,
            'lightcurve_generator': self.lcg.get_config() if self.lcg else None,
            'effects': [e.__class__.__name__ for e in self.effects],
            'stacker': self.stacker.get_config(),
        }

    def get_headers(self):

#        head['BJD0'] = (bjd0, '[day] base time subtracted from all BJD')
#        head['BJD'] = (bjd - bjd0, '[day] mid-exposure time - BJD0')
#        head['BJD_TDB'] = (bjd, '[day] BJD_TDB')
#        head['ANTISUN'] = (bjd_antisun - bjd0, '[day] time of antisun - BJD0')

# from the header ICD draft by chelsea and andras
# JD, JDTAI, JDTB - julian day of midexpo in some uniform time
# EXPTIME - gross exposure time
# BJD DX, BJD DY, BJD DZ - coordinates of spacecraft wrt solar system
#   barycenter.  if units are speed of light times SI day, then BJD-JD
#   correction can be retrieved by computing the scalar product of this and the
#   normal vector of the star.  J2000 coordinates are preferred.
# TIMESLICA, TIMELICB, TIMESLICC, TIMESLICD - in kepler it was TIMSLICE
# TIERRELA - uncertainty and/or precision about onboard timekeeping

# TESS QA, TESS QB, TESS QC, TESS QD - spacecraft quaternion
# RA NOM, DEC NOM, ROLL NOM, RAC(CRVAL1), DECC (CRVAL2) - nominal approximate
#   J2000 positions of the optical axes (including field rotation) and the ccd
# centers
# RA (CRVAL1) - ra for ccd center
# DECC (CRVAL2) - dec for ccd center

# pointing jitter reported by the ADCS
# position of moon and earth - moon vector wrt spacecraft?
#   earth, moon, and sun vectors?

# LILITH uses these:
# SECTOR
# CADENCE - frame number
# STARTTJD - e.g., 1209.979166
# ENDTJD - e.g., 1210.0

        return [
            ('TSIGVERS', (tsig.__version__, 'TSIG version number')),
            ('SECTOR', (self.mission.sector, 'sector')),
            ('STARTUTC', (self.mission.start_utc, '[UTC] mission start')),
            ('STARTTJD', (self.mission.start_tjd, '[tjd] mission start')),
            ('RES', (self.subpixel_resolution, 'subpixels per CCD pixel')),
            ('EXPTIME', (self.exp_time, '[s] single exposure time')),
        ]

    def get_2s_headers(self):
        return self.get_headers()

    def get_cadence_headers(self):
        headers = self.get_headers()
        return headers + [
            ('NUMEXP', (int(self.cadence_s / self.exp_time),
                        'number of exposures in this image')),
            ('CADENCE', (self.cadence_s, '[s] time between stacked images')),
            ('ECOUNTER', (self.e_count, 'number of exposures since start')),
        ]


class CCDImage(object):
    """
    Functions for dealing with CCD images.

    A full CCD image has the following characteristics:

    all_pixels        - 2136x2078
    science_pixels    - 2048x2048
    buffer_pixels     - 2048x10
    smear_pixels      - 2048x10
    virtual_pixels    - 2048x10
    underclock_pixels - 44x2078
    overclock_pixels  - 44x2078

    A 2-second image is 16-bits per pixel.
    A 2-minute or 30-minute stacked image is 32-bits per pixel.
    """

    IMAGE_WIDTH = 2136
    IMAGE_HEIGHT = 2078

    @staticmethod
    def make_hdulist(image, headers, *objs):
        """
        Produce an hdulist and add the headers from each object.  Use the
        specified image as the data for the science pixels of the image.

        The image must be 2048x2048 or 2048x2058 pixels, otherwise the
        geometry will be wrong.
        """
        if image is None:
            image = np.zeros(shape=(2058, 2048), dtype=np.uint32)
        head = fits.Header()
        for (name, value) in headers:
            head[name] = value
        for obj in objs:
            if hasattr(obj, "get_headers"):
                for (name, value) in obj.get_headers():
                    head[name] = value
        # create a full-sized image, including non-science pixels
        full_image = np.zeros(
            shape=(CCDImage.IMAGE_HEIGHT, CCDImage.IMAGE_WIDTH),
            dtype=image.dtype)
        full_image[0: image.shape[0], 44: 44 + image.shape[1]] = image
        hdu = fits.PrimaryHDU(full_image, header=head)
        hdulist = fits.HDUList([hdu])
        return hdulist

    @staticmethod
    def create_composite(image_dict):
        """
        Combine 4 CCD images into a single image.  Flip CCD 1 and 2.
        """
        # assume that all 4 CCD images have the same dimensions
        height = CCDImage.IMAGE_HEIGHT
        width = CCDImage.IMAGE_WIDTH
        total_width = 2 * width
        total_height = 2 * height
        image = np.zeros(shape=(total_height, total_width), dtype=np.uint32)
        if 1 in image_dict:
            image[height:, width: total_width] = image_dict[1].data[::-1, ::-1]
        if 2 in image_dict:
            image[height:, 0: width] = image_dict[2].data[::-1, ::-1]
        if 3 in image_dict:
            image[0: height, 0: width] = image_dict[3].data
        if 4 in image_dict:
            image[0: height, width: total_width] = image_dict[4].data
        head = fits.Header()
        # FIXME: add headers
        hdu = fits.PrimaryHDU(image, header=head)
        hdulist = fits.HDUList([hdu])
        return hdulist

    @staticmethod
    def apply_effects(hdulist, effects):
        """Take each effect in order and apply it to the FITS object"""
        for effect in effects:
            try:
                logger.debug("apply %s" % effect.__class__.__name__)
                hdulist = effect.apply(hdulist)
            except (NotImplementedError, ImportError) as err:
                logger.debug(str(err))
        return hdulist

    @staticmethod
    def apply_effect_headers(hdulist, effects):
        """
        Add a header for each effect, in the order in which the effects were
        applied. Just the effect name and version.
        """
        for i, effect in enumerate(effects):
            hdulist[0].header['EFFECT%d' % i] = (
                effect.version, '[version] %s' % effect.name)

    @staticmethod
    def get_filename(outdir, cam_id, ccd_id, frame, cadence, imgtype='full'):
        """
        The filenames for each time should be:
        
          tsig-%(frame)8d-cam%1d-ccd%1d-%(cadence)-%(imgtype).fits

        cadence: in seconds

        imgtype: raw, imagette, or full
                 raw - raw image (no effects applied)
                 tps - postage stamp (imagette)

        e.g., tsig-00000001-cam1-ccd1-120.fits
              tsig-00000001-cam1-ccd1-120-raw.fits
              tsig-00000001-cam1-ccd1-120-XXXX-tps.fits

        SPOC calls each CCD image a full-frame image, but POC refers to a
        camera image full-frame image.
        """
        if imgtype == 'imagette':
            imgtype_label = '-tps'
        elif imgtype == 'full':
            imgtype_label = ''
        elif imgtype == 'raw':
            imgtype_label = '-raw'
        elif imgtype == 'stacked':
            imgtype_label = '-stacked'
        else:
            raise TypeError("Unknown image type '%s'" % imgtype)

        # FIXME: if imagette, include XXXX identifier in filename

        parts = [outdir]
        if ccd_id is not None:
            parts.append("tsig-%08d-cam%1d-ccd%1d-%d%s.fits" % (
                frame, cam_id, ccd_id, cadence, imgtype_label))
        else:
            parts.append("tsig-%08d-cam%1d-%s%s.fits" % (
                frame, cam_id, cadence, imgtype_label))
        return '/'.join(parts)

    @staticmethod
    def save_image(hdulist, outdir, cam_id, ccd_id, frame, cadence, imgtype='full'):
        filename = CCDImage.get_filename(
            outdir, cam_id, ccd_id, frame, cadence, imgtype)
        logger.debug("save %s" % filename)
        hdulist.writeto(filename)

    @staticmethod
    def delete_2s_exposures(outdir):
        pattern = "%s/*-2.fits" % outdir
        logger.debug("delete 2s images matching %s" % pattern)
        for filename in glob.glob(pattern):
            try:
                logger.debug("delete 2s image %s" % filename)
                os.remove(filename)
            except IOError as e:
                logger.error("delete failed: %s" % e)


class Stacker(ConfigurableObject):

    # options for how to apply the cosmic mitigation
    SUBTRACT_MAX = 3
    SUBTRACT_MIN = 2
    SUBTRACT_MINMAX = 1
    SUBTRACT_NONE = 0

    ACTION_LABEL = {
        SUBTRACT_MAX: 'subtract_max',
        SUBTRACT_MIN: 'subtract_min',
        SUBTRACT_MINMAX: 'subtract_min_and_max',
        SUBTRACT_NONE: 'no_mitigation',
    }
    
    # maximum value of unsigned 32-bit integer
    MAX_VAL = 4294967295
    
    """
    Cosmic mitigation is an algorithm to reduce the effect of cosmic rays.
    The algorithm is applied while stacking two-second images.  For each
    block of N two-second images, and for each pixel in each image, drop
    the maximum values and the minimum values when adding up the values
    for each pixel.
    """
    def __init__(self, block_size=10, action=SUBTRACT_NONE):
        super(Stacker, self).__init__()
        """
        action - which items to subtract: min, max, both, or none
        block_size - number of 2s images to consider as a block
        """
        self.action = int(action)
        if self.action not in [0, 1, 2, 3]:
            raise ValueError("Unknown cosmic mitigation action %s" % action)
        logger.info("cosmic mitigation: action=%s" %
                    Stacker.ACTION_LABEL[self.action])
        self.block_size = int(block_size)
        logger.info("cosmic mitigation: block_size=%s" % block_size)

    def get_config(self):
        return {
            'block_size': self.block_size,
            'action': Stacker.ACTION_LABEL[self.action],
        }

    def stack_exposures(self, hdulist,
                        imgdir, cam_id, ccd_id, start_frame, end_frame):
        minval = np.zeros_like(hdulist[0].data)
        minval += Stacker.MAX_VAL
        maxval = np.zeros_like(hdulist[0].data)
        idx = 0
        for i in range(start_frame, end_frame + 1):
            filename = CCDImage.get_filename(imgdir, cam_id, ccd_id, i, 2)
            logger.debug("read exposure %s" % filename)
            try:
                new_hdulist = fits.open(filename)
                assert(hdulist[0].shape == new_hdulist[0].shape)
                hdulist[0].data += new_hdulist[0].data
                idx += 1
                if self.action != Stacker.SUBTRACT_NONE:
                    minval = np.minimum(minval, new_hdulist[0].data)
                    maxval = np.maximum(maxval, new_hdulist[0].data)
                    if idx % self.block_size == 0:
                        if self.action in [Stacker.SUBTRACT_MINMAX,
                                           Stacker.SUBTRACT_MIN]:
                            hdulist[0].data -= minval
                        if self.action in [Stacker.SUBTRACT_MINMAX,
                                           Stacker.SUBTRACT_MAX]:
                            hdulist[0].data -= maxval
                        # reset the hi/lo
                        minval = np.zeros_like(hdulist[0].data)
                        minval += Stacker.MAX_VAL
                        maxval = np.zeros_like(hdulist[0].data)
            except IOError as e:
                logger.error("read failed: %s" % e)
        return hdulist


class SubpixelBuffer(object):
    """
    Pixel buffer with sub-pixel resolution relative to a CCD.

    The subpixel buffer counts electrons.  When the subpixel buffer is
    downsampled to the CCD pixels, the electron count is converted to
    DHU count.
    """
    def __init__(self, cols, rows, resolution=1, num_buffer_pixels=0):
        """
        Create a pixel buffer that is higher resolution than the CCD.

        resolution - number of pixels per CCD pixel
        num_buffer_pixels - number of CCD pixels beyond the CCD pixels

        The subpixel pixels count electrons, and these values may be
        fractional.  So use float for the data type.        
        """
        width = int((cols + 2 * num_buffer_pixels) * resolution)
        height = int((rows + 2 * num_buffer_pixels) * resolution)
        logger.debug("allocate subpixel buffer %sx%s at resolution %s" %
                     (width, height, resolution))
        self.resolution = int(resolution)
        self.num_buffer_pixels = int(num_buffer_pixels)
        self.ccd_width = cols
        self.ccd_height = rows
        self.width = width
        self.height = height
        self.pixels = np.zeros(shape=(self.height, self.width), dtype=float)

    def get_label(self):
        return "SB %dx%d res=%d buf=%d" % (
            self.width, self.height, self.resolution, self.num_buffer_pixels)

    def clear(self):
        """set all pixel values to zero"""
        self.pixels[:] = 0.0

    @staticmethod
    def get_electrons(tmag, teff):
        """
        For a given magnitude, return the expected number of electrons.
        """
        exptime = 2.0 # seconds
        area = 73.0 # cm^2
        num_electrons = area * exptime * 1.6 * 10.0 ** (6.0 - 0.4 * tmag)
        return num_electrons

    @staticmethod
    def get_coord(x, y, granularity):
        """
        Transform CCD coordinates, in pixels, using the specified granularity,
        in pixels.  A granularity of 10 transforms the first 10 pixels to
        pixel 5, the next 10 pixels to pixel 15, and so on.  A granularity
        of None indicates no transformation.
        """
        if granularity is None:
            return x, y
        pixels_per_block_x = 2048 / granularity
        newx = int(pixels_per_block_x * (0.5 + x / pixels_per_block_x))
        pixels_per_block_y = 2048 / granularity
        newy = int(pixels_per_block_y * (0.5 + y / pixels_per_block_y))
        return newx, newy

    def apply_electrons_ccd(self, ccd_number, x, y, tmag, teff, psfsrc,
                            granularity=None, caching=None):
        """
        The x,y is in the reference frame and units of the CCD, not the
        subpixel buffer.
        """
        x = (self.num_buffer_pixels + x) * self.resolution
        y = (self.num_buffer_pixels + y) * self.resolution
        self.apply_electrons(ccd_number, x, y, tmag, teff, psfsrc,
                             granularity, caching)

    def apply_electrons(self, ccd_number, x, y, tmag, teff, psfsrc,
                        granularity=None, caching=None):
        """
        Put the electrons into each pixel.  Note that the pixels of the
        subpixel buffer count electrons, not ADUs.

        The x,y are in the reference frame and units of the subpixel buffer.

        This assumes all PSFs are are calculated relative to a single
        CCD where grid of PSFs has 0,0 corresponding to camera boresight
        and each PSF grid has 0,0 oriented toward camera boresight.
        """
        caching = caching or {}
        num_electrons = SubpixelBuffer.get_electrons(tmag, teff)
        if psfsrc is not None:
            # Transform from CCD reference frame to PSF reference frame
            if ccd_number == 1 or ccd_number == 3:
                psfx = self.width - x
                psfy = self.height - y
            else:
                psfx = x
                psfy = self.height - y
            # Convert from subpixel to pixel
            psfx /= self.resolution
            psfy /= self.resolution
            # Adjust coordinates depending on the granularity
            (query_x, query_y) = self.get_coord(psfx, psfy, granularity)
            # Get the exact stellar temperature
            stellar_temp = psfsrc.get_nearest_temperature(teff)
            # Query for the PSF based on field position in pixels
            psf = psfsrc.interpolated_position_query(
                query_x, query_y, self.resolution, stellar_temp,
                cacher=caching.get('cacher', None),
                cache_mode=caching.get('mode', 'interpolated'),
                cache_log=caching.get('log', False))
            # Rearrange rows/cols of the PSF grid
            if ccd_number == 1 or ccd_number == 3:
                electron_grid = psf.grid[::-1, ::-1]
            else:
                electron_grid = psf.grid[::-1, ::]
            # Apply the electron count across the grid
            electron_grid *= num_electrons
            # Add the PSF to the subpixel buffer
            self.overlay(self.pixels, electron_grid, x, y)
        else:
            try:
                self.pixels[int(y), int(x)] += num_electrons
            except IndexError:
                # ignore anything that is not on the grid
                pass

    @staticmethod
    def overlay(subpixel_grid, electron_grid, x, y):
        """
        Position the electron grid over the subpixel grid at x,y then 'drop'
        the electrons from the electron grid into the subpixel grid elements.

        This assumes that the subpixel and electron grids are the same
        resolution (wrt CCD pixels) and that they are not rotated relative
        to each other.
        """
        w_p = electron_grid.shape[1]
        h_p = electron_grid.shape[0]
        w_s = subpixel_grid.shape[1]
        h_s = subpixel_grid.shape[0]

        if x < w_p / 2:
            x0 = 0
            x1 = max(x + w_p / 2 + 1, 0)
            xp0 = max(w_p / 2 - x, 0)
            xp1 = w_p
        elif x >= w_s - w_p / 2:
            x0 = min(x - w_p / 2, w_s)
            x1 = w_s
            xp0 = 0
            xp1 = x1 - x0
        else:
            x0 = x - w_p / 2
            x1 = x + w_p / 2 + 1
            xp0 = 0
            xp1 = w_p

        if y < h_p / 2:
            y0 = 0
            y1 = max(y + h_p / 2 + 1, 0)
            yp0 = max(h_p / 2 - y, 0)
            yp1 = h_p
        elif y >= h_s - h_p / 2:
            y0 = min(y - h_p / 2, h_s)
            y1 = h_s
            yp0 = 0
            yp1 = y1 - y0
        else:
            y0 = y - w_p / 2
            y1 = y + w_p / 2 + 1
            yp0 = 0
            yp1 = h_p

        # be sure it is still in the field
        if (0 <= y0 < h_s and 0 < y1 <= h_s and
            0 <= x0 < w_s and 0 < x1 <= w_s and
            0 <= yp0 < h_p and 0 < yp1 <= h_p and
            0 <= xp0 < w_p and 0 < xp1 <= w_p):
            subpixel_grid[y0: y1, x0: x1] += electron_grid[yp0: yp1, xp0: xp1]

    def downsample_to_ccd(self):
        """Transform from subpixel pixels to CCD pixels"""
        logger.debug("downsample %dx%d to %dx%d" %
                     (self.width, self.height, self.ccd_width, self.ccd_height))
        center = self.pixels[
            self.num_buffer_pixels * self.resolution:
            (self.num_buffer_pixels + self.ccd_height) * self.resolution,
            self.num_buffer_pixels * self.resolution:
            (self.num_buffer_pixels + self.ccd_width) * self.resolution]
        return center.reshape(
            (self.ccd_height, self.resolution, self.ccd_width, -1)
        ).sum(axis=3).sum(axis=1)

    def make_plot(self, show_grid=True, ax=None):
        """
        Make a plot of the subpixel buffer.

        show_grid - if true, display a grid of pixel borders
        """
        import matplotlib.pylab as plt
        if ax is None:
            plt.figure(self.get_label())
            ax = plt.subplot()
        else:
            plt.sca(ax)
        ax.imshow(self.pixels, interpolation='none')
        ax.set_aspect(1)
        ax.set_title(self.get_label())
        ax.invert_yaxis()
        if show_grid:
            from matplotlib import collections
            edges = []
            for i in range(0, self.width, self.resolution):
                edges.append([(i, 0), (i, self.height)])
            for i in range(0, self.height, self.resolution):
                edges.append([(0, i), (self.width, i)])
            ax.add_collection(collections.LineCollection(edges, alpha=0.4))
        return plt.gcf()


class CatalogSelection(object):
    """This object stores data for various celestial objects."""

    def __init__(self, catalog=None):
        """Initialize by copying data from a catalog."""
        self.epoch = 0
        self._id = np.array([])
        self.ra = np.array([])
        self.dec = np.array([])
        self.pmra = np.array([])
        self.pmdec = np.array([])
        self.tmag = np.array([])
        self.teff = np.array([])
        self.projected_ra = np.array([])
        self.projected_dec = np.array([])
        self.lightcurve = []
        if catalog:
            self.add_stars(catalog)

    def add_stars(self, catalog):
        self.epoch = catalog.epoch # FIXME: this is a potential data skew
        self._id = np.append(self._id, catalog._id)
        self.ra = np.append(self.ra, catalog._ra)
        self.dec = np.append(self.dec, catalog._dec)
        self.pmra = np.append(self.pmra, catalog._pmra)
        self.pmdec = np.append(self.pmdec, catalog._pmdec)
        self.tmag = np.append(self.tmag, catalog._tmag)
        self.teff = np.append(self.teff, catalog._teff)

    def size(self):
        return self.tmag.size

    def apply_proper_motion(self, ts_year):
        """
        Calculate the positions of the items at the specified point in time.

        ts_year - requested time as decimal year

        Returns two arrays: the array of ra and array of dec.
        """
        # how many years since the catalog epoch?
        elapsed = ts_year - self.epoch # years
        logger.info('projecting %.3f years relative to %.2f for %d objects' %
                    (elapsed, self.epoch, len(self.ra)))
        # calculate the dec in degree/year, assuming pmdec in mas/year
        dec_rate = self.pmdec / 60.0 / 60.0 / 1000.0
        self.projected_dec = self.dec + elapsed * dec_rate
        # calculate unprojected rate of ra motion using the mean declination
        # between the catalog epoch and the requested time, in degrees of
        # ra/year assuming original was projected mas/year
        ra_rate = self.pmra / 60.0 / 60.0 / np.cos((self.dec + elapsed * dec_rate / 2.0) * np.pi / 180.0) / 1000.0
        self.projected_ra = self.ra + elapsed * ra_rate

    def apply_lightcurves(self, lcg):
        """Apply a lightcurve to each object in this selection."""
        if lcg is None:
            return
        self.lightcurve = lcg.get_lightcurves(self.tmag)
        self.lightcurve_indices = []
        for idx in range(len(self.lightcurve)):
            if self.lightcurve[idx] is not None:
                self.lightcurve_indices.append(idx)

    def save_lightcurves(self, outdir, cam_id):
        """Save the star identifier and associated lightcurve to file"""
        filename = "%s/lightcurves-cam%s" % (outdir, cam_id)
        logger.debug("save %s lightcurves to %s" %
                     (np.count_nonzero(self.lightcurve), filename))
        with open(filename, "w") as f:
            for idx in range(len(self.lightcurve)):
                if self.lightcurve[idx] is not None:
                    f.write("%d %s\n" %
                            (self._id[idx], self.lightcurve[idx].code))

    def get_magnitude(self, ts_day):
        """
        Return the magnitude of each object at a specific time.

        ts_day - desired time as decimal day
        """
        tmag = np.array(self.tmag)
        if len(self.lightcurve):
            assert(len(self.lightcurve) == len(tmag))
            logger.debug("calculate moments for %s stars" %
                         len(self.lightcurve_indices))
            for i in range(len(self.lightcurve_indices)):
                idx = self.lightcurve_indices[i]
                tmag[idx] += self.lightcurve[idx].get_mag(ts_day)
#            logger.debug("calculate moment from %s lightcurves" % len(tmag))
#            moment = np.array([c.get_mag(t) for c in self.lightcurve])
#            tmag += moment
        else:
            logger.debug("no lightcurves, using default magnitudes")
        return tmag


class TargetList(ConfigurableObject):
    """
    Figure out which stars should be rendered.
    """

    def __init__(self, filename=None, radius=0.35, **targets):
        """
        Targets can be specified individually or by a list in a file.

        filename - File that contains a list of targets.

        radius - Distance around target that should be queried, in degrees.
                 A value of 0 indicates query only the target.
        """
        super(TargetList, self).__init__()
        self.filename = filename
        self.radius = float(radius)
        self.targets = []
        if filename is not None:
            self.targets = self.read_target_file(filename)
        for tgt_label in targets:
            tgt_dict = dict()
            if 'loc' in targets[tgt_label]:
                ra, dec = targets[tgt_label].get('loc')
                tgt_dict['ra'] = float(ra)
                tgt_dict['dec'] = float(dec)
            if 'tic_id' in targets[tgt_label]:
                tgt_dict['tic_id'] = int(targets[tgt_label].get('tic_id'))
            # for backward compatibility, treat id as tic_id
            if 'id' in targets[tgt_label]:
                tgt_dict['tic_id'] = int(targets[tgt_label].get('id'))
            if 'radius' in targets[tgt_label]:
                tgt_dict['radius'] = float(targets[tgt_label].get('radius'))
            if tgt_dict:
                tgt_dict['label'] = tgt_label
                self.targets.append(tgt_dict)
            else:
                logger.info("no loc or tic_id specified for target %s" % tgt_label)
        if self.targets:
            logger.info("%d targets specified" % len(self.targets))
            logger.info("default target radius %s" % self.radius)

    def get_config(self):
        targets = dict()
        for i, t in enumerate(self.targets):
            targets["%d" % i] = t
        return {
            'radius': self.radius,
            'filename': self.filename,
            'targets': targets,
        }

    @staticmethod
    def parse_target(tgt_str):
        tgt_dict = dict()
        if tgt_str.count(',') == 1:
            parts = tgt_str.split(',')
            if parts[0] and parts[1]:
                tgt_dict['ra'] = float(parts[0])
                tgt_dict['dec'] = float(parts[1])
        elif tgt_str.count(',') == 3:
            parts = tgt_str.split(',')
            if parts[0]:
                tgt_dict['tic_id'] = int(parts[0])
            if parts[1] and parts[2]:
                tgt_dict['ra'] = float(parts[1])
                tgt_dict['dec'] = float(parts[2])
            if parts[3]:
                tgt_dict['radius'] = float(parts[3])
        else:
            parts = tgt_str.split()
            if len(parts) == 1:
                tgt_dict['tic_id'] = int(parts[0])
            elif len(parts) == 2:
                tgt_dict['ra'] = float(parts[0])
                tgt_dict['dec'] = float(parts[1])
            elif len(parts) == 3:
                tgt_dict['ra'] = float(parts[0])
                tgt_dict['dec'] = float(parts[1])
                tgt_dict['radius'] = float(parts[2])
        return tgt_dict

    @staticmethod
    def read_target_file(filename):
        """
        Read targets from a file.  Each line must contain one to three elements
        with the following meaning:

        tic_id,ra,dec,radius

        A line may contain the following:

        tic_id,,,
        tic_id,,,radius
        ,ra,dec,
        ,ra,dec,radius

        The following shortcut formats are recognized:

        ra,dec
        ra dec
        ra dec radius

        Lines that begin with # are ignored.
        """
        targets = []
        try:
            logger.debug("read target file %s" % filename)
            with open(filename) as f:
                for line in f:
                    target = line.strip()
                    if target.startswith('#'):
                        continue
                    tgt_dict = TargetList.parse_target(target)
                    if tgt_dict:
                        targets.append(tgt_dict)
        except OSError as e:
            logger.error("read failed for target file: %s" % e)
            raise
        logger.debug("read %s targets from file" % len(targets))
        return targets

    @staticmethod
    def _loc_to_cam(ra, dec, spacecraft):
        """
        Given a location as ra,dec, determine which camera field of view
        will capture that location.
        """
        model = spacecraft.get_model()
        cam_id = None
        cam_x = None
        cam_y = None
        cam_ccd = None
        for (cam_i, camera) in enumerate(spacecraft.camera):
            x, y, ccd_n = model.celestial_to_pixel(
                ra, dec, spacecraft.ra, spacecraft.dec, spacecraft.roll,
                camera.get_geometry(), camera.get_ccd_geometries())
            if ccd_n[0]:
                cam_id = cam_i + 1
                cam_x = x[0]
                cam_y = y[0]
                cam_ccd = ccd_n[0]
                logger.debug("target is on camera %s "
                             "(ra=%s,dec=%s)(x=%s,y=%s,ccd=%s)"
                             % (cam_id, ra, dec, cam_x, cam_y, cam_ccd))
        return cam_id, cam_x, cam_y, cam_ccd

    def get_targets(self, catalog, spacecraft, radius=None,
                    active_cameras=None):
        """
        Get the list of stars that will be rendered.

        Return one catalog selection for each camera.

        This function modifies the contents of the internal targets array.
        It adds information about each target as resolution of each target
        is attempted.
        """
        if active_cameras is None:
            active_cameras = (1, 2, 3, 4)
        stars = dict()
        if self.targets:
            logger.info("aquire targets from specified list")
            fail = 0
            for (cam_i, camera) in enumerate(spacecraft.camera):
                cam_id = cam_i + 1
                stars[cam_id] = CatalogSelection()
            for star in self.targets:
                ra_star = star.get('ra')
                dec_star = star.get('dec')
                radius = star.get('radius', self.radius)
                if ra_star is None or dec_star is None:
                    id_star = star.get('tic_id')
                    if id_star is not None:
                        result, _ = catalog.query_by_id(id_star, 'ra,dec')
                        # FIXME: need to do some rounding or later query fails
                        if result:
                            ra_star = result[0][0]
                            dec_star = result[0][1]
                if ra_star and dec_star:
                    cam_id, x, y, ccd_id = self._loc_to_cam(
                        ra_star, dec_star, spacecraft)
                    # confirm the values from lookup
                    star['x'] = x
                    star['y'] = y
                    star['ccd'] = ccd_id
                    star['cam'] = cam_id
                    if cam_id in active_cameras:
                        logger.info("target at %s,%s radius %s on camera %s"
                                    % (ra_star, dec_star, radius, cam_id))
                        catalog.query(ra_star, dec_star, radius)
                        if len(catalog._id) > 0:
                            stars[cam_id].add_stars(catalog)
                        else:
                            fail += 1
                            star['info'] = 'no such object'
                            logger.info("no object in catalog at %s,%s"
                                        % (ra_star, dec_star))
                    elif cam_id is not None:
                        fail += 1
                        star['info'] = 'inactive camera'
                        logger.info("target at %s,%s on inactive camera %s"
                                    % (ra_star, dec_star, cam_id))
                    else:
                        fail += 1
                        star['info'] = 'outside field of view'
                        logger.info("target at %s,%s is outside field of view"
                                    % (ra_star, dec_star))
                    catalog.clear_query()
                else:
                    fail += 1
                    star['info'] = 'no ra,dec found for this target'
                    logger.info("nothing found for target '%s'" % star)
                star['radius'] = radius # confirm the actual radius used
            if fail:
                logger.info("%s unusable targets of %s targets specified" %
                            (fail, len(self.targets)))
        else:
            logger.info("aquire targets from catalog")
            for (cam_i, camera) in enumerate(spacecraft.camera):
                cam_id = cam_i + 1
                if cam_id in active_cameras:
                    radius = radius or camera.fov_radius
                    ra_cam, dec_cam, roll_cam = camera.get_pointing()
                    logger.info("camera %s: pointing %s,%s,%s radius %s"
                                % (cam_id, ra_cam, dec_cam, roll_cam, radius))
                    catalog.query(ra_cam, dec_cam, radius)
                    stars[cam_id] = CatalogSelection(catalog)
                    catalog.clear_query()
                else:
                    stars[cam_id] = CatalogSelection()
        for cam_id in range(1, 5):
            logger.debug("camera %s: %s stars" % (cam_id, stars[cam_id].size()))
        return stars
