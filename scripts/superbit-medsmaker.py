import ngmix
import numpy as np
import meds
import os
from astropy.io import fits
'''
Goals:
  - Take as input calibrated images
  - Make masks/weight maps
    -- based on flats/darks: locate bad pixels there
    -- also choose a minimum distance from the edge to mask.
  - Make a catalog (SExtractor)
  - Build a psf model (PSFEx)
  - run the meds maker (use meds.Maker)
  - run ngmix (run library)

'''


class BITMeasurement():
    def __init__(self, image_files = None, flat_files = None, dark_files = None):
        '''
        :image_files: Python List of image filenames; must be complete relative or absolute path.
        :flat_files: Python List of image filenames; must be complete relative or absolute path.
        :dark_files: Python List of image filenames; must be complete relative or absolute path.
        '''

        self.image_files = image_files
        self.flat_files = flat_files
        self.dark_files = dark_files


    def make_mask(self):
        # Use the flats and darks to generate a bad pixel mask.

        for f in self.flat_files:
            hdu=fits.open(f)
            exptime=hdu[0].header['EXPTIME']
            norm_flat
        self.mask = None

   def _make_detection_image(self, output = 'detection.fits'):
       '''
       :output: output file where detection image is written.

       Runs SWarp on provided image files to make a detection image.
       '''
       ### Code to run SWARP

       self.detection_file = output


    def make_catalog(self):
        '''
        Wrapper for astromatic tools to make catalog from provided images.
        '''
        self._make_detection_image()
        image_list = ','.join(image_files)
        cmd = 'swarp '
        os.system(cmd)

        pass

    def make_psf(self):
        pass
