import ngmix
import numpy as np
import meds
import os
import psfex
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


    def make_mask(self, column_dark_thresh = None, global_dark_thresh = None, global_flat_thresh = None):
        # Use the flats and darks to generate a bad pixel mask.
        # make use of np.ravel/np.reshape for this
        med_flat_array=[]
        med_dark_array=[]
        for d in self.dark_files:
            hdu=fits.open(d)
            flattened=np.ravel(hdu[0].data)
            outrav=np.zeros(hdu[0].data.size)
            outrav[flattened>=column_dark_thresh]=1
            med_dark_array.append(outrav)
        sum_dark = np.sum(med_dark_array,axis=0)
        superdark=np.ones(sum_dark.size)
        superdark[sum_dark==(len(dark_files))]=0
        #outfile = fits.PrimaryHDU(superdark.reshape(np.shape(hdu[0].data)))
        #outfile.writeto('superdark.fits')

        for f in self.flat_files:
            hdu=fits.open(f)
            exptime=(hdu[0].header['EXPTIME'])/1000.
            med_flat_array.append(hdu[0].data/exptime)

        self.mask_file = None

   def _make_detection_image(self, output = 'detection.fits'):
       '''
       :output: output file where detection image is written.

       Runs SWarp on provided image files to make a detection image.
       '''
       ### Code to run SWARP

       self.detection_file = output


    def make_catalog(self, sextractor_config_file = './astro_config/sextractor.config', sextractor_param_file = './astro_config/sextractor.param',psfex_config_file = './astro_config/'):
        '''
        Wrapper for astromatic tools to make catalog from provided images.
        '''
        self._make_detection_image()
        cmd = 'swarp '+' '.join(image_files)

        sxcmd = ' '.join(['sex',image_file,'-c','astro_config/sextractor.config'])
        os.system(cmd)

        pass

    def _select_stars_for_psf(self,stars_file = None):
        '''
        select stars from self.catalog using a sensible s/g separation criterion.
        write these stars to disk so PSFEx can use them.
        '''

        self.stars_file = None
        pass

    def _make_psf_model(self,imagefile):
        '''
        Wrapper for PSFEx. Assumes stars have already been selected, and are in the .fits file self.stars_file
        '''

        return psfex_model_file

    def make_psf_models(self):
        self.select_stars_for_psf() # this writes a star catalog file
        self.psfEx_models = []
        for image_file in self.image_files:
            psfex_model_file = self._make_psf_model(image)
            self.psfEx_models.append( psfex.PSFEx(psfex_model_file))

     def make_image_info_struct(self,max_len_of_filepath = 120):

         image_info = meds.util.get_image_info_struct(len(self.image_files),max_len_of_filepath)
         for image_file in self.image_files:
			image_info[i]['image_path'] = image_file
			image_info[i]['image_ext'] = 0
			image_info[i]['weight_path'] = self.weight_file
			image_info[i]['weight_ext'] = 0
			image_info[i]['bmask_path'] = self.mask_file
			image_info[i]['bmask_ext'] = 0
        return image_info

    def make_meds_config(self,extra_parameters = None):
		'''
		:extra_parameters: dictionary of keys to be used to update the base MEDS configuration dict

		'''
		# sensible default config.
		config = {'cutout_types':['weight','seg','bmask'],'psf_type':'psfex'}
        if extra_parameters is not None:
            config.update(extra_parameters)
        return config

	def _meds_metadata(self):
		meta = np.empty(1,[('magzp_ref',np.float)])
		meta['magzp_ref'] = 0.0
		return meta

	def _calculate_box_size(self,angular_size,size_multiplier = 2.5 min_size = 16, max_size= 64, pixel_scale = 0.23):
		'''
		Calculate the cutout size for this survey.

		:angular_size: angular size of a source, with some kind of angular units.
		:size_multiplier: Amount to multiply angular size by to choose boxsize.
		:deconvolved:
		:min_size:
		:max_size:

		'''
		box_size_float = np.ceil( angular_size/pixel_scale)

		# Available box sizes to choose from -> 16 to 256 in increments of 2
		available_sizes = min_size * 2**(np.arange(np.ceil(np.log2(max_size)-np.log2(min_size)+1)).astype(int))

		# If a single angular_size was proffered:
		if isinstance(box_size_float, np.ndarray):
			available_sizes_matrix = available_sizes.reshape(1,available_sizes.size)
			available_sizes_matrix[box_size_float.reshape(box_size_float.size,1) > available_sizes.reshape(1,available_sizes.size)] = np.max(available_sizes)+1
			box_size = np.min(available_sizes_matrix,axis=1)
		else:
			box_size = np.min( available_sizes[ available_sizes > box_size_float ] )
		return box_size

    def make_object_info_struct(self,catalog=None):
		if catalog is None:
			raise Exception("A catalog object is required to create MEDS files.")

		obj_str = meds.util.get_meds_input_struct(catalog.size,extra_fields = [('KRON_RADIUS',np.float)])
		obj_str['id'] = catalog['NUMBER']
		obj_str['box_size'] = self._calculate_box_size(catalog['KRON_RADIUS'])
		obj_str['ra'] = catalog['ALPHAWIN_J2000']
		obj_str['dec'] = catalog['DELTAWIN_J2000']
        obj_str['KRON_RADIUS'] = catalog['KRON_RADIUS']
        return obj_str


    def run(self,outfile = "superbit.meds"):

        # Make a mask.
        self.make_mask()
        # Combine images, make a catalog.
        self.make_catalog()
        # Build a PSF model for each image.
        self.make_psf_models()
        # Make the image_info struct.
        image_info = self.make_image_info_struct()
        # Make the object_info struct.
        obj_info = self.make_object_info_struct()
        # Make the MEDS config file.
        meds_config = self.make_meds_config()
        # Finally, make and write the MEDS file.
        medsObj = meds.maker.MEDSMaker(obj_data,self.image_info,config=meds_config,psf_data = psf_models,meta_data=meta)
        medsObj.write(outfile)
