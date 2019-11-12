import os,sys
import importlib.util
import glob
import pdb, traceback
import esutil as eu
# Get the location of the main superbit package.
dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,dir)
from superbit import medsmaker

# Start by making a directory...
if not os.path.exists('../Data/calib'):
    os.mkdir('../Data/')
    os.mkdir('../Data/calib')

# This is picking out only Luminance images
science = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/superbit-ngmix/WCS_images/image*_???_WCS.fits')
flats = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/FlatImages/*')
biases = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/BiasImages/*')
darks = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/DarkImages/*')
try:
    bm = medsmaker.BITMeasurement(image_files=science,flat_files=flats, dark_files=darks, bias_files=biases)
    # The path names should be updated; as written the code also expects all
    # calibration files to be in the same directory
    '''
    bm.set_working_dir()
    bm.set_path_to_calib_data(path='/Users/jemcclea/Research/SuperBIT_2019/A2218/')
    bm.set_path_to_science_data(path='/Users/jemcclea/Research/SuperBIT_2019/A2218/ScienceImages/')
    bm.reduce(overwrite=False)
    bm.make_mask(overwrite=False)
    bm.make_catalog()
    bm.make_psf_models()
    image_info = bm.make_image_info_struct()
    obj_info = bm.make_object_info_struct()
    '''
    bm.run(clobber=False)

except:
    thingtype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
