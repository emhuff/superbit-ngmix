import os,sys
import importlib.util
import glob
import pdb
# Get the location of the main superbit package.
dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,dir)
from superbit import medsmaker

# Start by making a directory...
if not os.path.exists('../Data/calib'):
    os.mkdir('../Data/')
    os.mkdir('../Data/calib')


science = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/ScienceImages/image*fits')
flats = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/FlatImages/*')
biases = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/BiasImages/*')
darks = glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/DarkImages/*')

bm = medsmaker.BITMeasurement(image_files=science,flat_files=flats, dark_files=darks, bias_files=biases)
# The path names should be updated; as written the code also expects all
# calibration files to be in the same directory


bm.set_path_to_calib_data(path='/Users/jemcclea/Research/SuperBIT_2019/A2218/')
bm.set_path_to_science_data(path='/Users/jemcclea/Research/SuperBIT_2019/A2218/ScienceImages/')
bm.set_path_to_wcs_data(path='/Users/jemcclea/Research/SuperBIT_2019/A2218/ScienceImages/')
#bm.add_wcs_to_science_frames()
bm.reduce()
