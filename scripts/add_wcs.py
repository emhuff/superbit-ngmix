from astropy.io import fits
import numpy as np
from astropy import wcs
import sys,os
import glob
import string

"""
Script to add externally generated (e.g., astrometry-net) WCS information to
the headers of science FITS files.
 - Locate science (cluster) FITS images, probably with glob
 - Locate external WCS files, maybe with string.strip (if the file
    doesn't exist, don't proceed with that file)
 - Open the FITS file, 
"""

def main(argv):


if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
