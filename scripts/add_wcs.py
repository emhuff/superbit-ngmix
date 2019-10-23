from astropy.io import fits
import numpy as np
from astropy import wcs
import sys,os
import glob
import string
import pdb

"""
Script to add externally generated (e.g., astrometry-net) WCS information to
the headers of science FITS files.
 - Locate science (cluster) FITS images, probably with glob
 - Locate external WCS files, maybe with string.strip (if the file
    doesn't exist, don't proceed with that file)
 - Open the FITS file, open the wcs correction object as a wcs.WCS object_info
 - Use wcs.WCS methods to create a new header, preferably with some of the
    original header keywords!
 - Write a new FITS file with the image data and the new header
"""

def makeNewFITS(clusterName):
    '''
    Returns new cluster fits file with the
    updated WCS and some important keywords
    List of KW can probably be changed as needed
    '''
    clusterFITSName=os.path.join(imagepath,clusterName)
    #print "I get here 1"
    if os.path.exists(clusterFITSName):
        ClusterFITSFile=fits.open(clusterFITSName)
        ClusterHeader=ClusterFITSFile[0].header
        #print "I get here 2"
        WCSheader=get_wcs_info(clusterName)
        #print "I get here 3"

        try:
            for key in WCSheader.keys():
                ClusterHeader[key]=WCSheader[key]
            #print "I get here 4"
            outFITS=fits.PrimaryHDU(ClusterFITSFile[0].data,header=ClusterHeader)
            outFITS.writeto(os.path.join(outpath,clusterName.replace(".fits","WCS.fits")))
        except:
            "new FITS file not written..."
    else:
        print("File %s doesn't exist" % clusterFITSName)

    return

def get_wcs_info(clusterName):
    '''
    Return a new image header with WCS (SIP) information,
    or nothing if the WCS file doesn't exist
    '''
    try:
        splitted=string.split(clusterName,'_')
        wcsName=os.path.join(wcspath,str('wcs_'+splitted[2]+'_'+splitted[3]+'.fits'))
        inhead=fits.getheader(wcsName)
        w=wcs.WCS(inhead)
        wcs_sip_header=w.to_header(relax=True)
    except:
        print('cluster %s has no WCS, skipping...'%clusterName)
        #pdb.set_trace()
        wcs_sip_header=None
    return wcs_sip_header

def main(argv):
    global imagepath
    global wcspath
    global outpath
    imagepath = '/Users/jemcclea/Research/SuperBIT_2019/princeton_repo/b2k.princeton.edu/~wcj/transfer/superbit/timmins2019/quicklook'
    wcspath='/Users/jemcclea/Research/SuperBIT_2019/princeton_repo/b2k.princeton.edu/~wcj/transfer/superbit/timmins2019/quicklook'
    outpath='/Users/jemcclea/Research/SuperBIT_2019/WCS_images'

    imlist_name = 'cluster_images.txt'


    clusterImageNames=os.path.join(imagepath,imlist_name)
    with open(clusterImageNames,"r") as f:
        for cluster in f:
            print cluster
            makeNewFITS(cluster.strip('\n'))



if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
