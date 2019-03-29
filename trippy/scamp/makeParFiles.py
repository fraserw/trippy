#! /usr/bin/env python
from __future__ import (absolute_import, division, print_function, unicode_literals)
"""
Copyright (C) 2016  Wesley Fraser

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = 'Wesley Fraser (@wtfastro, github: fraserw <westhefras@gmail.com>), Academic email: wes.fraser@qub.ac.uk'

import os.path as osp


def outFile(name,strings,overwrite = False):
    if not osp.isfile(name) or overwrite:
        han=open(name,'w+')
        for i in range(len(strings)):
            print(strings[i],file=han)
        han.close()
    return

#write the default sextractor convolution kernel to the cwd
def writeConv(fileName='default.conv',overwrite=False):
    """Write a default sextractor convolution kernel.
    """
    strArr=['CONV NORM',
            '# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.',
            '1 2 1',
            '2 4 2',
            '1 2 1']

    outFile(fileName,strArr,overwrite=overwrite)
    return

#write a parameter list to the cwd
#this is a modified default.param from sextractor
def writeParam(fileName='def.param',numAps=1,overwrite=False):
    """Write a sex parameter list to the current working directory.
    Default file is called def.param unless otherwise specified.

    Generally speaking, this file writes out everything you'd need to run both sextractor and scamp.
    """

    strArr=['NUMBER',
            '',
            'XWIN_IMAGE',
            'YWIN_IMAGE',
            'ERRX2WIN_IMAGE',
            'ERRY2WIN_IMAGE',
            '',
            'ERRAWIN_IMAGE ',
            'ERRBWIN_IMAGE ',
            'ERRTHETAWIN_IMAGE',
            '',
            'FLUX_AUTO',
            'FLUX_RADIUS',
            'FLUXERR_AUTO',
            'FLUX_ISOCOR',
            'FLUXERR_ISOCOR'
            '',
            'FLAGS',
            '#FLAGS_WEIGHT',
            '#IMAFLAGS_ISO',
            '',
            'X_WORLD',
            'Y_WORLD',
            'ERRA_WORLD',
            'ERRB_WORLD',
            'ERRTHETA_WORLD',
            'MAG_AUTO',
            'MAG_APER',
            '',
            'ALPHA_SKY',
            'DELTA_SKY',
            '',
            'AWIN_IMAGE',
            'BWIN_IMAGE',
            'THETA_IMAGE',
            '',
            'FWHM_IMAGE']

    if numAps>1:
        for jj in range(len(strArr)):
            if 'MAG_APER' in strArr[jj]:
                strArr[jj]+='['+str(int(numAps))+']'
    outFile(fileName,strArr,overwrite=overwrite)
    return

#write a modified default scamp file
def writeScamp(fileName='def.scamp',distort=2,overwrite=False):
    """Write a modified nearly-default scamp file to the current
    working directory.

    Options - distort is the order of the geometric distortion for
              the scamp fit.

    Default file name is def.scamp unless otherwise specified.
    """

    strArr=['# Default configuration file for SCAMP modded by Fraser',
            ' ',
            '#----------------------------- Field grouping ---------------------------------',
            '',
            'FGROUP_RADIUS          1.0             # Max dist (deg) between field groups',
            ' ',
            '#---------------------------- Reference catalogs ------------------------------',
            ' ',
            'REF_SERVER         cocat1.u-strasbg.fr # Internet addresses of catalog servers',
            'ASTREF_CATALOG         USNO-B1         # NONE, FILE, USNO-A1, USNO-A2, USNO-B1,',
            '                                       # GSC-1.3, GSC-2.2, UCAC-1, UCAC-2,',
            '                                       # NOMAD-1, 2MASS, DENIS-3,',
            '                                       # SDSS-R3, SDSS-R5 or SDSS-R6',
            'ASTREF_BAND            DEFAULT         # Photom. band for astr.ref.magnitudes',
            '                                       # or DEFAULT, BLUEST, or REDDEST',
            'ASTREFCAT_NAME         astrefcat.cat   # Local astrometric reference catalogs',
            'ASTREFCENT_KEYS        X_WORLD,Y_WORLD # Local ref.cat.centroid parameters',
            'ASTREFERR_KEYS         ERRA_WORLD, ERRB_WORLD, ERRTHETA_WORLD',
            '                                       # Local ref.cat.error ellipse parameters',
            'ASTREFMAG_KEY          MAG             # Local ref.cat.magnitude parameter',
            'SAVE_REFCATALOG        Y               # Save ref catalogs in FITS-LDAC format?',
            'REFOUT_CATPATH         .               # Save path for reference catalogs',
            ' ',
            '#--------------------------- Merged output catalogs ---------------------------',
            ' ',
            'MERGEDOUTCAT_NAME      scamp.cat       # Merged output catalog filename',
            'MERGEDOUTCAT_TYPE      NONE            # NONE, ASCII_HEAD, ASCII, FITS_LDAC',
            ' ',
            '#----------------------------- Pattern matching -------------------------------',
            ' ',
            'MATCH                  Y               # Do pattern-matching (Y/N) ?',
            'MATCH_NMAX             0               # Max.number of detections for MATCHing',
            '                                       # (0=auto)',
            'PIXSCALE_MAXERR        1.2             # Max scale-factor uncertainty',
            'POSANGLE_MAXERR        5.0             # Max position-angle uncertainty (deg)',
            'POSITION_MAXERR        1.0             # Max positional uncertainty (arcmin)',
            'MATCH_RESOL            0               # Matching resolution (arcsec); 0=auto',
            'MATCH_FLIPPED          N               # Allow matching with flipped axes?',
            'MOSAIC_TYPE            UNCHANGED       # UNCHANGED, SAME_CRVAL, SHARE_PROJAXIS,',
            '                                       # FIX_FOCALPLANE or LOOSE',
            ' ',
            '#---------------------------- Cross-identification ----------------------------',
            ' ',
            'CROSSID_RADIUS         2.0             # Cross-id initial radius (arcsec)',
            ' ',
            '#---------------------------- Astrometric solution ----------------------------',
            ' ',
            'SOLVE_ASTROM           Y               # Compute astrometric solution (Y/N) ?',
            'ASTRINSTRU_KEY         FILTER,QRUNID   # FITS keyword(s) defining the astrom',
            'STABILITY_TYPE         EXPOSURE        # EXPOSURE, GROUP, INSTRUMENT or FILE',
            'CENTROID_KEYS          XWIN_IMAGE,YWIN_IMAGE # Cat. parameters for centroiding',
            'CENTROIDERR_KEYS       ERRAWIN_IMAGE,ERRBWIN_IMAGE,ERRTHETAWIN_IMAGE',
            '                                       # Cat. params for centroid err ellipse',
            'DISTORT_KEYS           XWIN_IMAGE,YWIN_IMAGE # Cat. parameters or FITS keywords',
            'DISTORT_GROUPS         1,1             # Polynom group for each context key',
            'DISTORT_DEGREES        '+str(distort)+'               # Polynom degree for each group',
            ' ',
            '#---------------------------- Photometric solution ----------------------------',
            ' ',
            'SOLVE_PHOTOM           N               # Compute photometric solution (Y/N) ?',
            'MAGZERO_OUT            0.0             # Magnitude zero-point(s) in output',
            'MAGZERO_INTERR         0.01            # Internal mag.zero-point accuracy',
            'MAGZERO_REFERR         0.03            # Photom.field mag.zero-point accuracy',
            'PHOTINSTRU_KEY         FILTER          # FITS keyword(s) defining the photom.',
            'MAGZERO_KEY            PHOT_C          # FITS keyword for the mag zero-point',
            'EXPOTIME_KEY           EXPTIME         # FITS keyword for the exposure time (s)',
            'AIRMASS_KEY            AIRMASS         # FITS keyword for the airmass',
            'EXTINCT_KEY            PHOT_K          # FITS keyword for the extinction coeff',
            'PHOTOMFLAG_KEY         PHOTFLAG        # FITS keyword for the photometry flag',
            'PHOTFLUX_KEY           FLUX_AUTO       # Catalog param. for the flux measurement',
            'PHOTFLUXERR_KEY        FLUXERR_AUTO    # Catalog parameter for the flux error',
            ' ',
            '#------------------------------- Check-plots ----------------------------------',
            ' ',
            'CHECKPLOT_DEV          PNG             # NULL, XWIN, TK, PS, PSC, XFIG, PNG,',
            '                                      # or JPEG',
            'CHECKPLOT_TYPE         FGROUPS,DISTORTION,ASTR_INTERROR2D,ASTR_INTERROR1D,ASTR_REFERROR2D,ASTR_REFERROR1D,ASTR_CHI2,PHOT_ERROR',
            'CHECKPLOT_NAME         fgroups,distort,astr_interror2d,astr_interror1d,astr_referror2d,astr_referror1d,astr_chi2,psphot_error # Check-plot filename(s)',
            ' ',
            '#------------------------------ Miscellaneous ---------------------------------',
            ' ',
            'SN_THRESHOLDS          10.0,100.0      # S/N thresholds (in sigmas) for all and',
            '                                       # high-SN sample',
            'FWHM_THRESHOLDS        0.0,100.0       # FWHM thresholds (in pixels) for sources',
            'AHEADER_SUFFIX         .ahead          # Filename extension for additional',
            '                                       # INPUT headers',
            'HEADER_SUFFIX          .head           # Filename extension for OUTPUT headers',
            'VERBOSE_TYPE           NORMAL          # QUIET, NORMAL, LOG or FULL',
            'WRITE_XML              Y               # Write XML file (Y/N)?',
            'XML_NAME               scamp.xml       # Filename for XML output',
            'NTHREADS               2               # Number of simultaneous threads for',
            '                                       # the SMP version of SCAMP',
            '                                       # 0 = automatic']
    outFile(fileName,strArr,overwrite=overwrite)
    return


#write a modified sex file for astrometry purposes in the scampe_.py scripts
def writeSex(fileName='def.sex',paramFileName='def.param',minArea=20, threshold=8, zpt=27.4, saturate=40000.,aperture=11,catalogType='FITS_LDAC',catalogName='def.cat',kron_factor=2.5,min_radius=3.5,overwrite=False):
    """Write a modified sextractor file.

    Options - paramFileName -the name of the associated parameter
                             file.
              minArea - the minimum threshold area for sources.
                        Default is 20.
              threshold - the snr threshold for sources.
                          default is 8
              zpt - the photometric zeropoint.
                    default is 27.4.
              saturation - the saturation level in adu.
                           default is 40000.
              aperture - the photometry aperture in pixels
                         default is 11.
    """

    if type(aperture) not in [type(1),type([1]),type(1.)]:
        print('Aperture can only be an integer, floating point, or LIST of those. A Numpy array will cause a failure here.')
        raise TypeError
    if type(aperture)==type([1]):
        aperString=''
        for jj in range(len(aperture)):
            aperString+=str(aperture[jj])+','
        aperString=aperString[:len(aperString)-1]
    else:
        aperString=str(aperture)

    if catalogType!='FITS_LDAC' and catalogType!='ASCII':
        import sys
        print(catalogType)
        print('Error: type should be either ASCII or FITS_LDAC')
        raise TypeError
        sys.exit()

    strArr=['# Default configuration file for SExtractor 2.3b2 modded by Fraser',
            ' ',
            '#-------------------------------- Catalog ------------------------------------',
            ' ',
            'CATALOG_NAME '+catalogName+'        # name of the output catalog',
            'CATALOG_TYPE '+catalogType+'        # "NONE","ASCII_HEAD","ASCII","FITS_1.0"',
            '                                # or "FITS_LDAC"',
            ' ',
            'PARAMETERS_NAME '+paramFileName+'   # name of the file containing catalog contents',
            ' ',
            '#------------------------------- Extraction ----------------------------------',
            ' ',
            'DETECT_TYPE     CCD             # "CCD" or "PHOTO"',
            'FLAG_IMAGE      flag.fits       # filename for an input FLAG-image',
            'DETECT_MINAREA  '+str(minArea)+'               # minimum number of pixels above threshold',
            'DETECT_THRESH   '+str(threshold)+'             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2',
            'ANALYSIS_THRESH 5             # <sigmas> or <threshold>,<ZP> in mag.arcsec-2',
            ' ',
            'FILTER          Y               # apply filter for detection ("Y" or "N")?',
            'FILTER_NAME     default.conv    # name of the file containing the filter',
            ' ',
            'DEBLEND_NTHRESH 32              # Number of deblending sub-thresholds',
            'DEBLEND_MINCONT 0.005           # Minimum contrast parameter for deblending',
            ' ',
            'CLEAN           Y               # Clean spurious detections? (Y or N)?',
            'CLEAN_PARAM     1.0             # Cleaning efficiency',
            ' ',
            'MASK_TYPE       CORRECT         # type of detection MASKing: can be one of',
            '                                # "NONE", "BLANK" or "CORRECT"',
            ' ',
            '#------------------------------ Photometry -----------------------------------',
            ' ',
            'PHOT_APERTURES  '+aperString+'               # MAG_APER aperture diameter(s) in pixels',
            'PHOT_AUTOPARAMS '+str(kron_factor)+' '+str(min_radius)+'        # MAG_AUTO parameters: <Kron_fact>,<min_radius>',
            ' ',
            'SATUR_LEVEL     '+str(saturate)+'         # level (in ADUs) at which arises saturation',
            ' ',
            'MAG_ZEROPOINT   '+str(zpt)+'             # magnitude zero-point',
            'MAG_GAMMA       4.0             # gamma of emulsion (for photographic scans)',
            'GAIN            2.6             # detector gain in e-/ADU',
            'PIXEL_SCALE     1.0             # size of pixel in arcsec (0=use FITS WCS info)',
            ' ',
            '#------------------------- Star/Galaxy Separation ----------------------------',
            ' ',
            'SEEING_FWHM     1.2             # stellar FWHM in arcsec',
            'STARNNW_NAME    default.nnw     # Neural-Network_Weight table filename',
            ' ',
            '#------------------------------ Background -----------------------------------',
            ' ',
            'BACK_SIZE       64              # Background mesh: <size> or <width>,<height>',
            'BACK_FILTERSIZE 3               # Background filter: <size> or <width>,<height>',
            ' ',
            'BACKPHOTO_TYPE  GLOBAL          # can be "GLOBAL" or "LOCAL"',
            ' ',
            '#------------------------------ Check Image ----------------------------------',
            ' ',
            'CHECKIMAGE_TYPE NONE            # can be one of "NONE", "BACKGROUND",',
            '                                # "MINIBACKGROUND", "-BACKGROUND", "OBJECTS",',
            '                                # "-OBJECTS", "SEGMENTATION", "APERTURES",',
            '                                # or "FILTERED"',
            'CHECKIMAGE_NAME check.fits      # Filename for the check-image',
            ' ',
            '#--------------------- Memory (change with caution!) -------------------------',
            ' ',
            'MEMORY_OBJSTACK 2000            # number of objects in stack',
            'MEMORY_PIXSTACK 200000          # number of pixels in stack',
            'MEMORY_BUFSIZE  1024            # number of lines in buffer',
            ' ',
            '#----------------------------- Miscellaneous ---------------------------------',
            ' ',
            'VERBOSE_TYPE    NORMAL          # can be "QUIET", "NORMAL" or "FULL"',
            '',
            '']
    outFile(fileName,strArr,overwrite)
    return

#write another modified scamp config file used in the scamp_*.py scripts
def writeScamp2(fileName='wes2.scamp'):
    strArr=['# Default configuration file for SCAMP 1.4.6-MP modded by Fraser',
            ' ',
            '#----------------------------- Field grouping ---------------------------------',
            ' ',
            'FGROUP_RADIUS          1.0             # Max dist (deg) between field groups',
            ' ',
            '#---------------------------- Reference catalogs ------------------------------',
            ' ',
            'REF_SERVER             FILE            # Internet addresses of catalog servers',
            'ASTREF_CATALOG         FILE            # NONE, FILE, USNO-A1, USNO-A2, USNO-B1,',
            '                                       # GSC-1.3, GSC-2.2, UCAC-1, UCAC-2,',
            '                                       # NOMAD-1, 2MASS, DENIS-3,',
            '                                       # SDSS-R3, SDSS-R5 or SDSS-R6',
            'ASTREF_BAND            DEFAULT         # Photom. band for astr.ref.magnitudes',
            '                                       # or DEFAULT, BLUEST, or REDDEST',
            'ASTREFCAT_NAME         astrefcat.cat   # Local astrometric reference catalogs',
            'ASTREFCENT_KEYS        X_WORLD,Y_WORLD # Local ref.cat.centroid parameters',
            'ASTREFERR_KEYS         ERRA_WORLD, ERRB_WORLD, ERRTHETA_WORLD',
            '                                       # Local ref.cat.error ellipse parameters',
            'ASTREFMAG_KEY          MAG_AUTO        # Local ref.cat.magnitude parameter',
            'SAVE_REFCATALOG        Y               # Save ref catalogs in FITS-LDAC format?',
            'REFOUT_CATPATH         .               # Save path for reference catalogs',
            ' ',
            '#--------------------------- Merged output catalogs ---------------------------',
            ' ',
            'MERGEDOUTCAT_NAME      scamp.cat       # Merged output catalog filename',
            'MERGEDOUTCAT_TYPE      NONE            # NONE, ASCII_HEAD, ASCII, FITS_LDAC',
            ' ',
            '#----------------------------- Pattern matching -------------------------------',
            ' ',
            'MATCH                  Y               # Do pattern-matching (Y/N) ?',
            'MATCH_NMAX             0               # Max.number of detections for MATCHing',
            '                                       # (0=auto)',
            'PIXSCALE_MAXERR        1.2             # Max scale-factor uncertainty',
            'POSANGLE_MAXERR        5.0             # Max position-angle uncertainty (deg)',
            'POSITION_MAXERR        1.0             # Max positional uncertainty (arcmin)',
            'MATCH_RESOL            0               # Matching resolution (arcsec); 0=auto',
            'MATCH_FLIPPED          N               # Allow matching with flipped axes?',
            'MOSAIC_TYPE            UNCHANGED       # UNCHANGED, SAME_CRVAL, SHARE_PROJAXIS,',
            '                                       # FIX_FOCALPLANE or LOOSE',
            ' ',
            '#---------------------------- Cross-identification ----------------------------',
            ' ',
            'CROSSID_RADIUS         2.0             # Cross-id initial radius (arcsec)',
            ' ',
            '#---------------------------- Astrometric solution ----------------------------',
            ' ',
            'SOLVE_ASTROM           Y               # Compute astrometric solution (Y/N) ?',
            'ASTRINSTRU_KEY         FILTER,QRUNID   # FITS keyword(s) defining the astrom',
            'STABILITY_TYPE         EXPOSURE        # EXPOSURE, GROUP, INSTRUMENT or FILE',
            'CENTROID_KEYS          XWIN_IMAGE,YWIN_IMAGE # Cat. parameters for centroiding',
            'CENTROIDERR_KEYS       ERRAWIN_IMAGE,ERRBWIN_IMAGE,ERRTHETAWIN_IMAGE',
            '                                       # Cat. params for centroid err ellipse',
            'DISTORT_KEYS           XWIN_IMAGE,YWIN_IMAGE # Cat. parameters or FITS keywords',
            'DISTORT_GROUPS         1,1             # Polynom group for each context key',
            'DISTORT_DEGREES        3               # Polynom degree for each group',
            ' ',
            '#---------------------------- Photometric solution ----------------------------',
            ' ',
            'SOLVE_PHOTOM           N               # Compute photometric solution (Y/N) ?',
            'MAGZERO_OUT            0.0             # Magnitude zero-point(s) in output',
            'MAGZERO_INTERR         0.01            # Internal mag.zero-point accuracy',
            'MAGZERO_REFERR         0.03            # Photom.field mag.zero-point accuracy',
            'PHOTINSTRU_KEY         FILTER          # FITS keyword(s) defining the photom.',
            'MAGZERO_KEY            PHOT_C          # FITS keyword for the mag zero-point',
            'EXPOTIME_KEY           EXPTIME         # FITS keyword for the exposure time (s)',
            'AIRMASS_KEY            AIRMASS         # FITS keyword for the airmass',
            'EXTINCT_KEY            PHOT_K          # FITS keyword for the extinction coeff',
            'PHOTOMFLAG_KEY         PHOTFLAG        # FITS keyword for the photometry flag',
            'PHOTFLUX_KEY           FLUX_AUTO       # Catalog param. for the flux measurement',
            'PHOTFLUXERR_KEY        FLUXERR_AUTO    # Catalog parameter for the flux error',
            ' ',
            '#------------------------------- Check-plots ----------------------------------',
            ' ',
            'CHECKPLOT_DEV          PNG             # NULL, XWIN, TK, PS, PSC, XFIG, PNG,',
            '                                       # or JPEG',
            'CHECKPLOT_TYPE         FGROUPS,DISTORTION,ASTR_INTERROR2D,ASTR_INTERROR1D,ASTR_REFERROR2D,ASTR_REFERROR1D,ASTR_CHI2,PHOT_ERROR',
            'CHECKPLOT_NAME         fgroups,distort,astr_interror2d,astr_interror1d,astr_referror2d,astr_referror1d,astr_chi2,psphot_error # Check-plot filename(s)',
            ' ',
            '#------------------------------ Miscellaneous ---------------------------------',
            ' ',
            'SN_THRESHOLDS          10.0,100.0      # S/N thresholds (in sigmas) for all and',
            '                                       # high-SN sample',
            'FWHM_THRESHOLDS        0.0,100.0       # FWHM thresholds (in pixels) for sources',
            'AHEADER_SUFFIX         .ahead          # Filename extension for additional',
            '                                       # INPUT headers',
            'HEADER_SUFFIX          .head           # Filename extension for OUTPUT headers',
            'VERBOSE_TYPE           NORMAL          # QUIET, NORMAL, LOG or FULL',
            'WRITE_XML              Y               # Write XML file (Y/N)?',
            'XML_NAME               scamp.xml       # Filename for XML output',
            'NTHREADS               2               # Number of simultaneous threads for',
            '                                       # the SMP version of SCAMP',
            '                                       # 0 = automatic']
    outFile(fileName,strArr)
    return


if __name__=="__main__":
    writeConv(fileName='default.conv')
    writeSex(fileName='test.sex',paramFileName='test.param',minArea=20, threshold=5, zpt=21.5, saturate=45000.,aperture=11)
    writeParam(fileName='test.param')
