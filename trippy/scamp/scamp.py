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


import subprocess
from astropy import wcs as WCS
from astropy.io import fits as pyf
import numpy as num
from os import path
import os,sys
#import future

pyversion = sys.version_info

def runScamp(scampFile,imageName):
    """
    Execute scamp using the input scamp parameter file on the provided image name.
    """

    if '.fits' in imageName:
        imageName=imageName.split('.fits')[0]
    if '.cat' not in imageName:
        comm='scamp '+imageName+'.cat -c '+scampFile
    else:
        comm='scamp '+imageName+' -c '+scampFile
    print(comm)
    print(subprocess.check_output(comm.split()))
    return

def checkSex(sexName):
    """Check whether Source Extracter is installed with a given name."""
    if (pyversion[0] == 3) & (pyversion[1] >= 6):
        try:
            _ = subprocess.Popen(sexName.split(), stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, encoding='utf8')
        except FileNotFoundError:
            return False
    else:
        try:
            _ = subprocess.Popen(sexName.split(), stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        except OSError:
            return False
    return True

def runSex(sexFile,imageName,options=None,verbose=False):
    """
    Execute sextractor using the input sextractor parameters file on the provided image name.

    Options should be a dictionary containing various command line options that sextractor takes. Example:
        scamp.runSex('my.sex','my.fits',options={'CATALOG_NAME': 'my.cat'})

    Set verbose to True to see the full sextractor output.

    The full sextractor output is returned.
    """

    # Source Extractor is sometimes installed as sex, sometimes as sextractor.
    sexNames = ['sex', 'sextractor']
    try:
        i = 0
        while not checkSex(sexNames[i]):
            i += 1
    except IndexError:
        print("Neither 'sex' nor 'sextractor' found.\n" +
              "Is Source Extractor installed?")
        raise
    sexName = sexNames[i]
    
    comm = sexName + ' ' + imageName + ' -c ' + sexFile

    if options:
        for ii in options:
            comm+=' -'+ii+' '+options[ii]
    if (pyversion[0] == 3) & (pyversion[1] >= 6):
        process = subprocess.Popen(comm.split(),stdout = subprocess.PIPE, stderr=subprocess.PIPE,encoding='utf8')
    else:
        process = subprocess.Popen(comm.split(),stdout = subprocess.PIPE, stderr=subprocess.PIPE)
    junk = process.communicate()
    #print(junk[1].split('\n\x1b[1M>'))
    #exit()
    if verbose:
        print(comm)
        for i in range(len(junk)):
            print(junk[i])
    return junk[1]


def getCatalog(catalogName,type='FITS_LDAC',paramFile=None):
    """
    Return the detections in the provided catalog name.

    Type is default FITS_LDAC, ASCII HAS NOT BEEN CODED YET!

    If paramFile is not provided, the raw array of detections is returned.
    If paramFile is provded, the detections are contained in a dictionary of numpy arrays,
    with keys as names of the entries in the paramaters file. For example:
    catalog['XWIN_IMAGE']=[x_array]
    catalog['YWIN_IMAGE']=[y_array]
    .
    .
    .
    """

    if type=='FITS_LDAC':
        han=pyf.open(catalogName,ignore_missing_end=True)
        data=han[2].data
        han.close()

        #print data[1]
        #print len(data[1])
        #sys.exit()
        if paramFile==None and pars==[]:
            return data

        else:
            indices=[]
            han=open(paramFile)
            d=han.readlines()
            han.close()

            filePars=[]
            for ii in range(len(d)):
                if '#' in d[ii]: continue
                s=d[ii].split()

                if len(s)==0: continue
                else:
                    filePars.append(s[0])
                    indices.append(len(indices))

            catalog={}
            for i in filePars:
                catalog[i]=[]

            for ii in range(len(data)):
                for jj in range(len(indices)):
                    catalog[filePars[jj]].append(data[ii][indices[jj]])
            for ii in filePars:
                catalog[ii]=num.array(catalog[ii])
                if 'MAG_APER' in ii and ii!='MAG_APER':
                    catalog['MAG_APER']=num.array(catalog[ii])
                    del catalog[ii]
            return catalog





def updateHeader(fileNameBase,overWrite=True):
    """
    Update the provided fits filename with the scamp header created from scamp.runScamp()

    If overWrite is true as is default, the header entries will be updated in the file directly.
    If cowardly, overWrite=False, a new file will be created with 's' prepended to the name.
    """

    print('Updating header of image %s.'%(fileNameBase))
    #update the headers and setup new images
    handle=open(fileNameBase+'.head')
    H=handle.readlines()
    handle.close()


    scampHead=[['FRASER','COMMENT','WCS generated using scamp.py Module'],['FRASER1','COMMENT','All Below are .head scamp output']]
    for j in range(3,len(H)-1):
        key=H[j].split('= ')[0]
        s2=H[j].split('= ')[1].split('/')[0]
        if "'" in s2:
            entry=s2.split("'")[1]
        else:
            entry=s2
        comment=H[j].split(' / ')[1].split('  ')[0]

        #print key.strip(),entry,comment

        try:
            if ('NAN' not in entry) and ('INF' not in entry) and ('-INF' not in entry):
                scampHead.append([key.strip(),float(entry),comment])
            else:
                entry=0.0
        except:
            scampHead.append([key.strip(),entry,comment])
        #print scampHead[len(scampHead)-1]

    han=pyf.open(fileNameBase+'.fits')
    header=han[0].header


    data=han[0].data
    han.close()
    for j in range(len(scampHead)):
        if scampHead[j][0] in header:
            header[scampHead[j][0]]=scampHead[j][1]
        else:
            header.set(scampHead[j][0],scampHead[j][1],scampHead[j][2])

    try:
        os.remove('s'+fileNameBase+'.fits')
    except:
        pass

    HDU=pyf.PrimaryHDU(data,header)
    List=pyf.HDUList([HDU])
    List.writeto('s'+fileNameBase+'.fits')


    raRms=han[0].header['ASTRRMS1']*3600
    decRms=han[0].header['ASTRRMS2']*3600

    han.close()

    print('done.')
    return (raRms,decRms)



def writeDS9Regions(sexCatalog,regionFile,wcsImage=None,radius=15,colour=None):
    """
    Utility to write ds9 regions of your sectractor detections.
    """

    if not path.isfile(sexCatalog):
        print("I can't seem to find the sexCatalog file %s.\n"%(sexCatalog))
        raise

    print(sexCatalog)
    binTable=True
    try:
        han=pyf.open(sexCatalog)
    except:
        binTable=False

    keys=[]
    if binTable:
        header=han[2].header
        data=han[2].data
        han.close()

        if not wcsImage:
            k=-1
            for ii in header:
                if 'TTYPE' in ii:
                    k+=1
                if header[ii]=='XWIN_IMAGE':
                    keys.append(k)
                    break
            k=-1
            for ii in header:
                if 'TTYPE' in ii:
                    k+=1
                if header[ii]=='YWIN_IMAGE':
                    keys.append(k)
                    break

            if len(keys)!=2:
                print("Cannot find XWIN_IMAGE,YWIN_IMAGE")
                raise


            k=-1
            for ii in header:
                if 'TTYPE' in ii:
                    k+=1
                if header[ii]=='FLAGS':
                    keys.append(k)
                    break


            regions=[]
            for ii in range(len(data)):
                x=data[ii][keys[0]]
                y=data[ii][keys[1]]

                f=''
                if len(keys)>2:
                    f=data[ii][keys[2]]
                regions.append([x,y,f])

            han=open(regionFile,'w+')
            for ii in range(len(regions)):
                [x,y,f]=regions[ii]
                region='circle(%s, %s, %s)'%(x,y,radius)
                if colour:
                    region+=' # color='+colour
                if f!=0:
                    region+=' text={'+str(f)+'}'
                print(region,file=han)
            han.close()


        else: #using the wcs of the input image
            with pyf.open(wcsImage) as wcsHan:
                wcsHeader=wcsHan[0].header
            wcs=WCS.WCS(wcsHeader)

            k=-1
            for ii in header:
                if 'TTYPE' in ii:
                    k+=1
                if header[ii]=='X_WORLD':
                    keys.append(k)
                    break
            k=-1
            for ii in header:
                if 'TTYPE' in ii:
                    k+=1
                if header[ii]=='Y_WORLD':
                    keys.append(k)
                    break

            if len(keys)!=2:
                print("Cannot find X_WORLD,Y_WORLD")
                raise


            k=-1
            for ii in header:
                if 'TTYPE' in ii:
                    k+=1
                if header[ii]=='FLAGS':
                    keys.append(k)
                    break


            regions=[]
            for ii in range(len(data)):
                ra=data[ii][keys[0]]
                dec=data[ii][keys[1]]

                (x,y)=wcs.wcs_world2pix(ra,dec)
                x+=1
                y+=1

                f=''
                if len(keys)>2:
                    f=data[ii][keys[2]]
                regions.append([x,y,f])

            han=open(regionFile,'w+')
            for ii in range(len(regions)):
                [x,y,f]=regions[ii]
                region='circle(%s, %s, %s)'%(x,y,radius)
                if colour:
                    region+=' # color='+colour
                if f!=0:
                    region+=' text={'+str(f)+'}'
                print(region,file=han)
            han.close()



    return
