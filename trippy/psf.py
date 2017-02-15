#! /usr/bin/env python

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

import numpy as num
from scipy import signal
import sys,os
import pylab as pyl
from scipy import optimize as opti,interpolate as interp
import scipy as sci
import bgFinder
#import weightedMeanSTD

import imp
try:
    imp.find_module('astropy')
    astropyFound = True
except ImportError:
    astropyFound = False
if astropyFound:
    from astropy.io import fits as pyf
else:
    import pyfits as pyf

from pill import pillPhot

from trippy_utils import *





class modelPSF:
    """
    Round and moving object psf class.

    The intent of this class is to provide a model PSF of both stationary and trailed point sources.
    The basic profile is a moffat profile with super sampled constant look up table. Both are used
    with linear convolution to calculate the trailed source PSF.

    modelPSF takes as input:
    -x,y are arrays of length equal to the width and height of the desired PSF image.
     eg. x=numpy.arange(50), y=numpy.arange(70) would create a psf 50x70 pix
    -alpha, beta are initial guesses for the moffat profile to be used. 5 and 2 are usually pretty good
    -repFact is the supersampling factor. 10 is default, though for improvement in speed, 5 can be used
     without much loss of PSF or photometric precision

    optional arguments:
    -verbose = True if you want to see a lot of unnecessary output
    -restore = psf filename if you want to restore from a previously saved psf file.

    The general steps for psf generation and photometry are:
    -initialization
    -lookup table generation
    -psf generation
    -line convolution
    -linear aperture correction estimation


    """

    def psfStore(self,fn):
        """
        Store the psf into a fits file that you can view and reopen at a later point. The only option is the fits file
        name.
        """
        name=fn.split('.fits')[0]

        HDU=pyf.PrimaryHDU(self.PSF)
        hdu=pyf.ImageHDU(self.psf)
        lookupHDU=pyf.ImageHDU(self.lookupTable)
        lineHDU=pyf.ImageHDU(self.longPSF)

        if self.aperCorrs is not None:
            aperCorrHDU=pyf.ImageHDU(num.array([self.aperCorrs,self.aperCorrRadii]))
        else:
            aperCorrHDU=pyf.ImageHDU(num.array([[-1],[-1]]))
        if self.lineAperCorrs is not None:
            lineAperCorrHDU=pyf.ImageHDU(num.array([self.lineAperCorrs,self.lineAperCorrRadii]))
        else:
            lineAperCorrHDU=pyf.ImageHDU(num.array([[-1],[-1]]))
        #distHDU=pyf.ImageHDU(num.array([self.rDist,self.fDist]))
        list=pyf.HDUList([HDU,lookupHDU,lineHDU,hdu,aperCorrHDU,lineAperCorrHDU])


        list[0].header.set('REPFACT',self.repFact)
        for ii in range(len(self.psfStars)):
            list[0].header.set('xSTAR%s'%(ii),self.psfStars[ii][0],'PSF Star x value.')
            list[0].header.set('ySTAR%s'%(ii),self.psfStars[ii][1],'PSF Star y value.')
        list[0].header['alpha']=self.alpha
        list[0].header['beta']=self.beta
        list[0].header['A']=self.A
        list[0].header['rate']=self.rate
        list[0].header['angle']=self.angle
        list[0].header['dt']=self.dt
        list[0].header['pixScale']=self.pixScale
        list.writeto(name + '.fits', clobber = True)

    def _fitsReStore(self,fn):
        """
        Hidden convenience function to restore a psf file.
        """
        print '\nRestoring PSF...'
        name=fn.split('.fits')[0]
        with pyf.open(name+'.fits') as inHan:

            #load the psf file
            self.PSF=inHan[0].data
            self.lookupTable=inHan[1].data
            self.longPSF=inHan[2].data
            self.psf=inHan[3].data
            self.aperCorrs=inHan[4].data[0]
            self.aperCorrRadii=inHan[4].data[1]
            self.lineAperCorrs=inHan[5].data[0]
            self.lineAperCorrRadii=inHan[5].data[1]
            #self.rDist=inHan[6].data[0]
            #self.fDist=inHan[6].data[1]

            self.psfStars=[]

            header=inHan[0].header
            self.repFact=header['REPFACT']

            x=header['xSTAR*'].values()
            y=header['ySTAR*'].values()
            for ii in range(len(x)):
                self.psfStars.append([x[ii],y[ii]])
            self.alpha=header['alpha']
            self.beta=header['beta']
            self.A=header['A']
            self.rate=header['RATE']
            self.angle=header['ANGLE']
            self.dt=header['DT']
            self.pixScale=header['PIXSCALE']


        self.boxSize=len(self.lookupTable)/self.repFact/2

        #now recompute the necessary parameters
        if len(self.aperCorrs)<>1:
            self.aperCorrFunc=interp.interp1d(self.aperCorrRadii*1.,self.aperCorrs*1.)
        if len(self.lineAperCorrs)<>1:
            self.lineAperCorrFunc=interp.interp1d(self.lineAperCorrRadii*1.,self.lineAperCorrs*1.)

        self.shape=self.psf.shape
        self.x=num.arange(self.shape[0])+0.5
        self.y=num.arange(self.shape[1])+0.5

        self.cent=num.array([len(self.y)/2.,len(self.x)/2.])
        self.centx=self.cent[0]
        self.centy=self.cent[1]

        self.psf=num.ones([len(self.y),len(self.x)]).astype('float')

        self.inds=num.zeros((len(self.y),len(self.x),2)).astype('int')
        for ii in range(len(self.y)):
            self.inds[ii,:,1]=num.arange(len(self.x))
        for ii in range(len(self.x)):
            self.inds[:,ii,0]=num.arange(len(self.y))

        self.coords=self.inds+num.array([0.5,0.5])
        self.r=num.sqrt(num.sum((self.coords-self.cent)**2,axis=2))



        self.X=num.arange(len(self.x)*self.repFact)/float(self.repFact)+0.5/self.repFact
        self.Y=num.arange(len(self.y)*self.repFact)/float(self.repFact)+0.5/self.repFact
        self.Inds=num.zeros((len(self.y)*self.repFact,len(self.x)*self.repFact,2)).astype('int')
        for ii in range(len(self.y)*self.repFact):
            self.Inds[ii,:,1]=num.arange(len(self.x)*self.repFact)
        for ii in range(len(self.x)*self.repFact):
            self.Inds[:,ii,0]=num.arange(len(self.y)*self.repFact)
        self.Coords=(self.Inds+num.array([0.5,0.5]))/float(self.repFact)

        self.R=num.sqrt(num.sum((self.Coords-self.cent)**2,axis=2))
        self.genPSF()
        self.fitted=True


        print '   PSF restored.\n'


    def __init__(self,x=-1,y=-1,alpha=-1,beta=-1,repFact=10,verbose=False,restore=False):
        """
        Initialize the PSF.

        x,y are the size of the PSF (width, height) in pixels. Can either be an integer value or a numpy.arange(x) array.
        alpha, beta are the initial moffat parameters
        repfact=5,10 is the supersampling factor. Only 5 and 10 are well tested!
        verbose to see a bunch of unnecessary, but informative output.
        restore=filename to restore a psf having the filename provided.
        """

        self.nForFitting=0
        self.imData=None

        if repFact not in [5,10]:
            raise Warning('This has only been robustly tested with repFact=5 or 10. I encourage you to stick with those.')

        if not restore:
            if type(x)==type(num.ones(1)):
                if len(x)==1:
                    if x[0]%2==0 or x[0]%2==0:
                        raise Exception('Please use odd width PSFs. Even has not been tested yet.')
                elif (len(x)%2==0 or len(y)%2==0):
                    raise Exception('Please use odd width PSFs. Even has not been tested yet.')
            else:
                if (x%2==0 or y%2==0):
                    raise Exception('Please use odd width PSFs. Even has not been tested yet.')

        if restore:
            self._fitsReStore(restore)

        else:
            self.A=None
            self.alpha=alpha
            self.beta=beta
            self.chi=None

            if type(x)<>type(num.ones(1)):
                self.x=num.arange(x)+0.5
                self.y=num.arange(y)+0.5
            elif len(x)==1:
                self.x=num.arange(x)+0.5
                self.y=num.arange(y)+0.5
            else:
                self.x=x*1.0+0.5
                self.y=y*1.0+0.5
            self.cent=num.array([len(self.y)/2.,len(self.x)/2.])
            self.centx=self.cent[0]
            self.centy=self.cent[1]
            self.repFact=repFact

            self.psf=num.ones([len(self.y),len(self.x)]).astype('float')

            self.inds=num.zeros((len(self.y),len(self.x),2)).astype('int')
            for ii in range(len(self.y)):
                self.inds[ii,:,1]=num.arange(len(self.x))
            for ii in range(len(self.x)):
                self.inds[:,ii,0]=num.arange(len(self.y))

            self.coords=self.inds+num.array([0.5,0.5])
            self.r=num.sqrt(num.sum((self.coords-self.cent)**2,axis=2))


            self.X=num.arange(len(self.x)*self.repFact)/float(self.repFact)+0.5/self.repFact
            self.Y=num.arange(len(self.y)*self.repFact)/float(self.repFact)+0.5/self.repFact
            self.Inds=num.zeros((len(self.y)*self.repFact,len(self.x)*self.repFact,2)).astype('int')
            for ii in range(len(self.y)*self.repFact):
                self.Inds[ii,:,1]=num.arange(len(self.x)*self.repFact)
            for ii in range(len(self.x)*self.repFact):
                self.Inds[:,ii,0]=num.arange(len(self.y)*self.repFact)
            self.Coords=(self.Inds+num.array([0.5,0.5]))/float(self.repFact)

            self.R=num.sqrt(num.sum((self.Coords-self.cent)**2,axis=2))


            self.PSF=self.moffat(self.R)
            self.PSF/=num.sum(self.PSF)
            self.psf=downSample2d(self.PSF,self.repFact)

            self.fullPSF=None
            self.fullpsf=None


            self.shape=self.psf.shape

            self.aperCorrFunc=None
            self.aperCorrs=None
            self.aperCorrRadii=None
            self.lineAperCorrFunc=None
            self.lineAperCorrs=None
            self.lineAperCorrRadii=None

            self.verbose=verbose
            self.fitted=False

            self.lookupTable=None
            self.lookupF=None
            self.lookupR=None
            #self.rDist=None
            #self.fDist=None

            self.line2d=None
            self.longPSF=None
            self.longpsf=None

            self.bgNoise=None


            #from fitting a psf to a source
            self.model=None
            self.residual=None

            self.psfStars=None


    def computeRoundAperCorrFromPSF(self,radii,useLookupTable=True,display=True,displayAperture=True):
        """
        This computes the aperture correction directly from the PSF. These vaules will be used for interpolation to
        other values. The aperture correction is with respect tothe largest aperture provided in radii. I recommend
        4*FWHM.

        radii is an array of radii on which to calculate the aperture corrections. I recommend at least 10 values
        between 1 and 4 FWHM.
        useLookupTable=True/False to calculate either with just the moffat profile, or with lookuptable included.
        display=True to show you some plots.
        displayAperture=True to show you the aperture at each radius.
        """

        self.aperCorrRadii=radii*1.0
        aperCorrs=[]

        (A,B)=self.PSF.shape
        if useLookupTable:
            phot=pillPhot(self.fullPSF,repFact=1)
        else:
            phot=pillPhot(self.PSF,repFact=1)

        """
        #old individual radii call version
        for iii in range(len(self.aperCorrRadii)):
            r=radii[iii]
            width=A/2#int(A/(r*self.repFact*2)+0.5)*0.75
            phot(B/2.,A/2.,radius=r*self.repFact,l=0.,a=0.,skyRadius=None,zpt=0.0,width=width,display=displayAperture)
            m=phot.magnitude
            aperCorrs.append(m)
        """

        #more efficient version with all radii passed at once.
        width=A/2
        phot(B / 2., A / 2., radius=radii * self.repFact, l=0., a=0., skyRadius=None, zpt=0.0, width=width,
             display=displayAperture)
        aperCorrs = phot.magnitude

        self.aperCorrs=num.array(aperCorrs)
        self.aperCorrFunc=interp.interp1d(self.aperCorrRadii*1.,self.aperCorrs*1.)

        if display:
            fig=pyl.figure('psf')
            pyl.plot(self.aperCorrRadii,self.aperCorrs,'k-o')
            pyl.xlabel('Aperture Radius (pix')
            pyl.ylabel('Normalized Magnitude')
            pyl.show()
        #still need to implement this!


    def roundAperCorr(self,r):
        """
        Return an aperture correction at given radius. Linear interpolation between values found in
        computeRoundAperCorrFromPSF is used.
        """

        if self.aperCorrFunc<>None:
            return self.aperCorrFunc(r)-num.min(self.aperCorrs)
        else:
            raise Exception('Must first fun computeRoundAperCorrFromPSF before the aperture corrections can be evaluated here.')


    def computeLineAperCorrFromTSF(self,radii,l,a,display=True,displayAperture=True):
        """
        This computes the aperture correction directly from the TSF. These vaules will be used for interpolation to
        other values. The aperture correction is with respect tothe largest aperture provided in radii. I recommend
        4*FWHM.

        radii is an array of radii on which to calculate the aperture corrections. I recommend at least 10 values
        between 1 and 4 FWHM.
        l and a are the length (in pixels) and angle of the pill aperture
        useLookupTable=True/False to calculate either with just the moffat profile, or with lookuptable included.
        display=True to show you some plots.
        displayAperture=True to show you the aperture at each radius.
        """

        self.lineAperCorrRadii=radii*1.0
        self.lineAperCorrs=[]

        (A,B)=self.PSF.shape
        phot=pillPhot(self.longPSF,repFact=1)

        """
        #old version where all radii are passed individually
        for ii in range(len(self.lineAperCorrRadii)):
            r=self.lineAperCorrRadii[ii]
            width=A/2#int(A/(r*self.repFact*2))
            phot(B/2.,A/2.,radius=r*self.repFact,l=l*self.repFact,a=a,skyRadius=None,zpt=0.0,width=width,display=displayAperture)
            m=phot.magnitude
            print '   ',r,phot.sourceFlux,m
            self.lineAperCorrs.append(m)
        """

        #new version where all radii are passed at once
        width = A / 2
        phot(B / 2., A / 2., radius=radii * self.repFact, l=l * self.repFact, a=a, skyRadius=None, zpt=0.0, width=width,
             display=displayAperture)
        fluxes = phot.sourceFlux
        self.lineAperCorrs = phot.magnitude
        print "    Radius  Flux      Magnitude"
        for ii in range(len(self.lineAperCorrRadii)):
            print '    {:6.2f} {:10.3f}  {:8.3f}'.format(radii[ii],phot.sourceFlux[ii],phot.magnitude[ii])

        self.lineAperCorrs=num.array(self.lineAperCorrs)
        self.lineAperCorrFunc=interp.interp1d(self.lineAperCorrRadii,self.lineAperCorrs)

        if display:
            fig=pyl.figure('psf')
            pyl.plot(self.lineAperCorrRadii,self.lineAperCorrs,'k-o')
            pyl.xlabel('Aperture Radius (pix')
            pyl.ylabel('Normalized Magnitude')
            pyl.show()



    def lineAperCorr(self,r):
        """
        Return an aperture correction at given radius. Linear interpolation between values found in
        computeRoundAperCorrFromTSF is used.
        """

        if self.lineAperCorrFunc<>None:
            return self.lineAperCorrFunc(r)-num.min(self.lineAperCorrs)
        else:
            raise Exception('Must first fun computeLineAperCorrFromMoffat before the aperture corrections can be evaluated here.')


    def moffat(self,rad):
        """
        Return a moffat profile evaluated at the radii in the input numpy array.
        """

        #normalized flux profile return 1.-(1.+(rad/self.alpha)**2)**(1.-self.beta)
        a2=self.alpha*self.alpha
        return (self.beta-1)*(num.pi*a2)*(1.+(rad/self.alpha)**2)**(-self.beta)

    def FWHM(self,fromMoffatProfile=False):
        """
        Return the moffat profile of the PSF. If fromMoffatProfile=True, or if the lookupTable is not yet calculated,
        the FWHM from a pure moffat profile is returned. Otherwise the lookup table is used.
        """

        if (not self.fitted) or fromMoffatProfile:
            r=num.arange(0,(2*max(self.x.shape[0]/2.,self.y.shape[0]/2.)**2)**0.5,0.005)
            m=self.moffat(r)
            m/=num.max(m)
            k=num.sum(num.greater(m,0.5))
            if k<0 or k>=len(m): return None
            return r[k]*2.
        else:
            a=self.y.shape[0]/2.
            b=self.x.shape[0]/2.
            rangeY=num.arange(-a*self.repFact,a*self.repFact)/float(self.repFact)
            rangeX=num.arange(-b*self.repFact,b*self.repFact)/float(self.repFact)
            dx2=(0.5/self.repFact-rangeX)**2
            repRads=[]
            for ii in range(len(rangeY)):
                repRads.append((0.5/self.repFact-rangeY[ii])**2+dx2)
            repRads=num.array(repRads)**0.5

            r=0.

            s=num.sum(self.fullPSF)
            while r<num.max(repRads):
                if num.sum(self.fullPSF[num.where(repRads<r)])>=s*0.5:
                    return r*2.
                r+=0.01


    def __getitem__(self,key):
        return self.psf[key]

    def line(self,rate,angle,dt,pixScale=0.2,display=False,useLookupTable=True):
        """
        Compute the TSF given input rate of motion, angle of motion, length of exposure, and pixelScale.

        Units choice is irrelevant, as long as they are all the same! eg. rate in "/hr, and dt in hr.
        Angle is in degrees +-90 from horizontal.

        display=True to see the TSF

        useLookupTable=True to use the lookupTable. OTherwise pure moffat is used.
        """

        self.rate=rate
        self.angle=angle
        self.dt=dt
        self.pixScale=pixScale

        angr=angle*num.pi/180.


        self.line2d=self.PSF*0.0
        w=num.where(( num.abs(self.X-self.centx)<num.cos(angr)*rate*dt/pixScale/2.))
        if len(w[0])>0:
            x=self.X[w]*1.0
            y=num.tan(angr)*(x-self.centx)+self.centy
            X=(x*self.repFact).astype('int')
            Y=(y*self.repFact).astype('int')
            self.line2d[Y,X]=1.0

            w=num.where(self.line2d>0)
            yl,yh=num.min(w[0]),num.max(w[0])
            xl,xh=num.min(w[1]),num.max(w[1])

            self.line2d=self.line2d[yl:yh+1,xl:xh+1]

        else:
            self.line2d=num.array([[1.0]])

        if useLookupTable:
            print 'Using the lookup table when generating the long PSF.'
            #self.longPSF=signal.convolve2d(self.moffProf+self.lookupTable*self.repFact*self.repFact, self.line2d,mode='same')
            self.longPSF=signal.fftconvolve(self.moffProf+self.lookupTable*self.repFact*self.repFact, self.line2d,mode='same')
            self.longPSF*=num.sum(self.fullPSF)/num.sum(self.longPSF)
        else:
            print 'Not using the lookup table when generating the long PSF'
            #self.longPSF=signal.convolve2d(self.moffProf,self.line2d,mode='same')
            self.longPSF=signal.fftconvolve(self.moffProf,self.line2d,mode='same')
            self.longPSF*=num.sum(self.moffProf)/num.sum(self.longPSF)
        self.psf=downSample2d(self.longPSF,self.repFact)

        if display:
            fig=pyl.figure('Line PSF')
            pyl.imshow(self.longPSF,interpolation='nearest',origin='lower')
            pyl.show()



    def plant(self,x,y,amp,indata,addNoise=True,useLinePSF=False,returnModel=False):
        """
        Plant a star at coordinates x,y with amplitude amp.

        indata is the array in which you want to plant the source.
        addNoise=True to add gaussian noise.
        useLinePSF=True to use the TSF rather than the circular PSF.
        returnModel=True to not actually plant in the data, but return an array of the same size containing the TSF or
        PSF without noise added.

        """



        self.boxSize=len(self.lookupTable)/self.repFact/2

        xint,yint=int(x)-self.boxSize,int(y)-self.boxSize
        cx,cy=x-int(x)+self.boxSize,y-int(y)+self.boxSize
        sx,sy=int(round((x-int(x))*self.repFact)),int(round((y-int(y))*self.repFact))
        cut=num.copy(indata[yint:yint+2*self.boxSize+1,xint:xint+2*self.boxSize+1])

        #repCut=expand2d(cut,self.repFact)

        if self.imData<>None:
            origData=num.copy(self.imData)
        else: origData=None

        self.imData=cut
        if type(cx)==type(1.0):
            self._flatRadial(num.array([cx]),num.array([cy]))
        else:
            self._flatRadial(cx,cy)
        if origData<>None:
            self.imData=origData

        if not useLinePSF:

            moff=downSample2d(self.moffat(self.repRads),self.repFact)*amp
            (pa,pb)=moff.shape

            #shift the lookuptable right and up to account for the off-zero centroid
            slu=num.copy(self.lookupTable)
            (a,b)=slu.shape

            if sx>0:
                sec=slu[:,b-sx:]
                slu[:,sx:]=slu[:,:b-sx]
                slu[:,:sx]=sec
            if sy>0:
                sec=slu[a-sy:,:]
                slu[sy:,:]=slu[:a-sy,:]
                slu[:sy,:]=sec
            slu=downSample2d(slu,self.repFact)*amp*self.repFact*self.repFact
            psf=slu+moff
        else:
            lpsf=num.copy(self.longPSF)
            (a,b)=lpsf.shape

            #cubic interpolation doesn't do as good as the x10 subsampling
            #quintic does just about as well, linear sucks
            #f=sci.interpolate.interp2d(self.dx,self.dy,downSample2d(lpsf,self.repFact),kind='linear')
            #psf=f(self.dx-float(sx)/self.repFact,self.dy-float(sy)/self.repFact)*amp


            if sx>0:
                sec=lpsf[:,b-sx:]
                lpsf[:,sx:]=lpsf[:,:b-sx]
                lpsf[:,:sx]=sec
            if sy>0:
                sec=lpsf[a-sy:,:]
                lpsf[sy:,:]=lpsf[:a-sy,:]
                lpsf[:sy,:]=sec
            psf=downSample2d(lpsf,self.repFact)*amp

            #this is a cheat to handle the outer edges that can go negative after convolution
            w=num.where(psf<0)
            psf[w]=0.0

        self.fitFluxCorr=1. #HACK! Could get rid of this in the future...

        (a,b)=psf.shape
        if addNoise:
            psf+=sci.randn(a,b)*psf**0.5

        (A,B)=indata.shape
        bigOut=num.zeros((A+2*self.boxSize,B+2*self.boxSize),dtype=indata.dtype)
        bigOut[yint+self.boxSize:yint+3*self.boxSize+1,xint+self.boxSize:xint+3*self.boxSize+1]+=psf

        if returnModel:
            return bigOut[self.boxSize:A+self.boxSize,self.boxSize:B+self.boxSize]

        indata+=bigOut[self.boxSize:A+self.boxSize, self.boxSize:B+self.boxSize]
        return indata


    def remove(self,x,y,amp,data,useLinePSF=False):
        """
        The opposite of plant.
        """

        mo = self.plant(x,y,amp,data,addNoise=False,returnModel=True,useLinePSF=useLinePSF)
        self.model = mo*1.
        return data-mo


    def writeto(self,name):
        """
        Convenient file saving function to save the round PSF. Probably not necessary.

        """

        try:
            os.remove(name)
        except: pass
        HDU=pyf.PrimaryHDU(self.psf)
        List=pyf.HDUList([HDU])
        List.writeto(name)

    def fitMoffat(self,imData,centX,centY,
                  boxSize=25,bgRadius=20,
                  verbose=False,mode='smart',fixAB=False,
                  fitXY=False,fitMaxRadius=-1.,logRadPlot=False):

        """
        Fit a moffat profile to the input data, imData, at point centX,centY.

        - boxSize is the width around the centre used in the fitting.
        - bgRadius is the radius beyond which the background is estimated.
        - verbose=True to see a lot of fittnig output and a radial plot of each fit.
        - logRadPlot=True to see the plot in log radius.
        - mode='smart' is the background determination method used. See bgFinder for details.
        - fixAB=True to fit only the amplitude.
        - fitXY=False *** this is currently not implemented***
        - fitMaxRadius ***not currently implemented***
        """



        self.verbose = verbose

        self.imData = imData*1.0
        self.boxSize = boxSize

        self._flatRadial(centX-0.5,centY-0.5)#set the radial distribution pixels

        #w = num.where(self.rDist>bgRadius)
        #bgf = bgFinder.bgFinder(self.fDist[w])
        w = num.where(self.rads>bgRadius)
        bgf = bgFinder.bgFinder(self.subSec[w])
        self.bg = bgf(method=mode)

        #peakGuess = (num.max(self.fDist)-self.bg)/(num.max(self.moffat(self.rDist)))
        peakGuess = (num.max(self.subSec)-self.bg)/(num.max(self.moffat(self.rads)))

        if fitXY:
            print 'This is hacky and really slow. Not really meant for production.'
            self.verbose = False
            best = [1.e8,-1.,-1.,-1.]
            print 'Fitting XYA'
            deltaX = num.arange(-0.2,0.2+1./self.repFact,1./self.repFact/2.)
            deltaY = num.arange(-0.2,0.2+1./self.repFact,1./self.repFact/2.)
            for ii in range(len(deltaX)):
                for jj in range(len(deltaY)):
                    self._flatRadial(centX+deltaX[ii],centY+deltaY[jj])
                    lsqf = opti.leastsq(self._residFAB,(peakGuess),args=(self.alpha,self.beta,fitMaxRadius),maxfev=1000)
                    res = num.sum(self._residFAB((lsqf[0][0]),self.alpha,self.beta,fitMaxRadius)**2)
                    if best[0]>= res:
                        best = [res,lsqf[0],deltaX[ii],deltaY[jj]]

            return (best[2],best[3])

        elif fixAB:
            lsqf = opti.leastsq(self._residFAB,(peakGuess),args=(self.alpha,self.beta,fitMaxRadius),maxfev=200)
        else:
            lsqf = opti.leastsq(self._resid,(peakGuess,self.alpha,self.beta),args=(fitMaxRadius),maxfev=250)
        if self.verbose: print lsqf
        self.A = lsqf[0][0]
        if not fixAB:
            self.alpha = lsqf[0][1]
            self.beta = lsqf[0][2]
        if fixAB:
            res=self._residFAB((self.A),self.alpha,self.beta,fitMaxRadius)
        else:
            res=self._resid((self.A,self.alpha,self.beta),fitMaxRadius)
        self.chi = num.sqrt(num.sum(res**2)/(len(res)-1))
        self.fitted = True

        self.PSF = self.moffat(self.R)
        self.PSF /= num.sum(self.PSF)
        self.psf = downSample2d(self.PSF,self.repFact)


        if self.verbose:
            print '   A:%s, alpha:%s, beta:%s'%(self.A,self.alpha,self.beta)
            fig = pyl.figure('Radial Profile')
            ax = fig.add_subplot(111)
            pyl.scatter(downSample2d(self.repRads,self.repFact),self.subSec)
            r = num.linspace(0,num.max(self.rads),100)
            pyl.plot(r,self.A*self.moffat(r)+self.bg,'r--')
            fw = self.FWHM(fromMoffatProfile=True)
            print 'FWHM: %.3f'%(fw)
            pyl.title('FWHM: %.3f alpha: %.3f beta: %.3f'%(fw,self.alpha,self.beta))
            if logRadPlot: ax.set_xscale('log')
            pyl.show()

        return res



    def genLookupTable(self,imData,centXs,centYs,verbose=False,bpMask=None,threeSigCut=True,bgRadius=20.):
        """
        Generate the lookup table from input imData and x/y coordinates in the numpy arrays centX,centY.

        verbose=True to see a lot of fitting output.
        bpMask=array to provide a bad pixel mask.
        threeSigCut=True to apply a 3 sigma cut before reporting the mean lookupTable. Only useful for ~5 or more stars.
        """

        adjCentXs=centXs-0.5
        adjCentYs=centYs-0.5

        self.verbose=verbose

        self.imData=imData*1.0
        self.boxSize=len(self.R[0])/self.repFact/2


        self.psfStars=[]

        if bpMask<>None:
            w=num.where(bpMask==0)
            imData[w]=num.median(imData)

        shiftIms=[]
        fluxes=[]
        for ii in range(len(centXs)):

            #store the psf star location
            self.psfStars.append([centXs[ii],centYs[ii]])


            xint,yint=int(adjCentXs[ii])-self.boxSize-2,int(adjCentYs[ii])-self.boxSize-2
            cx,cy=adjCentXs[ii]-int(adjCentXs[ii])+self.boxSize+2,adjCentYs[ii]-int(adjCentYs[ii])+self.boxSize+2
            cx+=0.5
            cy+=0.5

            cut=imData[yint:yint+2*self.boxSize+5,xint:xint+2*self.boxSize+5]

            self.fitMoffat(cut,num.array([cx]),num.array([cy]),self.boxSize,verbose=verbose,fixAB=True,fitXY=False,fitMaxRadius=3.,bgRadius=bgRadius)
            self.imData=num.copy(imData) #this is necessary because the imdata gets set to the shifted image subsection
            moff=downSample2d(self.moffat(self.repRads),self.repFact)*self.A


            diff=cut-self.bg
            diff[2:-2,2:-2]-=moff

            fluxes.append(self.A)
            self.psfStars[ii].append(self.A)

            repCut=expand2d(diff,self.repFact)

            cx,cy=adjCentXs[ii]-int(adjCentXs[ii])+self.boxSize+2,adjCentYs[ii]-int(adjCentYs[ii])+self.boxSize+2
            kx,ky=int(round(cx*self.repFact)),int(round(cy*self.repFact))


            shiftedImage=repCut[ky-self.repFact*self.boxSize:ky+self.repFact*self.boxSize+self.repFact,
                         kx-self.repFact*self.boxSize:kx+self.repFact*self.boxSize+self.repFact]

            shiftIms.append(shiftedImage)
        shiftIms=num.array(shiftIms)
        fluxes=num.array(fluxes)

        self.maxFlux=1.0
        invFluxes=self.maxFlux/fluxes

        for ii in range(len(shiftIms)):
            shiftIms[ii]*=invFluxes[ii]

        if threeSigCut:
            meanLUT=num.median(shiftIms,axis=0)
            stdLUT=num.std(shiftIms,axis=0)

            bigMean=num.repeat(num.array([meanLUT]),len(shiftIms),axis=0)
            w=num.where( num.abs(bigMean-shiftIms)>3*stdLUT)
            shiftIms[w]=num.nan
            self.lookupTable=num.nanmean(shiftIms,axis=0)/self.maxFlux
        else:
            self.lookupTable=num.mean(shiftIms,axis=0)/self.maxFlux
        self.psfStar=num.array(self.psfStars)

        self.genPSF()
        return None



    def genPSF(self,A=1.):
        """
        generate the psf with lookup table. Convenience function only.
        """
        self.moffProf=self.moffat(self.R-num.min(self.R))
        self.fullPSF=(self.moffProf+self.lookupTable)*A
        self.fullpsf=downSample2d(self.fullPSF,self.repFact)


    def _flatRadial(self,centX,centY):
        """
        Convenience function for the fitMoffat routines.
        """

        if type(centX)<>type(1.) and type(centX)<>type(num.float64(1.)):
            centX=centX[0]
            centY=centY[0]
        (A,B)=self.imData.shape
        a=int(max(0,centY-self.boxSize))
        b=int(min(A,centY+self.boxSize+1))
        c=int(max(0,centX-self.boxSize))
        d=int(min(B,centX+self.boxSize+1))

        self.subSec=self.imData[a:b,c:d]
        self.repSubsec=expand2d(self.subSec,self.repFact)




        rangeY=num.arange(a*self.repFact,b*self.repFact)/float(self.repFact)
        rangeX=num.arange(c*self.repFact,d*self.repFact)/float(self.repFact)
        dx2=(centX-rangeX)**2
        repRads=[]
        for ii in range(len(rangeY)):
            repRads.append((centY-rangeY[ii])**2+dx2)
        self.repRads=num.array(repRads)**0.5


        self.dX=centX-rangeX
        self.dY=centY-rangeY
        self.dx=centX-num.arange(c,d)
        self.dy=centY-num.arange(a,b)
        #there are more efficient ways to do this, but I leave it like this for clarity.
        subSec=[]
        arrR=[]
        for ii in range(a,b):
            arrR.append([])
            for jj in range(c,d):
                D=((centY-ii)**2+(centX-jj)**2)**0.5

                arrR[-1].append(D)


        #subSecFlat=self.subSec.reshape((b-a)*(c-d))

        arrR=num.array(arrR)
        self.rads=num.copy(arrR)
        #arrR=arrR.reshape((b-a)*(d-c))
        #arg=num.argsort(arrR)
        #self.rDist=arrR[arg]*1.
        #self.fDist=subSecFlat[arg]*1.

    def _resid(self,p,maxRad):
        (A,alpha,beta)=p
        self.alpha=alpha
        self.beta=beta
        err=(self.subSec-(self.bg+A*downSample2d(self.moffat(self.repRads),self.repFact))).reshape(self.subSec.size)
        #if maxRad>0:
        #    w=num.where(self.rDist<=maxRad)
        #else:
        #    w=num.arange(len(self.rDist))
        #moff=self.moffat(self.rDist[w])
        #err=self.fDist[w]-(self.bg+A*moff)
        if self.alpha<0 or self.beta<0: return num.inf


        if self.verbose: print A,alpha,beta,num.sqrt(num.sum(err**2)/(self.subSec.size-1.))
        return err

    def _residFAB(self,p,alpha,beta,maxRad):
        (A)=p
        self.alpha=alpha
        self.beta=beta
        err=(self.subSec-(self.bg+A*downSample2d(self.moffat(self.repRads),self.repFact))).reshape(self.subSec.size)
        #if maxRad>0:
        #    w=num.where(self.rDist<=maxRad)
        #else:
        #    w=num.arange(len(self.rDist))
        #err=self.fDist[w]-(self.bg+A*self.moffat(self.rDist[w]))

        if self.verbose: print A,alpha,beta,num.sqrt(num.sum(err**2)/(self.subSec.size-1.))
        return err






if __name__=="__main__":

    import pylab as pyl
    psfNoLine=modelPSF(num.arange(25),num.arange(25),alpha=1.5,beta=2.0,repFact=10)
    psfNoLine.writeto('noline.fits')
    print
    psfLine=modelPSF(num.arange(25),num.arange(25),alpha=1.5,beta=2.0,repFact=10)
    psfLine.line(4.0,32.,0.45)
    psfLine.writeto('line.fits')
    sys.exit()
    (A,B)=psf.shape
    for i in range(A/2,A/2+1):
        pyl.plot(psf.x,psf.psf[i,:])
    for i in range(A*10/2,A*10/2+1):
        pyl.plot(psf.X,psf.PSF[i,:],linestyle=':')
    pyl.show()
