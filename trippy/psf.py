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
__version__= '0.1'

import numpy as num
from scipy import signal
import sys,os
import pylab as pyl
from scipy import optimize as opti,interpolate as interp
import scipy as sci
import bgFinder
import emcee
import weightedMeanSTD

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


def lnprob(r,dat,lims,psf,ue,useLinePSF,verbose=False):
    psf.nForFitting+=1
    (x,y,amp)=r
    (a,b)=dat.shape
    if  amp<=0 or x>=b or x<=0 or y<=0 or y>=a: return -num.inf
    diff=psf.remove(x,y,amp,dat,useLinePSF=useLinePSF)[lims[0]:lims[1],lims[2]:lims[3]]
    chi=-0.5*num.sum(diff**2/ue**2)**0.5
    if verbose: print psf.nForFitting,x,y,amp,chi
    return chi


def lnprobDouble(r,dat,psf,ue,useLinePSF,verbose=False):
    psf.nForFitting+=1
    (A,B)=dat.shape
    (X,Y,AMP,x,y,amp)=r
    if amp<=0 or AMP<=0 or X<0 or X>B or x<0 or x>B or Y<0 or Y>A or y<0 or y>A: return -num.inf

    diff=psf.remove(X,Y,AMP,dat,useLinePSF=useLinePSF)
    diff=psf.remove(x,y,amp,diff,useLinePSF=useLinePSF)
    chi=-0.5*num.sum((diff**2/ue**2))#/(A*B-7)
    #chi=-num.sum(diff**2)**0.5
    if verbose:
        print psf.nForFitting,x,y,amp,X,Y,AMP,chi
    return chi



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
        name=fn.split('.fits')[0]

        HDU=pyf.PrimaryHDU(self.PSF)
        hdu=pyf.ImageHDU(self.psf)
        lookupHDU=pyf.ImageHDU(self.lookupTable)
        lineHDU=pyf.ImageHDU(self.longPSF)
        if self.aperCorrs<>None:
            aperCorrHDU=pyf.ImageHDU(num.array([self.aperCorrs,self.aperCorrRadii]))
        else:
            aperCorrHDU=pyf.ImageHDU(num.array([[-1],[-1]]))
        if self.lineAperCorrs<>None:
            lineAperCorrHDU=pyf.ImageHDU(num.array([self.lineAperCorrs,self.lineAperCorrRadii]))
        else:
            lineAperCorrHDU=pyf.ImageHDU(num.array([[-1],[-1]]))
        distHDU=pyf.ImageHDU(num.array([self.rDist,self.fDist]))
        list=pyf.HDUList([HDU,lookupHDU,lineHDU,hdu,aperCorrHDU,lineAperCorrHDU,distHDU])


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
        list.writeto(name+'.fits',clobber=True)

    def _fitsReStore(self,fn):
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
            self.rDist=inHan[6].data[0]
            self.fDist=inHan[6].data[1]

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


        #now recompute the necessary parameters
        self.aperCorrFunc=interp.interp1d(self.aperCorrRadii*1.,self.aperCorrs*1.)
        self.lineAperCorrFunc=interp.interp1d(self.lineAperCorrRadii*1.,self.lineAperCorrs*1.)

        self.shape=self.psf.shape
        self.x=num.arange(self.shape[0])+0.5
        self.y=num.arange(self.shape[1])+0.5

        self.cent=num.array([len(self.y)/2.,len(self.x)/2.])
        self.centx=self.cent[0]
        self.centy=self.cent[1]

        self.psf=num.ones([len(y),len(x)]).astype('float')

        self.inds=num.zeros((len(y),len(x),2)).astype('int')
        for ii in range(len(y)):
            self.inds[ii,:,1]=num.arange(len(x))
        for ii in range(len(x)):
            self.inds[:,ii,0]=num.arange(len(y))

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
        self.nForFitting=0
        self.imData=None

        if restore:
            self._fitsReStore(restore)

        else:
            self.A=None
            self.alpha=alpha
            self.beta=beta
            self.chi=None


            self.x=x*1.0+0.5
            self.y=y*1.0+0.5
            self.cent=num.array([len(y)/2.,len(x)/2.])
            self.centx=self.cent[0]
            self.centy=self.cent[1]
            self.repFact=repFact

            self.psf=num.ones([len(y),len(x)]).astype('float')

            self.inds=num.zeros((len(y),len(x),2)).astype('int')
            for ii in range(len(y)):
                self.inds[ii,:,1]=num.arange(len(x))
            for ii in range(len(x)):
                self.inds[:,ii,0]=num.arange(len(y))

            self.coords=self.inds+num.array([0.5,0.5])
            self.r=num.sqrt(num.sum((self.coords-self.cent)**2,axis=2))


            self.X=num.arange(len(x)*self.repFact)/float(self.repFact)+0.5/self.repFact
            self.Y=num.arange(len(y)*self.repFact)/float(self.repFact)+0.5/self.repFact
            self.Inds=num.zeros((len(y)*self.repFact,len(x)*self.repFact,2)).astype('int')
            for ii in range(len(y)*self.repFact):
                self.Inds[ii,:,1]=num.arange(len(x)*self.repFact)
            for ii in range(len(x)*self.repFact):
                self.Inds[:,ii,0]=num.arange(len(y)*self.repFact)
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
            self.rDist=None
            self.fDist=None

            self.line2d=None
            self.longPSF=None
            self.longpsf=None

            self.bgNoise=None


            #from fitting a psf to a source
            self.model=None
            self.residual=None

            self.psfStars=None


    def computeRoundAperCorrFromMoffat(self,radii,useLookupTable=True,display=True,displayAperture=True):
        """
        This computes the aperture correction between the provided aperture
        and the total sum of the psf.
        """
        self.aperCorrRadii=radii*1.0
        aperCorrs=[]

        (A,B)=self.PSF.shape
        if useLookupTable:
            phot=pillPhot(self.fullPSF,repFact=1)
        else:
            phot=pillPhot(self.PSF,repFact=1)
        for iii in range(len(self.aperCorrRadii)):
            r=radii[iii]
            width=A/2#int(A/(r*self.repFact*2)+0.5)*0.75
            phot(B/2.+0.5,A/2.+0.5,radius=r*self.repFact,l=0.,a=0.,skyRadius=None,zpt=0.0,width=width,display=displayAperture)
            m=phot.magnitude
            aperCorrs.append(m)
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
        if self.aperCorrFunc<>None:
            return self.aperCorrFunc(r)-num.min(self.aperCorrs)
        else:
            raise Exception('Must first fun computeRoundAperCorrFromMoffat before the aperture corrections can be evaluated here.')


    def computeLineAperCorrFromMoffat(self,radii,l,a,display=True,displayAperture=True):
        """This computes the aperture correction between the linear aperture
        and the total sum of the psf.
        """
        self.lineAperCorrRadii=radii*1.0
        self.lineAperCorrs=[]

        (A,B)=self.PSF.shape
        phot=pillPhot(self.longPSF,repFact=1)
        for ii in range(len(self.lineAperCorrRadii)):
            r=self.lineAperCorrRadii[ii]
            width=A/2#int(A/(r*self.repFact*2))
            phot(B/2.+0.5,A/2.+0.5,radius=r*self.repFact,l=l*self.repFact,a=a,skyRadius=None,zpt=0.0,width=width,display=displayAperture)
            m=phot.magnitude
            print '   ',r,phot.sourceFlux,m
            self.lineAperCorrs.append(m)
        self.lineAperCorrs=num.array(self.lineAperCorrs)
        self.lineAperCorrFunc=interp.interp1d(self.lineAperCorrRadii,self.lineAperCorrs)
        
        if display:
            fig=pyl.figure('psf')
            pyl.plot(self.lineAperCorrRadii,self.lineAperCorrs,'k-o')
            pyl.xlabel('Aperture Radius (pix')
            pyl.ylabel('Normalized Magnitude')
            pyl.show()



    def lineAperCorr(self,r):

        if self.lineAperCorrFunc<>None:
            return self.lineAperCorrFunc(r)-num.min(self.lineAperCorrs)
        else:
            raise Exception('Must first fun computeLineAperCorrFromMoffat before the aperture corrections can be evaluated here.')


    def moffat(self,rad):
        #normalized flux profile return 1.-(1.+(rad/self.alpha)**2)**(1.-self.beta)
        a2=self.alpha*self.alpha
        return (self.beta-1)*(num.pi*a2)*(1.+(rad/self.alpha)**2)**(-self.beta)

    def FWHM(self):
        if self.fitted:
            r=num.linspace(0,num.max(self.rDist),300)
        else:
            r=num.linspace(0,100,500)
        m=self.moffat(r)
        m/=num.max(m)
        k=num.sum(num.greater(m,0.5))
        if k<0 or k>=len(m): return None
        return r[k]*2.

    
    def __getitem__(self,key):
        return self.psf[key]

    def line(self,rate,angle,dt,pixScale=0.2,display=False,useLookupTable=True):
        """give the angle in degrees,
        rate in arcsec/time unit
        dt in same time unit as used in the rate
        -pixel scale in arcsec/pix
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
            self.longPSF=signal.convolve2d(self.moffProf+self.lookupTable*self.repFact*self.repFact, self.line2d,mode='same')
            self.longPSF*=num.sum(self.fullPSF)/num.sum(self.longPSF)
        else:
            print 'Not using the lookup table when generating the long PSF'
            self.longPSF=signal.convolve2d(self.moffProf,self.line2d,mode='same')
            self.longPSF*=num.sum(self.moffProf)/num.sum(self.longPSF)
        self.psf=downSample2d(self.longPSF,self.repFact)
        
        if display:
            fig=pyl.figure('Line PSF')
            pyl.imshow(self.longPSF,interpolation='nearest',origin='lower')
            pyl.show()



    def plant(self,x,y,amp,indata,addNoise=True,useLinePSF=False,returnModel=False):
        """
        plant a star at coordinates x,y

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

            w=num.where(psf<0)
            psf[w]=0.0 #this is a cheat to handle the outer edges of the lookup table that can get negative values when convolved
            self.fitFluxCorr=1. #HACK! Could get rid of this in the future...

        (a,b)=psf.shape
        if addNoise:
            psf+=sci.randn(a,b)*psf**0.5

        (A,B)=indata.shape
        bigOut=num.zeros((A+2*self.boxSize,B+2*self.boxSize),dtype=indata.dtype)
        bigOut[yint+self.boxSize:yint+3*self.boxSize+1,xint+self.boxSize:xint+3*self.boxSize+1]+=psf

        if returnModel:
            return bigOut[self.boxSize:A+self.boxSize,self.boxSize:B+self.boxSize]

        indata+=bigOut[self.boxSize:A+self.boxSize,self.boxSize:B+self.boxSize]
        return indata


    def remove(self,x,y,amp,data,useLinePSF=False):
        mo=self.plant(x,y,amp,data,addNoise=False,returnModel=True,useLinePSF=useLinePSF)
        self.model=mo*1.
        return data-mo


    def fitWithModelPSF(self,x_in,y_in,imageData,m_in=-1.,fitWidth=20,
                        nWalkers=20,nBurn=10,nStep=20,
                        confidenceRange=0.67,
                        bg=None,
                        useLinePSF=False,verbose=False):

        self.nForFitting=0

        if fitWidth>self.boxSize:
            raise NotImplementedError('Need to keep the fitWidth <= boxSize.')

        (A,B)=imageData.shape
        ai=max(0,int(y_in)-fitWidth)
        bi=min(A,int(y_in)+fitWidth+1)
        ci=max(0,int(x_in)-fitWidth)
        di=min(B,int(x_in)+fitWidth+1)
        dat=num.copy(imageData)



        if bg==None:
            bgf=bgFinder.bgFinder(imageData)
            bg=bgf.smartBackground()

        dat-=bg
        ue=(dat+bg)**0.5

        if m_in==-1.:
            if useLinePSF:
                m_in=self.repFact*self.repFact*num.sum(dat)/num.sum(self.longPSF)
            else:
                m_in=self.repFact*self.repFact*num.sum(dat)/num.sum(self.fullPSF)

        nDim=3
        r0=[]
        for ii in range(nWalkers):
            r0.append(num.array([x_in,y_in,m_in])+sci.randn(3)*num.array([0.1,0.1,m_in*0.25]))
        r0=num.array(r0)


        sampler=emcee.EnsembleSampler(nWalkers,nDim,lnprob,args=[dat,(ai,bi,ci,di),self,ue,useLinePSF,verbose])
        pos, prob, state=sampler.run_mcmc(r0,nBurn)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nStep, rstate0=state)
        samps=sampler.chain
        probs=sampler.lnprobability

        (Y,X)=probs.shape
        goodSamps=[]
        for ii in range(Y):
            for jj in range(X):
                goodSamps.append([samps[ii,jj][0],samps[ii,jj][1],samps[ii,jj][2],probs[ii,jj]])
        goodSamps=num.array(goodSamps)
        args=num.argsort(goodSamps[:,3])
        goodSamps=goodSamps[args]

        (x_out,y_out,m_out,chi)=goodSamps[-1]
        print 'Best point:',x_out,y_out,m_out,chi
        self.residual=self.remove(x_out,y_out,m_out,dat,useLinePSF=True)
        self.fitFlux=num.sum(self.model)*self.fitFluxCorr
        bestFit=(x_out,y_out,m_out,chi)

        uncert=[]
        for ii in range(3):
            args=num.argsort(goodSamps[:,ii])
            x=goodSamps[args][:,ii]
            uncert.append([x[int((confidenceRange/2)*len(x))],
                           x[int((1-confidenceRange/2)*len(x))]])
        return (bestFit,uncert)


    def fitDoubleWithModelPSF(self,x_in,y_in,X_in,Y_in,bRat_in,imageData,m_in=-1.,bg=None,
                              fitWidth=20,nWalkers=30,nBurn=50,nStep=100,
                              confidenceWidth=0.95,
                              useErrorMap=False,
                              useLinePSF=False,verbose=False):
        (A,B)=imageData.shape
        ai=max(0,int((y_in+Y_in)/2)-fitWidth)
        bi=min(A,int((y_in+Y_in)/2)+fitWidth)
        ci=max(0,int((x_in+X_in)/2)-fitWidth)
        di=min(B,int((x_in+X_in)/2)+fitWidth)
        dat=imageData[ai:bi,ci:di]

        if bg==None:

            bgf=bgFinder.bgFinder(imageData)
            bg=bgf.smartBackground()
            dat-=bg
        else:
            dat-=bg


        ue=(dat+bg)**0.5
        if not useErrorMap:
            ue*=0.0
            ue+=1.0


        if m_in==-1.:
            if useLinePSF:
                m_in=num.sum(dat)/num.sum(self.longPSF)
            else:
                m_in=num.sum(dat)/num.sum(self.fullPSF)

        nDim=6
        r0=[]
        for ii in range(nWalkers):
            r0.append(num.array([x_in-ci,y_in-ai,m_in,X_in-ci,Y_in-ai,m_in*bRat_in])+sci.randn(6)*num.array([1.,1.,
                                                                                                          m_in*0.4,
                                                                                                    1.,1.,
                                                                                               m_in*0.4*bRat_in]))
        r0=num.array(r0)

        sampler=emcee.EnsembleSampler(nWalkers,nDim,lnprobDouble,args=[dat,self,ue,useLinePSF,verbose])
        pos, prob, state=sampler.run_mcmc(r0,nBurn)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nStep, rstate0=state)
        samps=sampler.chain
        probs=sampler.lnprobability

        (Y,X)=probs.shape
        goodSamps=[]
        for ii in range(Y):
            for jj in range(X):
                goodSamps.append([samps[ii,jj][0],samps[ii,jj][1],samps[ii,jj][2],samps[ii,jj][3],samps[ii,jj][4],
                                  samps[ii,jj][5],probs[ii,jj]])
        goodSamps=num.array(goodSamps)
        args=num.argsort(goodSamps[:,6])
        goodSamps=goodSamps[args]

        A=len(goodSamps)
        c=(1-confidenceWidth)/2.
        limI=[int(A*c),int(A*(1-c))]
        lims=[]
        for i in range(6):
            x=num.sort(goodSamps[:,i])
            lims.append( [x[limI[0]],x[limI[1]]])

        dx=num.sort(goodSamps[:,0]-goodSamps[:,3])
        dy=num.sort(goodSamps[:,1]-goodSamps[:,4])
        br=num.sort(goodSamps[:,2]/goodSamps[:,5])

        lims.append([dx[limI[0]],dx[limI[1]]])
        lims.append([dy[limI[0]],dy[limI[1]]])
        lims.append([br[limI[0]],br[limI[1]]])
        lims=num.array(lims)

        (x_out,y_out,m_out,X_out,Y_out,M_out,chi)=num.median(goodSamps,axis=0)#[len(goodSamps)-1]
        pars=[x_out+ci,y_out+ai,m_out,X_out+ci,Y_out+ai,M_out,x_out-X_out,y_out-Y_out,m_out/M_out]
        lims[0]+=ci
        lims[1]+=ai
        lims[3]+=ci
        lims[4]+=ai
        return (pars,lims,chi,goodSamps)


    def writeto(self,name):
        try:
            os.remove(name)
        except: pass
        HDU=pyf.PrimaryHDU(self.psf)
        List=pyf.HDUList([HDU])
        List.writeto(name)

    def fitMoffat(self,imData,centX,centY,boxSize=25,bgRadius=20,verbose=False,mode='smart',fixAB=False,fitXY=False,fitMaxRadius=-1.,logRadPlot=False):

        """
        input coordinates are in numpy coordiantes, not iraf.
        and do not need the extra +0.5 added to them!
        """



        self.verbose=verbose

        self.imData=imData*1.0
        #self.centX=centX
        #self.centY=centY
        self.boxSize=boxSize

        self._flatRadial(centX-0.5,centY-0.5)#set the radial distribution pixels

        w=num.where(self.rDist>bgRadius)
        bgf=bgFinder.bgFinder(self.fDist[w])
        self.bg=bgf(method=mode)

        peakGuess=(num.max(self.fDist)-self.bg)/(num.max(self.moffat(self.rDist)))

        if fitXY:
            self.verbose=False
            best=[1.e8,-1.,-1.,-1.]
            print 'Fitting XYA'
            deltaX=num.arange(-0.2,0.2+1./self.repFact,1./self.repFact/2.)
            deltaY=num.arange(-0.2,0.2+1./self.repFact,1./self.repFact/2.)
            for ii in range(len(deltaX)):
                for jj in range(len(deltaY)):
                    self._flatRadial(centX+deltaX[ii],centY+deltaY[jj])
                    lsqf=opti.leastsq(self._residFAB,(peakGuess),args=(self.alpha,self.beta,fitMaxRadius),maxfev=1000)
                    res=num.sum(self._residFAB((lsqf[0][0]),self.alpha,self.beta,fitMaxRadius)**2)
                    #print res,deltaX[ii],deltaY[jj],lsqf[0][0]
                    if best[0]>=res:
                        best=[res,lsqf[0],deltaX[ii],deltaY[jj]]

            return (best[2],best[3])

        elif fixAB:
            lsqf=opti.leastsq(self._residFAB,(peakGuess),args=(self.alpha,self.beta,fitMaxRadius),maxfev=200)
        else:
            lsqf=opti.leastsq(self._resid,(peakGuess,self.alpha,self.beta),args=(fitMaxRadius),maxfev=250)
        if self.verbose: print lsqf
        self.A=lsqf[0][0]
        if not fixAB:
            self.alpha=lsqf[0][1]
            self.beta=lsqf[0][2]
        if fixAB:
            res=self._residFAB((self.A),self.alpha,self.beta,fitMaxRadius)
        else:
            res=self._resid((self.A,self.alpha,self.beta),fitMaxRadius)
        self.chi=num.sqrt(num.sum(res**2)/(len(res)-1))
        self.fitted=True

        self.PSF=self.moffat(self.R)
        self.PSF/=num.sum(self.PSF)
        self.psf=downSample2d(self.PSF,self.repFact)


        if self.verbose:
            print '   A:%s, alpha:%s, beta:%s'%(self.A,self.alpha,self.beta)
            fig=pyl.figure('Radial Profile')
            ax=fig.add_subplot(111)
            pyl.scatter(downSample2d(self.repRads,self.repFact),self.subSec)
            r=num.linspace(0,num.max(self.rDist),100)
            pyl.plot(r,self.A*self.moffat(r)+self.bg,'r--')
            print 'FWHM: %.3f'%(self.FWHM())
            pyl.title('FWHM: %.3f alpha: %.3f beta: %.3f'%(self.FWHM(),self.alpha,self.beta))
            if logRadPlot: ax.set_xscale('log')
            pyl.show()

        return res



    def genLookupTable(self,imData,centXs,centYs,verbose=False,bpMask=None):
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


            xint,yint=int(centXs[ii])-self.boxSize-2,int(centYs[ii])-self.boxSize-2
            cx,cy=centXs[ii]-int(centXs[ii])+self.boxSize+2,centYs[ii]-int(centYs[ii])+self.boxSize+2
            cx+=0.5
            cy+=0.5

            cut=imData[yint:yint+2*self.boxSize+5,xint:xint+2*self.boxSize+5]

            self.fitMoffat(cut,num.array([cx]),num.array([cy]),self.boxSize,verbose=verbose,fixAB=True,fitXY=False,fitMaxRadius=3.)
            self.imData=num.copy(imData) #this is necessary because the imdata gets set to the shifted image subsection
            moff=downSample2d(self.moffat(self.repRads),self.repFact)*self.A


            diff=cut-self.bg
            diff[2:-2,2:-2]-=moff

            #pyl.imshow(diff)
            #pyl.show()
            #sys.exit()

            fluxes.append(self.A)
            self.psfStars[ii].append(self.A)

            repCut=expand2d(diff,self.repFact)

            cx,cy=centXs[ii]-int(centXs[ii])+self.boxSize+2,centYs[ii]-int(centYs[ii])+self.boxSize+2
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

        self.lookupTable=num.mean(shiftIms,axis=0)/self.maxFlux
        self.psfStar=num.array(self.psfStars)

        return None



    def genPSF(self,A=1.):
        """
        generate the psf with lookup table
        """
        self.moffProf=self.moffat(self.R-num.min(self.R))
        self.fullPSF=(self.moffProf+self.lookupTable)*A
        self.fullpsf=downSample2d(self.fullPSF,self.repFact)


    def _flatRadial(self,centX,centY):
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


        subSecFlat=self.subSec.reshape((b-a)*(c-d))
        arrR=num.array(arrR)
        self.rads=num.copy(arrR)
        arrR=arrR.reshape((b-a)*(d-c))
        arg=num.argsort(arrR)
        self.rDist=arrR[arg]*1.
        self.fDist=subSecFlat[arg]*1.

    def _resid(self,p,maxRad):
        (A,alpha,beta)=p
        self.alpha=alpha
        self.beta=beta
        err=(self.subSec-(self.bg+A*downSample2d(self.moffat(self.repRads),self.repFact))).reshape(len(self.fDist))
        #if maxRad>0:
        #    w=num.where(self.rDist<=maxRad)
        #else:
        #    w=num.arange(len(self.rDist))
        #moff=self.moffat(self.rDist[w])
        #err=self.fDist[w]-(self.bg+A*moff)
        if self.alpha<0 or self.beta<0: return num.inf


        if self.verbose: print A,alpha,beta,num.sqrt(num.sum(err**2)/(len(self.fDist)-1.))
        return err

    def _residFAB(self,p,alpha,beta,maxRad):
        (A)=p
        self.alpha=alpha
        self.beta=beta
        err=(self.subSec-(self.bg+A*downSample2d(self.moffat(self.repRads),self.repFact))).reshape(len(self.fDist))
        #if maxRad>0:
        #    w=num.where(self.rDist<=maxRad)
        #else:
        #    w=num.arange(len(self.rDist))
        #err=self.fDist[w]-(self.bg+A*self.moffat(self.rDist[w]))

        if self.verbose: print A,alpha,beta,num.sqrt(num.sum(err**2)/(len(self.fDist)-1.))
        return err


        
def bgselect(event):
    global CA
    print CA.get_xlim()
    print CA.get_ylim()
        


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
