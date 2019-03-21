#! /usr/bin/env python

from __future__ import print_function, division
from collections import namedtuple

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

import numpy as np, scipy as sci,emcee
from trippy import bgFinder
import pickle
from scipy import optimize as opti
import pylab as pyl

def resid(p,cutout,psf,boxWidth = 7,useLinePSF = False,verbose=False):
    (x,y,m) = p
    xt,yt = int(x),int(y)
    (a,b) = cutout.shape
    res = psf.remove(x,y,m,cutout,useLinePSF=useLinePSF)[yt-boxWidth:yt+boxWidth+1,xt-boxWidth:xt+boxWidth+1]
    if verbose:
        print(np.sum(res**2)**0.5,x,y,m)
    return np.array(res).astype('float').reshape((boxWidth*2+1)**2)

def likelihood_for_LS(p,cutout,bg,psf,boxWidth = 7,useLinePSF = False,verbose=False):
    (x,y,m) = p
    res = resid(p,cutout-bg,psf,boxWidth = boxWidth,verbose=verbose,useLinePSF = useLinePSF)
    xt,yt = int(x),int(y)
    ue2 = np.abs(cutout[yt-boxWidth:yt+boxWidth+1,xt-boxWidth:xt+boxWidth+1].reshape((boxWidth*2+1)**2))
    return -0.5*np.sum(res**2/ue2)

def lnprob(r,dat,lims,psf,ue,useLinePSF, verbose=False, other=None):
    """
    can't recall what the purpose of parameter other is. Note to look into this in the future.
    """
    psf.nForFitting+=1
    if other != None:
        (x,y,amp) = other[:]
    if len(r) == 3:
        (x, y, amp) = r
    elif len(r) == 2:
        (x,y) = r
    elif len(r) == 1:
        amp = r[0]
    (a,b)=dat.shape
    if  amp <= 0 or x >= b or x <= 0 or y <= 0 or y >= a: return -np.inf
    diff = psf.remove(x,y,amp,dat,useLinePSF=useLinePSF)[lims[0]:lims[1],lims[2]:lims[3]]
    chi = -0.5*np.sum(diff**2/ue[lims[0]:lims[1],lims[2]:lims[3]]**2)
    if verbose: print('{:6d} {:8.3f} {:8.3f} {:8.3f} {:10.3f}'.format(psf.nForFitting,x,y,amp,chi))
    return chi


def lnprob_varRateAngle(r,dat,lims,psf,ue,useLinePSF, exptime, pixScale, verbose=False):
    """
    for now because I am lazy, rates, exptime, and pixScale should be given in "/hr, seconds, and "/pix
    """
    psf.nForFitting+=1
    (x, y, amp, rate, angle) = r[:]
    (a,b) = dat.shape
    if  amp<=0 or x>=b or x<=0 or y<=0 or y>=a or angle>90 or angle<-90 or rate <0: return -np.inf

    psf.line(rate, angle, exptime/3600., pixScale = pixScale, useLookupTable = True, verbose = False)

    diff = psf.remove(x,y,amp,dat, useLinePSF = useLinePSF)[lims[0]:lims[1],lims[2]:lims[3]]
    chi = -0.5*np.sum(diff**2/ue[lims[0]:lims[1],lims[2]:lims[3]]**2)
    if verbose: print('{:6d} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:10.3f}'.format(psf.nForFitting,x,y,amp,rate,angle,chi))
    return chi


def _lnprob_varRateAngle_LSSTHACK(r,dat,lims,psf,ue,useLinePSF, exptime, pixScale, verbose=False):
    """
    for now because I am lazy, rates, exptime, and pixScale should be given in "/hr, seconds, and "/pix
    """
    dT = 5.0

    psf.nForFitting += 1
    (x, y, amp, rate, angle) = r[:]
    (a,b) = dat.shape
    if  amp<=0 or x>=b or x<=0 or y<=0 or y>=a or angle>90 or angle<-90 or rate <0: return -np.inf

    psf.line(rate, angle, exptime/3600., pixScale = pixScale, useLookupTable = True, verbose = False)

    dl = rate * (exptime / 2.0 + dT/2.0) / 3600.0 / pixScale
    dx = dl * np.cos(angle * np.pi / 180.0)
    dy = dl * np.sin(angle * np.pi / 180.0)

    d = psf.remove(x-dx/2.0,y-dy/2.0,amp,dat,useLinePSF=useLinePSF)
    diff = psf.remove(x+dx/2.0,y+dy/2.0,amp,d,useLinePSF=useLinePSF)[lims[0]:lims[1],lims[2]:lims[3]]

    chi = -0.5*np.sum(diff**2/ue[lims[0]:lims[1],lims[2]:lims[3]]**2)
    if verbose: print('{:6d} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:10.3f}'.format(psf.nForFitting,x,y,amp,rate,angle,chi))
    return chi


def lnprobDouble(r,dat,lims,psf,ue,useLinePSF,verbose=False):
    psf.nForFitting+=1
    (A,B) = dat.shape
    (X,Y,AMP,x,y,amp) = r
    if amp<=0 or AMP<=0 or X<0 or X>B or x<0 or x>B or Y<0 or Y>A or y<0 or y>A: return -np.inf

    diff = psf.remove(X,Y,AMP,dat,useLinePSF=useLinePSF)
    diff = psf.remove(x,y,amp,diff,useLinePSF=useLinePSF)
    chi = -0.5*np.sum((diff**2/ue**2)[lims[0]:lims[1],lims[2]:lims[3]])
    #chi=-np.sum(diff**2)**0.5
    if verbose:
        print('{:6d} {: 8.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: 10.3f}'.format(psf.nForFitting,x,y,amp,X,Y,AMP,chi))
    return chi


class LSfitter(object):

    def __init__(self,psf,imageData):
        """
        Initialize a least squares fitter object which is essentially a wrapper
        around scipy.optimize.leastsq.

        Input:
        psf is the trippy model psf object
        imageData is the data on which the fit will be performed.

        Use restore=fileName to import a dump fit file saved by saveState.

        """
        self.psf = psf
        self.imageData = np.copy(imageData)
        self.fitted = False

    def fitWithModelPSF(self,x_in,y_in,m_in = -1.,fitWidth = 7,
                        bg = None,ftol=1.e-8,
                        useLinePSF = False,
                        verbose=False):
        """
        This is still experimental and hasn't been fully vetted yet.

        Use at your own risk.
        """

        self.useLinePSF = useLinePSF
        if fitWidth>self.psf.boxSize:
            raise NotImplementedError('Need to keep the fitWidth <= boxSize.')

        (A,B) = self.imageData.shape
        #ai = max(0,int(y_in)-fitWidth)
        #bi = min(A,int(y_in)+fitWidth+1)
        #ci = max(0,int(x_in)-fitWidth)
        #di = min(B,int(x_in)+fitWidth+1)
        dat = np.copy(self.imageData)

        if bg == None:
            bgf = bgFinder.bgFinder(self.imageData)
            bg = bgf.smartBackground()
            dmbg = dat - bg
            print('Subtracting background {}'.format(bg))
        else:
            dmbg = dat - bg
        if m_in==-1.:
            if useLinePSF:
                m_in = self.psf.repFact*self.psf.repFact*np.sum(dat)/np.sum(self.psf.longPSF)
            else:
                m_in = self.psf.repFact*self.psf.repFact*np.sum(dat)/np.sum(self.psf.fullPSF)


        lsqf = opti.leastsq(resid,(x_in,y_in,m_in),args=(dmbg,self.psf,fitWidth,useLinePSF,verbose),maxfev=1000,ftol=ftol)
        fitPars = lsqf[0]
        l = likelihood_for_LS(fitPars,dat,bg,self.psf,boxWidth = fitWidth,useLinePSF=useLinePSF)
        fitPars = np.concatenate([fitPars,np.array([l])])

        return fitPars


class MCMCfitter:

    def __init__(self,psf,imageData,restore=False):
        """
        Initialize a fitter object which essentially wraps convenience around the already convenient emcee code.

        Input:
        psf is the trippy model psf object
        imageData is the data on which the fit will be performed.

        Use restore=fileName to import a dump fit file saved by saveState.

        """
        self.psf = psf
        self.imageData = np.copy(imageData)
        self.fitted = False

        if restore:
            self._restoreState(restore)

    def fitWithModelPSF(self,x_in,y_in,m_in = -1.,fitWidth = 20,
                        nWalkers = 20,nBurn = 10,nStep = 20,
                        bg = None, useErrorMap = False,
                        useLinePSF = False, fitRateAngle = False,
                        rate_in = None, angle_in = None, exptime = None, pixScale = None,
                        verbose=False, rand_pos = 0.1):

        """
        Using emcee (It's hammer time!) the provided image is fit using
        the provided psf to find the best x,y and amplitude, and confidence
        range on the fitted parameters.

        x_in, y_in, m_in - initial guesses on the true centroid and amplitude of the object
        fitWidth - the width +- of x_in/y_in of the data used in the fit
        nWalkers, nBurn, nStep - emcee fitting paramters. If you don't know what these are RTFM
        bg - the background of the image. Needed for the uncertainty table.
             **Defaults to None.** When set to default, it will invoke
             the background measurement and apply that. Otherwise, it assumes you are dealing with
             background subtracted data already.
        useErrorMap - if true, a simple pixel uncertainty map is used in the fit. This is adopted as
                      ue_ij=(imageData_ij+bg)**0.5, that is, the poisson noise estimate. Note the fit confidence range
                      is only honest if useErrorMap=True.
        useLinePSF - use the TSF? If not, use the PSF
        verbose - if set to true, lots of information printed to screen
        """

        print("Initializing sampler")

        self.nForFitting = 0
        self.useLinePSF = useLinePSF

        if fitWidth>self.psf.boxSize:
            raise NotImplementedError('Need to keep the fitWidth <= boxSize.')

        (A,B) = self.imageData.shape
        ai = max(0,int(y_in)-fitWidth)
        bi = min(A,int(y_in)+fitWidth+1)
        ci = max(0,int(x_in)-fitWidth)
        di = min(B,int(x_in)+fitWidth+1)
        dat = np.copy(self.imageData)



        if bg == None:
            bgf = bgFinder.bgFinder(self.imageData)
            bg = bgf.smartBackground()
            dat -= bg
            print('Subtracting background {}'.format(bg))

        if not useErrorMap:
            ue = dat*0.0+1.
        else:
            ue = (dat+bg)**0.5

        self.fitted = True


        if m_in==-1.:
            if useLinePSF:
                m_in = self.psf.repFact*self.psf.repFact*np.sum(dat)/np.sum(self.psf.longPSF)
            else:
                m_in = self.psf.repFact*self.psf.repFact*np.sum(dat)/np.sum(self.psf.fullPSF)

        if not fitRateAngle:

            nDim = 2
            r0 = []
            for ii in range(nWalkers):
                r0.append(np.array([x_in,y_in])+sci.randn(2)*np.array([rand_pos,rand_pos]))
            r0 = np.array(r0)

            #fit first using input best guess amplitude
            sampler = emcee.EnsembleSampler(nWalkers,nDim,lnprob,args=[dat,(ai,bi,ci,di),self.psf,ue,useLinePSF,verbose,(-1,-1,m_in)])
            print("Executing xy burn-in... this may take a while.")
            pos, prob, state = sampler.run_mcmc(r0, nBurn)#, 10)
            sampler.reset()
            print("Executing xy production run... this will also take a while.")
            pos, prob, state = sampler.run_mcmc(pos, nStep, rstate0=state)
            self.samps = sampler.chain
            self.probs = sampler.lnprobability
            self.dat = np.copy(dat)

            out = self.fitResults()
            (x,y,junk) = out[0]
            dx = (out[1][0][1] - out[1][0][0])/2.0
            dy = (out[1][1][1] - out[1][1][0])/2.0

            nDim = 1 # need to put two here rather than one because the fitresults code does a residual subtraction
            r0 = []
            for ii in range(nWalkers):
                r0.append(np.array([m_in]) + sci.randn(1) * np.array([m_in*0.25]))
            r0 = np.array(r0)


            #now fit the amplitude using the best-fit x,y from above
            #could probably cut the nBurn and nStep numbers down by a factor of 2
            sampler = emcee.EnsembleSampler(nWalkers, nDim, lnprob,
                                            args=[dat, (ai, bi, ci, di), self.psf, ue, useLinePSF, verbose, (x, y, -1)])
            print("Executing amplitude burn-in... this may take a while.")
            pos, prob, state = sampler.run_mcmc(r0, max(int(nBurn/2),10))
            sampler.reset()
            print("Executing amplitude production run... this will also take a while.")
            pos, prob, state = sampler.run_mcmc(pos, max(int(nStep/2),10), rstate0=state)
            self.samps = sampler.chain
            self.probs = sampler.lnprobability
            self.dat = np.copy(dat)

            out = self.fitResults()
            amp = out[0][0]
            damp = (out[1][0][1]-out[1][0][0])/2.0


            #now do a full 3D fit using a small number of burn and steps.
            nDim=3
            r0=[]
            for ii in range(nWalkers):
                r0.append(np.array([x,y,amp])+sci.randn(3)*np.array([dx,dy,damp]))
            r0=np.array(r0)


            sampler=emcee.EnsembleSampler(nWalkers,nDim,lnprob,args=[dat,(ai,bi,ci,di),self.psf,ue,useLinePSF,verbose])
            print("Executing xy-amp burn-in... this may take a while.")
            pos, prob, state=sampler.run_mcmc(r0,nBurn)
            sampler.reset()
            print("Executing xy-amp production run... this will also take a while.")
            pos, prob, state = sampler.run_mcmc(pos, nStep, rstate0=state)

        else:

            nDim = 5
            r0 = []
            for ii in range(nWalkers):
                r0.append(np.array([x_in, y_in, m_in, rate_in, angle_in]) + sci.randn(nDim) * np.array([0.1, 0.1, m_in*0.1, rate_in*0.1, 4.0]))
            r0 = np.array(r0)

            #fit first using input best guess amplitude
            sampler = emcee.EnsembleSampler(nWalkers, nDim, lnprob_varRateAngle_LSSTHACK,
                                          args=[dat, (ai,bi,ci,di), self.psf, ue, useLinePSF, exptime, pixScale, verbose])
            #dat,lims,psf,ue,useLinePSF, exptime, pixScale, verbose=False):
            print("Executing burn-in... this may take a while.")
            pos, prob, state = sampler.run_mcmc(r0, nBurn, 10)
            sampler.reset()
            print("Executing production run... this will also take a while.")
            pos, prob, state = sampler.run_mcmc(pos, nStep, rstate0=state)
            self.samps = sampler.chain
            self.probs = sampler.lnprobability
            self.dat = np.copy(dat)

        self.samps = sampler.chain
        self.probs = sampler.lnprobability
        self.dat = np.copy(dat)

    def fitResults(self, confidenceRange=0.67, returnSamples=False):
        """
        Return the best point and confidence interval.

        confidenceRange - the range for the returned confidence interval

        Returns (bestPoint, confidenceArray) Will return None if a fit hasn't been run yet.

        If the fit is a binary fit (6 parameters) fitRange will have a 10 elements which is the range of uncertain on
        the brightness ratio, and the Primary-Secondary separation in x/y and total (in pixels).
        """

        if not self.fitted:
            print("You haven't actually run a fit yet!")
            return None

        (Y,X,b) = self.samps.shape
        goodSamps=[]
        for ii in range(Y):
            for jj in range(X):
                g = []
                for kk in range(b):
                    g.append(self.samps[ii,jj][kk])
                g.append(self.probs[ii,jj])
                goodSamps.append(g)
        goodSamps = np.array(goodSamps)
        args = np.argsort(goodSamps[:,b])
        goodSamps = goodSamps[args]

        bp = goodSamps[-1]
        print('Best point:',bp)
        if b == 3 or b == 6:
            self.residual = self.psf.remove(bp[0],bp[1],bp[2],self.dat,useLinePSF=self.useLinePSF)
        elif b == 6:
            self.residual = self.psf.remove(bp[3],bp[4],bp[5],self.residual,useLinePSF=self.useLinePSF)
        elif b == 1:
            self.residual = None
        self.fitFlux = np.sum(self.psf.model)*self.psf.fitFluxCorr

        uncert=[]
        for ii in range(b):
            args = np.argsort(goodSamps[:,ii])
            x = goodSamps[args][:,ii]
            a = int(len(x)*(1-confidenceRange)/2)
            b = int(1-a)
            uncert.append([x[int(a)],
                           x[int(b)]])

        if len(uncert)==6:
            x=np.sort(goodSamps[:,5]/goodSamps[:,2])
            a=int(len(x)*(1-confidenceRange)/2)
            b=int(1-a)
            uncert.append([x[int(a)],
                           x[int(b)]])
            x = np.sort(goodSamps[:, 0] - goodSamps[:, 3])
            uncert.append([x[int(a)],
                           x[int(b)]])
            x = np.sort(goodSamps[:, 1] - goodSamps[:, 4])
            uncert.append([x[int(a)],
                           x[int(b)]])
            x = np.sort(((goodSamps[:, 1] - goodSamps[:, 4])**2+(goodSamps[:, 0] - goodSamps[:, 3])**2)**0.5)
            uncert.append([x[int(a)],
                           x[int(b)]])
        if not returnSamples: return (bp, uncert)
        return (bp, uncert, goodSamps)


    def fitDoubleWithModelPSF(self,x_in,y_in,X_in,Y_in,bRat_in,m_in=-1.,bg=None,
                              fitWidth=20,nWalkers=30,nBurn=50,nStep=100,
                              useErrorMap=False,
                              useLinePSF=False,verbose=False):
        """
        Using emcee (It's hammer time!) two sources are fit using
        the provided psf to find the best x,y and amplitude, and confidence
        range on the fitted parameters.

        x_in, y_in, m_in, X_in, Y_in, bRat - initial guesses on the true centroids, the amplitude and brightness ratio
                                             of the two sources
        fitWidth - the width +- of x_in/y_in of the data used in the fit
        nWalkers, nBurn, nStep - emcee fitting paramters. If you don't know what these are RTFM
        bg - the background of the image. Needed for the uncertainty table.
             **Defaults to None.** When set to default, it will invoke
             the background measurement and apply that. Otherwise, it assumes you are dealing with
             background subtracted data already.
        useErrorMap - if true, a simple pixel uncertainty map is used in the fit. This is adopted as
                      ue_ij=(imageData_ij+bg)**0.5, that is, the poisson noise estimate. Note the fit confidence range
                      is only honest if useErrorMap=True.
        useLinePSF - use the TSF? If not, use the PSF
        verbose - if set to true, lots of information printed to screen
        """


        self.useLinePSF = useLinePSF

        (A,B) = self.imageData.shape
        ai = max(0,int((y_in+Y_in)/2)-fitWidth)
        bi = min(A,int((y_in+Y_in)/2)+fitWidth+1)
        ci = max(0,int((x_in+X_in)/2)-fitWidth)
        di = min(B,int((x_in+X_in)/2)+fitWidth+1)
        dat = np.copy(self.imageData)


        if bg==None:
            bgf = bgFinder.bgFinder(self.imageData)
            bg = bgf.smartBackground()
            dat -= bg


        if not useErrorMap:
            ue = dat*0.0+1.
        else:
            ue = (dat+bg)**0.5



        if m_in == -1.:
            if useLinePSF:
                m_in = np.sum(dat)/np.sum(self.psf.longPSF)
            else:
                m_in = np.sum(dat)/np.sum(self.psf.fullPSF)

        nDim=6
        r0=[]
        for ii in range(nWalkers):
            r0.append(np.array([x_in,y_in,m_in,X_in,Y_in,m_in*bRat_in])+sci.randn(6)*np.array([1.,1.,
                                                                                                          m_in*0.4,
                                                                                                    1.,1.,
                                                                                               m_in*0.4*bRat_in]))
        r0=np.array(r0)

        sampler = emcee.EnsembleSampler(nWalkers,nDim,lnprobDouble,args=[dat,(ai,bi,ci,di),self.psf,ue,self.useLinePSF,verbose])
        pos, prob, state = sampler.run_mcmc(r0,nBurn)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nStep, rstate0=state)
        self.samps = sampler.chain
        self.probs = sampler.lnprobability
        self.dat = np.copy(dat)
        self.fitted = True

    def saveState(self, fn = 'MCState.pickle'):
        """
        Save the fitted state to the provided filename.
        """
        if not self.fitted: raise Exception('You must run a fit before you can save the fit results.')
        with open(fn,'w+') as han:
            pickle.dump([self.samps,self.probs,self.dat,self.useLinePSF],han)

    def _restoreState(self,fn='MCState.pickle'):
        with open(fn) as han:
            (self.samps,self.probs,self.dat,self.useLinePSF)=pickle.load(han)
        self.fitted=True
