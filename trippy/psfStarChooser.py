
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


import numpy as np
import pylab as pyl
from astropy.visualization import interval

from . import psf, tzscale, bgFinder

class starChooser:
    """
    Helper class for psf.py to choose good psf stars. Basically, it will provide a window in which
    you can eliminate bad stars, or not point sources.

    This function will fit all input sources with moffat profiles, and then popup a 5 panel window.
    The top right panel is chi^2 of the fits versus FWHM in pix, with historgrams of both on the sides.
    Bottom left is the radial profiles of the fits. When one left-clicks on a point, it will show the measured radial
    profile of the selected source, as well as a cutout of that source.

    **To mark it bad, right click. Red marked sources are excluded from the fitting.**

    Further sources can be exlcuded by zooming the main panel. Only the  blue sources within the zoom window are considered.

    It takes as input:
    -data: the image data loaded up through pyfits or astropy.io.fits
    -XWIN_IMAGE, YWIN_IMAGE, FLUX_AUTO, FLUXERR_AUTO are numpy arrays of the x,y coordinates, flux, and flux uncertainty of a user
     provided list of candidate stars. I used the names of sextractor parameters because these can be easily extracted
     from sextractor output usig those names.
    -zscaleNsamp and zscaleContrast control the zscaling of the display.

    The actual chooser is launched with a self call to the initialized object. The call takes as parameters:
    - radius in pixels for the moffat fitting
    - the minimum SNR of the sources considered in the psf fitting
    - inital alpha and beta for the fitter starting point
    - repfact, the subssampling factor for the moffat fitting
    - xWidth, yWidth are the widths of the star cutouts displayed
    - minGoodVal is used FOR DISPLAY PURPOSES ONLY. If set to a real number, it flags the zscaling algorithm to ignore pixel values below that value.
      This is useful for images that contain large regions or edges of bad data, for example marked with -32768. eg. suprime-cam data.
    - cheesy saturation cut is probably something you don't want to use.


    Returned is an array containing the fitted parameters of all good stars, the median parameters, and the standard
    deviations of those parameters. Basically these should be fed directly to the PSF generation routine.

    the x and y coordinates of the good fit stars are indices 4 and 5 of the goodFits array.
    The median array is [FWHM, chi, alpha, beta]


    Use noVisualSelection=False to get returned the median FWHM, alpha, and beta of the sample

    run as example:
    starChooser=psfStarChooser.starChooser(x,y,f,e)
    (goodStars,meds,stds)=starChooser(40, 5. 2.)
    """

    def __init__(self,data,XWIN_IMAGE,YWIN_IMAGE,FLUX_AUTO,FLUXERR_AUTO,zscaleNsamp=200,zscaleContrast=1.,minGoodVal=None):
        self.XWIN_IMAGE=XWIN_IMAGE
        self.YWIN_IMAGE=YWIN_IMAGE
        self.FLUX_AUTO=FLUX_AUTO
        self.FLUXERR_AUTO=FLUXERR_AUTO
        self.data=data
        self.minGoodVal = minGoodVal

        if self.minGoodVal is not None:
            mask = np.zeros(self.data.shape)
            w = np.where(self.data<self.minGoodVal)
            mask[w] = 1
            (self.z1,self.z2)=tzscale.zscale(self.data,nsamples=zscaleNsamp,contrast=zscaleContrast,bpmask=mask)
        else:
            (self.z1,self.z2)=tzscale.zscale(self.data,nsamples=zscaleNsamp,contrast=zscaleContrast)
        self.normer=interval.ManualInterval(self.z1,self.z2)

        self._increment = 0

    def __call__(self,moffatWidth,moffatSNR,initAlpha=5.,
                 initBeta=2.,repFact=3,xWidth=51,yWidth=51,
                 bgRadius = 20.0,ftol=1.49012e-8,quickFit=True,
                 autoTrim=False,noVisualSelection=False,verbose=False):

        self.moffatWidth=moffatWidth
        self.moffatSNR=moffatSNR
        self.initAlpha=initAlpha
        self.initBeta=initBeta
        self.repFact=repFact
        self.bgRadius = bgRadius

        self.fwhms = []
        self.points = []
        self.moffs = []
        self.moffr = np.linspace(0,self.moffatWidth,self.moffatWidth*3)
        self.starsFlatR = []
        self.starsFlatF = []
        self.subsecs = []
        self.goodStars = []
        self.starsScat = None

        print('Fitting stars with moffat profiles...')
        print('      X         Y    chi    a     b    FWHM')

        for j in range(len(self.XWIN_IMAGE)):
            if self.FLUX_AUTO[j]/self.FLUXERR_AUTO[j]>self.moffatSNR:
                if self.XWIN_IMAGE[j]-1-(moffatWidth+1)<0 or self.XWIN_IMAGE[j]-1+(moffatWidth+1)>=self.data.shape[1] or self.YWIN_IMAGE[j]-1-(moffatWidth+1)<0 or self.YWIN_IMAGE[j]-1+(moffatWidth+1)>=self.data.shape[0]:
                    continue
                #this psf object creator has to be here. It's to do with the way PSF objects are initialized. Some time
                #in the future I will adjust this for a small speed boost.
                mpsf = psf.modelPSF(np.arange(xWidth), np.arange(yWidth), alpha=self.initAlpha, beta=self.initBeta,
                                    repFact=self.repFact)
                mpsf.fitMoffat(self.data, self.XWIN_IMAGE[j], self.YWIN_IMAGE[j],
                               boxSize=self.moffatWidth, verbose=verbose, bgRadius = self.bgRadius,
                               ftol=1.49e-8, quickFit=quickFit)

                fwhm = mpsf.FWHM(fromMoffatProfile=True)

                #norm=Im.normalise(mpsf.subsec,[self.z1,self.z2])
                if self.minGoodVal is not None:

                    norm=self.normer(np.clip(mpsf.subSec,self.minGoodVal,np.max(mpsf.subSec)))
                else:
                    norm=self.normer(mpsf.subSec)
                if (fwhm is not None) and not (np.isnan(mpsf.beta) or np.isnan(mpsf.alpha)):
                    print('   {: 8.2f} {: 8.2f} {:3.2f} {: 5.2f} {: 5.2f} {: 5.2f} '.format(self.XWIN_IMAGE[j],self.YWIN_IMAGE[j],mpsf.chiFluxNorm,mpsf.alpha,mpsf.beta,fwhm))

                    self.subsecs.append(norm*1.)
                    self.goodStars.append(True)

                    self.moffs.append(mpsf.moffat(self.moffr)*1.)

                    self.starsFlatR.append(psf.downSample2d(mpsf.repRads,mpsf.repFact))
                    self.starsFlatF.append((mpsf.subSec-mpsf.bg)/(mpsf.moffat(0)*mpsf.A))

                    self.moffs[len(self.moffs)-1] /= np.max(self.moffs[len(self.moffs)-1])

                    self.points.append([fwhm,mpsf.chi,mpsf.alpha,mpsf.beta,self.XWIN_IMAGE[j],self.YWIN_IMAGE[j],mpsf.bg])
        self.points = np.array(self.points)
        self.goodStars = np.array(self.goodStars)

        FWHM_mode_width = 1.0 #could be 0.5 just as well
        ab_std_width = 2.0
        if autoTrim:
            print('\nDoing auto star selection.')
            #first use Frasermode on the distribution of FWHM to get a good handle on the true FWHM of stars
            #select only those stars with FWHM of +-1 pixel of the mode.
            bg = bgFinder.bgFinder(self.points[:,0])
            mode = bg('median')
            w = np.where(np.abs(self.points[:,0]-mode)>FWHM_mode_width)
            self.goodStars[w] = False

            #now mean select on both moffat a and b parameters to get only those with common shapes
            w = np.where(self.goodStars)
            mean_a = np.mean(self.points[w][:,2])
            std_a = np.std(self.points[w][:,2])
            mean_b = np.mean(self.points[w][:,3])
            std_b = np.std(self.points[w][:,3])

            w = np.where( (np.abs(self.points[:,0]-mode)>FWHM_mode_width) | (np.abs(self.points[:,2]-mean_a)>ab_std_width*std_a) | (np.abs(self.points[:,3]-mean_b)>ab_std_width*std_b) )
            self.goodStars[w] = False

            #now eliminate the ones with sources nearby
            for ii in range(len(self.goodStars)):
                dist = ((self.XWIN_IMAGE-self.points[ii,4])**2 + (self.YWIN_IMAGE-self.points[ii,5])**2)**0.5
                args = np.argsort(dist)
                dist = dist[args]
                #print self.points[ii,4:6],dist
                if dist[1]<moffatWidth:
                    self.goodStars[ii] = False

            """

            #now print out the number of pixels that are 3-sigma above the naive expectation of the moffat profile itself
            mpsf = psf.modelPSF(np.arange(xWidth), np.arange(yWidth), alpha=mean_a, beta=mean_b,
                                repFact=self.repFact)
            print 'Fitting amplitudes to each good star.'
            for ii in range(len(self.moffs)):
                if self.goodStars[ii]:
                    mpsf.fitMoffat(self.data,
                                   self.XWIN_IMAGE[ii], self.YWIN_IMAGE[ii],
                                   boxSize = self.moffatWidth,
                                   fixAB = True)
                    a,b = int(self.YWIN_IMAGE[ii]-yWidth/2),int(self.YWIN_IMAGE[ii]+yWidth/2)+1
                    c,d = int(self.XWIN_IMAGE[ii]-xWidth/2),int(self.XWIN_IMAGE[ii]+xWidth/2)+1
                    x,y = self.XWIN_IMAGE[ii]-int(self.XWIN_IMAGE[ii])+xWidth/2+1.0, self.YWIN_IMAGE[ii]-int(self.YWIN_IMAGE[ii])+yWidth/2+1.0
                    cut = self.data[a:b,c:d]

                    removed = mpsf.remove(self.XWIN_IMAGE[ii]-0.5,self.YWIN_IMAGE[ii]-0.5, mpsf.A, self.data)[a:b,c:d]-mpsf.bg
                    print  ii,len(np.where( np.abs(removed/cut**0.5)>3)[0])

            exit()
            """


        self.figPSF = pyl.figure('Point Source Selector')
        self.sp1 = pyl.subplot2grid((4,4),(0,1),colspan=3,rowspan=2)
        pyl.scatter(self.points[:,0],self.points[:,1],picker=True)
        pyl.title('Select PSF range with zoom,\n' +
                  'inspect stars (a = next, d = previous, w = toggle on/off,\n' +
                  'or left/right click to show/toggle),\n' +
                  'then close the plot window.')
        self.sp2 = pyl.subplot2grid((4,4),(2,1),colspan=3,sharex=self.sp1,rowspan=1)
        bins = np.arange(np.min(self.points[:,0]),np.max(self.points[:,0])+0.5,0.5)
        pyl.hist(self.points[:,0],bins=bins)
        pyl.xlabel('FWHM (pix)')
        self.sp3 = pyl.subplot2grid((4,4),(0,0),rowspan=2,sharey=self.sp1)
        pyl.hist(self.points[:,1],bins=30,orientation='horizontal')
        pyl.ylabel('RMS')
        self.sp4 = pyl.subplot2grid((4,4),(2,0),rowspan=2)

        self.moffPatchList = []
        self.showing = []
        self.selected_star = -1

        for j in range(len(self.moffs)):
            self.moffPatchList.append(self.sp4.plot(self.moffr,self.moffs[j],zorder=0))
            self.showing.append(1)
        self.sp4.set_xlim(0,30)
        self.sp4.set_ylim(0,1.02)
        self.sp5 = pyl.subplot2grid((4,4),(3,1))
        self.sp5.set_aspect('equal')
        self.psfPlotLimits = [self.sp1.get_xlim(),self.sp1.get_ylim()]
        self.conn1 = self.sp1.callbacks.connect('ylim_changed',self.PSFrange)
        self.conn2 = pyl.connect('pick_event',self.ScatterPSF)
        self.conn3 = pyl.connect('key_press_event',self.ScatterPSF_keys)
        self.conn4 = pyl.connect('close_event',self.HandleClose)

        self.ScatterPSF(None)

        if not noVisualSelection:
            pyl.show()
        self._fwhm_lim=self.sp1.get_xlim()
        self._chi_lim=self.sp1.get_ylim()

        w=np.where((self.points[:,0]>self._fwhm_lim[0])&(self.points[:,0]<self._fwhm_lim[1])&(self.points[:,1]>self._chi_lim[0])&(self.points[:,1]<self._chi_lim[1])&(self.goodStars==True))
        pyl.clf()
        pyl.close()

        goodFits=self.points[w]
        goodMeds=np.median(goodFits[:4],axis=0)
        goodSTDs=np.std(goodFits[:4],axis=0)
        return (goodFits,goodMeds,goodSTDs)


    def PSFrange(self, junkAx):
        """
        Display function that you shouldn't call directly.
        """

        #ca=pyl.gca()
        pyl.sca(self.sp1)

        newLim = [self.sp1.get_xlim(),self.sp1.get_ylim()]
        self.psfPlotLimits=newLim[:]
        w = np.where((self.points[:,0]>=self.psfPlotLimits[0][0])&(self.points[:,0]<=self.psfPlotLimits[0][1])&(self.points[:,1]>=self.psfPlotLimits[1][0])&(self.points[:,1]<=self.psfPlotLimits[1][1]))[0]

        if self.starsScat!=None:
            self.starsScat.remove()
            self.starsScat=None

        for ii in range(len(self.showing)):
            if self.showing[ii]: self.moffPatchList[ii][0].remove()

        for ii in range(len(self.showing)):
            if ii not in w: self.showing[ii]=0
            else: self.showing[ii]=1

        for ii in range(len(self.showing)):
            if self.showing[ii]:
                self.moffPatchList[ii] = self.sp4.plot(self.moffr,self.moffs[ii])
                x_lim_max = np.max(self.moffr)+1

        self.sp4.set_xlim(0,self.moffatWidth)
        self.sp4.set_ylim(0,1.02)

        self.conn2 = pyl.connect('pick_event',self.ScatterPSF)
        self.conn3 = pyl.connect('key_press_event',self.ScatterPSF_keys)

        self._fwhm_lim = self.sp1.get_xlim()
        self._chi_lim = self.sp1.get_ylim()

        ## The above two lines need to be here again.
        ## This allows for zooming/inspecting in whatever order, over & over.
        pyl.draw()


    def ScatterPSFCommon(self, arg):
        """
        Display function that you shouldn't call directly.
        """
        w = np.where(self.goodStars)[0]
        W = np.where(self.goodStars != True)
        ##ca=pyl.gca()
        pyl.sca(self.sp1)
        xlim = self.sp1.get_xlim()
        ylim = self.sp1.get_ylim()
        title = self.sp1.get_title()
        self.sp1.cla()
        pyl.scatter(self.points[:, 0][w], self.points[:, 1][w], color='b',
                    picker=True, zorder=9)
        pyl.scatter(self.points[:, 0][W], self.points[:, 1][W],
                    picker=True, color='r', zorder=10)
        pyl.scatter(self.points[:, 0][arg], self.points[:, 1][arg],
                    marker='d', color='m', zorder=0, s=75)
        pyl.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
        pyl.title(title)
        max_x = self.sp5.get_xlim()[1]


    def ScatterPSF_keys(self, event):
        """
        Display function that you shouldn't call directly.
        """

        doClose = False

        self._increment = 0

        key = event.key
        npoints = len(self.points[:, 0])
        showingbool = np.array(self.showing).astype(np.bool)
        pointsshowing = self.points[showingbool]
        npointsshowing = len(pointsshowing[:, 0])
        args = np.arange(npoints)[showingbool]
        arg = args[np.argsort(pointsshowing[:, 0])[self.selected_star]]

        ca=pyl.gca()
        if self.starsScat!=None:
            self.starsScat.remove()
            self.starsScat=None

        if key in ['d','a','left','right']:  # Move forwards/backwards through points
            ## The below might seem overly complicated,
            ## but doing it this way ensures that you continue
            ## from same/next point after zooming.
            ## Hang on, no, this doesn't make it continue at the right point
            ## after a zoom. hm, fix later.
            if key == 'd' or key == 'right':
                self._increment = +1
            elif key == 'a' or key == 'left':
                self._increment = -1
            else:
                pass
            self.selected_star = (self.selected_star + self._increment) % npointsshowing
            arg = args[np.argsort(self.points[:, 0][showingbool])[self.selected_star]]
            self.ScatterPSFCommon(arg)

        elif key in ['w','up','down']:  # Remove a point.
            self.goodStars[arg] = not self.goodStars[arg]
            self.ScatterPSFCommon(arg)
            #pyl.sca(self.sp1)
            #self.sp1.set_xlim(xlim)
            #self.sp1.set_ylim(ylim)
            ##pyl.sca(ca)

        elif key in ['c','C']:
            doClose = True

        colour = 'b' if self.goodStars[arg] else 'r'
        self.starsScat = self.sp4.scatter(self.starsFlatR[arg],
                                          self.starsFlatF[arg],
                                          edgecolor=colour,facecolor='none',zorder=10)
        self.sp4.set_xlim(0, self.moffatWidth)
        self.sp4.set_ylim(0, 1.02)
        self.sp5.imshow(self.subsecs[arg])
        #self.sp5.set_xlabel("x,y = {:.1f}, {:.1f}".format(self.points[:, 4][arg],self.points[:, 5][arg]))


        self.conn1=self.sp1.callbacks.connect('ylim_changed',self.PSFrange)
        self.conn2=pyl.connect('pick_event',self.ScatterPSF)
        ## The above two lines need to be here again.
        ## This allows for zooming/inspecting in whatever order, over & over.
        pyl.sca(ca)
        pyl.draw()

        if doClose: pyl.close()


    def ScatterPSF(self, event):
        """
        Display function that you shouldn't call directly.
        """

        ca=pyl.gca()
        if event is not None:
            me=event.mouseevent


        if self.starsScat!=None:
            self.starsScat.remove()
            self.starsScat=None

        npoints = len(self.points[:, 0])
        showingbool = np.array(self.showing).astype(np.bool)
        pointsshowing = self.points[showingbool,:]
        ranks = np.zeros(len(pointsshowing[:,0]))
        if event is not None:
            args = np.argsort(np.abs(me.xdata - pointsshowing[:, 0]))
        else:
            args = np.argsort(np.abs(0.0 - pointsshowing[:, 0]))
        for ii in range(len(args)):
            ranks[args[ii]]+=ii
        if event is not None:
            args = np.argsort(np.abs(me.ydata - pointsshowing[:, 1]))
        else:
            args = np.argsort(np.abs(0.0 - pointsshowing[:, 1]))
        for ii in range(len(args)):
            ranks[args[ii]]+=ii

        args = np.arange(npoints)[showingbool]
        self.selected_star = np.argmin(ranks[np.argsort(pointsshowing[:, 0])])
        arg = args[np.argsort(pointsshowing[:, 0])[self.selected_star]]

        self.starsScat = self.sp4.scatter(self.starsFlatR[arg],self.starsFlatF[arg],facecolor='none',edgecolor='b',zorder=10)
        self.sp4.set_xlim(0, self.moffatWidth)
        self.sp4.set_ylim(0,1.02)

        self.sp5.imshow(self.subsecs[arg])
        #self.sp5.set_xlabel("x,y = {:.1f}, {:.1f}".format(self.points[:, 4][arg],self.points[:, 5][arg]))

        self.ScatterPSFCommon(arg)

        if event is not None:
            if me.button==3:
                self.goodStars[arg] = not self.goodStars[arg]
                self.ScatterPSFCommon(arg)

        self.conn1=self.sp1.callbacks.connect('ylim_changed',self.PSFrange)
        self.conn3=pyl.connect('key_press_event',self.ScatterPSF_keys)
        ## The above two lines need to be here again.
        ## This allows for zooming/inspecting in whatever order, over & over.
        pyl.sca(ca)
        pyl.draw()


    def HandleClose(self, evt):
        self._fwhm_lim = self.sp1.get_xlim()
        self._chi_lim = self.sp1.get_ylim()
