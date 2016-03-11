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


import pylab as pyl, numpy as num,psf
from stsci import numdisplay
from astropy.visualization import interval
from trippy import bgFinder

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

    def __init__(self,data,XWIN_IMAGE,YWIN_IMAGE,FLUX_AUTO,FLUXERR_AUTO,zscaleNsamp=200.,zscaleContrast=1.):
        self.XWIN_IMAGE=XWIN_IMAGE
        self.YWIN_IMAGE=YWIN_IMAGE
        self.FLUX_AUTO=FLUX_AUTO
        self.FLUXERR_AUTO=FLUXERR_AUTO
        self.data=data
        (self.z1,self.z2)=numdisplay.zscale.zscale(self.data,nsamples=zscaleNsamp,contrast=zscaleContrast)
        self.normer=interval.ManualInterval(self.z1,self.z2)

    def __call__(self,moffatWidth,moffatSNR,initAlpha=5.,initBeta=2.,repFact=5,xWidth=51,yWidth=51,
                 includeCheesySaturationCut=True,autoTrim=False,noVisualSelection=False,verbose=False):
        self.moffatWidth=moffatWidth
        self.moffatSNR=moffatSNR
        self.initAlpha=initAlpha
        self.initBeta=initBeta
        self.repFact=repFact

        self.fwhms=[]
        self.points=[]
        self.moffs=[]
        self.moffr=num.linspace(0,30,80)
        self.starsFlatR=[]
        self.starsFlatF=[]
        self.subsecs=[]
        self.goodStars=[]
        self.starsScat=None

        print 'Fitting stars with moffat profiles...'


        for j in range(len(self.XWIN_IMAGE)):
            if self.FLUX_AUTO[j]/self.FLUXERR_AUTO[j]>self.moffatSNR:
                if self.XWIN_IMAGE[j]-1<0 or self.XWIN_IMAGE[j]-1>=self.data.shape[1] or self.YWIN_IMAGE[j]-1<0 or self.YWIN_IMAGE[j]-1>=self.data.shape[0]:
                    continue
                mpsf=psf.modelPSF(num.arange(xWidth),num.arange(yWidth),alpha=self.initAlpha,beta=self.initBeta,repFact=self.repFact)
                mpsf.fitMoffat(self.data,self.XWIN_IMAGE[j],self.YWIN_IMAGE[j],boxSize=self.moffatWidth,verbose=verbose)

                fwhm=mpsf.FWHM(fromMoffatProfile=True)
                #if includeCheesySaturationCut:
                #    if (mpsf.fDist[0]-mpsf.bg)/(mpsf.moffat(0)*mpsf.A)<0.9: #cheesy saturation cut
                #        #print 'Saturated'
                #        continue

                #norm=Im.normalise(mpsf.subsec,[self.z1,self.z2])
                norm=self.normer(mpsf.subSec)
                if fwhm<>None and not (num.isnan(mpsf.beta) or num.isnan(mpsf.alpha)):
                    #print self.XWIN_IMAGE[j],self.YWIN_IMAGE[j],mpsf.alpha,mpsf.beta,fwhm
                    print '{: 8.2f} {: 8.2f} {: 5.2f} {: 5.2f} {: 5.2f}'.format(self.XWIN_IMAGE[j],self.YWIN_IMAGE[j],mpsf.alpha,mpsf.beta,fwhm)

                    self.subsecs.append(norm*1.)
                    self.goodStars.append(True)

                    self.moffs.append(mpsf.moffat(self.moffr)*1.)

                    self.starsFlatR.append(psf.downSample2d(mpsf.repRads,mpsf.repFact))
                    self.starsFlatF.append((mpsf.subSec-mpsf.bg)/(mpsf.moffat(0)*mpsf.A))

                    self.moffs[len(self.moffs)-1]/=num.max(self.moffs[len(self.moffs)-1])

                    self.points.append([fwhm,mpsf.chi,mpsf.alpha,mpsf.beta,self.XWIN_IMAGE[j],self.YWIN_IMAGE[j],mpsf.bg])
        self.points=num.array(self.points)
        self.goodStars=num.array(self.goodStars)

        if autoTrim:
            bg=bgFinder.bgFinder(self.points[:,0])
            mode=bg('fraserMode')
            w=num.where(num.abs(self.points[:,0]-mode)>0.5)
            self.goodStars[w]=False



        self.figPSF=pyl.figure('Point Source Selector')
        self.sp1=pyl.subplot2grid((4,4),(0,1),colspan=3,rowspan=2)
        pyl.scatter(self.points[:,0],self.points[:,1],picker=True)
        pyl.title('Select PSF range with zoom and then close the plot window.')
        self.sp2=pyl.subplot2grid((4,4),(2,1),colspan=3,sharex=self.sp1,rowspan=1)
        bins=num.arange(num.min(self.points[:,0]),num.max(self.points[:,0])+0.5,0.5)
        pyl.hist(self.points[:,0],bins=bins)
        pyl.xlabel('FWHM (pix)')
        self.sp3=pyl.subplot2grid((4,4),(0,0),rowspan=2,sharey=self.sp1)
        pyl.hist(self.points[:,1],bins=30,orientation='horizontal')
        pyl.ylabel('RMS')
        self.sp4=pyl.subplot2grid((4,4),(2,0),rowspan=2)

        self.moffPatchList=[]
        self.showing=[]

        for j in range(len(self.moffs)):
            self.moffPatchList.append(self.sp4.plot(self.moffr,self.moffs[j]))
            self.showing.append(1)
        self.sp4.set_xlim(0,30)
        self.sp4.set_ylim(0,1.02)
        self.sp5=pyl.subplot2grid((4,4),(3,1))
        self.sp5.set_aspect('equal')
        self.psfPlotLimits=[self.sp1.get_xlim(),self.sp1.get_ylim()]
        self.conn1=self.sp1.callbacks.connect('ylim_changed',self.PSFrange)
        self.conn2=pyl.connect('pick_event',self.ScatterPSF)
        if not noVisualSelection: pyl.show()


        fwhm_lim=self.sp1.get_xlim()
        chi_lim=self.sp1.get_ylim()

        w=num.where((self.points[:,0]>fwhm_lim[0])&(self.points[:,0]<fwhm_lim[1])&(self.points[:,1]>chi_lim[0])&(self.points[:,1]<chi_lim[1])&(self.goodStars==True))
        pyl.close()

        goodFits=self.points[w]
        goodMeds=num.median(goodFits[:4],axis=0)
        goodSTDs=num.std(goodFits[:4],axis=0)
        return (goodFits,goodMeds,goodSTDs)

    def PSFrange(self,junkAx):
        """
        Display function that you shouldn't call directly.
        """

        #ca=pyl.gca()
        pyl.sca(self.sp1)

        print self.starsScat

        newLim=[self.sp1.get_xlim(),self.sp1.get_ylim()]
        self.psfPlotLimits=newLim[:]
        w=num.where((self.points[:,0]>=self.psfPlotLimits[0][0])&(self.points[:,0]<=self.psfPlotLimits[0][1])&(self.points[:,1]>=self.psfPlotLimits[1][0])&(self.points[:,1]<=self.psfPlotLimits[1][1]))[0]

        if self.starsScat<>None:
            self.starsScat.remove()
            self.starsScat=None

        for ii in range(len(self.showing)):
            if self.showing[ii]: self.moffPatchList[ii][0].remove()

        for ii in range(len(self.showing)):
            if ii not in w: self.showing[ii]=0
            else: self.showing[ii]=1



        for ii in range(len(self.showing)):
            if self.showing[ii]:
                self.moffPatchList[ii]=self.sp4.plot(self.moffr,self.moffs[ii])
        self.sp4.set_xlim(0,30)
        self.sp4.set_ylim(0,1.02)

        pyl.draw()

    def ScatterPSF(self,event):
        """
        Display function that you shouldn't call directly.
        """

        ca=pyl.gca()
        me=event.mouseevent

        if self.starsScat<>None:
            self.starsScat.remove()
            self.starsScat=None

        ranks=self.points[:,0]*0.0
        args=num.argsort(num.abs(me.xdata-self.points[:,0]))
        for ii in range(len(args)):
            ranks[args[ii]]+=ii
        args=num.argsort(num.abs(me.ydata-self.points[:,1]))
        for ii in range(len(args)):
            ranks[args[ii]]+=ii

        arg=num.argmin(ranks)

        self.starsScat=self.sp4.scatter(self.starsFlatR[arg],self.starsFlatF[arg])
        self.sp4.set_xlim(0,30)
        self.sp4.set_ylim(0,1.02)

        self.sp5.imshow(self.subsecs[arg])

        #below should be cleaned up eventually
        ##ca=pyl.gca()
        pyl.sca(self.sp1)
        xlim=self.sp1.get_xlim()
        ylim=self.sp1.get_ylim()
        title=self.sp1.get_title()
        self.sp1.cla()
        w=num.where(self.goodStars)[0]
        W=num.where(self.goodStars<>True)
        pyl.scatter(self.points[:,0],self.points[:,1],color='b',picker=True,zorder=9)
        pyl.scatter(self.points[:,0][W],self.points[:,1][W],picker=True,color='r',zorder=10)
        pyl.scatter(self.points[:,0][arg],self.points[:,1][arg],marker='d',color='m',zorder=0,s=75)
        pyl.axis([xlim[0],xlim[1],ylim[0],ylim[1]])
        pyl.title(title)
        ##pyl.sca(ca)

        if me.button==3:
            if self.goodStars[arg]==True: self.goodStars[arg]=False
            else: self.goodStars[arg]=True

            w=num.where(self.goodStars)[0]
            W=num.where(self.goodStars<>True)

            ##ca=pyl.gca()
            xlim=self.sp1.get_xlim()
            ylim=self.sp1.get_ylim()
            title=self.sp1.get_title()
            ##pyl.sca(sp1)
            self.sp1.cla()
            pyl.scatter(self.points[:,0][w],self.points[:,1][w],picker=True,color='b',zorder=9)
            pyl.scatter(self.points[:,0][W],self.points[:,1][W],picker=True,color='r',zorder=10)
            pyl.scatter(self.points[:,0][arg],self.points[:,1][arg],marker='d',color='m',zorder=0,s=75)
            self.sp1.set_xlim(xlim)
            self.sp1.set_ylim(ylim)
            pyl.title(title)
            ##pyl.sca(ca)
        pyl.sca(ca)
        pyl.draw()
