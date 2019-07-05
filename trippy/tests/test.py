from __future__ import print_function, division

from trippy import psf, scamp, bgFinder, pill, psfStarChooser
import numpy as np,scipy as sci,pylab as pyl
from astropy.io import fits
import pickle
import unittest
import os,sys



class tester(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.rates = [0.4,1.0,2.5]
        self.angles = [37.0,0.0,-58.7]
        self.EXPTIME = 360.0

        #os.system('gunzip test_image.fits.gz')

        self.gened_bg = 1000.0
        try:
            with fits.open('test_image.fits') as han:
                self.image = han[0].data
            self.loadedPSF = psf.modelPSF(restore = 'test_psf.fits')

            with open('planted_locations.pickle','rb') as han:
                if sys.version_info[0]==3:
                    x = pickle.load(han, encoding='latin1')
                elif sys.version_info[0]==2:
                    x = pickle.load(han)
        except:
            print('You may not have the necessary fits files for these tests.')
            print('Please execute the following two wget commands:')
            print('wget https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/fraserw/test_psf.fits')
            print('wget https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/fraserw/test_image.fits')
            print('')
            exit()
        self.planted_locations = np.array(x)

        self.bg = bgFinder.bgFinder(self.image)

        scamp.makeParFiles.writeConv()
        scamp.makeParFiles.writeParam()
        scamp.makeParFiles.writeSex('test.sex', minArea = 2, threshold = 2)

        scamp.runSex('test.sex','test_image.fits',verbose = True)
        self.catalog = scamp.getCatalog('def.cat',paramFile = 'def.param')


        os.remove('default.conv')
        os.remove('def.param')
        os.remove('test.sex')
        os.remove('def.cat')
        #os.system('gzip test_image.fits')

        starChooser = psfStarChooser.starChooser(self.image,
                                                 self.catalog['XWIN_IMAGE'],self.catalog['YWIN_IMAGE'],
                                                 self.catalog['FLUX_AUTO'],self.catalog['FLUXERR_AUTO'])
        (self.goodFits,self.goodMeds,self.goodSTDs) = starChooser(30, 25, noVisualSelection=True, autoTrim=False,
                                                                  quickFit=False, repFact=5)
        (self.goodFitsQF,self.goodMedsQF,self.goodSTDsQF) = starChooser(30, 25, noVisualSelection=True, autoTrim=False,
                                                                        quickFit=True, repFact=5)

        #using manual alpha and beta because the fitting done with the star chooser results in
        #different answers with the different versions of scipy
        ALPHA = 29.81210639392963
        BETA = 19.32497948470224

        self.goodPSF = psf.modelPSF(np.arange(51),np.arange(51), alpha=ALPHA,beta=BETA,repFact=10)
        #self.goodPSF = psf.modelPSF(np.arange(51),np.arange(51), alpha=self.goodMeds[2],beta=self.goodMeds[3],repFact=10)
        self.goodPSF.genLookupTable(self.image,self.goodFits[:,4],self.goodFits[:,5],verbose=False)
        self.goodPSF.line(self.rates[0],
                          self.angles[0],
                          self.EXPTIME/3600.,
                          pixScale=0.185,
                          useLookupTable=True)
        #self.goodPSF.psfStore('test_psf.fits')

        self.pill = pill.pillPhot(self.image)
        rads = np.arange(5.0,25.0,5)
        x_test, y_test = 652.4552577047876, 101.62067493726078 #necessray to use custom coordinates to avoid errors due to different sextractor versions
        self.pill.computeRoundAperCorrFromSource(x_test,y_test,
                                             rads, width = 60.0, skyRadius = 50.0,display=False,displayAperture=False)

        self.pill(x_test,y_test,7.0,l=5.0,display=False,enableBGSelection = False)
        self.pill.SNR(verbose=True)

        self.distances = {}
        self.distances[26] = 0.2913793325
        self.distances[16] = 0.5179553032
        self.distances[7] = 0.1751143336
        self.distances[0] = 0.7428739071
        self.distances[23] = 0.3347620368
        self.distances[18] = 0.7867022753
        self.distances[14] = 0.1851933748
        self.distances[5] = 0.4436303377
        self.distances[28] = 0.3333371580
        self.distances[9] = 0.1997155696
        self.distances[29] = 0.3528487086
        self.distances[22] = 0.4013085365
        self.distances[19] = 0.5801378489
        self.distances[17] = 0.6873673797
        self.distances[1] = 0.1621235311
        self.distances[6] = 0.1248580739
        self.distances[13] = 2.0296137333
        self.distances[24] = 1.3238428831
        self.distances[3] = 0.7816261053
        self.distances[25] = 0.3067137897
        self.distances[6] = 0.8099660873



    def test_compSourceAper(self):
        self.assertAlmostEqual(self.pill.roundAperCorr(7.5),0.8348167541693421,msg = 'Round aperture correction from image seems to be discrepant.')

    def test_snr(self):
        self.assertAlmostEqual(self.pill.snr,22.629585517648817,msg = 'SNR is different than expected.')
    def test_flux(self):
        self.assertAlmostEqual(self.pill.sourceFlux,8602.32952903671,msg = 'Flux is different than expected.')
    def test_bg(self):
        self.assertAlmostEqual(self.pill.bg,1002.9928889886571,msg = 'Background is different than expected.')
    def test_numPix(self):
        self.assertEqual(self.pill.nPix,225.7,msg = 'Numpix is different than expected.')

    def test_roundAperFromPSF(self):
        rads = np.arange(5.0,25.0,5)
        r = (rads[1]+rads[0])/2.0
        self.goodPSF.computeRoundAperCorrFromPSF(rads, display=False,displayAperture=False)
        self.loadedPSF.computeRoundAperCorrFromPSF(rads, display=False,displayAperture=False)
        self.assertAlmostEqual(self.goodPSF.roundAperCorr(r),self.loadedPSF.roundAperCorr(r),6, msg = 'Line aperture corrections differ.')

    def test_lineAperFromTSF(self):
        rads = np.arange(5.0,25.0,5)
        r = (rads[1]+rads[0])/2.0
        self.goodPSF.computeLineAperCorrFromTSF(rads, self.rates[0]/10.0, self.angles[0],display=False,displayAperture=False)
        self.loadedPSF.computeLineAperCorrFromTSF(rads, self.rates[0]/10.0, self.angles[0],display=False,displayAperture=False)
        self.assertAlmostEqual(self.goodPSF.lineAperCorr(r),self.loadedPSF.lineAperCorr(r), 6,  msg = 'Line aperture corrections differ.')

    def test_fwhm(self):
        self.assertEqual(self.loadedPSF.FWHM(), self.goodPSF.FWHM(),msg = 'PSF FWHM differ.')

    def test_fwhmFromMoffat(self):
        self.assertEqual(self.loadedPSF.FWHM(fromMoffatProfile=True), self.goodPSF.FWHM(fromMoffatProfile=True),msg = 'PSF FWHM differ.')


    def test_lookupTable(self):
        diff = self.goodPSF.lookupTable - self.loadedPSF.lookupTable
        self.assertAlmostEqual(np.max(np.abs(diff)),0.0, 2, msg = 'Generated lookup table appears to be unusual.')

    def test_long_one(self):
        print('\n    Test Long PSF One:')
        self.goodPSF.line(self.rates[0],
                          self.angles[0],
                          self.EXPTIME/3600.,
                          pixScale=0.185,
                          useLookupTable=True)
        self.loadedPSF.line(self.rates[0],
                            self.angles[0],
                            self.EXPTIME/3600.,
                            pixScale=0.185,
                            useLookupTable=True)
        diff = self.goodPSF.longPSF - self.loadedPSF.longPSF
        self.assertAlmostEqual(np.max(np.abs(diff)),0.0,1, msg = 'Generated TSF one appears to be unusual.')

    def test_long_two(self):
        print('\n    Test Long PSF Two:')
        self.goodPSF.line(self.rates[1],
                          self.angles[1],
                          self.EXPTIME/3600.,
                          pixScale=0.185,
                          useLookupTable=True)
        self.loadedPSF.line(self.rates[1],
                            self.angles[1],
                            self.EXPTIME/3600.,
                            pixScale=0.185,
                            useLookupTable=True)
        diff = self.goodPSF.longPSF - self.loadedPSF.longPSF
        self.assertAlmostEqual(np.max(np.abs(diff)), 0.0, 1, msg = 'Generated TSF two appears to be unusual.')

    def test_long_three(self):
        print('\n    Test Long PSF Three:')
        self.goodPSF.line(self.rates[2],
                          self.angles[2],
                          self.EXPTIME/3600.,
                          pixScale=0.185,
                          useLookupTable=True)
        self.loadedPSF.line(self.rates[2],
                            self.angles[2],
                            self.EXPTIME/3600.,
                            pixScale=0.185,
                            useLookupTable=True)
        diff = self.goodPSF.longPSF - self.loadedPSF.longPSF
        self.assertAlmostEqual(np.max(np.abs(diff)), 0.0, 1, msg = 'Generated TSF three appears to be unusual.')

    def test_createPSF(self):
        diff = np.sum(self.goodFits[:,0] - np.array([11.42,11.23,11.68,12.19]))
        self.assertAlmostEqual(diff,0.0, msg = 'Fitted FWHM not the same. ')

    def test_createPSF_QF(self):
        diff = np.sum(self.goodFitsQF[:,0] - np.array([11.49,11.37,11.77,12.26]))
        self.assertAlmostEqual(diff,0.0, msg = 'Quick fitted FWHM not the same. ')


    def test_sextractor_coordinates(self):

        n_good = 0
        for i in range(len(self.catalog['XWIN_IMAGE'])):
            x = self.catalog['XWIN_IMAGE'][i]-1.0
            y = self.catalog['YWIN_IMAGE'][i]-1.0
            d = ((self.planted_locations[:,0] - x)**2 + (self.planted_locations[:,1]-y)**2)**0.5
            arg = np.argmin(d)
            #super lax distance to handle sextractor version differences
            #avoid index 6 which is source that sextractor cuts into two.
            if abs(d[arg] - self.distances[arg])<0.05 and arg!=6:
                n_good+=1
        self.assertEqual(n_good,19,'Didnt find 20 unique sources in the correct places.')

    def test_len_sex(self):
        self.assertEqual(len(self.catalog['XWIN_IMAGE']),21,'Did not detect exactly 21 sources.')

    def test_bg_median(self):
        median = self.bg('median')
        self.assertAlmostEqual(median, 1000.0341315254655,  msg = 'Median failed.')

    def test_bg_mean(self):
        mean = self.bg('mean')
        self.assertAlmostEqual(mean, 1000.0460484863409,  msg = 'Mean failed.')

    def test_bg_hist(self):
        hist = self.bg('histMode')
        self.assertAlmostEqual(hist, 1000.4190254669019,  msg = 'Hist method failed.')

    def test_bg_fraser(self):
        fraser = self.bg('fraserMode',0.1)
        self.assertAlmostEqual(fraser, 999.933901294518,  msg = 'Fraser method failed.')

    def test_bg_gauss(self):
        gauss = self.bg('gaussFit')
        self.assertAlmostEqual(gauss, 1000.0460761715437,  msg = 'Gaussian method failed.')

    def test_bg_smart(self):
        smart = self.bg('smart',3.0)
        self.assertAlmostEqual(smart, 1000.0460761715437,  msg = 'Smart method failed.')



if __name__ == "__main__":

    unittest.main()
