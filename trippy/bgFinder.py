# -*- coding: utf-8 -*-
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

__author__ = ('Wesley Fraser (@wtfastro, github: fraserw <westhefras@gmail.com>), '
              'Academic email: wes.fraser@qub.ac.uk')

import numpy as np
import pylab as pyl
from scipy import stats
from astropy.io import fits as pyf
from scipy import optimize as opti

#: The class for the return-value of :meth:`bgFinder._stats`
Stats = namedtuple('Stats', ('mode', 'stddev'))


class bgFinder(object):
    """
    Get the background estimate of the inputted data. eg.

    bgf=trippy.bgFinder.bgFinder(dataNumpyArray)

    Best to call it as
    bgf(mode,n)
    where n is optional and not called for every mode

    where mode can be
    median - simple median of the values
    histMode [n] - uses a histogram to estimate the mode, n is the number of bins in the histogram
    mean - simple mean of the values
    fraserMode [n] - the background modal estimate described in Fraser et al. (2016) TRIPPy paper.
                     n=0.1 to 0.2 seems to work best. This is the method most robust to background
                     sources in the data. Also works well in pure background data. This is probably
                     the most robust method.
    gaussFit - perform a approximate gaussian fit to the data, returning the mean fit value
    smart [n] - this first does a gaussFit. If the condition standard Deviation/mean**0.5 > n
                (where n is the # of standard deviations, ~3) is satisfied, it means you have
                contamination, in which case it reverts to fraserMode. Otherwise, the gaussian
                mean is returned.
    """
    def __init__(self, data):
        self.data = np.ravel(data)
        self.plotAxis = None

    def __call__(self, method='median', inp=None):
        if method == 'median':
            return self.median()
        elif method == 'mean':
            return self.mean()
        elif method == 'histMode':
            if inp is None:
                retval = self.histMode()
            else:
                retval = self.histMode(inp)
            return retval
        elif method == 'fraserMode':
            if inp is None:
                retval = self.fraserMode()
            else:
                retval = self.fraserMode(inp)
            return retval
        elif method == 'gaussFit':
            return self.gaussFit()
        elif method == 'smart':
            return self.smartBackground()
        else:
            raise ValueError('Unknown method {}'.format(method))

    def histMode(self, nbins=50 , display = False):
        if display:
            g = self._stats(nbins)[0]
            self.background_display(g)
            return g
        return self._stats(nbins)[0]

    def median(self, display = False):
        if display:
            g = np.median(nbins)
            self.background_display(g)
            return g
        return np.median(self.data)

    def mean(self, display = False):
        if display:
            g = np.mean(self.data)
            self.background_display(g)
            return g
        return np.mean(self.data)

    def fraserMode(self, multi=0.1, display = False):
        if display:
            g = self._fraserMode(multi)
            self.background_display(g)
            return g
        return self._fraserMode(multi)

    def gaussFit(self, display = False):
        if display:
            g = self._gaussFit()
            self.background_display(g)
            return g
        return self._gaussFit()

    @staticmethod
    def _ahist(data, nbins):
        # ahist and stats generously donated by JJ Kavelaars from jjkmode.py
        b = np.sort(data)
        ## use the top and bottom octile to set the histogram bounds
        mx = b[len(b) - max(1, int(len(b) / 100.0))]
        mn = b[len(b) - int(99 * len(b) / 100.0)]
        w = (int((mx - mn) / nbins))

        n = np.searchsorted(b, np.arange(mn, mx, w))
        n = np.concatenate([n, [len(b)]])
        retval = (n[1:] - n[:-1], w, mn)
        return retval

    def _stats(self, nbins=50):
        (b, w, l) = self._ahist(self.data, nbins)
        b[len(b) - 1] = 0
        b[0] = 0
        am = np.argmax(b)
        c = b[b > int(b[am] / 2.0)]
        mode = (am * w) + l
        stddev = (len(c) * w / 2.0) / 1.41
        retval = Stats(mode, stddev)
        return retval

    def _fraserMode(self, multi=0.1):
        y = np.array(self.data * multi).astype(int)
        mode = stats.mode(y)[0]
        w = np.where(y == mode)
        next_mode = stats.mode(y[np.where(y!=mode)])[0]
        if next_mode<mode and abs(next_mode-mode)==1:
            adjust = -0.5/multi
        elif next_mode>mode and abs(next_mode-mode)==1:
            adjust = 0.5/multi
        else:
            adjust = 0.0
        return np.median(self.data[w[0]])+adjust

    def _gaussFit(self):
        med = np.median(self.data)
        std = np.std(self.data)

        res = opti.fmin(self._gaussLike, [med, std], disp=False)
        self.gauss = res
        return res[0]

    def _gaussLike(self, x):
        [m, s] = x[:]

        X = -np.sum((self.data - m)**2) / (2.0 * s * s)
        X -= len(self.data) * np.log((2 * np.pi * s * s)**0.5)
        return -X

    def smartBackground(self, gaussStdLimit=1.1, backupMode='fraserMode', inp=None, verbose=False,
                        display=False, forceBackupMode = False):
        """
        guassStdLimit=1.1 seemed the best compromise in my experience

        If we want to use a backup mode only here, then set the backupmode of choice, and set forceBackgroundMode = True
        """
        self._gaussFit()
        g, s = self.gauss
        if (s / g**0.5) > gaussStdLimit or forceBackupMode:
            if inp is not None:
                if verbose:
                    print('\nUsing backup mode %s with parameter %s.\n' % (backupMode, inp))
                g = self(backupMode, inp)
            else:
                if verbose:
                    print('\nUsing backup mode %s.\n' % (backupMode))
                g = self(backupMode)

        if display:
            self.background_display(g)
        return g

    def background_display(self, g):
        runShow = False
        if self.plotAxis  is None:
            figHist = pyl.figure('backgroundHistogram')
            self.plotAxis = figHist.add_subplot(111)
            runShow = True
        self.plotAxis.hist(self.data, bins=min(100, int(len(self.data) / 10.0)))
        (y0, y1) = self.plotAxis.get_ylim()
        self.plotAxis.plot([g, g], [y0, y1], 'r-', lw=2)
        self.plotAxis.set_title('Background {:.3f}'.format(g))

        if runShow:
            pyl.show()

    """
    def midBackground(self):
        x=np.array([self._gaussFit(),self.median(),self.mean(),self.fraserMode(),self.histMode()])
        args=np.argsort(x)
        if args[2]==0:
            print 'Adopting the Gaussian Fit.'
        elif args[2]==1:
            print 'Adopting the median.'
        elif args[2]==2:
            print 'Adopting the mean.'
        elif args[2]==3:
            print 'Adopting the Fraser mode.'
        else:
            print 'Adopting the JJK mode.'
        return x[args[2]]
    """

if __name__ == "__main__":

    v = np.array([17.36,  3.17,  3.17,  4.78,  3.13,  3.15,  3.16,  3.12,  3.19,  4.27, 20.48,  3.2,
  3.24,  3.12,  3.23,  8.53,  3.19,  3.22,  3.19,  3.2,   3.22,  3.18,  3.19,  3.14,
  6.33,  5.38])
    bg = bgFinder(v)
    mean = bg.mean()
    median = bg.median()
    fmode = bg.fraserMode(0.5)
    gauss = bg.gaussFit()
    smart = bg.smartBackground(inp=0.1)
    print(smart,mean,median,fmode,gauss)
    exit()

    with pyf.open('junk.fits') as han:
        data = han[1].data

    #near source
    x, y = 3275, 2266
    #cosmic ray
    x, y = 3179, 2314
    #out of source
    x, y = 3205, 2260
    #funny place
    x, y = 3093, 2422

    w = 15

    data = data[y - w: y + w + 1, x - w: x + w + 1].reshape((2 * w + 1)**2)

    bg = bgFinder(data)
    mean = bg.mean()
    median = bg.median()
    histo = bg.histMode()
    fmode = bg.fraserMode(0.1)
    gauss = bg.gaussFit()
    smart = bg.smartBackground(inp=0.1)

    print('Mean', mean)
    print('Median', median)
    print('JJKMode', histo)
    print('FraserMode', fmode)
    print('Gauss Fit', gauss)
    print('Smart Background', smart)

    fig = pyl.figure(1)
    ax = fig.add_subplot(111)
    pyl.hist(data, bins=w * 20)
    (y0, y1) = ax.get_ylim()
    pyl.plot([mean, mean], [0, y1], label='mean', lw=2)
    pyl.plot([median, median], [0, y1], label='median', lw=2)
    pyl.plot([histo, histo], [0, y1], label='JJKMode', lw=2)
    pyl.plot([fmode, fmode], [0, y1], label='Fraser Mode', lw=2)
    pyl.plot([gauss, gauss], [0, y1], label='Gauss Fit', lw=2)
    pyl.legend()
    pyl.show()
