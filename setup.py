#! /usr/bin/env python

from setuptools import setup
import sys

#if sys.version_info[0] > 2:
#    print 'trippy is only compatible with Python version 2.7+, not yet 3.x'
#    sys.exit(-1)

print("Trippy unittests haven't been built into setup.py scripts. Please manually run tests/test.py to ensure everything works before production use.")

dependencies = ['numpy >= 1.16.2',
                'scipy',
                'matplotlib',
                'astropy >= 3.1']

setup(
  name = 'trippy',
  packages = ['trippy','trippy.scamp','trippy.MCMCfit'],
  version = '1.1',
  description = 'Pill aperture photometry for trailed astronomical sources',
  author = 'Wesley Fraser',
  author_email = 'westhefras@gmail.com',
  url = 'https://github.com/fraserw/trippy',
  download_url = 'https://github.com/fraserw/trippy/tarball/0.1',
  keywords = ['Photometry', 'Astronomy', 'PSF'],
  license = 'GNU',
  install_requires=dependencies,
  classifiers = [],
  extras_require = {
      'MCMCfit': ['emcee']
  },
  test_suite="trippy.tests",
  tests_require=['mock', ],
)
