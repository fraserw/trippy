
# TRIPPy: Python based Trailed Source Photometry 
*(published in the Astrophysical Journal)*

TRIPPy is a python package aimed to perform all the steps required to measure accurate photometry of both trailed and non-trailed (stationary) astronomical sources. This includes the ability to generate stellar and trailed point source functions, and to use circular and pill shaped apertures to measure photometry and estimate appropriate aperture corrections.

Citation: If you use TRIPPy in your science works, please cite Fraser, W. et al., 2016, To appear in AJ. 
DOI at Zenodo: http://dx.doi.org/10.5281/zenodo.48694

## Update to version 1

TRIPPy has been updated to version 1.0, which includes many changes around accuracy and performance. Performance increases are all over, and you'll see as you use it. 

*A bug was found* in the github verion prior to v1.0 which lightly affected the stellar source aperture correction estimates. __This did not affect the first released version, and has been fixed for v1.0.__

A few notable feature improvements are the ability to pass more than one radius as an array format when calling phot, a new space-saving PSF format, and significant improvements in the interaction with the star selector panel. 

## Installation

TRIPPy is compatible with python 2.7 and 3.5. Though some parts that require sklearn are only available in python 3.5.

### Dependencies

TRIPPy depends on a few separate, equally useful python packages. These packages are:
* numpy
* scipy
* matplotlib
* astropy
* stsci.numdisplay
* sklearn (optional, only compatible with python 3)
* emcee (optional)
* numba -- very very useful for big speed improvements. Highly recommended!!

An additional optional piece of software is the SExtractor package. This can often be installed with through standard
linux repository systems (yum, apt-get, etc.) or can be simply downloaded and compiled. [Source is available here].(http://www.astromatic.net/software/sextractor)


*Technically* pip should automatically install all of the above requirements. In practice however, pip falls over quiet
hilariously. Rather, the most reliable thing to do is install these requirements first.

All packages can be install using pip. eg. pip install stsci . Alternatively, the first four requirements can be
installed (and maintained!) with an anaconda python installation. [Instructions to install anaconda are available here.](https://www.continuum.io/)

Once numpy, scipy, matplotlib, and astropy are installed, stsci-python (which provides stsci.numdisplay) can be
installed by the two commands

    pip install stsci.distutils
    pip install stsci-python

This will compile and install all of the bits required by stsci-python.

Finally, two optional packages, sklearn and emcee (MC Hammer) can be installed. This will provide robust source fitting utilities in
the MCMCfit module.

Test that these modules are available by attempting to import each one:

    python -c "import numpy; import scipy; import matplotlib; import astropy; import stsci.numdisplay"

If the optional emcee, or sklearn packages were installed

    python -c "import emcee"
    python -c "import sklearn"


### TRIPPy installation

Once dependencies are installed, trippy can be installed with a simple pip invocation

    pip install trippy
    

Accesibility can be checked with a few import commands

    python -c "import trippy; from trippy import bgFinder; from trippy import psfStarChooser; from trippy import scamp"

and if the optional emcee package was installed

    python -c "from trippy import MCMCfit"


NOTE: Currently no internal tests have been written. This will eventually change of course.



## Usage

I have provided an ipython notebook which shows most of the functionality of the trippy package including PSF/TSF
generation, photometry, and source fitting and removal. Please checkout the [notebook.](https://github.com/fraserw/trippy/blob/master/tutorial/trippytutorial.ipynb)
