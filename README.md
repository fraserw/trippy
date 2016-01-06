
TRIPPy: Python based Trailed Source Photometry (to be published soon)

This package is presented in a manuscript that will soon be submitted to ApJ's new code journal. But for now, you can
see the draft science publication at this [authorea link.](https://www.authorea.com/users/44077/articles/93287/_show_article)

1. Installation
===============

1a. Dependencies
----------------
TRIPPy depends on a few separate, equally useful python packages. These packages are:
* numpy
* scipy
* matplotlib
* astropy
* stsci.numdisplay
* emcee (optional)

An additional optional piece of software is the SExtractor package. This can often be installed with through standard
linux repository systems (yum, apt-get, etc.) or can be simply downloaded and compiled. [Source is available here].(http://www.astromatic.net/software/sextractor)


*Technically* pip should automatically install all of the above requirements. In practice however, pip falls over quiet
hilariously. Rather, the most reliable thing to do is install these requirements first.

All packages can be install using pip. eg. pip install stsci . Alternatively, the first four requirements can be
installed (and maintained!) with an anaconda python installation. [Instructions to install anaconda are available here.](https://www.continuum.io/)

Once numpy, scipy, matplotlib, and astropy are installed, stsci-python (which provides stsci.numdisplay) can be
installed by the two commands

    pip install stsci.distutils
    pip install stscipython

This will compile and install all of the bits required by stsci-python.

Finally, the optional package emcee (MC Hammer) can be installed. This will provide robust source fitting utilities in
the MCMCfit module.

Test that these modules are available by attempting to import each one:

    python -c "import numpy; import scipy; import matplotlib; import astropy; import stsci.numdisplay"

If the optional emcee was installed

    python -c "import emcee"


1b. TRIPPy installation
-----------------------
Once dependencies are installed, trippy can be installed with a simple pip invocation

    pip install trippy

Accesibility can be checked with a few import commands

    python -c "import trippy; from trippy import bgFinder; from trippy import psfStarChooser; from trippy import scamp"

and if the optional emcee package was installed

    python -c "from trippy import MCMCfit"


NOTE: Currently no internal tests have been written. This will eventually change of course.



2. Usage
========

I have provided an ipython notebook which shows most of the functionality of the trippy package including PSF/TSF
generation, photometry, and source fitting and removal. Please checkout the [notebook.](https://github.com/fraserw/trippy/blob/master/tutorial/trippytutorial.ipynb)
