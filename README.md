![RADPy logo](https://github.com/spaceashley/radpy/blob/main/radpynobglogo.png)


# Robust Angular Diameters in Python (`RADPy`)
## Introduction to `RADPy`

`RADPy` stands for Robust Angular Diameters in Python. This was created to allow for multi-wavelength fits for angular diameters of stars measured with interferometric methods. Currently `RADPy` only has compatibility with the instruments on the Center for High Angular Resolution Astronomy (CHARA) Array. `RADPy` is currently configured for the following instruments at CHARA:

- Classic/CLIMB
- PAVO
- VEGA
- MIRC-X
- MYSTIC
- SPICA

## To install:

**Please read this section carefully!!**

Simply use pip to install `RADPy`. Due to naming conflicts, to install `RADPy`, you must use "rsadpy". 

```python
pip install rsadpy
```

The installation should also install all necessary additional packages you need to run everything. Just in case, here is a list of all the necessary packages that aren't default:
- `lmfit`
- `astropy`
- `astroquery`
- `gaiadr3-zeropoint`

If you would like to use the SED fitting feature, there are some additional packages you need to make sure that you have installed. 
- `SEDFit`
- `astroARIADNE`

These two packages have to be installed separately outside of `RADPy` due to some technical issues. The instructions to download them are below. Please note that if you are a Windows user, you will need to run a virtual environment that runs a linux or OS distribution. I recommend using WSL. The two packages are have dependencies that are not compatible with Windows machines. 

`SEDFit`:

The installation of this package requires a few additional packages that are unfortunately not compatible with windows machines. This will require the use of a virtual environment like WSL. 

To install this package, use the following command:

```python
pip install git+https://github.com/mkounkel/SEDFit.git
```

In addition, there are filter profiles that need to be downloaded and moved to the directory where `SEDFit` was just installed. The filter profiles needed are:

- GAIA.GAIA3.G
- GAIA.GAIA3.Gbp
- GAIA.GAIA3.Grp
- Hipparcos.Hipparcos.Hp_MvB
- Johnson.H
- Johnson.J
- Johnson.K
- Stromgren.b.dat
- Stromgren.u.dat
- Stromgren.v.dat
- Stromgren.y.dat
- TESS.TESS.Red.dat
- TYCHO.TYCHO.B_MvB
- TYCHO.TYCHO.V_MvB

You can download them from here: <https://github.com/spaceashley/radpy/tree/main/radpy/data>. 

If you are interested in having `RADPy` pull photometry for your star for you, you need to install `astroARIADNE`. This package also has dependencies that are not compatible with Windows machines.

To install this package:

```python 
pip install astroariadne
```

With `astroARIADNE`, you also need to import the necessary dustmaps. 

```python
import dustmaps.sfd
dustmaps.sfd.fetch()
```

However, the dependencies that are required for this package aren't all up to date and/or aren't working properly. This is perfectly fine. You should be able to use the features of the SED fitter without an issue. If there is one, please submit an issue to the repo. 

You can find more information about each package here:

[`SEDFit'`](https://github.com/mkounkel/SEDFit)

[`astroARIADNE`](https://github.com/jvines/astroARIADNE)



To test if the installation worked, import `RADPy`. If you did not get an error, you should be all set. 

```python
import radpy
```

NOTE: 

to _install_, use rsadpy. **Note the 's'**

to _import_, use radpy. **Note that there is no longer an s**

## What does `RADPy` actually do?
`RADPy` accepts data from an arbitrary number of beam-combiners from CHARA and allows the user to fit for the angular diameters (both uniform disk and limb-darkened disk) of single stars. With the fitted angular diameter, the user can also calculate the remaining fundamental stellar parameters of effective temperature, stellar luminosity, and radius of the measured star. The user can also plot the interferometric data with the chosen angular diameter fit (uniform or limb-darkened) which will output a publication ready plot. The plotting is highly customizable to the user's needs, including the type of model plotted, the ability to add the diameter in text to said plot, the binning of the data if the user choses to, and more. 

The core of `RADPy` is a Monte Carlo simulation that involves a custom-built bracket bootstrapping within. A bracket in the realm of interferometry describes a set of data taken at the same time. Several instruments at CHARA span a wavelength range, so for every one observation, there is a span of data points to cover the wavelength ranges. `RADPy` automatically assigns a bracket number to the data once the data files are read in. The bracket numbers are assigned based on time-stamp and for PAVO, based on the same UCOORD and VCOORD measurements (as PAVO data does not output a time stamp). 

For uniform disk diameters, `RADPy` will sample the wavelength of observations on a normal distribution. Within the bracket bootstrapping, the visibilities of each bracket chosen to be fit are sampled on a normal distribution. Using lmfit, the data are then fit using the uniform disk visibility squared equation. The final output results in a list of angular diameters calculated. The final uniform disk diameter is determined by taking the average of the uniform disk diameters and the error is determined by taking the mean absolute deviation. 

For limb-darkened disk diameters, `RADPy` follows a similar structure to the uniform disk diameters. There are a few differences which I'll highlight below:

- One needs the limb-darkening coefficient. To account for the limb-darkening coefficient, the tables of limb-darkening coefficients determined by Claret et al. 2011 are used. Based on the observation band, surface gravity (log g), and the effective temperature (Teff) of the star, `RADPy` will use an interpolated function based on the Claret tables to calculate the limb-darkening coefficient. If the effective temperature is less than 3500 and the surface gravity is between 3.5 and 5, the tables with the PHOENIX models are used. For all other stars, the tables with the ATLAS models are used.
- For each iteration of the MC, `RADPy` calculates a limb-darkening coefficient for each band used (i.e. R-band). Within the bootstrapping, `RADPy` samples the limb-darkening coefficient on a normal distribution using 0.02 has the "error". The limb-darkening coefficient is then used in the full visibility squared equation and the limb-darkened angular diameter is fit.
- To ensure `RADPy` is fitting for the optimal angular diameter, the limb-darkened disk fitting function will iterate until minimal change between the previous angular diameter and the one just calculated is seen. For robustness, the effective temperature is also checked as well. Minimal change is defined as being less than or equal to 0.05% difference.

## Tutorial notebooks

For a tutorial for single stars, go here: <https://github.com/spaceashley/radpy/blob/main/tests/SingleStarTutorial.ipynb>

For a tutorial on how to use batch mode, go here: <https://github.com/spaceashley/radpy/blob/main/tests/BatchModeTutorial.ipynb>

For a tutorial on how to use the SED fitting feature, go here: <https://github.com/spaceashley/radpy/blob/main/tests/SED%20Fitting%20Tutorial.ipynb>

## How to Cite

If you use `RADPy` in your research, please cite it through the following:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17488122.svg)](https://doi.org/10.5281/zenodo.19444006)

Official paper is coming soon.

In addition, if you decide to use the SED fitting feature and/or the photometry extraction for SED fitting, please cite the following as well:

`astroARIADNE` (if using the photometry extraction for SED fitting):
```
@ARTICLE{2022MNRAS.tmp..920V,
       author = {{Vines}, Jose I. and {Jenkins}, James S.},
        title = "{ARIADNE: Measuring accurate and precise stellar parameters through SED fitting}",
      journal = {\mnras},
     keywords = {stars:atmospheres, methods:data analysis, stars:fundamental parameters, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = apr,
          doi = {10.1093/mnras/stac956},
archivePrefix = {arXiv},
       eprint = {2204.03769},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.tmp..920V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

`SEDFit` (if using the SED fitting feature):
```
@software{sedfit,
	author = {{Kounkel}, Marina},
	doi = {10.5281/zenodo.8076500},
	month = jun,
	publisher = {Zenodo},
	title = {SEDFit},
	url = {https://doi.org/10.5281/zenodo.8076500},
	year = 2023}
```

## Contact
- Ashley Elliott (aelli76@lsu.edu)

## Logo Credits
Logo was designed by Emelly Tiburcio from LSU and made digital by Olivia Crowell from LSU.
