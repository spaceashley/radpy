![RADPy logo](https://github.com/spaceashley/radpy/tests/test_data/RADPy v10.png "RADPy logo")


# Robust Angular Diameters in Python (`RADPy`)
## Introduction to `RADPy`

`RADPy` stands for Robust Angular Diameters in Python. This was created to allow for multi-wavelength fits for angular diameters of stars measured with interferometric methods. Currently `RADPy` only has compatibility with the instruments on the Center for High Angular Resolution Astronomy (CHARA) Array. `RADPy` is currently configured for the following instruments at CHARA:

- Classic/CLIMB
- PAVO
- VEGA
- MIRC-X (in progress)

The remaining CHARA instruments SPICA and MYSTIC will be integrated soon.

## To install:

**Please read this section carefully!!**

Simply use pip to install `RADPy`. Due to naming conflicts, to install `RADPy`, you must use "rsadpy". 

`pip install rsadpy`

The installation should also install all necessary additional packages you need to run everything. Just in case, here is a list of all the necessary packages that aren't default:
- `lmfit`
- `astropy`
- `astroquery`
- `gaiadr3-zeropoint`

To test if the installation worked, import `RADPy`. If you did not get an error, you should be all set. 

`import radpy`

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

## What is in this notebook?

This tutorial is designed to show the general steps of how a user should implement `RADPy` for your own angular diameter fitting needs for single stars. This notebook will go over how to install, how to import the necessary modules from `RADPy`, how to read in your data, how to perform initial fits, how to run the Monte Carlo simulations, how to calculate the remaining stellar parameters, and how to generate publicaiton ready plots. This tutorial is useful for those that want to step through every part of `RADPy` and see the outputs of each step. 

## How to Cite

If you use `RADPy` in your research, please cite it as:

Elliott, Ashley (2025). *RADPy: Robust Angular Diameters in Python*. Version 0.3.9. Available on PyPI: https://pypi.org/project/rsadpy. Source code: https://github.com/spaceashley/radpy

## Contact
- Ashley Elliott (aelli76@lsu.edu)
