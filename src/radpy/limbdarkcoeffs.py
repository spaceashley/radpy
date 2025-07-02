import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from radpy.config import rppath, rapath, kppath, kapath, hppath, hapath, jppath, japath

def create_ldc_functions(df):
    ############################################################################
    # Function: create_ldc_function                                            #
    # Inputs: df -> the Claret data tables                                     #
    # Outputs: ldc_func -> the function to interpolate the ldc for the filter  #
    # What it does:                                                            #
    #      1. Drops any nans in the table                                      #
    #      2. If the model is the Phoenix models, the interpolator only takes  #
    #         the teff and logg and returns the mus.                           #
    #      3. If the model is the Atlas models, the interpolator takes the     #
    #         teff, logg, and metallicity, and returns the mus.                #
    #      4. Returns the interpolated function                                #
    ############################################################################
    df_clean = df.dropna()
    if df_clean['Mod'][0] == 'P':
        ldc_func = LinearNDInterpolator((df_clean['Teff'], df_clean['logg']), df_clean['u'])
        return ldc_func
    else:
        points = np.stack([df_clean['Teff'], df_clean['logg'], df_clean['Z']], axis = -1)
        ldc_func = LinearNDInterpolator(points, df_clean['u'])
        return ldc_func


def ldc_calc(teff, logg, feh, filt, verbose = False):
    ###########################################################################
    # Function: ldc_calc                                                      #
    # Inputs: teff -> the effective temperature                               #
    #         logg -> the surface gravity                                     #
    #         feh -> the metallicity                                          #
    #         filt -> the filter for the LDC                                  #
    #         verbose -> if set to true, prints the print statements          #
    #                    defaults to not print                                #
    # Outputs: the limb-darkening coefficient                                 #
    # What it does:                                                           #
    #       1. returns the lower-case value for the filter to make sure there #
    #          is no case mismatch                                            #
    #       2. defines a dictionary for the Phoenix functions                 #
    #          Phoenix is used if the temps are below 3500 K                  #
    #       3. defines a dictionary for the Atlas functions                   #
    #          Atlas is used if the temps are above 3500 K                    #
    #       4. if temps are below 3500 and the logg is between 3.5 and 5,     #
    #          uses the Phoenix models to generate a limb darkening coeff.    #
    #       5. if temps are above 3500, uses the Atlas models to generate     #
    #          a limb darkening coeff.                                        #
    #       6. Returns the limb darkening coefficient.                        #
    ###########################################################################
    filt = filt.lower()
    # Functions for teff < 3500
    low_teff_funcs = {
        'r': RPfunc,
        'h': HPfunc,
        'k': KPfunc,
        'j': JPfunc
    }
    # Functions for teff >= 3500
    high_teff_funcs = {
        'r': RAfunc,
        'h': HAfunc,
        'k': KAfunc,
        'j': JAfunc
    }
    if teff < 3500 and 3.5 <=logg <=5:
        if verbose:
            print('Using the Phoenix models.')
        func = low_teff_funcs.get(filt)
        if func is None:
            raise ValueError(f"Unknown filter: {filt}")
        mu = func([teff, logg])[0]
    else:
        if verbose:
            print("Using the Atlas models.")
        func = high_teff_funcs.get(filt)
        if func is None:
            raise ValueError(f"Unknown filter: {filt}")
        mu = func([teff, logg, feh])[0]
    return mu

ldc_RP = pd.read_csv(rppath)
ldc_RA = pd.read_csv(rapath)
ldc_KP = pd.read_csv(kppath)
ldc_KA = pd.read_csv(kapath)
ldc_HP = pd.read_csv(hppath)
ldc_HA = pd.read_csv(hapath)
ldc_JP = pd.read_csv(jppath)
ldc_JA = pd.read_csv(japath)

RPfunc = create_ldc_functions(ldc_RP)
RAfunc = create_ldc_functions(ldc_RA)
HPfunc = create_ldc_functions(ldc_HP)
HAfunc = create_ldc_functions(ldc_HA)
KPfunc = create_ldc_functions(ldc_KP)
KAfunc = create_ldc_functions(ldc_KA)
JPfunc = create_ldc_functions(ldc_JP)
JAfunc = create_ldc_functions(ldc_JA)