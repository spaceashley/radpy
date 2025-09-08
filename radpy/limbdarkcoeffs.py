import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from radpy.config import rppath, rapath, kppath, kapath, hppath, hapath, jppath, japath, rpfpath, rafpath, kpfpath, kafpath, hpfpath, hafpath, jpfpath, jafpath

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


def ldc_calc(teff, logg, feh, filt, method, verbose = False):
    ###########################################################################
    # Function: ldc_calc                                                      #
    # Inputs: teff -> the effective temperature                               #
    #         logg -> the surface gravity                                     #
    #         feh -> the metallicity                                          #
    #         filt -> the filter for the LDC                                  #
    #         method -> the method for finding the LDC                        #
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
    # Functions for teff < 3500 and using Least squares
    low_teff_Lfuncs = {
        'r': RPfunc,
        'h': HPfunc,
        'k': KPfunc,
        'j': JPfunc
    }
    # Functions for teff >= 3500 and using Least squares
    high_teff_Lfuncs = {
        'r': RAfunc,
        'h': HAfunc,
        'k': KAfunc,
        'j': JAfunc
    }

    # Functions for teff < 3500 and using flux conservation
    low_teff_Ffuncs = {
        'r': RPFfunc,
        'h': HPFfunc,
        'k': KPFfunc,
        'j': JPFfunc
    }
    # Functions for teff >= 3500 and using flux conservation
    high_teff_Ffuncs = {
        'r': RAFfunc,
        'h': HAFfunc,
        'k': KAFfunc,
        'j': JAFfunc
    }
    if method.lower() == 'l':
        if teff < 3500 and 3.5 <=logg <=5:
            if verbose:
                print('Using the Phoenix models and Least Squares method.')
            func = low_teff_Lfuncs.get(filt)
            if func is None:
                raise ValueError(f"Unknown filter: {filt}")
            mu = func([teff, logg])[0]
        else:
            if verbose:
                print("Using the Atlas models and Least Squares method.")
            func = high_teff_Lfuncs.get(filt)
            if func is None:
                raise ValueError(f"Unknown filter: {filt}")
            mu = func([teff, logg, feh])[0]
        return mu
    if method.lower() == 'f':
        if teff < 3500 and 3.5 <= logg <= 5:
            if verbose:
                print('Using the Phoenix models and flux conservation method.')
            func = low_teff_Ffuncs.get(filt)
            if func is None:
                raise ValueError(f"Unknown filter: {filt}")
            mu = func([teff, logg])[0]
        else:
            if verbose:
                print("Using the Atlas models and flux conservation method.")
            func = high_teff_Ffuncs.get(filt)
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

ldc_RPF = pd.read_csv(rpfpath)
ldc_RAF = pd.read_csv(rafpath)
ldc_KPF = pd.read_csv(kpfpath)
ldc_KAF = pd.read_csv(kafpath)
ldc_HPF = pd.read_csv(hpfpath)
ldc_HAF = pd.read_csv(hafpath)
ldc_JPF = pd.read_csv(jpfpath)
ldc_JAF = pd.read_csv(jafpath)

RPfunc = create_ldc_functions(ldc_RP)
RAfunc = create_ldc_functions(ldc_RA)
HPfunc = create_ldc_functions(ldc_HP)
HAfunc = create_ldc_functions(ldc_HA)
KPfunc = create_ldc_functions(ldc_KP)
KAfunc = create_ldc_functions(ldc_KA)
JPfunc = create_ldc_functions(ldc_JP)
JAfunc = create_ldc_functions(ldc_JA)

RPFfunc = create_ldc_functions(ldc_RPF)
RAFfunc = create_ldc_functions(ldc_RAF)
HPFfunc = create_ldc_functions(ldc_HPF)
HAFfunc = create_ldc_functions(ldc_HAF)
KPFfunc = create_ldc_functions(ldc_KPF)
KAFfunc = create_ldc_functions(ldc_KAF)
JPFfunc = create_ldc_functions(ldc_JPF)
JAFfunc = create_ldc_functions(ldc_JAF)