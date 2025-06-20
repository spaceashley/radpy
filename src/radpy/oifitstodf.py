from astropy.io import fits
import pandas as pd

# handler for oifits files
def oifits_to_pandas(filename):
    # Reads in an oifits file and converts into a pandas dataframe with the necessary information needed for RADPy
    # Extracts the following data:
    #   V2, dv2, ucoord, vcoord, mjd, time, effective wavelength, and effective bandwidth
    # Converts the wavelengths and bandwidth arrays into lists of lists to match the V2 and V2err lists
    # creates a dataframe
    # Sorts the dataframe by MJD and then assigns a bracket number based on the date groupings
    # returns the sorted and bracket labeled df

    data = fits.open(filename)
    v2 = data["OI_VIS2"].data["VIS2DATA"]
    dv2 = data["OI_VIS2"].data["VIS2ERR"]
    ucoord = data["OI_VIS2"].data["UCOORD"]
    vcoord = data["OI_VIS2"].data["VCOORD"]
    mjd = data["OI_VIS2"].data["MJD"]
    time = data["OI_VIS2"].data["TIME"]
    wl = data["OI_WAVELENGTH"].data["EFF_WAVE"]
    band = data["OI_WAVELENGTH"].data["EFF_BAND"]

    wl_list = [wl.tolist() for _ in range(len(v2))]
    band_list = [band.tolist() for _ in range(len(v2))]

    pd.set_option('display.float_format', '{:.12f}'.format)
    df = pd.DataFrame({'MJD': mjd, 'Time': time, 'V2': v2.tolist(), 'V2_Err': dv2.tolist(),
                       'EFF_Wave': wl_list, 'EFF_band': band_list, 'UCOORD': ucoord, 'VCOORD': vcoord})

    sorted_df = df.sort_values(by='MJD')
    sorted_df['Bracket'] = sorted_df.groupby('MJD').ngroup() + 1

    return sorted_df