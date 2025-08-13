import pandas as pd
import numpy as np
import astropy.io.fits as fits


def oifits_to_pandas(filename, inst_name):
    # Reads in an oifits file and converts into a pandas dataframe with the necessary information needed for RADPy
    # Extracts the following data:
    #   V2, dv2, ucoord, vcoord, mjd, time, effective wavelength, and effective bandwidth
    # Converts the wavelengths and bandwidth arrays into lists of lists to match the V2 and V2err lists
    # creates a dataframe
    # Sorts the dataframe by MJD and then assigns a bracket number based on the date groupings
    # Explodes the dataframe by extracting out each list for V2, V2_err, wavelength, and bandwidth
    # returns the exploded, sorted, and bracket labeled df

    data = fits.open(filename)
    count = check_multidatasets(data)
    new_df = unpack_multidatasets(data, count)
    sorted_df = brackets(new_df, inst_name)

    sorted_df['zipped'] = sorted_df.apply(
        lambda row: list(zip(row['V2'], row['V2_err'], row['Eff_wave[m]'], row['Eff_band[m]'])), axis=1)
    df_exploded = sorted_df.explode('zipped').reset_index(drop=True)
    df_exploded[['V2', 'V2_err', 'Eff_wave[m]', 'Eff_band[m]']] = pd.DataFrame(df_exploded['zipped'].tolist(),
                                                                               index=df_exploded.index)
    df_exploded = df_exploded.drop(columns='zipped')
    df_exploded['Instrument'] = [inst_name] * len(df_exploded)

    return df_exploded


def filename_extension(filename, inst_name, verbose=False, debug = False):
    #########################################################################
    # Function: filename_extension                                          #
    # Inputs: filename -> name of data file                                 #
    #         inst_name -> Instrument identifier                            #
    #                      C - Classic                                      #
    #                      P - PAVO                                         #
    #                      V - VEGA                                         #
    #                      M - MIRCX                                        #
    #                      MY - MYSTIC                                      #
    #                      S - SPICA                                        #
    #         verbose -> default is False, if true, allows print statements #
    # Outputs: data frame of the data with the instrument added as a column #
    # What it does:                                                         #
    #         1. Checks what format the file is in                          #
    #         If .csv:                                                      #
    #            2a. Uses pandas.read_csv to read in the file               #
    #            3a. Adds the Instrument column                             #
    #         If .txt:                                                      #
    #            2b. Opens the file and reads in the first line             #
    #            3b. Checks what delimiter the file is using                #
    #            4b. Reads in the file                                      #
    #            5b. Adds the instrument column                             #
    #         If .oifits or .fits:                                          #
    #            2c. uses the oifits_to_pandas function                     #
    #         Returns the dataframe, and number of brackets                 #
    #########################################################################

    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
        df['Instrument'] = [inst_name] * len(df)
        sorted_df = brackets(df, inst_name)
        num_brackets = sorted_df['Bracket'].max()
        print('Number of brackets:', num_brackets)
        return sorted_df, num_brackets
        # return sorted_df
    elif filename.endswith('.txt'):
        header = None
        data_start = 0
        with open(filename, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Remove '#' and newline, then split by pipe (or whatever header delimiter you expect)
                header = [col.strip() for col in line.strip()[1:].split('|')]
                data_start = i + 1
                if debug:
                    print(f"Header detected: {header}")
                break

        # Read the data lines (skip header and comments)
        # Drop empty lines and comments
        data_lines = [l for l in lines[data_start:] if l.strip() and not l.strip().startswith('#')]

        # Save to a temp string buffer for pandas
        from io import StringIO
        data_str = ''.join(data_lines)
        df = pd.read_csv(StringIO(data_str), sep=r'\s+', header=None, engine='python')
        if header and len(header) == df.shape[1]:
            df.columns = header
        df['Instrument'] = [inst_name] * len(df)
        sorted_df = brackets(df, inst_name)
        num_brackets = sorted_df['Bracket'].max()
        if verbose:
            print('Number of brackets:', num_brackets)
        return sorted_df, num_brackets


    elif filename.endswith('.oifits') or filename.endswith('.fits'):
        df = oifits_to_pandas(filename, inst_name)
        num_brackets = df['Bracket'].max()
        print('Number of brackets:', num_brackets)
        return df, num_brackets
        # return df

    else:
        header = None
        data_start = 0
        with open(filename, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Remove '#' and newline, then split by pipe (or whatever header delimiter you expect)
                header = [col.strip() for col in line.strip()[1:].split('|')]
                data_start = i + 1
            if debug:
                print(f"Header detected: {header}")
            break

        # Read the data lines (skip header and comments)
        # Drop empty lines and comments
        data_lines = [l for l in lines[data_start:] if l.strip() and not l.strip().startswith('#')]

        # Save to a temp string buffer for pandas
        from io import StringIO
        data_str = ''.join(data_lines)
        df = pd.read_csv(StringIO(data_str), sep=r'\s+', header=None, engine='python')
        if header and len(header) == df.shape[1]:
            df.columns = header
        df['Instrument'] = [inst_name] * len(df)
        sorted_df = brackets(df, inst_name)
        num_brackets = sorted_df['Bracket'].max()
        if verbose:
            print('Number of brackets:', num_brackets)
        return sorted_df, num_brackets

def brackets(df, instrument):
    # bracket generator
    # PAVO brackets are assigned via same baseline
    # Classic has no brackets
    # MIRCX/MYSTIC/SPICA are by MJD
    # Vega is uncertain for now but will be date im fairly certain
    if instrument == 'M' or instrument == 'm':
        pd.set_option('display.float_format', '{:.12f}'.format)
        sorted_df = df.sort_values(by='MJD')
        sorted_df['Bracket'] = sorted_df.groupby('MJD').ngroup() + 1
        return sorted_df
    if instrument == 'P' or instrument == 'p':
        pd.set_option('display.float_format', '{:.12f}'.format)
        sorted_df = df.sort_values(by='U(meters)')
        sorted_df['Bracket'] = sorted_df.groupby('U(meters)').ngroup() + 1
        return sorted_df
    if instrument == 'C' or instrument == 'c':
        pd.set_option('display.float_format', '{:.12f}'.format)
        df['Bracket'] = [1] * len(df)
        return df
    if instrument == 'S' or instrument == 's':
        pd.set_option('display.float_format', '{:.12f}'.format)
        sorted_df = df.sort_values(by='UCOORD[m]')
        sorted_df['Bracket'] = sorted_df.groupby('UCOORD[m]').ngroup() + 1
        return sorted_df
    if instrument == 'My' or instrument == 'my':
        pd.set_option('display.float_format', '{:.12f}'.format)
        sorted_df = df.sort_values(by='MJD')
        sorted_df['Bracket'] = sorted_df.groupby('MJD').ngroup() + 1
        return sorted_df
    if instrument == 'V' or instrument == 'v':
        pd.set_option('display.float_format', '{:.12f}'.format)
        sorted_df = df.sort_values(by='MJD')
        sorted_df['Bracket'] = sorted_df.groupby('MJD').ngroup() + 1
        return sorted_df


def combined(*dfs, fulldf=False):
    # combines the data into one big data frame if needed

    b = pd.concat([df['B'] for df in dfs], ignore_index=True)
    v2 = pd.concat([df['V2'] for df in dfs], ignore_index=True)
    dv2 = pd.concat([df['dV2'] for df in dfs], ignore_index=True)
    wave = pd.concat([df['Wave'] for df in dfs], ignore_index=True)
    band = pd.concat([df['Band'] for df in dfs], ignore_index=True)
    brack = pd.concat([df['Bracket'] for df in dfs], ignore_index=True)
    inst = pd.concat([df['Instrument'] for df in dfs], ignore_index=True)


    if not fulldf:
        return b, v2, dv2, wave, band, brack, inst

    if fulldf:
        return pd.DataFrame({
            'B': b, 'V2': v2, 'dV2': dv2,
            'Wave': wave, 'Band': band,
            'Bracket': brack, 'Instrument': inst})

def check_multidatasets(fitsfile, verbose = False):
    count = 0
    for i, hdu in enumerate(fitsfile):
        name = 'OI_VIS2'
        if hdu.name == name:
            if hdu.ver:
                count +=1
    if verbose:
        print("Nights:", count)
    return count


def unpack_multidatasets(fitsfile, count, verbose = False):
    if count > 1:
        v2_list = []
        dv2_list = []
        uc_list = []
        vc_list = []
        mjd_list = []
        time_list = []
        wl_list = []
        band_list = []
        if verbose:
            print("Multiple nights")
        for i in range(count):
            if verbose:
                print("Version:", i + 1)
            v2 = fitsfile["OI_VIS2", i + 1].data["VIS2DATA"]
            dv2 = fitsfile["OI_VIS2", i + 1].data["VIS2ERR"]
            ucoord = fitsfile["OI_VIS2", i + 1].data["UCOORD"]
            vcoord = fitsfile["OI_VIS2", i + 1].data["VCOORD"]
            mjd = fitsfile["OI_VIS2", i + 1].data["MJD"]
            time = fitsfile["OI_VIS2", i + 1].data["TIME"]
            wl = fitsfile["OI_WAVELENGTH", i + 1].data["EFF_WAVE"]
            band = fitsfile["OI_WAVELENGTH", i + 1].data["EFF_BAND"]

            wll = [wl.tolist() for _ in range(len(v2))]
            bandl = [band.tolist() for _ in range(len(v2))]

            v2_list.append(v2)
            dv2_list.append(dv2)
            uc_list.append(ucoord)
            vc_list.append(vcoord)
            mjd_list.append(mjd)
            time_list.append(time)
            wl_list.append(wll)
            band_list.append(bandl)

        pd.set_option('display.float_format', '{:.12f}'.format)
        df = pd.DataFrame({'MJD': mjd_list, 'Time': time_list, 'V2': v2_list, 'V2_err': dv2_list,
                           'Eff_wave[m]': wl_list, 'Eff_band[m]': band_list, 'UCOORD[m]': uc_list,
                           'VCOORD[m]': vc_list})
        df_e = df.apply(pd.Series.explode)
        return df_e
    else:
        if verbose:
            print("Only one night.")
        v2 = fitsfile["OI_VIS2"].data["VIS2DATA"]
        dv2 = fitsfile["OI_VIS2"].data["VIS2ERR"]
        ucoord = fitsfile["OI_VIS2"].data["UCOORD"]
        vcoord = fitsfile["OI_VIS2"].data["VCOORD"]
        mjd = fitsfile["OI_VIS2"].data["MJD"]
        time = fitsfile["OI_VIS2"].data["TIME"]
        wl = fitsfile["OI_WAVELENGTH"].data["EFF_WAVE"]
        band = fitsfile["OI_WAVELENGTH"].data["EFF_BAND"]

        wl_list = [wl.tolist() for _ in range(len(v2))]
        band_list = [band.tolist() for _ in range(len(v2))]

        pd.set_option('display.float_format', '{:.12f}'.format)
        df = pd.DataFrame({'MJD': mjd, 'Time': time, 'V2': v2.tolist(), 'V2_err': dv2.tolist(),
                           'Eff_wave[m]': wl_list, 'Eff_band[m]': band_list, 'UCOORD[m]': ucoord, 'VCOORD[m]': vcoord})

        return df

class InterferometryData:
    def __init__(self, df, instrument_code):
        self.raw = df.copy()
        self.instrument = instrument_code.lower()
        self.cleaned = None
        self.process()

    def process(self):
        raise NotImplementedError("Subclasses must implement the .process() method")

    def make_df(self, LDC=None):
        n = len(self.B)

        if LDC is None:
            ldc_col = [None] * n
        elif np.isscalar(LDC):
            ldc_col = [LDC] * n
        elif isinstance(LDC, (list, np.ndarray, pd.Series)) and len(LDC) == n:
            ldc_col = LDC
        else:
            raise ValueError(f"LDC must be None, a scalar, or a list/array of length {n}, got {LDC}.")

        return pd.DataFrame({
            "B": self.B,
            "V2": self.V2,
            "dV2": self.dV2,
            "Wave": self.Wave,
            "LDC": ldc_col,
            "Band": self.Band,
            "Bracket": self.Bracket,
            "Instrument": [self.instrument] * len(self.B)

        })

    def make_ldmcdf(self, LDC):
        # spf = self.B / self.Wave

        return pd.DataFrame({
            # "Spf":spf,
            "B": self.B,
            "V2": self.V2,
            "dV2": self.dV2,
            "LDC": LDC,
            "Bracket": self.Bracket,
            "Instrument": [self.instrument] * len(self.V2)
        })


class PavoData(InterferometryData):
    def __init__(self, df):
        super().__init__(df, instrument_code='p')

    def process(self):
        df = self.raw.dropna(subset=['V2', 'sigma_V2'])
        self.cleaned = df

        self.V2 = df['V2']
        self.dV2 = df['sigma_V2']
        self.U = df['U(meters)']
        self.V = df['V(meters)']
        self.B = np.sqrt(self.U ** 2 + self.V ** 2)
        self.Wave = self.B / df['B/lambda']
        self.Band = pd.Series(np.full(len(df), 5e-9))  # 5 nm
        self.Bracket = df['Bracket']


class ClassicData(InterferometryData):
    def __init__(self, df):
        super().__init__(df, instrument_code='c')

    def process(self):
        df = self.raw.dropna(subset=['Vis', 'Vis_e'])  # clean NaNs
        self.cleaned = df
        v = df['Vis']
        dv = df['Vis_e']

        self.B = df['B']
        self.V2 = v ** 2
        self.dV2 = self.V2 * np.sqrt(2 * (dv / v) ** 2)
        self.Wave = pd.Series(np.full(len(self.B), 2.1329e-6))  # meters
        self.Band = pd.Series(np.full(len(self.B), 5e-9))  # meters (5 nm)
        self.Bracket = df['Bracket']


class VegaData(InterferometryData):
    def __init__(self, df):
        super().__init__(df, instrument_code='v')

    def process(self):
        df = self.raw.dropna(subset=['V2', 'sigma_sys', 'sigma_stat'])  # clean NaNs
        self.cleaned = df
        dv2_sys = df['sigma_sys']
        dv2_stat = df['sigma_stat']

        self.B = df['Baseline length']
        self.V2 = df['V2']
        self.dV2 = np.sqrt((dv2_sys ** 2) + (dv2_stat) ** 2)
        self.Wave = df['lambda'] * 1e-9
        self.Band = pd.Series(np.full(len(df), 5e-9))
        self.Bracket = df['Bracket']


class MircxData(InterferometryData):
    def __init__(self, df):
        super().__init__(df, instrument_code='m')

    def process(self):
        df = self.raw.dropna(subset=['V2', 'V2_err'])  # clean NaNs
        self.cleaned = df

        ucoord = (df['UCOORD[m]'].values).astype('float')
        vcoord = (df['VCOORD[m]'].values).astype('float')
        self.B = np.sqrt((ucoord ** 2) + (vcoord ** 2))
        self.V2 = df['V2']
        self.dV2 = df['V2_err']
        self.Wave = df['Eff_wave[m]']
        self.Band = df['Eff_band[m]']
        self.Bracket = df['Bracket']

class MysticData(InterferometryData):
    def __init__(self, df):
        super().__init__(df, instrument_code='my')

    def process(self):
        df = self.raw.dropna(subset=['V2', 'V2_err'])  # clean NaNs
        self.cleaned = df

        ucoord = (df['UCOORD[m]'].values).astype('float')
        vcoord = (df['VCOORD[m]'].values).astype('float')
        self.B = np.sqrt((ucoord ** 2) + (vcoord ** 2))
        self.V2 = df['V2']
        self.dV2 = df['V2_err']
        self.Wave = df['Eff_wave[m]']
        self.Band = df['Eff_band[m]']
        self.Bracket = df['Bracket']

class SpicaData(InterferometryData):
    def __init__(self, df):
        super().__init__(df, instrument_code='s')

    def process(self):
        df = self.raw.dropna(subset=['V2', 'V2_err'])  # clean NaNs
        self.cleaned = df

        ucoord = df['UCOORD[m]']
        vcoord = df['VCOORD[m]']
        self.B = np.sqrt((ucoord ** 2) + (vcoord ** 2))
        self.V2 = df['V2']
        self.dV2 = df['V2_err']
        self.Wave = df['Eff_wave[m]']
        self.Band = df['Eff_band[m]']
        self.Bracket = df['Bracket']


