import io
import math
import pickle
import contextlib
import numpy as np
import pandas as pd
import pkg_resources
from SEDFit.sed import SEDFit
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.integrate import quad
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from radpy.config import svopath
import warnings
warnings.filterwarnings("ignore")

#%% Rewrite of the SEDFit functions
def definefilter(self, tmass=True, cousins=True, gaia=True, galex=True, johnson=True,
                 panstarrs=True, sdss=True, wise=True, xmm=True, spitzer=True, tycho=True, hip=True, tess=True,
                 stromgren=True,
                 new=True, empty=False, **kwargs):
    idx = []
    if new:
        for i in range(len(self.sed)):
            self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace(':', '.').replace('/', '.')
            self.sed['width'] = 0 * u.AA
        if tmass:
            if new:
                self.sed['sed_filter'] = self.sed['sed_filter'].astype(object)
                for i in range(len(self.sed)):
                    # self.sed['sed_filter'][i]=self.sed['sed_filter'][i].replace('Johnson.J','2MASS.J')
                    # self.sed['sed_filter'][i]=self.sed['sed_filter'][i].replace('Johnson.H','2MASS.H')
                    # self.sed['sed_filter'][i]=self.sed['sed_filter'][i].replace('Johnson.K','2MASS.Ks')
                    self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace('UKIDSS.J', '2MASS.J')
                    self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace('UKIDSS.H', '2MASS.H')
                    self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace('UKIDSS.Ks', '2MASS.Ks')
                    self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace('UKIDSS.K', '2MASS.Ks')
                    self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace('VISTA.J', '2MASS.J')
                    self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace('VISTA.H', '2MASS.H')
                    self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace('VISTA.Ks', '2MASS.Ks')
                    self.sed['sed_filter'][i] = self.sed['sed_filter'][i].replace('VISTA.K', '2MASS.Ks')
                    if self.sed['sed_filter'][i] == '2MASS:K': self.sed['sed_filter'][i] = '2MASS.Ks'
                self.sed['sed_filter'] = self.sed['sed_filter'].astype(str)

            filters = ['2MASS.J', '2MASS.H', '2MASS.Ks']
            width = np.array([0.152, 0.241, 0.251]) / 2 * u.micron
            la = np.array([12350, 16620, 21590]) * u.AA
            ind = np.array([39, 41, 43])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if cousins:
            filters = ['Cousins.U', 'Cousins.B', 'Cousins.V', 'Cousins.R', 'Cousins.I']
            width = np.array([0.0657, 0.10117, 0.55014, 0.13811, 0.101107]) / 2 * u.micron
            la = np.array([3511.89, 4382.77, 5501.4, 6414.42, 7858.32]) * u.AA
            ind = np.array([7, 12, 24, 27, 33])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if gaia:
            filters = ['GAIA.GAIA3.G', 'GAIA.GAIA3.Gbp', 'GAIA.GAIA3.Grp']
            width = np.array([0.405297, 0.21575, 0.292444]) / 2 * u.micron
            la = np.array([6217.59, 5109.71, 7769.02]) * u.AA
            ind = np.array([28, 18, 34])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if galex:
            filters = ['GALEX.FUV', 'GALEX.NUV']
            width = np.array([0.026557, 0.076831]) / 2 * u.micron
            la = np.array([1535.08, 2300.78]) * u.AA
            ind = np.array([0, 3])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if johnson:
            filters = ['Johnson.U', 'Johnson.B', 'Johnson.V', 'Johnson.R', 'Johnson.I', 'Johnson.J', 'Johnson.K',
                       'Johnson.H']
            width = np.array([0.0657, 0.09725, 0.08898, 0.207, 0.2316, 0.319355, 0.5785, 0.286263]) / 2 * u.micron
            la = np.array([3511.89, 4390.21, 5501.4, 6819.05, 8674.2, 12317.3, 21735.85, 16396.38]) * u.AA
            ind = np.array([9, 13, 23, 29, 36, 40, 44, 42])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if panstarrs:
            filters = ['PAN-STARRS.PS1.g', 'PAN-STARRS.PS1.r', 'PAN-STARRS.PS1.i', 'PAN-STARRS.PS1.z',
                       'PAN-STARRS.PS1.y']
            width = np.array([0.105308, 0.125241, 0.120662, 0.099772, 0.063898]) / 2 * u.micron
            la = np.array([4849.11, 6201.2, 7534.96, 8674.2, 9627.79]) * u.AA
            ind = np.array([16, 25, 30, 35, 38])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if sdss:
            filters = ['SDSS.u', 'SDSS.g', 'SDSS.r', 'SDSS.i', 'SDSS.z']
            width = np.array([0.054097, 0.106468, 0.105551, 0.110257, 0.116401]) / 2 * u.micron
            la = np.array([3556.52, 4702.5, 6175.58, 7489.98, 8946.71]) * u.AA
            ind = np.array([8, 17, 26, 31, 37])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if wise:
            filters = ['WISE.W1', 'WISE.W2', 'WISE.W3']
            width = np.array([0.662642, 0.662642, 5.505523]) / 2 * u.micron
            la = np.array([33526, 46028, 115608]) * u.AA
            ind = np.array([45, 48, 51])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if xmm:
            filters = ['XMM-OT.V', 'XMM-OT.B', 'XMM-OT.U', 'XMM-OT.UVW1', 'XMM-OT.UVM2', 'XMM-OT.UVW2']
            width = np.array([0.069956, 0.091023, 0.067513, 0.074398, 0.046194, 0.04355]) / 2 * u.micron
            la = np.array([5450.47, 4368.97, 3465.51, 2895.36, 2284.66, 2041.68]) * u.AA
            ind = np.array([21, 14, 6, 4, 2, 1])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if spitzer:
            filters = ['Spitzer.IRAC.3.6', 'Spitzer.IRAC.4.5', 'Spitzer.IRAC.5.8', 'Spitzer.IRAC.8.0',
                       'Spitzer.MIPS.24']
            width = np.array([0.683618, 0.864992, 1.256117, 2.52885, 5.296286]) / 2 * u.micron
            la = np.array([35378.41, 44780.49, 56961.78, 77978.40, 235937.78]) * u.AA
            ind = np.array([46, 47, 49, 50, 52])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if hip:
            filters = ['Hipparcos.Hipparcos.Hp_MvB']
            width = np.array([0.240569]) / 2 * u.micron
            la = np.array([5338.25]) * u.AA
            ind = np.array([20])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if tycho:
            filters = ['TYCHO.TYCHO.B_MvB', 'TYCHO.TYCHO.V_MvB']
            width = np.array([0.074139, 0.113355]) / 2 * u.micron
            la = np.array([4194.96, 5300.19]) * u.AA
            ind = np.array([11, 19])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if tess:
            filters = ['TESS.TESS.Red']
            width = np.array([0.389865]) / 2 * u.micron
            la = np.array([7697.6]) * u.AA
            ind = np.array([32])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))
        if stromgren:
            filters = ['Stromgren.u', 'Stromgren.v', 'Stromgren.b', 'Stromgren.y']
            width = np.array([0.037244, 0.022398, 0.020208, 0.02533]) / 2 * u.micron
            la = np.array([3443.6, 4105.46, 4666.25, 5475.17]) * u.AA
            ind = np.array([5, 10, 15, 22])
            idx.extend(self.selectflux(filters, width, la, ind, new=new, empty=empty))

        self.sed = self.sed[np.array(idx).flatten()]
        a = np.argsort(self.sed['la'])
        self.sed = self.sed[a]
        return


def downloadflux(self, userinput, deletevot=True, **kwargs):
    target = str(self.ra) + '%20' + str(self.dec)
    good = False
    if userinput is not None:
        self.sed = userinput
        # print("Using User input table")
        # print(self.sed)
        return
    else:
        if self.vizier_filename is not None:
            vot_fn = self.vizier_filename
            self.sed = Table.read(vot_fn)
            # print("Using VOT table")
            good = True
            if deletevot:
                os.remove(vot_fn)
        else:
            attempts, maxattempts = 0, 4
            while attempts < maxattempts:
                try:
                    target = str(self.ra) + '%20' + str(self.dec)
                    self.sed = Table.read(f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={target}&-c.rs={self.radius}")
                    # print("Using vizier search")
                    good = True
                    break
                except:
                    attempts += 1
                    time.sleep(attempts ** 2)
        if not good:
            self.sed = []
            # print("Nothing worked")
            return

        self.sed['index'] = int(0)
        self.sed['la'] = self.sed['sed_freq'].to(u.AA, equivalencies=u.spectral())
        a = np.where((self.sed['la'] < 30 * u.micron) & (self.sed['la'] > 1000 * u.AA))[0]
        self.sed = self.sed[a]

        self.sed["sed_flux"] = self.sed["sed_flux"].to((u.erg / u.s / (u.cm ** 2) / u.AA),
                                                       equivalencies=u.spectral_density(self.sed['la'].data * u.AA))
        self.sed["sed_eflux"] = self.sed["sed_eflux"].to((u.erg / u.s / (u.cm ** 2) / u.AA),
                                                         equivalencies=u.spectral_density(self.sed['la'].data * u.AA))

        a = np.where(self.sed['sed_flux'] > 0)[0]
        self.sed = self.sed[a]

        a = np.where((np.isnan(self.sed["sed_eflux"]) == True) | (self.sed["sed_eflux"] / self.sed["sed_flux"] < 0.02))[
            0]
        self.sed["sed_eflux"][a] = self.sed["sed_flux"][a] * 0.02

        self.sed['eflux'] = self.sed["sed_eflux"] / self.sed["sed_flux"] / np.log(10)
        self.sed['flux'] = np.log10(self.sed["sed_flux"].value * self.sed['la'])

        self.definefilter(**kwargs)
        self.sed = self.sed[['index', 'sed_filter', 'la', 'width', 'flux', 'eflux']]
        # print("Using vizier photometry search")
        # print(self.sed)
        return


def set_quality(self):
    with open(pkg_resources.resource_filename('SEDFit', 'quality.p'), 'rb') as file:
        model = pickle.load(file)
    n = len(self.sed)
    input = np.zeros((n, 42, 2)) - 1
    input[:, self.sed['index'].astype(int), 0] = np.tile(np.max(self.sed['flux']) - self.sed['flux'], (n, 1))
    input[:, :, 1] = 0
    input[range(len(self.sed)), self.sed['index'], 1] = 1
    q = np.round(model.predict(input, verbose=0), 3)
    self.sed['valid'] = q[:, 3]
    self.sed['excess'] = q[:, 2]
    self.sed['bad'] = q[:, 1]
    a = np.where(q[:, 3] > 0.2)[0]
    if len(a) / n < 0.3:
        print(
            'Warning: large number of fluxes rejected, due to IR excess, noise, or misattribution. Manual vetting suggested')
        return False
    return

#%% Beginning of my own functions
def match_filters(filt_name, verbose=False):
    ###########################################################
    # Function: match_filters                                 #
    # Inputs:                                                 #
    #    filt_name: name of the filter bandpass               #
    #    verbose: if True, returns the print statements       #
    # Outputs:                                                #
    #    match: filter name in the svo format                 #
    #    sedmatch: filter name in the SEDFit format           #
    # How it works:                                           #
    #    1. reads in filter name and forces it all to be in   #
    #       lower case                                        #
    #    2. defines the filter map that takes the read in     #
    #       filter name and maps it to the correct format     #
    #    3. matches the input filter to the formats           #
    #    4. Returns the formatted name                        #
    ###########################################################

    filt_name = filt_name.lower()

    # Define the filter mapping dictionary
    filter_mapping = {
        'gaia g': ('GAIA/GAIA3.G', 'GAIA.GAIA3.G'),
        'g': ('GAIA/GAIA3.G', 'GAIA.GAIA3.G'),
        'gaia gbp': ('GAIA/GAIA3.Gbp', 'GAIA.GAIA3.Gbp'),
        'gbp': ('GAIA/GAIA3.Gbp', 'GAIA.GAIA3.Gbp'),
        'gaia bp': ('GAIA/GAIA3.Gbp', 'GAIA.GAIA3.Gbp'),
        'gaia grp': ('GAIA/GAIA3.Grp', 'GAIA.GAIA3.Grp'),
        'grp': ('GAIA/GAIA3.Grp', 'GAIA.GAIA3.Grp'),
        'gaia rp': ('GAIA/GAIA3.Grp', 'GAIA.GAIA3.Grp'),
        '2mass k': ('2MASS/2MASS.Ks', '2MASS.Ks'),
        '2massk': ('2MASS/2MASS.Ks', '2MASS.Ks'),
        '2mass_k': ('2MASS/2MASS.Ks', '2MASS.Ks'),
        '2mass j': ('2MASS/2MASS.J', '2MASS.J'),
        '2massj': ('2MASS/2MASS.J', '2MASS.J'),
        '2mass_j': ('2MASS/2MASS.J', '2MASS.J'),
        '2mass h': ('2MASS/2MASS.H', '2MASS.H'),
        '2massh': ('2MASS/2MASS.H', '2MASS.H'),
        '2mass_h': ('2MASS/2MASS.H', '2MASS.H'),
        'tycho bt': ('TYCHO/TYCHO.B_MvB', 'TYCHO.TYCHO.B_MvB'),
        'bt': ('TYCHO/TYCHO.B_MvB', 'TYCHO.TYCHO.B_MvB'),
        'tycho vt': ('TYCHO/TYCHO.V_MvB', 'TYCHO.TYCHO.V_MvB'),
        'vt': ('TYCHO/TYCHO.V_MvB', 'TYCHO.TYCHO.V_MvB'),
        'hipparcos hp': ('Hipparcos/Hipparcos.Hp_MvB', 'Hipparcos.Hipparcos.Hp_MvB'),
        'hp': ('Hipparcos/Hipparcos.Hp_MvB', 'Hipparcos.Hipparcos.Hp_MvB'),
        'cousins u': ('Generic/Cousins.U', 'Cousins.U'),
        'uc': ('Generic/Cousins.U', 'Cousins.U'),
        'cousins v': ('Generic/Cousins.V', 'Cousins.V'),
        'cv': ('Generic/Cousins.V', 'Cousins.V'),
        'cousins b': ('Generic/Cousins.B', 'Cousins.B'),
        'bc': ('Generic/Cousins.B', 'Cousins.B'),
        'cousins r': ('Generic/Cousins.R', 'Cousins.R'),
        'rc': ('Generic/Cousins.R', 'Cousins.R'),
        'cousins i': ('Generic/Cousins.I', 'Cousins.I'),
        'ic': ('Generic/Cousins.I', 'Cousins.I'),
        'johnson u': ('Generic/Johnson_UBVRIJHKL.U', 'Johnson.U'),
        'u': ('Generic/Johnson_UBVRIJHKL.U', 'Johnson.U'),
        'johnson v': ('Generic/Johnson_UBVRIJHKL.V', 'Johnson.V'),
        'v': ('Generic/Johnson_UBVRIJHKL.V', 'Johnson.V'),
        'johnson b': ('Generic/Johnson_UBVRIJHKL.B', 'Johnson.B'),
        'b': ('Generic/Johnson_UBVRIJHKL.B', 'Johnson.B'),
        'johnson r': ('Generic/Johnson_UBVRIJHKL.R', 'Johnson.R'),
        'r': ('Generic/Johnson_UBVRIJHKL.R', 'Johnson.R'),
        'johnson i': ('Generic/Johnson_UBVRIJHKL.I', 'Johnson.I'),
        'i': ('Generic/Johnson_UBVRIJHKL.I', 'Johnson.I'),
        'johnson j': ('Generic/Johnson_UBVRIJHKL.J', 'Johnson.J'),
        'j': ('Generic/Johnson_UBVRIJHKL.J', 'Johnson.J'),
        'johnson k': ('Generic/Johnson_UBVRIJHKL.K', 'Johnson.K'),
        'k': ('Generic/Johnson_UBVRIJHKL.K', 'Johnson.K'),
        'johnson h': ('Generic/Johnson_UBVRIJHKL.H', 'Johnson.H'),
        'h': ('Generic/Johnson_UBVRIJHKL.H', 'Johnson.H'),
        'stromgren v': ('Generic/Stromgren.v', 'Stromgren.v'),
        'sv': ('Generic/Stromgren.v', 'Stromgren.v'),
        'stromgren b': ('Generic/Stromgren.b', 'Stromgren.b'),
        'sb': ('Generic/Stromgren.b', 'Stromgren.b'),
        'stromgren y': ('Generic/Stromgren.y', 'Stromgren.v'),
        'sy': ('Generic/Stromgren.y', 'Stromgren.v'),
        'tess': ('TESS/TESS.Red', 'TESS.TESS.Red'),
        'tess t': ('TESS/TESS.Red', 'TESS.TESS.Red'),
        't': ('TESS/TESS.Red', 'TESS.TESS.Red'),
        'galex fuv': ('GALEX/GALEX.FUV', 'Galaex.FUV'),
        'gfuv': ('GALEX/GALEX.FUV', 'Galaex.FUV'),
        'galex nuv': ('GALEX/GALEX.NUV', 'Galaex.NUV'),
        'gnuv': ('GALEX/GALEX.NUV', 'Galaex.NUV'),
        'ps1 g': ('PAN-STARRS/PS1.g', 'PAN-STARRS.PS1.g'),
        'panstarrs g': ('PAN-STARRS/PS1.g', 'PAN-STARRS.PS1.g'),
        'ps1 r': ('PAN-STARRS/PS1.r', 'PAN-STARRS.PS1.r'),
        'panstarrs r': ('PAN-STARRS/PS1.r', 'PAN-STARRS.PS1.r'),
        'ps1 i': ('PAN-STARRS/PS1.i', 'PAN-STARRS.PS1.i'),
        'panstarrs i': ('PAN-STARRS/PS1.i', 'PAN-STARRS.PS1.i'),
        'ps1 z': ('PAN-STARRS/PS1.z', 'PAN-STARRS.PS1.z'),
        'panstarrs z': ('PAN-STARRS/PS1.z', 'PAN-STARRS.PS1.z'),
        'ps1 y': ('PAN-STARRS/PS1.y', 'PAN-STARRS.PS1.y'),
        'panstarrs y': ('PAN-STARRS/PS1.y', 'PAN-STARRS.PS1.y'),
        'sdss u': ('SLOAN/SDSS.u', 'SDSS.u'),
        'sloan u': ('SLOAN/SDSS.u', 'SDSS.u'),
        'sdss g': ('SLOAN/SDSS.g', 'SDSS.g'),
        'sloan g': ('SLOAN/SDSS.g', 'SDSS.g'),
        'sdss r': ('SLOAN/SDSS.r', 'SDSS.r'),
        'sloan r': ('SLOAN/SDSS.r', 'SDSS.r'),
        'sdss i': ('SLOAN/SDSS.i', 'SDSS.i'),
        'sloan i': ('SLOAN/SDSS.i', 'SDSS.i'),
        'sdss z': ('SLOAN/SDSS.z', 'SDSS.z'),
        'sloan z': ('SLOAN/SDSS.z', 'SDSS.z'),
        'wise w1': ('WISE/WISE.W1', 'WISE.W1'),
        'w1': ('WISE/WISE.W1', 'WISE.W1'),
        'wise w2': ('WISE/WISE.W2', 'WISE.W2'),
        'w2': ('WISE/WISE.W2', 'WISE.W2'),
        'wise w3': ('WISE/WISE.W3', 'WISE.W3'),
        'w3': ('WISE/WISE.W3', 'WISE.W3'),
        'wise w4': ('WISE/WISE.W4', 'WISE.W4'),
        'w4': ('WISE/WISE.W4', 'WISE.W4'),
        'xmm v': ('XMM/OM.V_filter', 'XMM-OT.V'),
        'xmm b': ('XMM/OM.B_filter', 'XMM-OT.B'),
        'xmm u': ('XMM/OM.U_filter', 'XMM-OT.U'),
        'xmm w1': ('XMM/OM.UVW1_filter', 'XMM-OT.UVW1'),
        'xmm w2': ('XMM/OM.UVW2_filter', 'XMM-OT.UVW2'),
        'xmm m2': ('XMM/OM.UVM2_filter', 'XMM-OT.UVM2'),
        'spitzer irac 3.6': ('Spitzer/IRAC.I1', 'Spitzer.IRAC.3.6'),
        'irac 3.6': ('Spitzer/IRAC.I1', 'Spitzer.IRAC.3.6'),
        'spitzer irac 4.5': ('Spitzer/IRAC.I2', 'Spitzer.IRAC.4.5'),
        'irac 4.5': ('Spitzer/IRAC.I2', 'Spitzer.IRAC.4.5'),
        'spitzer irac 5.8': ('Spitzer/IRAC.I3', 'Spitzer.IRAC.5.8'),
        'irac 5.8': ('Spitzer/IRAC.I3', 'Spitzer.IRAC.5.8'),
        'spitzer irac 8.0': ('Spitzer/IRAC.I4', 'Spitzer.IRAC.8.0'),
        'irac 8.0': ('Spitzer/IRAC.I4', 'Spitzer.IRAC.8.0'),
        'spitzer mips 24': ('Spitzer/MIPS.24mu', 'Spitzer.MIPS.24'),
        'mips 24': ('Spitzer/MIPS.24mu', 'Spitzer.MIPS.24'),
    }

    # Find the corresponding match and sedmatch
    result = filter_mapping.get(filt_name)

    if result:
        match, sedmatch = result
        if verbose:
            print('Filter:', filt_name)
            print('Filter match:', match)
        return match, sedmatch

    # If filter name is not found, return None
    if verbose:
        print(f"No match found for filter: {filt_name}")
    return None, None

def set_filters(df, verbose = False):
    ##########################################################
    # Function: set_filters                                  #
    # Inputs:                                                #
    #    df: dataframe of photometry data                    #
    #    verbose: if True, returns an print statements       #
    # Outputs:                                               #
    #    df: same dataframe with new columns added           #
    # How it works:                                          #
    #    1. reads in the column 'Filter' to a variable       #
    #    2. in a for loop, calls the match_filters function  #
    #       which matches the input name to the correct      #
    #       format                                           #
    #    3. Creates new columns in the dataframd with the    #
    #       new filter names in the right formats            #
    #    4. Returns the dataframe                            #
    ##########################################################

    filter_name = df['Filter'].tolist()
    new_names = []
    sed_names = []
    for i in range(len(filter_name)):
        fn, sfn = match_filters(filter_name[i], verbose = verbose)
        new_names.append(fn)
        sed_names.append(sfn)
    df['Filter_name'] = new_names
    df['SED Filter name'] = sed_names
    return df


def get_zpts(df, verbose=False):
    ##########################################################
    # Function: get_zpts                                     #
    # Inputs:                                                #
    #    df: dataframe of photometry data                    #
    #    verbose: if True, returns an print statements       #
    # Outputs:                                               #
    #    df: same dataframe with new columns added           #
    # How it works:                                          #
    #    1. Reads in the svo filter data file                #
    #    2. Sets the columns needed to variables             #
    #    3. Runs through a for loop to compare the filter    #
    #       name to the svo filter data set                  #
    #    4. Pulls the zero point, reference wavelength,      #
    #       and effective width of the filter                #
    #    5. Adds the columns to the original dataframe       #
    #    6. Returns the dataframe                            #
    ##########################################################
    svo = pd.read_csv(svopath)
    svo_names = svo['Filter_name']
    svo_wref = svo['WavelengthRef']
    svo_weff = svo['WidthEff']
    svo_zpt = svo['ZPT (e-9)']

    filters = df['Filter_name']
    zpts = []
    zpt_err = []
    wref = []
    weff = []

    for i in range(len(filters)):
        name = filters[i]
        # print(name)
        for ii in range(len(svo_names)):
            if name == svo_names[ii]:
                if verbose:
                    print('Found a match.')
                    print('Input name:', name)
                    print('Matching:', svo_names[ii])
                zpts.append(svo_zpt[ii])
                zpt_err.append(0.05 * svo_zpt[ii])
                wref.append(svo_wref[ii])
                weff.append(svo_weff[ii])

    df['ZPT'] = zpts
    df['ZPT_err'] = zpt_err
    df['Ref wavelength'] = wref
    df['Effective width'] = weff

    return df

def read_in_photometry(filename, verbose = False):
    ##########################################################
    # Function: read_in_photometry                           #
    # Inputs:                                                #
    #    filename: filename of the photometry data           #
    #    verbose: if True, returns an print statements       #
    # Outputs:                                               #
    #    new_sed: astropy table                              #
    # How it works:                                          #
    #    1. Reads in the photometry file                     #
    #    2. Calls set_filters to re-format filter names      #
    #    3. Calls get_zpts to retrieve the svo filter info   #
    #    4. Separates out the needed data                    #
    #    5. Corrects the flux with the zero point            #
    #    6. Calculates the flux error                        #
    #    7. Creates a new dataframe with only needed data    #
    #    8. Converts new data frame to an astropy Table      #
    #    9. Converts the data into correct format for SEDFit #
    #   10. Adds units to data in Table.                     #
    #   11. Returns astropy Table in format for SEDFit       #
    ##########################################################
    phot = pd.read_csv(filename)
    phot1 = set_filters(phot)
    phot2 = get_zpts(phot1)
    mag = phot2['Magnitude']
    dmag = phot2['Error']
    zptf = phot2['ZPT']*(1e-9)
    dzptf = phot2['ZPT_err']*(1e-9)
    wave = phot2['Ref wavelength']
    pbwidth = phot['Effective width']/2

    flux = (zptf)*(10**(-0.4*(mag)))
    #Calculating the error for the new fluxes
    term1 = (10**(-0.4*mag))*(dzptf)
    term2 = -0.4*np.log(10)*(zptf)*(10**(-0.4*mag))*dmag
    flux_err = np.sqrt(term1**2 + term2**2)
    name = phot2['Filter']
    idx = np.arange(0,len(flux), 1)

    new_phot = pd.DataFrame({'index': idx,'sed_filter':phot['SED Filter name'], 'la':wave, 'width':pbwidth, 'flux':flux, 'eflux':flux_err})
    new_sed = Table.from_pandas(new_phot)
    new_sed['la'].unit = u.AA
    new_sed['width'].unit = u.AA
    new_sed['flux'] = np.log10(flux*wave)
    new_sed['eflux'] = 0.434*(flux_err/flux)
    new_sed['flux'].unit = u.erg/u.s/(u.cm**2)/u.AA
    new_sed['eflux'].unit = u.erg/u.s/(u.cm**2)/u.AA

    if verbose:
        new_sed

    return new_sed


def chi2red(x, numfp, verbose=False):
    ##########################################################
    # Function: chi2red                                      #
    # Inputs:                                                #
    #    x: sed object from SEDFit                           #
    #    numfp: number of fit parameters                     #
    #    verbose: if True, returns an print statements       #
    # Outputs:                                               #
    #    chi: chi squared value                              #
    #    chi2r: chi squared reduced value                    #
    # How it works:                                          #
    #    1. Sets the index range                             #
    #    2. Sets the "modeled" component to the synthetic    #
    #       fluxes from the model                            #
    #    3. Sets the "real" component to the input fluxes    #
    #    4. Sets the "error" component to the input flux err #
    #    5. Calculates chi squared                           #
    #    6. Calcualtes chi squared reduced                   #
    #    7. Returns chi squared and chi squared reduced.     #
    ##########################################################
    idx = np.arange(0, len(x.sed['index']))
    # print(idx)
    m = x.mags[idx]
    # print(m)
    r = x.sed['flux'][idx]
    # print(r)
    e = x.sed['eflux'][idx]
    # print(e)
    l = len(idx)
    # print(l)

    chi = np.sum(((m - r) / e) ** 2)
    chi2r = np.sum(((m - r) / e) ** 2) / (l - numfp)
    if verbose:
        print("Chi squared:", chi)
        print("Chi squared reduced:", chi2r)
    return (chi, chi2r)


def fit_sed(sed, star, initial_guess, model, teffrange=None, loggrange=None, fehrange=None, fitT=False, fit_logg=False,
            fit_feh=False, verbose=False):
    ##########################################################
    # Function: fit_sed                                      #
    # Inputs:                                                #
    #    sed: sed object                                     #
    #    star: StellarParams() object                        #
    #    initial_guess: intitial guess for fit parameters    #
    #                   [ teff, logg, feh]                   #
    #    model: model keyword                                #
    #           'phoenix' -> Phoenix model                   #
    #           'btsettl' -> BT-SETTL model                  #
    #           'kurucz' -> Kurucz model                     #
    #           'coehlo' -> Coehlo model                     #
    #    teffrange: if fitting for teff, sets the range of   #
    #               teff values the model can search through #
    #    loggrange: if fitting for logg, sets the range of   #
    #               logg values the model can search through #
    #    fehrange: if fitting for feh, sets the range of feh #
    #              values the model can search through       #
    #    fitT: default is False, if True, fits for teff      #
    #    fit_logg: default is False, if True, fits for logg  #
    #    fit_feh: default is False, if True, fits for feh    #
    #    verbose: if True, returns an print statements       #
    # Outputs:                                               #
    #    x: sed object with fitted parameters in it          #
    #    if verbose is true, returns the print statements    #
    #    displaying fit parameter solutions                  #
    # How it works:                                          #
    #    1. Reads in distance, RA, and dec from the star     #
    #    2. Initializes SEDfit object                        #
    #    3. Downloads user input photometry file             #
    #    4. Calls set_quality function to fix the SED object #
    #    5. Sets initial guess                               #
    #    6. Adds the intial guesses                          #
    #    7. Sets ranges for the fit variables if any         #
    #    8. Calls SEDFit.fit to fit the SED                  #
    #    9. Determines number of fit params based on what is #
    #       being fitted                                     #
    #   10. Calculates chi squared and chi squared reduced   #
    #   11. Sets the params to the stellar object            #
    #   12. Returns the SED object                           #
    ##########################################################
    d = star.dist
    ra = star.ra
    dec = star.dec

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        x = SEDFit(ra, dec, 1, grid_type=model)

    x.dist = d
    downloadflux(x, sed)
    set_quality(x)

    teff, logg, feh = initial_guess

    x.addguesses(teff=teff, logg=logg, feh=feh)
    if teffrange is not None:
        x.addrange(teff=teffrange)
    if loggrange is not None:
        x.addrange(logg=loggrange)
    if fehrange is not None:
        x.addrange(feh=fehrange)

    x.fit(use_gaia=False, idx=np.arange(0, len(x.sed['index'])), fitdist=False,
          fitteff=fitT, fitfeh=fit_feh, fitlogg=fit_logg, quality_check=False)
    numOparams = 2
    if fitT == True:
        numOparams += 1
        # print("fitT is set to True.")
        # print(numOparams)
        star.sed_teff = x.getteff()[0]
    if fit_logg == True:
        numOparams += 1
        # print("fitlogg is set to True.")
        # print(numOparams)
        star.sed_logg = x.getlogg()[0]
    if fit_feh == True:
        numOparams += 1
        # print("fitfeh is set to True.")
        # print(numOparams)
        star.sed_feh = x.getfeh()

    chi2, chi2r = chi2red(x, numOparams, verbose=verbose)
    if verbose:
        print("Distance: {} pc".format(x.getdist()))
        print("AV: {} mag".format(x.getav()))
        print("Radius: {} Rsun".format(x.getr()))
        print("Teff: {} K".format(x.getteff()))
        print("Log g: {} ".format(x.getlogg()))
        print("Fe/H: {}".format(x.getfeh()))

    star.sed_av = x.getav()
    star.sed_rad = x.getr()[0]
    return x


def convert(sed, unit):
    ##########################################################
    # Function: convert                                      #
    # Inputs:                                                #
    #    sed: sed object                                     #
    #    unit: string with what unit you want                #
    #          options are: 'micron' or 'AA'                 #
    # Outputs:                                               #
    #    lit_w: input wavelength                             #
    #    lit_f: input flux                                   #
    #    lit_dw: input wavelength error                      #
    #    lit_df: input flux error                            #
    #    model_w: model wavelength                           #
    #    model_f: model flux                                 #
    #    synth_f: model fluxes in the wavelength badpasses   #
    # How it works:                                          #
    #    1. Reads in the model wavelength and converts to    #
    #       microns if needed. Default is angstroms          #
    #    2. Reads in the model flux and converts it into     #
    #       flux from log10(flux/wavelength) and converts it #
    #       into micron units if needed. Default is angstrom #
    #    3. Reads in wavelength error and converts if needed #
    #       Default is angstroms                             #
    #    4. Reads in flux error and converts it from log     #
    #       and converts into microns if needed.             #
    #    5. Reads in model wavelengths and converts it to    #
    #       microns if needed.                               #
    #    6. Reads in model flux and converts it out of log   #
    #       and converts it to microns if needed             #
    #    7. Reads in synthetic fluxes and converts it out of #
    #       log and converts it into microns if needed       #
    #    8. Returns values                                   #
    ##########################################################
    if unit == 'AA':
        lit_w = sed.sed['la']  # literature wavelengths
        lit_f = (10 ** sed.sed['flux']) / np.array(lit_w)  # literature flux
        lit_dw = sed.sed['width']  # literature wavelength error
        lit_df = (sed.sed['eflux'] / 0.434) * lit_f  # literature flux error

        model_w = np.array(sed.la)  # model wavelength
        model_f = (10 ** (sed.fx.flatten())) / model_w  # model flux

        synth_f = (10 ** sed.mags) / np.array(lit_w)  # model fluxes for the wavelength

        # residuals = lit_f-synth_f

        return lit_w, lit_f, lit_dw, lit_df, model_w, model_f, synth_f

    if unit == 'micron':
        lit_w = sed.sed['la'] * (1e-4)  # literature wavelengths
        lit_f = ((10 ** sed.sed['flux']) / np.array(lit_w)) * (1e-4)  # literature flux
        lit_dw = sed.sed['width'] * (1e-4)  # literature wavelength error
        lit_df = ((sed.sed['eflux'] / 0.434) * lit_f) * (1e-4)  # literature flux error

        model_w = np.array(sed.la) * (1e-4)  # model wavelength
        model_f = ((10 ** (sed.fx.flatten())) / model_w) * (1e-4)  # model flux

        synth_f = ((10 ** sed.mags) / np.array(lit_w)) * (1e-4)  # model fluxes for the wavelength

        # residuals = np.array(lit_f)-np.array(synth_f)

        return np.array(lit_w), np.array(lit_f), np.array(lit_dw), np.array(lit_df), np.array(model_w), np.array(
            model_f), np.array(synth_f)

def calc_fbol(star,x, unit, verbose = False):
    ##########################################################
    # Function: calc_fbol                                    #
    # Inputs:                                                #
    #    star: star object                                   #
    #    x: sed object                                       #
    #    unit: unit string                                   #
    # Outputs:                                               #
    #    result: bolometric flux                             #
    #    error: error on bolometric flux                     #
    # How it works:                                          #
    #    1. Calls convert to generate the model values       #
    #    2. Defines a "model" for integration purposes       #
    #    3. Integrates the model                             #
    #    4. Sets bolometric flux value and error to star     #
    #    5. Returns values.                                  #
    ##########################################################
    _, _, _, _, mw, mf, _ = convert(x, unit=unit)
    def new_model(x):
        new_m = np.interp(x, model_w, model_f)
        return new_m

    result, error = quad(new_model, min(model_w), 1000000)
    if verbose:
        print('Fbol = ', round(result/(1e-8), 5), '+/-', round(error/(1e-8),5), 'x10^(-8) erg/s/cm^s/angstrom')

    star.fbol = round(result/(1e-8), 5)
    star.fbol_err = round(error/(1e-8), 5)
    return result, error


def set_values(x, unit, logplot=False, fbol_lam=False, verbose=False):
    ##########################################################
    # Function: set_values                                   #
    # Inputs:                                                #
    #    x: sed object                                       #
    #    unit: unit string                                   #
    #    logPlot: default is False, if True, indicates the   #
    #             log flag                                   #
    #    fbol_lam: default is False, if True, indicates the  #
    #             fbol_lam is flag                           #
    # Outputs:                                               #
    #    litp_xvals: input wavelength                        #
    #    litp_yvals: input flux                              #
    #    litp_dxvals: input wavelength error                 #
    #    litp_dyvals: input flux error                       #
    #    model_xvals: model wavelength                       #
    #    model_yvals: model flux                             #
    #    synth_yvals: model fluxes in the wavelength         #
    #                 bandpasses                             #
    #   residuals: lit flux values minus synth values        #
    # How it works:                                          #
    #    1. Calls convert to generate the model values       #
    #    2. Based on the flags set, converts the values to   #
    #       match what the flags say                         #
    #       logplot: convert everything back into log10      #
    #       fbol_lam: multiply the flux by wavelength        #
    #    3. Returns values                                   #
    ##########################################################
    iwave, iflux, idwave, idflux, mw, mf, msf = convert(x, unit=unit)

    if logplot and fbol_lam:
        # print('set values: Log and lambda')
        model_xvals = np.log10(mw)
        model_yvals = np.log10(mf * mw)
        litp_xvals = np.log10(iwave)
        litp_yvals = np.log10(iwave * iflux)
        litp_dyvals = 0.434 * (idflux / iflux)
        litp_dxvals = 0.434 * (idwave / iwave)
        synth_yvals = np.log10(msf * iwave)
        res = litp_yvals - synth_yvals

        return model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res

    if logplot and not fbol_lam:
        # print('set values Log and no lambda')
        model_xvals = np.log10(mw)
        model_yvals = np.log10(mf)
        litp_xvals = np.log10(iwave)
        litp_yvals = np.log10(iflux)
        litp_dyvals = 0.434 * (idflux / iflux)
        litp_dxvals = 0.434 * (idwave / iwave)
        synth_yvals = np.log10(msf)
        res = litp_yvals - synth_yvals
        # print('Log plot and no lambda')
        # print(res)
        return model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res

    if not logplot and not fbol_lam:
        # print('set values No log and no lambda')
        number = iflux[0]
        _, exp = normalize_number(number)
        # print('Dividing by:', exp)
        model_xvals = mw
        model_yvals = mf / (10 ** exp)
        litp_xvals = iwave
        litp_yvals = iflux / (10 ** exp)
        litp_dyvals = idflux / (10 ** exp)
        litp_dxvals = idwave
        synth_yvals = msf / (10 ** exp)
        res = litp_yvals - synth_yvals
        # print('No log and no lambda')
        # print(res)
        return model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res

    if not logplot and fbol_lam:
        # print('set values No log but lambda')
        number = (iflux[0] * iwave[0])
        _, exp = normalize_number(number)
        # print('Dividing by:', exp)
        model_xvals = mw
        model_yvals = (mf * mw) / (10 ** exp)
        litp_xvals = iwave
        litp_yvals = (iflux * iwave) / (10 ** exp)
        litp_dyvals = (idflux) / (10 ** exp)
        litp_dxvals = idwave
        synth_yvals = (msf * iwave) / (10 ** exp)
        res = litp_yvals - synth_yvals
        return model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res


def normalize_number(num):
    ##########################################################
    # Function: normalize_number                             #
    # Inputs:                                                #
    #    num: number you want normalized                     #
    # Outputs:                                               #
    #    normalized_num: normalized number                   #
    #    exponent: exponent that value was normalized by     #
    # How it works:                                          #
    #    1. Determines the power of 10                       #
    #    2. Determines normalization factor                  #
    #    3. Normalizes the number                            #
    #    4. Returns the normalized number and exoponent      #
    ##########################################################
    # Determine the power of 10 (exponent, X)
    if num != 0:
        exponent = math.floor(math.log10(abs(num)))
    else:
        exponent = 0  # Special case when num is 0 to avoid math domain error

    # Calculate the normalization factor (1eX)
    factor = 10 ** exponent

    # Divide the number by 10^X
    normalized_num = num / factor

    return normalized_num, exponent

def setaxislabels(unit, fbol_lam=False):
    ##########################################################
    # Function: setaxislabels                                #
    # Inputs:                                                #
    #    unit: string of unit wanted                         #
    #    fbol_lam: flag for fbol_lam                         #
    # Outputs:                                               #
    #    xlab: x axis label                                  #
    #    ylab: y axis label                                  #
    # How it works:                                          #
    #    1. Based on unit chosen, sets x label dependent on  #
    #       unit                                             #
    #    2. Based on unit chosen and if the fbol_lam flag    #
    #       has been set, sets y label                       #
    #    3. Returns x axis and y axis labels                 #
    ##########################################################
    if unit == 'AA':
        xlab = r'$\rm Wavelength~[\AA]$'
        if fbol_lam:
            ylab = r'$\rm \lambda F_{\lambda}~[\frac{erg}{cm^{2}~s}$]'
        else:
            ylab = r'$\rm F_{\lambda}~[\frac{erg}{cm^{2}~s~\AA}$]'
        return xlab, ylab
    if unit == 'micron':
        xlab = r'$\rm Wavelength~[\mu m]$'
        if fbol_lam:
            ylab = r'$\rm \lambda F_{\lambda}~[\frac{erg}{cm^{2}~s}$]'
        else:
            ylab = r'$\rm F_{\lambda}~[\frac{erg}{cm^{2}~s~\mu m}$]'
        return xlab, ylabs

def set_axis_limits(mw, mf, unit, fbol_lam = False):
    ##########################################################
    # Function: set_axis_labels                              #
    # Inputs:                                                #
    #    mw: model wavelength array                          #
    #    mf: model flux array                                #
    #    unit: string of unit wanted                         #
    #    fbol_lam: flag for fbol_lam                         #
    # Outputs:                                               #
    #    set_axis: the axis limits in the format of          #
    #              [xmin, xmax, ymin, ymax]                  #
    # How it works:                                          #
    #    1. Based on unit chosen and fbol_lam flag setting,  #
    #       sets the axis limits based on the minimum of the #
    #       arrays  and maxs of the arrays                   #
    #    2. Returns the sxis limits                          #
    ##########################################################
    if unit == 'AA':
        #print('Angstrom')
        if fbol_lam:
            #print('F x Lambda')
            set_axis = [round(min(mw)), round(max(mw)), min(mf*mw), max(mf*mw)]
            return set_axis
        else:
            #print('F')
            set_axis = [round(min(mw)), round(max(mw)), min(mf), max(mf)]
            return set_axis
    if unit == 'micron':
        #print('micron')
        if fbol_lam:
            #print('F x Lambda')
            set_axis = [min(mw), round(max(mw)), min((mf*mw)), max((mf*mw))]
            return set_axis
        else:
            #print('F')
            set_axis = [min(mw), round(max(mw)), min(mf), max(mf)]
            return set_axis


def plot_sed(x, unit, logplot=True, fbol_lam=True, set_axis=None, title=None, savefig=None, show=True, verbose=False):
    ##########################################################
    # Function: plot_sed                                     #
    # Inputs:                                                #
    #    x: sed object                                       #
    #    unit: unit chosen                                   #
    #    logplot: log flag                                   #
    #             if True, sets plot in log space            #
    #    fbol_lam: fbol_lam flag                             #
    #             if True, multiplies the flux by wavelength #
    #    set_axis: allows user to set their axis limits      #
    #             if None, will set based on the model vals  #
    #    title: allows user to set the plot title            #
    #    savefig: allows user to save fig                    #
    #            give a filename                             #
    #    show: shows Figure                                  #
    # Outputs:                                               #
    #    displays the plot                                   #
    # How it works:                                          #
    #    1. Determines the axis limits based on user input   #
    #       and flags set                                    #
    #    2. Calls set_values to generate the data to be      #
    #       plotted                                          #
    #    3. Plots everything                                 #
    ##########################################################

    iwave, iflux, idwave, idflux, mw, mf, msf = convert(x, unit)

    plt.rcParams.update({'font.size': 15})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['text.usetex'] = False

    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 3]}, sharex=True)

    if set_axis:
        xmin = set_axis[0]
        xmax = set_axis[1]
        ymin = set_axis[2]
        ymax = set_axis[3]

        # print('Log plot')
        if logplot:
            # print(xmin, xmax, ymin, ymax)
            axes[0].set_ylim(ymin, ymax)
            axes[1].set_xlim(xmin, xmax)

            y_loc = [round(ymin), round(ymax)]
            y_labels = [rf'$10^{{{y_loc[0]}}}$', rf'$10^{{{y_loc[1]}}}$']
            axes[0].set_yticks(y_loc)
            axes[0].set_yticklabels(y_labels)
            xaxis = np.linspace(xmin, xmax, 5)
            # print(xaxis)
            x_loc = []
            for i in range(len(xaxis)):
                xloc = (10 ** (xaxis[i]))
                x_loc.append(round(xloc, 1))
            x_labels = [f'{round(val)}' for val in x_loc]
            # print(x_loc)
            axes[1].set_xticks(xaxis)
            axes[1].set_xticklabels(x_labels)
        else:
            # print('No log plot')
            if fbol_lam:
                number = iflux[0] * iwave[0]
                _, exp = normalize_number(number)
                # print('Axis Dividing by:', exp)
            else:
                number = iflux[0]
                _, exp = normalize_number(number)
                # print('Axis Dividing by:', exp)
            if unit == 'micron':
                xmax = 10
            if unit == 'AA':
                xmax = 20000
            # print(xmin, xmax, ymin/10**exp, ymax/10**exp)
            axes[0].set_ylim(0, ymax / 10 ** exp)
            axes[1].set_xlim(xmin, xmax)
            x_loc = np.linspace(round(xmin), round(xmax), 5)
            x_labels = [f'{round(val)}' for val in x_loc]
            axes[1].set_xticks(x_loc)
            axes[1].set_xticklabels(x_labels)
    else:
        set_axis = set_axis_limits(mw, mf, unit, fbol_lam=fbol_lam)
        # print('Min of model')
        # print(min(mw))
        xmin = set_axis[0]
        xmax = set_axis[1]
        ymin = set_axis[2]
        ymax = set_axis[3]
        # print('axis beore log')
        # print(xmin, xmax, ymin, ymax)
        if logplot:
            # print('Log plot')

            xmin = round(np.log10(xmin), 1)
            xmax = round(np.log10(xmax), 1)
            ymin = round(np.log10(ymin), 1)
            ymax = round(np.log10(ymax), 1)
            # print(xmin, xmax, ymin, ymax)
            axes[0].set_ylim(ymin, ymax)
            axes[1].set_xlim(xmin, xmax)

            y_loc = [round(ymin), round(ymax)]
            y_labels = [rf'$10^{{{y_loc[0]}}}$', rf'$10^{{{y_loc[1]}}}$']
            axes[0].set_yticks(y_loc)
            axes[0].set_yticklabels(y_labels)
            xaxis = np.linspace(xmin, xmax, 5)
            # print(xaxis)
            x_loc = []
            for i in range(len(xaxis)):
                xloc = (10 ** (xaxis[i]))
                x_loc.append(round(xloc, 1))
            x_labels = [f'{round(val)}' for val in x_loc]
            # print(x_loc)
            axes[1].set_xticks(xaxis)
            axes[1].set_xticklabels(x_labels)
        else:
            # print('No log plot')
            if fbol_lam:
                number = iflux[0] * iwave[0]
                _, exp = normalize_number(number)
                # print('Axis Dividing by:', exp)
            else:
                number = iflux[0]
                _, exp = normalize_number(number)
                # print('Axis Dividing by:', exp)
            if unit == 'micron':
                xmax = 10
            if unit == 'AA':
                xmax = 20000
            # print(xmin, xmax, ymin/10**exp, ymax/10**exp)
            axes[0].set_ylim(0, ymax / 10 ** exp)
            axes[1].set_xlim(xmin, xmax)
            x_loc = np.linspace(round(xmin), round(xmax), 5)
            x_labels = [f'{round(val)}' for val in x_loc]
            axes[1].set_xticks(x_loc)
            axes[1].set_xticklabels(x_labels)

    model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res = set_values(x, unit,
                                                                                                              logplot=logplot,
                                                                                                              fbol_lam=fbol_lam,
                                                                                                              verbose=verbose)

    axes[0].plot(model_xvals, model_yvals, 'g', linewidth=1, label=r'$\rm Model~Spectrum$')
    axes[0].plot(litp_xvals, litp_yvals, 'b.', markersize=10, markerfacecolor='none', label=r'$\rm Photometry$')
    axes[0].errorbar(litp_xvals, litp_yvals, xerr=litp_dxvals, yerr=litp_dyvals, fmt='.', markerfacecolor='none',
                     color='blue', capsize=3)
    axes[0].plot(litp_xvals, synth_yvals, 'r.', markersize=10, label=r'$\rm Synthetic~Photometry $')

    axes[0].legend(prop={'size': 10}, loc='best')
    axes[0].tick_params(axis='x', labelbottom=False)

    axes[1].plot(litp_xvals, res, 'k.')
    axes[1].errorbar(litp_xvals, res, yerr=litp_dyvals, fmt='.', color='black')
    axes[1].axhline(y=0)
    axes[1].set_ylabel(r'$\rm Residuals$')
    xlab, ylab = setaxislabels(unit, fbol_lam=fbol_lam)
    axes[1].set_xlabel(xlab)
    axes[0].set_ylabel(ylab)
    plt.subplots_adjust(wspace=0, hspace=0)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    if title:
        axes[0].set_title(title)
    if savefig:
        f.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()

    return f, axes