import io
import os
import ast
import math
import pickle
import pyphot
import contextlib
import numpy as np
import pandas as pd
import dustmaps.sfd
import pkg_resources
from SEDFit.sed import SEDFit
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.table import Table
from radpy.config import svopath
from astroARIADNE.star import Star
from numpy.random import default_rng
from astroquery.simbad import Simbad
from astroARIADNE.fitter import Fitter
from astropy import coordinates as coord
from radpy.stellar import check_if_string
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

pyphot.config.set_units_backend('pint')
#dustmaps.sfd.fetch()

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  #suppress everything but error messages
import tensorflow as tf

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
            width = np.array([0.152026, 0.241018, 0.250619]) / 2 * u.micron
            la = np.array([12393.09, 16494.95, 21638.61]) * u.AA
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
            width = np.array([0.0657, 0.10117, 0.08898, 0.207, 0.2316, 0.319355, 0.5785, 0.286263]) / 2 * u.micron
            la = np.array([3511.89, 4382.77, 5501.4, 6819.05, 8657.44, 12317.3, 21735.85, 16396.38]) * u.AA
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
            filters = ['WISE.W1', 'WISE.W2', 'WISE.W3', 'WISE.W4']
            width = np.array([0.662642, 1.042266, 5.505523, 4.10168]) / 2 * u.micron
            la = np.array([33682.21, 46179.06, 120718.12, 221944.04]) * u.AA
            ind = np.array([45, 48, 51, 53])
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
    #target = str(self.ra) + '%20' + str(self.dec)
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

    # Keep dtype stable for TensorFlow
    inp = np.full((n, 42, 2), -1.0, dtype=np.float32)

    # Build index arrays once, with stable dtypes
    sed_idx = np.asarray(self.sed['index'], dtype=np.int32)
    flux = np.asarray(self.sed['flux'], dtype=np.float32)

    inp[:, sed_idx, 0] = np.tile(np.max(flux) - flux, (n, 1))
    inp[:, :, 1] = 0.0
    inp[np.arange(n, dtype=np.int32), sed_idx, 1] = 1.0

    # Call model directly (often less retracing-prone than predict in loops)
    pred = model(tf.convert_to_tensor(inp, dtype=tf.float32), training=False)
    q = np.round(pred.numpy(), 3)

    self.sed['valid'] = q[:, 3]
    self.sed['excess'] = q[:, 2]
    self.sed['bad'] = q[:, 1]
    a = np.where(q[:, 3] > 0.2)[0]
    if len(a) / n < 0.3:
        return False
    return

#%% Beginning of my own functions
def pull_coords(star_id, star, verbose=False):
    star_name = check_if_string(star_id, verbose=verbose)
    simbad_result = Simbad.query_object(star_name)
    radeg = simbad_result['ra'][0]
    decdeg = simbad_result['dec'][0]

    if verbose:
        print(f"\nCoordinates in decimal degrees for {star_name}:")
        print(f"RA: {radeg}, Dec: {decdeg}")

    coords = coord.SkyCoord(ra=radeg * u.deg, dec=decdeg * u.deg, frame='icrs')
    # Convert to sexagesimal string representation
    ra_sg = coords.ra.to_string(unit=u.hourangle, sep=':', precision=5)
    dec_sg = coords.dec.to_string(unit=u.deg, sep=':', precision=5)
    star.ra_deg = radeg
    star.dec_deg = decdeg
    star.ra_hms = ra_sg
    star.dec_dms = dec_sg
    if verbose:
        print(f"\nCoordinates in sexagesimal format for {star_name}:")
        print(f"RA: {ra_sg}, Dec: {dec_sg}")
    return radeg, decdeg, ra_sg, dec_sg

def pull_gaia_id(starid, star, verbose = False):
    star_name = check_if_string(starid, verbose = verbose)
    try:
            simbad_result = Simbad.query_objectids(star_name)
            if simbad_result is None:
                if verbose:
                    print(f"No results found in Simbad for {star_name}")
                    #d, dd = use_hipparcos(star_name, plx, dplx, verbose)
                #return round(d, 5), round(d, dd)

            # Find the column name for 'id' (case-insensitive)
            id_col = next((col for col in simbad_result.colnames if col.lower() == "id"), None)

            if id_col is not None:
                ids = simbad_result[id_col]
            else:
                raise KeyError("No column named 'id' or 'ID' found in simbad_result")

            gaiadr3mask = ['Gaia DR3' in name for name in ids]
            if any(gaiadr3mask):
                gaiadr3_name = ids[gaiadr3mask][0]
                if verbose:
                    print("Found Gaia DR3 ID:", gaiadr3_name)
                gdr3_id = gaiadr3_name.split()[-1]
                star.GaiaDR3ID = int(gdr3_id)
                return int(gdr3_id)

                # Only check DR2 if no DR3 found
            gaiadr2mask = ['Gaia DR2' in name for name in ids]
            if any(gaiadr2mask):
                gaiadr2_name = ids[gaiadr2mask][0]
                if verbose:
                    print("Found Gaia DR2 ID:", gaiadr2_name)
                gdr2_id = gaiadr2_name.split()[-1]
                star.GaiaDR2ID = int(gdr2_id)
                return int(gdr2_id)

            # Only check DR1 if no DR2 found
            gaiadr1mask = ['Gaia DR1' in name for name in ids]
            if any(gaiadr1mask):
                gaiadr1_name = ids[gaiadr1mask][0]
                if verbose:
                    print("Found Gaia DR1 ID:", gaiadr1_name)
                gdr1_id = gaiadr1_name.split()[-1]
                star.GaiaDR1ID = int(gdr1_id)
                return int(gdr1_id)
    except Exception as e:
        if verbose:
            print(f"Error in Simbad query: {e}")


def extract_photometry(starid, star, verbose=False):
    ra = star.ra_deg
    dec = star.dec_deg
    if hasattr(star, 'GaiaDR3ID'):
        if verbose:
            print('Star has Gaia DR3 ID.')
        gaia_id = star.GaiaDR3ID
    elif hasattr(star, 'GaiaDR2ID'):
        if verbose:
            print('Star has Gaia DR2 ID.')
        gaia_id = star.GaiaDR2ID
    elif hasattr(star, 'GaiaDR1ID'):
        if verbose:
            print('Star has Gaia DR1 ID.')
        gaia_id = star.GaiaDR1ID
    else:
        if verbose:
            print('Star does not have a Gaia ID')
        gaia_id = None
    if not verbose:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            s = Star(starid, ra, dec, g_id=gaia_id, ignore=['SkyMapper', 'GALEX', 'APASS', 'Pan-STARRS', 'SDSS', 'ASCC', 'GLIMPSE', 'TESS', 'STROMGREN_PAUNZ', 'STROMGREN_HAUCK'], verbose=False)

    if verbose:
        s = Star(starid, ra, dec, g_id=gaia_id, ignore=['SkyMapper', 'GALEX', 'APASS', 'Pan-STARRS', 'SDSS', 'ASCC', 'GLIMPSE', 'TESS', 'STROMGREN_PAUNZ', 'STROMGREN_HAUCK'], verbose=False)

    return s

def inspect_photometry(phot_obj):
    phot_obj.print_mags()

def add_photometry(phot_obj, filter_name, mag, mag_err, verbose = False):
    filter_name = filter_name.lower()
    filt_map = {
        'gaia g': ('GaiaDR2v2_G'),
        'g': ('GaiaDR2v2_G'),
        'gaia gbp': ('GaiaDR2v2_BP'),
        'gbp': ('GaiaDR2v2_BP'),
        'gaia bp': ('GaiaDR2v2_BP'),
        'gaia grp': ('GaiaDR2v2_RP'),
        'grp': ('GaiaDR2v2_RP'),
        'gaia rp': ('GaiaDR2v2_RP'),
        '2mass k': ('2MASS_Ks'),
        '2massk': ('2MASS_Ks'),
        '2mass_k': ('2MASS_Ks'),
        '2mass j': ('2MASS_J'),
        '2massj': ('2MASS_J'),
        '2mass_j': ('2MASS_J'),
        '2mass h': ('2MASS_H'),
        '2massh': ('2MASS_H'),
        '2mass_h': ('2MASS_H'),
        'tycho bt': ('TYCHO_B_MvB'),
        'bt': ('TYCHO_B_MvB'),
        'tycho vt': ('TYCHO_V_MvB'),
        'vt': ('TYCHO_V_MvB'),
        'cousins r': ('GROUND_COUSINS_R'),
        'rc': ('GROUND_COUSINS_R'),
        'cousins i': ('GROUND_COUSINS_I'),
        'ic': ('GROUND_COUSINS_I'),
        'johnson u': ('GROUND_JOHNSON_U'),
        'u': ('GROUND_JOHNSON_U'),
        'johnson v': ('GROUND_JOHNSON_V'),
        'v': ('GROUND_JOHNSON_V'),
        'johnson b': ('GROUND_JOHNSON_B'),
        'b': ('GROUND_JOHNSON_B'),
        'stromgren u': ('STROMGREN_u'),
        'su': ('STROMGREN_u'),
        'stromgren v': ('STROMGREN_v'),
        'sv': ('STROMGREN_v'),
        'stromgren b': ('STROMGREN_b'),
        'sb': ('STROMGREN_b'),
        'stromgren y': ('STROMGREN_y'),
        'sy': ('STROMGREN_y'),
        'tess': ('TESS'),
        'tess t': ('TESS'),
        't': ('TESS'),
        'galex fuv': ('GALEX_FUV'),
        'gfuv': ('GALEX_FUV'),
        'galex nuv': ('GALEX_NUV'),
        'gnuv': ('GALEX_NUV'),
        'ps1 g': ('PS1_g'),
        'panstarrs g': ('PS1_g'),
        'ps1 r': ('PS1_r'),
        'panstarrs r': ('PS1_r'),
        'ps1 i': ('PS1_i'),
        'panstarrs i': ('PS1_i'),
        'ps1 z': ('PS1_z'),
        'panstarrs z': ('PS1_z'),
        'ps1 y': ('PS1_y'),
        'panstarrs y': ('PS1_y'),
        'sdss u': ('SDSS_u'),
        'sloan u': ('SDSS_u'),
        'sdss g': ('SDSS_g'),
        'sloan g': ('SDSS_g'),
        'sdss r': ('SDSS_r'),
        'sloan r': ('SDSS_r'),
        'sdss i': ('SDSS_i'),
        'sloan i': ('SDSS_i'),
        'sdss z': ('SDSS_z'),
        'sloan z': ('SDSS_z'),
        'wise w1': ('WISE_RSR_W1'),
        'w1': ('WISE_RSR_W1'),
        'wise w2': ('WISE_RSR_W2'),
        'w2': ('WISE_RSR_W2'),
        'wise w3': ('WISE_RSR_W3'),
        'w3': ('WISE_RSR_W3'),
        'wise w4': ('WISE_RSR_W4'),
        'w4': ('WISE_RSR_W4')
    }

    filt_name = filt_map.get(filter_name)
    if verbose:
        phot_obj.add_mag(mag, mag_err, filt_name)

    if not verbose:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            phot_obj.add_mag(mag, mag_err, filt_name)

def remove_photometry(phot_obj, filter_name, verbose = False):
    filter_name = filter_name.lower()
    filt_map = {
        'gaia g': ('GaiaDR2v2_G'),
        'g': ('GaiaDR2v2_G'),
        'gaia gbp': ('GaiaDR2v2_BP'),
        'gbp': ('GaiaDR2v2_BP'),
        'gaia bp': ('GaiaDR2v2_BP'),
        'gaia grp': ('GaiaDR2v2_RP'),
        'grp': ('GaiaDR2v2_RP'),
        'gaia rp': ('GaiaDR2v2_RP'),
        '2mass k': ('2MASS_Ks'),
        '2massk': ('2MASS_Ks'),
        '2mass_k': ('2MASS_Ks'),
        '2mass j': ('2MASS_J'),
        '2massj': ('2MASS_J'),
        '2mass_j': ('2MASS_J'),
        '2mass h': ('2MASS_H'),
        '2massh': ('2MASS_H'),
        '2mass_h': ('2MASS_H'),
        'tycho bt': ('TYCHO_B_MvB'),
        'bt': ('TYCHO_B_MvB'),
        'tycho vt': ('TYCHO_V_MvB'),
        'vt': ('TYCHO_V_MvB'),
        'cousins r': ('GROUND_COUSINS_R'),
        'rc': ('GROUND_COUSINS_R'),
        'cousins i': ('GROUND_COUSINS_I'),
        'ic': ('GROUND_COUSINS_I'),
        'johnson u': ('GROUND_JOHNSON_U'),
        'u': ('GROUND_JOHNSON_U'),
        'johnson v': ('GROUND_JOHNSON_V'),
        'v': ('GROUND_JOHNSON_V'),
        'johnson b': ('GROUND_JOHNSON_B'),
        'b': ('GROUND_JOHNSON_B'),
        'stromgren v': ('STROMGREN_v'),
        'sv': ('STROMGREN_v'),
        'stromgren b': ('STROMGREN_b'),
        'sb': ('STROMGREN_b'),
        'stromgren y': ('STROMGREN_y'),
        'sy': ('STROMGREN_y'),
        'stromgren u': ('STROMGREN_u'),
        'su': ('STROMGREN_u'),
        'tess': ('TESS'),
        'tess t': ('TESS'),
        't': ('TESS'),
        'galex fuv': ('GALEX_FUV'),
        'gfuv': ('GALEX_FUV'),
        'galex nuv': ('GALEX_NUV'),
        'gnuv': ('GALEX_NUV'),
        'ps1 g': ('PS1_g'),
        'panstarrs g': ('PS1_g'),
        'ps1 r': ('PS1_r'),
        'panstarrs r': ('PS1_r'),
        'ps1 i': ('PS1_i'),
        'panstarrs i': ('PS1_i'),
        'ps1 z': ('PS1_z'),
        'panstarrs z': ('PS1_z'),
        'ps1 y': ('PS1_y'),
        'panstarrs y': ('PS1_y'),
        'sdss u': ('SDSS_u'),
        'sloan u': ('SDSS_u'),
        'sdss g': ('SDSS_g'),
        'sloan g': ('SDSS_g'),
        'sdss r': ('SDSS_r'),
        'sloan r': ('SDSS_r'),
        'sdss i': ('SDSS_i'),
        'sloan i': ('SDSS_i'),
        'sdss z': ('SDSS_z'),
        'sloan z': ('SDSS_z'),
        'wise w1': ('WISE_RSR_W1'),
        'w1': ('WISE_RSR_W1'),
        'wise w2': ('WISE_RSR_W2'),
        'w2': ('WISE_RSR_W2'),
        'wise w3': ('WISE_RSR_W3'),
        'w3': ('WISE_RSR_W3'),
        'wise w4': ('WISE_RSR_W4'),
        'w4': ('WISE_RSR_W4')
    }

    filt_name = filt_map.get(filter_name)
    if verbose:
        phot_obj.remove_mag(filt_name)

    if not verbose:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            phot_obj.remove_mag(filt_name)


def save_photometry(starid, phot_obj, out_dir, verbose=False):
    try:
        os.makedirs(out_dir, exist_ok=True)
        print(f"Directory '{out_dir}' created successfully or already exists.")
    except OSError as e:
        print(f"Error creating directory {out_dir}: {e}")
    star_id = starid.replace(" ", "")
    f = Fitter()
    f.star = phot_obj
    f.out_folder = out_dir
    f.star.save_mags(f.out_folder + '/' + star_id)
    if verbose:
        print('File saved:', f.out_folder + '/' + star_id + 'mags.dat')
    fn = f.out_folder + '/' + star_id + 'mags.dat'
    return fn


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
        'gaiadr2v2_g': ('GAIA/GAIA3.G', 'GAIA.GAIA3.G'),
        'gaia gbp': ('GAIA/GAIA3.Gbp', 'GAIA.GAIA3.Gbp'),
        'gbp': ('GAIA/GAIA3.Gbp', 'GAIA.GAIA3.Gbp'),
        'gaia bp': ('GAIA/GAIA3.Gbp', 'GAIA.GAIA3.Gbp'),
        'gaiadr2v2_bp': ('GAIA/GAIA3.Gbp', 'GAIA.GAIA3.Gbp'),
        'gaia grp': ('GAIA/GAIA3.Grp', 'GAIA.GAIA3.Grp'),
        'grp': ('GAIA/GAIA3.Grp', 'GAIA.GAIA3.Grp'),
        'gaia rp': ('GAIA/GAIA3.Grp', 'GAIA.GAIA3.Grp'),
        'gaiadr2v2_rp': ('GAIA/GAIA3.Grp', 'GAIA.GAIA3.Grp'),
        '2mass k': ('2MASS/2MASS.Ks', '2MASS.Ks'),
        '2massk': ('2MASS/2MASS.Ks', '2MASS.Ks'),
        '2mass_k': ('2MASS/2MASS.Ks', '2MASS.Ks'),
        '2mass_ks': ('2MASS/2MASS.Ks', '2MASS.Ks'),
        '2mass j': ('2MASS/2MASS.J', '2MASS.J'),
        '2massj': ('2MASS/2MASS.J', '2MASS.J'),
        '2mass_j': ('2MASS/2MASS.J', '2MASS.J'),
        '2mass h': ('2MASS/2MASS.H', '2MASS.H'),
        '2massh': ('2MASS/2MASS.H', '2MASS.H'),
        '2mass_h': ('2MASS/2MASS.H', '2MASS.H'),
        'tycho bt': ('TYCHO/TYCHO.B_MvB', 'TYCHO.TYCHO.B_MvB'),
        'bt': ('TYCHO/TYCHO.B_MvB', 'TYCHO.TYCHO.B_MvB'),
        'tycho_b_mvb': ('TYCHO/TYCHO.B_MvB', 'TYCHO.TYCHO.B_MvB'),
        'tycho vt': ('TYCHO/TYCHO.V_MvB', 'TYCHO.TYCHO.V_MvB'),
        'vt': ('TYCHO/TYCHO.V_MvB', 'TYCHO.TYCHO.V_MvB'),
        'tycho_v_mvb': ('TYCHO/TYCHO.V_MvB', 'TYCHO.TYCHO.V_MvB'),
        'hipparcos hp': ('Hipparcos/Hipparcos.Hp_MvB', 'Hipparcos.Hipparcos.Hp_MvB'),
        'hp': ('Hipparcos/Hipparcos.Hp_MvB', 'Hipparcos.Hipparcos.Hp_MvB'),
        'cousins u': ('Generic/Cousins.U', 'Cousins.U'),
        'uc': ('Generic/Cousins.U', 'Cousins.U'),
        'cousins v': ('Generic/Cousins.V', 'Cousins.V'),
        'cv': ('Generic/Cousins.V', 'Cousins.V'),
        'cousins b': ('Generic/Cousins.B', 'Cousins.B'),
        'bc': ('Generic/Cousins.B', 'Cousins.B'),
        'cousins r': ('Generic/Cousins.R', 'Cousins.R'),
        'ground_cousins_r': ('Generic/Cousins.R', 'Cousins.R'),
        'rc': ('Generic/Cousins.R', 'Cousins.R'),
        'cousins i': ('Generic/Cousins.I', 'Cousins.I'),
        'ground_cousins_i': ('Generic/Cousins.I', 'Cousins.I'),
        'ic': ('Generic/Cousins.I', 'Cousins.I'),
        'johnson u': ('Generic/Johnson_UBVRIJHKL.U', 'Johnson.U'),
        'ground_johnson_u': ('Generic/Johnson_UBVRIJHKL.U', 'Johnson.U'),
        'u': ('Generic/Johnson_UBVRIJHKL.U', 'Johnson.U'),
        'johnson v': ('Generic/Johnson_UBVRIJHKL.V', 'Johnson.V'),
        'ground_johnson_v': ('Generic/Johnson_UBVRIJHKL.V', 'Johnson.V'),
        'v': ('Generic/Johnson_UBVRIJHKL.V', 'Johnson.V'),
        'johnson b': ('Generic/Johnson_UBVRIJHKL.B', 'Johnson.B'),
        'ground_johnson_b': ('Generic/Johnson_UBVRIJHKL.B', 'Johnson.B'),
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
        'stromgren_v': ('Generic/Stromgren.v', 'Stromgren.v'),
        'stromgren u': ('Generic/Stromgren.u', 'Stromgren.u'),
        'stromgren_u': ('Generic/Stromgren.u', 'Stromgren.u'),
        'sv': ('Generic/Stromgren.v', 'Stromgren.v'),
        'stromgren b': ('Generic/Stromgren.b', 'Stromgren.b'),
        'stromgren_b': ('Generic/Stromgren.b', 'Stromgren.b'),
        'sb': ('Generic/Stromgren.b', 'Stromgren.b'),
        'stromgren y': ('Generic/Stromgren.y', 'Stromgren.y'),
        'stromgren_y': ('Generic/Stromgren.y', 'Stromgren.y'),
        'sy': ('Generic/Stromgren.y', 'Stromgren.y'),
        'tess': ('TESS/TESS.Red', 'TESS.TESS.Red'),
        'tess t': ('TESS/TESS.Red', 'TESS.TESS.Red'),
        't': ('TESS/TESS.Red', 'TESS.TESS.Red'),
        'galex fuv': ('GALEX/GALEX.FUV', 'GALEX.FUV'),
        'galex_fuv': ('GALEX/GALEX.FUV', 'GALEX.FUV'),
        'gfuv': ('GALEX/GALEX.FUV', 'GALEX.FUV'),
        'galex nuv': ('GALEX/GALEX.NUV', 'GALEX.NUV'),
        'galex_nuv': ('GALEX/GALEX.NUV', 'GALEX.NUV'),
        'gnuv': ('GALEX/GALEX.NUV', 'GALEX.NUV'),
        'ps1 g': ('PAN-STARRS/PS1.g', 'PAN-STARRS.PS1.g'),
        'ps1_g': ('PAN-STARRS/PS1.g', 'PAN-STARRS.PS1.g'),
        'panstarrs g': ('PAN-STARRS/PS1.g', 'PAN-STARRS.PS1.g'),
        'ps1 r': ('PAN-STARRS/PS1.r', 'PAN-STARRS.PS1.r'),
        'ps1_r': ('PAN-STARRS/PS1.r', 'PAN-STARRS.PS1.r'),
        'panstarrs r': ('PAN-STARRS/PS1.r', 'PAN-STARRS.PS1.r'),
        'ps1 i': ('PAN-STARRS/PS1.i', 'PAN-STARRS.PS1.i'),
        'ps1_i': ('PAN-STARRS/PS1.i', 'PAN-STARRS.PS1.i'),
        'panstarrs i': ('PAN-STARRS/PS1.i', 'PAN-STARRS.PS1.i'),
        'ps1 z': ('PAN-STARRS/PS1.z', 'PAN-STARRS.PS1.z'),
        'ps1_z': ('PAN-STARRS/PS1.z', 'PAN-STARRS.PS1.z'),
        'panstarrs z': ('PAN-STARRS/PS1.z', 'PAN-STARRS.PS1.z'),
        'ps1 y': ('PAN-STARRS/PS1.y', 'PAN-STARRS.PS1.y'),
        'ps1_y': ('PAN-STARRS/PS1.y', 'PAN-STARRS.PS1.y'),
        'panstarrs y': ('PAN-STARRS/PS1.y', 'PAN-STARRS.PS1.y'),
        'sdss u': ('SLOAN/SDSS.u', 'SDSS.u'),
        'sdss_u': ('SLOAN/SDSS.u', 'SDSS.u'),
        'sloan u': ('SLOAN/SDSS.u', 'SDSS.u'),
        'sdss g': ('SLOAN/SDSS.g', 'SDSS.g'),
        'sdss_g': ('SLOAN/SDSS.g', 'SDSS.g'),
        'sloan g': ('SLOAN/SDSS.g', 'SDSS.g'),
        'sdss r': ('SLOAN/SDSS.r', 'SDSS.r'),
        'sdss_r': ('SLOAN/SDSS.r', 'SDSS.r'),
        'sloan r': ('SLOAN/SDSS.r', 'SDSS.r'),
        'sdss i': ('SLOAN/SDSS.i', 'SDSS.i'),
        'sdss_i': ('SLOAN/SDSS.i', 'SDSS.i'),
        'sloan i': ('SLOAN/SDSS.i', 'SDSS.i'),
        'sdss z': ('SLOAN/SDSS.z', 'SDSS.z'),
        'sdss_z': ('SLOAN/SDSS.z', 'SDSS.z'),
        'sloan z': ('SLOAN/SDSS.z', 'SDSS.z'),
        'wise w1': ('WISE/WISE.W1', 'WISE.W1'),
        'wise_rsr_w1': ('WISE/WISE.W1', 'WISE.W1'),
        'w1': ('WISE/WISE.W1', 'WISE.W1'),
        'wise w2': ('WISE/WISE.W2', 'WISE.W2'),
        'wise_rsr_w2': ('WISE/WISE.W2', 'WISE.W2'),
        'w2': ('WISE/WISE.W2', 'WISE.W2'),
        'wise w3': ('WISE/WISE.W3', 'WISE.W3'),
        'wise_rsr_w3': ('WISE/WISE.W3', 'WISE.W3'),
        'w3': ('WISE/WISE.W3', 'WISE.W3'),
        'wise w4': ('WISE/WISE.W4', 'WISE.W4'),
        'wise_rsr_w4': ('WISE/WISE.W4', 'WISE.W4'),
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


def read_in_photometry(filename, verbose=False):
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
    if filename.endswith('.csv'):
        phot = pd.read_csv(filename)
    elif filename.endswith('.dat'):
        phot = pd.read_csv(filename, sep='\s+', skiprows=1, header=None)
        phot.columns = ['Filter', 'Magnitude', 'Error']

    phot1 = set_filters(phot)
    phot2 = get_zpts(phot1)
    mag = phot2['Magnitude']
    dmag = phot2['Error']
    zptf = phot2['ZPT'] * (1e-9)
    dzptf = phot2['ZPT_err'] * (1e-9)
    wave = phot2['Ref wavelength']
    pbwidth = phot['Effective width'] / 2

    flux = (zptf) * (10 ** (-0.4 * (mag)))
    # Calculating the error for the new fluxes
    term1 = (10 ** (-0.4 * mag)) * (dzptf)
    term2 = -0.4 * np.log(10) * (zptf) * (10 ** (-0.4 * mag)) * dmag
    flux_err = np.sqrt(term1 ** 2 + term2 ** 2)
    name = phot2['Filter']
    idx = np.arange(0, len(flux), 1)

    new_phot = pd.DataFrame(
        {'index': idx, 'sed_filter': phot['SED Filter name'], 'la': wave, 'width': pbwidth, 'flux': flux,
         'eflux': flux_err})
    new_sed = Table.from_pandas(new_phot)
    new_sed['la'].unit = u.AA
    new_sed['width'].unit = u.AA
    new_sed['flux'] = np.log10(flux * wave)
    new_sed['eflux'] = 0.434 * (flux_err / flux)
    new_sed['flux'].unit = u.erg / u.s / (u.cm ** 2) / u.AA
    new_sed['eflux'].unit = u.erg / u.s / (u.cm ** 2) / u.AA

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





def randomize_photometry(sed):
    """Ultra-fast version using numpy's new RNG"""
    rng = default_rng()  # Faster than np.random.normal

    new_sed = sed.copy()
    rand_f = rng.normal(new_sed['flux'], new_sed['eflux'])
    new_sed['flux'] = rand_f

    return new_sed


def calc_fbol(x, unit):
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
    _, _, _, _, model_w, model_f, _ = convert(x, unit = unit)

    F1 = np.trapezoid(model_f, model_w)

    # testing the model's convergence
    wav_fine = np.linspace(model_w.min(), model_w.max(), 100 * len(model_w))
    f_fine = np.interp(wav_fine, model_w, model_f)
    F2 = np.trapezoid(f_fine, wav_fine)

    convergence = ((F1 - F2) / F2) * 100
    if convergence >= 0.1:
        print("Integration does not converge.")

    return F1

def select_bestfit(star, sed_list, chi2list, chi2redlist):
    min_c2r = min(chi2redlist)
    minidx = np.argmin(chi2redlist)
    sed_bf = sed_list[minidx]
    minc2 = chi2list[minidx]
    star.SEDchi2 = minc2
    star.SEDchi2red = min_c2r
    return sed_bf


def fit_sed(sed, star, initial_guess, num_iter, unit, model, teffrange=None, loggrange=None, fehrange=None, avrange = None, fitT=False, fit_logg=False,
            fit_feh=False, fit_av = False, verbose=False, debug = False):
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
    import time

    d = star.dist
    ra = star.ra_hms
    dec = star.dec_dms
    if verbose:
        print("Initializing SEDFit object...")
    #f = io.StringIO()
    #with contextlib.redirect_stdout(f):
    #    try:
    #        x = SEDFit(ra, dec, 1, use_gaia_params=False, use_gaia_xp=False, grid_type=model)
    #        x.addguesses(r=[1])
    #    except AttributeError:
    #        x = SEDFit(ra, dec, 0.5, use_gaia_params=False, use_gaia_xp=False, grid_type=model)
    x = SEDFit(' ', ' ', 1, use_gaia_params = False, use_gaia_xp = False, grid_type = model)
    downloadflux(x, userinput = sed)
    teff, logg, feh, av = initial_guess
    x.dist = d
    x.addguesses(dist = d, av = av, r = [1.], teff=[teff], logg=[logg], feh=feh, alpha = 0., area = False)

    if teffrange is not None:
        x.addrange(teff=teffrange)
    if loggrange is not None:
        x.addrange(logg=loggrange)
    if fehrange is not None:
        x.addrange(feh=fehrange)
    if avrange is not None:
        x.addrange(av=avrange)

    numOparams = 1 + sum([fitT, fit_logg, fit_feh, fit_av])

    fbol = np.zeros(num_iter)
    chi2s = np.zeros(num_iter)
    chi2reds = np.zeros(num_iter)

    best_sed = None
    best_chi2red = np.inf
    best_idx = 0

    print(f"Starting {num_iter} iterations...")
    overall_start = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(num_iter):
            iter_start = time.time()

            if i < 3 or i % 10 == 0:
                print(f"\n[{i + 1}/{num_iter}] Starting iteration...", flush=True)

            rand_sed = randomize_photometry(sed)
            downloadflux(x, rand_sed)

            if i == 0:
                set_quality(x)

            x.fit(use_gaia=False, idx=np.arange(0, len(x.sed['index'])),
                  fitdist=False, fitteff=fitT, fitfeh=fit_feh,
                  fitlogg=fit_logg, fitav=fit_av, quality_check=False)

            chi2s, chi2r = chi2red(x, numOparams, verbose=False)
            chi2reds[i] = chi2r
            fbol[i] = calc_fbol(x, unit)

            if chi2r < best_chi2red:
                best_chi2red = chi2r
                best_chi2 = chi2s
                import copy
                best_sed = copy.deepcopy(x)

            iter_time = time.time() - iter_start

            if i < 3 or i % 10 == 0:
                elapsed = time.time() - overall_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (num_iter - i - 1)
                if debug:
                    print(f"[{i + 1}/{num_iter}] Complete in {iter_time:.1f}s | "
                          f"Avg: {avg_time:.1f}s | ETA: {remaining / 60:.1f} min", flush=True)
    if debug:
        total_time = time.time() - overall_start
        print(f"\n✓ All {num_iter} iterations complete in {total_time / 60:.1f} minutes or {total_time:.3f} seconds")

    avg_fbol = np.mean(fbol)
    std_fbol = np.std(fbol)
    fbol_err = np.sqrt((std_fbol) ** 2 + (0.02 * avg_fbol) ** 2)
    precision = (fbol_err / avg_fbol) * 100
    if verbose:
        print('Bolometric flux: ', round((avg_fbol / 1e-8), 5), '+/-', round((fbol_err / 1e-8), 5),
              'x 10^{-8) erg/s/cm^2')
        print('Fbol precision: ', round(precision, 3), '%')
    star.fbol = round((avg_fbol / 1e-8), 5)
    star.fbol_err = round((fbol_err / 1e-8), 5)

    star.SEDchi2red = best_chi2red
    star.SEDchi2 = best_chi2

    if fitT == True:
        star.SEDTeff = best_sed.getteff()[0]
    if fit_logg == True:
        star.SEDlogg = best_sed.getlogg()[0]
    if fit_feh == True:
        star.SEDfeh = best_sed.getfeh()
    if fit_av == True:
        star.SEDAv = best_sed.getav()

    if verbose:
        print("Distance: {} pc".format(best_sed.getdist()))
        print("AV: {} mag".format(best_sed.getav()))
        print("Radius: {} Rsun".format(best_sed.getr()[0]))
        print("Teff: {} K".format(best_sed.getteff()[0]))
        print("Log g: {} ".format(best_sed.getlogg()[0]))
        print("Fe/H: {}".format(best_sed.getfeh()))

    return best_sed


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
    w = sed.sed['la']
    f = sed.sed['flux']
    dw = sed.sed['width']
    df = sed.sed['eflux']
    sf = sed.mags
    mw = sed.la
    mf = sed.fx.flatten()
    if unit == 'AA':
        lit_w = w  # literature wavelengths
        lit_f = (10 ** f) / np.array(w)  # literature flux
        lit_dw = dw # literature wavelength error
        lit_df = (df / 0.434) * lit_f  # literature flux error

        model_w = mw  # model wavelength
        model_f = (10 ** mf) / mw  # model flux

        synth_f = (10 ** sf) / np.array(w)  # model fluxes for the wavelength

        # residuals = lit_f-synth_f

        return np.array(lit_w), np.array(lit_f), np.array(lit_dw), np.array(lit_df), np.array(model_w), np.array(model_f), np.array(synth_f)

    if unit == 'micron':
        lit_w = w * (1e-4)  # literature wavelengths
        lit_f = ((10 ** f) / np.array(w)) * (1e4)  # literature flux
        lit_dw = dw * (1e-4)  # literature wavelength error
        lit_df = (df / 0.434) * lit_f  # literature flux error

        model_w = mw * (1e-4)  # model wavelength
        model_f = ((10 ** mf) / np.array(mw)) * (1e4)  # model flux

        synth_f = ((10 ** sf) / np.array(w)) * (1e4)  # model fluxes for the wavelength

        # residuals = np.array(lit_f)-np.array(synth_f)

        return np.array(lit_w), np.array(lit_f), np.array(lit_dw), np.array(lit_df), np.array(model_w), np.array(
            model_f), np.array(synth_f)

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
        exp = 0

        return model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res, exp

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
        exp = 0
        return model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res, exp

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

        return model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res, exp

    if not logplot and fbol_lam:
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

        return model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res, exp


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

def setaxislabels(exp, unit, logplot = False, fbol_lam = False):
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
    if logplot:
        if fbol_lam:
            ylab = r'$\rm \lambda F_{\lambda}~[\frac{erg}{cm^{2}~s}$]'
        else:
            if unit == 'AA':
                ylab = r'$\rm F_{\lambda}~[\frac{erg}{cm^{2}~s~\AA}$]'
            elif unit == 'micron':
                ylab = r'$\rm F_{\lambda}~[\frac{erg}{cm^{2}~s~\mu m}$]'
        if unit == 'AA':
            xlab = r'$\rm Wavelength~[\AA]$'
        elif unit == 'micron':
            xlab = r'$\rm Wavelength~[\mu m]$'
        return xlab, ylab
    else:
        if fbol_lam:
            #print('Fbol lam')
            #print('Exponent:', exp)
            if exp < 0:
                #print('Exp < 0')
                ylab = rf'$\rm \lambda F_{{\lambda}}~[\times 10^{{{exp}}}~\frac{{\rm erg}}{{\rm cm^2~s}}]$'
            if unit == 'AA':
                #print('angstroms')
                xlab = r'$\rm Wavelength~[\AA]$'
            elif unit == 'micron':
                #print('Microns')
                xlab = r'$\rm Wavelength~[\mu m]$'
            return xlab, ylab
        else:
            #print('No fbol lam')
            #print('Exoponent:', exp)
            if unit == 'AA':
                #print('Angstroms')
                xlab = r'$\rm Wavelength~[\AA]$'
                if exp < 0:
                    #print('exp < 0')
                    ylab = rf'$\rm F_{{\lambda}}~[\times 10^{{{exp}}}~\frac{{\rm erg}}{{\rm cm^2~s~\AA}}]$'
            elif unit == 'micron':
                #print('Microns')
                xlab = r'$\rm Wavelength~[\mu m]$'
                if exp < 0:
                    #print('Exp < 0')
                    ylab = rf'$\rm F_{{\lambda}}~[\times 10^{{{exp}}}~\frac{{\rm erg}}{{\rm cm^2~s~\mu m}}]$'
            return xlab, ylab


def set_radpy_axis_limits(w, f, exp, unit, logplot):
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
    xmin = min(w)
    xmax = max(w)
    ymin = min(f)
    ymax = max(f)

    if unit == 'AA':
        if logplot:
            Xmin = xmin - (xmin * 0.05)
            Xmax = xmax + (xmax * 0.05)
            Ymin = ymin + (ymin * 0.05)
            Ymax = ymax - (ymax * 0.05)

            new_axis = [round(Xmin, 1), round(Xmax, 1), round(Ymin) * (10 ** (exp)), round(Ymax) * (10 ** (exp))]

            return new_axis
        else:
            Xmin = xmin - (xmin * 0.1)
            Xmax = xmax + (xmax * 0.1)
            Ymin = ymin + (ymin * 0.1)
            Ymax = ymax + (ymax * 0.2)

            new_axis = [round(Xmin, -2), round(Xmax, -3), round(Ymin, 1) * (10 ** (exp)), round(Ymax) * (10 ** (exp))]

            return new_axis
    if unit == 'micron':
        if logplot:
            Xmin = xmin - (xmin * 0.05)
            Xmax = xmax + (xmax * 0.05)
            Ymin = ymin + (ymin * 0.01)
            Ymax = ymax - (ymax * 0.01)

            new_axis = [round(Xmin, 2), round(Xmax, 2), round(Ymin) * (10 ** (exp)), round(Ymax) * (10 ** (exp))]

            return new_axis
        else:
            Xmin = xmin - (xmin * 0.05)
            Xmax = xmax + (xmax * 0.05)
            Ymin = ymin - (ymin * 0.05)
            Ymax = ymax + (ymax * 0.1)

            new_axis = [round(Xmin, 2), round(Xmax, 2), round(Ymin, 1) * (10 ** (exp)), round(Ymax, 1) * (10 ** (exp))]

            return new_axis


def setaxisticklabels(iwave, iflux, exp, unit, set_axis, logplot=False, fbol_lam=False, verbose=False):
    if set_axis:
        xmin = set_axis[0]
        xmax = set_axis[1]
        ymin = set_axis[2]
        ymax = set_axis[3]
    elif set_axis is None:
        set_axis = set_radpy_axis_limits(iwave, iflux, exp, unit, logplot=logplot)
        xmin = set_axis[0]
        xmax = set_axis[1]
        ymin = set_axis[2]
        ymax = set_axis[3]

    if logplot:
        xmin = round((xmin), 1)
        xmax = round((xmax), 1)
        ymin = round(ymin)
        ymax = round(ymax)
        # y ticks
        y_loc = [ymin, ymax]
        y_labels = [rf'$10^{{{y_loc[0]}}}$', rf'$10^{{{y_loc[1]}}}$']

        # x ticks
        xaxis = np.linspace(xmin, xmax, 5)
        # xaxis is in np.log10 space
        x_loc = []
        xl = []
        for i in range(len(xaxis)):
            # converting out of log10 space
            xloc = (10 ** (xaxis[i]))
            if unit == 'AA':
                # xlabel is out of log10 space
                xl.append(int((round(xloc, -3))))
                # xlocation is in log10 space
                x_loc.append(np.log10((round(xloc, -3))))
            if unit == 'micron':
                if xloc < 0.8:
                    # xlabel is out of log10 space
                    xl.append(round(xloc, 1))
                    # xlocation is in log10 space
                    x_loc.append(np.log10(round(xloc, 1)))
                else:
                    # xlabel is out of log10 space
                    xl.append(round(xloc))
                    # xlocation is in log10 space
                    x_loc.append(np.log10(round(xloc)))

        x_labels = [rf'$\rm {(val)}$' for val in xl]
        return x_loc, x_labels, y_loc, y_labels
    else:
        # y axis limits are in 10^exp space
        if fbol_lam:
            # y ticks
            # taking the yvals out of 10^exp space
            yloc = np.linspace(ymin / (10 ** exp), ymax / (10 ** exp), 4)
            y_loc = [round(val) for val in yloc]
            y_labels = [rf'$\rm {round(val)}$' for val in y_loc]

            # x ticks
            xaxis = np.linspace(xmin, xmax, 5)
            x_loc = []
            xl = []
            for i in range(len(xaxis)):
                xloc = xaxis[i]
                if unit == 'AA':
                    xl.append(int((round(xloc, -3))))
                    x_loc.append((round(xloc, -3)))
                if unit == 'micron':
                    if xloc < 1 and xloc > 0:
                        xl.append(round(xloc, 2))
                        x_loc.append(round(xloc, 2))
                    elif xloc == 0:
                        xl.append(round(xloc))
                        x_loc.append(round(xloc))
                    else:
                        if xloc % 1 < 0.5:
                            xl.append(round(np.floor(xloc)))
                            x_loc.append(round(np.floor(xloc)))
                        elif xloc % 1 > 0.5:
                            xl.append(round((xloc)))
                            x_loc.append(round((xloc)))

            x_labels = [rf'${(val)}$' for val in xl]
            return x_loc, x_labels, y_loc, y_labels
        else:
            # y ticks
            yloc = np.linspace(1, ymax / (10 ** exp), 4)
            y_loc = [round(val) for val in yloc]
            y_labels = [rf'$ \rm {round(val)}$' for val in y_loc]
            # x ticks
            xaxis = np.linspace(xmin, xmax, 5)
            x_loc = []
            xl = []
            for i in range(len(xaxis)):
                xloc = xaxis[i]
                if unit == 'AA':
                    xl.append(int((round(xloc, -3))))
                    x_loc.append((round(xloc, -3)))
                if unit == 'micron':
                    if xloc < 1 and xloc > 0:
                        xl.append(round(xloc, 2))
                        x_loc.append(round(xloc, 2))
                    elif xloc == 0:
                        xl.append(round(xloc))
                        x_loc.append(round(xloc))
                    else:
                        if xloc % 1 < 0.5:
                            xl.append(round(np.floor(xloc)))
                            x_loc.append(round(np.floor(xloc)))
                        elif xloc % 1 > 0.5:
                            xl.append(round((xloc)))
                            x_loc.append(round((xloc)))

            x_labels = [rf'${(val)}$' for val in xl]
            return x_loc, x_labels, y_loc, y_labels


def setaxislimits(iwave, iflux, exp, unit, set_axis, logplot=False, fbol_lam=False):
    if set_axis:
        xmin = set_axis[0]
        xmax = set_axis[1]
        ymin = set_axis[2]
        ymax = set_axis[3]
    else:
        set_axis = set_radpy_axis_limits(iwave, iflux, exp, unit, logplot=logplot)
        xmin = set_axis[0]
        xmax = set_axis[1]
        ymin = set_axis[2]
        ymax = set_axis[3]

    if logplot:
        return xmin - 0.1, xmax + 0.1, ymin - 0.5, ymax + 0.1
    else:
        if unit == 'AA':
            return xmin - 500, xmax + 500, (ymin / (10 ** (exp))) - 0.25, (ymax / (10 ** (exp))) + 0.25
        if unit == 'micron':
            return xmin - 0.1, xmax + 0.1, (ymin / (10 ** (exp))) - 0.25, (ymax / (10 ** (exp))) + 0.25

def set_res_axis(res, logplot = False):
    min_res = np.min(res)
    max_res = np.max(res)

    minres = round(min_res, 1)
    maxres = round(max_res, 1)

    if abs(minres) >= abs(maxres):
        if logplot:
            res_axis_min = (abs(minres)+0.05)*-1
            res_axis_max = abs(minres)+0.05
            res_loc = [abs(minres)*-1, 0, abs(minres)]
            res_labels = [rf'$\rm {val}$' for val in res_loc]
        else:
            res_axis_min = (abs(minres)+0.5)*-1
            res_axis_max = abs(minres)+0.5
            res_loc = [round(abs(minres)*-1), 0, round(abs(minres))]
            res_labels = [rf'$\rm {val}$' for val in res_loc]
    elif abs(minres) <= abs(maxres):
        if logplot:
            res_axis_min = (abs(maxres)+0.05)*-1
            res_axis_max = abs(maxres)+0.05
            res_loc = [abs(maxres)*-1, 0, abs(maxres)]
            res_labels = [rf'$\rm {val}$' for val in res_loc]
        else:
            res_axis_min = (abs(maxres)+0.5)*-1
            res_axis_max = abs(maxres)+0.5
            res_loc = [round(abs(maxres)*-1), 0, round(abs(maxres))]
            res_labels = [rf'$\rm {val}$' for val in res_loc]

    return res_axis_min, res_axis_max, res_loc, res_labels

def plot_sed(x, unit, logplot=True, fbol_lam=True, set_axis=None, title=None, savefig=None, uselatex = False, show=True, verbose=False):
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

    #iwave, iflux, idwave, idflux, mw, mf, msf = convert(x, unit)

    plt.rcParams.update({'font.size': 15})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['text.usetex'] = uselatex

    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 3]}, sharex=True)

    model_xvals, model_yvals, litp_xvals, litp_yvals, litp_dxvals, litp_dyvals, synth_yvals, res, exp = set_values(x, unit,
                                                                                                                   logplot=logplot,
                                                                                                                   fbol_lam=fbol_lam,
                                                                                                                   verbose=verbose)
    xmin, xmax, ymin, ymax = setaxislimits(litp_xvals, litp_yvals, exp, unit, set_axis, logplot=logplot, fbol_lam=fbol_lam)
    xloc, xlabels, yloc, ylabels = setaxisticklabels(litp_xvals, litp_yvals, exp, unit, set_axis, logplot=logplot, fbol_lam=fbol_lam)

    axes[0].set_ylim(ymin, ymax)
    axes[1].set_xlim(xmin, xmax)
    axes[0].set_yticks(yloc)
    axes[0].set_yticklabels(ylabels)
    axes[1].set_xticks(xloc)
    axes[1].set_xticklabels(xlabels)

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
    ramin, ramax, rloc, rlabel = set_res_axis(res, logplot = logplot)
    axes[1].set_ylim(ramin, ramax)
    axes[1].set_yticks(rloc)
    axes[1].set_yticklabels(rlabel)
    if logplot:
        if unit == 'micron':
            axes[1].set_ylabel(r'$\rm Residuals$', labelpad=5)
        if unit == 'AA':
            axes[1].set_ylabel(r'$\rm Residuals$', labelpad=5)
    else:
        if unit == 'micron':
            axes[1].set_ylabel(r'$\rm Residuals$', labelpad=0)
        if unit == 'AA':
            axes[1].set_ylabel(r'$\rm Residuals$', labelpad=0)

    xlab, ylab = setaxislabels(exp, unit, logplot=logplot, fbol_lam=fbol_lam)
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
