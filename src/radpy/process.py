import os
#import ipympl
import random
import corner
import statistics
import numpy as np
import pandas as pd
from lmfit import Model
from zero_point import zpt
import scipy.special as ss
from astropy.io import fits
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from astropy.stats import mad_std
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from ldtk import LDPSetCreator, SVOFilter
from scipy.interpolate import LinearNDInterpolator
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)

from radpy.stellar import temp, dist_calc, luminosity, ang_to_lin
from radpy.fitting import random_bracket, random_bracket_ld, chis, sigmacalc, weight_avg, percent_diff, V2, UDV2
from radpy.oifitstodf import oifits_to_pandas
from radpy.datareadandformat import *

"""
Entering the fitting routine steps
"""

#initial fit for the uniform disk diameter using lmfit to do so.
udmodel = Model(UDV2)
udparams = udmodel.make_params(theta = 0.4)
ud_result = udmodel.fit(v2, udparams, sf = spf, weights = 1/(dv2), scale_covar = False)


theta_ilm = ud_result.uvars['theta'].n               #theta value result
dtheta_ilm = ud_result.uvars['theta'].s              #error on the theta value
chisqr_ilm = ud_result.redchi                        #chi squared reduced of the fit
print('Initial fit with lmfit:')
print(ud_result.fit_report())
#sigmacalc(theta_ilm, dtheta_ilm, udtheta, uddtheta)

#initial fit for the limb-darkened disk diameter using lmfit to do so.
T_c = temp(Fbol, dF, theta_ilm, dtheta_ilm)
print(T_c)
class_ldc = ldc_func_K([T_c[0],logg])[0]
print(class_ldc)
ldmodel = Model(V2, independent_vars = ['sf', 'mu'])
ldparams = ldmodel.make_params(theta = 0.4)
ld_result = ldmodel.fit(v2_c, ldparams, sf = spf_c, mu = class_ldc, weights = 1/(dv2_c), scale_covar = False)


ldtheta_ilm = ld_result.uvars['theta'].n               #theta value result
lddtheta_ilm = ld_result.uvars['theta'].s              #error on the theta value
chisqr_ldilm = ld_result.redchi                        #chi squared reduced of the fit
print('Initial fit with lmfit:')
print(ld_result.fit_report())
#sigmacalc(ldtheta_ilm, lddtheta_ilm, ldtheta, lddtheta)

# Uniform disk fit for combined data
UD = []
udmcbs_spf = []
udmcbs_v2 = []
udmcbs_dv2 = []
udmcbs_index = []
MC_num = 71
BS_num = 71
theta_guess = 1
new_err_p = dv2_p * np.sqrt(chisqr_ilm)
new_err_c = dv2_c * np.sqrt(chisqr_ilm)
new_err_v = dv2_v * np.sqrt(chisqr_ilm)
count = 0
for k in range(MC_num):
    rand_wave_p = np.random.normal(wave_p, band_p)
    new_spf_p = B_p / rand_wave_p
    rand_data_p = list(zip(new_spf_p, v2_p, new_err_p, brack_p))
    new_df_p = pd.DataFrame(rand_data_p, columns=['Spf', 'V2', 'sigma_V2', 'Bracket'])

    rand_wave_c = np.random.normal(wave_c, band_c)
    new_spf_c = B_c / rand_wave_c
    rand_data_c = list(zip(new_spf_c, v2_c, new_err_c, brack_c))
    new_df_c = pd.DataFrame(rand_data_c, columns=['Spf', 'V2', 'sigma_V2', 'Bracket'])

    rand_wave_v = np.random.normal(wave_v, band_v)
    new_spf_v = B_v / rand_wave_v
    rand_data_v = list(zip(new_spf_v, v2_v, new_err_v, brack_v))
    new_df_v = pd.DataFrame(rand_data_v, columns=['Spf', 'V2', 'sigma_V2', 'Bracket'])

    for l in range(BS_num):
        new_v2_c = np.random.normal(v2_c, dv2_c)
        rand_data_c2 = list(zip(new_spf_c, new_v2_c, new_err_p))
        new_df_c2 = pd.DataFrame(rand_data_c2, columns=['Spf', 'V2', 'sigma_V2'])

        spfbr_p, v2br_p, dv2br_p, avg_dv2_p = random_bracket(new_df_p, 18)
        new_v2_p = np.random.normal(v2br_p, avg_dv2_p)
        rand_data_p2 = list(zip(spfbr_p, new_v2_p, dv2br_p))
        new_df_p2 = pd.DataFrame(rand_data_p2, columns=['Spf', 'V2', 'sigma_V2'])

        spfbr_v, v2br_v, dv2br_v, avg_dv2_v = random_bracket(new_df_v, 20)
        new_v2_v = np.random.normal(v2br_v, avg_dv2_v)
        rand_data_v2 = list(zip(spfbr_v, new_v2_v, dv2br_v))
        new_df_v2 = pd.DataFrame(rand_data_v2, columns=['Spf', 'V2', 'sigma_V2'])

        frames = [new_df_c2, new_df_p2, new_df_v2]
        new_df = pd.concat(frames, ignore_index=True)

        udbsparams = udmodel.make_params(theta=theta_guess)
        udbs_result = udmodel.fit(new_df['V2'], udbsparams, sf=new_df['Spf'], weights=1 / (new_df['sigma_V2']),
                                  scale_covar=True)

        theta_udbs = udbs_result.uvars['theta'].n  # theta value result

        UD.append(theta_udbs)
        udmcbs_spf.append(new_df['Spf'])
        udmcbs_v2.append(new_df['V2'])
        udmcbs_dv2.append(new_df['sigma_V2'])
        # udmcbs_index.append(ind)
        # count +=1
        # print('Iteration ',count, 'out of ',MC_num*BS_num )

avg_UD = np.mean(UD)
std_UD = mad_std(UD)
print('Uniform Disk Diameter after MC/BS:', round(avg_UD, 4), '+/-', round(std_UD, 5), 'mas')
chisq, chisqr = chis(v2, UDV2(spf, avg_UD), dv2, 1)
print("Chi-squared:", chisq)
print("Chi-squared reduced:", chisqr)

teff_ud = temp(Fbol, dF, avg_UD, std_UD)
T_ud = teff_ud[0]
print("Temperature:", round(teff_ud[0], 1), "+/-", round(teff_ud[1], 1), "K")


#initial fit for the PAVO data
udmodel = Model(UDV2)
udparams = udmodel.make_params(theta = 0.4)
ud_result = udmodel.fit(v2_p, udparams, sf = spf_p, weights = 1/(dv2_p), scale_covar = False)


theta_ilm = ud_result.uvars['theta'].n               #theta value result
dtheta_ilm = ud_result.uvars['theta'].s              #error on the theta value
chisqr_ilm = ud_result.redchi                        #chi squared reduced of the fit
print('Initial fit with lmfit:')
print(ud_result.fit_report())
#sigmacalc(theta_ilm, dtheta_ilm, udtheta, uddtheta)

# Uniform disk fit for PAVO data only
UD = []
udmcbs_spf = []
udmcbs_v2 = []
udmcbs_dv2 = []
udmcbs_index = []
MC_num = 71
BS_num = 71
theta_guess = 1
new_err_p = dv2_p * np.sqrt(chisqr_ilm)
count = 0
for k in range(MC_num):
    rand_wave = np.random.normal(wave_p, band_p)
    new_spf = B_p / rand_wave
    rand_data = list(zip(new_spf, v2_p, new_err_p, brack_p))
    new_df = pd.DataFrame(rand_data, columns=['Spf', 'V2', 'sigma_V2', 'Bracket'])

    for l in range(BS_num):
        spfbr, v2br, dv2br, avg_dv2 = random_bracket(new_df, 18)
        new_v2 = np.random.normal(v2br, avg_dv2)

        udbsparams = udmodel.make_params(theta=theta_guess)
        udbs_result = udmodel.fit(new_v2, udbsparams, sf=spfbr, weights=1 / (dv2br), scale_covar=True)

        theta_udbs = udbs_result.uvars['theta'].n  # theta value result

        UD.append(theta_udbs)
        udmcbs_spf.append(spfbr)
        udmcbs_v2.append(new_v2)
        udmcbs_dv2.append(dv2br)
        # udmcbs_index.append(ind)
        # count +=1
        # print('Iteration ',count, 'out of ',MC_num*BS_num )

avg_UD = np.mean(UD)
std_UD = mad_std(UD)
print('Uniform Disk Diameter after MC/BS:', round(avg_UD, 4), '+/-', round(std_UD, 5), 'mas')
chisq, chisqr = chis(v2_p, UDV2(spf_p, avg_UD), dv2_p, 1)
print("Chi-squared:", chisq)
print("Chi-squared reduced:", chisqr)

teff_ud = temp(Fbol, dF, avg_UD, std_UD)
T_ud = teff_ud[0]
print("Temperature:", round(teff_ud[0], 1), "+/-", round(teff_ud[1], 1), "K")

#Initial fit for VEGA data
udmodel = Model(UDV2)
udparams = udmodel.make_params(theta = 0.4)
ud_result = udmodel.fit(v2_v, udparams, sf = spf_v, weights = 1/(dv2_v), scale_covar = False)


theta_ilm = ud_result.uvars['theta'].n               #theta value result
dtheta_ilm = ud_result.uvars['theta'].s              #error on the theta value
chisqr_ilm = ud_result.redchi                        #chi squared reduced of the fit
print('Initial fit with lmfit:')
print(ud_result.fit_report())
#sigmacalc(theta_ilm, dtheta_ilm, udtheta, uddtheta)

# Uniform disk fit for VEGA data only
UD = []
udmcbs_spf = []
udmcbs_v2 = []
udmcbs_dv2 = []
udmcbs_index = []
MC_num = 71
BS_num = 71
theta_guess = 1
new_err_v = dv2_v * np.sqrt(chisqr_ilm)
count = 0
for k in range(MC_num):
    rand_wave = np.random.normal(wave_v, band_v)
    new_spf = B_v / rand_wave
    rand_data = list(zip(new_spf, v2_v, new_err_v, brack_v))
    new_df = pd.DataFrame(rand_data, columns=['Spf', 'V2', 'sigma_V2', 'Bracket'])

    for l in range(BS_num):
        spfbr, v2br, dv2br, avg_dv2 = random_bracket(new_df, 20)
        new_v2 = np.random.normal(v2br, avg_dv2)

        udbsparams = udmodel.make_params(theta=theta_guess)
        udbs_result = udmodel.fit(new_v2, udbsparams, sf=spfbr, weights=1 / (dv2br), scale_covar=True)

        theta_udbs = udbs_result.uvars['theta'].n  # theta value result

        UD.append(theta_udbs)
        udmcbs_spf.append(spfbr)
        udmcbs_v2.append(new_v2)
        udmcbs_dv2.append(dv2br)
        # udmcbs_index.append(ind)
        # count +=1
        # print('Iteration ',count, 'out of ',MC_num*BS_num )

avg_UD = np.mean(UD)
std_UD = mad_std(UD)
print('Uniform Disk Diameter after MC/BS:', round(avg_UD, 4), '+/-', round(std_UD, 5), 'mas')
chisq, chisqr = chis(v2_v, UDV2(spf_v, avg_UD), dv2_v, 1)
print("Chi-squared:", chisq)
print("Chi-squared reduced:", chisqr)

teff_ud = temp(Fbol, dF, avg_UD, std_UD)
T_ud = teff_ud[0]
print("Temperature:", round(teff_ud[0], 1), "+/-", round(teff_ud[1], 1), "K")

#Initial fit for CLASSIC data
udmodel = Model(UDV2)
udparams = udmodel.make_params(theta = 0.4)
ud_result = udmodel.fit(v2_c, udparams, sf = spf_c, weights = 1/(dv2_c), scale_covar = False)


theta_ilm = ud_result.uvars['theta'].n               #theta value result
dtheta_ilm = ud_result.uvars['theta'].s              #error on the theta value
chisqr_ilm = ud_result.redchi                        #chi squared reduced of the fit
print('Initial fit with lmfit:')
print(ud_result.fit_report())
#sigmacalc(theta_ilm, dtheta_ilm, udtheta, uddtheta)

# Uniform disk fit for CLASSIC data only
UD = []
udmcbs_spf = []
udmcbs_v2 = []
udmcbs_dv2 = []
udmcbs_index = []
MC_num = 71
BS_num = 71
theta_guess = 1
new_err_c = dv2_c * np.sqrt(chisqr_ilm)
count = 0
for k in range(MC_num):
    rand_wave = np.random.normal(wave_c, band_c)
    new_spf = B_c / rand_wave
    rand_data = list(zip(new_spf, v2_c, new_err_c, brack_c))
    new_df = pd.DataFrame(rand_data, columns=['Spf', 'V2', 'sigma_V2', 'Bracket'])

    for l in range(BS_num):
        # spfbr, v2br, dv2br,avg_dv2 = random_bracket(new_df, 20)
        new_v2 = np.random.normal(v2_c, new_err_c)

        udbsparams = udmodel.make_params(theta=theta_guess)
        udbs_result = udmodel.fit(new_v2, udbsparams, sf=new_spf, weights=1 / (new_err_c), scale_covar=True)

        theta_udbs = udbs_result.uvars['theta'].n  # theta value result

        UD.append(theta_udbs)
        udmcbs_spf.append(new_spf)
        udmcbs_v2.append(new_v2)
        udmcbs_dv2.append(new_err_c)
        # udmcbs_index.append(ind)
        # count +=1
        # print('Iteration ',count, 'out of ',MC_num*BS_num )

avg_UD = np.mean(UD)
std_UD = mad_std(UD)
print('Uniform Disk Diameter after MC/BS:', round(avg_UD, 4), '+/-', round(std_UD, 5), 'mas')
chisq, chisqr = chis(v2_c, UDV2(spf_c, avg_UD), dv2_c, 1)
print("Chi-squared:", chisq)
print("Chi-squared reduced:", chisqr)

teff_ud = temp(Fbol, dF, avg_UD, std_UD)
T_ud = teff_ud[0]
print("Temperature:", round(teff_ud[0], 1), "+/-", round(teff_ud[1], 1), "K")

# Limb-darkening fit for combined data
LD = []
ldc_rband = []
ldc_kband = []
new_err_p = dv2_p * np.sqrt(chisqr_ilm)
new_err_c = dv2_c * np.sqrt(chisqr_ilm)
new_err_v = dv2_v * np.sqrt(chisqr_ilm)
MC_num = 71
BS_num = 71
theta_guess = avg_UD
T_new = T_ud
ldmodel = Model(V2, independent_vars=['sf', 'mu'])
ldbsparams = ldmodel.make_params(theta=theta_guess)
for i in range(5):
    LD = []
    ldmcbs_spf = []
    ldmcbs_v2 = []
    ldmcbs_dv2 = []
    ldmcbs_mu = []
    ldc_pv = ldc_func_R([T_new, logg])[0]
    ldc_c = ldc_func_K([T_new, logg])[0]

    for k in range(MC_num):
        rand_wave_p = np.random.normal(wave_p, band_p)
        new_spf_p = B_p / rand_wave_p
        rand_LDC_p = np.random.normal(ldc_pv, 0.02)
        rand_ldc_p = np.ones(len(new_spf_p)) * rand_LDC_p
        rand_data_p = list(zip(new_spf_p, v2_p, new_err_p, rand_ldc_p, brack_p))
        new_df_p = pd.DataFrame(rand_data_p, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        rand_wave_c = np.random.normal(wave_c, band_c)
        new_spf_c = B_c / rand_wave_c
        rand_LDC_c = np.random.normal(ldc_c, 0.02)
        rand_ldc_c = np.ones(len(new_spf_c)) * rand_LDC_c
        rand_data_c = list(zip(new_spf_c, v2_c, new_err_c, rand_ldc_c, brack_c))
        new_df_c = pd.DataFrame(rand_data_c, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        rand_wave_v = np.random.normal(wave_v, band_v)
        new_spf_v = B_v / rand_wave_v
        rand_ldc_v = np.ones(len(new_spf_v)) * rand_LDC_p
        rand_data_v = list(zip(new_spf_v, v2_v, new_err_v, rand_ldc_v, brack_v))
        new_df_v = pd.DataFrame(rand_data_v, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        for l in range(BS_num):
            new_v2_c = np.random.normal(v2_c, new_err_c)
            rand_data_C = list(zip(new_spf_c, new_v2_c, new_err_c, rand_ldc_c))
            new_df_C = pd.DataFrame(rand_data_C, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            spfbr_p, v2br_p, dv2br_p, bsldc_p, avg_dv2_p = random_bracket_ld(new_df_p, 18)
            new_v2_p = np.random.normal(v2br_p, avg_dv2_p)
            rand_data_P = list(zip(spfbr_p, v2br_p, dv2br_p, bsldc_p))
            new_df_P = pd.DataFrame(rand_data_P, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            spfbr_v, v2br_v, dv2br_v, bsldc_v, avg_dv2_v = random_bracket_ld(new_df_v, 20)
            new_v2_v = np.random.normal(v2br_v, avg_dv2_v)
            rand_data_V = list(zip(spfbr_v, v2br_v, dv2br_v, bsldc_v))
            new_df_V = pd.DataFrame(rand_data_V, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            frames = [new_df_C, new_df_P, new_df_V]
            new_df = pd.concat(frames, ignore_index=True)

            ldbs_result = ldmodel.fit(new_df['V2'], ldbsparams, sf=new_df['Spf'], mu=new_df['LDC'],
                                      weights=1 / (new_df['sigma_V2']), scale_covar=True)
            theta_ldbs = ldbs_result.uvars['theta'].n  # theta value result

            LD.append(theta_ldbs)
            ldmcbs_spf.append(new_df['Spf'])
            ldmcbs_v2.append(new_df['V2'])
            ldmcbs_dv2.append(new_df['sigma_V2'])
            # ldmcbs_mu.append(bsldc)

    ldc_rband.append(ldc_pv)
    ldc_kband.append(ldc_c)

    avg_LD = np.mean(LD)
    std_LD = mad_std(LD)
    avg_mu_r = ldc_rband[i]
    avg_mu_k = ldc_kband[i]
    print("Iteration", i + 1)
    print('Limb-darkened Disk Diameter after MC/BS:', round(avg_LD, 4), '+/-', round(std_LD, 5), 'mas')
    print("Limb-darkening coefficient in R:", round(avg_mu_r, 5))
    print("Limb-darkening coefficient in K:", round(avg_mu_k, 5))
    chisqld_r, chisqrld_r = chis(v2, V2(spf, avg_LD, avg_mu_r), dv2, 1)
    chisqld_k, chisqrld_k = chis(v2_c, V2(spf_c, avg_LD, avg_mu_k), dv2_c, 1)
    print("Chi-squared for R band:", chisqld_r)
    print("Chi-squared reduced for R band:", chisqrld_r)
    print("Chi-squared for K band:", chisqld_k)
    print("Chi-squared reduced for K band:", chisqrld_k)

    teff_ld = temp(Fbol, dF, avg_LD, std_LD)
    T_old = T_new
    T_new = teff_ld[0]
    print("Temperature:", round(teff_ld[0], 1), "+/-", round(teff_ld[1], 1), "K")
    percent_diff(T_new, T_old)

# Limb-darkening fit for combined data
LD = []
ldc_rband = []
ldc_kband = []
new_err_p = dv2_p * np.sqrt(chisqr_ilm)
new_err_c = dv2_c * np.sqrt(chisqr_ilm)
new_err_v = dv2_v * np.sqrt(chisqr_ilm)
MC_num = 71
BS_num = 71
theta_guess = avg_UD
T_new = T_ud
ldmodel = Model(V2, independent_vars=['sf', 'mu'])
ldbsparams = ldmodel.make_params(theta=theta_guess)
for i in range(5):
    LD = []
    ldmcbs_spf = []
    ldmcbs_v2 = []
    ldmcbs_dv2 = []
    ldmcbs_mu = []
    ldc_pv = ldc_func_R([T_new, logg])[0]
    ldc_c = ldc_func_K([T_new, logg])[0]

    for k in range(MC_num):
        rand_wave_p = np.random.normal(wave_p, band_p)
        new_spf_p = B_p / rand_wave_p
        rand_LDC_p = np.random.normal(ldc_pv, 0.02)
        rand_ldc_p = np.ones(len(new_spf_p)) * rand_LDC_p
        rand_data_p = list(zip(new_spf_p, v2_p, new_err_p, rand_ldc_p, brack_p))
        new_df_p = pd.DataFrame(rand_data_p, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        rand_wave_c = np.random.normal(wave_c, band_c)
        new_spf_c = B_c / rand_wave_c
        rand_LDC_c = np.random.normal(ldc_c, 0.02)
        rand_ldc_c = np.ones(len(new_spf_c)) * rand_LDC_c
        rand_data_c = list(zip(new_spf_c, v2_c, new_err_c, rand_ldc_c, brack_c))
        new_df_c = pd.DataFrame(rand_data_c, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        rand_wave_v = np.random.normal(wave_v, band_v)
        new_spf_v = B_v / rand_wave_v
        rand_ldc_v = np.ones(len(new_spf_v)) * rand_LDC_p
        rand_data_v = list(zip(new_spf_v, v2_v, new_err_v, rand_ldc_v, brack_v))
        new_df_v = pd.DataFrame(rand_data_v, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        for l in range(BS_num):
            new_v2_c = np.random.normal(v2_c, new_err_c)
            rand_data_C = list(zip(new_spf_c, new_v2_c, new_err_c, rand_ldc_c))
            new_df_C = pd.DataFrame(rand_data_C, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            spfbr_p, v2br_p, dv2br_p, bsldc_p, avg_dv2_p = random_bracket_ld(new_df_p, 18)
            new_v2_p = np.random.normal(v2br_p, avg_dv2_p)
            rand_data_P = list(zip(spfbr_p, v2br_p, dv2br_p, bsldc_p))
            new_df_P = pd.DataFrame(rand_data_P, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            spfbr_v, v2br_v, dv2br_v, bsldc_v, avg_dv2_v = random_bracket_ld(new_df_v, 20)
            new_v2_v = np.random.normal(v2br_v, avg_dv2_v)
            rand_data_V = list(zip(spfbr_v, v2br_v, dv2br_v, bsldc_v))
            new_df_V = pd.DataFrame(rand_data_V, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            frames = [new_df_C, new_df_P, new_df_V]
            new_df = pd.concat(frames, ignore_index=True)

            ldbs_result = ldmodel.fit(new_df['V2'], ldbsparams, sf=new_df['Spf'], mu=new_df['LDC'],
                                      weights=1 / (new_df['sigma_V2']), scale_covar=True)
            theta_ldbs = ldbs_result.uvars['theta'].n  # theta value result

            LD.append(theta_ldbs)
            ldmcbs_spf.append(new_df['Spf'])
            ldmcbs_v2.append(new_df['V2'])
            ldmcbs_dv2.append(new_df['sigma_V2'])
            # ldmcbs_mu.append(bsldc)

    ldc_rband.append(ldc_pv)
    ldc_kband.append(ldc_c)

    avg_LD = np.mean(LD)
    std_LD = mad_std(LD)
    avg_mu_r = ldc_rband[i]
    avg_mu_k = ldc_kband[i]
    print("Iteration", i + 1)
    print('Limb-darkened Disk Diameter after MC/BS:', round(avg_LD, 4), '+/-', round(std_LD, 5), 'mas')
    print("Limb-darkening coefficient in R:", round(avg_mu_r, 5))
    print("Limb-darkening coefficient in K:", round(avg_mu_k, 5))
    chisqld_r, chisqrld_r = chis(v2, V2(spf, avg_LD, avg_mu_r), dv2, 1)
    chisqld_k, chisqrld_k = chis(v2_c, V2(spf_c, avg_LD, avg_mu_k), dv2_c, 1)
    print("Chi-squared for R band:", chisqld_r)
    print("Chi-squared reduced for R band:", chisqrld_r)
    print("Chi-squared for K band:", chisqld_k)
    print("Chi-squared reduced for K band:", chisqrld_k)

    teff_ld = temp(Fbol, dF, avg_LD, std_LD)
    T_old = T_new
    T_new = teff_ld[0]
    print("Temperature:", round(teff_ld[0], 1), "+/-", round(teff_ld[1], 1), "K")
    percent_diff(T_new, T_old)

# Limb-darkening fit for combined data
LD = []
ldc_rband = []
ldc_kband = []
new_err_p = dv2_p * np.sqrt(chisqr_ilm)
new_err_c = dv2_c * np.sqrt(chisqr_ilm)
new_err_v = dv2_v * np.sqrt(chisqr_ilm)
MC_num = 71
BS_num = 71
theta_guess = avg_UD
T_new = T_ud
ldmodel = Model(V2, independent_vars=['sf', 'mu'])
ldbsparams = ldmodel.make_params(theta=theta_guess)
for i in range(5):
    LD = []
    ldmcbs_spf = []
    ldmcbs_v2 = []
    ldmcbs_dv2 = []
    ldmcbs_mu = []
    ldc_pv = ldc_func_R([T_new, logg])[0]
    ldc_c = ldc_func_K([T_new, logg])[0]

    for k in range(MC_num):
        rand_wave_p = np.random.normal(wave_p, band_p)
        new_spf_p = B_p / rand_wave_p
        rand_LDC_p = np.random.normal(ldc_pv, 0.02)
        rand_ldc_p = np.ones(len(new_spf_p)) * rand_LDC_p
        rand_data_p = list(zip(new_spf_p, v2_p, new_err_p, rand_ldc_p, brack_p))
        new_df_p = pd.DataFrame(rand_data_p, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        rand_wave_c = np.random.normal(wave_c, band_c)
        new_spf_c = B_c / rand_wave_c
        rand_LDC_c = np.random.normal(ldc_c, 0.02)
        rand_ldc_c = np.ones(len(new_spf_c)) * rand_LDC_c
        rand_data_c = list(zip(new_spf_c, v2_c, new_err_c, rand_ldc_c, brack_c))
        new_df_c = pd.DataFrame(rand_data_c, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        rand_wave_v = np.random.normal(wave_v, band_v)
        new_spf_v = B_v / rand_wave_v
        rand_ldc_v = np.ones(len(new_spf_v)) * rand_LDC_p
        rand_data_v = list(zip(new_spf_v, v2_v, new_err_v, rand_ldc_v, brack_v))
        new_df_v = pd.DataFrame(rand_data_v, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        for l in range(BS_num):
            new_v2_c = np.random.normal(v2_c, new_err_c)
            rand_data_C = list(zip(new_spf_c, new_v2_c, new_err_c, rand_ldc_c))
            new_df_C = pd.DataFrame(rand_data_C, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            spfbr_p, v2br_p, dv2br_p, bsldc_p, avg_dv2_p = random_bracket_ld(new_df_p, 18)
            new_v2_p = np.random.normal(v2br_p, avg_dv2_p)
            rand_data_P = list(zip(spfbr_p, v2br_p, dv2br_p, bsldc_p))
            new_df_P = pd.DataFrame(rand_data_P, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            spfbr_v, v2br_v, dv2br_v, bsldc_v, avg_dv2_v = random_bracket_ld(new_df_v, 20)
            new_v2_v = np.random.normal(v2br_v, avg_dv2_v)
            rand_data_V = list(zip(spfbr_v, v2br_v, dv2br_v, bsldc_v))
            new_df_V = pd.DataFrame(rand_data_V, columns=['Spf', 'V2', 'sigma_V2', 'LDC'])

            frames = [new_df_C, new_df_P, new_df_V]
            new_df = pd.concat(frames, ignore_index=True)

            ldbs_result = ldmodel.fit(new_df['V2'], ldbsparams, sf=new_df['Spf'], mu=new_df['LDC'],
                                      weights=1 / (new_df['sigma_V2']), scale_covar=True)
            theta_ldbs = ldbs_result.uvars['theta'].n  # theta value result

            LD.append(theta_ldbs)
            ldmcbs_spf.append(new_df['Spf'])
            ldmcbs_v2.append(new_df['V2'])
            ldmcbs_dv2.append(new_df['sigma_V2'])
            # ldmcbs_mu.append(bsldc)

    ldc_rband.append(ldc_pv)
    ldc_kband.append(ldc_c)

    avg_LD = np.mean(LD)
    std_LD = mad_std(LD)
    avg_mu_r = ldc_rband[i]
    avg_mu_k = ldc_kband[i]
    print("Iteration", i + 1)
    print('Limb-darkened Disk Diameter after MC/BS:', round(avg_LD, 4), '+/-', round(std_LD, 5), 'mas')
    print("Limb-darkening coefficient in R:", round(avg_mu_r, 5))
    print("Limb-darkening coefficient in K:", round(avg_mu_k, 5))
    chisqld_r, chisqrld_r = chis(v2, V2(spf, avg_LD, avg_mu_r), dv2, 1)
    chisqld_k, chisqrld_k = chis(v2_c, V2(spf_c, avg_LD, avg_mu_k), dv2_c, 1)
    print("Chi-squared for R band:", chisqld_r)
    print("Chi-squared reduced for R band:", chisqrld_r)
    print("Chi-squared for K band:", chisqld_k)
    print("Chi-squared reduced for K band:", chisqrld_k)

    teff_ld = temp(Fbol, dF, avg_LD, std_LD)
    T_old = T_new
    T_new = teff_ld[0]
    print("Temperature:", round(teff_ld[0], 1), "+/-", round(teff_ld[1], 1), "K")
    percent_diff(T_new, T_old)

# Limb-darkening fitting routine for CLASSIC data only
LD = []
ldc_kband = []
new_err_c = dv2_c * np.sqrt(chisqr_ilm)
MC_num = 71
BS_num = 71
theta_guess = avg_UD
T_new = T_ud
ldmodel = Model(V2, independent_vars=['sf', 'mu'])
ldbsparams = ldmodel.make_params(theta=theta_guess)

for i in range(5):
    LD = []
    ldmcbs_spf = []
    ldmcbs_v2 = []
    ldmcbs_dv2 = []
    ldmcbs_mu = []
    ldc_c = ldc_func_K([T_new, logg])[0]

    for k in range(MC_num):
        rand_wave_c = np.random.normal(wave_c, band_c)
        new_spf_c = B_c / rand_wave_c
        rand_LDC_c = np.random.normal(ldc_c, 0.02)
        rand_ldc_c = np.ones(len(new_spf_c)) * rand_LDC_c
        rand_data_c = list(zip(new_spf_c, v2_c, new_err_c, rand_ldc_c, brack_c))
        new_df_c = pd.DataFrame(rand_data_c, columns=['Spf', 'V2', 'sigma_V2', 'LDC', 'Bracket'])

        for l in range(BS_num):
            new_v2 = np.random.normal(v2_c, new_err_c)

            ldbs_result = ldmodel.fit(new_v2, ldbsparams, sf=new_spf_c, mu=rand_ldc_c, weights=1 / (dv2_c),
                                      scale_covar=True)
            theta_ldbs = ldbs_result.uvars['theta'].n  # theta value result

            LD.append(theta_ldbs)
            ldmcbs_spf.append(new_spf_c)
            ldmcbs_v2.append(new_v2)
            ldmcbs_dv2.append(new_err_c)

    ldc_kband.append(ldc_c)

    avg_LD = np.mean(LD)
    std_LD = mad_std(LD)
    print("Iteration", i + 1)
    print('Limb-darkened Disk Diameter after MC/BS:', round(avg_LD, 4), '+/-', round(std_LD, 5), 'mas')
    print("Limb-darkening coefficient in K:", round(ldc_kband[i], 5))
    chisqld_k, chisqrld_k = chis(v2_c, V2(spf_c, avg_LD, ldc_kband[i]), dv2_c, 1)
    print("Chi-squared for R band:", chisqld_k)
    print("Chi-squared reduced for R band:", chisqrld_k)

    teff_ld = temp(Fbol, dF, avg_LD, std_LD)
    T_old = T_new
    T_new = teff_ld[0]
    print("Temperature:", round(teff_ld[0], 1), "+/-", round(teff_ld[1], 1), "K")
    percent_diff(T_new, T_old)


print("Angular diameter:", theta, "+/-", dtheta, "[mas]")
print("LDC R:", muR)
print("LDC K:", muK)
print("Bolometric Flux:", Fbol, "+/-", dF, "[10^-8 ergs/cm^2/s]")
print("GAIA Corrected Distance:", D, "+/-", dD, "[pc]")
print("Parallax:", p, "+/-", pc_err, "[mas]")
ang_to_lin(theta, dtheta, D, dD)
teff = temp(Fbol, dF, theta, dtheta)
print("Temperature: ", teff[0] ,"+/-",teff[1],"[K]")
luminosity(D, dD, Fbol*(1e-8), dF*(1e-8))