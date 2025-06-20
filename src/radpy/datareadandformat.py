import pandas as pd
import numpy as np
from radpy.stellar import dist_calc
from zero_point import zpt
from scipy.interpolate import LinearNDInterpolator
from radpy.config import classicpath, pavopath, vegapath, ldcipath, ldckpath, ldcrpath

#reading in the PAVO data
#PAVO data has 5 columns, spatial frequency (baseline/lambda), the V2, error on the V2, then the u and v coordinates
#the u and v coordinates can be used to determine the baseline by adding them in quadrature. Then we can get the wavelength
#out by dividing the baseline by the spatial frequency

pavo = pd.read_csv(pavopath)
spf_p = pavo['B/lambda']                    #spatial frequency in rad^-1
v2_p = pavo['V2']                           #Visibility squared
dv2_p = pavo['sigma_V2']                    #error on V2
ucoord = pavo['U(meters)']                #U coords (meters)
vcoord = pavo['V(meters)']                #V coords (meters)
brack_p = pavo['Bracket']
B_p = np.sqrt((ucoord**2) + (vcoord**2))    #Baseline (meters)
wave_p = B_p/spf_p                              #wavelengths for each v2 (meters)
band_p = 5e-9                               #error on the wavelength taken as 5nm (converted into meters)
inst_p = pavo['Instrument']
np.random.seed()

#below are PAVOs results
udtheta = 1.037             #PAVO UD angular diameter in mas
uddtheta = 0.008            #PAVO UD error in ang diam in mas
ldtheta = 1.119             #PAVO LD angular diameter
lddtheta = 0.011            #PAVO LD angular diamter
pavo_ldc = 0.7039725        #PAVO LDC (used my interpolation function to find this originally)

#Reading in the CLASSIC data
#Classic data has 7 columns, MJD, baseline in meters, a column tabby told me to ignore, the visibility, and the error, a flag, and the JD again
#Note: Classic visibilities are not squared so need to do that myself
classic = pd.read_csv(classicpath)
B_c = classic['B']            #baseline (meters)
v_c = classic['Vis']          #visbility
dv_c = classic['Vis_e']       #vis error
brack_c = classic['Bracket']  #"bracket": for classic they are split into dates of observations
v2_c = v_c**2                 #visibility squared
dv2_c = v2_c*np.sqrt(2*(dv_c/v_c)**2)   #error on v2
wave_c = pd.Series(2.1329e-6*np.ones(len(B_c)))            #CLASSIC operates in K' band which is centered on this (meters)
band_c = 5e-9                 #error on wavelength taken as 5 nm (converted to meters)
inst_c = classic['Instrument']
spf_c = B_c/wave_c
avg_dv2_c = classic['Avg_v2_err']   #average error for v2 per night

#Reading in the VEGA data
#Vega's data was taken from Roxanne's paper. Has MJD, UT, Telescope, Baseline, sequence, S/N, V2, sigma_Stat, sigma_sys, lambda, and bandwidth
vega = pd.read_csv(vegapath)
B_v = vega['Baseline length']   #baseline (meters)
v2_v = vega['V2']               #Visibility squared
dv2_sys = vega['sigma_sys']     #systematic error on v2
dv2_stat = vega['sigma_stat']   #statistical error on v2
wave_v = vega['lambda']*1e-9    #wavelength in meters
band_v = 5e-9                   #error on wavelength taken as 5 nm
brack_v = vega['Bracket']       #bracket on vega data, separated into time stamps: total of 20 brackets
dv2_v = np.sqrt((dv2_sys**2) + (dv2_stat)**2)  #combined error
inst_v = vega['Instrument']
spf_v = B_v/wave_v
pavo_ldc = 0.7039725        #PAVO LDC (used my interpolation function to find this originally)

#combining the three data sets into one dataframe to work with
b = pd.concat([B_p, B_c, B_v], axis = 0, ignore_index = True)
v2 = pd.concat([v2_p, v2_c, v2_v],axis = 0, ignore_index = True)
dv2 = pd.concat([dv2_p, dv2_c, dv2_v], axis = 0, ignore_index = True)
brack = pd.concat([brack_p, brack_c, brack_v], axis = 0, ignore_index = True)
wave = pd.concat([wave_p, wave_c, wave_v], axis = 0, ignore_index = True)
inst = pd.concat([inst_p, inst_c, inst_v], axis = 0, ignore_index = True)

spf = b/wave

#getting the GAIA corrected parallax, distance, and reading in the rest of the values needed
zpt.load_tables()

phot_g_mean_mag = 5.231896
nu_eff_used_in_astrometry = 1.472
pseudocolour = 7.1
ecl_lat = 54.54138702773
astrometric_params_solved = 31
correction = zpt.get_zpt(phot_g_mean_mag, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved)

p = 152.864                 #units in mas
pc_err = 0.0494             #units in mas
Fbol = 21.751               #bolometric flux in 10e-8 ergs/cm^2/s
dF = 0.585                  #bolometric flux error in ergs/cm^2/s
logg = 4.5

distance = dist_calc(p,pc_err,correction)
D = distance[0]
dD = distance[1]

# LDCs from Claret et al. 2011
# Using the Phoenix model and least squares method
#Need the K and R filter to accomodate CLASSIC (K band), VEGA (R band), and PAVO (R band)
ldc_dataR = pd.read_csv(ldcrpath)
tempsR = ldc_dataR['Teff'].tolist()
gravityR = ldc_dataR['logg'].tolist()
musR = ldc_dataR['u'].tolist()

ldc_dataK = pd.read_csv(ldckpath)
tempsK = ldc_dataK['Teff'].tolist()
gravityK = ldc_dataK['logg'].tolist()
musK = ldc_dataK['u'].tolist()
# Define the interpolation function
ldc_func_R = LinearNDInterpolator((tempsR, gravityR), musR)
ldc_func_K = LinearNDInterpolator((tempsK, gravityK), musK)


