
from radpy.stellar import *
from radpy.datareadandformat import *
from radpy.plotting import plot_v2_fit
from radpy.LDfitting import initial_LDfit, run_LDfit
from radpy.config import vegapath, classicpath, pavopath
from radpy.UDfitting import initial_UDfit, run_udmcbs_fit, udfit_values


#Reading in the data files first
filenamec = classicpath
filenamep = pavopath
filenamev = vegapath

datav, num_brack_v = filename_extension(filenamev, 'V')
datac, num_brack_c = filename_extension(filenamec, 'C')
datap, num_brack_p = filename_extension(filenamep, 'P')

pavo_data = PavoData(datap)
vega_data = VegaData(datav)
classic_data = ClassicData(datac)

b, v2, dv2, wave, band, brack, inst = combined(pavo_data.make_df(), classic_data.make_df(), vega_data.make_df())
spf = b/wave

#Stellar parameters
p = 152.864
dp = 0.0494
f = 21.751
df = 0.585
logg = 4.5
dlogg = 0.1
m = 0.09
dm = 0.08
#distances("HD 219134", verbose = True)
D, dD = distances('HD 219134', verbose = True)

star = StellarParams()
star.fbol = f
star.fbol_err = df
star.logg = logg
star.logg_err = dlogg
star.feh = m
star.feh_err = dm
star.dist = D
star.dist_err = dD
star.plx = p
star.plx_err = dp

print("Stellar parameters:")
print(star)

theta1, dtheta1, chisqr1 = initial_UDfit(spf, v2, dv2, 0.4, star, verbose = False)
#initial_UDfit(spf, v2, dv2, 0.4, star, verbose = False)
theta2, dtheta2, chisqr2 = initial_LDfit(spf, v2, dv2, star, 'R', verbose = True)
#initial_LDfit(spf, v2, dv2, star, 'R', verbose = True)

print("Initial fitting results:")
print(star)

results = run_udmcbs_fit(10, 10, datasets = [pavo_data, vega_data, classic_data], stellar_params = star)
udfit_values(spf, v2, dv2, results, stellar_params = star, verbose = True)

thetaf, dthetaf, tf, dtf, ldcsf, chisf = run_LDfit(2, 2, ogdata = [spf, v2, dv2], datasets = [pavo_data, vega_data, classic_data], stellar_params = star, verbose = True)

print("Final fitting results:")
print(star)

print("Final stellar parameters:")
calc_star_params(star, verbose = True)
print(star)


data_dict = {'pavo':pavo_data, 'vega':vega_data, 'classic':classic_data}
plot_v2_fit(
    data_dict=data_dict,
    star=star,          # Your Star instance with theta, ldcR, etc., and .V2 method
    line_spf=np.linspace(0.00001, 2.5e8, 1000),
    set_axis = [0, 2.5e8, -0.05,1.1],
    ldc_band='ldc_R',        # or 'ldcR', 'ldcH', etc.
    datasets_to_plot=['pavo', 'vega', 'classic'],
    plot_udmodel = True,
    to_bin = ['pavo'],
    eq_text=True,
    show = True
)



