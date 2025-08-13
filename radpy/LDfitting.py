import numpy as np
import pandas as pd
from lmfit import Model
import concurrent.futures
import scipy.special as ss
from radpy.stellar import temp
from astropy.stats import mad_std
from radpy.UDfitting import chis, weight_avg, percent_diff, safe_theta_extraction
from radpy.limbdarkcoeffs import ldc_calc

import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.")
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns")
#Limb-darkened disk V2 equation
def V2(sf, theta, mu, V0):
    alpha = 1-mu
    beta = mu
    x = np.pi*sf*(theta/(206265*1000))
    vis = (V0**2)*((((alpha/2)+(beta/3))**(-2))*((alpha*(ss.jv(1,x)/x))+ beta*(np.sqrt(np.pi/2)*(ss.jv(3/2,x)/(x**(3/2)))))**2)
    return vis
##########################################################################################
# Random bracket function for bootstrapping for limb-darkening
def random_bracket_ld(df, num_of_brackets):
    ###########################################################################
    # Function: random_bracket                                                #
    # Inputs: df -> dataframe with spatial frequency, v2, v2_err, and bracket #
    #         num_of_brackets -> number of brackets in total you have         #
    # Outputs: spf_br -> the spatial frequencies randomized                   #
    #          v2_br -> the visibility squared randomized                     #
    #          dv2_br -> the error on the v2 randomized                       #
    #          ldc_br -> the limb-darkening coefficients
    #          wavgs -> weighted averages of the v2                           #
    # What it does:                                                           #
    #      1. sets the seed                                                   #
    #      2. picks a random number between 2 and the number of brackets      #
    #      3. Selects that many unique bracket labels at random               #
    #      4. Filters the dataframe down to only those with those specific    #
    #         bracket labels                                                  #
    #      5. Groups by the bracket labels                                    #
    #      6. Applies the weight average to the grouped data                  #
    #      7. Adds the weighted average as a column in the data frame         #
    #      8. Merges the groups into a new group for weighted average         #
    #         based on the bracket                                            #
    #      9. Splits up the dataframe into spatial frequency, v2, dv2, ldcs,  #
    #         and wavg                                                        #
    #     10. Returns spatial frequency, v2, dv2, and wavg                    #
    ###########################################################################
    np.random.seed()
    xdata = []
    ydata = []
    dydata = []
    ldcdata = []
    numbr = np.random.randint(2, num_of_brackets)
    # chatgpt wrote the next couple lines
    random_group_ids = df['Bracket'].drop_duplicates().sample(n=numbr).values

    # Filter the DataFrame for the selected groups
    random_groups = df[df['Bracket'].isin(random_group_ids)]

    grouped = random_groups.groupby('Bracket')
    results = grouped.apply(weight_avg).reset_index()
    results.columns = ['Bracket', 'Wavg']
    random_groups_with_avg = random_groups.merge(results, on='Bracket')

    spf_br = random_groups_with_avg['Spf']  # spatial frequency in rad^-1
    v2_br = random_groups_with_avg['V2']  # Visibility squared
    dv2_br = random_groups_with_avg['dV2']
    ldc_br = random_groups_with_avg['LDC']
    wavgs = random_groups_with_avg['Wavg']

    return spf_br, v2_br, dv2_br, ldc_br, wavgs

##########################################################################################
def initial_LDfit(spf, v2, dv2, star_params, filt, v0_flag = True, verbose=False):
    #####################################################################
    # Function: initial_LDfit                                           #
    # Inputs: spf -> spatial frequency                                  #
    #         v2 -> visibilitity squared                                #
    #         dv2 -> error on the V2                                    #
    #         theta_guess -> initial guess for theta                    #
    #         star_params -> stellar class object                       #
    #         verbose -> if set to True, allows print statements        #
    #                    defaults to False                              #
    # Outputs: ldtheta_ilm -> initial uniform disk diameter             #
    #          lddtheta_ilm -> error on the diameter                    #
    #          chisqr_ldilm -> chi squared reduced value                #
    # What it does:                                                     #
    #        1. Calculates the temperature using the initial UD         #
    #        2. Calculated the LDC                                      #
    #        3. Initialized the model                                   #
    #        4. initializes the parameters                              #
    #        5. Fits for the UD diameter using lmfit                    #
    #           uses for the weights as 1/dv2                           #
    #        6. pulls out the theta, dtheta, and chi squared reduced    #
    #        7. updates the stellar object                              #
    #        8. Returns the theta, dtheta, and chi squared reduced      #
    #####################################################################
    t, dt = temp(star_params.fbol, star_params.fbol_err, star_params.udthetai, star_params.udthetai_err)
    ldc = ldc_calc(t, star_params.logg, star_params.feh, filt)

    ldmodel = Model(V2, independent_vars=['sf', 'mu'])
    ldparams = ldmodel.make_params(theta=star_params.udthetai, V0 = star_params.udv0i)
    if not v0_flag:
        ldparams['V0'].set(value = 1.0, vary = False)
    ld_result = ldmodel.fit(v2, ldparams, sf=spf, mu=ldc, weights= 1 / (dv2), scale_covar=False)
    ldtheta_ilm, lddtheta_ilm, ldv0_ilm, lddv0_ilm = safe_param_extraction(ld_result)
    chisqr_ldilm = ld_result.redchi  # chi squared reduced of the fit

    star_params.update(ldthetai=round(ldtheta_ilm,5), ldthetai_err = round(lddtheta_ilm,5), teff=round(t,5), teff_err=round(dt,5), ldv0i = round(ldv0_ilm, 5), ldv0i_err = round(lddv0_ilm))
    if verbose:
        print("Effective temperature:", round(t,5), "+/-", round(dt,5), "K")
        print("LDC for filter ", filt, ":", round(ldc,5))
        print('Initial fit with lmfit:')
        print(ld_result.fit_report())

    return ldtheta_ilm, lddtheta_ilm, chisqr_ldilm, ldv0_ilm, lddv0_ilm


def bootstrap_ld(df, inst):
    ###########################################################
    # Function: bootstrap_ld                                  #
    # Inputs: df -> the data dataframe                        #
    #         inst -> the intstrument                         #
    # Outputs: the new_df                                     #
    # What it does:                                           #
    #       1. If the instrument is set to c (Classic),       #
    #          samples the V2 on a normal distribution        #
    #          and creates a new dataframe with that.         #
    #       2. If the instrument is any others, determines    #
    #          the number of brackets in the dataset.         #
    #       3. calls the random_bracket function              #
    #       4. samples the V2 on a normal distribution        #
    #       5. creates a new dataframe with that              #
    #       6. Returns the new dataframe                      #
    ###########################################################
    if inst == 'c' or inst == 'C':
        newv2 = np.random.normal(df['V2'], df['dV2'])
        new_df = pd.DataFrame(np.column_stack((df['Spf'], newv2, df['dV2'], df['LDC'])),
                              columns=['Spf', 'V2', 'dV2', 'LDC'])
        return new_df
    else:
        num_brackets = df['Bracket'].max()
        spfbr, v2br, dv2br, ldcbr, avgdv2 = random_bracket_ld(df, num_brackets)
        newv2 = np.random.normal(v2br, avgdv2)
        new_df = pd.DataFrame(np.column_stack((spfbr, newv2, dv2br, ldcbr)), columns=['Spf', 'V2', 'dV2', 'LDC'])
        return new_df


def ldfit(df, stellar_params, v0_flag = True, verbose=False):
    #####################################################################
    # Function: ldfit                                                   #
    # Inputs: df -> dataframe with data in it                           #
    #         star_params -> stellar class object                       #
    #         verbose -> if set to True, allows print statements        #
    #                    defaults to False                              #
    # Outputs: theta_ld -> initial uniform disk diameter                #
    # What it does:                                                     #
    #        1. Initialized the model                                   #
    #        2. initializes the parameters                              #
    #        3. Fits for the LD diameter using lmfit                    #
    #           uses for the weights as 1/dv2                           #
    #        4. pulls out the theta                                     #
    #        5. Returns the theta                                       #
    #####################################################################
    ldmodel = Model(V2, independent_vars=['sf', 'mu'])
    ld_params = ldmodel.make_params(theta=stellar_params.udtheta, V0 = stellar_params.udv0)
    #print(v0_flag)
    if not v0_flag:
        ld_params['V0'].set(value = 1.0, vary = False)
    ld_result = ldmodel.fit(df['V2'], ld_params, sf=df['Spf'], mu=df['LDC'], weights=1 / (df['dV2']), scale_covar=True)
    theta_ld, _, v0_ld, _ = safe_param_extraction(ld_result)
    #theta_ld = ld_result.uvars['theta'].n
    #print(v0_ld)

    return theta_ld, v0_ld


def ldfit_values(x, y, dy, LD, V0, ldcs, stellar_params, v0_flag = True, verbose=False):
    ##################################################################
    # Function: ldfit_values                                         #
    # Inputs: x -> the spatial frequencies                           #
    #         y -> the V2                                            #
    #        dy -> the error on the V2                               #
    #        LD -> the list of diameters                             #
    #        ldcs -> limb darkening coefficients                     #
    #        stellar_params -> the star object                       #
    #        verbose - > if true, returns print statements           #
    # Outputs: avg_LD -> average limb darkened diameter              #
    #          std_LD -> the median absolute deviation of LD theta   #
    #          teff_ld[0] -> effective temperature                   #
    #          teff_ld[1] -> error on the effective temperature      #
    #          ldc_results -> the ldc for each band                  #
    #          chisq_results -> the chi square and chi square red    #
    #                           values for each ldc band             #
    # What it does:                                                  #
    #     1. Takes the mean of the limb-darkened disk diameters      #
    #     2. Takes the median absolute deviation of the LDs          #
    #     3. Calculates the effective temperature using the mean     #
    #     4. Initializes the ldc_results and chisq_results           #
    #        to store dynamically                                    #
    #     5. For each band in the ldcs, calculates the V2 model      #
    #     6. Calculates the chi squared and chi squared reduced      #
    #        for each LDC band                                       #
    #     7. Stores the results in the ldc_results and chisq_results #
    #     8. Returns the avg_LD, std_LD, teff and teff error, the    #
    #        ldc results, and the chi squared results                #
    ##################################################################
    avg_LD = np.mean(LD)
    std_LD = mad_std(LD)
    avg_V0 = np.mean(V0)
    std_V0 = mad_std(V0)
    if not v0_flag:
        avg_V0 = 1.0
        std_V0 = 0.0
    teff_ld = temp(stellar_params.fbol, stellar_params.fbol_err, avg_LD, std_LD)
    # Store results dynamically
    ldc_results = {}
    chisq_results = {}

    for band in ldcs:
        ldc_val = ldcs[band]
        if ldc_val is not None:
            model_v2 = V2(x, avg_LD, ldc_val, avg_V0)
            chisq, chisqr = chis(y, model_v2, dy, 1)
            ldc_results[band] = ldc_val
            chisq_results[band] = {"chisq": chisq, "chisqr": chisqr}

    if verbose:
        print('Limb-darkened Disk Diameter after MC/BS:', round(avg_LD, 4), '+/-', round(std_LD, 5), 'mas')
        print('V0^2:', round(avg_V0, 5), '+/-', std_V0)
        for band, ldc_val in ldc_results.items():
            print(f"Limb-darkening coefficient in {band}:", round(ldc_val, 5))
            print(f"Chi-squared for {band} band:", round(chisq_results[band]["chisq"], 3))
            print(f"Reduced chi-squared for {band} band:", round(chisq_results[band]["chisqr"], 3))
        print("Temperature:", round(teff_ld[0], 1), "+/-", round(teff_ld[1], 1), "K")

    return avg_LD, std_LD, avg_V0, std_V0, teff_ld[0], teff_ld[1], ldc_results, chisq_results


def mcbs_worker(args):
    #############################################################
    # Function: mcbs_worker                                     #
    # Inputs: args -> the mc_dfs, the bs_num, stellar_params,   #
    #                 and verbose                               #
    # Outputs: the limb darkened disk list                      #
    # What it does:                                             #
    #       1. unpacks the arguments                            #
    #       2. Initializes the limb-darkened disk list          #
    #       3. Enters the bootstrap loop                        #
    #       4. For each dataframe created in the Monte Carlo    #
    #          loop, it determines which instrument, then       #
    #          calls the bootstrap function for the ld          #
    #       5. Appends the resulting dataframe to the list      #
    #       6. Concatenates all the bootstrapped dfs into one   #
    #       7. Calls ldfit and fits for the limb-darkened theta #
    #       8. Appends results to the LD list                   #
    #       9. Returns the LD list                              #
    #############################################################

    mc_dfs, bs_num, stellar_params, v0_flag, verbose = args
    #print(v0_flag)
    LD = []
    V0 = []
    for _ in range(bs_num):
        bs_dfs = []
        for df in mc_dfs:
            inst = df["Instrument"].iloc[0]
            boot_df = bootstrap_ld(df, inst)
            bs_dfs.append(boot_df)
        new_df = pd.concat(bs_dfs, ignore_index=True)
        theta_ldbs, V0_ldbs = ldfit(new_df, stellar_params, v0_flag, verbose)
        LD.append(theta_ldbs)
        V0.append(V0_ldbs)
    return LD, V0



def run_LDfit(mc_num, bs_num, ogdata, datasets, stellar_params, v0_flag = True, verbose=False, debug=False):
    ######################################################################
    # Function: run_ldmcbs_fit_parallel                                  #
    # Inputs: mc_num -> number of Monte Carlo iterations                 #
    #         bs_num -> number of bootstrap iterations                   #
    #         ogdata -> original data sets                               #
    #         datasets -> the datasets you want fit                      #
    #                     format: [inst1, inst2, inst3]                  #
    #         stellar_params -> star object                              #
    #         verbose -> if True, allows print statements                #
    #                    default is False                                #
    #         debug -> allows debug statements to show                   #
    #                  default is set to False                           #
    # Outputs: theta_ld-> final limb-darkened disk diameter              #
    #          dtheta_ld -> error on the ld diameter                     #
    #          T -> effective temperature                                #
    #          dT -> error on the effective temperature                  #
    #          final_ldcs -> the final ldcs for each detected band       #
    #          final_chisqrs -> the final chi square and chi square      #
    #                           reduced values                           #
    # What it does:                                                      #
    #      1. Initializes a filter map dictionary relating each          #
    #         instrument to a filter                                     #
    #      2. sets the T_new to be the current temperature in the star   #
    #         object                                                     #
    #      3. sets an arbitrary number for the diff_teff and diff_theta  #
    #      4. sets the minimum percent difference                        #
    #      5. unpacks the ogdata for comparison later                    #
    #      6. Starts the while loop that compares the percent difference #
    #         between the theta and teff of the iteration before and the #
    #         theta and teff of the current iteration                    #
    #      7. Initializes the empty list for the diameters and a         #
    #         dynamic list for the ldcs per filter                       #
    #      8. For each data set, it calculates a ldc depending on the    #
    #         instrument                                                 #
    #      9. enters the Monte Carlo loop                                #
    #     10. Creates the dataframes for each dataset                    #
    #     11. For each dataframe, it samples the ldc on a normal         #
    #         distribution.
    #     12. For each dataframe, samples the wavelength of observation  #
    #         on a normal distribution. Then calculates new spatial      #
    #         frequencies                                                #
    #     13. Begins running all the bootstrapping loops in parallel     #
    #         by calling mcbs_worker                                     #
    #     14. Appends each result of the mcbs_worked to the LD list      #
    #     15. Resets the while loop iterators                            #
    #     16. Calcualtes a new LD theta, LD dtheta, teff and dteff by    #
    #         calling ldfit_values                                       #
    #     17. Updates the stellar object                                 #
    #     18. Calculates the new percent difference for theta and teff   #
    #     19. After the final iteration of the while loop,               #
    #         calls ldfit_values to do a final fit for the theta, theta  #
    #         error, teff, teff error, ldc_values, and chi-square vals   #
    #     20. Updates stellar object with the ldc values for each filter #
    #     21. Calculates final percent differences for teff and theta    #
    #     22. Returns final theta, theta err, teff, teff error, ldc_vals #
    #         and chi-sqr vals.                                          #
    ######################################################################
    filter_map_i = {
        'p': 'R',
        'v': 'R',
        'c': 'K',
        'm': 'H',
        'my': 'K',
        's': 'R'
        # Add other instruments as needed
    }
    T_new = stellar_params.teff
    theta_new = stellar_params.udtheta
    diff_theta = 5
    diff_teff = 5
    min_percent = 0.05
    iter = 0
    x = ogdata[0]
    y = ogdata[1]
    dy = ogdata[2]
    while diff_theta >= min_percent or diff_teff >= min_percent:
        LD = []
        V0 = []
        ldc_per_filter = {}
        for d in datasets:
            inst = d.instrument.lower()
            filt = filter_map_i[inst]
            if filt not in ldc_per_filter:
                ldc_val = ldc_calc(stellar_params.teff,
                                   stellar_params.logg,
                                   stellar_params.feh, filt)
                ldc_per_filter[filt] = ldc_val

        mc_args = []
        for _ in range(mc_num):
            mc_dfs = []
            for d in datasets:
                inst = d.instrument.lower()
                filt = filter_map_i[inst]
                mu = np.random.normal(ldc_per_filter[filt], 0.02)
                df = d.make_df(LDC=mu)
                df['Spf'] = df['B'] / np.random.normal(df['Wave'], df['Band'])
                mc_dfs.append(df)
            mc_args.append((mc_dfs, bs_num, stellar_params, v0_flag, verbose))

        # Parallel execute
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(mcbs_worker, mc_args))
            for res in results:
                LD.extend(res)
                V0.extend(res)

        T_old = T_new
        theta_old = theta_new
        theta_new, _,V0_new, _, T_new, _, _, _ = ldfit_values(x, y, dy, LD,V0, ldc_per_filter, stellar_params, v0_flag,verbose=debug)
        #print(V0_new)
        stellar_params.update(teff=round(T_new,5), ldtheta=round(theta_new,5))
        diff_teff = percent_diff(T_old, T_new, verbose=debug)
        diff_theta = percent_diff(theta_old, theta_new, verbose=debug)
        iter += 1
    if verbose:
        print("Final Values after ", iter, " iterations:")
    theta_ld, dtheta_ld, V0_ld, dV0_ld, T, dT, final_ldcs, final_chis = ldfit_values(x, y, dy, LD, V0, ldc_per_filter, stellar_params, v0_flag,
                                                                      verbose)
    stellar_params.update(teff=round(T,5), ldtheta=round(theta_ld,5), ldtheta_err=round(dtheta_ld,5), ldv0 = round(V0_ld, 5), ldv0_err = round(dV0_ld))
    for filt, mu in final_ldcs.items():
        setattr(stellar_params, f"ldc_{filt}", round(mu, 5))
    diff_teff = percent_diff(T_old, T_new, verbose)
    diff_theta = percent_diff(theta_old, theta_new, verbose)

    return theta_ld, dtheta_ld, V0_ld, dV0_ld, T, dT, final_ldcs, final_chis

