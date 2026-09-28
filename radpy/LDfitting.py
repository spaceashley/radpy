import numpy as np
import pandas as pd
from lmfit import Model
import concurrent.futures
import scipy.special as ss
from radpy.stellar import temp
from astropy.stats import mad_std
from lmfit import Parameters, Minimizer
from radpy.UDfitting import chis, weight_avg, percent_diff, safe_theta_extraction, safe_thetaV0_extraction
from radpy.limbdarkcoeffs import ldc_calc

import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.")
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns")
#Limb-darkened disk V2 equation
def V2(sf, theta, mu):
    sf = np.asarray(sf, dtype = float)
    theta = float(theta)
    my = np.asarray(mu, dtype = float)
    alpha = 1-mu
    beta = mu
    x = np.pi*sf*(theta/(206265*1000))
    vis = (((alpha/2)+(beta/3))**(-2))*((alpha*(ss.jv(1,x)/x))+ beta*(np.sqrt(np.pi/2)*(ss.jv(3/2,x)/(x**(3/2)))))**2
    return vis
###########################################################################################
def scaledV2(sf, theta, mu, V0):
    alpha = 1-mu
    beta = mu
    x = np.pi*sf*(theta/(206265*1000))
    vis = (V0**2)*((((alpha/2)+(beta/3))**(-2))*((alpha*(ss.jv(1,x)/x))+ beta*(np.sqrt(np.pi/2)*(ss.jv(3/2,x)/(x**(3/2)))))**2)
    return vis
##########################################################################################
def multi_v0_residual(params, df):
    #################################################################
    # Function: multi_v0_residual                                   #
    # Inputs: params -> fitting parameters                          #
    #         df -> data frame of data being fit                    #
    # Outputs: residual                                             #
    # What it does:                                                 #
    #      1. Assigns the theta value                               #
    #      2. Extracts out each V0 value and maps it                #
    #      3. Checks to make sure that there is no nans, infs, etc  #
    #      4. Generates the base V2 model                           #
    #      5. Multiplies base model by the V0^2 values according to #
    #         night and instrument.                                 #
    #      6. Calculates and returns the residual.                  #
    #################################################################
    theta = params["theta"].value

    v0_by_group = {
        name.removeprefix("V0_"): parameter.value
        for name, parameter in params.items()
        if name.startswith("V0_")
    }

    v0_values = df["V0_group"].map(v0_by_group)

    if v0_values.isna().any():
        missing_groups = (df.loc[v0_values.isna(), "V0_group"].drop_duplicates().tolist())
        raise ValueError(f"No V0 parameter exists for groups: {missing_groups}")

    base_model = V2(df["Spf"].to_numpy(), theta, df["LDC"].to_numpy(), )
    model = base_model * v0_values.to_numpy() ** 2
    residual = (df["V2"].to_numpy() - model) / df["dV2"].to_numpy()
    return residual


def fit_ld_with_v0_groups(df, stellar_params):
    """
    Fit one shared theta and one V0 for every unique V0_group.
    """

    ##################################################################
    # Function: fit_ld_with_v0_groups                                #
    # Inputs: df -> data frame of data being fit                     #
    #         stellar_params -> StellarParams() object               #
    # Outputs: theta -> fitted angular diameter                      #
    #          dtheta -> error for angular diameter                  #
    #          v0_by_group -> group of V0 values                     #
    #          dv0_by_group -> group of the errors for the V0 values #
    #          result.redchi -> chi-squared value for the fit        #
    # What it does:                                                  #
    #      1. Determines the V0 value groups for the datasets        #
    #      2. Makes and adds the Parameters for lmfit                #
    #      3. Creates the Minimizer object to perform the fit and    #
    #         performs the fit                                       #
    #      5. Extracts the results and separates the V0 values by    #
    #         group.                                                 #
    #      6. Returns the fitted parameters and the chi squared      #
    ##################################################################
    groups = sorted(df["V0_group"].astype(str).drop_duplicates().tolist())
    params = Parameters()
    params.add("theta", value=stellar_params.udtheta, min=0.0001, max=100, )

    for group in groups:
        params.add(f"V0_{group}", value=1.0, min=0.0, max=2.0, )

    minner = Minimizer(multi_v0_residual, params, fcn_args=(df,), )
    result = minner.minimize()

    theta = result.params["theta"].value
    dtheta = result.params["theta"].stderr

    if dtheta is None:
        dtheta = np.nan

    v0_by_group = {}
    dv0_by_group = {}

    for group in groups:
        parameter = result.params[f"V0_{group}"]
        v0_by_group[group] = parameter.value
        dv0_by_group[group] = (parameter.stderr
                               if parameter.stderr is not None
                               else np.nan)

    return (theta, dtheta, v0_by_group, dv0_by_group, result.redchi,)


def make_v0_group(df):
    ###################################################################
    # Function: make_v0_group                                         #
    # Inputs: df -> dataframe containing the data for all instruments #
    # Outputs: group_list -> list with the instrument and night       #
    #                      assignments                                #
    # What it does:                                                   #
    #        1. Creates an empty list                                 #
    #        2. Loops through the Instrument column and assigns       #
    #           the correct instrument with the correct night         #
    #        3. Returns the list.                                     #
    ###################################################################

    group_list = []
    for i in range(len(df['Instrument'])):
        instrument = df['Instrument'][i]
        night = df['Night'][i]
        group = f"{str(instrument).upper()}_{str(int(night))}"

        group_list.append(group)

    return group_list


def aggregate_v0_results(v0_results):
    """
    Convert a list of V0 dictionaries into per-group statistics.
    """
    ##################################################################
    # Function: aggregate_v0_results                                 #
    # Inputs: v0_results -> list of the fitted v0 values in the MCMC #
    # Outputs: avg_v0 -> average V0 value for each instrument and    #
    #                  night                                         #
    #          std_v0 -> standard deviation for each instrument and  #
    #                  night                                         #
    # What it does:                                                  #
    #      1. Sorts the v0_values by instrument/night                #
    #      2. Calculates the average and std of each group.          #
    #      3. Returns the results.                                   #
    ##################################################################
    groups = sorted({
        group
        for result in v0_results
        for group in result
    })

    avg_v0 = {}
    std_v0 = {}

    for group in groups:
        values = np.array([
            result[group]
            for result in v0_results
            if group in result
        ])

        avg_v0[group] = np.mean(values)
        std_v0[group] = mad_std(values)

    return avg_v0, std_v0


def assign_v0_value(og_df, v0s):
    ################################################################
    # Function: assign_v0_value                                    #
    # Inputs: og_df -> original data being fitted                  #
    #         v0s -> list of the V0 values calculated in the MCMC  #
    # Outputs: og_df -> original data being fitted with the V0     #
    #                 values added to it                           #
    # What it does:                                                #
    #      1. Calls aggregate_v0_results and calculates the        #
    #         average and standard deviation of the V0 values      #
    #      2. Adds the averaged value to the original data frame   #
    #         according to the Instrument and Night                #
    #      3. Returns the new data frame                           #
    ################################################################
    v0_results = aggregate_v0_results(v0s)
    num_nights = len(v0_results[0])
    for i in range(num_nights):
        for ii in range(len(og_df['Night'])):
            if int(og_df['Night'][ii]) == i + 1:
                og_df.loc[ii, 'V0'] = list(v0_results[0].values())[i]

    return og_df

# Random bracket function for bootstrapping for limb-darkening
def random_bracket_ld(df, num_of_brackets):
    ###########################################################################
    # Function: random_bracket                                                #
    # Inputs: df -> dataframe with spatial frequency, v2, v2_err, and bracket #
    #         num_of_brackets -> number of brackets in total you have         #
    # Outputs: spf_br -> the spatial frequencies randomized                   #
    #          v2_br -> the visibility squared randomized                     #
    #          dv2_br -> the error on the v2 randomized                       #
    #          ldc_br -> the limb-darkening coefficients                      #
    #          wavgs -> weighted averages of the v2                           #
    #          nights_br -> the V0_group assigned to each value               #
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
    #     10. Returns spatial frequency, v2, dv2, wavg, and nights_br         #
    ###########################################################################
    np.random.seed()
    xdata = []
    ydata = []
    dydata = []
    ldcdata = []
    nightdata = []
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
    nights_br = random_groups_with_avg['V0_group']

    return spf_br, v2_br, dv2_br, ldc_br, wavgs, nights_br
##########################################################################################
def initial_LDfit(spf, v2, dv2, star_params, filt, ldc_method, v0_flag = False, verbose=False, debug = False):
    #####################################################################
    # Function: initial_LDfit                                           #
    # Inputs: spf -> spatial frequency                                  #
    #         v2 -> visibilitity squared                                #
    #         dv2 -> error on the V2                                    #
    #         theta_guess -> initial guess for theta                    #
    #         star_params -> stellar class object                       #
    #         v0_flag -> if set to true, allows fitting for a scaling   #
    #                    factor as well                                 #
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
    ldc = ldc_calc(t, star_params.logg, star_params.feh, filt, ldc_method)
    if not np.isfinite(ldc):
        prev = ldc
        if np.isfinite(prev):
            ldc = prev
        else:
            ldc = 0.20 if filt == 'H' else 0.16 if filt == 'K' else 0.30
        if debug:
            print(f"[WARN] ldc_calc returned {ldc_val} fallback for filter {filt}")

    if not v0_flag:
        #print("No Scaling used")
        ldmodel = Model(V2, independent_vars=['sf', 'mu'])
        ldparams = ldmodel.make_params(theta=star_params.udthetai)
        ld_result = ldmodel.fit(v2, ldparams, sf=spf, mu=ldc, weights= 1 / (dv2), scale_covar=False)
        ldtheta_ilm, lddtheta_ilm = safe_theta_extraction(ld_result)
        chisqr_ldilm = ld_result.redchi  # chi squared reduced of the fit

        star_params.update(ldthetai=round(ldtheta_ilm,5), ldthetai_err = round(lddtheta_ilm,5), teff=round(t,5), teff_err=round(dt,5))
        if verbose:
            print("Effective temperature:", round(t,5), "+/-", round(dt,5), "K")
            print("LDC for filter ", filt, ":", round(ldc,5))
            print('Initial fit with lmfit:')
            print(ld_result.fit_report())

        return (ldtheta_ilm, lddtheta_ilm, chisqr_ldilm)

    if v0_flag:
        #print("Scaling used")
        ldmodel = Model(scaledV2, independent_vars=['sf', 'mu'])
        ldparams = ldmodel.make_params(theta=star_params.udthetai, V0 = 1.0)
        ld_result = ldmodel.fit(v2, ldparams, sf=spf, mu=ldc, weights= 1 / (dv2), scale_covar=False)
        ldtheta_ilm, lddtheta_ilm, ldv0_ilm, lddv0_ilm = safe_thetaV0_extraction(ld_result)
        chisqr_ldilm = ld_result.redchi  # chi squared reduced of the fit
        star_params.update(ldthetai=round(ldtheta_ilm,5), ldthetai_err = round(lddtheta_ilm,5), ldv0i = round(ldv0_ilm, 5), ldv0i_err = round(lddv0_ilm, 5),
                           teff=round(t,5), teff_err=round(dt,5))
        if verbose:
            print("Effective temperature:", round(t,5), "+/-", round(dt,5), "K")
            print("LDC for filter ", filt, ":", round(ldc,5))
            print('Initial fit with lmfit:')
            print(ld_result.fit_report())

        return (ldtheta_ilm, lddtheta_ilm, ldv0_ilm, lddv0_ilm, chisqr_ldilm)

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
        new_df = pd.DataFrame(np.column_stack((df['Spf'], newv2, df['dV2'], df['LDC'], df['V0_group'])),
                              columns=['Spf', 'V2', 'dV2', 'LDC', 'V0_group'])
        return new_df
    else:
        num_brackets = df['Bracket'].max()
        spfbr, v2br, dv2br, ldcbr, avgdv2, nightsbr = random_bracket_ld(df, num_brackets)
        newv2 = np.random.normal(v2br, avgdv2)
        new_df = pd.DataFrame(np.column_stack((spfbr, newv2, dv2br, ldcbr, nightsbr)), columns=['Spf', 'V2', 'dV2', 'LDC', 'V0_group'])
        return new_df


def ldfit(df, stellar_params, v0_flag = False, verbose=False):
    #####################################################################
    # Function: ldfit                                                   #
    # Inputs: df -> dataframe with data in it                           #
    #         star_params -> stellar class object                       #
    #         v0_flag - > allows you to fit for a scaling factor if True#
    #         verbose -> if set to True, allows print statements        #
    #                    defaults to False                              #
    # Outputs: theta_ld -> initial uniform disk diameter                #
    # What it does:                                                     #
    #        1. Initialized the model                                   #
    #        2. initializes the parameters                              #
    #        3. Fits for the LD diameter using lmfit                    #
    #           uses for the weights as 1/dv2                           #
    #        4. If v0_flag is True: calls fit_ld_with_v0_groups         #
    #        5. pulls out the theta                                     #
    #        6. Returns the theta                                       #
    #           If v0_flag is True, will pull V0s                       #
    #####################################################################
    if not v0_flag:
        #print("No scaling use")
        ldmodel = Model(V2, independent_vars=['sf', 'mu'])
        ld_params = ldmodel.make_params(theta=stellar_params.udtheta)
        ld_params['theta'].set(min=0.0001, max = 100)
        ld_result = ldmodel.fit(df['V2'], ld_params, sf=df['Spf'], mu=df['LDC'], weights=1 / (df['dV2']), scale_covar=True)
        theta_ld, _ = safe_theta_extraction(ld_result)
        #theta_ld = ld_result.uvars['theta'].n
        return (theta_ld)
    if v0_flag:
        #print("Scaling used")
        result = fit_ld_with_v0_groups(df, stellar_params)
        theta_ld = result[0]
        v0s = result[2]
        return (theta_ld, v0s)


def ldfit_values(x, y, dy, inst, nights, mc_results, ldcs, stellar_params, v0_flag=False, verbose=False):
    ##################################################################
    # Function: ldfit_values                                         #
    # Inputs: x -> the spatial frequencies                           #
    #         y -> the V2                                            #
    #        dy -> the error on the V2                               #
    #        inst -> instrument                                      #
    #        nights -> which night for V0 scaling                    #
    #        LD -> the list of diameters                             #
    #        ldcs -> limb darkening coefficients                     #
    #        stellar_params -> the star object                       #
    #        v0_flag - > if true, fits for the scaling factor        #
    #        verbose - > if true, returns print statements           #
    # Outputs: avg_LD -> average limb darkened diameter              #
    #          std_LD -> the median absolute deviation of LD theta   #
    #          avg_V0 -> the average V0^2 values if flag is set      #
    #          std_V0 -> the standard deviation of the V0^2 values   #
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
    #        If v0_Flag is set, will also return avg_V0 and std_V0   #
    ##################################################################
    if not v0_flag:
        LD = mc_results
        avg_LD = np.mean(LD)
        std_LD = mad_std(LD)
        teff_ld = temp(stellar_params.fbol, stellar_params.fbol_err, avg_LD, std_LD)
        # Store results dynamically
        ldc_results = {}
        chisq_results = {}

        for band in ldcs:
            ldc_val = ldcs[band]
            if ldc_val is not None:
                model_v2 = V2(x, avg_LD, ldc_val)
                chisq, chisqr = chis(y, model_v2, dy, 1)
                ldc_results[band] = ldc_val
                chisq_results[band] = {"chisq": chisq, "chisqr": chisqr}

        if verbose:
            print('Limb-darkened Disk Diameter after MC/BS:', round(avg_LD, 4), '+/-', round(std_LD, 5), 'mas')
            for band, ldc_val in ldc_results.items():
                print(f"Limb-darkening coefficient in {band}:", round(ldc_val, 5))
                print(f"Chi-squared for {band} band:", round(chisq_results[band]["chisq"], 3))
                print(f"Reduced chi-squared for {band} band:", round(chisq_results[band]["chisqr"], 3))
            print("Temperature:", round(teff_ld[0], 1), "+/-", round(teff_ld[1], 1), "K")

        return avg_LD, std_LD, teff_ld[0], teff_ld[1], ldc_results, chisq_results

    if v0_flag:
        LD = mc_results[0]
        V0_results = mc_results[1]
        avg_LD = np.mean(LD)
        std_LD = mad_std(LD)
        avg_V0, std_V0 = aggregate_v0_results(V0_results)
        # std_V0 = mad_std(V0)

        teff_ld = temp(stellar_params.fbol, stellar_params.fbol_err, avg_LD, std_LD)
        # Store results dynamically
        ldc_results = {}
        chisq_results = {}
        og_df = pd.DataFrame({"Spf": x, "V2": y, "dV2": dy, "Inst": inst, "Night": nights})
        ogdf_v0s = assign_v0_value(og_df, V0_results)

        for band in ldcs:
            ldc_val = ldcs[band]
            if ldc_val is not None:
                model_v2 = ((ogdf_v0s['V0']).to_numpy() ** 2) * V2(x, avg_LD, ldc_val)
                # chisq, chisqr = chis(y, model_v2, dy, 2)
                ldc_results[band] = ldc_val
                number_of_params = 1 + len(avg_V0)
                chisq, chisqr = chis(y, model_v2, dy, number_of_params)
                chisq_results[band] = {"chisq": chisq, "chisqr": chisqr}

        if verbose:
            print('Limb-darkened Disk Diameter after MC/BS:', round(avg_LD, 4), '+/-', round(std_LD, 5), 'mas')
            for group in sorted(avg_V0):
                v0 = avg_V0[group]
                dv0 = std_V0[group]

                print(f"  {group}: V0 = {v0:.5f} +/- {dv0:.5f}; "f"V0^2 = {v0 ** 2:.5f}")
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
    #                 v0_flag, and verbose                      #
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
    #          If v0_flag = True, will append results to the V0 #
    #          list                                             #
    #       9. Returns the LD list (and the V0 list if True)    #
    #############################################################

    mc_dfs, bs_num, stellar_params, v0_flag, verbose = args
    LD = []
    V0 = []
    for _ in range(bs_num):
        bs_dfs = []
        for df in mc_dfs:
            inst = df["Instrument"].iloc[0]
            boot_df = bootstrap_ld(df, inst)
            bs_dfs.append(boot_df)
        new_df = pd.concat(bs_dfs, ignore_index=True)
        if not v0_flag:
            #print("No scaling")
            theta_ldbs = ldfit(new_df, stellar_params,v0_flag, verbose)
            LD.append(theta_ldbs)
        if v0_flag:
            #print("Scaling")
            theta_ldbs, v0_ldbs = ldfit(new_df, stellar_params, v0_flag, verbose)
            LD.append(theta_ldbs)
            V0.append(v0_ldbs)
    if not v0_flag:
        #print("no scaling again")
        return (LD)
    if v0_flag:
        #print("Scaling again")
        return (LD, V0)


def run_LDfit(mc_num, bs_num, ogdata, datasets, stellar_params, ldc_method, v0_flag=False, verbose=False, debug=False):
    ######################################################################
    # Function: run_ldmcbs_fit_parallel                                  #
    # Inputs: mc_num -> number of Monte Carlo iterations                 #
    #         bs_num -> number of bootstrap iterations                   #
    #         ogdata -> original data sets                               #
    #         datasets -> the datasets you want fit                      #
    #                     format: [inst1, inst2, inst3]                  #
    #         stellar_params -> star object                              #
    #         ldc_method -> which method for LDC calculation             #
    #         v0_flag -> allows you to fit for a scaling factor, V0^2    #
    #                    Default is False                                #
    #         verbose -> if True, allows print statements                #
    #                    default is False                                #
    #         debug -> allows debug statements to show                   #
    #                  default is set to False                           #
    # Outputs: theta_ld-> final limb-darkened disk diameter              #
    #          dtheta_ld -> error on the ld diameter                     #
    #          v0^2 and error - > if v0_Flag is set to true              #
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
    #         calling ldfit_values (if v0_Flag is true, will also calc   #
    #         the v0^2 value)                                            #
    #     17. Updates the stellar object                                 #
    #     18. Calculates the new percent difference for theta and teff   #
    #     19. After the final iteration of the while loop,               #
    #         calls ldfit_values to do a final fit for the theta, theta  #
    #         error, teff, teff error, ldc_values, and chi-square vals   #
    #         If v0_flag = True, will also calcualte the final val for   #
    #         v0^2 and its error                                         #
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
    inst = ogdata[3]
    nights = ogdata[4]
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
                                   stellar_params.feh, filt, ldc_method)
                if not np.isfinite(ldc_val):
                    prev = ldc_per_filter.get(filt, np.nan)
                    if np.isfinite(prev):
                        ldc_val = prev
                    else:
                        ldc_val = 0.20 if filt == 'H' else 0.16 if filt == 'K' else 0.30
                    if debug:
                        print(f"[WARN] ldc_calc returned {ldc_val} fallback for filter {filt}"
                              f"(teff={stellar_params.teff}, logg={stellar_params.logg}, feh={stellar_params.feh})")
                ldc_per_filter[filt] = float(ldc_val)

        mc_args = []
        for _ in range(mc_num):
            mc_dfs = []
            for d in datasets:
                inst = d.instrument.lower()
                filt = filter_map_i[inst]
                mu = np.random.normal(ldc_per_filter[filt], 0.02)
                df = d.make_df(LDC=mu)
                df['V0_group'] = make_v0_group(df)
                df['Spf'] = df['B'] / np.random.normal(df['Wave'], df['Band'])
                mc_dfs.append(df)
            mc_args.append((mc_dfs, bs_num, stellar_params, v0_flag, verbose))

        # Parallel execute
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(mcbs_worker, mc_args))
            if not v0_flag:
                # print("No scaling")
                for res in results:
                    # print(len(res))
                    LD.extend(res)
            if v0_flag:
                # print("Scaling")
                for res in results:
                    # print(len(res))
                    LD.extend(res[0])
                    V0.extend(res[1])

        T_old = T_new
        theta_old = theta_new
        if not v0_flag:
            theta_new, _, T_new, _, _, _ = ldfit_values(x, y, dy, inst, nights, LD, ldc_per_filter, stellar_params,
                                                        v0_flag, verbose=debug)
            stellar_params.update(teff=round(T_new, 5), ldtheta=round(theta_new, 5))
            diff_teff = percent_diff(T_old, T_new, verbose=debug)
            diff_theta = percent_diff(theta_old, theta_new, verbose=debug)
            iter += 1
        if v0_flag:
            theta_new, _, V0_new, _, T_new, _, _, _ = ldfit_values(x, y, dy, inst, nights, [LD, V0], ldc_per_filter,
                                                                   stellar_params, v0_flag, verbose=debug)
            stellar_params.update(teff=round(T_new, 5), ldtheta=round(theta_new, 5))
            diff_teff = percent_diff(T_old, T_new, verbose=debug)
            diff_theta = percent_diff(theta_old, theta_new, verbose=debug)
            iter += 1
    if verbose:
        print("Final Values after ", iter, " iterations:")
    if not v0_flag:
        theta_ld, dtheta_ld, T, dT, final_ldcs, final_chis = ldfit_values(x, y, dy, inst, nights, LD, ldc_per_filter,
                                                                          stellar_params, v0_flag,
                                                                          verbose)
        stellar_params.update(teff=round(T, 5), ldtheta=round(theta_ld, 5), ldtheta_err=round(dtheta_ld, 5))
        for filt, mu in final_ldcs.items():
            setattr(stellar_params, f"ldc_{filt}", round(mu, 5))
        diff_teff = percent_diff(T_old, T_new, verbose)
        diff_theta = percent_diff(theta_old, theta_new, verbose)

        return (theta_ld, dtheta_ld, T, dT, final_ldcs, final_chis)
    if v0_flag:
        theta_ld, dtheta_ld, v0_ld, dv0_ld, T, dT, final_ldcs, final_chis = ldfit_values(x, y, dy, inst, nights,
                                                                                         [LD, V0], ldc_per_filter,
                                                                                         stellar_params, v0_flag,
                                                                                         verbose)

        v0_squared = {group: value ** 2 for group, value in v0_ld.items()}
        dv0_squared = {group: value ** 2 for group, value in dv0_ld.items()}

        stellar_params.update(teff=round(T, 5), ldtheta=round(theta_ld, 5), ldtheta_err=round(dtheta_ld, 5),
                              ldv02_by_group=v0_squared, ldv02_err_by_group=dv0_squared)
        for filt, mu in final_ldcs.items():
            setattr(stellar_params, f"ldc_{filt}", round(mu, 5))
        diff_teff = percent_diff(T_old, T_new, verbose)
        diff_theta = percent_diff(theta_old, theta_new, verbose)

        return (theta_ld, dtheta_ld, v0_squared, dv0_squared, T, dT, final_ldcs, final_chis)