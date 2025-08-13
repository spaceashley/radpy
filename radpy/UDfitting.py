import numpy as np
import pandas as pd
from lmfit import Model
import scipy.special as ss
from astropy.stats import mad_std
from radpy.stellar import temp
import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0 may give unexpected results.")
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns")
#Uniform disk V2 equation
def UDV2(sf, theta, V0):
    x = np.pi*sf*(theta/(206265*1000))
    if V0 == 1:
        vis2 = (((2*ss.jv(1,x))/x)**2)
        return vis2
    else:
        vis2 = (V0**2)*(((2*ss.jv(1,x))/x)**2)
        return vis2

##########################################################################################
# calculates chi squared and chi squared reduced
def chis(y, exp_y, yerr, dof):
    chisqr = np.sum(((y - exp_y) / (yerr)) ** 2)
    chisqr_red = chisqr / (len(y) - dof)
    return chisqr, chisqr_red


##########################################################################################
# Weighted average function
def weight_avg(df):
    x = df['V2']
    dx = df['dV2']
    weight = 1 / (dx ** 2)
    wavg = sum(weight * x) / sum(weight)
    dwavg = 1 / np.sqrt(sum(weight))
    return dwavg


##########################################################################################
# Random bracket function for bootstrapping
def random_bracket(df, num_of_brackets):
    ###########################################################################
    # Function: random_bracket                                                #
    # Inputs: df -> dataframe with spatial frequency, v2, v2_err, and bracket #
    #         num_of_brackets -> number of brackets in total you have         #
    # Outputs: spf_br -> the spatial frequencies randomized                   #
    #          v2_br -> the visibility squared randomized                     #
    #          dv2_br -> the error on the v2 randomized                       #
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
    #      9. Splits up the dataframe into spatial frequency, v2, dv2, and    #
    #         wavg                                                            #
    #     10. Returns spatial frequency, v2, dv2, and wavg                    #
    ###########################################################################
    np.random.seed()
    xdata = []
    ydata = []
    dydata = []
    numbr = np.random.randint(2, num_of_brackets)
    # chatgpt wrote the next couple lines
    random_group_ids = df['Bracket'].drop_duplicates().sample(n=numbr).values

    # Filter the DataFrame for the selected groups
    random_groups = df[df['Bracket'].isin(random_group_ids)]

    grouped = random_groups.groupby('Bracket')

    # results = grouped.apply(weight_avg, include_group = False).reset_index()
    results = grouped.apply(weight_avg).reset_index()
    # Version-aware apply for future compatibility
    # pandas_major = int(pd.__version__.split('.')[0])
    # pandas_minor = int(pd.__version__.split('.')[1])
    # if pandas_major >= 2:
    #    results = grouped.apply(weight_avg, include_group=False).reset_index()
    # else:
    #    results = grouped.apply(weight_avg).reset_index()

    results.columns = ['Bracket', 'Wavg']
    random_groups_with_avg = random_groups.merge(results, on='Bracket')

    spf_br = random_groups_with_avg['Spf']  # spatial frequency in rad^-1
    v2_br = random_groups_with_avg['V2']  # Visibility squared
    dv2_br = random_groups_with_avg['dV2']
    wavgs = random_groups_with_avg['Wavg']

    return spf_br, v2_br, dv2_br, wavgs


##########################################################################################
# Percent difference function
def percent_diff(x1, x2, verbose=False):
    diff = (abs(x1 - x2) / ((x1 + x2) / 2)) * 100
    if verbose:
        print("Percent difference:", round(diff, 2), "%")
    return round(diff, 3)

def safe_param_extraction(result):
    tparam = result.params['theta']
    vparam = result.params['V0']
    tvalue = tparam.value
    tstderr = tparam.stderr
    vvalue = vparam.value
    vstderr = vparam.stderr
    if tstderr is None or tstderr == 0 or tstderr is None or tstderr == 0:
        return tvalue, None, vvalue, None
    else:
        uvart = result.uvars['theta']
        uvarv = result.uvars['V0']
        return uvart.n, uvart.s, uvarv.n, uvarv.s
##########################################################################################
def initial_UDfit(spf, v2, dv2, theta_guess, V0_guess, star_params, v0_flag=False, verbose=False):
    #####################################################################
    # Function: initial_UDfit                                           #
    # Inputs: spf -> spatial frequency                                  #
    #         v2 -> visibilitity squared                                #
    #         dv2 -> error on the V2                                    #
    #         theta_guess -> initial guess for theta                    #
    #         star_params -> stellar class object                       #
    #         v0_flag -> set to fit with a scaling factor or not        #
    #                    True: fits with a scaling factor               #
    #         verbose -> if set to True, allows print statements        #
    #                    defaults to False                              #
    # Outputs: theta_ilm -> initial uniform disk diameter               #
    #          dtheta_ilm -> error on the diameter                      #
    #          chisqr_ilm -> chi squared reduced value                  #
    # What it does:                                                     #
    #        1. Initialized the model                                   #
    #        2. initializes the parameters                              #
    #        3. If the v0_flag is not set, will fix V0 to be 1.0        #
    #        4. Fits for the UD diameter using lmfit                    #
    #           uses for the weights as 1/dv2                           #
    #        5. pulls out the theta, dtheta, and chi squared reduced    #
    #        6. updates the stellar object                              #
    #        7. Returns the theta, dtheta, and chi squared reduced      #
    #####################################################################

    udmodel = Model(UDV2)
    udparams = udmodel.make_params(theta=theta_guess, V0=V0_guess)
    if not v0_flag:
        # Fix V0 to 1.0 (or V0_guess, but default should be 1.0)
        udparams['V0'].set(value=1.0, vary=False)
    ud_result = udmodel.fit(v2, udparams, sf=spf, weights=1 / (dv2), scale_covar=False)
    # Extract parameters
    if v0_flag:
        theta_ilm, dtheta_ilm, v0_ilm, dv0_ilm = safe_param_extraction(ud_result)
    else:
        theta_ilm, dtheta_ilm, v0_ilm, dv0_ilm = safe_param_extraction(
            ud_result)  # extract, but v0_ilm will be 1.0, dv0_ilm may be 0
    chisqr_ilm = ud_result.redchi
    if verbose:
        print('Initial fit with lmfit:')
        print(ud_result.fit_report())
    star_params.update(
        udthetai=round(theta_ilm, 5),
        udthetai_err=round(dtheta_ilm, 5),
        udv0i=round(v0_ilm, 5),
        udv0i_err=round(dv0_ilm, 5)
    )
    return theta_ilm, dtheta_ilm, chisqr_ilm, v0_ilm, dv0_ilm


def make_df(B, v2, dv2, wave, band, brack, inst, fulldf=False):
    rand_wave = np.random.normal(wave, band)
    new_spf = B / rand_wave
    if not fulldf:
        return new_spf
    if fulldf:
        new_df = pd.DataFrame(np.column_stack((new_spf, v2, dv2, brack, inst)),
                              columns=['Spf', 'V2', 'dV2', 'Bracket', 'Instrument'])
        return new_df

def bootstrap(df, inst):
    ###########################################################
    # Function: bootstrap                                     #
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
        new_df = pd.DataFrame(np.column_stack((df['Spf'], newv2, df['dV2'])), columns=['Spf', 'V2', 'dV2'])
        return new_df
    else:
        num_brackets = df['Bracket'].max()
        spfbr, v2br, dv2br, avgdv2 = random_bracket(df, num_brackets)
        newv2 = np.random.normal(v2br, avgdv2)
        new_df = pd.DataFrame(np.column_stack((spfbr, newv2, dv2br)), columns=['Spf', 'V2', 'dV2'])
        return new_df


def udfit(df, stellar_params, v0_flag = False, verbose=False):
    #####################################################################
    # Function: udfit                                                   #
    # Inputs: df -> dataframe with data in it                           #
    #         star_params -> stellar class object                       #
    #         verbose -> if set to True, allows print statements        #
    #                    defaults to False                              #
    # Outputs: theta_ud -> initial uniform disk diameter                #
    # What it does:                                                     #
    #        1. Initialized the model                                   #
    #        2. initializes the parameters                              #
    #        3. Fits for the UD diameter using lmfit                    #
    #           uses for the weights as 1/dv2                           #
    #        4. pulls out the theta                                     #
    #        5. Returns the theta                                       #
    #####################################################################
    udmodel = Model(UDV2)
    ud_params = udmodel.make_params(theta=stellar_params.udthetai, V0 = stellar_params.udv0i)
    if not v0_flag:
        # Fix V0 to 1.0 (or V0_guess, but default should be 1.0)
        ud_params['V0'].set(value=1.0, vary=False)
    ud_result = udmodel.fit(df['V2'], ud_params, sf=df['Spf'], weights=1 / (df['dV2']), scale_covar=True)

    #theta_ud = ud_result.uvars['theta'].n
    if v0_flag:
        theta_ud = ud_result.uvars['theta'].n
        v0_ud = ud_result.uvars['V0'].n
        return theta_ud, v0_ud
    if not v0_flag:
        theta_ud = ud_result.uvars['theta'].n
        return theta_ud

def run_UDfit(mc_num, bs_num, datasets, stellar_params, v0_flag = False, verbose=False):
    ######################################################################
    # Function: run_udmcbs_fit                                           #
    # Inputs: mc_num -> number of Monte Carlo iterations                 #
    #         bs_num -> number of bootstrap iterations                   #
    #         datasets -> the datasets you want fit                      #
    #                     format: [inst1, inst2, inst3]                  #
    #         stellar_params -> star object                              #
    #         verbose -> if True, allows print statements                #
    #                    default is False                                #
    # Outputs: UD -> a list of all the uniform disk diameters calculated #
    #                during the fitting routine                          #
    # What it does:                                                      #
    #      1. Initializes the empty list for the diameters               #
    #      2. If verbose is set, it initializes the empty lists for      #
    #         spf, v2, and dv2                                           #
    #      3. enters the Monte Carlo loop                                #
    #      4. Creates the dataframes for each dataset                    #
    #      5. For each dataframe, samples the wavelength of observation  #
    #         on a normal distribution. Then calculates new spatial      #
    #         frequencies                                                #
    #      6. Enters the bootstrapping loop                              #
    #      7. Pulls the instrument for each data set                     #
    #      8. Calls the bootstrap function with the instrument's df and  #
    #         the instrument ID                                          #
    #      9. appends the output of the bootstrap function to a list.    #
    #     10. Concatenates all the dataframes created in the bootstrap   #
    #         loop.                                                      #
    #     11. Calculates the uniform disk diameter with udfit            #
    #     12. Appends the diameter to the list called UD.                #
    #     13. After the loops, returns the UD.                           #
    ######################################################################
    UD = []
    V0 = []
    if verbose:
        udmcbs_spf = []
        udmcbs_v2 = []
        udmcbs_dv2 = []

    for _ in range(mc_num):
        dfs = [d.make_df() for d in datasets]

        for df in dfs:
            df['Spf'] = df['B'] / np.random.normal(df['Wave'], df['Band'])

        for _ in range(bs_num):
            bs_dfs = []

            for df in dfs:
                inst = df["Instrument"].iloc[0]
                # print('Instrument:', inst)
                boot_df = bootstrap(df, inst)
                bs_dfs.append(boot_df)

            new_df = pd.concat(bs_dfs, ignore_index=True)

            if v0_flag:
                theta_udbs, v0_udbs = udfit(new_df, stellar_params, v0_flag, verbose)
                UD.append(theta_udbs)
                V0.append(v0_udbs)

                if verbose:
                    udmcbs_spf.append(new_df['Spf'])
                    udmcbs_v2.append(new_df['V2'])
                    udmcbs_dv2.append(new_df['dV2'])
                    return UD, V0, udmcbs_spf, udmcbs_v2, udmcbs_dv2
                return UD, V0

            if not v0_flag:
                theta_udbs = udfit(new_df, stellar_params, v0_flag, verbose)
                UD.append(theta_udbs)

                if verbose:
                    udmcbs_spf.append(new_df['Spf'])
                    udmcbs_v2.append(new_df['V2'])
                    udmcbs_dv2.append(new_df['dV2'])
                    return UD, udmcbs_spf, udmcbs_v2, udmcbs_dv2
                return UD


def udfit_values(x, y, dy, UD, stellar_params, v0_flag = False, verbose=False):
    ################################################################
    # Function: udfit_values                                       #
    # Inputs: x -> the spatial frequencies                         #
    #         y -> the V2                                          #
    #        dy -> the error on the V2                             #
    #        UD -> the list of diameters                           #
    #        stellar_params -> the star object                     #
    #        verbose - > if true, returns print statements         #
    # Outputs: None                                                #
    # What it does:                                                #
    #     1. Takes the mean of the uniform disk diameter list      #
    #     2. Takes the median absolute deviation of the UDs        #
    #     3. Calculates the chi squared and chi squared reduced    #
    #        of the uniform disk diameter fit.                     #
    #     4. Calculates the effective temperature using the mean   #
    #     5. Updates the stellar object with the new parameters    #
    ################################################################
    avg_UD = np.mean(UD)
    std_UD = mad_std(UD)
    if v0_flag:
        avg_V0 = np.mean(V0)
        std_V0 = mad_std(V0)
        chisq, chisqr = chis(y, UDV2(x, avg_UD, avg_V0), y, 2)
        stellar_params.update(teff=round(teff_ud[0], 5), teff_err=round(teff_ud[1], 5), udtheta=round(avg_UD, 5),
                              udtheta_err=round(std_UD, 5), udv0=round(avg_V0, 5), udv0_err=round(std_V0, 5))
    if not v0_flag:
        chisq, chisqr = chis(y, UDV2(x, avg_UD, 1), y, 1)

        stellar_params.update(teff=round(teff_ud[0], 5), teff_err=round(teff_ud[1], 5), udtheta=round(avg_UD, 5),
                              udtheta_err=round(std_UD, 5))

    teff_ud = temp(stellar_params.fbol, stellar_params.fbol_err, avg_UD, std_UD)

    if verbose:
        print('Uniform Disk Diameter after MC/BS:', round(avg_UD, 4), '+/-', round(std_UD, 5), 'mas')
        if v0_flag:
            print('V0**2:', round(avg_V0, 4), '+/-', round(std_V0, 4))
        print("Chi-squared:", round(chisq, 3))
        print("Chi-squared reduced:", round(chisqr, 3))
        print("Temperature:", round(teff_ud[0], 1), "+/-", round(teff_ud[1], 1), "K")

