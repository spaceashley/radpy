import numpy as np
import scipy.special as ss

#Uniform disk V2 equation
def UDV2(sf, theta):
    x = np.pi*sf*(theta/(206265*1000))
    vis = (2*ss.jv(1,x))/x
    return vis**2
##########################################################################################
#Limb-darkened disk V2 equation
def V2(sf, theta, mu):
    #mu = 0.7039725
    alpha = 1-mu
    beta = mu
    x = np.pi*sf*(theta/(206265*1000))
    vis = (((alpha/2)+(beta/3))**(-2))*((alpha*(ss.jv(1,x)/x))+ beta*(np.sqrt(np.pi/2)*(ss.jv(3/2,x)/(x**(3/2)))))**2
    return vis


# calculates the sigma off of the observed values
def sigmacalc(expval, experr, obsval, obserr):
    upper = obsval + obserr
    lower = obsval - obserr
    sigma = 0
    if expval < lower:
        while expval < lower:
            expval = expval + experr
            # print(dumval)
            sigma += 1
    elif expval > upper:
        while expval > upper:
            expval = expval - experr
            sigma += 1

    return print("Value is", sigma, "sigma off")

#calculates chi squared and chi squared reduced
def chis(y, exp_y, yerr, dof):
    chisqr = np.sum(((y-exp_y)/(yerr))**2)
    chisqr_red  = chisqr/(len(y)-dof)
    return chisqr, chisqr_red
##########################################################################################
#Weighted average function
def weight_avg(df):
    x = df['V2']
    dx = df['sigma_V2']
    weight = 1/(dx**2)
    wavg = sum(weight*x)/sum(weight)
    dwavg = 1/np.sqrt(sum(weight))
    return dwavg


# Random bracket funciton for bootstrapping
def random_bracket(df, num_of_brackets):
    # 18 for PAVO, 4 for CLASSIC, 20 for VEGA, 42 in total
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

    results = grouped.apply(weight_avg)
    results = grouped.apply(weight_avg).reset_index()
    results.columns = ['Bracket', 'Wavg']
    random_groups_with_avg = random_groups.merge(results, on='Bracket')

    spf_br = random_groups_with_avg['Spf']  # spatial frequency in rad^-1
    v2_br = random_groups_with_avg['V2']  # Visibility squared
    dv2_br = random_groups_with_avg['sigma_V2']
    wavgs = random_groups_with_avg['Wavg']

    return spf_br, v2_br, dv2_br, wavgs


# Percent difference function
def percent_diff(x1, x2):
    diff = (abs(x1 - x2) / ((x1 + x2) / 2)) * 100
    print("Percent difference:", round(diff, 2), "%")


# Random bracket function for bootstrapping for limb-darkening
def random_bracket_ld(df, num_of_brackets):
    # 18 for PAVO, 4 for CLASSIC, 20 for VEGA, 42 in total
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

    results = grouped.apply(weight_avg)
    results = grouped.apply(weight_avg).reset_index()
    results.columns = ['Bracket', 'Wavg']
    random_groups_with_avg = random_groups.merge(results, on='Bracket')

    spf_br = random_groups_with_avg['Spf']  # spatial frequency in rad^-1
    v2_br = random_groups_with_avg['V2']  # Visibility squared
    dv2_br = random_groups_with_avg['sigma_V2']
    ldc_br = random_groups_with_avg['LDC']
    wavgs = random_groups_with_avg['Wavg']

    return spf_br, v2_br, dv2_br, ldc_br, wavgs


