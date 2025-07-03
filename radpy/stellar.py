import numpy as np
from zero_point import zpt
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


# function to calculate temperature from bolometric flux and angular diameter
def temp(Fbol, dF, theta, dtheta, verbose=False):
    T = 2341 * (Fbol / theta ** 2) ** (1 / 4)
    dT = np.sqrt(
        (2341 / (4 * Fbol ** (3 / 4) * theta ** (1 / 2)) * dF) ** 2
        + (2341 * Fbol ** (1 / 4) / (2 * theta ** (3 / 2)) * dtheta) ** 2
    )
    # dT = T*np.sqrt(((dF/Fbol)*(1/4))**2 + ((dtheta/theta)*(1/2))**2)
    if verbose:
        print("Temperature: ", T, "+/-", dT, "[K]")
    return T, dT


##########################################################################################
# calculates the distance given parallax
def dist_calc(p, p_err, zpc, verbose=False):
    pc = (p + zpc) / 1000
    D = 1 / pc
    dD = D * ((p_err / 1000) / pc)
    if verbose:
        print("Corrected parallax:", round(pc,5) * 1000)
        print("Distance:", round(D,5), "+/-", round(dD,5), "[pc]")

    return D, dD


# Luminosity function
def luminosity(D, dD, Fbol, dF):
    pc_conv = 3.091e16  # meters in a parsec
    d = D * pc_conv * 100
    L = (4 * np.pi * (d ** 2) * Fbol) / (3.846e33)
    dL = L * np.sqrt(((2 * dD) / D) ** 2 + (dF / Fbol) ** 2)
    # return(print("Luminosity: ", L ,"+/-",dL, "[L_solar]"))
    return L, dL


##########################################################################################
# Physical radius function
def ang_to_lin(theta, dtheta, D, dD, verbose=False):
    Rs = 6.957e8  # meters in a solar radii
    pc_conv = 3.091e16  # meters in a parsec
    mas_conv = 206265 * 1000  # radians to mas conversion

    R = (theta * (D * pc_conv)) / (2 * mas_conv * Rs)
    # dR = R*np.sqrt((dtheta/theta)**2 + (dD/D)**2)                         #my equation same as Jonas just different derivation
    dR = (pc_conv / (2 * mas_conv * Rs)) * np.sqrt(
        (dtheta * D) ** 2 + (dD * theta) ** 2)  # jonas equation matches the first one
    if verbose:
        print("Linear Radius: ", R, "+/-", dR, "[R_solar]")
    return R, dR


def gaia_correct_distance(gaiadrname, plx=None, dplx=None, verbose=False):
    #####################################################################
    # Function name: gaia_correct_distance                              #
    # Inputs: gaiadrname -> Gaia ID for the star                        #
    #         plx, dplx -> parallax and error, default is None          #
    #                      if None, function pulls parallax from        #
    #                      Gaia catalog                                 #
    #         verbose flag -> default is False                          #
    #                         if True, allows print statements          #
    # Outputs: the Gaia zero-point corrected distance in parsecs        #
    # What it does:                                                     #
    #    1. reads in the source_id, parallax, and parallax error        #
    #    2. checks to make sure the Gaia name is a string               #
    #    3. splits the Gaia name into just the source id numbers        #
    #    4. checks to see which Gaia DR we need to use                  #
    #    5. queries the correct catalog for the needed parameters       #
    #       source id, ra, dec, phot_g_mean_mag, parallax,              #
    #       parallax error, nu_eff_used_in_astrometry,                  #
    #       pseudocolour, ecliptic latitude, and                        #
    #       astrometric_params_solved                                   #
    #    6. if no parallax/parallax error is given by the user,         #
    #       it sets the parallax pulled from the Gaia DR                #
    #    7. Sets the variable names                                     #
    #    8. Checks to make sure the zero_point function is only         #
    #       interpolating, not extrapolating                            #
    #    9. Calculates the zero-point correction value                  #
    #   10. Calculates the distance                                     #
    #####################################################################
    zpt.load_tables()
    name = check_if_string(gaiadrname, verbose)
    source_id = name.split()[-1]
    if verbose:
        print('Source ID', source_id)

    if 'Gaia DR3' in name:
        job = Gaia.launch_job(f"""
            SELECT source_id, ra, dec, phot_g_mean_mag, parallax, parallax_error, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved
            FROM gaiadr3.gaia_source
            WHERE source_id = {source_id}
        """)
        result = job.get_results()
    elif 'Gaia DR2' in name:
        job = Gaia.launch_job(f"""
            SELECT source_id, ra, dec, phot_g_mean_mag, parallax, parallax_error, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved
            FROM gaiadr2.gaia_source
            WHERE source_id = {source_id}
        """)
        result = job.get_results()
    elif 'Gaia DR1' in name:
        job = Gaia.launch_job(f"""
            SELECT source_id, ra, dec, phot_g_mean_mag, parallax, parallax_error, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved
            FROM gaiadr3.gaia_source
            WHERE source_id = {source_id}
        """)
        result = job.get_results()
    else:
        if verbose:
            print('Incorrect input. Please make sure you have a Gaia ID for DR3, DR2, or DR1.')

    if plx is None:
        plx = result['parallax'][0]
    if dplx is None:
        dplx = result['parallax_error'][0]
    gmag = result['phot_g_mean_mag'][0]
    nueff = result['nu_eff_used_in_astrometry'][0]
    pseudo = result['pseudocolour'][0]
    eclat = result['ecl_lat'][0]
    soltype = result['astrometric_params_solved'][0]

    valid = soltype > 3
    zpcorrect = zpt.get_zpt(gmag[valid], nueff[valid], pseudo[valid], eclat[valid], soltype[valid], _warnings=verbose)[
        0]

    distance = dist_calc(plx, dplx, zpcorrect, verbose)
    d = distance[0]
    dd = distance[1]

    return round(d,5), round(dd,5)


def use_hipparcos(star_name, plx=None, dplx=None, verbose=False):
    ###################################################################
    # Function: use_hipparcos                                         #
    # Inputs: star_name -> name of star                               #
    #         plx, dplx -> parallax and error                         #
    #                      If no user input, pulls value from catalog #
    #         verbose flag -> Default is False                        #
    #                         if True, allows print statements        #
    # Outputs: distance and distance error                            #
    # What it does                                                    #
    #     1. Queries "Hipparcos: the New Reduction" catalog           #
    #     2. if it finds results,                                     #
    #           a. assigns parallax and error if set to None          #
    #           b. calculates distance                                #
    #           c. returns distance                                   #
    #     3. if it doesn't find results:                              #
    #           a. returns None, None                                 #
    ###################################################################

    hip_result = Vizier.query_object(star_name, catalog="I/311/hip2")
    if hip_result:
        if verbose:
            print('Found in Hipparcos catalog')
        hip = hip_result[0]
        if plx is None:
            plx = hip['Plx'][0]
        if dplx is None:
            dplx = hip['e_Plx'][0]
        distance = dist_calc(plx, dplx, 0, verbose)
        d = distance[0]
        dd = distance[1]
        return round(d,5), round(dd,5)
    if not hip_result:
        if verbose:
            print('Not found in Hipparcos catalog.')
        return None, None


def check_if_string(name, verbose=False):
    # checks if the name is a string or not
    # if it isn't, converts to a string
    if not isinstance(name, str):
        if verbose:
            print('Star name is not a valid type. Should be a string. Correcting it now.')
        name = str(name)
        return name
    else:
        return name


def distances(star_name, plx=None, dplx=None, use_Hipp=False, verbose=False):
    ################################################################
    # Function: distances                                          #
    # Inputs: star_name -> star name, needs to be a string         #
    #         plx, dplx -> parallax and error, default is None     #
    #                      if default is set, code will pull the   #
    #                      Gaia parallax or the Hipparcos parallax #
    #         use_Hipp -> indicates if user wants to use Hipparcos #
    #                     instead of Gaia                          #
    #                     default is Gaia                          #
    #         verbose flag -> Default is False                     #
    #                         if True, allows print statements     #
    # Outputs: distance and distance error in units of parsecs     #
    # What it does:                                                #
    #     1. Checks to make sure the star name is a string.        #
    #     If use_Hipp flag is set:                                 #
    #     2. calculates the distance using use_hipparcos           #
    #     If use_Hipp flag is not set or it failed:                #
    #     2. Queries Simbad for a list of star IDs using the name. #
    #     3. Checks which Gaia ID it found. First choice is DR3.   #
    #        If there is a DR3 ID:                                 #
    #        a. pulls the the Gaia DR3 ID.                         #
    #        b. Prints a congrats there was one.                   #
    #        c. Calls the gaia_correct_distance function which     #
    #            returns the gaia zero-point corrected distance    #
    #            and error.                                        #
    #        If there is no DR3 ID:                                #
    #        a. Checks for a Gaia DR2 ID                           #
    #        b. pulls the DR2 and prints a congrats we found one.  #
    #        c. Calls the gaia_correct_distance function which     #
    #           returns the gaia zero-point corrected distance     #
    #           and error.                                         #
    #        If there is no DR2 ID:                                #
    #        a. Checks for a Gaia DR1 ID                           #
    #        b. pulls the DR1 and prints a congrats we found one.  #
    #        c. Calls the gaia_correct_distance function which     #
    #           returns the gaia zero-point corrected distance     #
    #           and error.                                         #
    #        If there is no DR1 ID:                                #
    #        a. Prints "No Gaia ID. Searching for Hipparcos'       #
    #        b. Queries the van leeuwen (2007) updated reduction   #
    #           Hipparcos catalog                                  #
    #        c. Pulls the parallax and parallax error              #
    #        d. Calculates the distance                            #
    #     4. Returns the distance.                                 #
    ################################################################
    star_name = check_if_string(star_name, verbose)
    used_hipparcos = False

    if use_Hipp:
        if verbose:
            print('Using Hipparcos parallaxes instead of Gaia.')
        d, dd = use_hipparcos(star_name, plx, dplx, verbose)
        if d is not None:
            used_hipparcos = True
            return round(d,5), round(dd,5)

    if not used_hipparcos:
        try:
            simbad_result = Simbad.query_objectids(star_name)
            if simbad_result is None:
                if verbose:
                    print(f"No results found in Simbad for {star_name}")
                    d, dd = use_hipparcos(star_name, plx, dplx, verbose)
                return round(d, 5), round(d, dd)

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
                    print("Found Gaia DR3:", gaiadr3_name)
                d, dd = gaia_correct_distance(gaiadr3_name, plx, dplx, verbose)
                return round(d,5), round(dd,5)

                # Only check DR2 if no DR3 found
            gaiadr2mask = ['Gaia DR2' in name for name in ids]
            if any(gaiadr2mask):
                gaiadr2_name = ids[gaiadr2mask][0]
                if verbose:
                    print("Found Gaia DR2:", gaiadr2_name)
                d, dd = gaia_correct_distance(gaiadr2_name, plx, dplx, verbose)
                return round(d,5), round(dd,5)

            # Only check DR1 if no DR2 found
            gaiadr1mask = ['Gaia DR1' in name for name in ids]
            if any(gaiadr1mask):
                gaiadr1_name = ids[gaiadr1mask][0]
                if verbose:
                    print("Found Gaia DR1:", gaiadr1_name)
                d, dd = gaia_correct_distance(gaiadr1_name, plx, dplx, verbose)
                return round(d,5), round(dd,5)
        except Exception as e:
            if verbose:
                print(f"Error in Simbad query: {e}")

        # Finally fall back to Hipparcos
        if verbose:
            print("No Gaia name found. Checking Hipparcos.")
        d, dd = use_hipparcos(star_name, plx, dplx, verbose)
        return round(d,5), round(dd,5)

        if d is None:
            if verbose:
                print('No valid distance could be found from Gaia or Hipparcos')
            return None, None

def calc_star_params(stellar_params, verbose=False):
    #################################################################
    # Function: calc_star_params                                    #
    # Inputs: stellar_params -> star object                         #
    #         verbose -> if True, allows the print statements       #
    # Outputs: None                                                 #
    # What it does:                                                 #
    #       1. Unpacks the star object                              #
    #       2. Calculated the physical radius and error using       #
    #          and_to_lin                                           #
    #       3. Calculates the effective temperature and error       #
    #          using temp                                           #
    #       4. Calculates the luminosity and error using luminosity #
    #       5. Updates the stellar object with the new params       #
    #################################################################
    theta = stellar_params.ldtheta
    dtheta = stellar_params.ldtheta_err
    d = stellar_params.dist
    dd = stellar_params.dist_err
    fbol = stellar_params.fbol
    dfbol = stellar_params.fbol_err

    r, dr = ang_to_lin(theta, dtheta, d, dd)
    t, dt = temp(fbol, dfbol, theta, dtheta)
    l, dl = luminosity(d, dd, fbol * (1e-8), dfbol * (1e-8))

    stellar_params.teff = round(t, 5)
    stellar_params.teff_err = round(dt,5)
    stellar_params.rad = round(r,5)
    stellar_params.rad_err = round(dr,5)
    stellar_params.lum = round(l,5)
    stellar_params.lum_err = round(dl,5)

    if verbose:
        print("Linear Radius: ", round(r, 3), "+/-", round(dr, 3), "[R_solar]")
        print("Luminosity: ", round(l, 3), "+/-", round(dl, 3), "[L_solar]")
        print("Effective temperature: ", round(t, 3), "+/-", round(dt, 3), "[K]")


class StellarParams:
    # Attributes can include:
    # fbol and fbol_err: bolometric flux and its error in units of 10^-8 ergs/ s*cm^2
    # logg and logg_err: surface gravity in units of dex
    # feh and feh_err: metallicity of star in units of dex
    # dist and dist_err: distance and its error in units of parsecs
    # plx and plx_err: parallax and its error in units of mas
    # initial udtheta and udtheta_err: uniform disk diameter and its error in units of mas
    # initial ldtheta and ldtheta_err: limb darkened disk diameter and its error in units of mas
    # final udtheta and udtheta_err: uniform disk diameter and its error in units of mas
    # final ldtheta and ldtheta_err: limb darkened disk diameter and its error in units of mas
    # teff and teff_err: temperature and its error in units of Kelvin
    # lum and lum_err: luminosity and its error in units of Lsol
    # rad and rad_err: physical radius and its error in units of Rsol

    __units__ = {
        'fbol': '10⁻⁸ erg/s/cm²',
        'fbol_err': '10⁻⁸ erg/s/cm²',
        'logg': 'dex',
        'logg_err': 'dex',
        'feh': 'dex',
        'feh_err': 'dex',
        'plx': 'mas',
        'plx_err': 'mas',
        'dist': 'pc',
        'dist_err': 'pc',
        'udthetai': 'mas',
        'udthetai_err': 'mas',
        'ldthetai': 'mas',
        'ldthetai_err': 'mas',
        'udtheta': 'mas',
        'udtheta_err': 'mas',
        'ldtheta': 'mas',
        'ldtheta_err': 'mas',
        'teff': 'K',
        'teff_err': 'K',
        'lum': 'L☉',
        'lum_err': 'L☉',
        'rad': 'R☉',
        'rad_err': 'R☉',
        'ldc_R': ' ',
        'ldc_K': ' ',
        'ldc_H': ' ',
        'ldc_J': ' '
    }

    def __init__(self, **kwargs):
        # Set all known attributes to None by default
        for key in self.__class__.__units__:
            setattr(self, key, None)
        # Then overwrite with provided values
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        lines = []
        for k, v in self.__dict__.items():
            unit = self.__units__.get(k, '')
            unit_str = f" [{unit}]" if unit else ''
            lines.append(f"{k} = {v}{unit_str}")
        return "\n".join(lines)
