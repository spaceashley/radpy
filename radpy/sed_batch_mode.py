from radpy.sedfit import *
from radpy.stellar import *
from radpy.batchmode import extract_id, find_files_for_star, convert_names_to_latex, format_catalog_name, save_plot

def get_stellar_params(file_path):
    ######################################################
    # Function: get_stellar_params                       #
    # Inputs: file_path -> path to file                  #
    # Outputs: star_names -> names of the stars          #
    #          star_params -> dictionary of stellar      #
    #                         params                     #
    # What it does:                                      #
    #     1. reads in the file                           #
    #     2. extracts the star names                     #
    #     3. For each star name, extracts the stellar    #
    #        parameters and adds it to the dictionary    #
    #     4. Returns the star names and the dictionary   #
    ######################################################
    df = pd.read_csv(file_path)
    star_names = df['Star'].tolist()
    params_dict = {}
    fitting_dict = {}
    for _, row in df.iterrows():
        params_dict[row['Star']] = {
            'logg': row['logg'],
            'logg_err': row['dlogg'],
            'feh': row['feh'],
            'feh_err': row['dfeh'],
        }
    for _, row in df.iterrows():
        fitting_dict[row['Star']] = {
            'fitTeff': row['fitteff'],
            'Trange': row['fitteff_range'],
            'fitLogg': row['fitlogg'],
            'Loggrange':row['fitlogg_range'],
            'fitFeh': row['fitfeh'],
            'Fehrange': row['fitfeh_range'],
            'fitAv': row['fitav'],
            'Avrange': row['fitav_range'],
            'model': row['model'],
        }
    return star_names, params_dict, fitting_dict


def convert_fit_params(star_name, star, fit_params_dict, verbose=False):
    fits = fit_params_dict.get(star_name, {})

    fitT = fits['fitTeff']
    fitLG = fits['fitLogg']
    fitFEH = fits['fitFeh']
    fitAV = fits['fitAv']

    model = fits['model']

    init_logg = star.logg
    init_feh = star.feh
    init_av = 0
    if fitT:
        teff_range = ast.literal_eval(fits['Trange'])
        init_teff = (teff_range[0] + teff_range[1]) / 2
    else:
        init_teff = 5000
        teff_range = None
    if fitLG:
        logg_range = ast.literal_eval(fits['Loggrange'])
    else:
        logg_range = None
    if fitFEH:
        feh_range = ast.literal_eval(fits['Fehrange'])
    else:
        feh_range = None
    if fitAV:
        av_range = ast.literal_eval(fits['Avrange'])
    else:
        av_range = None

    init_vals = [init_teff, init_logg, init_feh, init_av]
    ranges = [teff_range, logg_range, feh_range, av_range]
    fitflags = [fitT, fitLG, fitFEH, fitAV]
    return init_vals, ranges, fitflags, model


def create_photometry_file(star_id, out_dir, verbose=False):
    starname = star_id
    star = StellarParams()
    ra_deg, dec_deg, ra_hms, dec_hms = pull_coords(starname, star, verbose=verbose)
    gaia_id = pull_gaia_id(starname, star, verbose=verbose)
    photometry = extract_photometry(starname, star, verbose=verbose)
    out_dir = out_dir
    if verbose:
        print(f"Extracted photometry for {starname}")
    remove_photometry(photometry, 'su', verbose=verbose)
    filename = save_photometry(starname, photometry, out_dir, verbose=verbose)

    return filename

def write_results(starname, sed_obj, star,  out_dir, rows_for_results, rows_for_diams):
    star_teff = round(sed_obj.getteff()[0], 2)
    star_av = round(sed_obj.getav(), 4)
    star_rad = round(sed_obj.getr()[0], 3)
    star_fbol = star.fbol
    star_fbol_err = star.fbol_err
    star_logg = star.logg
    star_logg_err = star.logg_err
    star_feh = star.feh
    star_feh_err = star.feh_err
    star_chi2 = star.SEDchi2
    star_chi2r = star.SEDchi2red

    rows_for_diams.append({
        "Star": starname,
        "fbol": star_fbol,
        "dfbol": star_fbol_err,
        "logg": star_logg,
        "dlogg": star_logg_err,
        "feh": star_feh,
        "dfeh": star_feh_err
    })

    rows_for_results.append({
        "Star": starname,
        "SED Teff": star_teff,
        "SED Av": star_av,
        "SED Radius": star_rad,
        "SED Chi2": star_chi2,
        "SED Chi2Red" : star_chi2r
    })

    return rows_for_diams, rows_for_results

def write_table(df, out_dir, out_file):
    #####################################################
    # Function: write_latex_table                       #
    # Inputs: df -> dataframe to write to table         #
    #         out_file -> output tex file               #
    # Outputs: writes to a latex file                   #
    # What it does:                                     #
    #       1. Opens the out_file                       #
    #       2. Writes to the file                       #
    #####################################################
    os.chdir(out_dir)
    df.to_csv(out_file, sep='\t', index=False)


def sed_process_star(star_name, data_dir, output_dir, stellar_param_dict, fitting_param_dict, unit, num_iter, set_axis, image_ext,
                     result_rows, diam_rows, uselatex, logplot, fbol_lam, own_photometry=False, verbose=False):
    star_id = extract_id(star_name)
    print("--------------------------------------------------")
    print(f"Starting processing for {star_name}")
    if star_id is None:
        star_id = star_name

    if own_photometry:
        if verbose:
            print("Finding user generated photometry files")
        files = find_files_for_star(star_id, data_dir)
        phot_data = read_in_photometry(files[0])
        if verbose:
            print(f"Files found for {star_name}:", files)
        if not files:
            print(f"No files found for {star_name} ({star_id})")
            return
    if not own_photometry:
        if verbose:
            print(f"Extracting photometry for {star_name}")
        files = create_photometry_file(star_name, data_dir, verbose=verbose)
        phot_data = read_in_photometry(files)

    star = StellarParams()
    params = stellar_param_dict.get(star_name, {})
    for param, value in params.items():
        setattr(star, param, value)
    ra_deg, dec_deg, ra_hms, dec_hms = pull_coords(star_name, star, verbose=verbose)
    gaia_id = pull_gaia_id(star_name, star, verbose=verbose)
    D, dD = distances(star_name, verbose=verbose)
    star.dist = D
    star.dist_err = dD

    init_values, fit_ranges, fit_flags, model = convert_fit_params(star_name, star, fitting_param_dict, verbose=False)

    sed_fit = fit_sed(phot_data, star, init_values, num_iter, model, teffrange=fit_ranges[0], loggrange=fit_ranges[1],
                      fehrange=fit_ranges[2], avrange=fit_ranges[3], fitT=fit_flags[0],
                      fit_logg=fit_flags[1], fit_feh=fit_flags[2], fit_av=fit_flags[3], verbose=verbose)


    sed_results, diam_resuls = write_results(star_name, sed_fit, star, output_dir, result_rows, diam_rows)

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    star_title = convert_names_to_latex([star_name])

    fig, _ = plot_sed(sed_fit,
                      unit=unit,
                      logplot=logplot,
                      fbol_lam=fbol_lam,
                      set_axis=set_axis,
                      title=star_title[0],
                      uselatex=uselatex,
                      verbose=verbose)

    save_plot(fig, plot_dir, star_name.replace(" ", ""), "SEDfit", image_ext)

    print(f"Finished processing {star_name}")


def sed_batchmode(starfile, data_dir, out_dir, res_out, diam_out, unit, num_iter, set_axis, image_ext, uselatex, logplot,
                  fbol_lam, own_photometry, verbose=False):
    os.chdir(data_dir)
    star_names, star_params, fit_params = get_stellar_params(starfile)
    res_rows = []
    diam_rows = []
    count = 0
    for star_name in star_names:
        sed_process_star(star_name, data_dir, out_dir, star_params, fit_params, unit, num_iter, set_axis, image_ext, res_rows,
                         diam_rows, uselatex, logplot, fbol_lam, own_photometry=own_photometry, verbose=verbose)
        count += 1

    res_df = pd.DataFrame(res_rows)
    diam_df = pd.DataFrame(diam_rows)

    write_table(res_df, out_dir, res_out)
    write_table(diam_df, out_dir, diam_out)

    print(
        f"Batch complete. Fit {count} stars. Plots in {os.path.join(out_dir, 'plots')}, SED fit results in {res_out}, File for diameter fitting in {diam_out}")
