import re
import os
from radpy.stellar import *
from radpy.datareadandformat import *
from radpy.plotting import plot_v2_fit
from radpy.LDfitting import initial_LDfit, run_LDfit
from radpy.UDfitting import initial_UDfit, run_UDfit, udfit_values

def extract_id(star_name):
    #############################################
    # Function: extract_id                      #
    # Inputs: star_name -> name of the star     #
    # Outputs: just the numbers of the star ID  #
    # What it does:                             #
    #     1. Takes in the star name             #
    #     2. Searches the string for one or     #
    #        more digits.                       #
    #     3. If it found some, returns the      #
    #        exact sequence of digits.          #
    #     4. If none are found, returns None    #
    #############################################
    match = re.search(r'\d+', star_name)
    return match.group(0) if match else None


def find_files_for_star(star_id, data_dir):
    ####################################################
    # Function: find_files_for_star                    #
    # Inputs: star_id -> star id numbers               #
    #         data_dir -> data directory               #
    # Outputs: matches -> list of files matching to    #
    #                     the star ID                  #
    # What it does:                                    #
    #      1. sets the ignore extension variable for   #
    #         any image extension                      #
    #      2. loops through the files in the data      #
    #         directory                                #
    #      3. Searches for the star id in the file     #
    #         name.                                    #
    #      4. Checks the file extension on each file   #
    #      5. If file is not in the ignore list,       #
    #         appends file to the matches list         #
    #      6. Returns the file list                    #
    ####################################################
    ignore_ext = {'.jpg', '.jpeg', '.png', '.pdf', '.eps', '.gif', '.tif', '.tiff', '.bmp'}
    pattern = re.compile(rf'_{star_id}\b')  # underscore, then star_id, then word boundary
    matches = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if pattern.search(file) and ext not in ignore_ext:
                matches.append(os.path.join(root, file))
    return matches

def extract_instrument_from_filename(filename):
    ####################################################
    # Function: extract_instrument_from_filename       #
    # Inputs: filename -> filename                     #
    # Outputs: key -> instrument key                   #
    # What it does:                                    #
    #      1. Creates instrument dictionary with keys  #
    #      2. Pulls instrument name from file name     #
    #      3. Checks for filename and matches the key  #
    #      4. As a fall back, returns the first letter #
    #         of the filename                          #
    #      5. If not found, returns "UNKNOWN"          #
    ####################################################
    instruments = {'C': 'classic', 'P': 'pavo', 'V': 'vega', 'M': 'mircx', 'MY': 'mystic', 'S': 'spica'}
    name = os.path.basename(filename).upper()
    # Try to detect instrument by filename
    if "PAVO" in name: return "P"
    if "VEGA" in name: return "V"
    if "CLASSIC" in name or "CLIMB" in name: return "C"
    if "MIRCX" in name: return "M"
    if "MYSTIC" in name: return "MY"
    if "SPICA" in name: return "S"
    # Fallback: use first letter prefix
    for key in instruments:
        if name.startswith(key):
            return key
    return "UNKNOWN"

def save_plot(fig, out_dir, star_id, plot_type, extension):
    ##########################################################
    # Function: save_plot                                    #
    # Inputs: fig -> the figure                              #
    #         out_dir -> output directory                    #
    #         star_id -> name of star                        #
    #         plot_type -> type of model being plotted       #
    #         extension -> type of file extension            #
    # Outputs: saves figure                                  #
    # What it does:                                          #
    #      1. Checks to see if output directory has been     #
    #         made. If not, creates it.                      #
    #      2. Based on file extension, saves figure to       #
    #         output directory                               #
    ##########################################################
    os.makedirs(out_dir, exist_ok=True)
    if extension == '.png':
        fig.savefig(os.path.join(out_dir, f"{star_id}_{plot_type}.png"), bbox_inches = 'tight', dpi = 200)
    if extension == '.jpg':
        fig.savefig(os.path.join(out_dir, f"{star_id}_{plot_type}.jpg"), bbox_inches='tight', dpi = 200)
    if extension == '.eps':
        fig.savefig(os.path.join(out_dir, f"{star_id}_{plot_type}.eps"), bbox_inches='tight', dpi = 200)
    if extension == '.pdf':
        fig.savefig(os.path.join(out_dir, f"{star_id}_{plot_type}.pdf"), bbox_inches='tight', dpi = 200)

def write_latex_table(df, out_file):
    #####################################################
    # Function: write_latex_table                       #
    # Inputs: df -> dataframe to write to table         #
    #         out_file -> output tex file               #
    # Outputs: writes to a latex file                   #
    # What it does:                                     #
    #       1. Opens the out_file                       #
    #       2. Writes to the file                       #
    #####################################################
    with open(out_file, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))

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
    df = pd.read_csv(file_path, sep='\t')
    star_names = df['Star'].tolist()
    params_dict = {}
    for _, row in df.iterrows():
        params_dict[row['Star']] = {
            'fbol': row['fbol'],
            'fbol_err': row['dfbol'],
            'logg': row['logg'],
            'logg_err': row['dlogg'],
            'feh': row['feh'],
            'feh_err': row['dfeh'],
        }
    return star_names, params_dict


def converttoIDobj(dfs, verbose=False):
    ####################################################
    # Function: converttoIDobj                         #
    # Inputs: dfs -> dictionaries                      #
    # Outputs: wrapped_data -> instrument class        #
    # What it does:                                    #
    #        1. Creates instrument class map           #
    #        2. Loops through the dictionaries and     #
    #           sorts the files by instrument          #
    #        3. Converts the data file to the instru-  #
    #           ment class                             #
    #        4. Returns the instrument class object    #
    ####################################################
    # Allows for dynamic wrapping of the different instrument classes
    instrument_class_map = {
        "P": PavoData,
        "V": VegaData,
        "C": ClassicData,
        "M": MircxData,
        # "My": MysticData,
        # "S": SpicaData
    }
    wrapped_data = {}
    for inst_code, df in dfs.items():
        data_class = instrument_class_map.get(inst_code)
        if data_class:
            try:
                wrapped_data[inst_code] = data_class(df)
            except Exception as e:
                print(f"Failed to wrap data for {inst_code}: {e}")
        else:
            print(f"No class found for '{inst_code}'")
    return wrapped_data


def data_dict_plotting(wrapped_data):
    #####################################################
    # Function: data_dict_plotting                      #
    # Inputs: wrapped_data -> Instrument Class object   #
    # Outputs: data_dict -> data dictionary             #
    # What it does:                                     #
    #       1. Creates instrument map                   #
    #       2. Creates data dictionary for each data    #
    #          file                                     #
    #       3. Returns dictionary                       #
    #####################################################
    instrument_name_map = {
        "P": "pavo",
        "V": "vega",
        "C": "classic",
        "M": "mircx",
        "My": "mystic",
        "S": "spica"
    }
    data_dict = {
        instrument_name_map[code]: data
        for code, data in wrapped_data.items()
        if code in instrument_name_map
    }

    return data_dict

def format_catalog_name(name):
    ########################################################
    # Function: format_catalog_name                        #
    # Inputs: name -> star name                            #
    # Outputs: latex formatted name                        #
    # What it does:                                        #
    #       1. Checks to see if the name is already in     #
    #          correct format                              #
    #       2. Replaces the underscores in the name with   #
    #          spaces if there are underscores             #
    #       3. Inserts a space between the prefix and the  #
    #          numbers only at the first transition        #
    #       4. Replaces the spaces with '~'                #
    #       5. Returns the latex formatted name            #
    ########################################################

    if name.startswith('$\\rm') and name.endswith('$'):
        return name
    name = name.replace('_', ' ')
    name = re.sub(r'^([A-Za-z+\-]+)\s*([0-9].*)', r'\1 \2', name)
    return rf'$\rm {name.replace(" ", "~")}$'

def convert_names_to_latex(names):
    #####################################################
    # Function: convert_names_to_latex                  #
    # Inputs: names -> list of names                    #
    # Outputs: names2 -> list of latex formatted names  #
    # What it does:                                     #
    #      1. For every name in the names list, formats #
    #         the name with format_catalog_name()       #
    #      2. Returns formatted names                   #
    #####################################################
    names2 = [format_catalog_name(name) for name in names]
    return names2


def process_star(star_name, data_dir, output_dir, stellar_param_dict, latex_rows, mc_num=71, bs_num=71,
                 set_axis=None, image_ext=None, binned=None, ldc_band=None, verbose=True):
    ##################################################################
    # Function: process_star                                         #
    # Inputs: star_name -> name of star                              #
    #         data_dir -> data directory                             #
    #         output_dir -> output directory                         #
    #         stellar_params_dict -> dictionary with stellar params  #
    #         latex_rows -> for latex table                          #
    #         mc_num -> number of Monte Carlo iterations             #
    #                   defaults to 71                               #
    #         bs_num -> number of bootstrap iterations               #
    #                   defaults to 71                               #
    #         set_axis -> axis limits [xmin, xmax, ymin, ymax]       #
    #                     defaults to none                           #
    #         image_ext -> image extension for file                  #
    #                      Options: '.jpg', '.png', '.pdf', '.eps'   #
    #         binned -> data you want binned i.e. ['pavo']           #
    #         ldc_band -> limb darkening band you want               #
    #                     Options: 'ldc_R', 'ldc_H', 'ldc_K', 'ldc_J'#
    #         verbose -> allows print statements to screen           #
    # Outputs: None                                                  #
    # What it does:                                                  #
    #      1. Extracts the star name                                 #
    #      2. Finds the files for the star                           #
    #      3. Groups the files by instrument.                        #
    #      4. Process the files by instrument                        #
    #      5. Combines the data files per instrument.                #
    #      6. wraps the data into InterferometryData Objects         #
    #      7. Combines all data into one dataframe                   #
    #      8. Sets up the StellarParams object                       #
    #      9. Calculates the distance                                #
    #     10. Performs initial fits for uniform disk and limb-dark   #
    #     11. Performs the UD MC fit                                 #
    #     12. Calculates parameters after UD fit                     #
    #     13. Performs the LD MC fit                                 #
    #     14. Calculates final stellar parameters                    #
    #     15. Sets up plot directory                                 #
    #     16. If data are to be binned, creates a binned data dict   #
    #     17. converts the star name into latex format               #
    #     18. Plots the UD fit                                       #
    #     19. Saves the plot                                         #
    #     20. Plots the LD fit                                       #
    #     21. Saves the plot                                         #
    #     22. Appends the stellar parameters to the latex_rows       #
    ##################################################################
    star_id = extract_id(star_name)
    print("--------------------------------------------------")
    print(f"Starting processing for {star_name}")
    files = find_files_for_star(star_id, data_dir)
    if not files:
        print(f"No files found for {star_name} ({star_id})")
        return

    # Group files by instrument
    grouped_files = {}
    for file in files:
        inst = extract_instrument_from_filename(file)
        grouped_files.setdefault(inst, []).append(file)

    # Process files by instrument
    dataframes = {}
    for inst, filelist in grouped_files.items():
        dfs = []
        for file in filelist:
            try:
                df, _ = filename_extension(file, inst, verbose=verbose)
                dfs.append(df)
            except Exception as e:
                print(f"Failed to process {file} ({inst}): {e}")
        if dfs:
            # Concatenate all files for this instrument
            dataframes[inst] = pd.concat(dfs, ignore_index=True)

    # Convert to InterferometryData objects and wrap into instrument classes
    wrap_data = converttoIDobj(dataframes, verbose=False)

    # Combine all available data using RADPy's combined() utility
    combined_args = [obj.make_df() for obj in wrap_data.values()]

    b, v2, dv2, wave, band, brack, inst = combined(*combined_args)
    spf = b / wave

    # Set up stellar parameters
    star = StellarParams()
    params = stellar_param_dict.get(star_name, {})
    for param, value in params.items():
        setattr(star, param, value)
    # Calculate distance
    if params.get('plx', None) is not None:
        D, dD = distances(star_name, plx=params['plx'], dplx=params['plx_err'], verbose=verbose)
        star.dist = D
        star.dist_err = dD
    else:
        D, dD = distances(star_name, verbose=verbose)
        star.dist = D
        star.dist_err = dD

    # Initial fits
    initial_UDfit(spf, v2, dv2, 0.4, star, verbose=verbose)
    initial_LDfit(spf, v2, dv2, star, 'R', verbose=verbose)

    # Monte Carlo uniform-disk fit
    datasets = list(wrap_data.values())
    results_UD = run_UDfit(bs_num, mc_num, datasets=datasets, stellar_params=star)
    udfit_values(spf, v2, dv2, results_UD, stellar_params=star, verbose=verbose)

    # Monte Carlo limb-darkened fit
    run_LDfit(bs_num, mc_num, ogdata=[spf, v2, dv2], datasets=datasets, stellar_params=star, verbose=verbose)

    # Calculate additional stellar parameters
    calc_star_params(star, verbose=verbose)

    # Save plots
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    data_dict = data_dict_plotting(wrap_data)
    if binned:
        bin_only = [k for k in binned if k in data_dict]
    else:
        bin_only = None

    star_title = convert_names_to_latex([star_name])
    # Uniform disk plot
    fig1, _ = plot_v2_fit(
        data_dict=data_dict,
        star=star,
        datasets_to_plot=list(data_dict.keys()),
        to_bin=bin_only,
        plot_udmodel=True,
        eq_text=True,
        set_axis=set_axis,
        title=star_title[0],
        show=False
    )
    save_plot(fig1, plot_dir, star_id, "UDfit", image_ext)

    # Limb-darkened R band plot
    fig2, _ = plot_v2_fit(
        data_dict=data_dict,
        star=star,
        datasets_to_plot=list(data_dict.keys()),
        to_bin=bin_only,
        plot_ldmodel=True,
        ldc_band=ldc_band,
        title=star_title[0],
        set_axis=set_axis,
        eq_text=True,
        show=False
    )
    save_plot(fig2, plot_dir, star_id, "LDfit", image_ext)

    # Collect results for LaTeX
    latex_rows.append({
        "Star": star_name,
        "D (pc)": star.dist,
        r"$\Delta \rm D (pc)$": star.dist_err,
        r"$\theta_{\rm UD}$ (mas)": star.udtheta,
        r"$\Delta\theta_{\rm UD}$ (mas)": star.udtheta_err,
        r"$\theta_{\rm LD}$ (mas)": star.ldtheta,
        r"$\Delta\theta_{\rm LD}$ (mas)": star.ldtheta_err,
        r"$T_{\rm eff}$ (K)": star.teff,
        r"$\Delta T_{\rm eff}$ (K)": star.teff_err,
        r"$L_{\star} (\rm L_{\odot})$": star.lum,
        r"$\Delta L_{\star} (\rm L_{\odot})$": star.lum_err,
        r"$R_{\star} (\rm R_{\odot})$": star.rad,
        r"$\Delta R_{\star} (\rm R_{\odot})$": star.rad_err,
        r"$\mu_{\rm R}$": star.ldc_R,
        r"$\mu_{\rm K}$": star.ldc_K,
        r"$\mu_{\rm H}$": star.ldc_H,
        r"$\mu_{\rm J}$": star.ldc_J,
    })

    print(f"Finished processing {star_name}")

def batch_mode(star_file, data_dir, output_dir, latex_out, mc_num=71, bs_num=71, set_axis = None, image_ext=None, binned=None, ldc_band=None, verbose=True):
    ######################################################
    # Function: batch_mode                               #
    # Inputs: star_file -> stellar param file            #
    #         data_dir -> data directory                 #
    #         output_dir -> output directory             #
    #         latex_out -> name of latex file            #
    #         mc_num -> number of MC iterations          #
    #                   defaults to 71                   #
    #         bs_num -> number of bootstrap iterations   #
    #                   defaults to 71                   #
    #         set_axis -> sets the axis limits           #
    #                   defaults to None (automatically  #
    #                   assigns)                         #
    #         image_ext -> sets image extension          #
    #         binned -> datasets you want binned         #
    #         ldc_band -> limb-darkening coefficent you  #
    #                     want plotted                   #
    #         verbose -> allows print statements         #
    # Outputs: None                                      #
    # What it does:                                      #
    #       1. Changes directory to the data directory   #
    #       2. Extracts star names and stellar params    #
    #       3. Assigns empty rows for latex_rows         #
    #       4. Initializes count iterator                #
    #       5. Loops through the star names and runs     #
    #          process_star for each                     #
    #       6. Updates loop counter                      #
    #       7. Outside loop, creates a DataFrame of the  #
    #          stellar parameters                        #
    #       8. Writes the dataframe to a latex table     #
    ######################################################
    os.chdir(data_dir)
    star_names, star_params = get_stellar_params(star_file)
    latex_rows = []
    count = 0
    for star_name in star_names:
        process_star(star_name, data_dir, output_dir, star_params, latex_rows, mc_num=mc_num, bs_num=bs_num, set_axis = set_axis,
                     image_ext=image_ext, binned=binned, ldc_band=ldc_band, verbose=verbose)
        count += 1

    latex_df = pd.DataFrame(latex_rows)
    write_latex_table(latex_df, latex_out)
    print(f"Batch complete. Fit {count} stars. Plots in {os.path.join(output_dir, 'plots')}, results in {latex_out}")