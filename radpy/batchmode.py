import re
import os
import pandas as pd
from radpy.stellar import *
from radpy.datareadandformat import *
from radpy.plotting import plot_v2_fit
from radpy.LDfitting import initial_LDfit, run_LDfit
from radpy.UDfitting import initial_UDfit, run_UDfit, udfit_values

def extract_id(star_name):
    match = re.search(r'\d+', star_name)
    return match.group(0) if match else None


def find_files_for_star(star_id, data_dir):
    matches = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if star_id in file:
                matches.append(os.path.join(root, file))
    return matches

def extract_instrument_from_filename(filename):
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
    with open(out_file, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))

def get_stellar_params(csv_path):
    """
    Read stellar parameters for each star from a CSV with columns:
    star_name, fbol, fbol_err, logg, logg_err, feh, feh_err
    """
    df = pd.read_csv(csv_path, sep='\t')
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


def process_star(star_name, data_dir, output_dir, stellar_param_dict, latex_rows, mc_num=71, bs_num=71,
                 image_ext=None, binned=None, ldc_band=None, verbose=True):
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

    # Uniform disk plot
    fig1, _ = plot_v2_fit(
        data_dict=data_dict,
        star=star,
        datasets_to_plot=list(data_dict.keys()),
        to_bin=bin_only,
        plot_udmodel=True,
        eq_text=True,
        title=f"{star_name}",
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
        title=f"{star_name}",
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

def batch_mode(star_file, data_dir, output_dir, latex_out, mc_num=71, bs_num=71, image_ext=None, binned=None, ldc_band=None, verbose=True):
    os.chdir(data_dir)
    star_names, star_params = get_stellar_params(star_file)
    latex_rows = []
    count = 0
    for star_name in star_names:
        process_star(star_name, data_dir, output_dir, star_params, latex_rows, mc_num=mc_num, bs_num=bs_num,
                     image_ext=image_ext, binned=binned, ldc_band=ldc_band, verbose=verbose)
        count += 1

    latex_df = pd.DataFrame(latex_rows)
    write_latex_table(latex_df, latex_out)
    print(f"Batch complete. Fit {count} stars. Plots in {os.path.join(output_dir, 'plots')}, results in {latex_out}")