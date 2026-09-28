import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from radpy.UDfitting import UDV2, scaledUDV2
from radpy.LDfitting import V2, scaledV2
plt.rcParams['text.usetex'] = True


def assign_inst_night_v0s(fitted_v0s, datasets):
    ###########################################################
    # Function: assign_inst_night_v0s                         #
    # Inputs: fitted_v0s -> dictionary fo the v0 values       #
    #                       assigned to each night/instrument #
    #         datasets -> datasets being fitted               #
    # Outputs: InterferometryData object with the attribute   #
    #          V0s added to it                                #
    # What it does:                                           #
    #       1. Generated a list from the v0 dictionary keys   #
    #       2. Renames each key with an underscore            #
    #       3. Searches each dataset and matches the correct  #
    #          V0 value with the correct night and instrument #
    #       4. Adds the values to each InterferometryData     #
    #          object.                                        #
    ###########################################################

    v0_keys = list(fitted_v0s.keys())
    for i in range(len(v0_keys)):
        string = v0_keys[i]
        base_id = re.sub(r'[\W_]+', '', string)
        match = re.match(r'([A-Za-z]+)[\s_-]*(\d+)', base_id)
        prefix, number = match.groups()

        if prefix == 'MY' and datasets.instrument == 'my':
            for m_y in range(len(datasets.Night)):
                if datasets.Night[m_y] == float(number):
                    datasets.V0s[m_y] = fitted_v0s[string]
        if prefix == 'M' and datasets.instrument == 'm':
            for m_x in range(len(datasets.Night)):
                if datasets.Night[m_x] == float(number):
                    datasets.V0s[m_x] = fitted_v0s[string]
        if prefix == 'S':
            for s in range(len(spica_data.Night)):
                if spica_data.Night[s] == float(number):
                    spica_data.V0s[s] = fitted_v0s[string]


def V2_wrapper(star, wave, spf):
    #########################################################
    # Function: V2_wrapper                                  #
    # Inputs: star -> StellarParams() object                #
    #         wave -> wavelength array from the combined    #
    #                 data                                  #
    #         spf -> spatial frequnecy array from the       #
    #                combined data                          #
    # Outputs: y -> V2 values                               #
    # What it does:                                         #
    #        1. Reads in the theta value                    #
    #        2. Creates an empty array.                     #
    #        3. Reads in the ldc values                     #
    #        4. If each ldc is not None, applies a mask to  #
    #           the spf data according the the wavelength   #
    #        5. Calculates the V2 value using the ldc and   #
    #           theta value for each mask.                  #
    #        6. Adds the V2 value to the empty array for    #
    #           each mask.                                  #
    #########################################################
    theta = star.ldtheta
    y = np.empty_like(spf)

    ldcK = star.ldc_K
    ldcH = star.ldc_H
    ldcR = star.ldc_R

    if ldcK is not None:
        maskK = (wave > 1.85e-6)
        spfK = spf[maskK]
        y[maskK] = V2(spfK, theta, ldcK)
    if ldcH is not None:
        maskH = (wave < 1.85e-6) & (wave > 9.5e-9)
        spfH = spf[maskH]
        y[maskH] = V2(spfH, theta, ldcH)
    if ldcR is not None:
        maskR = (wave < 9.5e-9)
        spfR = spf[maskR]
        y[maskR] = V2(spfR, theta, ldcR)

    return y


def create_ldc_label(star):
    m_labels = []
    if star.ldc_H is not None:
        model_label = fr"$\rm \mu_H = {star.ldc_H:0.5}$"
        m_labels.append(model_label)
    if star.ldc_K is not None:
        model_label = fr"$ \rm \mu_K = {star.ldc_K:0.5}$"
        m_labels.append(model_label)
    if star.ldc_R is not None:
        model_label = fr"$ \rm \mu_R = {star.ldc_R:0.5}$"
        m_labels.append(model_label)

    if len(m_labels) == 1:
        ldc_label = m_label[0]
        return ldc_label

    if len(m_labels) == 2:
        ldc_label = '{0} \n{1}'.format(m_labels[0], m_labels[1])
        return ldc_label

    if len(m_labels) == 3:
        ldc_label = '{0} \n{1} \n{2}'.format(m_labels[0], m_labels[1], m_labels[2])
        return ldc_label


def extract_v0s_for_latex(star):
    fitted_v0s = star.ldv02_by_group
    fitted_dv0s = star.ldv02_err_by_group

    v0_rows = []
    for key, value in fitted_v0s.items():
        v0_rows.append({r"$V_{0}^{2}$ " + key: value, })

    dv0_rows = []
    for dkey, dvalue in fitted_dv0s.items():
        dv0_rows.append({r"$\Delta V_{0}^{2}$ " + dkey: dvalue, })

    base_v0s = {}
    base_dv0s = {}
    for i in range(len(v0_rows)):
        base_v0s.update(v0_rows[i])
        base_dv0s.update(dv0_rows[i])

    v0dv0 = base_v0s | base_dv0s

    return v0dv0

# Function to bin the PAVO data
def bin_data(x, y, dy, bin_width=5e6, min_points_per_bin=1):
    ###########################################################################
    # Function: bin_data                                                      #
    # Inputs: x -> the x data to bin                                          #
    #         y -> the y data to bin                                          #
    #         bin_width -> the width of the data bins                         #
    #                      default is 5e6 but can be changed                  #
    #         min_points_per_bin-> ensures every bin has 1 data point in it   #
    #                              to avoid nans and issues                   #
    # Outputs: binned_x, binned_y, and binned_dy                              #
    # What it does:                                                           #
    #       1. ensures the input data are arrays                              #
    #       2. sorts the data to make sure its binning properly               #
    #       3. determines the number of bins needed for the data              #
    #       4. sets the bin indices                                           #
    #       5. goes through the data and assigns the data into a bin          #
    #       6. takes the weighted average of the values in each bin           #
    #       7. appends the weighted average into a list                       #
    #       8. Returns the weighted average of each bin                       #
    ###########################################################################

    # Ensure input is numpy array
    x = np.asarray(x)
    y = np.asarray(y)
    dy = np.asarray(dy)

    # Sort by x
    order = np.argsort(x)
    x, y, dy = x[order], y[order], dy[order]

    min_x = x.min()
    max_x = x.max()
    num_bins = max(1, int(np.ceil((max_x - min_x) / bin_width)))
    if num_bins < 1:
        num_bins = 1

    bins = np.linspace(min_x, max_x, num_bins + 1)
    inds = np.digitize(x, bins, right=True)

    avg_x = []
    avg_y = []
    avg_dy = []
    for i in range(1, len(bins)):
        mask = inds == i
        if np.any(mask) and np.sum(mask) >= min_points_per_bin:
            weights = 1 / dy[mask] ** 2
            # weighted mean for x and y
            wx = np.average(x[mask], weights=weights)
            wy = np.average(y[mask], weights=weights)
            wdy = 1 / np.sqrt(np.sum(weights))
            avg_x.append(wx)
            avg_y.append(wy)
            avg_dy.append(wdy)
    return np.array(avg_x), np.array(avg_y), np.array(avg_dy)

##########################################################################################
def plot_v2_fit(data_dict, star, line_spf=None, eq_text=False,
                datasets_to_plot=None, plot_ldmodel=False, plot_udmodel=False,
                to_bin=None, v0_flag=False, title=None, set_axis=None, uselatex=False, savefig=None, show=True):
    ###########################################################################
    # Function: plot_v2_fit                                                   #
    # Inputs: data_dict -> dict of InterferometryData objects,                #
    #                    e.g. {'pavo': pavo_obj, ...}                         #
    #         star-> star object with .theta and .ldc* attributes             #
    #                (ldcR, ldcK, etc.), and .V2(line_spf, theta, ldc)        #
    #         line_spf -> x values for model curve                            #
    #         eq_text -> optional string for annotation                       #
    #         datasets_to_plot-> list of keys in data_dict to plot            #
    #                            (default: all)                               #
    #         plot_ldmodel-> bool, whether to plot the ld model curve         #
    #         plot_udmodel -> bool, whether to plot the ud model curve        #
    #         to_bin -> list of kets in data_dict to bin                      #
    #         set_axis -> sets the axis limits                                #
    #         title -> allows user to set a plot title                        #
    #         savefig-> filename to save, if desired                          #
    #         show-> whether to plt.show()                                    #
    # Outputs: the plot                                                       #
    # What it does:                                                           #
    #        1. Initializes plotting parameters                               #
    #        2. Checks to see what datasets the user wants plotted            #
    #        3. Defines dictionaries for each instrument and the marker,      #
    #           color, label, and alpha value for each one                    #
    #        4. Checks to see if set_axis has been assigned. If it has,       #
    #           sets the axis limits and adjusts line_spf accordingly. If     #
    #           not, sets line_spf to be max(spf) with padding                #
    #        Starts with the top plot                                         #
    #        5. For each data set, sets the keys for the color, marker, label #
    #           and alpha value                                               #
    #        6. Checks to see if the to_bin has been set                      #
    #        7. if to_bin has been set, plots the unbinned data for each      #
    #           data set, and then bins the data sets indicated then plots    #
    #           those                                                         #
    #        8. If to_bin has not been set, it plots the unbinned data        #
    #        9. For the model, if plot_ldmodel is set, pulls the ldtheta,     #
    #           error on the theta, and the ldcs and calculates the fits for  #
    #           the relevant filter                                           #
    #       10. Plots the model for the filter indicated                      #
    #       11. If eq_text is set, annotates the plot with the theta val      #
    #       12. If plot_udmodel is set, pulls the udtheta and error, and      #
    #           calculates the UD model and plots it                          #
    #       13. If eq_text is set, annotates the plot the theta val           #
    #       14. Checks to see how many datasets are being plotted. If more    #
    #           than 1, sets the legend. If not, does not plot legend.        #
    #       Bottom plot:                                                      #
    #       15. For each data set, it pulls the respective keys for the       #
    #           color, label, alpha, and marker for each instrument           #
    #       16. For each data set, checks to see if the to_bin value has been #
    #           set.                                                          #
    #       17. If plot_ldmodel has been set, it calculates the residuals for #
    #           the data and the model for the filter indicated.              #
    #       18. If to_bin has been set, it calculates the residuals for the   #
    #           binned data as well.                                          #
    #       19. If plot_udmodel has been set, it calculates the residuals for #
    #           the data and the ud model.                                    #
    #       20. If to_bin has been set, it calcualtes the residuals for the   #
    #           binned data as well.                                          #
    #       21. Plots the residuals for the unbinned (and binned if set)      #
    #       22. Saves fig if save_fig has been set                            #
    #       23. Shows fig if show has been set.                               #
    ###########################################################################
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['text.usetex'] = uselatex
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 3]}, sharex=True)

    if datasets_to_plot is None:
        datasets_to_plot = list(data_dict.keys())

    color_map = {
        'pavo': '#02ccfe',
        'classic': '#738595',
        'vega': '#5edc1f',
        'mircx': '#efc8ff',
        'mystic': '#ffcfdc',
        'spica': '#ff964f'
    }
    binned_color_map = {
        'pavo': '#030aa7',
        'classic': '#000000',
        'vega': '#028f1e',
        'mircx': '#7e1e9c',
        'mystic': '#ff028d',
        'spica': '#fe4b03'
    }
    marker_map = {
        'pavo': '.',
        'classic': 's',
        'vega': '^',
        'mircx': 'v',
        'mystic': '*',
        'spica': 'D'
    }
    label_map = {
        'pavo': r'$\rm PAVO$',
        'classic': r'$\rm Classic$',
        'vega': r'$\rm VEGA$',
        'mircx': r'$\rm MIRCX$',
        'mystic': r'$\rm MYSTIC$',
        'spica': r'$\rm SPICA$'
    }
    binlabel_map = {
        'pavo': r'$\rm PAVO~(binned)$',
        'classic': r'$\rm Classic~(binned)$',
        'vega': r'$\rm VEGA~(binned)$',
        'mircx': r'$\rm MIRCX~(binned)$',
        'mystic': r'$\rm MYSTIC~(binned)$',
        'spica': r'$\rm SPICA~(binned)$'
    }
    alpha_map = {
        'pavo': 0.15,
        'classic': 0.5,
        'vega': 0.5,
        'mircx': 0.15,
        'mystic': 0.15,
        'spica': 0.15
    }

    band_map = {
        'mircx': 'H',
        'mystic': 'K',
        'classic': 'H',
        'pavo': 'R',
        'vega': 'R',
        'spica': 'R'
    }

    if set_axis and line_spf is None:
        xmin = set_axis[0]
        xmax = set_axis[1]
        ymin = set_axis[2]
        ymax = set_axis[3]
        a0.set_xlim(xmin, xmax)
        a0.set_ylim(ymin, ymax)
        line_spf = np.linspace(0.00001, xmax, 1000)
    else:
        all_spf = []
        ldcs = []
        for key in datasets_to_plot:
            data = data_dict[key]
            spf = np.array(data.B) / np.array(data.Wave)
            all_spf.extend(spf)

        all_spf = np.array(all_spf)
        min_spf = np.min(all_spf)
        max_spf = np.max(all_spf)
        line_spf = np.linspace(min_spf, max_spf, 1000)  # slight padding
    if v0_flag:
        fitted_v0s = star.ldv02_by_group
    # --- Top: V2 ---
    for key in datasets_to_plot:
        data = data_dict[key]
        data.V0s = np.ones(len(data.B))
        color = color_map.get(key, None)
        bin_color = binned_color_map.get(key, None)
        bin_label = binlabel_map.get(key, None)
        marker = marker_map.get(key, '.')
        label = label_map.get(key, key.capitalize())
        alpha = alpha_map.get(key, 0.5)
        spf = np.array(data.B) / np.array(data.Wave)

        if v0_flag:
            y_data = data.V2 / data.V0s
            assign_inst_night_v0s(fitted_v0s, data)
        else:
            y_data = data.V2

        is_binned = to_bin and key in to_bin
        # Always plot both, but only one gets the label
        if is_binned:
            # Plot unbinned points, no label
            a0.plot(spf, y_data, linestyle='None', marker=marker, markersize=3, color=color, alpha=alpha)
            a0.errorbar(spf, y_data, yerr=abs(data.dV2), fmt=marker, markersize=3, linestyle='None', linewidth=0.5,
                        color=color, capsize=3, alpha=alpha)
            # Plot binned points, with label
            binned_spf, binned_v2, binned_dv2 = bin_data(spf, y_data, data.dV2)
            a0.plot(binned_spf, binned_v2, linestyle='None', marker=marker, markersize=6, color=bin_color,
                    label=label)
            a0.errorbar(binned_spf, binned_v2, yerr=abs(binned_dv2), fmt=marker, linestyle='None', markersize=6,
                        color=bin_color,
                        capsize=3)
        else:
            # Plot unbinned points, with label
            a0.plot(spf, y_data, linestyle='None', marker=marker, markersize=6, color=color, alpha=alpha, label=label)
            a0.errorbar(spf, y_data, yerr=abs(data.dV2), fmt=marker, markersize=6, linestyle='None', linewidth=0.5,
                        color=color, capsize=3, alpha=alpha)
    # --- Model ---
    if plot_ldmodel:
        if not v0_flag:
            # ldc_value = np.ones(len(spf))*getattr(star, ldc_band, None)
            theta = getattr(star, "ldtheta", None)
            dtheta = getattr(star, "ldtheta_err", None)
            model_wl = 2.2e-6
            wl_plot = np.full_like(line_spf, model_wl)
            y_plot = V2_wrapper(star, wl_plot, line_spf)
            if theta is not None:
                model_label = create_ldc_label(star)
                a0.plot(line_spf, y_plot, '--', color='black', label=model_label)
                if eq_text:
                    eq1 = fr"$\theta_{{\rm LD}} = {round(theta, 3):.3f} \pm {round(dtheta, 3):.3f} \rm ~mas$"
                    a0.text(0.05, 0.05, eq1, transform=a0.transAxes, color='black', fontsize=15)

            else:
                print(f"Warning: ldtheta not present for star, skipping model plot.")
        if v0_flag:
            theta = getattr(star, "ldtheta", None)
            dtheta = getattr(star, "ldtheta_err", None)
            model_wl = 2.2e-6
            wl_plot = np.full_like(line_spf, model_wl)
            y_plot = V2_wrapper(star, wl_plot, line_spf)

            if theta is not None:
                model_label = create_ldc_label(star)
                a0.plot(line_spf, y_plot, '--', color='black', label=model_label)
                if eq_text:
                    eq1 = fr"$\theta_{{\rm LD}} = {round(theta, 3):.3f} \pm {round(dtheta, 3):.3f} \rm ~mas$"
                    a0.text(0.05, 0.05, eq1, transform=a0.transAxes, color='black', fontsize=15)

            else:
                print(f"Warning: ldtheta not present for star, skipping model plot.")

    if plot_udmodel:
        if not v0_flag:
            theta = getattr(star, "udtheta", None)
            dtheta = getattr(star, "udtheta_err", None)
            if theta is not None:
                model_label = fr"$\rm Uniform~Disk~Model$"
                a0.plot(line_spf, UDV2(line_spf, theta), '--', color='black', label=model_label)
                if eq_text:
                    eq1 = fr"$\theta_{{\rm UD}} = {round(theta, 3):.3f} \pm {round(dtheta, 3):.3f} \rm ~mas$"
                    a0.text(0.05, 0.05, eq1, transform=a0.transAxes, color='black', fontsize=15)
            else:
                print(f"Warning: udtheta not present for star, skipping model plot.")
        if v0_flag:
            theta = getattr(star, "udtheta", None)
            dtheta = getattr(star, "udtheta_err", None)
            if theta is not None:
                model_label = fr"$\rm Uniform~Disk~Model$"
                a0.plot(line_spf, UDV2(line_spf, theta), '--', color='black', label=model_label)
                if eq_text:
                    eq1 = fr"$\theta_{{\rm UD}} = {round(theta, 3):.3f} \pm {round(dtheta, 3):.3f} \rm ~mas$"
                    a0.text(0.05, 0.05, eq1, transform=a0.transAxes, color='black', fontsize=15)
            else:
                print(f"Warning: udtheta not present for star, skipping model plot.")

    if len(datasets_to_plot) > 1:
        a0.legend(fontsize=12, loc='upper right')

    a0.set_ylabel(r'$V^2$', labelpad=25)
    a0.tick_params(axis='x', labelbottom=False)
    a0.xaxis.set_minor_locator(AutoMinorLocator())
    a0.yaxis.set_minor_locator(AutoMinorLocator())
    a0.set_title(title)

    # --- Bottom panel: Residuals ---
    for key in datasets_to_plot:
        data = data_dict[key]
        color = color_map.get(key, None)
        bin_color = binned_color_map.get(key, None)
        marker = marker_map.get(key, '.')
        alpha = alpha_map.get(key, 0.5)
        spf = np.array(data.B) / np.array(data.Wave)
        if v0_flag:
            y_data = data.V2 / data.V0s
        else:
            y_data = data.V2
        is_binned = to_bin and key in to_bin  # e.g. to_bin = ['pavo']

        # --- Model and Residuals for Unbinned ---
        if plot_ldmodel and theta is not None:
            model_wl = 2.2e-6
            wl_plot = np.full_like(spf, model_wl)
            model_v2 = V2_wrapper(star, wl_plot, spf)
            # model_v2 = V2(spf, theta, ldc_value)
            residuals = np.array(y_data) - model_v2

            a1.plot(spf, residuals, linestyle='None', marker=marker, markersize=3, color=color, alpha=alpha)
            a1.errorbar(spf, residuals, yerr=abs(data.dV2), fmt=marker, markersize=3, linestyle='None', linewidth=0.5,
                        color=color,
                        capsize=3, alpha=alpha)

            # --- Model and Residuals for Binned ---
            if is_binned:
                binned_spf, binned_v2, binned_dv2 = bin_data(spf, y_data, data.dV2)
                model_wl = 2.2e-6
                binned_wv = np.full_like(binned_spf, model_wl)
                model_binv2 = V2_wrapper(star, binned_wv, binned_spf)
                # model_binv2 = V2_wrapper(sta
                binned_res = binned_v2 - model_binv2
                a1.plot(binned_spf, binned_res, linestyle='None', marker=marker, markersize=6, color=bin_color)
                a1.errorbar(binned_spf, binned_res, yerr=abs(binned_dv2), fmt=marker, linestyle='None', markersize=6,
                            color=bin_color, capsize=3)

        # --- (Repeat similar for UD model if desired) ---
        if plot_udmodel and theta is not None:
            model_udv2 = UDV2(spf, theta)
            ud_res = np.array(y_data) - model_udv2
            a1.plot(spf, ud_res, linestyle='None', marker=marker, markersize=3, color=color, alpha=alpha)
            a1.errorbar(spf, ud_res, yerr=abs(data.dV2), fmt=marker, markersize=3, linestyle='None', linewidth=0.5,
                        color=color, capsize=5, alpha=alpha)

            if is_binned:
                binned_spf, binned_v2, binned_dv2 = bin_data(spf, y_data, data.dV2)
                model_binudv2 = UDV2(binned_spf, theta)
                binned_udres = binned_v2 - model_binudv2

                a1.plot(binned_spf, binned_udres, linestyle='None', marker=marker, markersize=6, color=bin_color)
                a1.errorbar(binned_spf, binned_udres, yerr=abs(binned_dv2), fmt=marker, linestyle='None', markersize=6,
                            color=bin_color, capsize=3)

    a1.axhline(y=0, color='black', linestyle='--')
    a1.set_ylabel(r'$\rm Residual$', labelpad=3)
    plt.yticks([-.25, 0, 0.25])
    a1.set_ylim([-0.35, 0.35])
    a1.xaxis.set_minor_locator(AutoMinorLocator())
    a1.yaxis.set_minor_locator(AutoMinorLocator())

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xlabel(r'$\rm Spatial$ $\rm frequency$ [$\rm rad^{-1}$]')

    if savefig:
        f.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    return f, (a0, a1)