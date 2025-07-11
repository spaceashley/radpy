import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from radpy.UDfitting import UDV2
from radpy.LDfitting import V2
plt.rcParams['text.usetex'] = True


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
def plot_v2_fit(data_dict, star, line_spf=None, ldc_band=None, eq_text=False,
                datasets_to_plot=None, plot_ldmodel=False, plot_udmodel=False,
                to_bin=None, title=None, set_axis = None, savefig=None, show=True):
    ###########################################################################
    # Function: plot_v2_fit                                                   #
    # Inputs: data_dict -> dict of InterferometryData objects,                #
    #                    e.g. {'pavo': pavo_obj, ...}                         #
    #         star-> star object with .theta and .ldc* attributes             #
    #                (ldcR, ldcK, etc.), and .V2(line_spf, theta, ldc)        #
    #         line_spf -> x values for model curve                            #
    #         ldc_band -> string (e.g. "ldcR", "ldcK") for which LDC          #
    #                     coefficient to use                                  #
    #         eq_text -> optional string for annotation                       #
    #         datasets_to_plot-> list of keys in data_dict to plot            #
    #                            (default: all)                               #
    #         plot_ldmodel-> bool, whether to plot the ld model curve         #
    #         plot_udmodel -> bool, whether to plot the ud model curve        #
    #         to_bin -> list of kets in data_dict to bin                      #
    #         title -> allows user to set a plot title                        #
    #         savefig-> filename to save, if desired                          #
    #         show-> whether to plt.show()                                    #
    # Outputs: the plot                                                       #
    # What it does:                                                           #
    #        1. Initializes plotting parameters                               #
    #        2. Checks to see what datasets the user wants plotted            #
    #        3. Defines dictionaries for each instrument and the marker,      #
    #           color, label, and alpha value for each one                    #
    #        Starts with the top plot                                         #
    #        4. for each data set, sets the keys for the color, marker, label #
    #           and alpha value                                               #
    #        5. Checks to see if the to_bin has been set                      #
    #        6. if to_bin has been set, plots the unbinned data for each      #
    #           data set, and then bins the data sets indicated then plots    #
    #           those                                                         #
    #        7. If to_bin has not been set, it plots the unbinned data        #
    #        8. For the model, if plot_ldmodel is set, pulls the ldtheta,     #
    #           error on the theta, and the ldcs and calculates the fits for  #
    #           the relevant filter                                           #
    #        9. Plots the model for the filter indicated                      #
    #       10. If eq_text is set, annotates the plot with the theta val      #
    #       11. If plot_udmodel is set, pulls the udtheta and error, and      #
    #           calculates the UD model and plots it                          #
    #       12. If eq_text is set, annotates the plot the theta val           #
    #       Bottom plot:                                                      #
    #       13. For each data set, it pulls the respective keys for the       #
    #           color, label, alpha, and marker for each instrument           #
    #       14. For each data set, checks to see if the to_bin value has been #
    #           set.                                                          #
    #       15. If plot_ldmodel has been set, it calculates the residuals for #
    #           the data and the model for the filter indicated.              #
    #       16. If to_bin has been set, it calculates the residuals for the   #
    #           binned data as well.                                          #
    #       17. If plot_udmodel has been set, it calculates the residuals for #
    #           the data and the ud model.                                    #
    #       18. If to_bin has been set, it calcualtes the residuals for the   #
    #           binned data as well.                                          #
    #       19. Plots the residuals for the unbinned (and binned if set)      #
    #       20. Saves fig if save_fig has been set                            #
    #       21. Shows fig if show has been set.                               #
    ###########################################################################
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 3]}, sharex=True)

    if datasets_to_plot is None:
        datasets_to_plot = list(data_dict.keys())

    color_map = {
        'pavo': '#02ccfe',
        'classic': '#738595',
        'vega': '#5edc1f',
        'mircx': '#efc8ff',
        'mystic': '#ff81c0',
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
        'mircx': 'o',
        'mystic': '*',
        'spica': 'D'
    }
    label_map = {
        'pavo': r'$\rm PAVO$',
        'classic': r'$\rm Classic$',
        'vega': r'$\rm VEGA$',
        'mircx': r'$\rm MIRC-X$',
        'mystic': r'$\rm MYSTIC$',
        'spica': r'$\rm SPICA$'
    }
    alpha_map = {
        'pavo': 0.25,
        'classic': 0.25,
        'vega': 0.25,
        'mircx': 0.25,
        'mystic': 0.25,
        'spica': 0.25
    }

    if line_spf is None:
        all_spf = []
        for key in datasets_to_plot:
            data = data_dict[key]
            spf = np.array(data.B) / np.array(data.Wave)
            all_spf.extend(spf)
        all_spf = np.array(all_spf)
        max_spf = np.max(all_spf)
        line_spf = np.linspace(0.00001, max_spf * 1.1, 1000)  # slight padding
    # --- Top: V2 ---
    for key in datasets_to_plot:
        data = data_dict[key]
        color = color_map.get(key, None)
        bin_color = binned_color_map.get(key, None)
        marker = marker_map.get(key, '.')
        label = label_map.get(key, key.capitalize())
        alpha = alpha_map.get(key, 0.5)
        spf = np.array(data.B) / np.array(data.Wave)

        is_binned = to_bin and key in to_bin
        # Always plot both, but only one gets the label
        if is_binned:
            # Plot unbinned points, no label
            a0.plot(spf, data.V2, linestyle='None', marker=marker, markersize=3, color=color, alpha=alpha)
            a0.errorbar(spf, data.V2, yerr=abs(data.dV2), fmt=marker, markersize=3, linestyle='None', linewidth=0.5,
                        color=color,
                        capsize=3, alpha=alpha)
            # Plot binned points, with label
            binned_spf, binned_v2, binned_dv2 = bin_data(spf, data.V2, data.dV2)
            a0.plot(binned_spf, binned_v2, linestyle='None', marker=marker, markersize=7, color=bin_color, label=label)
            a0.errorbar(binned_spf, binned_v2, yerr=abs(binned_dv2), fmt=marker, linestyle='None', markersize=7,
                        color=bin_color,
                        capsize=3)
        else:
            # Plot unbinned points, with label
            a0.plot(spf, data.V2, linestyle='None', marker=marker, markersize=3, color=color, alpha=alpha, label=label)
            a0.errorbar(spf, data.V2, yerr=abs(data.dV2), fmt=marker, markersize=3, linestyle='None', linewidth=0.5,
                        color=color,
                        capsize=3, alpha=alpha)
    # --- Model ---
    if plot_ldmodel:
        ldc_value = getattr(star, ldc_band, None)
        theta = getattr(star, "ldtheta", None)
        dtheta = getattr(star, "ldtheta_err", None)
        if ldc_value is not None and theta is not None:
            model_label = fr"$ \rm Model ({ldc_band.replace('ldc_', '').upper()})$"
            a0.plot(line_spf, V2(line_spf, theta, ldc_value), '--', color='black', label=model_label)
            if eq_text:
                eq1 = fr"$\theta_{{\rm LD}} = {round(theta, 3)} \pm {round(dtheta, 3)} \rm ~[mas]$"
                a0.text(0.05, 0.05, eq1, transform=a0.transAxes, color='black', fontsize=15)
        else:
            print(f"Warning: {ldc_band} or ldtheta not present for star, skipping model plot.")

    if plot_udmodel:
        theta = getattr(star, "udtheta", None)
        dtheta = getattr(star, "udtheta_err", None)
        if theta is not None:
            model_label = fr"$\rm Uniform~Disk~Model$"
            a0.plot(line_spf, UDV2(line_spf, theta), '--', color='black', label=model_label)
            if eq_text:
                eq1 = fr"$\theta_{{\rm UD}} = {round(theta, 3)} \pm {round(dtheta, 3)} \rm ~[mas]$"
                a0.text(0.05, 0.05, eq1, transform=a0.transAxes, color='black', fontsize=15)
        else:
            print(f"Warning: udtheta not present for star, skipping model plot.")

    a0.legend(fontsize=12)
    a0.set_ylabel(r'$V^2$', labelpad=17)
    a0.tick_params(axis='x', labelbottom=False)
    a0.xaxis.set_minor_locator(AutoMinorLocator())
    a0.yaxis.set_minor_locator(AutoMinorLocator())
    if set_axis:
        xmin = set_axis[0]
        xmax = set_axis[1]
        ymin = set_axis[2]
        ymax = set_axis[3]
        a0.set_xlim(xmin, xmax)
        a0.set_ylim(ymin, ymax)

    # --- Bottom panel: Residuals ---
    for key in datasets_to_plot:
        data = data_dict[key]
        color = color_map.get(key, None)
        bin_color = binned_color_map.get(key, None)
        marker = marker_map.get(key, '.')
        alpha = alpha_map.get(key, 0.5)
        spf = np.array(data.B) / np.array(data.Wave)

        is_binned = to_bin and key in to_bin  # e.g. to_bin = ['pavo']

        # --- Model and Residuals for Unbinned ---
        if plot_ldmodel and ldc_value is not None and theta is not None:
            model_v2 = V2(spf, theta, ldc_value)
            residuals = np.array(data.V2) - model_v2
            a1.plot(spf, residuals, linestyle='None', marker=marker, markersize=3, color=color, alpha=alpha)
            a1.errorbar(spf, residuals, yerr=abs(data.dV2), fmt=marker, markersize=3, linestyle='None', linewidth=0.5,
                        color=color,
                        capsize=3, alpha=alpha)

            # --- Model and Residuals for Binned ---
            if is_binned:
                model_binv2 = V2(binned_spf, theta, ldc_value)
                binned_res = binned_v2 - model_binv2
                a1.plot(binned_spf, binned_res, linestyle='None', marker=marker, markersize=7, color=bin_color)
                a1.errorbar(binned_spf, binned_res, yerr=abs(binned_dv2), fmt=marker, linestyle='None', markersize=7,
                            color=bin_color, capsize=3)

        # --- (Repeat similar for UD model if desired) ---
        if plot_udmodel and theta is not None:
            model_udv2 = UDV2(spf, theta)
            ud_res = np.array(data.V2) - model_udv2
            a1.plot(spf, ud_res, linestyle='None', marker=marker, markersize=3, color=color, alpha=alpha)
            a1.errorbar(spf, ud_res, yerr=abs(data.dV2), fmt=marker, markersize=3, linestyle='None', linewidth=0.5,
                        color=color, capsize=5,
                        alpha=alpha)

            if is_binned:
                model_binudv2 = UDV2(binned_spf, theta)
                binned_udres = binned_v2 - model_binudv2
                a1.plot(binned_spf, binned_udres, linestyle='None', marker=marker, markersize=7, color=bin_color)
                a1.errorbar(binned_spf, binned_udres, yerr=abs(binned_dv2), fmt=marker, linestyle='None', markersize=7,
                            color=bin_color, capsize=3)

    a1.axhline(y=0, color='black', linestyle='--')
    a1.set_ylabel(r'$\rm Residual$', labelpad=3)
    plt.yticks([-.25, 0, 0.25])
    a1.set_ylim([-0.35, 0.35])
    a1.xaxis.set_minor_locator(AutoMinorLocator())
    a1.yaxis.set_minor_locator(AutoMinorLocator())

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xlabel(r'$\rm Spatial$ $\rm frequency$ [$\rm rad^{-1}$]')
    a0.set_title(rf'$\rm {title}$')


    if savefig:
        f.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    return f, (a0, a1)