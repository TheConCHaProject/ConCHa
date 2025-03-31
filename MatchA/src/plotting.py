# plotting.py
"""
This module creates two plots:
  1) The cumulative halo number density (nvir) vs. (1+z)
  2) The evolution of stellar mass vs. (1+z)

Now, if a key contains 'p', we replace 'p' with '.' in the legend text.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

def plot_results(results,
                 keys,
                 data_filename='sham_lognormal_distributions.dat',
                 user_colors=None,
                 colormap='viridis'):
    """
    Generates two plots:
      1) The cumulative halo number density (nvir) vs. (1+z)
      2) The evolution of stellar mass vs. (1+z)

    :param results: dict of computed results (e.g., keys like '9', '9p5', etc.)
    :param keys: list of string keys to plot (e.g., ['9','9p5','10'])
    :param data_filename: str, path to the data file with additional reference data
    :param user_colors: list of colors (e.g., [(R,G,B), ...] or ["#RRGGBB", ...]),
                        same length as 'keys', optional
    :param colormap: str, name of a matplotlib colormap (used if user_colors is None)
    """

    # -------------------------------------------------------------------------
    # 1) Set up the style for LaTeX, font, axes thickness, and ticks
    # -------------------------------------------------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=35)
    plt.rc('axes', linewidth=3)
    
    # -------------------------------------------------------------------------
    # 2) Load data from file
    # -------------------------------------------------------------------------
    data = np.genfromtxt(data_filename, names=True)
    
    # -------------------------------------------------------------------------
    # 3) If user_colors is provided, it must match the length of keys.
    #    Otherwise, we sample from a colormap.
    # -------------------------------------------------------------------------
    if user_colors is not None:
        if len(user_colors) != len(keys):
            raise ValueError("Length of 'user_colors' must match length of 'keys'.")
        color_list = user_colors
    else:
        cmap = plt.get_cmap(colormap, len(keys))
        color_list = [cmap(i) for i in range(len(keys))]
    
    # Map each key to a color
    key_color_map = dict(zip(keys, color_list))
    
    # -------------------------------------------------------------------------
    # 4) Create a dictionary of data rows for each key
    # -------------------------------------------------------------------------
    data_dict = {}
    for key in keys:
        float_val = float(key.replace('p', '.'))
        data_dict[key] = data[data['logMs_0'] == float_val]
    
    # -------------------------------------------------------------------------
    # 5) Create the figure and subplots (1 row, 2 columns)
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    ax1, ax2 = axs

    for ax in axs:
        ax.tick_params(which="major", length=10, width=3, direction="out")
        ax.tick_params(which="minor", length=5, width=3, direction="out")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # -------------------------------------------------------------------------
    # 6) First plot: cumulative halo density (nvir) vs. (1 + z)
    # -------------------------------------------------------------------------
    for key in keys:
        res = results[key]
        ax1.plot(1. + res['z'], res['nvir'], 
                 color=key_color_map[key], ls='-', linewidth=4, zorder=2)
        
        ax1.plot(1. + data_dict[key]['z'], data_dict[key]['ngal_eval_logMs_prog_deconv_gsmf'], 
                 color=key_color_map[key], ls=':', linewidth=4, zorder=2)
    
    # Example reference lines
    first_key = keys[0]
    M_ref = data_dict[first_key]
    ax1.plot(10 ** M_ref['logMs_prog_deconv_gsmf'], 
             10 ** M_ref['logMs_prog_deconv_gsmf'],
             color='k', ls='-', linewidth=4, zorder=2,
             label=r'${\rm Evolving\ halo\ cumulative\ number\ density}$')
    ax1.plot(10 ** M_ref['logMs_prog_deconv_gsmf'], 
             10 ** M_ref['logMs_prog_deconv_gsmf'],
             color='k', ls=':', linewidth=4, zorder=2,
             label=r'${\rm Accounting\ for\ random\ errors\ in\ the\ observed\ GSMF}$')

    ax1.set_ylabel(r'$n_{\rm vir} \; [\mathrm{Mpc}^{-3}]$', fontsize=35)
    ax1.set_xlabel(r'$1+z$', fontsize=35)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(loc='upper left', fontsize=20, frameon=False)
    ax1.axis([1, 11, 1E-5, 1])
    
    x_ticks = np.arange(1, 12)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'${int(x)}$' for x in x_ticks], fontsize=35)
    
    # -------------------------------------------------------------------------
    # 7) Second plot: evolution of stellar mass (logMs) vs. (1 + z)
    # -------------------------------------------------------------------------
    for key in keys:
        res = results[key]
        # Replace 'p' with '.' in the legend exponent
        key_label = key.replace('p', '.')
        ax2.plot(1. + res['z'], 10 ** res['logMs'], 
                 color=key_color_map[key], ls='-', linewidth=4, zorder=2,
                 label=rf'$M_{{\ast}} = 10^{{{key_label}}}\,\mathrm{{M}}_\odot$')
        
        ax2.plot(1. + data_dict[key]['z'], 10 ** data_dict[key]['logMs_prog_deconv_gsmf'],
                 color=key_color_map[key], ls=':', linewidth=4, zorder=2)
    
    ax2.set_ylabel(r'$M_{\ast}(z) \; [\mathrm{M}_{\odot}]$', fontsize=35)
    ax2.set_xlabel(r'$1+z$', fontsize=35)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc='lower left', fontsize=20, frameon=False)
    ax2.axis([1, 11, 1E5, 8E11])
    
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'${int(x)}$' for x in x_ticks], fontsize=35)
    
    # -------------------------------------------------------------------------
    # 8) Tight layout and show
    # -------------------------------------------------------------------------
    fig.tight_layout()
    plt.show()