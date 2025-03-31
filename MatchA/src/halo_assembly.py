# halo_assembly.py
"""
Module containing functions related to halo assembly, progenitor mass evolution,
and subhalo corrections. This streamlined version maintains core functionality 
(median_log10Mvir_progenitors, Total_cumulative_halo_function, etc.) while removing 
redundancies or purely unused code. Comments are in English.
"""

import numpy as np

############################
# CONSTANTS
############################
H0    = 100.0   # Hubble constant in km/s/Mpc (if needed for dimension checks)
h_BP  = 0.678   # Baseline h parameter for reference scaling
logM13 = 13.0   # log10(1e13 Msun) used in some empirical fits

############################
# BASIC COSMOLOGY / HELPER FUNCTIONS
############################

def Om_m(Om_mat, Om_lambda, z):
    """
    Computes Omega_m(z) = Om_mat*(1+z)^3 / [Om_lambda + Om_mat*(1+z)^3].
    """
    return Om_mat * (1. + z)**3 / (Om_lambda + Om_mat * (1. + z)**3)

def Om_l(Om_mat, Om_lambda, z):
    """
    Computes Omega_lambda(z) = Om_lambda / [Om_lambda + Om_mat*(1+z)^3].
    """
    return Om_lambda / (Om_lambda + Om_mat * (1. + z)**3)

def g_factor(Om_m_val, Om_l_val, z):
    """
    Helper function for D_gfactor. Empirical expression 
    depending on Omega_m(z) and redshift.
    """
    # 'Amplitude' depends on Om_m_val and z
    Amplitude = Om_m_val / (1. + z)
    # Denominator is a known fitting function used in some halo growth models
    return Amplitude / (
        Om_m_val**0.571428571
        - Om_l_val
        + (1. + Om_m_val * 0.5) * (1. + Om_l_val * 0.014285714)
    )

def D_gfactor(Om_mat, Om_lambda, z):
    """
    Returns D(z)/D(0), the ratio of growth factors between redshift z and z=0.
    """
    Om_m_z = Om_m(Om_mat, Om_lambda, z)
    Om_l_z = Om_l(Om_mat, Om_lambda, z)
    return g_factor(Om_m_z, Om_l_z, z) / g_factor(Om_mat, Om_lambda, 0)

def scale_factor(z):
    """
    Simple function: a(z) = 1/(1+z).
    Might be used elsewhere to convert redshifts to scale factors.
    """
    return 1.0 / (1.0 + z)

############################
# FITTING FUNCTIONS FOR PROGENITOR EVOLUTION
############################

def f_norm(x, logMvir0, dw):
    """
    Part of the fitting function in median_log10Mvir_progenitors.
    (1+dw)^alpha * (1+0.5*dw)^beta * exp(gamma*dw).
    """
    alpha = x[1]
    beta  = x[2]
    gamma = x[3]
    return (1. + dw)**alpha * (1. + 0.5*dw)**beta * np.exp(gamma * dw)

def a0_func(x, logMvir0, scale):
    """
    Helper for f_func. 
    x[4], x[5], x[6], x[7] are fitting parameters.
    """
    ratio = x[5] - logMvir0
    return x[4] - np.log10(10**(x[6] * ratio) + 1.)

def g_func(x, logMvir0, dw):
    """
    Another piece of the empirical fit for f_func.
    """
    sc = 1.0 / (1.0 + dw)  # scale factor ~ a(z)
    d_scale = sc - a0_func(x, logMvir0, sc)
    return 1. + np.exp(-x[7] * d_scale)

def f_func(x, logMvir0, dw):
    """
    Ties together f_norm and g_func to form part of the 
    median_log10Mvir_progenitors formula.
    """
    return (logMvir0 - logM13) * g_func(x, logMvir0, 0.) / g_func(x, logMvir0, dw)

############################
# MAIN FUNCTION: PROGENITOR MASSES
############################

def median_log10Mvir_progenitors(log10Mh0, z0, z, Cosmology):
    """
    Computes the median log10(Mvir) of halo progenitors for a given halo at z0,
    across an array of redshifts z. Uses an empirical fitting approach.

    :param log10Mh0: float, log10 of Mvir at redshift z0
    :param z0: float, the reference redshift
    :param z: array-like, the redshifts where we want the progenitor mass
    :param Cosmology: array of cosmological parameters: 
                      [0, Om_mat, Om_lambda, Ob0, sigma8, h_0, delta_c]
    :return: array of log10(Mvir) at each z
    """
    # Example array of fitting parameters (taken from some reference or prior calibration)
    x = np.array([0., 1.52947, -3.4087, -0.404274, 
                  0.285509, 11.9943, 0.143375, 4.07574])
    
    Om_mat = Cosmology[1]
    Om_lambda = Cosmology[2]
    h = Cosmology[5]
    delta_c = Cosmology[6]  # might be used as part of dw

    # Adjust log10Mh0 by the ratio (h/h_BP) if needed
    log10Mh0_adj = log10Mh0 + np.log10(h / h_BP)

    # Growth factor difference from z to z0
    dw_array = delta_c / D_gfactor(Om_mat, Om_lambda, z) - \
               delta_c / D_gfactor(Om_mat, Om_lambda, z0)

    # Combine f_norm and f_func
    first_term = np.log10(f_norm(x, log10Mh0_adj, dw_array))
    second_term = f_func(x, log10Mh0_adj, dw_array)
    Mvirz_array = logM13 + first_term + second_term

    # Re-adjust for h
    return Mvirz_array - np.log10(h / h_BP)

############################
# SUBHALO CORRECTIONS
############################

def subhalos_correction_factor(logMpeak, z, h):
    """
    Empirical subhalo correction factor for total cumulative halo function.
    :param logMpeak: float, log10(peak mass)
    :param z: float, redshift
    :param h: float, Hubble parameter (like 0.678)
    :return: float, factor representing subhalo contribution
    """
    logMpeak_h = logMpeak + np.log10(h)
    z2 = z * z
    
    # Example expansions
    Csub_z_over_Csub_0 = (0.008670 * z 
                          - 0.011330 * z2 
                          - 0.003892 * z2 * z 
                          + 0.000370 * z2 * z2)
    Normalization = 1.78 * 10**(Csub_z_over_Csub_0)

    logMcut_off = (11.904572 
                   - 0.636422*z 
                   - 0.020686*z2 
                   + 0.022034*z*z2 
                   - 0.001151*(z2**2))

    ratio = logMpeak_h - logMcut_off
    return Normalization * np.exp(-10**(0.220586 * ratio))

def Total_cumulative_halo_function(logMpeak, n_vir, z, h):
    """
    Applies the subhalo correction factor to the base halo number density.
    :param logMpeak: float or array, log10(peak mass)
    :param n_vir: float or array, base halo number density
    :param z: float, redshift
    :param h: float, Hubble parameter
    :return: float or array, corrected halo number density
    """
    correction = subhalos_correction_factor(logMpeak, z, h)
    return n_vir * (1. + correction)

############################
# OPTIONAL / LEGACY FUNCTIONS
############################

# def function():
#     """
#     ...
#     """
#     pass