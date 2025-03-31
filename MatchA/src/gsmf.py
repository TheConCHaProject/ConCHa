# gsmf.py
"""
Galaxy Stellar Mass Function (GSMF) module.

This version preserves all essential functions from the original code:
 - Z_func_RP20: parameter evolution fit
 - Star-forming (SF) galaxy functions: log_phi_SF, alpha_SF, beta_SF, etc.
 - Quiescent (Q) galaxy functions: log_phi_1_Q, log_phi_2_Q, etc.
 - Combined GSMF: phi_GSMF_SF and phi_GSMF_Q
 - High-level entry points: phi_GSMF() and integrate_phi_GSMF()

"""

import numpy as np
from . import halo_assembly as hal
from scipy.integrate import quad

###############################################################################
# 1) Helper function for parameter evolution
###############################################################################
def Z_func_RP20(p0, p1, p2, p3, z):
    """
    Computes a redshift-dependent function used in various GSMF parameters.
    It depends on a scale factor a(z) = 1 / (1+z).

    :param p0, p1, p2, p3: float, polynomial or log-based coefficients
    :param z: float, redshift
    :return: float, evaluated function value
    """
    return (
        p0
        + p1 * (1.0 - hal.scale_factor(z))
        + p2 * np.log10(hal.scale_factor(z))
        + p3 * z
    )

###############################################################################
# 2) Star-Forming Galaxy Parameter Functions
#    These functions define the Schechter-like parameters (phi, alpha, beta, etc.)
#    for star-forming galaxies, including a second population (SF2) if needed.
###############################################################################

def log_phi_SF(x, z):
    """
    Logarithm of the normalization (phi*) for the first SF population.

    :param x: array-like, containing fit parameters
    :param z: float, redshift
    :return: float, log10(phi_s)
    """
    phi_s = Z_func_RP20(x[1], x[2], x[3], x[4], z)
    return phi_s

def log_phi_SF2(x, z):
    """
    Logarithm of the normalization (phi*) for the second SF population.

    :param x: array-like, containing fit parameters (includes x[20])
    :param z: float, redshift
    :return: float, log10(phi_s) offset by x[20]
    """
    return log_phi_SF(x, z) + x[20]

def alpha_SF(x, z):
    """
    Low-mass slope (alpha) of the SF GSMF.

    :param x: array-like, containing fit parameters
    :param z: float, redshift
    :return: float, alpha_s
    """
    alpha_s = Z_func_RP20(x[5], x[6], 0.0, x[7], z)
    return alpha_s

def alpha_SF2(x, z):
    """
    Variation of alpha for the second SF population.

    :param x: array-like, containing fit parameters
    :param z: float, redshift
    :return: float, alpha_s + 1
    """
    return alpha_SF(x, z) + 1.0

def beta_SF(x, z):
    """
    High-mass cutoff (beta) for the SF GSMF.

    :param x: array-like, containing fit parameters
    :param z: float, redshift
    :return: float, beta_s
    """
    beta_s = Z_func_RP20(x[8], 0.0, 0.0, 0.0, z)
    return beta_s

def log10Mchar_SF(x, z):
    """
    Characteristic mass, log10(Mchar), for the SF GSMF.

    :param x: array-like, containing fit parameters
    :param z: float, redshift
    :return: float, log10(Mchar)
    """
    Mchar = Z_func_RP20(x[9], x[10], x[11], x[12], z)
    return Mchar

###############################################################################
# 3) Quiescent Galaxy Parameter Functions
#    Similar approach as SF but for Q populations. 
#    Some 'Q' parameters are built on the SF param logic for consistency.
###############################################################################

def log_phi_1_Q(x, z):
    """
    log10(phi*) for the first Q population. 
    It builds on log_phi_2_Q plus an offset from Z_func_RP20(x[13], -2, 0, 0, z).
    """
    phi_s = log_phi_2_Q(x, z) + Z_func_RP20(x[13], -2.0, 0.0, 0.0, z)
    return phi_s

def alpha_1_Q(x, z):
    """
    Low-mass slope alpha for the first Q population, 
    uses the same slope as alpha_SF.
    """
    return alpha_SF(x, z)

def beta_1_Q(x, z):
    """
    High-mass cutoff for the first Q population, 
    reusing beta_SF logic or parameters.
    """
    return beta_SF(x, z)

def log10Mchar_1_Q(x, z):
    """
    Characteristic mass for the first Q population, 
    reusing log10Mchar_SF logic.
    """
    return log10Mchar_SF(x, z)

def log_phi_2_Q(x, z):
    """
    log10(phi*) for the second Q population. 
    Uses Z_func_RP20(x[14], x[15], x[16], x[17], z).
    """
    phi_s = Z_func_RP20(x[14], x[15], x[16], x[17], z)
    return phi_s

def alpha_2_Q(x, z):
    """
    Low-mass slope for the second Q population, 
    again referencing alpha_SF but with an additional factor.
    """
    alpha_s = alpha_SF(x, z)
    return alpha_s + 2.0 - hal.scale_factor(z)

def beta_2_Q(x, z):
    """
    High-mass cutoff for the second Q population, reusing beta_SF.
    """
    return beta_SF(x, z)

def log10Mchar_2_Q(x, z):
    """
    Characteristic mass for the second Q population, 
    based on log10Mchar_SF plus an extra Z_func_RP20 term.
    """
    Mchar = log10Mchar_SF(x, z) + Z_func_RP20(x[18], x[19], 0.0, 0.0, z)
    return Mchar

def log_phi_3_Q(x, z):
    """
    log10(phi*) for a third Q population, offset from log_phi_2_Q by x[21].
    """
    return log_phi_2_Q(x, z) + x[21]

def beta_3_Q(x, z):
    """
    High-mass cutoff for a third Q population, offset from beta_2_Q by x[22].
    """
    beta_s = beta_2_Q(x, z) + x[22]
    return beta_s

###############################################################################
# 4) Core Schechter Function
###############################################################################

def generalized_schechter_function(phi, alpha, beta, log10Mchar, log10Ms):
    """
    Computes the number density at log10Ms given a generalized Schechter function:
      phi_star, alpha, beta, and characteristic mass log10Mchar.

    :param phi: float, log10(phi*) 
    :param alpha: float, slope at low mass
    :param beta: float, modifies the exponential cutoff at high mass
    :param log10Mchar: float, log10 of the characteristic mass
    :param log10Ms: float, log10 of the stellar mass
    :return: float, the number density [Mpc^-3 dex^-1], in linear space
    """
    x_ratio = log10Ms - log10Mchar
    base_change = np.log10(np.exp(1.0))  # factor to convert ln->log10

    # phi_star = phi + (alpha+1)*x_ratio - 10^(beta*x_ratio)*base_change - log10(base_change)
    phi_star = (
        phi
        + (alpha + 1.0) * x_ratio
        - 10.0 ** (beta * x_ratio) * base_change
        - np.log10(base_change)
    )
    return 10.0 ** phi_star

###############################################################################
# 5) Star-Forming and Quiescent GSMF Calculation
###############################################################################

def phi_GSMF_SF(x, logMs, z):
    """
    Computes the total star-forming (SF) GSMF at logMs, z, 
    by summing two generalized Schechter components (SF1 and SF2).

    :param x: array-like, fit parameters
    :param logMs: float, log10(stellar mass)
    :param z: float, redshift
    :return: float, combined SF number density
    """
    # First SF component
    phi_s1  = log_phi_SF(x, z)
    alpha_s1 = alpha_SF(x, z)
    beta_s1  = beta_SF(x, z)
    logMc1   = log10Mchar_SF(x, z)

    # Second SF component
    phi_s2   = log_phi_SF2(x, z)
    alpha_s2 = alpha_SF2(x, z)
    beta_s2  = beta_SF(x, z)
    logMc2   = log10Mchar_SF(x, z)

    # Evaluate each Schechter function
    phi_SF_1 = generalized_schechter_function(phi_s1, alpha_s1, beta_s1, logMc1, logMs)
    phi_SF_2 = generalized_schechter_function(phi_s2, alpha_s2, beta_s2, logMc2, logMs)

    return phi_SF_1 + phi_SF_2

def phi_GSMF_Q(x, logMs, z):
    """
    Computes the total quiescent (Q) GSMF at logMs, z,
    by summing up to three generalized Schechter components (Q1, Q2, Q3).

    :param x: array-like, fit parameters
    :param logMs: float, log10(stellar mass)
    :param z: float, redshift
    :return: float, combined Q number density
    """
    # Q1
    phi_1   = log_phi_1_Q(x, z)
    alpha_1 = alpha_1_Q(x, z)
    beta_1  = beta_1_Q(x, z)
    logMc1  = log10Mchar_1_Q(x, z)

    # Q2
    phi_2   = log_phi_2_Q(x, z)
    alpha_2 = alpha_2_Q(x, z)
    beta_2  = beta_2_Q(x, z)
    logMc2  = log10Mchar_2_Q(x, z)

    # Q3
    phi_3   = log_phi_3_Q(x, z)
    alpha_3 = alpha_2_Q(x, z)  # same alpha as Q2?
    beta_3  = beta_3_Q(x, z)
    logMc3  = log10Mchar_2_Q(x, z)  # same Mchar as Q2?

    # Evaluate each Schechter function
    phi_Q_1 = generalized_schechter_function(phi_1, alpha_1, beta_1, logMc1, logMs)
    phi_Q_2 = generalized_schechter_function(phi_2, alpha_2, beta_2, logMc2, logMs)
    phi_Q_3 = generalized_schechter_function(phi_3, alpha_3, beta_3, logMc3, logMs)

    return phi_Q_1 + phi_Q_2 + phi_Q_3

###############################################################################
# 6) High-Level GSMF Interface
###############################################################################

def phi_GSMF(logMs, z, choose_mode):
    """
    High-level function to compute the total GSMF (SF + Q) at logMs, z,
    for a chosen mode. Three modes available: "observed_smf", "true_smf", 
    "intrinsic_smf". Each mode sets a 
    specific array of parameters 'param' to feed into phi_GSMF_SF and phi_GSMF_Q.

    :param logMs: float, log10(stellar mass)
    :param z: float, redshift
    :param choose_mode: str, one of {"observed_smf", "true_smf", 
                                     "intrinsic_smf"}
    :return: float, total GSMF (SF + Q) at logMs, z
    """
    if choose_mode == "observed_smf":
        param = np.array([
            0, -2.97903, 0.711457, 2.13684, -0.143942,
            -1.43664, -0.182969, -0.0652577, 0.924276, 10.3495,
            -0.852267, -3.43822, -0.294256, -0.687046, -2.64856,
            -0.22453, -2.01024, -0.9555, 0.469221, -1.03017,
            0.386681, -0.782124, -0.292956
        ])
    elif choose_mode == "true_smf":
        param = np.array([
            0, -3.12587, 1.0149, 2.96305, 0.0322943,
            -1.5068, -0.0962951, -0.0678136, 0.984965, 10.4309,
            -0.936285, -3.60355, -0.439795, -0.789231, -2.6914,
            -0.0222751, -1.59791, -0.877709, 0.431888, -0.762424,
            0.531231, -0.748871, -0.326509
        ])
    elif choose_mode == "intrinsic_smf":
        param = np.array([
            0, -3.13858, 0.776621, 2.46962, 0.0151814,
            -1.50836, -0.0840718, -0.0680439, 1.01977, 10.4759,
            -0.936793, -3.48557, -0.447515, -0.861007, -2.65967,
            -0.0227114, -1.49839, -0.849498, 0.351052, -0.582103,
            0.502357, -0.876961, -0.352607
        ])
    else:
        raise ValueError(
            "No valid mode selected; choose between 'observed_smf', "
            "'true_smf', or 'intrinsic_smf'."
        )

    # Sum quiescent + star-forming
    return phi_GSMF_Q(param, logMs, z) + phi_GSMF_SF(param, logMs, z)

def integrate_phi_GSMF(logM_i, z, choose_mode):
    """
    Integrates the GSMF from logM_i to 13 (i.e., from 10^logM_i up to 10^13 Msun).

    :param logM_i: float, lower bound for integration in log10(M)
    :param z: float, redshift
    :param choose_mode: str, one of the three GSMF modes
    :return: float, integrated number density (units of Mpc^-3)
    """
    # We integrate phi_GSMF(logMs) d(logMs) from logM_i to 13
    result, error = quad(lambda logMs: phi_GSMF(logMs, z, choose_mode),
                         logM_i, 13)
    return result