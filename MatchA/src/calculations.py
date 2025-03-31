# calculations.py
"""
This module contains the MassFunctionCalculator class, which handles
the main computations (SHAM, HMF, etc.).
"""

import numpy as np
from hmf import MassFunction
from scipy import interpolate
import scipy.optimize as opt
# Relative imports for halo_assembly and gsmf
from . import halo_assembly as hal
from . import gsmf as smf

class MassFunctionCalculator:
    """
    Encapsulates methods for computing halo masses via SHAM, 
    halo mass functions, and related quantities.
    """
    
    def __init__(self, cosmology, z0=0):
        """
        Initialize the calculator with a given cosmology and initial redshift.

        :param cosmology: dict containing cosmological parameters (h_0, O_m0, O_b0, etc.)
        :param z0: float, the initial redshift (default 0)
        """
        self.h_0 = cosmology['h_0']
        self.O_m0 = cosmology['O_m0']
        self.O_b0 = cosmology['O_b0']
        self.n = cosmology['n']
        self.sigma_8 = cosmology['sigma_8']
        self.z0 = z0
        
        # If the user provided 'delta_c', use it; otherwise default to 1.686
        delta_c = cosmology.get('delta_c', 1.686)
        
        # We assume O_l0 = 1 - O_m0 as a simplified model
        O_l0 = 1 - self.O_m0
        
        # Build a NumPy array representing some cosmological parameters
        self.Cosmology = np.array([0, self.O_m0, O_l0, self.O_b0, self.sigma_8, self.h_0, delta_c])

    def SHAM(self, z, n_gal):
        """
        SubHalo Abundance Matching function (placeholder logic).
        This method finds the halo mass corresponding to a given 
        number density of galaxies at redshift z.

        :param z: float, redshift
        :param n_gal: float, galaxy number density
        :return: float, log10 of the halo mass that matches n_gal
        """
        # Create a MassFunction object from the 'hmf' library
        mf = MassFunction(
            z=z,
            cosmo_params={
                "Om0": self.O_m0,
                "Ob0": self.O_b0,
                "Tcmb0": 2.725,
                "Neff": 3.05,
                "H0": 100.0 * self.h_0
            },
            n=self.n,
            sigma_8=self.sigma_8,
            Mmin=9,
            Mmax=16,
            dlog10m=0.01,
            transfer_model="EH",
            mdef_model="SOVirial",
            hmf_model="Behroozi"
        )
        
        # Total cumulative halo function
        nvir = hal.Total_cumulative_halo_function(np.log10(mf.m), mf.ngtm, z, self.h_0)
        
        # We interpolate log(nvir) -> log(mass)
        hmf_int = interpolate.interp1d(np.log10(nvir), np.log10(mf.m))
        
        # Return log10 of the mass that matches the desired n_gal
        return hmf_int(np.log10(n_gal))
    
    def hmf(self, z, logMvir, epsilon=1e-5):
        """
        Compute the halo number density nvir for a given log10 halo mass 
        using the 'hmf' library at a specified redshift.

        :param z: float, redshift
        :param logMvir: float, log10(Mvir)
        :param epsilon: float, small interval for Mmin/Mmax in hmf
        :return: float, nvir
        """
        mf = MassFunction(
            z=z,
            cosmo_params={
                "Om0": self.O_m0,
                "Ob0": self.O_b0,
                "Tcmb0": 2.725,
                "Neff": 3.05,
                "H0": 100.0 * self.h_0
            },
            n=self.n,
            sigma_8=self.sigma_8,
            Mmin=logMvir - epsilon,
            Mmax=logMvir + epsilon,
            dlog10m=epsilon / 2.0,
            transfer_model="EH",
            mdef_model="SOVirial",
            hmf_model="Behroozi"
        )
        
        nvir = hal.Total_cumulative_halo_function(np.log10(mf.m), mf.ngtm, z, self.h_0)
        hmf_int = interpolate.interp1d(np.log10(mf.m), np.log10(nvir))
        
        # Return 10^(interpolated value)
        return 10 ** hmf_int(logMvir)
    
    def func_solve(self, logMs, z, n_vir):
        """
        Helper function for SHAM_ste, comparing the halo number density 
        with integrated GSMF.

        :param logMs: float, log10(stellar mass)
        :param z: float, redshift
        :param n_vir: float, halo number density
        :return: float, difference between log(n_vir) and log(GSMF)
        """
        # Compare log n_vir vs log (integrated GSMF)
        return np.log10(n_vir) - np.log10(
            smf.integrate_phi_GSMF(logMs, z, "deconvolved_including_halo_dispersion")
        )
    
    def SHAM_ste(self, z, n_vir):
        """
        Bisect function to find logMs such that the integrated GSMF 
        matches the given halo number density.

        :param z: float, redshift
        :param n_vir: float, halo number density
        :return: float, log10(stellar mass)
        """
        return opt.bisect(
            lambda logMs: self.func_solve(logMs, z, n_vir),
            a=1,
            b=12.5
        )
    
    def compute_values(self, num_samples=100):
        """
        High-level method that:
          1) Computes galaxy number densities for various logMs thresholds,
          2) Computes halo masses via SHAM for z=0,
          3) Builds an array of redshifts,
          4) Finds the progenitors for each initial halo mass,
          5) Computes corresponding nvir and logMs at those redshifts.

        :param num_samples: int, number of points in the z array
        :return: dict containing computed results for each key ('9', '9p5', '10', etc.)
        """
        # 1) Compute galaxy densities at z0
        ngal_values = {
            '9': smf.integrate_phi_GSMF(9,   self.z0, "deconvolved_including_halo_dispersion"),
            '9p5': smf.integrate_phi_GSMF(9.5, self.z0, "deconvolved_including_halo_dispersion"),
            '10': smf.integrate_phi_GSMF(10,  self.z0, "deconvolved_including_halo_dispersion"),
            '10p5': smf.integrate_phi_GSMF(10.5, self.z0, "deconvolved_including_halo_dispersion"),
            '11': smf.integrate_phi_GSMF(11,  self.z0, "deconvolved_including_halo_dispersion"),
            '11p5': smf.integrate_phi_GSMF(11.5, self.z0, "deconvolved_including_halo_dispersion")
        }
        
        # 2) Compute the initial logMvir for each threshold
        initial_logMvir = {
            key: self.SHAM(self.z0, ngal) for key, ngal in ngal_values.items()
        }
        
        # 3) Build an array of redshifts (from z0 to something higher)
        #    Here we logspace from (z0+1) to 12, then subtract 1
        z_array = np.logspace(np.log10(self.z0 + 1), np.log10(12), num_samples) - 1
        
        # 4) For each key, compute the progenitors and the associated nvir, logMs
        results = {}
        for key, logMvir0 in initial_logMvir.items():
            # median_log10Mvir_progenitors is a placeholder function in halo_assembly
            prog = hal.median_log10Mvir_progenitors(
                logMvir0, self.z0, z_array, self.Cosmology
            )
            
            # Compute nvir for each redshift in z_array
            nvir = np.array([self.hmf(z, prog[i]) for i, z in enumerate(z_array)])
            
            # Use SHAM_ste to find the corresponding logMs
            logMs = np.array([self.SHAM_ste(z, nvir[i]) for i, z in enumerate(z_array)])
            
            results[key] = {
                'z': z_array,
                'nvir': nvir,
                'logMs': logMs,
                'prog': prog
            }
        
        return results