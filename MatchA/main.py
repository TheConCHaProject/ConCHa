# main.py
"""
Main script of the application. It sets up the cosmology, creates a 
MassFunctionCalculator for mass functions, obtains results, and
then calls the new plotting function with user-defined 'keys' and colors.
"""

from src.calculations import MassFunctionCalculator
from src.plotting import plot_results

def main():
    """
    Main function to execute the overall application.
    """
    # Example cosmology dictionary with necessary parameters
    cosmology = {
        'h_0': 0.678,
        'O_m0': 0.307115,
        'O_b0': 0.048,
        'n': 0.96,
        'sigma_8': 0.823,
        'delta_c': 1.686
    }

    # Create an instance of the calculator, specifying the initial redshift z0=0
    calculator = MassFunctionCalculator(cosmology, z0=0)

    # Compute the halo/gsmf values (here we choose 100 sample points)
    results = calculator.compute_values(num_samples=100)

    # -------------------------------------------------------------------------
    # Define which keys to plot. You can freely modify this list 
    # to include any subset: e.g., ['9','9p5','10','10p5','11','11p5']
    # -------------------------------------------------------------------------
    keys_to_plot = ['9', '9p5', '10', '10p5', '11', '11p5']

    # -------------------------------------------------------------------------
    # Option 1: Use a list of user-defined colors (same length as keys).
    #           Each color can be an RGB tuple or a hex string (e.g., "#RRGGBB")
    # e.g., user_colors = [(0.2, 0.4, 0.8), "#FF00FF", ...]
    # 
    # Option 2: Pass None or skip the parameter to let the function sample
    #           from a matplotlib colormap.
    # -------------------------------------------------------------------------
    user_colors = None  # or a list like: ["#FF0000","#00FF00","#0000FF","#FFFF00","#FF00FF","#00FFFF"]

    # -------------------------------------------------------------------------
    # Call the updated plot_results function
    # -------------------------------------------------------------------------
    plot_results(results,
                 keys=keys_to_plot,
                 data_filename='sham_lognormal_distributions.dat',
                 user_colors=user_colors,
                 colormap='coolwarm')

if __name__ == '__main__':
    main()