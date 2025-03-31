# example.py
"""
Demo script to showcase how to use the MassFunctionCalculator class 
and print some basic results without generating full plots.
"""

# Import halo_assembly (placeholder module) to ensure it's recognized
import src.halo_assembly  
from src.calculations import MassFunctionCalculator

def demo():
    """
    Demo function that creates a MassFunctionCalculator,
    computes values, and prints a subset of the results.
    """
    # Example cosmology parameters
    cosmology = {
        'h_0': 0.678,
        'O_m0': 0.307115,
        'O_b0': 0.048,
        'n': 0.96,
        'sigma_8': 0.823,
        'delta_c': 1.686
    }
    
    # Instantiate the calculator
    calc = MassFunctionCalculator(cosmology, z0=0)
    
    # Compute results with fewer points for a quicker demo
    results = calc.compute_values(num_samples=50)
    
    # Print the results for each key (e.g., '9', '9p5', '10', etc.)
    for key, res in results.items():
        print(f"For initial M* = 10^{key}:")
        print("  z array:", res['z'])
        print("  nvir array:", res['nvir'])
        print("  logMs array:", res['logMs'])
        print()

if __name__ == '__main__':
    # Entry point if running example.py directly
    demo()