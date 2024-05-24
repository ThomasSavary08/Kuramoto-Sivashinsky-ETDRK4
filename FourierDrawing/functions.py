# Libraries
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from coefficients import coeffs_fft, order_coeffs

# Abstract class for dynamical systems
class DrawableFunction:

    def __init__(self, name: str, f: Callable[[float], complex], T: float):
        '''
        Instantiation of a drawable function.
            Parameters:
                name (str): Name of the function to draw.
                f (function): Periodic parametric function f such as f(t) = z.
                T (float): Period of the function.
        '''
        self.name = name
        self.f = f
        self.T = T

    def get_coefficients(self, K: int, ordered: bool = True):
        '''
        Get coefficients of the Fourier expansion.
            Parameters:
                K (int): Number of coefficients to compute (2K+1).
                ordered (bool): [c_0, c_(-1), c_1, ..., c_(-K), c_K] is True [c_(-K), ..., c_0, ..., c_K] otherwise.
            Returns:
                Coefficients of the Fourier expansion.
         '''
        times = np.linspace(0., self.T, 2*K+1, endpoint = False)
        f_t = np.asarray([self.f(t) for t in times])
        if ordered:
            return order_coeffs(coeffs_fft(K, f_t), K)
        else:
            return coeffs_fft(K, f_t)
        
    def plot_function(self, n: int):
        '''
        Plot the parametric function on one period.
            Parameters:
                n (int): Number of points to plot the function.
        '''
        times = np.linspace(0., self.T, n, endpoint = False)
        f_t = np.asarray([self.f(t) for t in times])
        plt.figure(figsize = (10,6))
        plt.plot(np.real(f_t), np.imag(f_t), 'ro', label = r'$f(t)$')
        plt.title("{} function".format(self.name))
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.legend()
        plt.show()

# Fish function
name = "Fish"
f = lambda t: complex(np.cos(t) + np.sqrt(8)*np.cos(t/2.), np.sin(t))
T = 4*np.pi
fish_function = DrawableFunction(name, f, T)

# Hearth function
name = "Heart"
f = lambda t: complex(np.cos(t), np.sin(t) + np.sqrt(np.abs(np.cos(t))))
T = 2*np.pi
heart_function = DrawableFunction(name, f, T)

# List of function (and their names)
drawable_functions = [fish_function, heart_function]
drawable_functions_names = [drawable.name for drawable in drawable_functions]