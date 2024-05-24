# Libraries
import numpy as np

def coeffs_integration(K: int, f_t: np.ndarray, t: np.ndarray, T: float, dt: float):
    '''
    Returns the coefficients [c_(-K), ..., c_0, ..., c_K] of the Fourier expansion of a periodic function using the integration method.
        Parameters:
            K (int): The number of coefficients to compute.
            f_t (np.ndarray): Array containing the values of the function at certain times.
            t (np.ndarray): The times associated with the previous array.
            T (float): Period of the function.
            df (float): Integration time step.
        Returns:
            coeffs (np.ndarray): Array of dimension (2K+1,) containing the coefficients [c_(-K), ..., c_0, ..., c_K] of the Fourier expansion.
    '''
    # Instanciate coefficients
    coeffs = np.zeros(2*K+1, dtype = complex)

    # Checks before computation
    assert len(f_t) == len(t)

    # Computation of each coefficient
    for n in range(-K,K+1):
        c_n = 0. + 0j
        for i in range(len(t)):
            c_n += f_t[i]*np.exp(-1j*(2*np.pi/T)*n*t[i])*dt
        coeffs[n+K] = (1./T)*c_n
    
    return coeffs

def coeffs_collocation(K: int, f_t: np.ndarray, t: np.ndarray, T: float):
    '''
    Returns the coefficients [c_(-K), ..., c_0, ..., c_K] of the Fourier expansion of a periodic function using the collocation method.
        Parameters:
            K (int): The number of coefficients to compute.
            f_t (np.ndarray): Array containing the values of the function at certain times.
            t (np.ndarray): The times associated with the previous array.
            T (float): Period of the function.
        Returns:
            coeffs (np.ndarray): Array of dimension (2K+1,) containing the coefficients [c_(-K), ..., c_0, ..., c_K] of the Fourier expansion.
    '''
    # Checks before computation
    assert len(f_t) == len(t)
    assert len(f_t) >= (2*K+1)

    # Computation of coefficients
    A = np.zeros((2*K+1,2*K+1), dtype = complex)
    b = np.zeros(2*K+1, dtype = complex)
    for i in range(2*K+1):
        for k in range(-K,K+1):
            A[i,k+K] = np.exp(1j*(2*np.pi/T)*k*t[i])
        b[i] = f_t[i]
    coeffs = np.linalg.solve(A,b)

    return coeffs

def coeffs_fft(K: int, f_t: np.ndarray):
    '''
    Returns the coefficients [c_(-K), ..., c_0, ..., c_K] of the Fourier expansion of a periodic function using Numpy's fft.
        Parameters:
            K (int): The number of coefficients to compute.
            f_t (np.ndarray): Array containing the values of the function at certain times.
        Returns:
            coeffs (np.ndarray): Array of dimension (2K+1,) containing the coefficients [c_(-K), ..., c_0, ..., c_K] of the Fourier expansion.
    '''
    # Computation of coefficients
    coeffs = np.fft.fft(f_t)
    coeffs = (1./(2.*K+1.)) * np.fft.fftshift(coeffs)

    return coeffs

def order_coeffs(coeffs: np.ndarray, K: int):
    '''
    Returns the coefficients [c_0, c_(-1), c_1, ..., c_(-K), c_K] of the Fourier expansion for drawing.
        Parameters:
            K (int): The number of coefficients to compute.
            coeffs (np.ndarray): Array of dimension (2K+1,) containing the coefficients [c_(-K), ..., c_0, ..., c_K] of the Fourier expansion.
        Returns:
            ordered_coeffs (np.ndarray): Array of dimension (2K+1,) containing the coefficients [c_0, c_(-1), c_1, ..., c_(-K), c_K] of the Fourier expansion.
    '''
    ordered_coeffs = [coeffs[K]]
    for i in range(1,len(coeffs)):
        if (i%2 == 0):
            ordered_coeffs.append(coeffs[K + int(i//2)])
        else:
            ordered_coeffs.append(coeffs[K - int(i//2 + 1)])
    ordered_coeffs = np.asarray(ordered_coeffs)

    return ordered_coeffs
