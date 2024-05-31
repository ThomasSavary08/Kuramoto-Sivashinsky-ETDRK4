# Libraries
import math as m
import numpy as np

# Coefficient of the numerator for Padé approximant
def N(d: int, l: int):
    '''
    Coefficient of the numerator of the (d,d)-Padé approximant of phi_l.
        Parameters:
            d (int): order of the Padé approximant.
            l (int): number of phi function.
        Returns:
            coeffs (np.ndarray): numpy array of dimension (d+1,) containing coefficients of the numerator of the (d,d)-Padé approximant of phi_l.
        '''
    coeffs = []
    for i in range(d+1):
        coeff = 0.
        for j in range(i+1):
            coeff += (m.factorial(2*d+l-j)*((-1.)**j))/(m.factorial(j)*m.factorial(d-j)*m.factorial(l+i-j))
        coeffs.append(coeff)
    return np.asarray(coeffs)

# Coefficient of the denominator for the Padé approximant
def D(d: int, l: int):
    '''
    Coefficient of the denominator of the (d,d)-Padé approximant of phi_l.
        Parameters:
            d (int): order of the Padé approximant.
            l (int): number of phi function.
        Returns:
            coeffs (np.ndarray): numpy array of dimension (d+1,) containing coefficients of the denominator of the (d,d)-Padé approximant of phi_l.
    '''
    coeffs = []
    for i in range(d+1):
        coeff = m.factorial(2*d+l-i)/(m.factorial(i)*m.factorial(d-i))
        coeffs.append(coeff)
    return np.asarray(coeffs)

# phi_1
N_61 = N(6,1) 
D_61 = D(6,1)
assert len(N_61) == len(D_61)

def phi_1(z: np.ndarray):
    '''
    Computation of phi_1(z) with (6,6)-Padé approximant.
        Parameters:
            z (np.ndarray): input vector of dimension (n,).
        Returns:
            res (np.ndarray): array of dimension (n,) containing phi_1(z).
    '''
    N, D = np.zeros_like(z), np.zeros_like(z)
    for i in range(len(N_61)):
        N += N_61[i]*(z**i)
        D += D_61[i]*((-z)**i)
    res = N/D
    return res

# phi_2
N_62 = N(6,2) 
D_62 = D(6,2)
assert len(N_62) == len(D_62)

def phi_2(z: np.ndarray):
    '''
    Computation of phi_2(z) with (6,6)-Padé approximant.
        Parameters:
            z (np.ndarray): input vector of dimension (n,).
        Returns:
            res (np.ndarray): array of dimension (n,) containing phi_2(z).
    '''
    N, D = np.zeros_like(z), np.zeros_like(z)
    for i in range(len(N_62)):
        N += N_62[i]*(z**i)
        D += D_62[i]*((-z)**i)
    res = N/D
    return res

# phi_3
N_63 = N(6,3) 
D_63 = D(6,3)
assert len(N_63) == len(D_63)

def phi_3(z: np.ndarray):
    '''
    Computation of phi_3(z) with (6,6)-Padé approximant.
        Parameters:
            z (np.ndarray): input vector of dimension (n,).
        Returns:
            res (np.ndarray): array of dimension (n,) containing phi_3(z).
    '''
    N, D = np.zeros_like(z), np.zeros_like(z)
    for i in range(len(N_63)):
        N += N_63[i]*(z**i)
        D += D_63[i]*((-z)**i)
    res = N/D
    return res

# phi_l(z) for l in [0,1,2,3]
def phi(z: np.ndarray):
    '''
    Computation of phi_l(z) with (6,6)-Padé approximant for l in [0,1,2,3].
        Parameters:
            z (np.ndarray): input vector of dimension (n,).
        Returns:
            res (np.ndarray): array of diemnsion (n,4) contaning phi_l(z) for l in [0,1,2,3].
    '''
    # Scaling
    p = int(np.ceil(np.log2(np.max(np.abs(z)))))
    m = max(0,p+1)
    z_scaled = z/(2.**m)

    # Compute phi_l(z_scaled) and undo scaling
    res = np.zeros((len(z_scaled),4))
    res[:,0] = np.exp(z_scaled)
    res[:,1] = phi_1(z_scaled)
    res[:,2] = phi_2(z_scaled)
    res[:,3] = phi_3(z_scaled)

    if (m == 0):
        return res
    else:
        for i in range(1,m+1):
            res_int = np.zeros_like(res)
            res_int[:,0] = res[:,0]**2
            res_int[:,1] = 0.5*(res[:,0]*res[:,1] + res[:,1])
            res_int[:,2] = 0.25*(res[:,1]**2 + 2.*res[:,2])
            res_int[:,3] = 0.125*(res[:,1]*res[:,2] + 2.*res[:,3] + res[:,2])
            res = res_int
        return res
    