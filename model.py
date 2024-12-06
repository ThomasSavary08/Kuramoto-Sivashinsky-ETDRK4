# Libraries
import numpy as np
from pade import *

# One-dimensional Kuramoto-Sivashinsky equation
class KSE_1D():

    def __init__(self, L: float, N: int, nu: float, u0: np.ndarray, t0: float, dt: float):
        '''
        Instantiation of a solver for the one-dimensional Kuramoto-Sivashinsky equation.
            Parameters:
                L (float): length of the spatial domain.
                N (int): number of discretization points for the spatial domain.
                nu (float): viscosity parameter.
                u0 (np.ndarray): initial condition fo dimension (N,).
                t0 (float): initial time.
                dt (float): time step to integrate the system.
        '''
        self.L = L
        self.N = N
        self.K = int((N-1)//2)
        self.x = np.linspace(0., L, N, endpoint = False)
        self.w = np.fft.rfftfreq(N, d = L/(2*np.pi*N))
        self.nu = nu
        self.c = self.w**2 - nu*(self.w**4)
        assert len(u0) == N
        self.u_hat = np.fft.rfft(u0)
        self.t = t0
        self.h = dt
    
    def next(self, method: str = 'ETDRK4'):
        '''
        Compute the state of the system after one time step with ETDRK2 or ETDRK4.
            Parameters:
                method (str): method to use for integration.
        '''
        # Compute next state
        if (method == 'ETDRK2'):
            phis = phi(self.h*self.c)
            U1 = self.u_hat
            N_U1 = -0.5j*self.w*np.fft.rfft(np.fft.irfft(U1, n = self.N)**2)
            U2 = self.h*phis[:,1]*N_U1 + phis[:,0]*self.u_hat
            N_U2 = -0.5j*self.w*np.fft.rfft(np.fft.irfft(U2, n = self.N)**2)
            self.u_hat = self.h*(phis[:,1]-phis[:,2])*N_U1 + self.h*phis[:,2]*N_U2 + phis[:,0]*self.u_hat
            self.t += self.h
        else:
            phis1, phis2 = phi(self.h*self.c), phi((self.h*self.c)/2.)
            U1 = self.u_hat
            N_U1 = -0.5j*self.w*np.fft.rfft(np.fft.irfft(U1, n = self.N)**2)
            U2 = (self.h/2.)*phis2[:,1]*N_U1 + phis2[:,0]*self.u_hat
            N_U2 = -0.5j*self.w*np.fft.rfft(np.fft.irfft(U2, n = self.N)**2)
            U3 = (self.h/2.)*phis2[:,1]*N_U2 + phis2[:,0]*self.u_hat
            N_U3 = -0.5j*self.w*np.fft.rfft(np.fft.irfft(U3, n = self.N)**2)
            U4 = (self.h/2.)*phis2[:,1]*(phis2[:,0]-1.)*N_U1 + self.h*phis2[:,1]*N_U3 + phis1[:,0]*self.u_hat
            N_U4 = -0.5j*self.w*np.fft.rfft(np.fft.irfft(U4, n = self.N)**2)
            self.u_hat = self.h*(phis1[:,1]-3*phis1[:,2]+4.*phis1[:,3])*N_U1 + self.h*(2*phis1[:,2]-4.*phis1[:,3])*(N_U2+N_U3) + self.h*(4*phis1[:,3]-phis1[:,2])*N_U4 + phis1[:,0]*self.u_hat
            self.t += self.h
        
    def next_LTM(self, W: np.ndarray, method: str = 'ETDRK4'):
        '''
        Compute the state of a deviation vectors after one time step.
            Parameters:
                W (numpy.ndarray): Array of deviations vectors.
                method (str): method to use for integration.
            Returns:
                res (numpy.ndarray): Array of deviations vectors at next time step.
        '''
        if (method == 'ETDRK2'):
            phis = phi(self.h*self.c)
            W1 = W
            N_W1 = -1.j*np.diag(self.w)@np.fft.rfft((np.diag(np.fft.irfft(self.u_hat, n = self.N))@np.fft.irfft(W1, n = self.N, axis = 0)), axis = 0)
            W2 = self.h*np.diag(phis[:,1])@N_W1 + np.diag(phis[:,0])@W
            N_W2 = -1.j*np.diag(self.w)@np.fft.rfft((np.diag(np.fft.irfft(self.u_hat, n = self.N))@np.fft.irfft(W2, n = self.N, axis = 0)), axis = 0)
            res = self.h*np.diag(phis[:,1]-phis[:,2])@N_W1 + self.h*np.diag(phis[:,2])@N_W2 + np.diag(phis[:,0])@W
            return res
        else:
            phis1, phis2 = phi(self.h*self.c), phi((self.h*self.c)/2.)
            W1 = W
            N_W1 = -1.j*np.diag(self.w)@np.fft.rfft((np.diag(np.fft.irfft(self.u_hat, n = self.N))@np.fft.irfft(W1, n = self.N, axis = 0)), axis = 0)
            W2 = (self.h/2.)*np.diag(phis2[:,1])@N_W1 + np.diag(phis2[:,0])@W
            N_W2 = -1.j*np.diag(self.w)@np.fft.rfft((np.diag(np.fft.irfft(self.u_hat, n = self.N))@np.fft.irfft(W2, n = self.N, axis = 0)), axis = 0)
            W3 = (self.h/2.)*np.diag(phis2[:,1])@N_W2 + np.diag(phis2[:,0])@W
            N_W3 = -1.j*np.diag(self.w)@np.fft.rfft((np.diag(np.fft.irfft(self.u_hat, n = self.N))@np.fft.irfft(W3, n = self.N, axis = 0)), axis = 0)
            W4 = (self.h/2.)*np.diag(phis2[:,1]*(phis2[:,0]-1.))@N_W1 + self.h*np.diag(phis2[:,1])@N_W3 + np.diag(phis1[:,0])@W
            N_W4 = -1.j*np.diag(self.w)@np.fft.rfft((np.diag(np.fft.irfft(self.u_hat, n = self.N))@np.fft.irfft(W4, n = self.N, axis = 0)), axis = 0)
            res = self.h*np.diag(phis1[:,1]-3*phis1[:,2]+4.*phis1[:,3])@N_W1 + self.h*np.diag(2*phis1[:,2]-4.*phis1[:,3])@(N_W2+N_W3) + self.h*np.diag(4*phis1[:,3]-phis1[:,2])@N_W4 + np.diag(phis1[:,0])@W
            return res
    
    def forward(self, n_steps: int, keep_traj: bool = True):
        '''
        Forward the system for n_steps with ETDRK4 method.
            Parameters:
                n_steps (int): number of simulation steps to do.
                keep_traj (bool): return or not the system trajectory.
            Returns:
                traj (numpy.ndarray): trajectory of the system of dimension (n_steps + 1, N) if keep_traj.
        '''
        if (keep_traj):
            traj = np.zeros((n_steps + 1, self.N))
            traj[0,:] = np.fft.irfft(self.u_hat, n = self.N)
            for i in range(1, n_steps + 1):
                self.next(method = 'ETDRK4')
                traj[i,:] = np.fft.irfft(self.u_hat, n = self.N)
            return traj
        else:
            for _ in range(n_steps):
                self.next(method = 'ETDRK4')
    
    def LCE(self, p : int, n_forward : int, n_compute : int, keep : bool):
        '''
        Compute LCE.
            Parameters:
                p (int): Number of LCE to compute.
                n_forward (int): number of time steps before starting the computation of LCE. 
                n_compute (int): number of steps to compute the LCE, can be adjusted using keep_evolution.
                keep (bool): if True returns a numpy array of dimension (n_compute,p) containing the evolution of LCE.
            Returns:
                LCE (numpy.ndarray): Lyapunov Charateristic Exponents.
                history (numpy.ndarray): Evolution of LCE during the computation.
        '''
        # Forward the system before the computation of LCE
        self.forward(n_forward, False)

        # Computation of LCE
        W = np.fft.rfft(np.eye(self.N)[:,:p], axis = 0)
        LCE = np.zeros(p)
        if keep:
            history = np.zeros((n_compute, p))
            for i in range(1, n_compute + 1):
                W = self.next_LTM(W)
                self.forward(1, False)
                W, R = np.linalg.qr(np.fft.irfft(W, n = self.N, axis = 0))
                for j in range(p):
                    LCE[j] += np.log(np.abs(R[j,j]))
                    history[i-1,j] = LCE[j] / (i * self.h)
                W = np.fft.rfft(W, axis = 0)
            LCE = LCE / (n_compute * self.h)
            return LCE, history
        else:
            for _ in range(n_compute):
                W = self.next_LTM(W)
                self.forward(1, False)
                W, R = np.linalg.qr(np.fft.irfft(W, n = self.N, axis = 0))
                for j in range(p):
                    LCE[j] += np.log(np.abs(R[j,j]))
                W = np.fft.rfft(W, axis = 0)
            LCE = LCE / (n_compute * self.h)
            return LCE
    
    def LLE(self, p : int, n_forward : int, n_compute : int, keep : bool):
        '''
        Compute LLE (Local Lyapunov Exponents).
            Parameters:
                p (int): Number of LLE to compute.
                n_forward (int): number of time steps before starting the computation of LLE. 
                n_compute (int): number of steps to compute the LLE.
                keep (bool): if True returns a numpy array of dimension (n_compute,self.N) containing the trajectory of the system.
            Returns:
                LLE (numpy.ndarray): numpy array of dimension (n_compute,p) containing LLE.
                history (numpy.ndarray): trajectory of the system during the computation.
        '''
        # Forward the system before the computation of LLE
        self.forward(n_forward, False)

        # Computation of LLE
        W = np.fft.rfft(np.eye(self.N)[:,:p], axis = 0)
        LLE = np.zeros((n_compute, p))
        if keep:
            history = np.zeros((n_compute, self.N))
            for i in range(1, n_compute + 1):
                W = self.next_LTM(W)
                self.forward(1, False)
                W, R = np.linalg.qr(np.fft.irfft(W, n = self.N, axis = 0))
                for j in range(p):
                    LLE[i-1,j] = np.log(np.abs(R[j,j]))/self.h
                history[i-1,:] = np.fft.irfft(self.u_hat, n = self.N)
                W = np.fft.rfft(W, axis = 0)
            return LLE, history
        else:
            for i in range(1, n_compute + 1):
                W = self.next_LTM(W)
                self.forward(1, False)
                W, R = np.linalg.qr(np.fft.irfft(W, n = self.N, axis = 0))
                for j in range(p):
                    LLE[i-1,j] = np.log(np.abs(R[j,j]))/self.h
                W = np.fft.rfft(W, axis = 0)
            return LLE

    def CLV(self, p : int, n_forward : int, n_A : int, n_B : int, n_C : int, traj : bool):
        '''
        Compute CLVs.
            Parameters:
                p (int): number of CLV to compute at each time step.
                n_forward (int): number of time steps before starting the computation of CLVs. 
                n_A (int): number of steps for the orthogonal matrice Q to converge to BLV.
                n_B (int): number of time steps for which Phi and R matrices are stored and for which CLV are computed.
                n_C (int): number of steps for which R matrices are stored in order to make A converge to A-. 
                traj (bool): if True returns a numpy array of dimension (n_B, self.N) containing system's trajectory at the times CLVs are computed.
            Returns:
                CLV (List): list of numpy.array containing CLV computed during n_B time steps.
                history (numpy.ndarray): trajectory of the system during the computation of CLV.
        '''
        # Forward the system before the computation of CLV
        self.forward(n_forward, keep_traj = False)

        # Make W converge to Phi
        W = np.fft.rfft(np.eye(self.N)[:,:p], axis = 0)
        for _ in range(n_A):
            W = self.next_LTM(W)
            W, _ = np.linalg.qr(np.fft.irfft(W, n = self.N, axis = 0))
            W = np.fft.rfft(W, axis = 0)
            self.forward(1, keep_traj = False)
        
        # We continue but now Q and R are stored to compute CLV later
        Phi_list, R_list1 = [np.fft.irfft(W, n = self.N, axis = 0)], []
        if traj:
            history = np.zeros((n_B+1, self.N))
            history[0,:] = np.fft.irfft(self.u_hat, n = self.N)
        for i in range(n_B):
            W = self.next_LTM(W)
            W, R = np.linalg.qr(np.fft.irfft(W, n = self.N, axis = 0))
            Phi_list.append(W)
            R_list1.append(R)
            W = np.fft.rfft(W, axis = 0)
            self.forward(1, keep_traj = False)
            if traj:
                history[i+1,:] = np.fft.irfft(self.u_hat, n = self.N)
        
        # Now we only store R to compute A- later
        R_list2 = []
        for _ in range(n_C):
            W = self.next_LTM(W)
            W, R = np.linalg.qr(np.fft.irfft(W, n = self.N, axis = 0))
            R_list2.append(R)
            W = np.fft.rfft(W, axis = 0)
            self.forward(1, keep_traj = False)
        
        # Generate A make it converge to A-
        A = np.triu(np.random.rand(p,p))
        for R in reversed(R_list2):
            C = np.diag(1. / np.linalg.norm(A, axis = 0))
            B = A @ C
            A = np.linalg.solve(R, B)
        del R_list2

        # Compute CLV
        CLV = [Phi_list[-1] @ A]
        for Q, R in zip(reversed(Phi_list[:-1]), reversed(R_list1)):
            C = np.diag(1. / np.linalg.norm(A, axis = 0))
            B = A @ C
            A = np.linalg.solve(R, B)
            CLV_t = Q @ A
            CLV.append(CLV_t / np.linalg.norm(CLV_t, axis = 0))
        del R_list1
        del Phi_list
        CLV.reverse()

        if traj:
            return CLV, history
        else:
            return CLV

    # Compute adjoints of CLV
    def ADJ(self, CLV : list):
        '''
        Compute adjoints vectors of CLV.
            Parameters:
                CLV (list): List of np.ndarray containing CLV at each time step: [CLV(t1), ...,CLV(tn)].
            Returns:
                ADJ (List): List of numpy.array containing adjoints of CLV at each time step (each column corresponds to an adjoint).
        '''
        ADJ = []
        for n in range(len(CLV)):
            try:
                ADJ_t = np.linalg.solve(np.transpose(CLV[n]), np.eye(CLV[n].shape[0]))
                ADJ.append(ADJ_t / np.linalg.norm(ADJ_t, axis = 0))
            except:
                ADJ_t = np.zeros_like(CLV[n])
                for j in range(ADJ_t.shape[1]):
                    columns = [i for i in range(ADJ_t.shape[1])]
                    columns.remove(j)
                    A = np.transpose(CLV[n][:,columns])
                    _, _, Vh = np.linalg.svd(A)
                    theta_j = Vh[-1] / np.linalg.norm(Vh[-1])
                    ADJ_t[:,j] = theta_j
                ADJ.append(ADJ_t)
        return ADJ
