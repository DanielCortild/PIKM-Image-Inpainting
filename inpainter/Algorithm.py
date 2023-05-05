# !/usr/bin/env python
# encoding: utf-8
"""
Algorithm2.py - Implements the Algorithm
~ Daniel Cortild, 21 March 2023
"""

# External imports
import numpy as np
import scipy as sp
import scipy.linalg
from tqdm.auto import trange                # type: ignore
import time


# The proximal operator of the nuclear norm
def svd_shrink(X, r):
    """
    Returns the proximal of the nuclear norm
    """
    try:
        U, S, VT = sp.linalg.svd(X, full_matrices=False)
        S_shrink = np.maximum(S - r, 0)
        X_shrink = ( U * S_shrink ) @ VT
        return X_shrink
    except np.linalg.LinAlgError as err:
        return X
    
def nuc_norm(X):
    return np.linalg.norm(fold(X, 0), ord='nuc') + np.linalg.norm(fold(X, 1), ord='nuc')

# Corresponds to (.)_(1) and (.)_(2)
def fold(X, axis):
    """
    Transforms a (N, M, 3) tensor to a (N, 3*M) or (3*N, M) tensor
    Reverses un fold
    """
    (a, b) = (1, 3) if axis == 0 else (3, 1)
    return X.reshape(X.shape[0] * a, X.shape[1] * b, order='F')

# Corresponds to iota_1 and iota_2
def unfold(X, axis):
    """
    Transforms a (N, 3*M) or (3*N, M) tensor to a (N, M, 3) tensor
    Reverses fold
    """
    (a, b) = (1, 3) if axis == 0 else (3, 1)
    return X.reshape(X.shape[0] // a, X.shape[1] // b, 3, order='F')


class Algorithm:
    """
    Solves minimisation problems over a Hilbert space H of the type:
            min_{x in H} f(x) + g(x) + h(Lx)
    Parameters:
        proxf               The proximal operator of f
        proxg               The proximal operator of g
        LgradhL             The operator L^*(grad_h(L))
        Z_init              Initial guess of Z
        lamb                Value of lambda in (0,1) [Default: 0.5]
        rho                 Value of rho in (0,2) [Default: 1]
        beta                The inverse of the Lipschitz constant of grad_h [Default: 1]
        alpha_static        Boolean expression whether alpha is static or not [Default: False]
    Public Methods:
        run                Runs the algorithm
    Private Methods:
        iterate            Runs a single iteration of the algorithm
        error              Compute error estimate used for stopping criterion
    """

    def __init__(self, image, mask, rho, lamb, sigma, tolerance, max_it, method, perturbed):
        self.rho = rho
        self.lamb = lamb
        self.tolerance = tolerance
        self.max_it = max_it
        self.sigma = sigma
        
        self.R = lambda X1, X0: np.linalg.norm(X1 - X0) / np.linalg.norm(X0)
        self.A = lambda X: mask.mask_image(X)
        self.proxf = lambda X: unfold(svd_shrink(fold(X, 0), rho*sigma), 0)
        self.proxg = lambda X: unfold(svd_shrink(fold(X, 1), rho*sigma), 1)
        
        eta = 2 * lamb / (4 - rho)
        if method == "accelerated": # beta=0
            a = (1 - eta) - (1/eta - 1)
            b = (1 - eta) + 2 * (1/eta - 1)
            c =  - (1/eta - 1)
            if abs(a) < 1e-4:
                delta = -c / b
            else:
                delta = (-b + np.sqrt(b**2 - 4*a*c) ) / (2 * a)
            alpha = (1 - eta) * delta / (1 - lamb)
            self.get_alpha = lambda k: (1 - 1 / (k+1)) * alpha
            self.get_beta  = lambda k: 0
        elif method == "inertial": # alpha=beta
            a = 1 - (1/eta - 1)
            b = 1 + 2 * (1/eta - 1)
            c =  - (1/eta - 1)
            if abs(a) < 1e-4:
                alpha = -c / b
            else:
                alpha = (-b + np.sqrt(b**2 - 4*a*c) ) / (2 * a)
            self.get_alpha = lambda k: (1 - 1 / (k+1)) * alpha
            self.get_beta  = lambda k: (1 - 1 / (k+1)) * alpha
        else:
            raise ValueError("No method specified")
            
        self.get_epsilon = lambda k: np.random.rand(*image.shape) / (30 * (k + 1)) if perturbed else 0
        self.get_rho     = lambda k: np.random.rand(*image.shape) / (30 * (k + 1)) if perturbed else 0
        self.get_theta   = lambda k: np.random.rand(*image.shape) / (30 * (k + 1)) if perturbed else 0
        
        self.X_corrupt = image.copy()
        self.F = lambda X: np.linalg.norm(X-self.X_corrupt)**2/2 + sigma * nuc_norm(X)

    def __iterate(self, X_previous, X_actual, k):
        """ @private
        Perform the iterations according to Algorithm 2
        """
        alpha    = self.get_alpha(k)
        beta     = self.get_beta(k)
        epsilon  = self.get_epsilon(k)
        rho      = self.get_rho(k)
        theta    = self.get_theta(k)
        lambd    = self.lamb
        
        Y       = X_actual + alpha * (X_actual - X_previous) + epsilon
        Z       = X_actual + beta * (X_actual - X_previous) + rho
        X_g     = self.proxg(Z)
        X_T     = Z - X_g + self.proxf(2 * X_g - Z - self.rho * (self.A(X_g) - self.X_corrupt))
        X_next  = (1 - lambd) * Y + lambd * X_T + theta
        
        return X_actual, X_next, X_T

    def run(self, bregman=False, verbose=True):
        """ @public
        Run the algorithm given the number of iterations and the iterator
        """
        X_previous = np.zeros_like(self.X_corrupt)
        X_next = self.X_corrupt
        X_hist = []
        X_T_hist = []
        X_F_hist = []
        its = 0

        start = time.time()
        
        for its in trange(self.max_it, disable = not verbose):
            X_previous, X_next, X_T = self.__iterate(X_previous, X_next, its)
            X_hist.append(X_previous)
            X_F_hist.append(self.F(X_previous))
            X_T_hist.append(X_T)
            if self.R(X_next, X_previous) < self.tolerance:
                break

        solution = self.proxg(X_next)
                
        return solution, X_hist, X_T_hist, X_F_hist, its + 1, time.time() - start, self.F
