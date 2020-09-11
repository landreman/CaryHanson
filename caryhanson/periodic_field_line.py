#!/usr/bin/env python3

"""
This module provides a function for finding periodic field lines and the magnetic axis.
"""

import numpy as np
from scipy.optimize import fsolve
from .spectral_diff_matrix import spectral_diff_matrix
#import matplotlib.pyplot as plt

def periodic_field_line(field, n, periods=1, R0=None, Z0=None):
    """
    Solves for a periodic field line.

    field: An object with a BR_Bphi_BZ method and nfp attribute..
    periods: The number of field periods over which the field line will be periodic.
    n: The number of grid points to use for the solve. If n is even, 1 will be added
       so n always ends up odd.
    R0: An initial guess for R. Either a float or a numpy array of shape (n,). 
       If a single float, B will be roughly integrated to generate an array.
    Z0: An initial guess for Z. Either a float or a numpy array of shape (n,). 
       If a single float, B will be roughly integrated to generate an array

    R0 and Z0 must either be both numpy ndarrays, or else both floats.
    """
    
    # Ensure n is odd:
    if np.mod(n, 2) == 0:
        n += 1
    
    if R0 is None:
        R0 = 1.0
    if Z0 is None:
        Z0 = 0.0

    phimax = periods * 2 * np.pi / field.nfp
    dphi = phimax / n
    phi = np.linspace(0, phimax, n, endpoint=False)
    assert np.abs(phi[1] - phi[0] - dphi) < 1.0e-13

    array_input = isinstance(R0, np.ndarray)
    if array_input:
        assert isinstance(Z0, np.ndarray)
        assert R0.shape == Z0.shape
        
    if not array_input:
        # Use a crude 1st-order Euler step to generate the initial guess.
        R0 = np.full(n, R0)
        Z0 = np.full(n, Z0)
        for j in range(n - 1):
            BR, Bphi, BZ = field.BR_Bphi_BZ(R0[j], dphi * j, Z0[j])
            R0[j + 1] = R0[j] + dphi * R0[j] * BR / Bphi
            Z0[j + 1] = Z0[j] + dphi * R0[j] * BZ / Bphi
            
        print('Generated initial condition:')
        print('R0: ', R0)
        print('Z0: ', Z0)

    D = spectral_diff_matrix(n, xmin=0, xmax=phimax)
    
    def func(x):
        """
        This is the vector-valued function that returns the residual.
        """
        R = x[0:n]
        Z = x[n:2 * n]
        #print('func eval ')
        #print('R=',R)
        #print('Z=',Z)
        BR = np.zeros(n)
        Bphi = np.zeros(n)
        BZ = np.zeros(n)
        for j in range(n):
            BR[j], Bphi[j], BZ[j] = field.BR_Bphi_BZ(R[j], phi[j], Z[j])
        R_residual = R * BR / Bphi - np.matmul(D, R)
        Z_residual = R * BZ / Bphi - np.matmul(D, Z)
        return np.concatenate((R_residual, Z_residual))
        
    
    state = np.concatenate((R0, Z0))
    #root, infodict, ier, mesg = fsolve(func, state)
    root = fsolve(func, state, xtol=1e-13)
    R = root[0:n]
    Z = root[n:2 * n]
    """
    bigphi = np.concatenate((phi, phi + phimax))
    plt.subplot(1, 2, 1)
    plt.plot(bigphi, np.concatenate((R, R)), '.-', label='R')
    plt.xlabel('phi')
    plt.title('R')
    plt.subplot(1, 2, 2)
    plt.plot(bigphi, np.concatenate((Z, Z)), '.-', label='Z')
    plt.xlabel('phi')
    plt.title('Z')
    plt.tight_layout()
    plt.show()
    """
    return R, phi, Z

