#!/usr/bin/env python3

"""
This module provides a function for finding properties derived from the full-orbit tangent map
of a periodic field line, including the rotational transform, Greene's residue, and the {e||, eperp} vectors.
"""

import numpy as np
from scipy.integrate import solve_ivp

def tangent_map_integrand(phi, RZU, field):
    """
    This is the function used for integrating the field line ODE.
    See the following Word doc for details:
    20200923-01 Equation for tangent map.docx
    """
    R = RZU[0]
    Z = RZU[1]
    u00 = RZU[2]
    u01 = RZU[3]
    u10 = RZU[4]
    u11 = RZU[5]
    #print('tangent_map_integrand called at R={}, phi={}, Z={}'.format(R, phi, Z))
    BR, Bphi, BZ = field.BR_Bphi_BZ(R, phi, Z)
    grad_B = field.grad_B(R, phi, Z)
    # VR = R * BR / Bphi
    # VZ = R * BZ / Bphi
    d_VR_d_R = BR / Bphi + R * grad_B[0, 0] / Bphi - R * BR / (Bphi * Bphi) * grad_B[1, 0]
    d_VR_d_Z = R * grad_B[0, 2] / Bphi - R * BR / (Bphi * Bphi) * grad_B[1, 2]
    d_VZ_d_R = BZ / Bphi + R * grad_B[2, 0] / Bphi - R * BZ / (Bphi * Bphi) * grad_B[1, 0]
    d_VZ_d_Z = R * grad_B[2, 2] / Bphi - R * BZ / (Bphi * Bphi) * grad_B[1, 2]
    
    dydt = [R * BR / Bphi,
            R * BZ / Bphi,
            d_VR_d_R * u00 + d_VR_d_Z * u10,
            d_VR_d_R * u01 + d_VR_d_Z * u11,
            d_VZ_d_R * u00 + d_VZ_d_Z * u10,
            d_VZ_d_R * u01 + d_VZ_d_Z * u11]
    return dydt
    
class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """
    pass

def tangent_map(field, periods=1, R0=1, Z0=0, rtol=1e-6, atol=1e-9):
    """
    Computes the full-orbit tangent map for a periodic fieldline, as well
    as quantities derived from the map like the rotational transform and Greene's residue.

    field: An object with BR_Bphi_BZ and grad_B methods and nfp attribute.
    periods: The number of field periods over which the field line will be periodic.
    R0: Location of the periodic field line at phi=0.
    Z0: Location of the periodic field line at phi=0.
    rtol: relative tolerance for integration along the field line.
    atol: absolute tolerance for integration along the field line.

    R0 and Z0 should each be a float, previously computed using periodic_field_line().
    """
    
    phimax = periods * 2 * np.pi / field.nfp
    t_span = (0, phimax)

    # The state vector has 6 unknowns: R, Z, and the 4 elements of the U matrix.
    x0 = [R0, Z0, 1, 0, 0, 1]

    def tangent_map_integrand_wrapper(t, y):
        return tangent_map_integrand(t, y, field)
    
    soln = solve_ivp(tangent_map_integrand_wrapper, t_span, x0, rtol=rtol, atol=atol)

    print('# of function evals: ', soln.nfev)
    
    # Make sure we got to the end:
    assert np.abs(soln.t[-1] - phimax) < 1e-13
    
    R = soln.y[0, :]
    Z = soln.y[1, :]
    # Make sure field line was indeed periodic:
    print('R(end) - R(0): ', R[-1] - R0)
    print('Z(end) - Z(0): ', Z[-1] - Z0)

    tol = 1e-5
    if np.abs(R[-1] - R0) > tol or np.abs(Z[-1] - Z0) > tol:
        raise RuntimeError('Field line is not closed. Values of R0 and Z0 provided must have been incorrect')

    # Form the full-orbit tangent map:
    M = np.array([[soln.y[2, -1], soln.y[3, -1]],
                  [soln.y[4, -1], soln.y[5, -1]]])

    print('M: ', M)
    det = np.linalg.det(M)
    print('determinant: ', det)
    if np.abs(det - 1) > 1e-4:
        raise RuntimeError('Determinant of tangent map is not close to 1!')
    
    eigvals, eigvects = np.linalg.eig(M)
    print('eigvals: ', eigvals)
    print('eigvects: ', eigvects)
    
    iota_per_period = np.angle(eigvals[0]) / (2 * np.pi)
    iota = iota_per_period * field.nfp
    residue = 0.25 * (2 - np.trace(M))
    print('iota per period: {},  total iota: {},  residue: {}'.format(iota_per_period, iota, residue))

    sigma = np.array([[0, 1], [-1, 0]])
    tempmat = np.matmul(sigma, M)
    W = 0.5 * (tempmat + tempmat.transpose())
    W_eigvals, W_eigvects = np.linalg.eig(W)
    print('W_eigvals: ', W_eigvals)
    print('W_eigvects: ', W_eigvects)
    
    results = Struct()
    results.mat = M
    results.eigvals = eigvals
    results.eigvects = eigvects
    results.iota_per_period = iota_per_period
    results.iota = iota
    results.residue = residue

    return results
