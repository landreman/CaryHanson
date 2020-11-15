#!/usr/bin/env python3

"""
This module contains objective functions for optimization
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, least_squares

from .helicalcoil import HelicalCoil
from .periodic_field_line import periodic_field_line
from .tangent_map import tangent_map

def num_poloidal_turns(R0, field, Raxis, L):
    """
    This function is used by find_R0.
    """

    phimax = L * 2 * np.pi / field.nfp
    t_span = (0, phimax)
    Z0 = 0
    sol = solve_ivp(field.d_RZ_d_phi, t_span, [R0, Z0], rtol=1e-9, atol=1e-12)
    R = sol.y[0, :]
    Z = sol.y[1, :]
    theta = np.arctan2(Z, R - Raxis)
    print('R:',R)
    print('Z:',Z)
    print('theta:',theta)
    # Eliminate jumps by 2pi:
    for j in range(1, len(R)):
        if theta[j] > theta[j-1] + np.pi / 2:
            # Big increase
            theta[j:] -= 2 * np.pi
        elif theta[j] < theta[j-1] - np.pi / 2:
            # Big decrease
            theta[j:] += 2 * np.pi
    print('theta:',theta)

    poloidal_turns = (theta[-1] - theta[0]) / (2 * np.pi)
    print("*******************************")
    print("At R=", R0, " poloidal_turns=", poloidal_turns)
    print("*******************************")
    return poloidal_turns

def find_R0(field, L, R1, poloidal_turns):
    """
    Compute a good guess for R0 for a periodic field line using a
    more robust method than periodic_field_line.
    """
    
    # First, find the location of the magnetic axis:
    pfl_axis = periodic_field_line(field, 31, periods=1, R0=1.0)
    # For now, assume R along the axis is nearly constant:
    Raxis = np.mean(pfl_axis.R)

    def objective(R):
        return num_poloidal_turns(R, field, Raxis, L) - poloidal_turns
    
    #R2 = Raxis
    #objective1 = objective(R1)
    # Vary R2 until we have bracketed the objective
    
    # Find the range to search:
    if Raxis > R1:
        Rmin = R1
        #Rmax = Raxis
        Rmax = R1 + 0.7 * (Raxis - R1)
    else:
        Rmax = R1
        Rmin = Raxis + 0.05 * (R1 - Raxis)
        #Rmin = Raxis
        
    sol = root_scalar(objective, bracket=[Rmin, Rmax])
    print("Solution:")
    print(sol)
    print(sol.root)
    return sol.root

def find_R0_brute_force(field, L, R_min, R_max, N):
    """
    Find a good guess for the initial R for a periodic field line by a direct search.
    """
    Rs = np.linspace(R_min, R_max, N)
    residuals = np.zeros(N)
    phimax = L * 2 * np.pi / field.nfp
    t_span = (0, phimax)
    Z0 = 0
    for j in range(N):
        sol = solve_ivp(field.d_RZ_d_phi, t_span, [Rs[j], Z0], rtol=1e-9, atol=1e-12)
        R = sol.y[0, :]
        Z = sol.y[1, :]
        # Measure the error in final - initial location:
        residuals[j] = (R[-1] - R[0]) ** 2 + Z[-1] ** 2
        
    index = np.argmin(residuals)
    R0 = Rs[index]
    print('find_R0_brute_force residuals:', residuals, ' best index=', index, ' best residual=', residuals[index], ' best R0=', Rs[index])
    return R0

def I0307_objective(x):
    """
    x should be a 2 element vector giving A(2,1) and B(1,1)
    """
    f = open('I0307_points', 'a')
    f.write('{:20.15} {:20.15}\n'.format(x[0], x[1]))
    f.close()

    A = [[0, np.pi / 2], \
         [0, x[0]]]

    B = [[0, 0], \
         [x[1], 0]]

    field = HelicalCoil(I=np.array([-1,1])*0.0307, A=A, B=B)

    L = 3

    # The periodic field line on the right is easier to find:
    R0_right = 1.0

    # The periodic field line on the left is harder to find, so we
    # need a more robust search:
    #R0_left = find_R0(field, 3, 0.81, -1)
    R0_left = find_R0_brute_force(field, 3, 0.81, 0.88, 30)

    residuals = []
    for j, R0 in enumerate([R0_left, R0_right]):
        pfl = periodic_field_line(field, 199, periods=L, R0=R0, Z0=0)
        if pfl.residual > 1e-8:
            raise RuntimeError('periodic_field_line residual is large')
        if np.abs(pfl.Z_k[0]) > 1e-6:
            raise RuntimeError('Initial Z of periodic_field_line is not 0')
        if np.max(pfl.R_k) - np.min(pfl.R_k) < 1e-4:
            raise RuntimeError('Found the magnetic axis instead of the desired X or O point')
        if j==0 and np.argmin(pfl.R_k) != 0:
            raise RuntimeError('Failed finding left periodic field line')
        if j==1 and np.argmax(pfl.R_k) != 0:
            raise RuntimeError('Failed finding right periodic field line')
        
        tm = tangent_map(field, pfl, atol=1e-8, rtol=1e-11)
        residuals.append(tm.residue)
        
    f = open('I0307_evals', 'a')
    f.write('{:20.15} {:20.15} {:20.15} {:20.15}\n'.format(x[0], x[1], residuals[0], residuals[1]))
    f.close()
    
    return np.array(residuals)

def solve_I0307():
    """
    Driver for the optimization of the I = 0.0307 coils.
    """
    # Delete files, if they exist:
    f = open('I0307_points', 'w')
    f.write('x[0], x[1]\n')
    f.close()
    f = open('I0307_evals', 'w')
    f.write('x[0], x[1], y[0], y[1]\n')
    f.close()

    lower_bounds = [0, 0]
    upper_bounds = [0.4, 0.4]
    x0 = [0.3, 0.3]
    soln = least_squares(I0307_objective, x0, bounds=(lower_bounds, upper_bounds), verbose=2, jac='3-point', diff_step=1e-5)
    print('solution:', soln)
    print('x at solution:', soln.x)
    print('function at solution:', soln.fun)
