#!/usr/bin/env python3

"""
This module provides a function to generate Poincare plots
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.integrate import solve_ivp

def poincare(field, R0, Z0, npoints=20, rmin=0.75, rmax=1.2,
             zmin=-0.15, zmax=0.15, pdf=False, rtol=1e-6, atol=1e-9,
             marker_size=1, extra_str=""):
   """
   Generate a Poincare plot

   field: A magnetic field object, which must have nfp attribute.
   R0: Initial R values. Can be an array of any dimension, or a single value.
   Z0: Initial Z values
   npoints: Number of points to compute for each initial condition for the Poincare plot
   rmin, rmax, zmin, zmax: Stop tracing field line if you leave this region
   pdf: bool indicating whether to save a PDF.
   rtol, atol: tolerances for the integration
   marker_size: size of points in the plot
   extra_str: string added to plot title and filename
   """
   data = compute_poincare(field, R0, Z0, npoints=npoints, rmin=rmin,
                           rmax=rmax, zmin=zmin, zmax=zmax, rtol=rtol, atol=atol)
   
   plot_poincare(data, pdf=pdf, marker_size=marker_size, extra_str=extra_str)
   
def compute_poincare(field, R0, Z0, npoints=20, rmin=0.75, rmax=1.2,
                     zmin=-0.15, zmax=0.15, rtol=1e-6, atol=1e-9):
   """
   Generate the raw (x,y) data for a Poincare plot, without actually plotting it
   
   field: A magnetic field object, which must have nfp attribute.
   R0: Initial R values. Can be an array of any dimension, or a single value.
   Z0: Initial Z values
   npoints: Number of points to compute for each initial condition for the Poincare plot
   rmin, rmax, zmin, zmax: Stop tracing field line if you leave this region
   rtol, atol: tolerances for the integration
   """
   comm = MPI.COMM_WORLD
   mpi_N_procs = comm.Get_size()
   mpi_rank = comm.Get_rank()
   print('Hello from MPI proc {:4d} of {:4d}'.format(mpi_rank,mpi_N_procs))

   R0 = np.array(R0).flatten()
   Z0 = np.array(Z0).flatten()
   
   assert R0.shape == Z0.shape
   
   def R_too_small(phi, RZ):
      return RZ[0] - rmin

   R_too_small.terminal = True

   def R_too_big(phi, RZ):
      return RZ[0] - rmax

   R_too_big.terminal = True

   def Z_too_small(phi, RZ):
      return RZ[1] - zmin

   Z_too_small.terminal = True

   def Z_too_big(phi, RZ):
      return RZ[1] - zmax

   Z_too_big.terminal = True

   N_field_lines = len(R0)
   Poincare_data = [0]*N_field_lines
   for j in range(N_field_lines):
      if j%mpi_N_procs != mpi_rank:
         continue
      print('Proc {:4d} is tracing field line {:4d} of {:4d}.'.format(mpi_rank, j, N_field_lines))
      nfp = field.nfp
      phi_range = (0, (2*np.pi*npoints)/nfp)
      # Factors of 4 below are to get data at 1/4, 1/2, 3/4 period:
      phi_to_report = np.arange(npoints * 4) * 2 * np.pi / (4 * nfp) 
      RZ_initial = [R0[j], Z0[j]]
      solution = solve_ivp(field.d_RZ_d_phi, phi_range, RZ_initial, \
                           events=[R_too_small, R_too_big, Z_too_small, Z_too_big], \
                           t_eval=phi_to_report, rtol=rtol, atol=atol)
      phi = solution.t
      R = solution.y[0,:]
      Z = solution.y[1,:]
      #print("phi:",solution.t)
      #print("[R,Z]:",solution.y)
      #print("R:",R)
      #print("Z:",Z)
      Poincare_data[j] = solution.y

   # Send results from each processor to the root:
   for j in range(N_field_lines):
      index = j%mpi_N_procs
      if index == 0:
         # Proc 0 did this field line, so no communication is needed
         temp = 0
      else:
         if mpi_rank == 0:
            print('Root is receiving field line {:5d} from proc {:4d}'.format(j,index))
            Poincare_data[j] = comm.recv(source = index, tag=index)
         elif index == mpi_rank:
            print('Proc {:4d} is sending field line {:5d} to root'.format(mpi_rank, j))
            comm.send(Poincare_data[j], dest=0, tag=index)

   # Pack data into a dict
   data = {'data': Poincare_data,
           'rtol': rtol,
           'atol': atol,
           'mpi_N_procs': mpi_N_procs,
           'npoints': npoints,
           'nlines': N_field_lines}
   return data


def plot_poincare(data, pdf=False, marker_size=1, extra_str=""):
   """
   Generate a Poincare plot using precomputed (R,Z) data.

   data: a dict returned by compute_poincare().
   pdf: bool indicating whether to save a PDF.
   marker_size: size of points in the plot
   extra_str: string added to plot title and filename
   """
   comm = MPI.COMM_WORLD
   mpi_N_procs = comm.Get_size()
   mpi_rank = comm.Get_rank()
   if mpi_rank != 0:
      return

   Poincare_data = data['data']
   N_field_lines = data['nlines']
   
   fig = plt.figure(figsize=(14,7))
   num_rows = 2
   num_cols = 2
   for j_quarter in range(4):
      plt.subplot(num_rows, num_cols, j_quarter + 1)
      for j in range(N_field_lines):
         plt.scatter(Poincare_data[j][0,j_quarter:-1:4], Poincare_data[j][1,j_quarter:-1:4], s=marker_size, edgecolors='none')
      plt.xlabel('R')
      plt.ylabel('Z')
      plt.gca().set_aspect('equal',adjustable='box')
      # Turn on minor ticks, since it is necessary to get minor grid lines
      from matplotlib.ticker import AutoMinorLocator
      plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10))
      plt.gca().yaxis.set_minor_locator(AutoMinorLocator(10))
      plt.grid(which='major',linewidth=0.5)
      plt.grid(which='minor',linewidth=0.15)

   rtol = data['rtol']
   atol = data['atol']
   npoints = data['npoints']
   mpi_N_procs = data['mpi_N_procs']
   title_string = extra_str + ' Rtol='+str(rtol)+', Atol='+str(atol)+', N_procs='+str(mpi_N_procs) \
                  + ', Npoints=' + str(npoints) + ', Nlines=' + str(N_field_lines)
   plt.figtext(0.5, 0.995, title_string, fontsize=10, ha='center', va='top')

   plt.tight_layout()
   filename = 'Poincare_'+extra_str+'_rtol'+str(rtol)+'_atol'+str(atol) \
       +'_Npoints'+str(npoints)+'_Nlines'+str(N_field_lines)+'_Nprocs'+str(mpi_N_procs)+'.pdf'
   print(filename)

   if pdf:
      print('Saving PDF')
      plt.savefig(filename)
   else:
      plt.show()
