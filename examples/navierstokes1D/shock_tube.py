"""
1D Advection-Diffusion 

"""

from __future__ import absolute_import,division, print_function

import time
import numpy as np
import math
from numba import vectorize, jit

#import matplotlib.pyplot as plt
from pylab import plt

import ragnarok
from gr import pygr

plot_flag = True
plot_step = 10

def plot(variable):
    pygr.plot(variable,ylim=(-0.15,0.15),accelerate=True)
    
# ------------------------------------------------------------
# Parameters
N   = 800
nu  = 0.06
Tmax = 10000

apply_bc = True

# ------------------------------------------------------------
# Initialize solver
solver = ragnarok.NavierStokes1D(nu=nu, Nx=N)

# ------------------------------------------------------------
# Get parameters
u   = solver.u
Nx  = solver.Nx
x = solver.x

# ------------------------------------------------------------

# Step density profile
solver.rho[solver.x < Nx/2] = 1.5
solver.rho[solver.x >= Nx/2] = 1.0

# Correct equilibrium
solver.correct_feq()

# Initialize population
solver.f[:] = solver.feq.copy()

# Correct macroscopic 
solver.correct_macroscopic()

if plot_flag:
    #plot(solver.rho)
    plot(solver.u)

# ------------------------------------------------------------
# Time stepping 

for t in range(1,Tmax+1):
    if t==1: # skip JIT overhead
        startTime = time.time()
    
    # Print time
    print('T = %d' % t)

    # Plot
    if plot_flag and t % plot_step == 0:
        plot(solver.u)
    
    # Step 1: Apply bounce-back. - general (skipped 0th population)
    if apply_bc:
        solver.f[1:,0] = solver.f[[2,1],0]
        solver.f[1:,-1] = solver.f[[2,1],-1]
    
    # Step 2: Streaming / advection step: f'_i(x) <- f^n_i(x-c_i)
    solver.stream() # non-periodic

    # if apply_bc:
    #     solver.f[1:,0] = solver.f[[2,1],0]
    #     solver.f[1:,-1] = solver.f[[2,1],-1]

    # Step 3: Correct the macroscopic parameters: {rho',ux',uy',P'} = f(f',c)
    solver.correct_macroscopic()
    
    # Step 3: Correct the equilbirum population: f_eq' = f(rho',ux',uy',P')
    solver.correct_feq()
    
    # Step 4: Relaxation / collision step: f^{n+1}_i(x) <- f'_i + \alpha\beta [f^{eq}'_i(x,t) - f'_i(x,t)]
    solver.relax()
    
    if solver.rho.min() <= 0.:
        print('Density is negative!')
        break

# Done
print('It took %g seconds.' % (time.time()-startTime))
