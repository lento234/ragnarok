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
    pygr.plot(variable,ylim=(0.9,1.7),accelerate=True)

# ------------------------------------------------------------
# Parameters
N   = 800
u   = 0.1
nu   = 5e-8
Tmax = 4000

# ------------------------------------------------------------
# Initialize solver
solver = ragnarok.AdvectionDiffusion1D(u=u, nu=nu, Nx=N)

# ------------------------------------------------------------
# Get parameters
u   = solver.u
Nx  = solver.Nx
x = solver.x

# ------------------------------------------------------------

# Steep gaussian profile
rho0 = 1 + 0.5*np.exp(-5000*(np.arange(N)/float(N) - 0.25)**2 )
solver.rho = rho0

# Correct equilibrium
solver.correct_feq_python()

# Initialize population
solver.f = solver.feq.copy()

# Correct macroscopic 
#solver.correct_macroscopic()
solver.correct_macroscopic_python()

if plot_flag:
    plot(solver.rho)

# ------------------------------------------------------------
# Time stepping 

for t in range(Tmax):
    if t==1: # skip JIT overhead
        startTime = time.time()
    
    # Print time
    print('T = %d' % t)

    # Plot
    if plot_flag and t % plot_step == 0:
        plot(solver.rho)
   
    # Step 1: Streaming / advection step: f'_i(x) <- f^n_i(x-c_i)
    solver.stream_python()
    
    # Step 2: Correct the macroscopic parameters: {rho',ux',uy',P'} = f(f',c)
    solver.correct_macroscopic_python()
    
    # Step 3: Correct the equilbirum population: f_eq' = f(rho',ux',uy',P')
    solver.correct_feq_python()
    
    # Step 4: Relaxation / collision step: f^{n+1}_i(x) <- f'_i + \alpha\beta [f^{eq}'_i(x,t) - f'_i(x,t)]
    solver.relax_python()
    
    if solver.rho.min() <= 0.:
        print('Density is negative!')
        break

# Done
print('It took %g seconds.' % (time.time()-startTime))
