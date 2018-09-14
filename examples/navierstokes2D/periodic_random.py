"""
2D Navier-Stokes example

   - 2D turbulence cascade

"""

from __future__ import absolute_import,division, print_function

import time
import numpy as np
import math
#import matplotlib.pyplot as plt
from numba import vectorize, jit
import gr
gr.inline('mov')
from gr import pygr

import ragnarok


# ------------------------------------------------------------
# Functions
@vectorize(['float64(float64,float64)'],target='parallel')
def calcnorm(ux,uy):
    return math.sqrt(ux**2+uy**2)

@jit
def curl(u):
    dudx,dudy = np.gradient(u[0], 1.0/Nx,1.0/Ny)
    dvdx,dvdy = np.gradient(u[1], 1.0/Nx,1.0/Ny)
    return dvdx - dudy

# ------------------------------------------------------------
# Parameters
Re  = 50000
T   = 10000
U   = 0.1
Nx  = 200
Ny  = 200

# Flags
apply_bc = True
plot_step = 100
plotSave = False
plotFlag = True

# ------------------------------------------------------------
# Initialize solver
solver = ragnarok.NavierStokes2D(U=U, Re=Re,
                                 Nx=Nx, Ny=Ny)

# ------------------------------------------------------------
# Get parameters
L   = solver.L
U   = solver.U
nu  = solver.nu
Nx  = solver.Nx
Ny  = solver.Ny
cs  = solver.cs
x = solver.x[0]
y = solver.x[1]

# ------------------------------------------------------------

# Setup initial conditions (Doubly periodic shear layer)
#ux, uy = doublyperiodicshearlayer()
np.random.seed(2018)
import scipy.ndimage as spndim
ux = spndim.filters.gaussian_filter(U*(np.random.rand(Nx,Ny)-0.5)*2,2.5)
uy = spndim.filters.gaussian_filter(U*(np.random.rand(Nx,Ny)-0.5)*2,2.5)

solver.initialize(ux=ux,uy=uy)

# ------------------------------------------------------------
# plotting

def plot(i):
    vortz = curl(solver.u)
    pygr.surface(vortz,rotation=0, tilt=90,colormap=44,
                 zlim=(-0.5,0.5),accelerate=True)

# ------------------------------------------------------------
# Time stepping 

for t in range(T+1):
    if t==1: # for JIT
        startTime = time.time()
    
    # Print
    print('T = %d' % t)

    # Plot
    if plotFlag and t % plot_step == 0:
        plot(t)

    # Step 1: Streaming / advection step: f'_i(x) <- f^n_i(x-c_i)
    solver.stream()
    
    # Step 2: Apply boundary condition
    solver.apply_periodic()
    
    # Step 3: Relaxation / collision step: f^{n+1}_i(x) <- f'_i + \alpha\beta [f^{eq}'_i(x,t) - f'_i(x,t)]
    solver.relax()
    
    if solver.rho.min() <= 0.:
        print('Density is negative!')
        break


# Done
print('It took %g seconds.' % (time.time()-startTime))
