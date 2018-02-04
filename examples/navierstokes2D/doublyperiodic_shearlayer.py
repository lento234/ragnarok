"""
2D Navier-Stokes example


.. math::
    
    \partial_t\rho  + \partial_{\alpha}\left( \rho\ u_{\alpha} \right) = 0
    ??? \partial_t\left( \rho\ u_{\alpha} \right) + \partial_{\alpha}\left( P_{\alpha \beta} - 2\nu \rho \partial_{\beta}u_\alpha \right) = 0

with

.. math::

    \nu = \tau c_s^2

Options:
    - Double Periodic shear layer

"""

from __future__ import absolute_import,division, print_function

import time
import numpy as np
import math
import matplotlib.pyplot as plt
from numba import vectorize, jit

import ragnarok

plotFlag = True

# Plotting
if plotFlag:
    from gr import pygr

# ------------------------------------------------------------
# Functions
@vectorize(['float64(float64,float64)'],target='parallel')
def calcnorm(ux,uy):
    return math.sqrt(ux**2+uy**2)

@jit
def curl(u):
    dudx,dudy = np.gradient(u[0])
    dvdx,dvdy = np.gradient(u[1])
    return dvdx - dudy

def doublyperiodicshearlayer():
    delta = 0.05
    kappa = 80.0
    u0 = 0.01
    ux = np.zeros(x.shape)
    ux[y<=Ny/2.0] = u0*np.tanh(kappa*(y[y<=Ny/2.0]/float(Ny) - 0.25))
    ux[y>Ny/2.0]  = u0*np.tanh(kappa*(0.75 - y[y>Ny/2.0]/float(Ny)))
    uy = delta*u0*np.sin(2*np.pi*(x/float(Nx) + 0.25))
    return ux, uy

# ------------------------------------------------------------
# Parameters
Re  = 30000
T   = 50000
U   = 0.1
Nx  = 200
Ny  = 200

# Flags
apply_bc = True
plot_step = 100

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
ux, uy = doublyperiodicshearlayer()

solver.initialize(ux=ux,uy=uy)

# ------------------------------------------------------------
# plotting

def plotsurface(variable):
    pygr.surface(variable,
        rotation=0, tilt=90,
        xlabel='x',
        ylabel='y',
        title='\omega_z',
        zlim=(-0.0025,0.0025),
        accelerate=True)

vortz = curl(solver.u)
if plotFlag: plotsurface(vortz)

# ------------------------------------------------------------
# Time stepping 

for t in range(T):
    if t==1: # for JIT
        startTime = time.time()
    
    # Print
    print('T = %d' % t)

    # Plot
    if plotFlag and t % plot_step == 0:
        vortz = curl(solver.u)
        plotsurface(vortz)

    # Step 1: Streaming / advection step: f'_i(x) <- f^n_i(x-c_i)
    #solver.stream_python()
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