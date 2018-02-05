"""
2D Navier-Stokes example

    - Taylor green vortex

"""

from __future__ import absolute_import,division, print_function

import time
import numpy as np
import math
import matplotlib.pyplot as plt
from numba import vectorize, jit

# Lattice boltzmann solver
import ragnarok

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

def taylorgreen(t=0.0):
    lambdax = lambday = 1.0
    Vmax = 0.05
    Ma = Vmax / cs

    Kx = 2.0*np.pi/(lambdax*Nx)
    Ky = 2.0*np.pi/(lambday*Ny)
    Ksquared = Kx**2 + Ky**2

    rho = 1.0 - (Ma**2/(2.0*Ksquared))*(Ky*Ky*np.cos(2.0*Kx*x) + Kx*Kx*np.cos(2.0*Ky*y))
    ux = - (Vmax * Ky / np.sqrt(Ksquared)) * np.exp(-nu * Ksquared * t) * np.sin(Ky*y) * np.cos(Kx * x)
    uy = (Vmax * Kx / np.sqrt(Ksquared)) * np.exp(-nu * Ksquared * t) * np.sin(Kx*x) * np.cos(Ky * y)
    
    return rho, ux, uy

# ------------------------------------------------------------
# Parameters
Re  = 5000
T   = 10000
U   = 0.1
Nx  = 100
Ny  = 100

# Flags
apply_bc = True
plot_step = 1000
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
ux = solver.u[0]
uy = solver.u[1]
rho = solver.rho

# ------------------------------------------------------------

# Setup initial conditions (Doubly periodic shear layer)
rho0, ux0, uy0 = taylorgreen(t=0.0)

solver.initialize(rho=rho0,ux=ux0,uy=uy0)

# ------------------------------------------------------------
# plotting

def plot(i):
    plt.figure('plot')
    plt.clf()
    vortz = curl(solver.u)
    levels = np.linspace(-0.05,0.05,11)
    plt.title('T = %d' % i)
    plt.contourf(x,y,vortz/0.05,levels,cmap='RdBu',extend='both')
    plt.colorbar()
    skip=5
    plt.quiver(x[::skip,::skip],y[::skip,::skip],ux[::skip,::skip],uy[::skip,::skip],scale=0.7)
    plt.axis('scaled')
    plt.axis([0,Nx,0,Ny])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.pause(0.1)

k = 0
if plotFlag:
    plot(0)
    plt.savefig('taylor_green_%04d.png' % 0, dpi=400)

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
        plt.savefig('taylor_green_%04d.png' % t, dpi=400)

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