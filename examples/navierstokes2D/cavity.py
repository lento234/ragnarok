"""
2D Navier-Stokes example


2D lid driven cavity flow

"""

from __future__ import absolute_import,division, print_function

import time
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from numba import vectorize, jit

# Lattice Boltzmann solver
import ragnarok

#plt.ioff()

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
Re  = 1000
T   = 100000#200000
U   = 0.1
Nx  = 200
Ny  = 200

# Flags
apply_bc = True
plot_step = 1000

plotFlag = True
plotSave = True

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

# ------------------------------------------------------------

# Setup initial conditions = default rho = 1.0, u = (0, 0)
solver.initialize()

ulid = 0.1
equilibriumbcdict = {'top': dict(rho=1.0, u=(-ulid,0.0))}

# ------------------------------------------------------------
# plotting

def plot(i):
    plt.figure('plot')
    plt.clf()
    vortz = curl(solver.u)
    levels = [-3,-2,-1,-0.5, 0, 0.5, 1, 2, 3, 4, 5]
    plt.contourf(x,y,vortz/ulid,levels,cmap='jet',extend='both')
    CS = plt.contour(x, y, vortz/ulid, levels, linewidths=0.5, colors='k')
    plt.clabel(CS, fontsize=6, inline=1)
    #plt.colorbar()
    plt.axis('scaled')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis([0,Nx,0,Ny])
    plt.pause(0.1)

# ------------------------------------------------------------
# Time stepping 
for t in range(T):
    if t==1: # for JIT
        startTime = time.time()
    # Print
    print('T = %d' % t)

    # Plot
    if plotFlag and t % plot_step == 0:
        plot(t)
        plt.savefig('image_t%05d.png' % (t/plot_step),dpi=300)
    
    # Step 1: Apply equilibirum b.c.
    solver.apply_equilibrium(equilibriumbcdict)

    #Step 2: Apply bounce-back
    solver.apply_bounceback(left=True,right=True,top=False,bottom=True)


    # Step 3: Streaming / advection step: f'_i(x) <- f^n_i(x-c_i)
    solver.stream()

    # Step 4: Relaxation / collision step: f^{n+1}_i(x) <- f'_i + \alpha\beta [f^{eq}'_i(x,t) - f'_i(x,t)]
    solver.relax()

    if solver.rho.min() <= 0.:
        print('Density is negative!')
        break

plt.savefig('ldcf_final.pdf')

# Done
print('It took %g seconds.' % (time.time()-startTime))
