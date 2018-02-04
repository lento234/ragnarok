"""
2D Navier-Stokes example


2D lid driven cavity flow

"""

from __future__ import absolute_import,division, print_function

#from os import environ
#environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'
#environ['GKS_WSTYPE'] = 'cairox11'

import time
import numpy as np
import math
import matplotlib.pyplot as plt
#plt.ion()
from numba import vectorize, jit
import pyevtk
import ragnarok

#import gr
#gr.inline('mov')

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

# ------------------------------------------------------------
# Parameters
Re  = 1000
T   = 500000
U   = 0.1#0.05
Nx  = 250
Ny  = 250

# Flags
apply_bc = True
plot_step = 500

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

equilibriumbcdict = {'top': dict(rho=1.0, u=(0.1,0.0))}

# ------------------------------------------------------------
# plotting

# def plotsurface(variable):
#          pygr.surface(x.T[0],y[0],variable,
#          rotation=0, tilt=90,#colormap=42,
#          xlabel='x',
#          ylabel='y',
#          title='unorm',
#          zlim=(0, 0.01),
#          accelerate=True)

#vortz = curl(solver.u)
#unorm = calcnorm(ux,uy)
#if plotFlag: plotsurface(vortz)
#if plotFlag: plotsurface(unorm)

def plotcontourf():
    plt.figure('plot')
    plt.clf()
    vortz = curl(solver.u)
    #skip = 5
    levels = [-4,-3,-2,-1,0,1,2,3,4,5,6]
    plt.contourf(x,y,-vortz/(0.1*(1.0/Nx)),levels,cmap='jet',extend='both')
    # plt.quiver(x[::skip,::skip],y[::skip,::skip],
    #             ux[::skip,::skip],uy[::skip,::skip])
    plt.colorbar()
    plt.axis('scaled')
    plt.axis([0,Nx,0,Ny])
    plt.show(block=False)
    plt.pause(0.1)

plotcontourf()

# ------------------------------------------------------------
# Time stepping 
k = 1
for t in range(T):
    if t==1: # for JIT
        startTime = time.time()
    
    # Print
    print('T = %d' % t)

    # Plot
    if plotFlag and t % plot_step == 0:
        k += 1
        plotcontourf()
    
    # Step 3: Streaming / advection step: f'_i(x) <- f^n_i(x-c_i)
    solver.stream()
    
    solver.apply_equilibrium(equilibriumbcdict)

    #Step 2: Apply bounce-back
    solver.apply_bounceback(left=True,right=True,top=False,bottom=True)
    
    # Step 3.5: Apply boundary condition
    #solver.apply_periodic()
    
    # Step 4: Relaxation / collision step: f^{n+1}_i(x) <- f'_i + \alpha\beta [f^{eq}'_i(x,t) - f'_i(x,t)]
    solver.relax()

    if solver.rho.min() <= 0.:
        print('Density is negative!')
        break


# Done
print('It took %g seconds.' % (time.time()-startTime))
