"""
2D Navier-Stokes example


2D lid driven cavity flow

"""

from __future__ import absolute_import,division, print_function

# from os import environ
# environ['MPLBACKEND'] = 'module://gr.matplotlib.backend_gr'

import time
import numpy as np
import math
# import matplotlib.pyplot as plt
# plt.ion()
from numba import vectorize, jit

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
Re  = 3000
T   = 50000
U   = 0.1
Nx  = 1000
Ny  = 500

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
ux = solver.u[0]
uy = solver.u[1]

# ------------------------------------------------------------

# Setup initial conditions = default rho = 1.0, uy =  0
ux0 = 0.1*np.cos(np.pi*(y+0.5-Ny/2)/Ny) + np.random.rand(Nx,Ny)*0.1
uy0 = 0.01*np.cos(5.0*np.pi*(x+0.5-Nx/2)/Nx) + np.random.rand(Nx,Ny)*0.1

solver.initialize(ux=ux0,uy=uy0)

# ------------------------------------------------------------
# plotting

def plotsurface(variable,zlim=(0.,0.1)):
    pygr.surface(variable,
        # rotation=0, tilt=90,
        #rotation=45, tilt=45,colormap=1,
        rotation=0, tilt=90,colormap=44,
        xlabel='x',
        ylabel='y',
        title='unorm',
        zlim=zlim,
        accelerate=True)

#vortz = curl(solver.u)
unorm = calcnorm(ux,uy)
if plotFlag: plotsurface(unorm)

#g = Re**2*0.01**2/100**3

DeltaU = g = Re**2*0.01**2/100**3
force = DeltaU * 1.0
#force = np.array([0.0001,0.0])

# ------------------------------------------------------------
# Time stepping 

for t in range(T):
    if t==1: # for JIT
        startTime = time.time()
    
    # Print
    print('T = %d' % t)

    # Plot
    if plotFlag and t % plot_step == 0:
        #vortz = curl(solver.u)
        unorm = calcnorm(ux,uy)
        #plotsurface(vortz)
        plotsurface(unorm)

    # Step 1: Apply equilibrium b.c at (x=all,y=N)
    # for q in range(solver.Q):
    #     solver.f[q,:,-1] = feq0[q]
    
    #solver.apply_equilibrium(equilibriumbcdict)

    #Step 2: Apply bounce-back
    solver.apply_bounceback(left=False,right=False,top=True,bottom=True)
    
    # Step 3: Streaming / advection step: f'_i(x) <- f^n_i(x-c_i)
    solver.stream()
    
    # Step 3.5: Apply boundary condition
    solver.apply_periodic(left_right=True,top_bottom=False)
    
    # Step 4: Relaxation / collision step: f^{n+1}_i(x) <- f'_i + \alpha\beta [f^{eq}'_i(x,t) - f'_i(x,t)]
    solver.relax(force=force)

    if solver.rho.min() <= 0.:
        print('Density is negative!')
        break


# Done
print('It took %g seconds.' % (time.time()-startTime))
