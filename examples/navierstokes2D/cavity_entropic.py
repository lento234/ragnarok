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
#import matplotlib.pyplot as plt
#plt.ion()
from numba import vectorize, jit, prange

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
T   = 10000
U   = 0.1
Nx  = 100
Ny  = 100

# Flags
apply_bc = True
plot_step = 5

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

equilibriumbcdict = {'top': dict(rho=1.0, u=(0.01,0.0))}

# ------------------------------------------------------------
# plotting

# def plotsurface(variable):
#          pygr.surface(x.T[0],y[0],variable,
#          rotation=0, tilt=90,colormap=44,
#          xlabel='x',
#          ylabel='y',
#          title='\omega_z',
#          zlim=(-0.005,0.005),
#          accelerate=True)

#vortz = curl(solver.u)
# unorm = calcnorm(ux,uy)
# if plotFlag: plotsurface(unorm)



# ------------------------------------------------------------
# Time stepping 


for t in range(1,T+1):
    if t==1: # for JIT
        startTime = time.time()
    
    # Print
    print('T = %d' % t)

    # Plot
    # if plotFlag and t % plot_step == 0:
    #     #vortz = curl(solver.u)
    #     unorm = calcnorm(ux,uy)
    #     #plotsurface(vortz)
    #     plotsurface(unorm)
    
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


# ------------------------------------------------------------
# Entropic lattice boltzmann

def calc_H(Q,W,f):
    H = np.zeros(f.shape[1:])
    for q in range(Q):
        H[:] += f[q]*np.log(f[q]/W[q])
    return H
    
f = solver.f
Q = solver.Q
W = solver.W
rho = solver.rho
u = solver.u
feq = solver.calc_feq_python(rho,u)



@jit(parallel=True)
def calc_alpha(Nx,Ny,Q,f,feq,alpha):
    for i in prange(Nx):
        for j in prange(Ny):
            alphaMax = 1.0
            ratio = 0.0
            for q in range(Q):
                Delta = feq[q,i,j]-f[q,i,j]
            #if Delta < 0:
                alphaMax = min(alphaMax,-f[q,i,j]/Delta)
                ratio = max(ratio,abs(Delta/f[q,i,j]))
            if alphaMax >= 1.0:# and alphaMax < 2.0:
                alpha[i,j] = alphaMax
            if ratio < 1e-4:
                alpha[i,j] = 2.0
        

#H = calc_H(Q,W,f)
#Delta = feq-f
#alpha = np.ones((Nx,Ny))
#maskOpt2 = np.max(Delta/f,axis=0) < 1e-4
#alpha[maskOpt2] = 2.0
#alphaMax = np.max((np.min(-f/Delta,axis=0),f[0]*0+1),axis=0)
#maskOpt1 = alphaMax < 2.0
#alpha[maskOpt1] = alphaMax[maskOpt1]
#alpha[maskOpt2] = 2.0

# alpha = np.ones((Nx,Ny))
# calc_alpha(Nx,Ny,Q,f,feq,alpha)

solver.calc_M()

pygr.surface(solver.M[1,0],rotation=0, tilt=90,colormap=1)


