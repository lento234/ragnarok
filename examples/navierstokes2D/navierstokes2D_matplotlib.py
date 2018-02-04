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
    - Taylor-Green vortex

"""

from __future__ import absolute_import,division, print_function

import time
import numpy as np
import math
import matplotlib.pyplot as plt
from numba import vectorize, jit
#plt.ion()

import ragnarok


# ------------------------------------------------------------
# Functions
@vectorize(['float64(float64,float64)'],target='parallel')
def calcnorm(ux,uy):
    return math.sqrt(ux**2+uy**2)

#@vectorize(['float64(float64,float64,float64,float64)'],target='parallel')
def curl(u):
    dudx,dudy = np.gradient(u[0])
    dvdx,dvdy = np.gradient(u[1])
    return dvdx - dudy
    #return (uy2-uy1) + (ux2-ux1)

def taylorgreen(t):
    lamb = 1.0
    Kx = 2.0*math.pi/(lamb*L)
    Ky = 2.0*math.pi/(lamb*L)
    K = math.sqrt(Kx*Kx + Ky*Ky)
    Ma = U/cs
    rho = 1.0 - (Ma*Ma)/(2*K*K) * (Ky*Ky*np.cos(2.0*Kx*x) + Kx*Kx*np.cos(2.0*Ky*y))
    ux = - (U*Ky)/K * np.exp(-nu*K*K*t) * np.sin(Ky*y) * np.cos(Kx*x)
    uy = - (U*Kx)/K * np.exp(-nu*K*K*t) * np.sin(Kx*x) * np.cos(Ky*y)
    return rho, ux, uy

def doublyperiodicshearlayer():
    delta = 0.05
    kappa = 80.0
    ux = np.zeros(x.shape)
    ux[y<=Ny/2.0] = U*np.tanh(kappa*(y[y<=Ny/2.0]/Ny - 0.25))
    ux[y>Ny/2.0]  = U*np.tanh(kappa*(0.75 - y[y>Ny/2.0]/Ny))
    uy = delta*U*np.sin(x/Nx + 0.25)
    return ux, uy

# ------------------------------------------------------------
# Parameters
Re  = 30000
T   = 10000

# Flags
apply_bc = True
update_plot = False
plot_step = 10

# ------------------------------------------------------------
# Initialize solver
solver = ragnarok.NavierStokes2D(Re=Re)

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
# Setup initial conditions (Taylor-green vortex)
#rho0, ux0, uy0 = taylorgreen(0.)
#solver.rho = rho0.copy()
#solver.u[0] = ux0.copy()
#solver.u[1] = uy0.copy()

# Setup initial conditions (Doubly periodic shear layer)
solver.u[0], solver.u[1] = doublyperiodicshearlayer()

# Correct equilibrium
solver.correct_feq()
# Initialize population
solver.f = solver.feq.copy()

# ------------------------------------------------------------
# plotting

# plt.figure('rho')
# plt.imshow(solver.rho)
# plt.colorbar()
# plt.axis('scaled')
# plt.show(block=False)

# plt.figure('ux0')
# plt.pcolor(x,y,solver.u[0])
# plt.colorbar()
# plt.axis('scaled')
# plt.show(block=False)


# plt.figure('ux')
# plt.pcolor(x,y,solver.u[0])
# plt.colorbar()
# plt.axis('scaled')
# plt.show(block=False)

#vortz = curl(solver.u[0][:-1,1:],solver.u[0][1:,1:],solver.u[1][1:,:-1],solver.u[1][1:,1:])
vortz = curl(solver.u)
plt.figure('vortz')
#plt.pcolor(x,y,vortz,cmap='RdBu')
plt.imshow(vortz,cmap='RdBu')
#plt.contourf(x,y,vortz,cmap='RdBu')
plt.axis('scaled')
plt.show(block=False)
plt.pause(0.1)

# ------------------------------------------------------------
# Time stepping 
Q = solver.Q
nx = solver.Nx
ny = solver.Ny
c = solver.c.T
W = solver.W

def equilibrium(rho,u):              # Equilibrium distribution function.
    cu   = 3.0 * np.dot(c,u.transpose(1,0,2))
    usqr = 3./2.*(u[0]**2+u[1]**2)
    feq = np.zeros((Q,nx,ny))
    for i in range(Q): feq[i,:,:] = rho*W[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return feq

for t in range(T):
    if t==1:
        startTime = time.time()
    print('T = %d' % t)   

    # Step A : apply out-flow boundary condition
    # Step B : calculate macroscopic density and velocity.
    # Step C : apply inlet boundary condition
    # Step D : apply other bc.
    # Step E : collision step

    solver.stream()
    solver.correct_macroscopic()
    solver.correct_feq()
    solver.relax()    
    
    
    if solver.rho.min() <= 0.:
        print('Density is negative!')
        break
    
    
    # Step 1: Streaming / advection step: f'_i(x) <- f^n_i(x-c_i)
    #solver.stream() # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    # for q in range(solver.Q): # Streaming step.
    #     solver.f[q,:,:] = np.roll(np.roll(solver.f[q,:,:],c[q,0],axis=0),c[q,1],axis=1)

    #for q in range()

    # Step 2: Correct the macroscopic parameters: {rho',ux',uy',P'} = f(f',c)
    # solver.correct_macroscopic() # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    # solver.rho = np.sum(solver.f,axis=0) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # solver.u = np.dot(c.transpose(), solver.f.transpose((1,0,2)))/solver.rho # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # for n in range(solver.D):
    #         solver.u[n] = np.sum(solver.f*np.reshape(solver.c[n],[solver.Q,1,1]),axis=0)/solver.rho


    # Step 3: Correct the equilbirum population: f_eq' = f(rho',ux',uy',P')
    # solver.correct_feq() # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    #solver.feq = equilibrium(solver.rho,solver.u) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Ax = 2.0 - np.sqrt(1.0 + 3.0*solver.u[0]**2)
    # Ay = 2.0 - np.sqrt(1.0 + 3.0*solver.u[1]**2)
    # Bx = (2.0*solver.u[0] + np.sqrt(1.0 + 3.0*solver.u[0]**2))/(1.0-solver.u[0])
    # By = (2.0*solver.u[1] + np.sqrt(1.0 + 3.0*solver.u[1]**2))/(1.0-solver.u[1])
    # for q in range(solver.Q):
    #     solver.feq[q] = solver.rho*solver.W[q]*Ax*Ay*(Bx**solver.c[0,q])*(By**solver.c[1,q])

    # Step 4: Relaxation / collision step: f^{n+1}_i(x) <- f'_i + \alpha\beta [f^{eq}'_i(x,t) - f'_i(x,t)]
    #solver.relax() # >>>>>>>>>>>>>>>>>>>>>>>>>>>
    #solver.f += solver.alpha*solver.beta*(solver.feq - solver.f)
    #solver.f = solver.f - solver.alpha*solver.beta * (solver.f - solver.feq)  # Collision step. # <<<<<<<<<<<<<<<<<<<<<<<<

    # Step 5: Correct the macroscopic parameters: {rho,ux,uy,P} = f(f^{n+1},c)
    #solver.correct_macroscopic()

    if t % 1 == 0:
        vortz = curl(solver.u)
        #plt.figure('vortz')
        plt.clf()
        #plt.pcolor(x,y,vortz,cmap='RdBu')
        plt.contourf(x,y,vortz,cmap='RdBu')
        plt.axis('scaled')
        plt.show(block=False)
        plt.pause(0.01)
    

print('It took %g seconds.' % (time.time()-startTime))
# f_comp = solver.f.copy()

# rho, ux, uy = taylorgreen(1.)
# solver.rho[:]  = rho
# solver.u[0][:] = ux
# solver.u[1][:] = uy
# # Correct equilibrium
# solver.correct_feq()

# f_anal = solver.feq.copy()


# plt.figure('u')
# rho, ux, uy = taylorgreen(T)
# #plt.imshow(calcnorm(solver.u[0],solver.u[1])-calcnorm(ux,uy))
# plt.imshow(solver.rho-rho0)
# #plt.imshow(calcnorm(solver.u[0],solver.u[1]))
# plt.colorbar()
# plt.axis('scaled')
# plt.show(block=False)
# plt.figure('ux2')
# plt.pcolor(x,y,solver.u[0])
# plt.colorbar()
# plt.axis('scaled')
# plt.show(block=False)
# 1) collide / relax
# 2) advect / stream
# 3) apply boundary condition
