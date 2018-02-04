"""
Class for 1D Navier-Stokes

# Copyright (c) 2017-2018, Lento Manickathan.
#
# This file is part of Ragnarok, which is free software distributed
# under the terms of the GPLv3 license.  A copy of the license should
# have been included in the file 'LICENSE.txt', and is also available
# online at <http://www.gnu.org/licenses/gpl-3.0.html>.
"""

from __future__ import division, absolute_import, print_function


from numba import jit, njit, prange
import numpy as np
import math


__all__ = ['NavierStokes1D']


@njit(parallel=True) 
def _calc_rho(Q,Nx,f,rho):
    for i in prange(Nx):
        rho[i] = 0.0
        for q in range(Q):
            rho[i] += f[q,i]

@njit(parallel=True) 
def _calc_u(Q,Nx,f,c,rho,u):
    for i in prange(Nx):
        u[i] = 0.0
        for q in range(Q):
            u[i] += f[q,i]*c[q]
        u[i] /= rho[i]

@njit(parallel=True)
def _correct_feq(Q,W,c,Nx,feq,rho,u):
    for i in prange(Nx):
        usqrt = math.sqrt(1.0 + 3.0*u[i]**2)
        A = 2.0 - usqrt
        B = (2.0*u[i] + usqrt)/(1-u[i])
        for q in range(Q):
            feq[q][i] = rho[i]*W[q]*A*(B**c[q])

# @njit(parallel=True)
# def _stream(Q,Nx,ftemp,buffersize,f,c):
#     for q in range(1,Q):
#         for i in prange(Nx):
#             itarget = (i+c[q]+buffersize)
#             ftemp[q,itarget] = f[q,i]
#         for i in prange(Nx):
#             f[q,i] = ftemp[q,i+buffersize]

# @njit(parallel=True)
# def _stream(Q,Nx,f,c):
#     for q in range(1,Q):
#         if c[q] < 0:
#             for i in range(Nx):
#                 itarget = (i+c[q])
#                 f[q,itarget] = f[q,i]
#         if c[q] > 0:
#             for i in range(Nx-1,-1,-1):
#                 itarget = (i+c[q])
#                 f[q,itarget] = f[q,i]

# @njit(parallel=True)
# def _stream(Q,Nx,f,c):
#     for q in range(1,Q):
#         if c[q] < 0:
#             for i in range(Nx):
#                 itarget = (i+c[q])
#                 f[q,itarget] = f[q,i]
#         if c[q] > 0:
#             for i in range(Nx-1,-1,-1):
#                 itarget = (i+c[q])
#                 f[q,itarget] = f[q,i]

@njit
def _stream(Q,Nx,f,bs,c):
    for q in prange(1,Q):
        # Determine loop order
        if c[q] > 0:
            istart, iend, istep = Nx+1-bs,-1+bs,-1
        else:
            istart, iend, istep = bs, Nx + bs, 1
        # Stream
        for i in range(istart,iend,istep):
            itarget = (i+c[q])
            f[q,itarget] = f[q,i]

#_stream(self.Q,self.Nx,self._f,self.buffersize,self.c)

@njit(parallel=True)
def _relax(Q,Nx,alpha,beta,f,feq):
    for q in range(Q):
        for i in prange(Nx):
            f[q][i] += alpha*beta*(feq[q,i]-f[q,i])


class NavierStokes1D(object):
    """
    1D Navier-Stokes Lattice Boltzmann solver
    
    Lattice: D1Q3
    
    Usage
    -----
    .. code-block:: python
        
        solver = NavierStokes1D()
        
    Parameters
    ----------
    U : float64, unit (m/s)
        characteristic velocity
    Re : float64
         Reynolds number
    Nx : int64
    """

    def __init__(self, nu, Nx):
        
        # # Reynolds number is unknown
        self.nu  = nu
        self.Nx = Nx
        self.L  = self.Nx
        
        # -----------------------------------
        self.D = 1
        self.Q = 3
        self.latticeType = 'D%dQ%d' % (self.D, self.Q)
        self.cs2 = 1.0/3.0
        self.cs = 1.0/math.sqrt(3.0)
        self.buffersize = 1
        self.c = np.array([ 0, 1, -1])

        self.alpha = 2.0
        self.beta = 1.0/(2.0*self.nu/self.cs2 + 1)
        self.W = np.array([2.0/3.0,  1.0/6.0,  1.0/6.0])
        self.x = np.arange(self.Nx)

        # Initialize population
        self._f = np.zeros((self.Q,self.Nx+self.buffersize*2))
        self._feq = np.zeros((self.Q,self.Nx+self.buffersize*2))
        #self.ftemp = np.zeros((self.Q,self.Nx+self.buffersize*2))
        self.rho = np.ones(self.Nx)
        self.u = np.zeros(self.Nx)
        self.Ma = np.zeros(self.Nx)

    @property
    def f(self):
        return self._f[:,self.buffersize:-self.buffersize]

    @property
    def feq(self):
        return self._feq[:,self.buffersize:-self.buffersize]

    def info(self):
        print(self.__doc__)

    def correct_feq_python(self):
        self.Ma[:] = self.u/self.cs
        self.feq[0][:] = 2.0*self.rho*(2.0-np.sqrt(1.0+self.u**2/self.cs2))/3.0
        self.feq[1][:] = 1.0*self.rho*(( self.u*self.c[1]-self.cs2)/(2.0*self.cs2)+np.sqrt(1.0+self.u**2/self.cs2))/3.0
        self.feq[2][:] = 1.0*self.rho*(( self.u*self.c[2]-self.cs2)/(2.0*self.cs2)+np.sqrt(1.0+self.u**2/self.cs2))/3.0

    def correct_feq(self):
        self.Ma[:] = self.u/self.cs
        _correct_feq(self.Q,self.W,self.c,self.Nx,self.feq,self.rho,self.u)

    def calc_feq_python(self,rho,u):
        Ma = u**2/self.cs2
        feq0 = 2.0*rho*(2.0-np.sqrt(1.0+u**2/self.cs2))/3.0
        feq1 = 1.0*rho*(( u*self.c[1]-self.cs2)/(2.0*self.cs2)+np.sqrt(1.0+u**2/self.cs2))/3.0
        feq2 = 1.0*rho*(( u*self.c[2]-self.cs2)/(2.0*self.cs2)+np.sqrt(1.0+u**2/self.cs2))/3.0

        return np.array([feq0, feq1, feq2])


    def correct_macroscopic_python(self):
        '''update macroscpic properties: rho'''
        self.rho[:] = np.sum(self.f,axis=0)
        self.u[:]   = np.sum(self.f*self.c.reshape(-1,1),axis=0)/self.rho

    def correct_macroscopic(self):
        '''update macroscpic properties: rho, u'''
        _calc_rho(self.Q,self.Nx,self.f,self.rho)
        _calc_u(self.Q,self.Nx,self.f,self.c,self.rho,self.u)

    def stream_python(self):
        '''perodic stream f'_i(x) <- f^n_i(x-c_i)'''
        for q in range(1,self.Q):
            self.f[q][:] = np.roll(self.f[q],self.c[q],axis=0)

    def stream(self):
        '''f'_i(x) <- f^n_i(x-c_i) non-periodic'''
        _stream(self.Q,self.Nx,self._f,self.buffersize,self.c)

    def relax_python(self):
        '''f^{n+1}_i(x) <- f'_i + 2\beta [f^{eq}'_i(x,t) - f'_i(x,t)]'''
        for q in range(self.Q):
            self.f[q][:] += self.alpha*self.beta*(self.feq[q]-self.f[q])

    def relax(self):
        '''f^{n+1}_i(x) <- f'_i + 2\beta [f^{eq}'_i(x,t) - f'_i(x,t)]'''
        _relax(self.Q,self.Nx,self.alpha,self.beta,self.f,self.feq)