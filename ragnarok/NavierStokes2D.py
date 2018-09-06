"""
Class for 2D Navier-Stokes

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


__all__ = ['NavierStokes2D']


@njit(parallel=True) 
def _calc_rho(Q,Nx,Ny,f,rho):
    for i in prange(Nx):
        for j in prange(Ny):
            rho[i,j] = 0.0
            for q in range(Q):
                rho[i,j] += f[q,i,j]

@njit(parallel=True)   
def _calc_u(D,Q,Nx,Ny,f,c,rho,u):
    for i in prange(Nx):
        for j in prange(Ny):
            for n in range(D):
                u[n,i,j] = 0.0
                for q in range(Q):
                    u[n,i,j] += f[q,i,j]*c[n,q]
                u[n,i,j] /= rho[i,j]

@njit(parallel=True)    
def _calc_P(D,Q,Nx,Ny,f,c,rho,P):
    for i in prange(Nx):
        for j in prange(Ny):
            for nx in range(D): 
                for ny in range(D):
                    P[nx,ny,i,j] = 0.
                    for q in range(Q):
                        P[nx,ny,i,j] += f[q,i,j]*c[nx,q]*c[ny,q]  
                    #P[nx,ny,i,j] /= rho[i,j]


@njit(parallel=True)
def _relax(Q,W,c,Nx,Ny,alpha,beta,f,rho,u,force):
    for i in prange(Nx):
        for j in prange(Ny):

            # Velocity
            ux = u[0,i,j]
            uy = u[1,i,j]

            # Feq coefficients
            uxsqrt = math.sqrt(1.0 + 3.0*ux**2)
            uysqrt = math.sqrt(1.0 + 3.0*uy**2)
            Ax = 2.0 - uxsqrt
            Ay = 2.0 - uysqrt
            Bx = (2.0*ux + uxsqrt)/(1.0-ux)
            By = (2.0*uy + uysqrt)/(1.0-uy)

            # Feqf coefficients
            ux = ux + force[0]/rho[i,j]
            uy = uy + force[1]/rho[i,j]
            uxsqrt = math.sqrt(1.0 + 3.0*ux**2)
            uysqrt = math.sqrt(1.0 + 3.0*uy**2)
            Axf = 2.0 - uxsqrt
            Ayf = 2.0 - uysqrt
            Bxf = (2.0*ux + uxsqrt)/(1.0-ux)
            Byf = (2.0*uy + uysqrt)/(1.0-uy)

            for q in range(Q):
                feq = rho[i,j]*W[q]*Ax*Ay*(Bx**c[0,q])*(By**c[1,q])
                feqf = rho[i,j]*W[q]*Axf*Ayf*(Bxf**c[0,q])*(Byf**c[1,q])
                f[q,i,j] += alpha*beta*(feq-f[q,i,j]) + (feqf - feq)
        

@njit(parallel=True)
def _stream(Q,Nx,Ny,f,bs,c):
    for q in prange(1,Q):
        # Determine loop order
        if c[0,q] > 0:
            istart, iend, istep = Nx+1-bs,-1+bs,-1
        else:
            istart, iend, istep = bs, Nx + bs, 1
        if c[1,q] > 0:
            jstart, jend, jstep = Ny+1-bs,-1+bs,-1
        else:
            jstart, jend, jstep = bs, Ny + bs, 1

        # Stream
        for i in range(istart,iend,istep):
            for j in range(jstart,jend,jstep):
                itarget = (i+c[0,q])
                jtarget = (j+c[1,q])
                f[q,itarget,jtarget] = f[q,i,j]


                        
class NavierStokes2D(object):
    """
    2D Navier-Stokes Lattice Boltzmann solver
    
    Lattice: D2Q9
    
    Usage
    -----
    .. code-block:: python
        
        solver = NavierStokes2D()
        
    Parameters
    ----------
    U : float64, unit (m/s)
        characteristic velocity
    Re : float64
         Reynolds number
    Nx : int64
         
    Ny : int64
        
    """

    def __init__(self, U, Re, Nx, Ny):
        
        self.Nx = Nx
        self.Ny = Ny
        self.L  = max(self.Nx,self.Ny)
        self.U  = U
        self.Re = Re
        self.nu = self.U*self.L/self.Re

        # -----------------------------------            
        # Setup 2D lattice - D2Q9
        self.D = 2
        self.Q = 9
        # 0: centre
        # 1: right, 2: top, 3: left, 4: bottom
        # 5: top-right, 6: top-left, 7: bottom-left, 8: bottom-right
        self.c = np.array([[ 0, 1, 0, -1,  0, 1, -1, -1,  1],
                           [ 0, 0, 1,  0, -1, 1,  1, -1, -1]])
        self.W = np.array([4.0/9.0,  1.0/9.0,  1.0/9.0,  1.0/9.0, 1.0/9.0,
                           1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])        
        self.latticeType = 'D%dQ%d' % (self.D, self.Q)
        self.cs2 = 1.0/3.0
        self.cs = 1.0/math.sqrt(3.0)
        self.buffersize = 1
        self.Dt = 1.0
        self.Dx = 1.0
        
        # -----------------------------------            
        # Calculated parameters
        self.alpha = self._calc_alpha()
        self.beta = self._calc_beta()
        self.tau = self._calc_tau()
        self.x = np.array(np.meshgrid(np.arange(self.Nx),
                                      np.arange(self.Ny),indexing='ij'))
        
        # Initialize population
        self._f = np.zeros((self.Q,self.Nx+2*self.buffersize,self.Ny+2*self.buffersize))
        self.feq = np.zeros((self.Q,self.Nx,self.Ny))
        self.rho = np.ones((self.Nx,self.Ny))
        self.u = np.zeros((self.D,self.Nx,self.Ny))
        self.P = np.ones((self.D,self.D,self.Nx,self.Ny))
        self.H = np.ones((self.Nx,self.Ny))

        self.force = np.zeros(self.D)

        self.M = np.zeros((self.D+1,self.D+1,self.Nx,self.Ny))

        # Assert 
        assert(U <= 0.1), 'U should be smaller than 0.1'
        assert(self.beta >= 0 and self.beta <= 1), '0 <= beta <= 1'
        
        print('nu = %f' % self.nu)
        print('beta = %f' % self.beta)
        print('omega = %f' % (self.alpha*self.beta))

    def initialize(self,rho=None,ux=None,uy=None):
        ''' Initialize the density, velocity and population '''
        if rho is not None:
            assert(rho.dtype == np.float64), 'rho is not %s' % np.float64
            self.rho[:,:] = rho[:,:]
        if ux is not None:
            assert(ux.dtype == np.float64), 'ux is not %s' % np.float64
            self.u[0,:,:] = ux[:,:]
        if uy is not None:
            assert(uy.dtype == np.float64), 'uy is not %s' % np.float64
            self.u[1,:,:] = uy[:,:]            
        
        # Initialize population
        self.f[:,:,:] = self.calc_feq_python(self.rho,self.u) #self.feq[:,:,:]

        # Correct macroscopic 
        self.correct_macroscopic() # redundant probably

    def correct_macroscopic(self):
        '''update macroscpic properties: rho, ux, uy, P'''
        _calc_rho(self.Q,self.Nx,self.Ny,self.f,self.rho)
        _calc_u(self.D,self.Q,self.Nx,self.Ny,self.f,self.c,self.rho,self.u)
        #_calc_P(self.D,self.Q,self.Nx,self.Ny,self.f,self.c,self.rho,self.P)

    def stream(self):
        '''f'_i(x) <- f^n_i(x-c_i) non-periodic'''
        _stream(self.Q,self.Nx,self.Ny,self._f,self.buffersize,self.c)

    def relax(self,force=None):
        '''f^{n+1}_i(x) <- f'_i + 2\beta [f^{eq}'_i(x,t) - f'_i(x,t)]'''
        # Correct macroscopic properties: rho, u, P
        self.correct_macroscopic()
        
        if force is not None:
            self.force[:] = force
    
        # Calculate Delta_u due to the external force
        _relax(self.Q,self.W,self.c,self.Nx,self.Ny,self.alpha,self.beta,self.f,self.rho,self.u,self.force)
    

    def apply_equilibrium(self,bcdict):
        for bckey,bc in bcdict.items():
            feq0 = self.calc_feq_python(bc['rho'],bc['u'])
            istart,iend,jstart,jend = None,None,None,None
            if bckey == 'top':
                jstart = -1
            elif bckey == 'bottom':
                jend = 1
            elif bckey == 'left':
                iend = 1
            elif bckey == 'right':
                istart = -1
            for q in range(self.Q):
                self.f[q,istart:iend,jstart:jend] = feq0[q]


    def apply_periodic(self,left_right=True,top_bottom=True):
        '''
        Apply boundary conditions:

        periodic (1D): 
            f_+(0,t) <- f_+(N,t) and f_-(N,t) <- f_-(0,t)
        bounce-back (1D):
            f_+(0,t) <- f_-(0,t) and f_-(N,t) <- f_+(N,t)
        equilibrium bc:
            f_i(x,t) <- f^{eq}_i (rho_0,u_0)
        '''
        # 0: centre
        # 1: right, 2: top, 3: left, 4: bottom
        # 5: top-right, 6: top-left, 7: bottom-left, 8: bottom-right
        
        if not left_right and not top_bottom:
            print('Warning: no periodic b.c applied')
        
        bs = self.buffersize
        
        if left_right:
            # Left wall: i=0,j=all
            # populations: 1-right, 5-topright, 8-bottomright
            self.f[1][0,:] = self._f[1][-bs,bs:-bs] # f_+(0,y) <- f_+(N,y)
            self.f[5][0,1:] = self._f[5][-bs,bs+1:-bs] # f_
            self.f[5][0,0] = self._f[5][-bs,-bs]
            self.f[8][0,:-1] = self._f[8][-bs,bs:-1-bs]
            self.f[8][0,-1] = self._f[8][-bs,0]

            # Right wall: i=-1,j=all
            # populations: 3-left, 6-topleft, 7-bottomleft
            self.f[3][-1,:] = self._f[3][0,bs:-bs] # f_+(0,y) <- f_+(N,y)
            self.f[6][-1,1:] = self._f[6][0,bs+1:-bs] # f_
            self.f[6][-1,0] = self._f[6][0,-bs]
            self.f[7][-1,:-1] = self._f[7][0,bs:-1-bs]
            self.f[7][-1,-1] = self._f[7][0,0]

        if top_bottom:
            # Bottom wall: i=all, j=0
            # populations: 2-top, 5-topright, 6-topleft
            self.f[2][:,0] = self._f[2][bs:-bs,-bs]
            self.f[5][1:,0] = self._f[5][bs+1:-bs,-bs]
            #self.f[5][0,0] = self.ftemp[5][-bs,-bs] # redundant
            self.f[6][:-1,0] = self._f[6][bs:-1-bs,-bs]
            #self.f[6][-1,0] = self.ftemp[6][0,-bs] # redundant

            # Top wall: i=all, j=-1
            # populations: 4-bottom, 7-bottomleft, 8-bottomright
            self.f[4][:,-1] = self._f[4][bs:-bs,0]
            self.f[7][:-1,-1] = self._f[7][bs:-bs-1,0]
            #self.f[7][-1,-1] = self.ftemp[7][0,0] # redundant
            self.f[8][1:,-1] = self._f[8][bs+1:-bs,0]
            #self.f[8][0,-1] = self.ftemp[8][-bs,0] # redundant

    def apply_bounceback(self,left=True,right=True,top=True,bottom=True):
        ''' apply bounce-back b.c 
        
        bounce-back (1D):
            f_+(0,t) <- f_-(0,t) and f_-(N,t) <- f_+(N,t)
        '''
        if not left and not right and not top and not bottom:
            print('Warning: no bounce-back b.c applied')

        # 0: centre
        # 1: right, 2: top, 3: left, 4: bottom
        # 5: top-right, 6: top-left, 7: bottom-left, 8: bottom-right   

        # Bounce-back
        # center (0)
        # right (1) --> left (3)
        # top (2) --> bottom (4)
        # left (3) --> right (1)
        # bottom (4)
        # top-right (5) --> top-left (6)
        # top-left (6) --> top-right (5)
        # bottom-left (7) --> bottom-right (8)
        # bottom-right (8) --> bottom left (7)

        #bs = self.buffersize
        reorder = [0,3,4,1,2,7,8,5,6]

        if left:
            self.f[:,0,1:-1] = self.f[reorder,0,1:-1]
        if right:
            self.f[:,-1,1:-1] = self.f[reorder,-1,1:-1]
        if bottom:
            self.f[:,:,0] = self.f[reorder,:,0]
        if top:
            self.f[:,:,-1] = self.f[reorder,:,-1]
       

    # ------------------------------------------   

    def calc_feq_python(self,rho,u):
        '''
        calc f_eq for a given scalar rho and u
        '''
        ux = u[0]
        uy = u[1]
        uxsqrt = np.sqrt(1.0 + 3.0*ux**2)
        uysqrt = np.sqrt(1.0 + 3.0*uy**2)
        Ax = 2.0 - uxsqrt
        Ay = 2.0 - uysqrt
        Bx = (2.0*ux + uxsqrt)/(1.0-ux)
        By = (2.0*uy + uysqrt)/(1.0-uy)
        if type(rho) == np.ndarray or type(u) == np.ndarray:
            feq = np.zeros((self.Q,self.Nx,self.Ny))
        else:
            feq = np.zeros(self.Q)
        for q in range(self.Q):
            feq[q] = rho*self.W[q]*Ax*Ay*(Bx**self.c[0,q])*(By**self.c[1,q])
        return feq

    def _calc_alpha(self):
        '''entropic stabilizer'''
        return np.float64(2.0)
    
    def _calc_beta(self):
        '''relaxation parameter'''
        return np.float64(1.0/(2.0*self.nu/self.cs2+1))
        
    def _calc_tau(self):
        '''relaxation time'''
        return self.nu/self.cs2

    def _calc_nu(self,beta):
        '''nu from beta'''
        return np.float64(0.5*self.cs2*(1.0/beta-1.0))
    
    def info(self):
        print(self.__doc__)

    # Properties
    @property
    def f(self):
        return self._f[:,self.buffersize:-self.buffersize,self.buffersize:-self.buffersize]





# --------------------------------------------------------
# Work in progress part

"""



    def correct_macroscopic_python(self):
        '''update macroscpic properties: rho, ux, uy, P'''
        self.rho[:,:] = np.sum(self.f,axis=0)
        self.u[0,:,:] = np.sum(self.f * np.reshape(self.c[0], (self.Q,1,1)), axis=0) / self.rho
        self.u[1,:,:] = np.sum(self.f * np.reshape(self.c[1], (self.Q,1,1)), axis=0) / self.rho
        

    def correct_feq_python(self):
        # A = [(2.0 - np.sqrt(1.0 + 3.0*self.u[n]**2)) for n in range(self.D)]
        # B = [((2.0*self.u[n] + np.sqrt(1.0 + 3.0*self.u[n]**2))/(1.0-self.u[n])) for n in range(self.D)]
        # for q in range(self.Q):
        #    self.feq[q] = self.rho*self.W[q]*A[0]*A[1]*(B[0]**self.c[0,q])*(B[1]**self.c[1,q])
        ####
        #cu = 2.0 * (self.u[0]*self.c[0].reshape(-1,1,1) + self.u[1]*self.c[1].reshape(-1,1,1))
        #usqr = 3./2.*(self.u[0]**2 + self.u[1]**2)
        cu = self.u[0]*self.c[0].reshape(-1,1,1) + self.u[1]*self.c[1].reshape(-1,1,1)
        usqr = self.u[0]**2 + self.u[1]**2
        for q in range(self.Q):
            self.feq[q] = self.rho*self.W[q]*(1.0 + 3.0*cu[q] + 4.5*cu[q]**2.0 - 1.5*usqr)
            #self.feq[q] = self.rho*self.W[q]*(1.0 + cu[q]+0.5*cu[q]**2-usqr)

    def relax_python(self):
        '''f^{n+1}_i(x) <- f'_i + 2\beta [f^{eq}'_i(x,t) - f'_i(x,t)]'''
        for q in range(self.Q):
            self.f[q] += self.alpha*self.beta*(self.feq[q]-self.f[q])

    def stream_python(self):
        '''periodic stream: f'_i(x) <- f^n_i(x-c_i)'''
        for q in range(1,self.Q):
            self.f[q] = np.roll(np.roll(self.f[q],self.c[0,q],axis=0),self.c[1,q],axis=1)
        

@njit(parallel=True)
def _H(H,Q,W,Nx,Ny,f):
    for i in prange(Nx):
        for j in prange(Ny):
            H[i,j] = 0.0
            for q in range(Q):
                H[i,j] += f[q,i,j]*math.log(f[q,i,j]/W[q])

@njit(parallel=True)
def _M(D,Q,Nx,Ny,f,c,rho,M):
    for i in prange(Nx):
        for j in prange(Ny):
            for nx in range(D+1):
                for ny in range(D+1):
                    M[nx,ny,i,j] = 0.0
                    for q in range(Q):
                        M[nx,ny,i,j] += f[q,i,j]*(c[0,q]**nx)*(c[1,q]**ny)
                    #M[nx,ny,i,j] /= rho[i,j]
                    if nx>0 or ny>0:
                        M[nx,ny,i,j] /= M[0,0,i,j]



    def calc_H(self):
        _H(self.H,self.Q,self.W,self.Nx,self.Ny,self.f)
  
    def calc_M(self):
        _M(self.D,self.Q,self.Nx,self.Ny,self.f,self.c,self.rho,self.M)


    def calc_f_from_M(self):
        '''
        (0,0)   <-- 0: centre 
        (1,0)   <-- 1: right
        (0,1)   <-- 2: top
        (-1,0)  <-- 3: left
        (0,-1)  <-- 4: bottom

        (1,1)   <-- 5: top-right
        (-1,1)  <-- 6: top-left
        (-1,-1) <-- 7: bottom-left
        (1,-1)  <-- 8: bottom-right
        '''
        rho = self._rho
        ux = self._ux
        uy = self._uy
        T = self._T
        N = self._N
        Pixy = self._Pixy
        Qxyy = self._Qxyy
        Qyxx = self._Qyxx
        A = self._A

        self.f[0][:,:] = rho*(1.0 - T + A) # f_(0,0)

        self.f[1][:,:] = 0.5*rho*(0.5*(T+N) + (1.0)*ux - (1.0)*Qxyy - A) # f_(1,0)
        self.f[3][:,:] = 0.5*rho*(0.5*(T+N) + (-1.0)*ux - (-1.0)*Qxyy - A) # f_(-1,0)

        self.f[2][:,:] = 0.5*rho*(0.5*(T-N) + (1.0)*uy - (1.0)*Qyxx - A) # f_(0,1)
        self.f[4][:,:] = 0.5*rho*(0.5*(T-N) + (-1.0)*uy - (-1.0)*Qyxx - A) # f_(0,-1)

        self.f[5][:,:] = 0.25*rho*(A + (1.0*1.0)*Pixy + (1.0)*Qxyy + (1.0)*Qyxx) # f_(1,1)
        self.f[6][:,:] = 0.25*rho*(A + (-1.0*1.0)*Pixy + (-1.0)*Qxyy + (1.0)*Qyxx) # f_(-1,1)
        self.f[7][:,:] = 0.25*rho*(A + (-1.0*-1.0)*Pixy + (-1.0)*Qxyy + (-1.0)*Qyxx) # f_(-1,-1)
        self.f[8][:,:] = 0.25*rho*(A + (1.0*-1.0)*Pixy + (1.0)*Qxyy + (-1.0)*Qyxx) # f_(1,-1)



    @property
    def _T(self):
        return self.M[2,0] + self.M[0,2]

    @property
    def _N(self):
        return self.M[2,0] - self.M[0,2]

    @property
    def _Pixy(self):
        return self.M[1,1]

    @property
    def _Qxyy(self):
        return self.M[1,2]

    @property
    def _Qyxx(self):
        return self.M[2,1]

    @property
    def _A(self):
        return self.M[2,2]

    @property
    def _Pixy_t(self):
        return self._Pixy - self._ux*self._uy

    @property
    def _N_t(self):
        return self._N - (self._ux**2 - self._uy**2)

    @property
    def _T_t(self):
        return self._T - (self._ux**2 + self._uy**2)

    @property
    def _Qxyy_t(self):
        return self._Qxyy - (2.0*self._uy*self._Pixy_t + self._ux*(-0.5*self._N_t + 0.5*self._T_t + self._uy**2))

    @property
    def _Qyxx_t(self):
        return self._Qyxx - (2.0*self._ux*self._Pixy_t + self._uy*( 0.5*self._N_t + 0.5*self._T_t + self._ux**2))

    @property
    def _A_t(self):
        return self._A - (  2.0*(self._ux*self._Qxyy_t + self._uy*self._Qyxx_t)
                          + 4.0*self._ux*self._uy*self._Pixy_t
                          + 0.5*(self._ux**2 + self._uy**2)*self._T_t
                          - 0.5*(self._ux**2 - self._uy**2)*self._N_t
                          + (self._ux**2)*(self._uy**2) )

    @property
    def _rho(self):
        return self.M[0,0]
    
    @property
    def _ux(self):
        return self.M[1,0]

    @property
    def _uy(self):
        return self.M[0,1]

"""
