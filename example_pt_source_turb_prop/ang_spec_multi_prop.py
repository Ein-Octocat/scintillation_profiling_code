import scipy.fftpack as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Adapted from Matlab code written by (Schmidt,2010-Chp 2-pg 36)
"""
def ft2(g, delta):
   
    return sp.fftshift( sp.fft2( sp.fftshift(g) ) ) * delta**2

"""
Adapted from Matlab code written by (Schmidt,2010,Chp2,pg37)
"""
def ift2(G, delta_f):
    
    N = G.shape[0]
    
    return sp.ifftshift( sp.ifft2( sp.ifftshift(G) ) ) * (N * delta_f)**2


"""
Diagram of propagation geometry 
- potentially useful for inspecting to insure code is setting up simulation correctly
- potentially useful for displaying propagation distances and spacings
Potentially alot of points to plot in 3d plot. Sinces it's really only for display purposes --> downsample option
sample_rate = blah #only plots every blah point
"""

def set_up_geometry(Uin, delta, z, n, sample_rate = 1 ):
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    planes = []

    N = Uin.shape[0]    
    nx,ny = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2))

    for idx in range(0,n):

        xi = nx * delta[idx]
        yi = ny * delta[idx]

        planes.append([xi[::sample_rate,::sample_rate],yi[::sample_rate,::sample_rate]])

    flat = np.zeros_like(nx)  
    flat = flat[::sample_rate,::sample_rate]

    for i, plane in enumerate(planes):
        
        ax.plot_wireframe(flat + z[i],plane[0],plane[1], alpha = 0.75)

    ax.set_zlabel('Grid Spacing $(y) [m]$')
    ax.set_ylabel('Grid Spacing $(x) [m]$')
    ax.set_xlabel('Propagation Distance $(z) [m]$')    
        
    return


"""
Adapted from Matlab code written by (Schmidt,2010,Chp3,45)
"""

def corr2_ft(u1, u2, mask, delta):
    
    N = len(u1)
    
    delta_f = 1 / (N*delta)
    
    U1 = ft2(u1 * mask, delta)
    U2 = ft2(u2 * mask, delta)
    
    U12corr = ift2(np.conj(U1) * U2, delta_f)
    
    maskcorr = ift2(np.abs(ft2(mask, delta))**2, delta_f) * delta**2
    
    c = U12corr / maskcorr * mask
    
    return c


def ang_spec_multi_prop_func(Uin, wvl, delta1, deltan, z, sg, phz):
    
    k = 2*np.pi/wvl
    
    n = int(z.shape[0]) 
    delta_z = np.diff(z)
    alpha = z / z[-1] 

    delta = (1 - alpha) * delta1 + alpha * deltan

    m = delta[1:] / delta[0:-1] #scaling factor from i+1/i    
    
    N = Uin.shape[0]    
    nx,ny = np.meshgrid(np.arange(-N/2, N/2), np.arange(-N/2, N/2))
    
    x1 = nx * delta[0]
    y1 = ny * delta[0]

    r1sq = x1**2 + y1**2

    """
    Phase Factor 1
    """
    Q1 =  np.exp(1j * k/2 * (1-m[0])/delta_z[0] * r1sq)

    Uin = Uin * Q1 * phz[0]
    
    for idx in range(0,n-1):

        xi = nx * delta[idx]
        yi = ny * delta[idx]

        deltaf = 1 / (N * delta[idx])

        fX = nx * deltaf
        fY = ny * deltaf

        fsq = fX**2 + fY**2

        Z = delta_z[idx]
        mag = m[idx]

        """
        Phase Factor 2 (Quadratic Phase Factor)
        """

        Q2 = np.exp(-1j * np.pi**2 * 2 * Z/mag/k*fsq)

        Uin = ft2(Uin / mag, delta[idx])    
        Uin = ift2(Q2 * Uin, deltaf) 

        Uin = sg * phz[idx] * Uin #apply boundary absorber and turb screen phase 
        
    xn = nx * delta[-1]
    yn = ny * delta[-1]

    rnsq = xn**2 + yn**2

    """
    Phase Factor 3
    """
    Q3 = np.exp(1j * k/2. * (m[-1]-1)/(m[-1]*Z) * rnsq)

    Uout = Q3 * Uin
        
    return xn, yn, Uout
