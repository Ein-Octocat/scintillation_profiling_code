"""
Adapted from Matlab code written by (Schmidt,2010-Chp 2-pg 36)
"""
def ft2(g, delta):
    
    import scipy.fftpack as sp
   
    return sp.fftshift( sp.fft2( sp.fftshift(g) ) ) * delta**2

"""
Adapted from Matlab code written by (Schmidt,2010,Chp2,pg37)
"""
def ift2(G, delta_f):
    
    import scipy.fftpack as sp
    
    N = G.shape[0]
    
    return sp.ifftshift( sp.ifft2( sp.ifftshift(G) ) ) * (N * delta_f)**2


def ang_spec_multi_prop_func(Uin, wvl, delta1, deltan, z, sg, phz):
    
    import numpy as np
    
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

        Uin = sg * phz[idx] * Uin #apply boundary absorber  
        
    xn = nx * delta[-1]
    yn = ny * delta[-1]

    rnsq = xn**2 + yn**2

    """
    Phase Factor 3
    """
    Q3 = np.exp(1j * k/2. * (m[-1]-1)/(m[-1]*Z) * rnsq)

    Uout = Q3 * Uin
        
    return xn, yn, Uout
