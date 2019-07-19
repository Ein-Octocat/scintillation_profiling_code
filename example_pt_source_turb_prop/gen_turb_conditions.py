import numpy as np
from scipy.optimize import least_squares

def gen_turb_conditions_func(Cn2, k, Dz, nscr):    
    
    #SW and PW coherence diameters [m]
    r0sw = (0.423 * k**2 * Cn2 * 3.0/8 * Dz)**(-3.0/5)    
    r0pw = (0.423 * k**2 * Cn2 * Dz)**(-3.0/5)
    p = np.linspace(0, Dz, 1000)
    
    rytov = 0.563 * k**(7.0/6) * np.sum( Cn2 * (1 - p/Dz)**(5.0/6) * p**(5.0/6) * (p[1] - p[0]) )
    
    A = np.zeros((2,nscr))

    alpha = np.arange(0,nscr)/(nscr-1)    
    
    A[0] = alpha**(5.0/3)
    A[1] = (1 - alpha)**(5.0/6) * alpha**(5.0/6)

    B = np.asarray([r0sw**(-5.0/3), rytov/1.33 * (k/Dz)**(5.0/6)])  
    
    #initial guess
    x0 = (nscr/3*r0sw * np.ones(nscr))**(-5.0/3)    
    
    #objective function

    def fun(x, A, B):
        return B - A.dot(x)

    #minimise using scipy.optimize.least_squares - allows contraint that x>0 through setting bounds
    res = least_squares(fun, x0, args =(A,B), bounds = (0,np.inf))  
    
    x = res.x
    r0scrn = x**(-3.0/5)
    
    return r0scrn, A, B, r0sw, r0pw, rytov