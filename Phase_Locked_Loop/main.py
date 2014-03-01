'''
Created on 26.02.2014

@author: Fabio Marti
'''
import numpy as np
import pylab as py
import message as msg
#import utilities as util
from matplotlib import rc


def main():
    # Parameters
    nOfSteps = 1000
    steps = np.arange(nOfSteps)
    phi = 90
    omega = 2*np.pi/144
    amplitude = 5
    variance = 2
    gamma = 1
    
    A = np.array([[np.cos(omega),np.sin(omega)],[-np.sin(omega),np.cos(omega)]])
    C = amplitude*np.array([[1,0]])
    
    # Computation of the inverse of A (Remains fixed)
    I = np.identity(A.shape[1])
    A_inv = np.linalg.solve(A, I)
    
    # Initial values
    x_k = np.array([[np.cos(phi)],[np.sin(phi)]])
    W_x = np.identity(A.shape[0])
    Wm_x = np.array([[1],[0]])
    y = np.zeros(nOfSteps)
    y_tilde = np.zeros(nOfSteps)
    phase = np.zeros(nOfSteps)
    
    # Solving the PLL problem iteratively via factor graphs
    for step in steps:
        x_k = np.dot(A, x_k)
        y_k = np.dot(C, x_k)
        y[step] = y_k
        z_k = variance*np.random.rand()
        y_tildek = y_k+z_k
        y_tilde[step] = y_tildek
        
        # Apply the message passing algorithm
        [W_x, Wm_x] = msg.directForwardMessagePassing(A_inv, C, variance, y_tildek, W_x, Wm_x)
        mean_k = np.linalg.solve(W_x, Wm_x)
        alpha = 1/(np.sqrt(mean_k[0]**2+mean_k[1]**2))  # Eliminate rounding errors
        phase_k = np.arccos(alpha*mean_k[0])
        phase[step] = phase_k
        
        # Incorporate forgetting factor
        W_x = W_x/gamma
        Wm_x = Wm_x/gamma
        
    # Plot graphs
    py.subplot(4,1,1)
    py.plot(steps,y)
    py.subplot(4,1,2)
    py.plot(steps,y_tilde)
    py.subplot(4,1,3)
    py.plot(steps,phase,'x')
    py.subplot(4,1,4)
    py.plot(steps,amplitude*np.cos(phase))
    py.show()
        
        
if __name__ == '__main__':
    
    main()