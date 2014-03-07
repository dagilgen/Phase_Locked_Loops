'''
This script serves for testing and experimental purposes only

Created on 28.02.2014

@authors: Fabio Marti, Daniel Gilgen
'''
import numpy as np
import pylab as py
import messages as msg


def main():
    # Parameters for reference signal
    n = 500    
    phi = 90
    omega = 2*np.pi/120
    amplitude = 3
    variance = 2
    gamma = 1
    
    # Rotation matrix A
    A = np.array([[np.cos(omega),np.sin(omega)],[-np.sin(omega),np.cos(omega)]])
    C = amplitude*np.array([[1,0]])
    
    # Computation of the inverse of A (Remains fixed)
    I = np.identity(A.shape[1])
    A_inv = np.linalg.solve(A, I)
    
    # Initial values
    x_k = np.array([[np.cos(phi)],[np.sin(phi)]])
    W_x = np.identity(A.shape[0])
    Wm_x = np.array([[1],[0]])
    y = np.zeros(n)
    ytilde = np.zeros(n)
    phase = np.zeros(n)
    
    # Solving the PLL problem by message passing
    steps = np.arange(n)
    for n in steps:
        x_k = np.dot(A, x_k)
        y_k = np.dot(C, x_k)
        y[n] = y_k
        z_k = variance*np.random.rand()
        ytilde_k = y_k+z_k
        ytilde[n] = ytilde_k
        
        # Message passing algorithm
        [W_x, Wm_x] = msg.directForwardMessagePassing(A_inv, C, variance, ytilde_k, W_x, Wm_x)
        mean_k = np.linalg.solve(W_x, Wm_x)
        alpha = 1/(np.sqrt(mean_k[0]**2+mean_k[1]**2))  # Eliminate rounding errors
        phase_k = np.arccos(alpha*mean_k[0])
        phase[n] = phase_k
        
        # Forgetting factor
        W_x = W_x/gamma
        Wm_x = Wm_x/gamma
        
    # Plot graphs
    py.subplot(2,2,1)
    py.plot(steps,y)
    py.title('Clean sine wave')
#    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    py.subplot(2,2,2)
    py.plot(steps,ytilde)
    py.title('Noisy input to PLL')
#    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    py.subplot(2,2,3)
    py.plot(steps,phase,'x')
    py.title('PLL output')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Phase $\phi$')
    
    py.subplot(2,2,4)
    py.plot(steps,amplitude*np.cos(phase))
    py.title('Best-fitting signal')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    py.show()
        
        
if __name__ == '__main__':
    
    main()