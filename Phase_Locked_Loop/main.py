'''
Created on 26.02.2014

@author: Fabio Marti
'''
import numpy as np
import pylab as py
import message as msg
import utilities as util
from matplotlib import rc


def main():
    # Parameters
    nOfSteps = 288
    omega = 2*np.pi/144
    phi = [0,np.pi/4,np.pi/2,np.pi]
    harmonicFrequencies = [1,2,3]
    amplitudes = [5,2,3]
    variance = 2
    gamma = 1
    
    
    # Generation of the state space model matrices
    nOfFrequencies = len(harmonicFrequencies)
    matrixList = []
    C = np.zeros((1,2*nOfFrequencies))

    for i in range(0,nOfFrequencies):
        f = harmonicFrequencies[i]
        rotationalMatrix = np.array([[np.cos(f*omega),np.sin(f*omega)],[-np.sin(f*omega),np.cos(f*omega)]])
        matrixList.append(rotationalMatrix)
        C[0,2*i] = amplitudes[i]
    A = util.blockDiag(matrixList)
    
    
    # Computation of the inverse of A (Remains fixed)
    I = np.identity(A.shape[0])
    A_inv = np.linalg.solve(A, I)
    
    
    # Initial values
    x_k = np.zeros((2*nOfFrequencies,1))
    for i in range(0,nOfFrequencies):
        f = harmonicFrequencies[i]
        x_k[2*i,0] = np.cos(phi[i])
        x_k[2*i+1,0] = np.sin(phi[i])
        
    W_x = I
    Wm_x = np.transpose(np.tile(np.array([[1,0]]),nOfFrequencies))
    y = np.zeros(nOfSteps)
    y_tilde = np.zeros(nOfSteps)
    phase = np.zeros((nOfSteps,nOfFrequencies))
    
    
    # Solving the PLL problem iteratively via factor graphs
    steps = np.arange(1,nOfSteps+1)
    for step in steps:
        x_k = np.dot(A, x_k)
        y_k = np.dot(C, x_k)
        y[step-1] = y_k
        z_k = variance*np.random.rand()
        y_tildek = y_k+z_k
        y_tilde[step-1] = y_tildek
        
        
        # Apply the message passing algorithm
        [W_x, Wm_x] = msg.directForwardMessagePassing(A_inv, C, variance, y_tildek, W_x, Wm_x)
        mean_k = np.linalg.solve(W_x, Wm_x)
        
        # Compute phases of harmonic sinusoids
        for i in range(0,nOfFrequencies):
            alpha = 1/(np.sqrt(mean_k[2*i]**2+mean_k[2*i+1]**2))  # Scale mean value
            phase_k = np.arccos(alpha*mean_k[2*i])
            phase[step-1,i] = phase_k
        
        # Incorporate forgetting factor
        W_x = W_x/gamma
        Wm_x = Wm_x/gamma
        
    estimatedHarmonicSig = np.zeros(nOfSteps)
    for i in range(0,nOfFrequencies):
        estimatedHarmonicSig += amplitudes[i]*np.cos(phase[:,i])
    
    
    # Enable Latex functionality and fonts
    rc('text',usetex=True)
    rc('font',**{'family':'serif','serif':['Computer Modern']})
    
    
    # Plot estimation results
    py.figure(1)
    
    py.subplot(2,2,1)
    py.plot(steps,y)
    py.title('Reference input signal')
    py.xlabel('Timestep $k$')
    py.ylabel('Amplitude')
    
    py.subplot(2,2,2)
    py.plot(steps,y_tilde)
    py.title('Noisy signal with $\sigma^2$ = ' + str(variance))
    py.xlabel('Timestep $k$')
    py.ylabel('Amplitude')
    
    py.subplot(2,2,3)
    py.plot(steps,phase[:,1],'x')
    py.title('Estimated phase')
    py.xlabel('Timestep $k$')
    py.ylabel('Amplitude')
    
    py.subplot(2,2,4)
    py.plot(steps,estimatedHarmonicSig)
    py.title('Estimated signal')
    py.xlabel('Timestep $k$')
    py.ylabel('Amplitude')
    
    
    # Plot the harmonics and the resultiing signal
    py.figure(2)
    for i in range(0,nOfFrequencies):
        harmonic = amplitudes[i]*np.cos(steps*harmonicFrequencies[i]*omega+phi[i])
        py.plot(steps,harmonic,color=util.colors(i),linewidth = 1.5)
    
    py.plot(steps,y,color = 'k',linewidth = 2.5)
    py.title('Reference input signals and corresponding harmonics')
    py.xlabel('Timestep $k$')
    py.ylabel('Amplitude')
    py.show()
        
if __name__ == '__main__':
    
    main()