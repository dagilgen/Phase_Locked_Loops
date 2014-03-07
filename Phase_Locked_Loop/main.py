'''
Created on 26.02.2014

@authors: Daniel Gilgen, Fabio Marti
'''
import numpy as np
import pylab as py
import message as msg
import utilities as util
import time
from matplotlib import rc


def main():
    # Parameters
    nOfSamples = 36000
    T_s = 1.0/8000                      # Sampling Period [s]
    f_W = 2                             # Fundamental frequency [Hz]
    omega = 2*np.pi*f_W*T_s
    harmonicFrequencies = [1]           # Multiples of the fundamental frequency
    amplitudes = [4,4,4]                # Amplitudes of harmonics
    phi = [0,np.pi/3,np.pi/2]           # Phase shifts of harmonics
    variance = 4                        # Noise variance
    gamma = 0.999                       # Forgetting factor
    zeroThreshold = 1e-10               # Threshold below which numbers are treated as zero
    
    print("Status:\n")
    
    # Generation of the state space model matrices
    nOfFrequencies = len(harmonicFrequencies)
    matrixList = []
    C = np.zeros((1,2*nOfFrequencies))

    for i in range(0,nOfFrequencies):
        f = harmonicFrequencies[i]
        rotationalMatrix = np.array([[np.cos(f*omega),-np.sin(f*omega)],[np.sin(f*omega),np.cos(f*omega)]])
        matrixList.append(rotationalMatrix)
        C[0,2*i] = amplitudes[i]
    A = util.blockDiag(matrixList)
    
    
    # Since A is orthogonal, the transpose of A is equal to its inverse
    A_inv = np.transpose(A)
    
    
    # Initial values
    x_k = np.zeros((2*nOfFrequencies,1))
    for i in range(0,nOfFrequencies):
        f = harmonicFrequencies[i]
        x_k[2*i,0] = np.cos(phi[i])
        x_k[2*i+1,0] = np.sin(phi[i])
    I = np.identity(A.shape[0])    
    W_x = I
    Wm_x = np.transpose(np.tile(np.array([[1,0]]),nOfFrequencies))
    y = np.zeros(nOfSamples)
    y_tilde = np.zeros(nOfSamples)
    phase = np.zeros((nOfSamples,nOfFrequencies))
    
    
    # Solving the PLL problem iteratively via factor graphs
    samplingTime = T_s*np.arange(1,nOfSamples+1)
    startTime = time.time()
    for k in range(1,nOfSamples+1):
        x_k = np.dot(A, x_k)
        y_k = np.dot(C, x_k)
        y[k-1] = y_k
        z_k = variance*np.random.randn()    # Add white Gaussian noise
        y_tildek = y_k+z_k
        y_tilde[k-1] = y_tildek
        
        
        # Apply the message passing algorithm with incorporated forgetting factor
        [W_x, Wm_x] = msg.directForwardMessagePassing(A_inv, C, variance, y_tildek, W_x, Wm_x)
        mean_k = np.linalg.solve(W_x, Wm_x)
        
        # Compute phases of harmonic sinusoids
        for i in range(0,nOfFrequencies):
            alpha = 1/(np.sqrt(mean_k[2*i]**2+mean_k[2*i+1]**2))  # Scale mean value
            cos_phase = np.arccos(alpha*mean_k[2*i])
            sin_phase = np.arcsin(alpha*mean_k[2*i+1])
            toggle = np.sign(sin_phase)
            phase[k-1,i] = cos_phase*toggle + (1-np.floor(np.abs(toggle)))*np.pi
        
        #phase[k-1,0:nOfFrequencies] = util.phaseEstimator2(phase[k-1,0:nOfFrequencies],omega*np.array(harmonicFrequencies),T_s,k)
        W_x = W_x*gamma
        Wm_x = Wm_x*gamma   
        
    estimatedHarmonicSig = np.zeros(nOfSamples)
    for i in range(0,nOfFrequencies):
        estimatedHarmonicSig += amplitudes[i]*np.cos(phase[:,i])
    stopTime = time.time()
    executionTime = (stopTime-startTime)
    print("Completed! The computation took %f seconds.\n" % executionTime)
    
    # Enable LaTex functionality and fonts
    rc('text',usetex=True)
    rc('font',**{'family':'serif','serif':['Computer Modern']})
    
    
    # Plot estimation results
    py.figure(1)
    
    py.subplot(2,2,1)
    py.plot(samplingTime,y)
    py.title('Reference input signal')
#    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    py.subplot(2,2,2)
    py.plot(samplingTime,y_tilde)
    py.title('Noisy signal with $\sigma^2$ = ' + str(variance))
#    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    py.subplot(2,2,3)
    py.plot(samplingTime,phase[:,0],'x')
    py.title('Estimated phase of fundamental harmonic')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    py.subplot(2,2,4)
    py.plot(samplingTime,estimatedHarmonicSig)
    py.title('Estimated signal')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    
    # Plot the harmonics and the resulting signal
    py.figure(2)
    
    py.subplot(3,1,1)
    for i in range(0,nOfFrequencies):
        harmonic = amplitudes[i]*np.cos(2*np.pi*samplingTime*harmonicFrequencies[i]*f_W+phi[i])
        py.plot(samplingTime,harmonic,color=util.colors(i),linewidth = 1.5)
    py.plot(samplingTime,y,color = 'k',linewidth = 2.5)
    py.title('Reference input signal and corresponding harmonics')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    
    # Error analysis  
    rawPhase = (phi[0] + (omega*np.arange(1,nOfSamples+1)))%(2*np.pi)       # Calculate the phase error
    toggle = np.floor((np.sign(np.sin(rawPhase)+zeroThreshold))*0.5)        # Eliminate rounding errors
    exactPhase = rawPhase + toggle*2*np.pi                                  # Shifts the phase into the interval
                                                                            # [-pi,pi] instead of [0,2pi]
    absPhaseDifferenceBottom = np.abs((phase[:,0]-exactPhase))              # Absolute phase difference towards 0
    absPhaseDifferenceTop = 2*np.pi-np.abs((phase[:,0]-exactPhase))         # Absolute phase difference towards 2*pi
    absPhaseDifference = np.minimum(absPhaseDifferenceBottom,absPhaseDifferenceTop)
    
    squaredPhaseError = ((np.abs(absPhaseDifference)%(2*np.pi)))**2
    error = estimatedHarmonicSig-y
    maxError = np.max(np.abs(error))
    RMSError = np.sqrt(np.sum(error**2)/nOfSamples)
    print("Error analysis")
    print("Maximum absolute error:\t %f" % maxError)
    print("Root-mean-square error:\t %f" % RMSError)
    
    py.subplot(3,1,2)
    py.axhline(color = 'k',linewidth = 1)
    py.plot(samplingTime,error,color = 'r',linewidth = 1.5)
    py.title('Absolute estimation error')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Error $E$')
    
    
    py.subplot(3,1,3)
    py.axhline(color = 'k',linewidth = 1)
    py.plot(samplingTime,squaredPhaseError,color = 'r',linewidth = 1.5)
    py.title('Squared fundamental phase error $|\hat{\phi}-\phi|^2$')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Error $E$')
    
    py.show()
        
if __name__ == '__main__':
    
    main()