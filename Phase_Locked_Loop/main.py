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
    useCompleteModel = False            # Choose the message passing model
    multHarmonics = True                # Multiple harmonics
    inputFundamental = True
    harmonicLocked = 0                  # Choose which harmonic do you want to track
    phaseJump = 0                       # If phaseJump is nonzero, the phase of all harmonics changes abruptly
    
    nOfSamples = 2000
    T_s = 1.0/1000                      # Sampling Period [s]
    f_W = 2                             # Fundamental frequency [Hz]
    omega = 2*np.pi*f_W*T_s
    harmonicFrequencies = [1,2,3,4]       # Multiples of the fundamental frequency
    amplitudes = [4,3,4,2]                # Amplitudes of harmonics
    phi = [0,np.pi/3,np.pi/2,np.pi/5]           # Phase shifts of harmonics
    variance = 2                        # Noise variance
    gamma = 0.9995                      # Forgetting factor
    zeroThreshold = 1e-10               # Threshold below which numbers are treated as zero
    
    
    print("Status:\n")
    
    # Generation of the state space model matrices
    nOfFrequencies = len(harmonicFrequencies)
    systemMatrixList = []
    precisionMatrixList = []
    c = np.array([[1,0]])
    C = np.tile(c,(1,nOfFrequencies))
    C_phaseLocked = np.array([np.zeros(2*nOfFrequencies, int)])
    C_phaseLocked[0][harmonicLocked*2] = 1

    for i in range(0,nOfFrequencies):
        f = harmonicFrequencies[i]
        rotationalMatrix = util.rotMat(f*omega)
        W_ss = msg.steadyStatePrecisionMatrix(gamma, variance, f*omega)
        systemMatrixList.append(rotationalMatrix)
        precisionMatrixList.append(W_ss)
    A = util.blockDiag(systemMatrixList)
    
    
    # Since A is orthogonal, the transpose of A is equal to its inverse
    A_inv = np.transpose(A)
    
    
    # Initial values
    x = np.zeros((2*nOfFrequencies,nOfSamples))
    x_k = np.zeros((2*nOfFrequencies,1))
    for i in range(0,nOfFrequencies):
        x_k[2*i,0] = amplitudes[i]*np.cos(phi[i])
        x_k[2*i+1,0] = amplitudes[i]*np.sin(phi[i])      

    W_x = util.blockDiag(precisionMatrixList)
    Wm_x = np.transpose(np.tile(np.array([[1,0]]),nOfFrequencies))
    y = np.zeros(nOfSamples)
    y_prime = np.zeros(nOfSamples)
    y_tilde = np.zeros(nOfSamples)
    phase = np.zeros((nOfSamples,nOfFrequencies))
    phaseLocked = np.zeros((nOfSamples,1))
    estimatedHarmonicSig = np.zeros(nOfSamples)
    
    # Solving the PLL problem iteratively via factor graphs
    samplingTime = T_s*np.arange(1,nOfSamples+1)
    startTime = time.time()
    identity = np.identity(2)
    rotationalMatrix_inv_dotk = np.tile(identity, (nOfFrequencies,1,1))

    for k in range(1,nOfSamples+1):
        x_k = np.dot(A, x_k)
        if phaseJump != 0 and k%(nOfSamples/2)== 0 and k<(nOfSamples/2)+1:
            abruptRotation = util.rotMat(phaseJump)
            jumpMatrix = util.blockDiag(nOfFrequencies*[abruptRotation])
            x_k = np.dot(jumpMatrix, x_k)
        x[:,k-1:k] = x_k
        if multHarmonics:        
            y_prime[k-1] = np.dot(C_phaseLocked, x_k)                        
        y_k = np.dot(C, x_k)
        y[k-1] = y_k
        z_k = variance*np.random.randn()    # Add white Gaussian noise
        y_tildek = y_k+z_k
        y_tilde[k-1] = y_tildek
         
        
        # Apply the message passing algorithm with incorporated forgetting factor
        if useCompleteModel:
            #[W_x, Wm_x] = msg.forwardMessagePassingComplete(A_inv, C, variance, y_tildek, W_x, Wm_x)
            Wm_x = msg.computeWeightedMeanComplete(A_inv, C, variance, y_tildek, Wm_x)
        else:
            #[W_x, Wm_x] = msg.forwardMessagePassingSplit(A_inv, c, variance, y_tildek, W_x, Wm_x)
            Wm_x = msg.computeWeightedMeanSplit(A_inv, c, variance, y_tildek, Wm_x)
        mean_k = np.linalg.solve(W_x, Wm_x)
        
        
        # Compute phases of harmonic sinusoids
        for i in range(0,nOfFrequencies):
            alpha = 1/(np.sqrt(mean_k[2*i]**2+mean_k[2*i+1]**2))  # Scale mean value
            cos_phase = np.arccos(alpha*mean_k[2*i])
            sin_phase = np.arcsin(alpha*mean_k[2*i+1])
            toggle = np.sign(sin_phase)
            phase[k-1,i] = cos_phase*toggle + (1-np.floor(np.abs(toggle)))*np.pi                 


        # Apply glue factor
        if multHarmonics:
            matrixList_D = []
            matrixList_E = []
            for i in range(0,nOfFrequencies):                          
                f = harmonicFrequencies[i]
                rotationalMatrix_inv = np.transpose(util.rotMat(f*omega))
                rotationalMatrix_inv_dotk[i] = np.dot(rotationalMatrix_inv_dotk[i], rotationalMatrix_inv)
                matrixList_D.append(rotationalMatrix_inv_dotk[i])
                
                phi_delta = phi[harmonicLocked] - phi[i]
                E_i = amplitudes[harmonicLocked]/amplitudes[i]*util.rotMat(phi_delta)
                matrixList_E.append(E_i)          
                
            D = util.blockDiag(matrixList_D)
            E = util.blockDiag(matrixList_E)
            
            [W_tildex, Wm_tildex] = msg.applyGlueFactor(W_x, Wm_x, D, E)
            mean_tildek = np.linalg.solve(W_tildex, Wm_tildex)
            
            
            # Turn the phases back to the state at time k
            mean_tildek = np.dot(np.transpose(matrixList_D[harmonicLocked]),mean_tildek)
            
            
            # Compute phase of locked phase
            alpha = 1/(np.sqrt(mean_tildek[0]**2+mean_tildek[1]**2))  # Scale mean value
            cos_phase = np.arccos(alpha*mean_tildek[0])
            sin_phase = np.arcsin(alpha*mean_tildek[1])
            toggle = np.sign(sin_phase)
            phaseLocked[k-1] = cos_phase*toggle + (1-np.floor(np.abs(toggle)))*np.pi
        #W_tildex = W_tildex*gamma
        Wm_x = Wm_x*gamma
        
        
    
    if multHarmonics:
        estimatedHarmonicSig = amplitudes[harmonicLocked]*np.cos(phaseLocked[:,0])
        stopTime = time.time()
        executionTime = (stopTime-startTime)
    else:
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
    py.subplots_adjust(left=None, bottom=None, right=None, top=None,    # Adjustment of subplots
                wspace=0.3, hspace=0.3)
#    py.tight_layout
    
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
    if multHarmonics:
        py.plot(samplingTime,phaseLocked[:,0],'x')
    else:
        py.plot(samplingTime,phase[:,harmonicLocked],'x')
        
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
    py.subplots_adjust(left=None, bottom=None, right=None, top=None,    # Adjustment of subplots
                wspace=0.7, hspace=0.7)
    
    py.subplot(3,1,1)
    py.axhline(color = 'k',linewidth = 1)
    for i in range(0,nOfFrequencies):
        harmonic = np.transpose(np.dot(c,x[2*i:2*i+2,:]))
        py.plot(samplingTime,harmonic,color=util.colors(i),linewidth = 1.5)
    py.plot(samplingTime,y,color = 'k',linewidth = 2.5)
    py.title('Reference input signal and corresponding harmonics')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Amplitude $A$')
    
    
    # Error analysis  
    rawPhase = (phi[harmonicLocked] + (harmonicFrequencies[harmonicLocked]*omega*np.arange(1,nOfSamples+1)))%(2*np.pi)       # Calculate the phase error
    toggle = np.floor((np.sign(np.sin(rawPhase)+zeroThreshold))*0.5)        # Eliminate rounding errors
    exactPhase = rawPhase + toggle*2*np.pi                                  # Shifts the phase into the interval
                                                                            # [-pi,pi] instead of [0,2pi]
    if multHarmonics:
        absPhaseDifferenceBottom = np.abs((phaseLocked[:,0]-exactPhase))              # Absolute phase difference towards 0
        absPhaseDifferenceTop = 2*np.pi-np.abs((phaseLocked[:,0]-exactPhase))         # Absolute phase difference towards 2*pi
    else:
        absPhaseDifferenceBottom = np.abs((phase[:,harmonicLocked]-exactPhase))              # Absolute phase difference towards 0
        absPhaseDifferenceTop = 2*np.pi-np.abs((phase[:,harmonicLocked]-exactPhase))         # Absolute phase difference towards 2*pi
    absPhaseDifference = np.minimum(absPhaseDifferenceBottom,absPhaseDifferenceTop)
        
    squaredPhaseError = ((np.abs(absPhaseDifference)%(2*np.pi)))**2
    
    py.figure(2)
    if multHarmonics:
        error = estimatedHarmonicSig - y_prime
    else:
        error = estimatedHarmonicSig - y
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
    
    # Additional plot
    py.figure(3)
    py.subplots_adjust(left=None, bottom=None, right=None, top=None,    # Adjustment of subplots
                wspace=0.7, hspace=0.7)
    
    py.subplot(2,1,1)
    py.plot(samplingTime,exactPhase,color = 'r',linewidth = 1.5)
    py.plot(samplingTime,phase[:,harmonicLocked],color = 'b',linewidth = 1.5)
    py.plot(samplingTime,phaseLocked[:,0],color = 'g',linewidth = 1.5)
    py.title('Phase')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Phase')
    
    py.subplot(2,1,2)
    py.plot(samplingTime,estimatedHarmonicSig,color = 'r',linewidth = 1.5)
    py.plot(samplingTime,y,color = 'b',linewidth = 1.5)
    py.plot(samplingTime,y_prime,color = 'g',linewidth = 1.5)
    py.title('Squared fundamental phase error $|\hat{\phi}-\phi|^2$')
    py.xlabel('Time $t$ $[s]$')
    py.ylabel('Error $E$')
    
    py.show()
        
if __name__ == '__main__':
        
    main()