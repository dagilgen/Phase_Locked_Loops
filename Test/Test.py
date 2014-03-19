'''
Created on 17.03.2014

@author: dagilgen
'''
import numpy as np
import pylab as py
import time
import utilities as util
from matplotlib import rc


def main():
    
    A = np.zeros((2, 6))
    print(A)
#         # Parameters
#     useCompleteModel = False            # Choose the message passing model
#     abruptSigChange = False             # Abrupt signal change, impemented for only one harmonic
#     multHarmonics = True                # Multiple harmonics
#     nOfSamples = 2000
#     T_s = 1.0/1000                      # Sampling Period [s]
#     f_W = 2                             # Fundamental frequency [Hz]
#     omega = 2*np.pi*f_W*T_s
#     harmonicFrequencies = [1,2,3]       # Multiples of the fundamental frequency
#     amplitudes = [4,4,4]                # Amplitudes of harmonics
#     phi = [0,np.pi/3,np.pi/2]           # Phase shifts of harmonics
#     variance = 2                        # Noise variance
#     gamma = 0.9995                      # Forgetting factor
#     zeroThreshold = 1e-10               # Threshold below which numbers are treated as zero
#     harmonicLocked = 0
#     
#     nOfFrequencies = len(harmonicFrequencies)
#     
#     
#     matrixList_D = []
#     matrixList_E = []  
# 
#     for i in range(0,nOfFrequencies):
#         f = harmonicFrequencies[i]
#         rotationalMatrix = np.array([[np.cos(f*omega),-np.sin(f*omega)],[np.sin(f*omega),np.cos(f*omega)]])
#         rotationalMatrix_inv = np.transpose(rotationalMatrix)
#         rotationalMatrix_inv_dotk = rotationalMatrix_inv
#         #rotationalMatrix = np.array([[np.cos(f*omega),-np.sin(f*omega)],[np.sin(f*omega),np.cos(f*omega)]]) # Computation of D matrix
#         rotationalMatrix_inv = np.transpose(rotationalMatrix)
#         rotationalMatrix_inv_dotk = np.dot(rotationalMatrix_inv_dotk, rotationalMatrix_inv)
#         matrixList_D.append(rotationalMatrix_inv_dotk)
#         #phi_delta = harmonicFrequencies[harmonicLocked]*omega-f*omega # Computation of E matrix
#         phi_delta = phi[harmonicLocked] - phi[i] # Computation of E matrix
#         E_i = amplitudes[harmonicLocked]/amplitudes[i]*np.array([[np.cos(phi_delta),-np.sin(phi_delta)],[np.sin(phi_delta),np.cos(phi_delta)]])
#         matrixList_E.append(E_i)          
#                  
#     D = util.blockDiag(matrixList_D)
#     E = util.blockDiag(matrixList_E)
#     print(D)
#     print(E)
#     print(D[0:2,0:2])
#     print(D[2:4,2:4])
#     print(D[4:6,4:6])
        
if __name__ == '__main__':
        
    main()