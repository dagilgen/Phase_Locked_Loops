'''
Created on 28.02.2014

@authors: Daniel Gilgen, Fabio Marti
'''

import numpy as np


"""
Error Class. Used to raise errors.
"""
class Error(Exception):
    pass


def colors(colorIndex):
    """
    Returns the respective RGB color for a certain input number.
    
    Inputs:
    
    colorIndex : int
        Integer indexing the desired color.
        
        
    Outputs:
    
    color : list
        The selected RGB color.
    """
    
    if colorIndex == 0:             # red
        color = [1,0.2,0.2]
    elif colorIndex == 1:           # orange
        color = [1,0.8,0]
    elif colorIndex == 2:           # brown
        color = [0.4,0.4,0]
    elif colorIndex == 3:           # violet
        color = [0.2,0.1,0.4]
    elif colorIndex == 4:           # blue
        color = [0.2,0.2,1]
    elif colorIndex == 5:           # light blue
        color = [0.6,0.6,1]
    elif colorIndex == 6:           # light green
        color = [0.3,1,0.3]
    elif colorIndex == 7:           # green
        color = [0,0.7,0]
    elif colorIndex == 8:           # dark green
        color = [0,0.4,0]
    elif colorIndex == 9:           # pink
        color = [1,0,1]
    else:                           # grey
        color = [0.6,0.6,0.6]
    
    
    return color

def diagonalSum(M):
    """
    Returns the sum of all diagonal elements of a matrix.
    
    Inputs:
    
    M : int
        Matrix with diagonal elements of interest.
        
        
    Outputs:
    
    diogonalSum : int
        The resulting diagonal sum.
    """
    diogonalSum = []
    print(M)
    for i in xrange(0, M.shape[0]):
        diogonalSum += M[i][i]
        print(i)
    
    #print(diogonalSum)
    return diogonalSum

def blockDiag(matrixList):
    """
    Takes square matrices as input and constructs a block diagonal
    matrix in order of their appearance as input arguments.
    
    Inputs:
    
    matrixList : list
        A list of arrays serving as diagonal matrices of the
        resulting block diagonal matrix.
        
        
    Outputs:
    
    blockMatrix : array_like
        Resulting block diagonal matrix.
    """
    
    # Check if all input matrices are square matrices
    dimension = 0
    for block in matrixList:
        if block.shape[0] != block.shape[1]:
            raise Error("Non-square input matrix.")
        dimension += block.shape[0]
    
    # Construct diagonal block matrix
    index = 0
    blockMatrix = np.zeros((dimension, dimension))
    for block in matrixList:
        matSize = block.shape[0]
        blockMatrix[index:index+matSize,index:index+matSize] = block
        index += matSize
        
    return blockMatrix


def phaseEstimator(phases,omegas,T_s,k):
    """
    !!DRAFT!! Estimates the phase shifts !!DRAFT!!
    """
    length = phases.shape[0]
    pis = np.tile(2*np.pi,length)
    a = phases - T_s*k*omegas
    phaseShifts = np.mod(a,pis)
    b = phases-phaseShifts
    omega_hat = np.mod(b,pis)
    n = omega_hat/omegas
    estimatedTime = np.sum(n)/length
    
    estimatedPhase = phaseShifts + estimatedTime*omegas
    
    return estimatedPhase



def phaseEstimator2(phases,omegas,T_s,k):
    """
    !!DRAFT!! Estimates the phase shift if all shifts of the harmonics are the same !!DRAFT!!
    """
    
    
    length = phases.shape[0]
    pis = np.tile(2*np.pi,length)
    a = phases - k*omegas
    phaseShifts = np.mod(a,pis)

    averagedPhaseShift = np.sum(phaseShifts)/length
    
    estimatedPhase = np.mod(averagedPhaseShift + k*omegas,pis)
    #estimatedPhase = np.array([np.pi/2,np.pi/2,np.pi/2]) + k*omegas
    
    return estimatedPhase