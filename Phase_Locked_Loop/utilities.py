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