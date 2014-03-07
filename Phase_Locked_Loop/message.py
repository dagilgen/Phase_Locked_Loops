'''
Created on 26.02.2014

@author: Daniel Gilgen, Fabio Marti
'''

import numpy as np

def equalityConstraint(W_x, W_y, Wm_x, Wm_y):
    '''
    Computes the matrix equality constraint
    
    Inputs:    W_x, W_y, Wm_x, Wm_y
    Outputs:   W_z, Wm_z
    '''
    
    W_z = W_x + W_y
    Wm_z = Wm_x + Wm_y
    
    return [W_z, Wm_z]
    
    
def forwardSumConstraint(V_x, V_y, m_x, m_y):
    '''
    Computes the forward matrix sum constraint
    
    Inputs:    V_x, V_y, m_x, m_y
    Outputs:   V_z, m_z
    '''
    
    V_z = V_x + V_y
    m_z = m_x + m_y
    
    return [V_z, m_z]
    
    
def backwardSumConstraint(V_y, V_z, m_y, m_z):
    '''
    Computes the backward matrix sum constraint
    
    Inputs:    V_y, V_z, m_y, m_z
    Outputs:   V_x, m_x
    '''
    
    V_x = V_z + V_y
    m_x = m_z - m_y
    
    return [V_x, m_x]
    
    
def forwardMatrixMultConstraint(V_x, m_x, A):
    '''
    Computes the forward matrix multiplication constraint
    
    Inputs:    V_x, m_x, A
    Outputs:   V_y, m_y
    '''
    
    temp = np.dot(A, V_x)
    V_y = np.dot(temp, np.transpose(A))
    m_y = np.dot(A,m_x)
    
    return [V_y, m_y]
    
    
def backwardMatrixMultConstraint(W_y, Wm_y, A):
    '''
    Computes the backward matrix multiplication constraint
    
    Inputs:    W_y, Wm_y, A
    Outputs:   W_x, Wm_x
    '''
    
    temp = np.dot(np.transpose(A), W_y)
    W_x = np.dot(temp, A)
    Wm_x = np.dot(A, Wm_y)
    
    return [W_x, Wm_x]
    
    
def forwardMessagePassing(A, C, variance, y_tilde, W_x, Wm_x):
    '''
    Performs forward message passing for one iteration
    
    Inputs:    A, C, variance, y_tilde, W_x, Wm_x
    Outputs:   W_xnew, Wm_xnew
    '''
    
    [V_y, m_y] = backwardSumConstraint(0, variance, y_tilde, 0)
    W_y = 1/V_y     # Only works if V_y is scalar
    [Wprime_x, Wmprime_x] = backwardMatrixMultConstraint(W_y, np.dot(W_y,m_y), C)
    [Wtilde_x, Wmtilde_x] = equalityConstraint(W_x, Wprime_x, Wm_x, Wmprime_x)
    Vtilde_x = np.linalg.inv(Wtilde_x)
    mtilde_x = np.dot(Vtilde_x,Wmtilde_x)
    [W_xnew, Wm_xnew] = forwardMatrixMultConstraint(Vtilde_x, mtilde_x, A)
    
    return [W_xnew, Wm_xnew]
    
    
def directForwardMessagePassing(A_inv, C, variance, y_tilde, W_x, Wm_x):
    '''
    Performs forward message passing for one iteration by using the
    explicit formula for one step of the PLL factor graph
    
    Inputs:    
    
    A_inv : array-like
        Inverse of the factor graph's state space matrix A
    C : array-like
        Output matrix for state-output mapping
    variance : float
        Variance of the white Gaussian noise applied on the output
    y_tilde : float
        Currently observed output value
    W_x : array-like
        Current forward message of inverse of the covariance matrix
    Wm_x : array-like
        Current forward message of the combined covaraince-mean message
    
    
    Outputs:
    
    W_xnew : array-like
        New forward message of inverse of the covariance matrix
    Wm_xnew : array-like
        New forward message of the combined covaraince-mean message
    '''
    
    temp = np.dot(np.transpose(A_inv), W_x)
    W_xnew = np.dot(temp, A_inv) + np.dot(np.transpose(C), C)/variance
    Wm_xnew = np.dot(np.transpose(A_inv), Wm_x) + np.dot(np.transpose(C), y_tilde)/variance
    
    return [W_xnew, Wm_xnew]
    