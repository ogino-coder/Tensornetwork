import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

##################################################################################################
# cost M * D ** 3
# Calculate the largest absolute eigenvalue and the right eigenvector of the transfer matrix T(A)
# A does not need to be normalized
# If you cannot understand RightEigs, please see Simple_RightEigs
##################################################################################################
def RightEigs(A,dtype=np.dtype("float")):
    # A.shape = (M,D,M)
    T_ope = RightEigsOpe(A,dtype)
    valR,vecR = sp.sparse.linalg.eigs(T_ope,k=1)
    if ( dtype == np.dtype("float") ):
        vecR = vecR.real; valR = valR.real
    vecR /= vecR[0,0]/np.abs(vecR[0,0]) # fix a complex phase of the eigenvector
    return valR[0],vecR[:,0]

class RightEigsOpe(sp.sparse.linalg.LinearOperator):
    def __init__(self,A,dtype):
        self.A = A
        self.D = A.shape[1]
        self.M = A.shape[0]
        self.shape = [self.M * self.M, self.M * self.M]
        self.dtype = dtype
    def _matvec(self,vvec):
        R1 = np.tensordot(self.A,vvec.reshape(self.M,self.M),([2],[0]))
        R = np.tensordot(R1,np.conj(self.A),([1,2],[1,2]))
        return R
        
def Simple_RightEigs(A,dtype):
    M = A.shape[0];
    T = np.einsum("asc,bsd -> abcd",A,np.conj(A)).reshape(M*M,M*M)
    valR,vecR = sp.sparse.linalg.eigs(T,k=1)
    if ( dtype == np.dtype("float") ):
        vecR = vecR.real; valR = valR.real
    vecR /= vecR[0,0]/np.abs(vecR[0,0]) # fix a complex phase of the eigenvector
    return valR[0],vecR[:,0]

##################################################################################################
# cost M * D ** 3
# Calculate the largest absolute eigenvalue and the right eigenvector of the extended transfer matrix T(A,O)
# A does not need to be normalized
# If you cannot understand ExtendedRightEigs, please see Simple_ExtendedRightEigs
##################################################################################################
def ExtendedRightEigs(A,O,dtype=np.dtype("float")):
    # A.shape = (M,D,M)
    T_ope = ExtendedRightEigsOpe(A,O,dtype)
    valR,vecR = sp.sparse.linalg.eigs(T_ope,k=1)
    if ( dtype == np.dtype("float") ):
        vecR = vecR.real; valR = valR.real
    vecR /= vecR[0,0]/np.abs(vecR[0,0]) # fix a complex phase of the eigenvector
    return valR[0],vecR[:,0]
    
class ExtendedRightEigsOpe(sp.sparse.linalg.LinearOperator):
    def __init__(self,A,O,dtype):
        self.A = A
        self.D = A.shape[1]
        self.M = A.shape[0]
        self.O = O
        self.shape = [self.M * self.M, self.M * self.M]
        self.dtype = dtype
    def _matvec(self,vvec):
        R1 = np.tensordot(self.A,vvec.reshape(self.M,self.M),([2],[0]))
        R2 = np.tensordot(R1,self.O,([1],[0]))
        R = np.tensordot(R2,np.conj(self.A),([2,1],[1,2]))
        return R

def Simple_ExtendedRightEigs(A,O,dtype):
    M = A.shape[0]
    T = np.einsum("asc,st,btd -> abcd",A,O,np.conj(A)).reshape(M*M,M*M)
    valR,vecR = sp.sparse.linalg.eigs(T,k=1)
    if ( dtype == np.dtype("float") ):
        vecR = vecR.real; valR = valR.real
    vecR /= vecR[0,0]/np.abs(vecR[0,0]) # fix a complex phase of the eigenvector
    return valR[0],vecR[:,0]


def Spin(D):
    S = 0.5 * ( D - 1.0 )
    Su = np.zeros((D,D))
    for i in range(D-1):
        m = i - S
        Su[i][i+1] = np.sqrt((S-m)*(S+m+1))
    Sd = Su.T
    Sx = 0.5 * ( Su + Sd )
    Sy = - 0.5j * ( Su - Sd )
    Sz = 0.5 * np.tensordot(Su,Sd,([1],[0])) - 0.5 * np.tensordot(Sd,Su,([1],[0]))
    return Sx,Sy,Sz,Su,Sd
