import numpy as np
import scipy as sp
import MathFunctions as MF

# transform a tensor A into the mixed canonical form
def MixedCanonicalForm(A,dtype=np.dtype("float"),Normalized=False):
    if ( not Normalized ):
        A = Normalize(A,dtype)
        Normalized = True
    VR,VL = RightLeftEigs(A,dtype,Normalized)
    DR,WR = sp.linalg.eigh(VR)
    X = np.einsum("ab,bc -> ac",WR,np.diag(np.sqrt(DR)))
    DL,WL = sp.linalg.eigh(VL)
    YT = np.einsum("ab,bc -> ac",WL,np.diag(np.sqrt(DL)))
    AC = np.einsum("ab,bsc,cd -> asd",YT.T,A,X)
    C = np.einsum("ab,bc -> ac",YT.T,X)
    AR = np.einsum("ab,bsc,cd -> asd",np.linalg.inv(X),A,X)
    AL = np.einsum("ab,bsc,cd -> asd",YT.T,A,np.linalg.inv(YT.T))
    return AC,C,AR,AL

# obtain the dominant right and left eigenvector
def RightLeftEigs(A,dtype=np.dtype("float"),Normalized=False):
    M = A.shape[0]
    if ( not Normalized ): A = Normalize(A,dtype)
    valR,vecR = MF.RightEigs(A,dtype)
    #valR,vecR = MF.Simple_RightEigs(A,dtype)
    valL,vecL = MF.RightEigs(A.transpose(2,1,0),dtype)
    #valL,vecL = MF.Simple_RightEigs(A.transpose(2,1,0),dtype)
    vecR /= np.inner(vecR,vecL)
    VR = vecR.reshape(M,M)
    VL = vecL.reshape(M,M)
    return VR,VL
    
def Normalize(A,dtype=np.dtype("float")):
    valR,vecR = MF.RightEigs(A,dtype)
    #valR,vecR = MF.Simple_RightEigs(A,dtype)
    A /= ( valR ** 0.5 )
    return A

def ExpectationValue(A,O,VR,VL):
    D = A.shape[1]; M = A.shape[0]
    AO = np.einsum("asb,st -> atb",A,O)
    VLAO = np.einsum("ac,atb -> ctb",VL,AO)
    VLAOVR = np.einsum("ctb,bd -> ctd",VLAO,VR)
    EXPO = np.inner(VLAOVR.reshape(M*D*M),np.conj(A).reshape(M*D*M))
    return EXPO
    
def Simple_ExpectationValue(A,O,dtype,Normalized=False):
    D = A.shape[1]; M = A.shape[0]
    if ( not Normalized ):
        A = Normalize(A)
        Normalized = True
    VR,VL = RightLeftEigs(A,dtype)
    AO = np.einsum("asb,st -> atb",A,O)
    VLAO = np.einsum("ac,atb -> ctb",VL,AO)
    VLAOVR = np.einsum("ctb,bd -> ctd",VLAO,VR)
    EXPO = np.inner(VLAOVR.reshape(M*D*M),np.conj(A).reshape(M*D*M))
    return EXPO

# Use C calculated by MixedCanonicalForm(A)
def EntanglementEntropy (C):
    c = EntanglementSpectrum(C)
    S = 0
    for i in range (len(c)):
        S -= c[i]**2 * np.log(c[i]**2)
    return S

# Use C calculated by MixedCanonicalForm(A)
def EntanglementSpectrum (C):
    _,c,_ = np.linalg.svd(C)
    norm = np.sum(c**2)
    c /= np.sqrt(norm)
    return c



