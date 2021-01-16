import numpy as np
import scipy as sp
import MathFunctions as MF

#######################################################################################
# single-site unit cell
#######################################################################################

# transform a tensor A into the mixed canonical form
def MixedCanonicalForm(A,dtype=np.dtype("float"),Normalized=False):
    if ( not Normalized ):
        A = Normalize(A,dtype)
        Normalized = True
    VR,VL = RightLeftEigs(A,dtype,Normalized)
    DR,WR = sp.linalg.eigh(VR)
    X = np.einsum("ab,bc -> ac",WR,np.diag(np.sqrt(abs(DR))))
    DL,WL = sp.linalg.eigh(VL)
    YT = np.einsum("ab,bc -> ac",WL,np.diag(np.sqrt(abs(DL))))
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
    # A /= ( valR ** 0.5 )
    A /= ( np.sqrt(abs(valR)) )
    return A
  
# Calculate <Psi(A)|O|Psi(A)>
# O.shape = (D**Osize,D**Osize)
# Energy is given by <Psi(A)|h|Psi(A)>
def Simple_ExpectationValue(A,O,dtype,Osize=2,Normalized=False):
    D = A.shape[1]; M = A.shape[0]
    if ( not Normalized ):
        A = Normalize(A)
        Normalized = True
    VR,VL = RightLeftEigs(A,dtype)
    Aall = A
    for i in range(1,Osize):
        Aall = np.tensordot(Aall,A,([2],[0])).reshape(M,D**(i+1),M)
    O = O.reshape(D**Osize,D**Osize)
    AallO = np.einsum("asb,st -> atb",Aall,O)
    VLAallO = np.einsum("ac,atb -> ctb",VL,AallO)
    VLAallOVR = np.einsum("ctb,bd -> ctd",VLAallO,VR)
    EXPO = np.inner(VLAallOVR.reshape(M*M*D**Osize),np.conj(Aall).reshape(M*M*D**Osize))
    return EXPO
    
#######################################################################################

#def ExpectationValue(A,O,VR,VL):
#    D = A.shape[1]; M = A.shape[0]
#    AO = np.einsum("asb,st -> atb",A,O)
#    VLAO = np.einsum("ac,atb -> ctb",VL,AO)
#    VLAOVR = np.einsum("ctb,bd -> ctd",VLAO,VR)
#    EXPO = np.inner(VLAOVR.reshape(M*D*M),np.conj(A).reshape(M*D*M))
#    return EXPO

#######################################################################################
#def Simple_ExpectationValue(A,O,dtype,Normalized=False):
#    D = A.shape[1]; M = A.shape[0]
#    if ( not Normalized ):
#        A = Normalize(A)
#        Normalized = True
#    VR,VL = RightLeftEigs(A,dtype)
#    AO = np.einsum("asb,st -> atb",A,O)
#    VLAO = np.einsum("ac,atb -> ctb",VL,AO)
#    VLAOVR = np.einsum("ctb,bd -> ctd",VLAO,VR)
#    EXPO = np.inner(VLAOVR.reshape(M*D*M),np.conj(A).reshape(M*D*M))
#    return EXPO
#######################################################################################

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

def CorrelationLegnth(A,dtype):
    T_ope = MF.RightEigsOpe(A,dtype)
    valR,vecR =  sp.sparse.linalg.eigs(T_ope,k=2)
    corr = -1/np.log(abs(valR[1]))
    return corr

# If O.shape = (D,D) e.g. Sx,Sy,Sz, Osize = 1
# If O.shape = (D**2,D**2) e.g. SS, Osize = 2
def CorrealtionFunction(A,O,length,dtype,Osize=1):
    M = A.shape[0]; D = A.shape[1]
    VR,VL = RightLeftEigs(A,dtype,Normalized=True)
    R = CalcR(A,O,VR,Osize=Osize)
    L = CalcL(A,O,VL,Osize=Osize)
    dis_list = []; corr_list = []
    for r in range(Osize,length):
        LR = np.inner(R.reshape(M*M),L.reshape(M*M))
        L = AddTM(L,A)
        dis_list.append(r); corr_list.append(LR)
    return dis_list,corr_list

def CalcR(A,O,VR,Osize=1):
    M = A.shape[0]; D = A.shape[1]
    Aall = A
    for i in range(1,Osize):
        Aall = np.tensordot(Aall,A,([2],[0])).reshape(M,D**(i+1),M)
    AallVR = np.tensordot(Aall,VR,([2],[0]))
    AallVRO = np.tensordot(AallVR,O,([1],[0]))
    R = np.tensordot(AallVRO,np.conj(Aall),([1,2],[2,1]))
    return R
    
def CalcL(A,O,VL,Osize=1):
    M = A.shape[0]; D = A.shape[1]
    Aall = A
    for i in range(1,Osize):
        Aall = np.tensordot(Aall,A,([2],[0])).reshape(M,D**(i+1),M)
    AallVL = np.tensordot(Aall,VL,([0],[0]))
    AallVLO = np.tensordot(AallVL,O,([0],[0]))
    L = np.tensordot(AallVLO,np.conj(Aall),([1,2],[0,1]))
    return L

#def CalcR(A,O,VR):
#    AVR = np.tensordot(A,VR,([2],[0]))
#    AVRO = np.tensordot(AVR,O,([1],[0]))
#    R = np.tensordot(AVRO,np.conj(A),([1,2],[2,1]))
#    return R

#def CalcL(A,O,VL):
#    AVL = np.tensordot(A,VL,([0],[0]))
#    AVLO = np.tensordot(AVL,O,([0],[0]))
#    L = np.tensordot(AVLO,np.conj(A),([1,2],[0,1]))
#    return L
    
# add a transfer matrix to a left vector L
def AddTM(L,A):
    L2 = np.tensordot(L,A,([0],[0]))
    L3 = np.tensordot(L2,np.conj(A),([0,1],[0,1]))
    return L3

#######################################################################################
# two-site unit cell
#######################################################################################

# O.shape = (D*D,D*D) e.g. S_j S_j+1
def CorrealtionFunction_twosite_unitcell(A0,A1,O,length,dtype):
    M = A0.shape[0]; D = A0.shape[1]
    A01 = np.tensordot(A0,A1,([2],[0])).reshape(M,D*D,M)
    A10 = np.tensordot(A1,A0,([2],[0])).reshape(M,D*D,M)
    VR01,VL01 = RightLeftEigs(A01,dtype,Normalized=True)
    VR10,VL10 = RightLeftEigs(A10,dtype,Normalized=True)
    R01 = CalcR(A01,O,VR01)
    R10 = CalcR(A10,O,VR10)
    L01 = CalcL(A01,O,VL01)
    dis_list = []; corr_list = []
    for r in range(2,length):
        if r % 2 == 0:
            LR = np.inner(L01.reshape(M*M),R01.reshape(M*M))
            L10 = AddTM(L01,A0)
        else:
            LR = np.inner(L10.reshape(M*M),R10.reshape(M*M))
            L01 = AddTM(L10,A1)
        dis_list.append(r); corr_list.append(LR)
    return dis_list,corr_list

# O.shape = (D*D*D,D*D*D) e.g. S_j S_j+1 - S_j+1 S_j+2
def CorrealtionFunction_twosite_unitcell2(A0,A1,O,length,dtype):
    M = A0.shape[0]; D = A0.shape[1]
    A01 = np.tensordot(A0,A1,([2],[0])).reshape(M,D*D,M)
    A10 = np.tensordot(A1,A0,([2],[0])).reshape(M,D*D,M)
    A010 = np.tensordot(A01,A0,([2],[0])).reshape(M,D*D*D,M)
    A101 = np.tensordot(A1,A01,([2],[0])).reshape(M,D*D*D,M)
    VR01,VL01 = RightLeftEigs(A01,dtype,Normalized=True)
    VR10,VL10 = RightLeftEigs(A10,dtype,Normalized=True)
    R010 = CalcR(A010,O,VR10)
    R101 = CalcR(A101,O,VR01)
    L010 = CalcL(A010,O,VL01)
    dis_list = []; corr_list = []
    for r in range(3,length):
        if r % 2 == 0:
            LR = np.inner(L101.reshape(M*M),R010.reshape(M*M))
            L010 = AddTM(L101,A0)
        else:
            LR = np.inner(L010.reshape(M*M),R101.reshape(M*M))
            L101 = AddTM(L010,A1)
        dis_list.append(r); corr_list.append(LR)
    return dis_list,corr_list


#######################################################################################
# multi-site unit cell
#######################################################################################

def CorrelationLegnth_multi(A,dtype,Asize=1):
    M = A[0].shape[0]; D = A[0].shape[1];
    Aall = A[0]
    for i in range(1,Asize):
        Aall = np.tensordot(Aall,A[i],([2],[0])).reshape(M,D**(i+1),M)
    T_ope = MF.RightEigsOpe(Aall,dtype)
    valR,vecR =  sp.sparse.linalg.eigs(T_ope,k=2)
    corr = -Asize/np.log(abs(valR[1]))
    return corr






