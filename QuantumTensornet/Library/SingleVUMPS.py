#############Caution#############
# Do not use this code !!!
# This code is not completed !!!
#################################

##################################################################################
# Reference
# [1] Phys. Rev. B 97, 045145 (2018)
# https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.045145
# https://arxiv.org/abs/1701.07035
##################################################################################
# Variational uniform Matrix Product State algorithm for single-site unit cells
# see Algorithms 1, 2 in Ref. [1]
##################################################################################
# Simple_* means that the code is easy to read but the computational cost is high
##################################################################################

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

import MathFunctions as MF
import MPSOperators as MO

##################################################################################
# Line 2 in Algorith 1
##################################################################################

# Eq. (12)
def Calc_hr(AR,h):
    ARAR = np.tensordot(AR,AR,([2],[0]))
    ARARh = np.tensordot(ARAR,h,([1,2],[0,1]))
    hr = np.tensordot(ARARh,np.conj(ARAR),([2,3,1],[1,2,3]))
    return hr

# Eq. (12)
def Calc_hl(AL,h):
    ALAL = np.tensordot(AL,AL,([2],[0]))
    ALALh = np.tensordot(ALAL,h,([1,2],[0,1]))
    hl = np.tensordot(ALALh,np.conj(ALAL),([0,2,3],[0,1,2]))
    return hl

# Calculate the Energy by using hr
def Calc_right_energy(AR,h,dtype):
    M = AR.shape[0]
    hr = Calc_hr(AR,h)
    _,vecL = MF.RightEigs(AR.transpose(2,1,0),dtype)
    VL = vecL.reshape(M,M)
    VL /= np.trace(VL)
    energy = np.inner(hr.reshape(M*M),VL.reshape(M*M))
    return energy

# Calculate the Energy by using hl
def Calc_left_energy(AL,h,dtype):
    M = AL.shape[0]
    hl = Calc_hl(AL,h)
    _,vecR = MF.RightEigs(AL,dtype)
    VR = vecR.reshape(M,M)
    VR /= np.trace(VR)
    energy = np.inner(hl.reshape(M*M),VR.reshape(M*M))
    return energy

##################################################################################
# Line 3 in Algorith 1
##################################################################################

# Eq. (13)
# use previous HR
def Calc_HR(AR,HR,h,dtype,tol=1e-17):
    M = AR.shape[0]
    hr = Calc_hr(AR,h)
    er = Calc_right_energy(AR,h,dtype)
    hr -= er * np.eye(M,M)
    #HR = Calc_HR_PowerMethod(AR,hr)
    #HR = Calc_HR_Inverse(AR,hr)
    HR = Simple_Calc_HR_Linear(AR,HR,hr,dtype,tol)
    return HR,er

# Eq. (13)
# use previous HL
def Calc_HL(AL,HL,h,dtype,tol=1e-17):
    M = AL.shape[0]
    hl = Calc_hl(AL,h)
    el = Calc_left_energy(AL,h,dtype)
    hl -= el * np.eye(M,M)
    #HL = Calc_HL_PowerMethod(AL,hl)
    #HL = Calc_HL_Inverse(AL,hl)
    HL = Simple_Calc_HL_Linear(AL,HL,hl,dtype,tol)
    return HL,el

# Eq. (15)
def Simple_Calc_HR_Linear(AR,HR,hr,dtype,tol):
    M = AR.shape[0]
    TR = np.einsum("asc,bsd -> abcd",AR,np.conj(AR)).reshape(M*M,M*M)
    _,vecL = MF.RightEigs(AR,dtype)
    VL = vecL.reshape(M,M)
    VL /= np.trace(VL)
    P = np.einsum("ab,cd -> abcd",np.eye(M),VL).reshape(M*M,M*M)
    PL = np.eye(M*M) - P
    R = np.einsum("ab,b -> a",PL,hr.reshape(M*M))
    T = np.eye(M*M) - TR + P
    HR,_ = sp.sparse.linalg.bicgstab(T,R,atol=tol,tol=tol,x0=HR.reshape(M*M))
    return HR.reshape(M,M)

# Eq. (15)
def Simple_Calc_HL_Linear(AL,HL,hl,dtype,tol):
    M = AL.shape[0]
    TL = np.einsum("asc,bsd -> abcd",AL,np.conj(AL)).reshape(M*M,M*M)
    _,vecR = MF.RightEigs(AL.transpose(2,1,0),dtype)
    VR = vecR.reshape(M,M)
    VR /= np.trace(VR)
    P = np.einsum("ab,cd -> abcd",np.eye(M),VR).reshape(M*M,M*M)
    PR = np.eye(M*M) - P
    R = np.einsum("ab,b -> a",PR,hl.reshape(M*M))
    T = np.eye(M*M) - TL.T + P
    HL,_ = sp.sparse.linalg.bicgstab(T,R,atol=tol,tol=tol,x0=HL.reshape(M*M))
    return HL.reshape(M,M)
    
def Calc_HR_Inverse(AR,hr):
    M = AR.shape[0]
    TR = np.einsum("asc,bsd -> abcd",AR,np.conj(AR)).reshape(M*M,M*M)
    Tinf = sp.linalg.inv(np.eye(M*M) - TR)
    HR = np.einsum("ab,b -> a",Tinf,hr.reshape(M*M))
    return HR.reshape(M,M)

def Calc_HL_Inverse(AL,hl):
    M = AL.shape[0]
    TL = np.einsum("asc,bsd -> abcd",AL,np.conj(AL)).reshape(M*M,M*M)
    Tinf = sp.linalg.inv(np.eye(M*M) - TL)
    HL = np.einsum("a,ab -> b",hl.reshape(M*M),Tinf)
    return HL.reshape(M,M)

# Eq. (14)
def Calc_HR_PowerMethod(AR,hr,tol=1e-15,max_iter=10000):
    M = hr.shape[0]
    HR = hr
    for i in range(max_iter):
        HR_ini = HR
        ARHR = np.tensordot(AR,HR,([2],[0]))
        TRHR = np.tensordot(ARHR,np.conj(AR),([1,2],[1,2]))
        HR = TRHR + hr
        if ( np.allclose(HR,HR_ini,rtol=0,atol=tol*M*M) ):
            print(i)
            break
    return HR

# Eq. (14)
def Calc_HL_PowerMethod(AL,hl,tol=1e-15,max_iter=10000):
    M = hl.shape[0]
    HL = hl
    for i in range(max_iter):
        HL_ini = HL
        ALHL = np.tensordot(HL,AL,([0],[0]))
        TLHL = np.tensordot(ALHL,np.conj(AL),([0,1],[0,1]))
        HL = TLHL + hl
        if ( np.allclose(HL,HL_ini,rtol=0,atol=tol*M*M) ):
            print(i)
            break
    return HL

##################################################################################
# Line 5,6,7 in Algorithm 2
##################################################################################

# Line 6 in Algorithm 2
# Calculate updated AC from Eq. (11)
def Simple_Next_AC(AC,AR,AL,HR,HL,h,dtype):
    M = AR.shape[0]; D = AR.shape[1]
    HAC = Simple_EffectiveHamiltonian_HAC(AR,AL,HR,HL,h)
    val,vec = sp.sparse.linalg.eigs(HAC,k=1,which="SR",v0=AC)
    if ( dtype == np.dtype("float") ): vec = vec.real
    return vec.reshape(M,D,M)

# Line 7 in Algorithm 2
# Calculate updated C from Eq. (16)
def Simple_Next_C(C,AR,AL,HR,HL,h,dtype):
    M = AR.shape[0]; D = AR.shape[1]
    HC = Simple_EffectiveHamiltonian_HC(AR,AL,HR,HL,h)
    val,vec = sp.sparse.linalg.eigs(HC,k=1,which="SR",v0=C)
    if ( dtype == np.dtype("float") ): vec = vec.real
    return vec.reshape(M,M)
  
# Line 5 in Algorithm 2
# Eq. (9), Eq. (11)
def Simple_EffectiveHamiltonian_HAC(AR,AL,HR,HL,h):
    M = AR.shape[0]; D = AR.shape[1]
    Block1 = np.einsum("ghd,gia,heib,fc -> abcdef",AL,np.conj(AL),h,np.eye(M)).reshape(M*D*M,M*D*M)
    Block2 = np.einsum("fhg,cig,ehbi,da -> abcdef",AR,np.conj(AR),h,np.eye(M)).reshape(M*D*M,M*D*M)
    Block3 = np.einsum("ab,cd -> bdac",HL,np.eye(D*M)).reshape(M*D*M,M*D*M)
    Block4 = np.einsum("ab,cd -> bdac",np.eye(D*M),HR).reshape(M*D*M,M*D*M)
    return Block1 + Block2 + Block3 + Block4

# Line 5 in Algorithm 2
# Eq. (10), Eq. (16)
def Simple_EffectiveHamiltonian_HC(AR,AL,HR,HL,h):
    M = AR.shape[0];
    Block1 = np.einsum("efc,eha,dgj,bij,fghi",AL,np.conj(AL),AR,np.conj(AR),h).reshape(M*M,M*M)
    Block2 = np.einsum("ab,cd -> bdac",HL,np.eye(M)).reshape(M*M,M*M)
    Block3 = np.einsum("ab,cd -> bdac",np.eye(M),HR).reshape(M*M,M*M)
    return Block1 + Block2 + Block3
    
##################################################################################
# Line 8 in Algorithm 2
##################################################################################

# Eqs. (19), (20)
def Next_AR_SVD(AC,C):
    M = AC.shape[0]; D = AC.shape[1]
    ACR = AC.reshape(M,D*M)
    CACR = np.einsum("ab,bc -> ac",np.conj(C).T,ACR)
    U,s,V = sp.linalg.svd(CACR,full_matrices=False)
    AR = np.einsum("ab,bc -> ac",U,V)
    return AR.reshape(M,D,M)

# Eqs. (19), (20)
def Next_AL_SVD(AC,C):
    M = AC.shape[0]; D = AC.shape[1]
    ACL = AC.reshape(M*D,M)
    ACLC = np.einsum("ab,bc -> ac",ACL,np.conj(C).T)
    U,s,V = sp.linalg.svd(ACLC,full_matrices=False)
    AL = np.einsum("ab,bc -> ac",U,V)
    return AL.reshape(M,D,M)

# Eqs. (21), (22)
def Next_AR_PolarDecomposition(AC,C):
    M = AC.shape[0]; D = AC.shape[1]
    ACR = AC.reshape(M,D*M)
    UAR,PAR = sp.linalg.polar(ACR,side = "right")
    UC,PC = sp.linalg.polar(C,side = "right")
    AR = np.einsum("ab,bc -> ac",np.conj(UC).T,UAR)
    return AR.reshape(M,D,M)

# Eqs. (21), (22)
def Next_AL_PolarDecomposition(AC,C):
    M = AC.shape[0]; D = AC.shape[1]
    ACL = AC.reshape(M*D,M)
    UAL,PAL = sp.linalg.polar(ACL)
    UC,PC = sp.linalg.polar(C)
    AL = np.einsum("ab,bc -> ac",UAL,np.conj(UC).T)
    return AL.reshape(M,D,M)
    
##################################################################################
# Line 9 in Algorithm 2
##################################################################################

# Eqs. (18), (24)
def Calc_B(AC,C,AR,AL):
    BL_square = 0
    D = AC.shape[1]
    for i in range(D):
        B = AC[:,i,:] - np.einsum("ab,bc -> ac",AL[:,i,:],C)
        BL_square += np.sum(abs(B)**2)
    BR_square = 0
    for i in range(D):
        B = AC[:,i,:] - np.einsum("ab,bc -> ac",C,AR[:,i,:])
        BR_square += np.sum(abs(B)**2)
    B_square = max(BL_square,BR_square)
    return np.sqrt(B_square)
