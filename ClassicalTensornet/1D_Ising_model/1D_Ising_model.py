import numpy as np

# The exact free energy
def f_exact (beta,J,N):
    K = beta * J
    f = - np.log( 2 * np.cosh(K) ) / beta - np.log( 1 + np.tanh(K) ** N ) / ( N * beta )
    return f

# The initial state of the transfer matrix T
def init_tensor_1D_Ising (beta,J):
    K = beta * J
    W = np.array([[np.sqrt(np.cosh(K)),np.sqrt(np.sinh(K))],
                  [np.sqrt(np.cosh(K)),-np.sqrt(np.sinh(K))]])
    T = np.einsum("ab,bc -> ac",W,W.T)
    return T

# Calculate tr[T^N]/N, where N = 2 ** N_itr
def renormalize_1D (T,N_itr):
    lnZ = 0.0
    for x in range(N_itr):
        alpha = np.amax(abs(T))
        T = T / alpha # normalize
        lnZ += 2 ** ( N_itr - x ) * np.log(alpha)
        T = np.einsum("ab,bc -> ac",T,T)
    trace = T[0][0] + T[1][1]
    lnZ += np.log(trace)
    return lnZ / 2 ** N_itr

beta = 2.0; J = 1.0; N_itr = 20
T = init_tensor_1D_Ising(beta,J)
lnZ = renormalize_1D(T,N_itr)
f = - lnZ / beta
f_ex = f_exact(beta,J,2**N_itr)

print ("exact",f_ex,"calculation",f,"error",f-f_ex)

