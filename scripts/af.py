import numpy as np

# Adaptive Filter(NLMS)
def NLMS(d, x, N, mu):
    # inputs
    #   d: Desired response(signal + noise)
    #   x: Input data(noise)
    #   N: Filter length
    #   mu: step size
    # output
    #   s: output data(signal)

    # signal length
    L = len(x)

    # init
    phi = np.zeros(N)
    w = np.zeros(N)
    s = np.zeros(L)
    
    # Adaptive Filter(NLMS)
    for i in range(L):
        # phi: The vector of buffered input data at step i
        phi[1:] = phi[0:-1]
        phi[0] = x[i]

        # error
        e = d[i] - np.dot(w, phi)
        
        # filter update
        w = w + mu * e / (0.01 + np.dot(phi, phi)) * phi

        s[i] = e

    return s
