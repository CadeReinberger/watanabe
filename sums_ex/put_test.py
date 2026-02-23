import taylor_lib
import numpy as np

def A(n):
    # First, let's get the c we're gonna need
    N = 2*n-1
    leading = taylor_lib.fill(1, -3, N)
    following = taylor_lib.series_power(taylor_lib.fill(1, -14, 9, N), .5)
    c = .25 * (leading - following)
    # Next, let's make our matrix A
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = c[i+j+1]
    # Get the determinant
    det_A = np.linalg.det(A)
    print(np.linalg.eigvals(A))
    return det_A

def trA(n):
    # First, let's get the c we're gonna need
    N = 2*n-1
    leading = taylor_lib.fill(1, -3, N)
    following = taylor_lib.series_power(taylor_lib.fill(1, -14, 9, N), .5)
    c = .25 * (leading - following)
    print(c)
    # Next, let's make our matrix A
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = c[i+j+1]
    # Get the determinant
    tr_A = np.trace(A)
    return tr_A

# print(f'tr_A: {[trA(n) for n in range(1, 5)]}')

print(f'A: {[np.log10(A(n)) for n in range(1, 6)]}')
    
            