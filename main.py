import numpy as np
from functools import reduce
np.set_printoptions(suppress=True)
def get_norm(U):
    return np.sqrt(np.dot(U, U))

def unit_shorten(U):
    norm = get_norm(U)
    return U * 1/norm

def projection(U, V):
    U = U.flatten()
    V = V.flatten()
    dot = np.dot(U, V)
    norm = get_norm(U) ** 2
    proj = dot/norm * U
    return proj


def check_convergence(A, B):
    D = A - B
    for i in D.flatten():
        if i > 1e-6:
            return False
    return True

def qr_decompose(arr):
    m = arr.shape[0]
    n = arr.shape[1]
    orthmat = arr.copy()
    upper = np.zeros(arr.shape)

    # set 1, 1 element of R
    upper[0, 0] = get_norm(arr[:, 0])

    for i in range(orthmat.shape[1]):
        v = orthmat[:, i].reshape(-1,)
        if i > 0:
            residual_sum = np.zeros(v.shape)
            U = np.split(orthmat[:, :i], i, axis=1)
            for u in U:
                residual = projection(u, v)
                residual_sum += residual
            v -= residual_sum
            upper[i, i] = get_norm(v)
        orthmat[:, i] = unit_shorten(v)
# calculate the entries in R that are above the diagonal
    for i in range(m):
        for j in range(n):
            if j > i:
                upper[i, j] = np.dot(arr[:, j], orthmat[:, i])
    return orthmat, upper

def eigvals(A):
    iters = 0
    A_next = A.copy()  # Initialize A_next with a copy of A
    A_prev = np.eye(A.shape[0], dtype=np.float64)
    Q_record = []
    # Iterative QR decomposition
    while not check_convergence(A_next, A_prev):
        A_prev = A_next  # Store the current A_next as A_prev before updating
        Q, R = qr_decompose(A_next)
        A_next = R @ Q
        Q_record.append(Q)
        iters += 1
        if iters > 999:
            print('Warning: 999 iterations reached without convergence!\n'
                  'Likely reason is that there is no exact QR solution')
            break
    eigenmatrix = reduce(np.matmul, Q_record)
    eigenvalues = np.array([A_next[i, i] for i in range(A.shape[0])])
    print('Eigenvalues are calculated as: ', eigenvalues)
    print('Eigenvectors are calculated as: ', Q_record)
    return eigenvalues, eigenmatrix

if __name__ == '__main__':
    import sys
    print(sys.version)
   # eigvals(A=np.array([[2, 1, 3], [0, 1, 4], [0, 0, 5]], dtype=float))
  #  print('done!')