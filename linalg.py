import numpy as np
from functools import reduce
import copy

np.set_printoptions(suppress=True)


def get_norm(U):
    return np.sqrt(np.dot(U, U))


def unit_shorten(U):
    norm = get_norm(U)
    return U * 1 / norm

def subspace_projection(u, V): # WRONG
    proj = np.zeros_like(V[:, 1])
    V = qr_decompose(V)[0] # gram_schmidt
    for j in range(V.shape[1]):
        coefficient = np.dot(u, V[:, j])/np.dot(V[:, j], V[:, j])
        proj += coefficient * V[:, j]
    return proj

def subspace_projection_v2(u, V):
    VtV = V.transpose() @ V
    Vtu = V.transpose() @ u
    x = np.linalg.solve(VtV, Vtu)
    return V @ x

def least_squares(A, b):
    R, Q = qr_decompose(A)
    y = (invert(R) @ Q.transpose())
    e = np.linalg.norm(b - A @ y)
    return y, e

def subspace_projection_v3(u, V):
    return V @ invert(V.transpose() @ V) @ V.transpose() @ u

def projection(U, V):
    U = U.flatten()
    V = V.flatten()
    dot = np.dot(U, V)
    norm = get_norm(U) ** 2
    proj = dot / norm * U
    return proj

def round(num):
    return np.round(num, 5)


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
        v = orthmat[:, i].reshape(-1, )
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


def invert(A):
    if A.ndim == 0:
        return 1/np.expand_dims(A, axis=0)
    pivcol = 0
    check = 1
    n = A.shape[0]
    augment = np.eye(n)
    augmented = np.concatenate([A, augment], axis=1)
    while not np.all(augmented[0:n, 0:n] == np.eye(n)):  # gaussian elimination until inverse is found
        while np.abs(augmented[pivcol, pivcol]) == 0:  # swap until nonzero pivot is found
            if np.abs(augmented[pivcol + check, pivcol]) != 0:
                temp = copy.deepcopy(augmented[pivcol + check, :])
                augmented[pivcol + check, :] = augmented[pivcol, :]
                augmented[pivcol, :] = temp
                check = 1
            check += 1
            if pivcol + check > n:
                raise ValueError('Low rank matrix detected!')
        augmented[pivcol, :] /= augmented[pivcol, pivcol]

        for i in range(pivcol + 1, n):
            augmented[i, :] -= augmented[pivcol, :] * augmented[i, pivcol]
        # matrix is now in row-echelon form
        pivcol += 1
        if pivcol == n:
            break  # matrix is now in row-echelon form

    for i in reversed(range(1, pivcol)):
        for j in reversed(range(0, i)):
            augmented[j, :] -= augmented[i, :] * augmented[j, i]
    return augmented[:, n:]


def diagonalize(A):  # Iterative Givens Rotations to find QR decomposition
    iters = 0
    A_next = A.copy()
    A_prev = np.eye(A.shape[0], M=A.shape[1], dtype=np.float64)
    Q_record = []
    # Iterative QR decomposition
    while not check_convergence(A_next, A_prev):
        A_prev = A_next
        Q, R = qr_decompose(A_next)
        A_next = R @ Q
        Q_record.append(Q)
        iters += 1
        if iters > 999:
            print('Warning: 999 iterations reached without convergence!\n'
                  'Likely reason is that there is no exact QR solution')
            break
    diagonal_matrix = A_next
    val_matrix = reduce(np.matmul, Q_record)
    v2_matrix = np.eye(A.shape[0])
    for i in range(len(Q_record)):
        v2_matrix = np.matmul(v2_matrix, Q_record[i])
    inverse_val_matrix = invert(val_matrix)


    return inverse_val_matrix, diagonal_matrix, val_matrix


if __name__ == '__main__':
    A = np.array([[4, 0, 1], [2, 3, 2], [1, 0, 4]], dtype=float)  # Diagonalizable matrix
   # A = np.array([[4, 0, 1], [0, 3, 2], [0, 0, 4]], dtype=float)  # Deficient Matrix
  #  A = np.array([[1.0, 6.0, -1.0],[6.0, -1.0, -2.0],[-1.0, -2.0, -1.0]])
    inverse_val_matrix, diagonal_matrix, val_matrix = diagonalize(A)

    val_inverse_str = np.array2string(inverse_val_matrix, formatter={'float_kind': lambda x: f"{x:6.4f}"})
    val_matrix_str = np.array2string(val_matrix, formatter={'float_kind': lambda x: f"{x:6.4f}"})
    diagonal_matrix_str = np.array2string(diagonal_matrix, formatter={'float_kind': lambda x: f"{x:6.4f}"})

    print(
        f'Diagonalization of A:\n {A} = __________\n {val_matrix_str} *\n ________________________\n {diagonal_matrix_str} *\n ________________________\n {val_inverse_str} \n ________________________\n = P * D * P_inverse')
    print(inverse_val_matrix @ val_matrix  @ diagonal_matrix)