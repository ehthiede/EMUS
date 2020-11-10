# -*- coding: utf-8 -*-
"""
Collection of linear algebra routines used in the EMUS algorithm and
associated error analysis.
"""
from scipy.linalg import lu
from scipy.linalg import qr
from scipy.linalg import inv
from scipy.linalg import solve
import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr as qr_decompose
from scipy.linalg import solve_triangular
from scipy.linalg.lapack import dtrtri as invert_upper_triangular
from emus import usutils as uu


def _stationary_distrib_QR(F, residtol=1.E-10, max_iter=1000):
    """
    Calculates the eigenvector of the matrix F with eigenvalue 1 (if it exists).

    Parameters
    ----------
    F : ndarray
        A matrix known to have a single left eigenvector with
        eigenvalue 1.

    residtol : float or scalar
        To improve the accuracy of the computation, the algorithm will
        "polish" the final result using several iterations of the power
        method, z^T F = z^T.  Residtol gives the tolerance for the
        associated relative residual to determine convergence.

    maxiter : int
        Maximum number of iterations to use the power method to reduce
        the residual.  In practice, should never be reached.

    Returns
    -------
    z : ndarray
        The eigenvector of the matrix F with eigenvalue 1.  For a Markov
        chain stationary distribution, this is the stationary distribution.
        Normalization is chosen s.t. entries sum to one.

    """

    L = len(F)  # Number of states
    M = np.eye(L)-F
    q, r = qr(M)
    z = q[:, -1]  # Stationary dist. is last column of QR fact
    z /= np.sum(z)  # Normalize Trajectory
    # Polish solution using power method.
    for itr in range(max_iter):
        znew = np.dot(z, F)
        maxresid = np.max(np.abs(znew - z)/z)  # Convergence Criterion
        if maxresid < residtol:
            break
        else:
            z = znew

    return z/np.sum(z)  # Return normalized (by convention)


def stationary_distrib(F, fix=None, residtol=1.E-10, max_iter=10000, verbose=False):
    """
    Depricated routine to calculate the stationar distribution of F.
    """
    L = len(F)  # Number of states
    # If no fixed state is specified, we find a state with high weight in z.
    if fix is None:
        testz = stationary_distrib(F, 1)
        fix = np.argmax(testz)
    # We get the matrix subminor, and the fix'th row of F
    submat = _submatrix(F, fix)
    Fi = F[fix, :]
    # (I-Fsub)^T
    ImFt = np.transpose(np.eye(L-1)-submat)
    Fi = np.delete(Fi, fix)
    z = solve(ImFt, Fi)  # Partition fxns of the other states.
    z = np.insert(z, fix, 1.0)  # Put the state we fixed to 1. back in
    # Polish solution using power method.
    for itr in range(max_iter):
        znew = np.dot(z, F)
        maxresid = np.max(np.abs(znew - z)/z)  # Convergence Criterion
        if maxresid < residtol:
            if verbose:
                print("Reached Tolerance")
            break
        else:
            z = znew
    if verbose:
        print("Used %d iterations" % itr)
    return z/np.sum(z)


def _submatrix(F, i):
    """
    Calculates the submatrix of F with the i'th row and column removed.

    Parameters
    ----------
    F : ndarray
        A matrix with at least i rows and columns
    i : int
        The row or column to delete

    Returns
    -------
    submatrix: ndarray
        The ensuing submatrix with the i'th row and column deleted.
    """
    submat = np.copy(F)
    submat = np.delete(submat, i, axis=1)
    submat = np.delete(submat, i, axis=0)
    return submat


def groupInverse(M):
    """
    Computes the group inverse of stochastic matrix using the algorithm
    given by Golub and Meyer in:
    G. H. Golub and C. D. Meyer, Jr, SIAM J. Alg. Disc. Meth. 7, 273-
    281 (1986)

    Parameters
    ----------
        M : ndarray
            A square matrix with index 1.

    Returns
    -------
        grpInvM : ndarray
            The group inverse of M.
    """
    L = np.shape(M)[1]
    q, r = qr(M)
    piDist = q[:, L-1]
    piDist = (1/np.sum(piDist))*piDist
    specProjector = np.identity(L)-np.outer(np.ones(L), piDist)
    u = r[0:(L-1), 0:(L-1)]  # remember 0:(L-1) actually means 0 to L-2!
    uInv = inv(u)  # REPLACE W. lapack, invert triangular matrix ROUTINE
    uInv = np.real(uInv)
    grpInvM = np.zeros((L, L))
    grpInvM[0:(L-1), 0:(L-1)] = uInv
    grpInvM = np.dot(specProjector, np.dot(
        grpInvM, np.dot(q.transpose(), specProjector)))
    return grpInvM


def expanded_group_inv(B):
    A = B[:-2, :-2]
    v = B[:-2, -2:]
    b = B[-2:, -2:]
    B_inv = np.zeros(B.shape)

    A_inv = groupInverse(A)
    b_inv = inv(b)
    B_inv[:-2, :-2] = A_inv
    B_inv[-2:, -2:] = b_inv

    # Build right column
    h = (np.eye(len(A)) - A @ A_inv) @ v @ b_inv @ b_inv
    right_col = - A_inv @ v @ b_inv + h
    B_inv[:-2, -2:] = right_col
    return B_inv


def groupInverse_for_iteravar(M):
    """
    Computes the group inverse of a matrix using LU decomposition.

    Parameters
    ----------
        M : ndarray
            A square matrix with index 1.

    Returns
    -------
        grpInvM : ndarray
            The group inverse of M.
    """
    p, l, u = lu(M)
    R = np.dot(p, l)
    T = np.dot(u, R)
    T_inv = groupInverse_partial(T)
    R_inv = np.linalg.inv(R)
    return np.dot(np.dot(R, T_inv), R_inv)


def groupInverse_partial(T):
    """
    Computes the group inverse of a matrix with all zeros as the last row.
    """
    L = np.shape(T)[1]
    T1 = T[0:(L-1), 0:(L-1)]
    T1_inv = inv(T1)
    T2 = T[0:(L-1), L-1]
    T_inv = np.zeros((L, L))
    T_inv[0:(L-1), 0:(L-1)] = T1_inv
    T_inv[0:(L-1), L-1] = np.linalg.multi_dot([T1_inv, T1_inv, T2])
    '''
    print(np.linalg.norm(np.dot(T,T_inv)-np.dot(T_inv,T)))
    print(np.linalg.norm(np.linalg.multi_dot([T,T_inv,T])-T))
    print(np.linalg.norm(np.linalg.multi_dot([T_inv,T,T_inv])-T_inv))
    '''
    return T_inv


def build_G(psis, z, neighbors, kappa):
    L = len(z)
    G = np.zeros((L, L))
    for i, n_i in enumerate(neighbors):
        # Get data structures for neighboring windows
        psis_i = np.array(psis[i])
        neighb_kappas = kappa[n_i]
        neighb_zs = z[n_i]
        # Build stochastic matrix
        weighted_psis = psis_i * (neighb_kappas / neighb_zs)  # Weight by kappa and z
        weighted_psis /= np.sum(weighted_psis, axis=1, keepdims=1)  # row normalize
        Gi = np.mean(weighted_psis, axis=0)
        G[i] = uu.unpack_nbrs(Gi, n_i, L)
    return G.T


def calculate_GI_from_QR(psis, z, neighbs, kappa=None, return_T=False, return_div_z=False):
    L = len(z)
    avar_G_mat = build_G(psis, z, neighbs, kappa)

    print("Checking that G is column stochastic: ||e(I-G)||=", norm(np.ones(L) @ (np.eye(L) - avar_G_mat)))
    B = np.diag(1. / z) @ (np.eye(L) - avar_G_mat)
    Q, R = qr_decompose(np.eye(L) - avar_G_mat)
    U = R[:-1, :-1]
    r = R[:-1, -1]
    a = solve_triangular(U, -r)
    u = np.append(a, [1])
    zu_norm = np.dot(z, u)
    projector_mat = np.eye(L) - np.outer(u, z) / zu_norm
    Uinv, info = invert_upper_triangular(U)
    if info != 0:
        raise RuntimeError("Error in upper triangular inversion!  Error code %d" % info)
    B_inv = np.zeros((L, L))
    B_inv[:-1, :-1] = Uinv
    B_inv = B_inv @ Q.T
    B_inv = projector_mat @ (B_inv*z) @ projector_mat
    B_ginv = B_inv
    Bmat = B
    print("new residues")
    print(np.linalg.norm(np.dot(Bmat, B_ginv)-np.dot(B_ginv, Bmat)))
    print(np.linalg.norm(np.linalg.multi_dot([Bmat, B_ginv, Bmat])-Bmat)/np.linalg.norm(Bmat))
    print(np.linalg.norm(np.linalg.multi_dot([B_ginv, Bmat, B_ginv])-B_ginv)/np.linalg.norm(B_ginv))
    if return_div_z:
        return B_inv, projector_mat @ B_inv @ projector_mat, np.outer(u, z) / zu_norm
    elif return_T:
        return B_inv, np.outer(u, z) / zu_norm
    else:
        return B_inv


def GI_expanded(psis, z, g1, g2, g1data, g2data, neighbors, kappa):
    L = len(z)
    v = np.zeros((L, 2))
    for i in range(L):
        psis_i = np.array(psis[i])
        Lneighb = len(neighbors[i])
        denom = np.sum(np.array([kappa[neighbors[i][j]]*psis_i[:, j]/z[neighbors[i][j]] for j in range(Lneighb)]), axis=0)
        v1 = -kappa[i]*g1data[i]/g1/denom/z[i]
        v[i, 0] = float(np.mean(v1, axis=0))
        v2 = -kappa[i]*g2data[i]/g2/denom/z[i]
        v[i, 1] = float(np.mean(v2, axis=0))
    b_inv = np.diag([g1, g2])
    avar_G_mat = build_G(psis, z, neighbors, kappa)
    avar_G_mat = np.diag(1/z)@(np.eye(L)-avar_G_mat)
    Bmat = np.vstack((np.hstack((avar_G_mat, v)), np.hstack((np.zeros((2, L)), np.diag([1/g1, 1/g2])))))
    G_inv, T = calculate_GI_from_QR(psis, z, neighbors, kappa, return_T=True)
    right_col = np.vstack((-G_inv@v@b_inv+T@v@b_inv@b_inv, b_inv))
    B_ginv = np.hstack((np.vstack((G_inv, np.zeros((2, L)))), right_col))
    print("new residues of B")
    print(np.linalg.norm(np.dot(Bmat, B_ginv)-np.dot(B_ginv, Bmat)))
    print(np.linalg.norm(np.linalg.multi_dot([Bmat, B_ginv, Bmat])-Bmat)/np.linalg.norm(Bmat))
    print(np.linalg.norm(np.linalg.multi_dot([B_ginv, Bmat, B_ginv])-B_ginv)/np.linalg.norm(B_ginv))
    return np.vstack((-G_inv@v@b_inv+T@v@b_inv@b_inv, b_inv))
