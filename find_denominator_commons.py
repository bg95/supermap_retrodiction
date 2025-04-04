import cvxpy as cp
import numpy as np
numpy = np
import scipy
import math, cmath
import time
import pickle
import datetime
import matplotlib.pyplot as plt
import itertools
import sympy

#from print_mat import *

def save(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load(filename):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj

def rand_probability(d):
    return numpy.random.dirichlet([1] * d)

def rand_hermitian(d):
    U = random_unitary(d)
    gamma = U @ np.diag(rand_probability(d)) @ U.conj().T
    gamma = (gamma + gamma.conj().T) / 2
    return gamma

def shift_mineigval(gamma, mineig=1e-15):
    gamma_mineig = min(scipy.linalg.eigvalsh(gamma))
    return gamma + max(0, mineig - gamma_mineig) * np.eye(gamma.shape[0])

def random_state(din, mineig=1e-15):
    gamma = shift_mineigval(rand_hermitian(din), mineig=mineig)
    gamma = gamma / np.trace(gamma)
    return gamma

def is_sympy(*args):
    return all(isinstance(A, sympy.Basic | sympy.Matrix) for A in args)

def powmh(A, t):
    if is_sympy(A):
        return A ** t
    w, v = scipy.linalg.eigh(A)
    return v @ np.diag(w ** t) @ v.conj().T

def sqrtmh(A):
    return powmh(A, 1/2)

def kron(*args):
    r = 1
    for a in args:
        r = np.kron(r, a)
    return r

def channel_choi_project(C, dout, din): # projects C into the subspace of Choi operators of channels
    C = (C + C.conj().T) / 2
    return C - cp.kron(np.eye(dout) / dout, cp.partial_trace(C, [dout, din], 0)) + cp.trace(C) * np.eye(din * dout) / (din * dout)

def channel_choi_project_inout(C, din, dout): # projects C into the subspace of Choi operators of channels
    C = (C + C.conj().T) / 2
    return C - cp.kron(cp.partial_trace(C, [din, dout], 1), np.eye(dout) / dout) + cp.trace(C) * np.eye(din * dout) / (din * dout)

def np_channel_choi_project_inout(C, din, dout): # projects C into the subspace of Choi operators of channels
    C = (C + C.conj().T) / 2
    return C - np.kron(np_partial_trace(C, [din, dout], 1), np.eye(dout) / dout) + np.trace(C) * np.eye(din * dout) / (din * dout)

def is_choi(C, din, dout, rtol=1e-05, atol=1e-08): # order of indices is: in(row), out(row); in(col), out(col)
    return np.allclose(np_channel_choi_project_inout(C, din, dout), C, rtol=rtol, atol=atol)

def random_channel_choi(dout, din, mineig=1e-15):
    d2 = din
    d3 = dout
    Gamma = rand_hermitian(d3 * d2)
    Gamma = channel_choi_project(Gamma, d3, d2).value
    Gamma = shift_mineigval(Gamma, mineig=mineig)
    Gamma = Gamma / np.trace(Gamma) * d2
    return Gamma

def random_channel_choi_inout(din, dout, mineig=1e-15):
    return cp.Constant(
        random_channel_choi(dout=dout, din=din, mineig=mineig).reshape((dout, din, dout, din)).transpose(1, 0, 3, 2).reshape((din * dout, din * dout))
    )

random_unitary = scipy.stats.unitary_group.rvs

def unitary_choi(U):
    n = U.shape[0]
    return np.einsum('ab,cd->abdc', U, U.conj().T).reshape((n * n, n * n))

def unitary_choi_inout(U):
    n = U.shape[0]
    return np.einsum('ab,cd->bacd', U, U.conj().T).reshape((n * n, n * n))

def superchannel_project(W, d1, d2, d3, d4):
    W = (W + W.conj().T) / 2
    return W - (
        cp.kron(cp.partial_trace(W, [d1 * d2 * d3, d4], 1), np.eye(d4)) / d4 # Tr_4[W] \otimes I_4 / d4
        - cp.kron(cp.partial_trace(W, [d1 * d2, d3 * d4], 1), np.eye(d3 * d4)) / (d3 * d4) # Tr_34[W] \otimes I_34 / d34
        + cp.kron(cp.partial_trace(W, [d1, d2 * d3 * d4], 1), np.eye(d2 * d3 * d4)) / (d2 * d3 * d4) # Tr_234[W] \otimes I_234 / d234
        - cp.trace(W) * np.eye(d1 * d2 * d3 * d4) / (d1 * d2 * d3 * d4)
    )
def np_superchannel_project(W, d1, d2, d3, d4):
    W = (W + W.conj().T) / 2
    return W - (
        np.kron(np_partial_trace_single(W, [d1 * d2 * d3, d4], 1), np.eye(d4)) / d4 # Tr_4[W] \otimes I_4 / d4
        - np.kron(np_partial_trace_single(W, [d1 * d2, d3 * d4], 1), np.eye(d3 * d4)) / (d3 * d4) # Tr_34[W] \otimes I_34 / d34
        + np.kron(np_partial_trace_single(W, [d1, d2 * d3 * d4], 1), np.eye(d2 * d3 * d4)) / (d2 * d3 * d4) # Tr_234[W] \otimes I_234 / d234
        - np.trace(W) * np.eye(d1 * d2 * d3 * d4) / (d1 * d2 * d3 * d4)
    )
def np_superchannel_criteria(W, d1, d2, d3, d4):
    # asserted W = (W + W.conj().T) / 2
    return (
        np_partial_trace_single(W, [d1 * d2 * d3, d4], 1) # Tr_4[W]
        - np.kron(np_partial_trace_single(W, [d1 * d2, d3 * d4], 1), np.eye(d3)) / (d3) # Tr_34[W] \otimes I_3 / d3
        + np.kron(np_partial_trace_single(W, [d1, d2 * d3 * d4], 1), np.eye(d2 * d3)) / (d2 * d3) # Tr_234[W] \otimes I_23 / d23
        - np.trace(W) * np.eye(d1 * d2 * d3) / (d1 * d2 * d3)
    )

def random_superchannel(d1, d2, d3, d4, mineig=1e-15):
    W = rand_hermitian(d1 * d2 * d3 * d4)
    W = np_superchannel_project(W, d1, d2, d3, d4)
    W = shift_mineigval(W, mineig=mineig)
    W = W / np.trace(W) * d1 * d3
    return W

def choi_to_transition_inout(C, din, dout):
    return C.reshape((din, dout, din, dout)).transpose(1, 3, 0, 2).reshape((dout * dout, din * din))

def transition_to_choi_inout(C, din, dout):
    return C.reshape((dout, dout, din, din)).transpose(2, 0, 3, 1).reshape((din * dout, din * dout))

def choi_pinv_map_inout(C, din, dout): # din dout are the dimensions of C, the returned inverse has input dim dout and output dim din
    return transition_to_choi_inout(scipy.linalg.pinv(choi_to_transition_inout(C, din, dout)), dout, din)

def kraus_to_choi_inout(K):
    '''
      ┌───┐      ┌───┐
  ─x─a│ K │─┐ ┌──│K^+│c─x─
   │  └───┘ │ │  └───┘  │
  ─x─b──────┘ └───────d─x─
    '''
    return np.einsum('ab,dc->badc', K, K.conj().T).reshape((K.size, K.size))
    

def hermitian_basis(dim):
    basis = []
    for i in range(dim):
        tmp = np.zeros((dim, dim))
        tmp[i, i] = 1
        basis.append(tmp)
        for j in range(i+1, dim):
            tmp = np.zeros((dim, dim), dtype=complex)
            tmp[i, j] = 1
            basis.append((tmp + tmp.conj().T) / math.sqrt(2))
            tmp[i, j] = 1j
            basis.append((tmp + tmp.conj().T) / math.sqrt(2))
    return basis

def check_linearity(f, din, rounds = 10, tolerance=1e-4):
    basis = hermitian_basis(din)
    f_basis = [f(b) for b in basis]
    for i in range(rounds):
        a = rand_hermitian(din)
        true_fa = f(a)
        coe = [np.trace(b.conj().T @ a) for b in basis]
        linear_fa = sum(f_basis[i] * coe[i] for i in range(len(basis)))
        if np.linalg.norm(true_fa - linear_fa) > tolerance:
            return 'Not linear', None
    return 'Linear', sum(np.kron(f_basis[i], basis[i].T) for i in range(len(basis))) # return the choi matrix (out, in)

def independent_indices(F, abs_tol=1e-9):
    vf = None
    indep_ind = []
    for i in range(len(F)):
        g = np.concatenate((F[i].real, F[i].imag), axis=0)
        if not numpy.isclose(numpy.linalg.norm(g), 0, atol=abs_tol):
            if vf is None:
                vf = g.reshape((g.size, 1))
                indep_ind.append(i)
            else:
                va = g.reshape((g.size, 1))
                x, residuals, rank, s = numpy.linalg.lstsq(vf, va, rcond=None)
                if not numpy.isclose(numpy.linalg.norm(va - vf @ x), 0, atol=abs_tol):
                    vf = np.concatenate((vf, va), axis=1)
                    indep_ind.append(i)
    return indep_ind

def independent_subset(F, abs_tol=1e-9):
    subset = independent_indices(F, abs_tol)
    return [F[i] for i in subset]

def partial_trace(obj, dims, traced_axes): # defining partial_trace to support multiple traced axes
    res = obj
    dims = dims.copy()
    if isinstance(traced_axes, int):
        traced_axes = [traced_axes]
    traced_axes = sorted(traced_axes, reverse=True) ## sort in descending order
    for a in traced_axes:
        #print('partial_trace {0}, {1}'.format(dims, a))
        res = cp.partial_trace(res, dims, a)
        dims.pop(a)
    return res

def np_partial_trace(obj, dims, traced_axes): # defining partial_trace to support multiple traced axes
    res = obj
    dims = dims.copy()
    if isinstance(traced_axes, int):
        traced_axes = [traced_axes]
    traced_axes = sorted(traced_axes, reverse=True) ## sort in descending order
    for a in traced_axes:
        #print('partial_trace {0}, {1}'.format(dims, a))
        res = np_partial_trace_single(res, dims, a)
        dims.pop(a)
    return res

def np_partial_trace_single(obj, dims, axis):
    remaining_dims = math.prod(dims) // dims[axis]
    return obj.reshape(dims + dims).trace(axis1=axis, axis2=axis + len(dims)).reshape((remaining_dims, remaining_dims))

def np_partial_transpose(A, sys, index):
    return cp.partial_transpose(A, sys, index).value

def swap_systems(obj, dims, a1, a2):
    if a1 == a2:
        return obj
    dleft = math.prod(dims[0:a1])
    d1 = dims[a1]
    dmiddle = math.prod(dims[a1+1:a2])
    d2 = dims[a2]
    dright = math.prod(dims[a2+1:])
    size = dleft * d1 * dmiddle * d2 * dright
    assert size == obj.shape[0] and size == obj.shape[1]
    perm_mat = np.zeros((size, size))
    perm = [0] * size
    for ileft, i1, imiddle, i2, iright in itertools.product(range(dleft), range(d1), range(dmiddle), range(d2), range(dright)):
        from_index = (((ileft * d1 + i1) * dmiddle + imiddle) * d2 + i2) * dright + iright
        to_index = (((ileft * d2 + i2) * dmiddle + imiddle) * d1 + i1) * dright + iright
        perm[to_index] = from_index
        perm_mat[to_index, from_index] = 1
    
    return cp.Constant(perm_mat) @ obj @ cp.Constant(perm_mat.T)

def np_swap_systems(obj, dims, a1, a2):
    return swap_systems(obj, dims, a1, a2).value

def middle_kron(obj1, dleft, obj2, dright=None): # kron with obj2 inserted in between
    if dright is None:
        dright = round(obj1.shape[0] / dleft)
    assert obj1.shape[0] == dleft * dright
    return swap_systems(cp.kron(obj1, obj2), [dleft, dright, obj2.shape[0]], 1, 2)

def np_middle_kron(obj1, dleft, obj2, dright=None): # kron with obj2 inserted in between
    if dright is None:
        dright = round(obj1.shape[0] / dleft)
    assert obj1.shape[0] == dleft * dright
    return np_swap_systems(np.kron(obj1, obj2), [dleft, dright, obj2.shape[0]], 1, 2)

def swap_gate(d):
    return np.eye(d * d).reshape((d, d, d, d)).transpose(0, 1, 3, 2).reshape((d * d, d * d))

def super_from_LR(L, R, dw, dx, dy, dz): # L: dw -> dx*dm,  R: dy*dm -> dz, result: dw,dx,dy,dz
    dm = L.shape[0] // dw // dx
    assert dm == R.shape[0] // dy // dz
    return np_partial_trace(np.kron(np_partial_transpose(L, [dw, dx, dm], 2), np.eye(dy * dz)) @ np.kron(np.eye(dw * dx), np_swap_systems(R, [dy, dm, dz], 0, 1)),
                            [dw, dx, dm, dy, dz], 2)
def compose_choi(L, R, dw, dm, dz):
    return super_from_LR(L, R, dw, 1, 1, dz)

def np_apply_supermap(supermap, channel, d1, d2, d3, d4):
    return np_partial_trace(supermap @ kron(np.eye(d1), channel.T, np.eye(d4)), [d1, d2 * d3, d4], 1)

def apply_supermap(supermap, channel, d1, d2, d3, d4): #order of channel choi is in - out
    if is_sympy(supermap, channel):
        return sp_partial_trace(supermap @ kron(np.eye(d1), channel.T, np.eye(d4)), [d1, d2 * d3, d4], 1)
    return np_partial_trace(supermap @ kron(np.eye(d1), channel.T, np.eye(d4)), [d1, d2 * d3, d4], 1)

def adjoint_map_choi(A, din, dout):
    return np_swap_systems(A.T, [din, dout], 0, 1)

def kron_choi(A, dinA, B, dinB):
    doutA = A.shape[0] // dinA
    doutB = B.shape[0] // dinB
    return swap_systems(np.kron(A, B), [dinA, doutA, dinB, doutB], 1, 2)

def depolarizing_channel(d, p=1):
    return kraus_to_choi_inout(np.eye(d)) * (1-p) + np.eye(d * d) / d * p

def dephasing_channel(d, p=1):
    return kraus_to_choi_inout(np.eye(d)) * (1-p) + kraus_to_choi_inout(np.diag([cmath.rect(1, k / d * math.pi * 2) for k in range(d)])) * p

# classical controlled Petz map
def ccPetz(N, S, dx, dy, dz): # choi operators ordered as in-out
    xbasis = [np.array([0] * i + [1] + [0] * (dx - i - 1)) for i in range(dx)]
    gamma_x = [np_partial_trace(N @ np.kron(np.outer(x.conj(), x).T, np.eye(dy)), [dx, dy], 0) for x in xbasis]
    S_gamma_x = [np_partial_trace(S @ np.kron(g.T, np.eye(dz)), [dy, dz], 0) for g in gamma_x]
    dm = dx
    dw = dx
    L = kraus_to_choi_inout(sum(np.outer(np.kron(x, x), x.conj()) for x in xbasis))
    R1 = sum(kraus_to_choi_inout(
        np.kron(powmh(S_gamma_x[x], -1/2), np.outer(xbasis[x].conj(), xbasis[x]))
    ) for x in range(len(xbasis))
            )
    S_dag_id = kron_choi(adjoint_map_choi(S, dy, dz), dz, unitary_choi_inout(np.eye(dx)), dx) #dz dy dx dx
    R3 = sum(kraus_to_choi_inout(
        np.kron(powmh(gamma_x[x], 1/2), xbasis[x].conj())
    ) for x in range(len(xbasis))
            )
    R = compose_choi(compose_choi(R1, S_dag_id, dz * dx, dz * dx, dy * dx), R3, dz * dx, dy * dx, dy)
    return super_from_LR(L, R, dw, dx, dy, dz)


def rearrange_systems(a, dims, order):
    return a.reshape(dims * 2).transpose(list(order) + [x + len(dims) for x in order]).reshape((math.prod(dims), math.prod(dims)))
def rearrange_systems_lr(a, ldims, lorder, rdims, rorder):
    return a.reshape(ldims + rdims).transpose(list(lorder) + [x + len(ldims) for x in rorder]).reshape((math.prod(ldims), math.prod(rdims)))

# for scipy optimization
def flatten_hermitian(A):
    n = A.shape[0]
    return np.concatenate((A[np.triu_indices(n)].real, A[np.triu_indices(n, 1)].imag))[1:]

def flatten_complex(A):
    v = A.flatten()
    return np.concatenate((v.real, v.imag))
    
def unflatten_complex(v2, r, c):
    n2 = v2.size
    n = n2 // 2
    assert n == r * c
    v = v2[0 : n] + 1j * v2[n : n2]
    return v.reshape((r, c))

# parameterization of SU(2)
def unflatten_su2(alpha, beta, theta):
    eia = cmath.exp(1j * alpha)
    eib = cmath.exp(1j * beta)
    ct = math.cos(theta)
    st = math.sin(theta)
    return np.array([[eia * ct, -eib.conjugate() * st],
                     [eib * st, eia.conjugate() * ct]], dtype=complex)

def flatten_su2(V):
    ct, alpha = cmath.polar(V[0, 0])
    st, beta = cmath.polar(V[1, 0])
    theta = math.atan2(st, ct)
    return alpha, beta, theta

# parameterization of SU(d)
def unflatten_sud(v):
    d_choose_2 = len(v) // 3
    d = round((1 + math.sqrt(8 * d_choose_2 + 1)) / 2)
    assert len(v) == 3 * d * (d-1) // 2, 'invalid length of vector, must be 3 * (d choose 2)'
    k = 0
    res = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            Uk_2 = unflatten_su2(v[3 * k], v[3 * k + 1], v[3 * k + 2])
            Uk_d = np.eye(d, dtype=complex)
            Uk_d[np.ix_([i,j],[i,j])] = Uk_2
            res = res @ Uk_d
            k += 1
    return res

def flatten_sud(V, epsi=1e-7):
    res = []
    d = V.shape[0]
    assert V.shape[1] == d
    for i in range(d):
        for j in range(i + 1, d):
            a = V[i, i]
            b = V[j, i]
            r = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
            if r < epsi:
                Uk_2 = np.eye(2)
            else:
                Uk_2 = np.array([[a.conj(), b.conj()], [-b, a]]) / r
            res.extend(flatten_su2(Uk_2.conj().T))
            Uk_d = np.eye(d, dtype=complex)
            Uk_d[np.ix_([i,j],[i,j])] = Uk_2
            V = Uk_d @ V
    return res

def flatten_sud_size(d_u):
    return d_u * (d_u - 1) * 3 // 2

def flatten_ud(V, epsi=1e-7):
    d = np.linalg.det(V)
    V_adjust = V.copy()
    if d != 0:
        V_adjust[:,0] /= d / abs(d)
    return np.concatenate([cmath.phase(d), flatten_sud(V_adjust, epsi=epsi)], axis=None)
def unflatten_ud(x):
    p = x[0]
    V_adjust = unflatten_sud(x[1:])
    V_adjust[:,0] *= cmath.exp(1j * p)
    return V_adjust
def flatten_ud_size(d):
    return flatten_sud_size(d) + 1


def flatten_isometry_size(rows, cols):
    if rows == cols:
        return flatten_ud_size(rows)
    if rows < cols:
        rows, cols = cols, rows
    res = 0
    for i in range(cols):
        for j in range(i + 1, rows):
            res += 3
    return res
def flatten_isometry(V, epsi=1e-7): #flatten an isometry or transpose of an isometry
    rows, cols = V.shape
    if rows == cols:
        return flatten_ud(V, epsi=epsi)
    res = []
    if rows < cols:
        V = V.T
        rows, cols = V.shape
    # now rows > cols
    for i in range(cols):
        for j in range(i + 1, rows):
            a = V[i, i]
            b = V[j, i]
            r = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
            if r < epsi:
                Uk_2 = np.eye(2)
            else:
                Uk_2 = np.array([[a.conj(), b.conj()], [-b, a]]) / r
            res.extend(flatten_su2(Uk_2.conj().T))
            Uk_d = np.eye(rows, dtype=complex)
            Uk_d[np.ix_([i,j],[i,j])] = Uk_2
            V = Uk_d @ V
    return res
def unflatten_isometry(v, rows, cols):
    if rows == cols:
        return unflatten_ud(v)
    if rows < cols:
        return unflatten_isometry(v, cols, rows).T
    # now rows > cols
    k = 0
    res = np.eye(rows)
    for i in range(cols):
        for j in range(i + 1, rows):
            Uk_2 = unflatten_su2(v[3 * k], v[3 * k + 1], v[3 * k + 2])
            Uk_d = np.eye(rows, dtype=complex)
            Uk_d[np.ix_([i,j],[i,j])] = Uk_2
            res = res @ Uk_d
            k += 1
    return res[0:rows, 0:cols]
    

def unflatten_so2(theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    return np.array([[ct, -st], [st, ct]])

def flatten_so2(V, epsi=1e-7):
    assert np.linalg.norm(V.imag) < epsi, 'Input matrix is not a real matrix'
    return math.atan2(V[1, 0].real, V[1, 1].real)

def unflatten_sod(v):
    d_choose_2 = len(v)
    d = round((1 + math.sqrt(8 * d_choose_2 + 1)) / 2)
    assert len(v) == d * (d-1) // 2, 'invalid length of vector, must be (d choose 2)'
    k = 0
    res = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            Uk_2 = unflatten_so2(v[k])
            Uk_d = np.eye(d, dtype=complex)
            Uk_d[np.ix_([i,j],[i,j])] = Uk_2
            res = res @ Uk_d
            k += 1
    return res

def flatten_sod(V, epsi=1e-7):
    assert np.linalg.norm(V.imag) < epsi, 'Input matrix is not a real matrix'
    V = V.real
    res = []
    d = V.shape[0]
    assert V.shape[1] == d
    for i in range(d):
        for j in range(i + 1, d):
            a = V[i, i]
            b = V[j, i]
            r = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
            if r < epsi:
                res.append(0)
                continue
            Uk_2 = np.array([[a, b], [-b, a]]) / r
            res.append(flatten_so2(Uk_2.T))
            Uk_d = np.eye(d, dtype=complex)
            Uk_d[np.ix_([i,j],[i,j])] = Uk_2
            V = Uk_d @ V
    return res

def flatten_sod_size(d_u):
    return d_u * (d_u - 1) // 2

def remove_small_elements(V, epsi=1e-4):
    quantized_V = V.copy()
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            if abs(V[i,j]) < epsi:
                quantized_V[i,j] = 0
    return quantized_V

def swap_basis(d, a, b):
    s = np.eye(d)
    s[a,b] = 1
    s[b,a] = 1
    s[a,a] = 0
    s[b,b] = 0
    return s

import sympy as sp
import sympy.physics
import sympy.physics.quantum

def sp_partial_trace(obj, dims, traced_axes): # defining partial_trace to support multiple traced axes
    res = obj
    dims = dims.copy()
    if isinstance(traced_axes, int):
        traced_axes = [traced_axes]
    traced_axes = sorted(traced_axes, reverse=True) ## sort in descending order
    for a in traced_axes:
        #print('partial_trace {0}, {1}'.format(dims, a))
        res = sp_partial_trace_single(res, dims, a)
        dims.pop(a)
    return res

def sp_partial_trace_single(obj, dims, axis):
    res_dims = math.prod(dims) // dims[axis]
    dl = math.prod(dims[0:axis])
    dm = dims[axis]
    dr = res_dims // dl
    assert math.prod(dims) == dl * dm * dr
    # i, j, k  ->  (i * dm + j) * dr + k
    # i1, i2, j, k1, k2:  result[i1 * dr + k1, i2 * dr + k2] = sum_j obj[(i1 * dm + j) * dr + k1, (i2 * dm + j) * dr + k2]
    result = sp.zeros(res_dims, res_dims)
    for i1, k1, i2, k2 in itertools.product(range(dl), range(dr), range(dl), range(dr)):
        for j in range(dm):
            result[i1 * dr + k1, i2 * dr + k2] += obj[(i1 * dm + j) * dr + k1, (i2 * dm + j) * dr + k2]
    return result

def sp_kron_single(a, b):
    br, bc = b.shape
    ar, ac = a.shape
    result = sp.zeros(ar * br, ac * bc)
    for i, j, k, l in itertools.product(range(ar), range(br), range(ac), range(bc)):
        result[i * br + j, k * bc + l] = a[i, k] * b[j, l]
    return result

def sp_kron(*args):
    r = sp.eye(1)
    for a in args:
        r = sp_kron_single(r, a)
    return r

def sp_superchannel_project(W, d1, d2, d3, d4):
    #W = (W + W.conj().T) / 2
    return W - (
        sp_kron(sp_partial_trace_single(W, [d1 * d2 * d3, d4], 1), sp.eye(d4)) / d4 # Tr_4[W] \otimes I_4 / d4
        - sp_kron(sp_partial_trace_single(W, [d1 * d2, d3 * d4], 1), sp.eye(d3 * d4)) / (d3 * d4) # Tr_34[W] \otimes I_34 / d34
        + sp_kron(sp_partial_trace_single(W, [d1, d2 * d3 * d4], 1), sp.eye(d2 * d3 * d4)) / (d2 * d3 * d4) # Tr_234[W] \otimes I_234 / d234
        - sp.trace(W) * sp.eye(d1 * d2 * d3 * d4) / (d1 * d2 * d3 * d4)
    )

def sp_superchannel_criteria(W, d1, d2, d3, d4):
    Tr_4_W = sp_partial_trace_single(W, [d1 * d2 * d3, d4], 1)
    return (
        Tr_4_W
        - sp_kron(sp_partial_trace_single(Tr_4_W, [d1 * d2, d3], 1), sp.eye(d3)) / (d3) # Tr_34[W] \otimes I_3 / d3
        + sp_kron(sp_partial_trace_single(Tr_4_W, [d1, d2 * d3], 1), sp.eye(d2 * d3)) / (d2 * d3) # Tr_234[W] \otimes I_23 / d23
        - sp.trace(Tr_4_W) * sp.eye(d1 * d2 * d3) / (d1 * d2 * d3)
    )

def commutator(A, B):
    return A @ B - B @ A

#differentials of matrices
def diff_sqrt(X, dX): #return d(sqrt(X))
    # by solving dX = sqrt(X) @ d(sqrt(X)) + d(sqrt(X)) @ sqrt(X)
    # X must be positive hermitian
    sqrtX = sqrtmh(X)
    d = X.shape[0]
    if is_sympy(X, dX): #symbolic
        A = sp_kron(sqrtX, sp.eye(d)) + sp_kron(sp.eye(d), sqrtX.T)
        return (powmh(A, -1) @ dX.reshape(len(dX), 1)).reshape(*X.shape)
    A = kron(sqrtX, np.eye(d)) + kron(np.eye(d), sqrtX.T)
    return numpy.linalg.solve(A, dX.flatten()).reshape(X.shape)
    
def diff_invh(X, dX): #only applicable to Hermitian matrices!
    #invX = scipy.linalg.inv(X)
    invX = powmh(X, -1)
    return -invX @ dX @ invX

def diff_inv_sqrt(X, dX):
    return diff_invh(sqrtmh(X), diff_sqrt(X, dX))

def starpt_outin(choi, rho):
    din = rho.shape[0]
    dout = round(choi.shape[0] / din)
    s_rho = np.kron(np.eye(dout), sqrtmh(rho).T)
    return s_rho @ choi @ s_rho

def starpt_inout(choi, rho):
    din = rho.shape[0]
    dout = round(choi.shape[0] / din)
    s_rho = np.kron(sqrtmh(rho).T, np.eye(dout))
    return s_rho @ choi @ s_rho
    
def Petz(choi, gamma): # in (original out), out (original in)
    din = gamma.shape[0]
    dout = round(choi.shape[0] / din)
    C_star_gamma = starpt_outin(choi, gamma)
    C_gamma = np_partial_trace(C_star_gamma, [dout, din], 1)
    C_gamma_minushalf = powmh(C_gamma, -1/2)
    return (np.kron(C_gamma_minushalf, np.eye(din)) @ C_star_gamma @ np.kron(C_gamma_minushalf, np.eye(din))).T

def var_Petz(choi, gamma, tau): # out, in
    din = gamma.shape[0]
    dout = round(choi.shape[0] / din)
    C_star_gamma = starpt_outin(choi, gamma)
    C_gamma = np_partial_trace(C_star_gamma, [dout, din], 1)
    sqrt_tau = sqrtmh(tau)
    D = sqrt_tau @ powmh(sqrt_tau @ C_gamma @ sqrt_tau, -1/2)
    return (np.kron(D.conj().T, np.eye(din)) @ C_star_gamma @ np.kron(D, np.eye(din))).T