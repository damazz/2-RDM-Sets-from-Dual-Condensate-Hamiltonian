import numpy as np
import sparse
import numba
from .prune import prune


@numba.jit(nopython=True, nogil=True)
def comb1(n, k):
    if n < k:
        return 0
    elif n == k:
        return 1
    c = n - k
    pa = np.int64(1)
    pb = np.int64(1)
    if k > c:
        for i in range(k + 1, n + 1):
            pa *= i
        for i in range(1, c + 1):
            pb *= i
    else:
        for i in range(c + 1, n + 1):
            pa *= i
        for i in range(1, k + 1):
            pb *= i
    #    return np.int64(pa/pb)
    return pa // pb


@numba.jit(nopython=True, nogil=True)
def antiindex(indx, N, num):
    state = np.zeros(2 * N, dtype=np.int64)
    cnt = 0
    for i in range(state.shape[0]):
        if indx == 0 and num != 0:
            for j in range(num):
                state[-j - 1] += 1
            break
        c = comb1(state.shape[0] - i - 1, num)
        if c <= indx and c != 0:
            state[i] += 1
            num -= 1
            indx -= c
    return state


@numba.jit(nopython=True, nogil=True)
def index(state):
    N = np.sum(state)
    out = 0
    for i in range(state.shape[0]):
        if state.shape[0] - i == N:
            break
        out += comb1(state.shape[0] - i - 1, N) * state[i]
        N -= state[i]
    if out < 0:
        print("error")
    return out


@numba.jit(nopython=True, nogil=True, parallel=True)
def prep(N):
    size = comb1(2 * N, N)
    out = np.zeros((size, 2 * N), dtype=np.int8)
    for i in numba.prange(size):
        out[i, :] = antiindex(i, N, N)
    return out


@numba.jit(nopython=True, nogil=True)
def indexer(state, a, b, c, d, N):
    out = np.int64(0)
    for i in range(state.shape[0]):
        if state.shape[0] - i == N:
            break
        elif i == a:
            out += comb1(state.shape[0] - i - 1, N)
            N -= 1
        elif i == b:
            out += comb1(state.shape[0] - i - 1, N)
            N -= 1
        elif i == c:
            pass
        elif i == d:
            pass
        else:
            out += comb1(state.shape[0] - i - 1, N) * state[i]
            N -= state[i]
    return out


@numba.jit(nopython=True, nogil=True)
def cntter(st):
    ncnt = np.int64(0)
    pcnt = np.int64(0)
    for i in range(st.shape[0]):
        if st[i] < 0:
            ncnt += 1
        elif st[i] > 0:
            pcnt += 1
    return ncnt, pcnt


@numba.jit(nopython=True, nogil=True, parallel=True)
def lamb(states, N):
    degen = states[:, :N] - states[:, N:]
    for i in range(degen.shape[0]):
        ncnt, pcnt = cntter(degen[i])
        degen[i] = comb1(ncnt, 2) * 2 + comb1(pcnt, 2) * 2
    degen = degen[:, 0]
    degen2 = np.zeros(degen.shape, dtype=np.int64)
    for i in range(1, degen.shape[0]):
        degen2[i] = degen2[i - 1] + degen[i - 1]
    out = np.zeros((2, np.sum(degen)), dtype=np.int64)
    for i in numba.prange(states.shape[0]):
        dcnt = 0
        for a in range(N):
            for b in range(N):
                if a != b:
                    if states[i, a] == 1 and states[i, b] == 1:
                        if states[i, a + N] == 0 and states[i, b + N] == 0:
                            indx = indexer(states[i], a + N, b + N, a, b, N)
                            out[0, degen2[i] + dcnt] = i
                            out[1, degen2[i] + dcnt] = indx
                            dcnt += 1
                    elif states[i, a] == 0 and states[i, b] == 0:
                        if states[i, a + N] == 1 and states[i, b + N] == 1:
                            indx = indexer(states[i], a, b, a + N, b + N, N)
                            out[0, degen2[i] + dcnt] = i
                            out[1, degen2[i] + dcnt] = indx
                            dcnt += 1
    return out


@numba.jit(nopython=True, nogil=True)
def gamcnt(st, N):
    tcnt = np.int64(0)
    ucnt = np.int64(0)
    dcnt = np.int64(0)
    for i in range(N):
        if st[i] == 1 and st[i + N] == 1:
            tcnt += 1
        elif st[i] == 1 and st[i + N] == 0:
            ucnt += 1
        elif st[i] == 0 and st[i + N] == 1:
            dcnt += 1
    return np.int64(ucnt * dcnt + tcnt)


@numba.jit(nopython=True, nogil=True, parallel=True)
def gamma(states, N):
    degen = np.zeros(states.shape[0], dtype=np.int64)
    for i in range(states.shape[0]):
        degen[i] = gamcnt(states[i], N)
    degen2 = np.zeros(degen.shape, dtype=np.int64)
    for i in range(1, degen.shape[0]):
        degen2[i] = degen2[i - 1] + degen[i - 1]
    out = np.zeros((2, np.sum(degen)), dtype=np.int64)
    for i in numba.prange(states.shape[0]):
        dcnt = 0
        for a in range(N):
            for b in range(N):
                if states[i, a] == 1 and states[i, b + N] == 1:
                    if (states[i, a + N] == 0 or a == b) and (
                        states[i, b] == 0 or a == b
                    ):
                        indx = indexer(states[i], a + N, b, a, b + N, N)
                        out[0, degen2[i] + dcnt] = i
                        out[1, degen2[i] + dcnt] = indx
                        dcnt += 1
    return out


@numba.jit(nopython=True, nogil=True)
def Gcnt(st, N):
    pcnt = np.int64(0)
    for i in range(N):
        if st[2 * i] == 1 and st[2 * i + 1] == 1:
            pcnt += 1
    return 2 * (comb1(pcnt, 2) + pcnt)


@numba.jit(nopython=True, nogil=True, parallel=True)
def G(states, N):
    degen = np.zeros(states.shape[0], dtype=np.int64)
    for i in range(states.shape[0]):
        degen[i] = Gcnt(states[i], N)
    degen2 = np.zeros(degen.shape, dtype=np.int64)
    for i in range(1, degen.shape[0]):
        degen2[i] = degen2[i - 1] + degen[i - 1]
    out = np.zeros((2, np.sum(degen)), dtype=np.int64)
    for i in numba.prange(states.shape[0]):
        dcnt = 0
        for a in range(N):
            for b in range(N):
                if states[i, 2 * a] == 1 and states[i, 2 * a + 1] == 1:
                    if (states[i, 2 * b] == 0 or a == b) and (
                        states[i, 2 * b + 1] == 0 or a == b
                    ):
                        indx = indexer(states[i], 2 * b, 2 * b + 1, 2 * a, 2 * a + 1, N)
                        out[0, degen2[i] + dcnt] = i
                        out[1, degen2[i] + dcnt] = indx
                        dcnt += 1
    return out


def Hinit(N):
    size = comb1(2 * N, N)
    epm = sparse.diagonalize(N * np.ones(size))
    print("epm done")
    states = prep(N)
    print("states done")
    lamm = sparse.COO(lamb(states, N), data=1.0, shape=epm.shape)
    print("lamm done")
    gamm = 2 * sparse.COO(gamma(states, N), data=1, shape=epm.shape)
    print("gamm done")
    gm = sparse.COO(G(states, N), data=1.0, shape=epm.shape)
    return epm, lamm, gamm, gm


# TODO add prune function
"""
N = 8
p = 100
from h import Hinit as oldh

epm, lamm, gamm, gm = oldh(N)
epm1, lamm1, gamm1, gm1 = Hinit(N)
print(epm.shape, epm1)
epm1 = epm1.todense()
lamm1 = lamm1.todense()
gamm1 = gamm1.todense()
gm1 = gm1.todense()
rand = np.random.uniform(-1,1,(4,p))
Ho = np.tensordot(rand[0],lamm,axes =0)
Hn = np.tensordot(rand[0],lamm1,axes =0)
for i in range(p):
    Ho[i]+=rand[1,i]*gamm+rand[2,i]*gm+rand[3,i]*epm
    Hn[i]+=rand[1,i]*gamm1+rand[2,i]*gm1+rand[3,i]*epm1
import numpy.linalg as LA
vo = LA.eigh(Ho)[1][:,:,0]
vn = LA.eigh(Hn)[1][:,:,0]
expo = np.zeros((3,p))
expn = np.zeros((3,p))
for i in range(p):
    expo[0,i] = np.matmul(vo[i],np.matmul(lamm,vo[i]))
    expn[0,i] = np.matmul(vn[i],np.matmul(lamm1,vn[i]))
    expo[1,i] = np.matmul(vo[i],np.matmul(gamm,vo[i]))
    expn[1,i] = np.matmul(vn[i],np.matmul(gamm1,vn[i]))
    expo[2,i] = np.matmul(vo[i],np.matmul(gm,vo[i]))
    expn[2,i] = np.matmul(vn[i],np.matmul(gm1,vn[i]))

#print(np.round(expo,5))
#print(np.round(expn,5))
print(np.argwhere(np.round(expo,5)-np.round(expn,5)))
"""
