import numpy as np
import numba


@numba.jit(nopython=True, nogil=True, cache = True)
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
    return pa // pb


@numba.jit(nopython=True, nogil=True, cache = True)
def hx(num, ln):
    out = np.zeros(ln, dtype=np.int8)
    for i in range(ln):
        out[i] = num // (6 ** (ln - i - 1))
        num -= out[i] * (6 ** (ln - i - 1))
    return out


@numba.jit(nopython=True, nogil=True, cache = True)
def index(state, N):
    out = np.int64(0)
    for i in range(state.shape[0]):
        if state.shape[0] - i == N:
            break
        out += comb1(state.shape[0] - i - 1, N) * state[i]
        N -= state[i]
    if out < 0:
        print("error")
    return out


@numba.jit(nopython=True, nogil=True, cache = True)
def prune(N):
    prods = np.zeros((6 ** (N // 2), N // 2), dtype=np.int8)
    for i in numba.prange(6 ** (N // 2)):
        prods[i] = hx(i, N // 2)
    for i in numba.prange(6 ** (N // 2)):
        cnt = 0
        for j in prods[i, :]:
            if j == 0:
                cnt += 0
            elif j < 5 and j != 0:
                cnt += 2
            elif j == 5:
                cnt += 4
        if cnt != N:
            prods[i, :] = -1
    bob = np.argwhere(prods[:, 0] != -1)
    prods1 = np.zeros((bob.shape[0], N // 2), dtype=np.int8)
    for i in numba.prange(bob.shape[0]):
        prods1[i] = prods[bob[i]]
    states = np.zeros((bob.shape[0], 2 * N), dtype=np.int64)
    for i in numba.prange(bob.shape[0]):
        for k, j in enumerate(prods1[i, :]):
            if j == 1:
                states[i, 2 * k] += 1
                states[i, 2 * k + 1] += 1
            elif j == 2:
                states[i, 2 * k + N] += 1
                states[i, 2 * k + N + 1] += 1
            elif j == 3:
                states[i, 2 * k] += 1
                states[i, 2 * k + N + 1] += 1
            elif j == 4:
                states[i, 2 * k + N] += 1
                states[i, 2 * k + 1] += 1
            elif j == 5:
                states[i, 2 * k] += 1
                states[i, 2 * k + 1] += 1
                states[i, 2 * k + N] += 1
                states[i, 2 * k + N + 1] += 1
    out = np.zeros(bob.shape[0], dtype=np.int64)
    for i in numba.prange(bob.shape[0]):
        out[i] = index(states[i], N)
    return np.sort(out)
