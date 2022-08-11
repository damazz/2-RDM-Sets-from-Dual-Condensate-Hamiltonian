import numpy as np
from itertools import combinations, chain
from multiprocessing import Pool
from functools import partial
from .prune import prune


def init(N):
    comb = list(combinations(range(2 * N), N))
    combs = list(combinations(range(2 * N), N - 1))
    combss = list(combinations(range(2 * N), N - 2))
    return comb, combs, combss


def d2func(i, N, comb):
    #    print(i/len(comb))
    listy = []
    for a in range(2 * N):
        for b in range(2 * N):
            for c in range(2 * N):
                for d in range(2 * N):
                    try:
                        ket = list(comb[i])
                        ket.remove(a)
                        ket.remove(b)
                        ket.append(c)
                        ket.append(d)
                        ket.sort()
                        indx = comb.index(tuple(ket))
                        listy.extend([indx, i, d, c, b, a])
                    except ValueError:
                        pass
    return np.array(listy)


def d2gen(N, ind=None):
    comb1 = list(combinations(range(2 * N), N))
    if type(ind) == str:
        comb = comb1
    else:
        if np.all(ind == None):
            ind = prune(N)
        comb = []
        for i in ind:
            comb.append(comb1[int(i)])
    g = partial(d2func, N=N, comb=comb)
    p = Pool()
    inx = p.map(g, range(0, len(ind)))
    p.close()
    inx = np.array(list(chain(*inx))).reshape(-1)
    inx = inx.reshape(int(len(inx) / 6), 6)
    return inx


def g2func(i, N, comb):
    #    print(i/len(comb))
    listy = []
    for a in range(2 * N):
        for b in range(2 * N):
            for c in range(2 * N):
                for d in range(2 * N):
                    try:
                        ket = list(comb[i])
#                        if i == 2:
#                            print(ket)
                        ket.remove(a)
                        ket.append(b)
                        ket.sort()
#                        if i == 2:
#                            print(ket)
                        comb.index(tuple(ket))
                        ket.remove(c)
                        ket.append(d)
#                        if i == 2:
#                            print(ket)
                        ket.sort()
                        indx = comb.index(tuple(ket))
                        listy.extend([indx, i, d, c, b, a])
                    except ValueError:
#                        print('\n')
                        pass
                        
    return np.array(listy)


def g2gen(N, ind=None):
    comb1 = list(combinations(range(2 * N), N))
    if type(ind) == str:
        comb = comb1
    else:
        if np.all(ind == None):
            ind = prune(N)
        comb = []
        for i in ind:
            comb.append(comb1[int(i)])
#    print(comb)
    g = partial(g2func, N=N, comb=comb)
    p = Pool()
    inx = p.map(g, range(0, len(ind)))
    p.close()
    inx = np.array(list(chain(*inx))).reshape(-1)
    inx = inx.reshape(int(len(inx) / 6), 6)
    return inx
