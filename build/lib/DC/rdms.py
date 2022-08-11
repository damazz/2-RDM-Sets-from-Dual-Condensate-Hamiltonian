import numpy as np
import numba 

@numba.jit(nopython = True , nogil=True, cache = True)
def comb1(n,k):
    if n<k:
        return 0
    elif n==k:
        return 1
    c = n-k
    pa = np.int64(1)
    pb = np.int64(1)
    if k > c:
        for i in range(k+1,n+1):
            pa *= i
        for i in range(1,c+1):
            pb *= i
    else:
        for i in range(c+1,n+1):
            pa *= i
        for i in range(1,k+1):
            pb *= i
    return pa // pb


@numba.jit(nopython = True, nogil = True, cache = True)
def indexerg(state,a,b,c,d,N):
    out = 0
    for i in range(state.shape[0]):
        if state.shape[0]-i == N:
            break
        elif i == a:
            out += comb1(state.shape[0]-i-1,N)
            N -= 1
        elif i == b:
            pass
        elif i == c:
            out += comb1(state.shape[0]-i-1,N)
            N -= 1
        elif i == d:
            pass
        else:
            out += comb1(state.shape[0]-i-1,N)*state[i]
            N -= state[i]
    return out

@numba.jit(nopython = True, nogil = True, cache = True)
def indexerd(state,a,b,c,d,N):
    out = 0
    for i in range(state.shape[0]):
        if state.shape[0]-i == N:
            break
        elif i == a:
            out += comb1(state.shape[0]-i-1,N)
            N -= 1
        elif i == b:
            out += comb1(state.shape[0]-i-1,N)
            N -= 1
        elif i == c:
            pass
        elif i == d:
            pass
        else:
            out += comb1(state.shape[0]-i-1,N)*state[i]
            N -= state[i]
    return out

@numba.jit(nopython=True,nogil=True, cache = True)
def find(indx, ind):
    for i in range(ind.shape[0]):
        if indx == ind[i]:
            return i
    return -1


@numba.jit(nopython=True,nogil=True,parallel = True, cache = True)
def lazygmatcal(N,states,v, ind):
    N = np.int64(2*N)
    num = np.sum(states[0])
    indx = np.int64(0)
    g2mat = np.zeros((v.shape[0],N**2,N**2), dtype = np.float64)
    for cat in numba.prange(N**2):
        a = cat%N
        b = cat//N
        for c in range(N):
            for d in range(N):
                for i in range(states.shape[0]):
                    if states[i,d]==1:
                        if states[i,c]==0 or c==d:
                            if (states[i,b]==1 and b!=d) or b==c:
                                if (states[i,a]==0 and a!=c) or a==b or (a==d and a!=c) or (a==d and a==c and a==b):
                                    indx = indexerg(states[i],a,b,c,d,num)
                                    indx = find(indx, ind)
                                    if indx < 0:
                                        break
                                    for bob in range(v.shape[0]):
                                        g2mat[bob,a*N+b,c+d*N] += (v[bob,i]*v[bob,indx])
#                                    if a ==0 and b==0 and c==0 and d ==0:
#                                        print(states[i], states[indx], i , indx)
#                                        print(g2mat[0,0,0])
#                                        print(v[0])
    return g2mat 


@numba.jit(nopython=True,nogil=True, parallel = True, cache = True)
def lazyrdmcal(N,states,v,ind):
    N = np.int64(2*N)
    num = np.sum(states[0])
    indx = np.int64(0)
    d2mat = np.zeros((v.shape[0],N**2,N**2), dtype = np.float64)
    for cat in numba.prange(N**2):
        a = cat%N
        b = cat//N
        for c in range(N):
            for d in range(N):
                if a!=b and c!=d:
                    for i in range(states.shape[0]):
                        if states[i,d]==1:
                            if states[i,c]==1:
                                if (states[i,b]==0 and states[i,a]==0) or (states[i,b]==0 and (c==a or d==a)) or (states[i,a]==0 and (c==b or d==b)) or (c==a and d==b) or (c==b and d==a):
                                    indx = indexerd(states[i],a,b,c,d,num)
                                    indx = find(indx, ind)
                                    if indx < 0:
                                        break
                                    for bob in range(v.shape[0]):
                                        d2mat[bob,a*N+b,c*N+d] += (v[bob,i]*v[bob,indx])
    return d2mat 

