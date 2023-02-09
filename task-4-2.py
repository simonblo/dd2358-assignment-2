import array
import numpy
import pytest
import random
import timeit

def test_dgemm_array():
    N = 128
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]
    C = [[random.random() for _ in range(N)] for _ in range(N)]
    X = array.array("d", (v for r in A for v in r))
    Y = array.array("d", (v for r in B for v in r))
    Z = array.array("d", (v for r in C for v in r))
    dgemm_lists(A, B, C, N)
    dgemm_array(X, Y, Z, N)
    assert numpy.allclose(numpy.array(C).flatten(), Z)

def test_dgemm_numpy():
    N = 128
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]
    C = [[random.random() for _ in range(N)] for _ in range(N)]
    X = numpy.array(A, dtype="float64")
    Y = numpy.array(B, dtype="float64")
    Z = numpy.array(C, dtype="float64")
    dgemm_lists(A, B, C, N)
    dgemm_numpy(X, Y, Z, N)
    assert numpy.allclose(C, Z)

def dgemm_lists(A, B, C, N):
    t0 = timeit.default_timer()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    t1 = timeit.default_timer()
    return t1 - t0

def dgemm_array(A, B, C, N):
    t0 = timeit.default_timer()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[N*i+j] = C[N*i+j] + A[N*i+k] * B[N*k+j]
    t1 = timeit.default_timer()
    return t1 - t0

def dgemm_numpy(A, B, C, N):
    t0 = timeit.default_timer()
    numpy.add(C, numpy.matmul(A, B), C)
    t1 = timeit.default_timer()
    return t1 - t0
