import array
import numpy
import random
import tabulate
import timeit

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

if __name__ == "__main__":
    table   = []
    headers = ["", "dgemm_lists", "dgemm_array", "dgemm_numpy"]
    for n in [4, 8, 16, 32, 64, 128, 256]:
        A = [[random.random() for _ in range(n)] for _ in range(n)]
        B = [[random.random() for _ in range(n)] for _ in range(n)]
        C = [[random.random() for _ in range(n)] for _ in range(n)]
        U = array.array("d", (v for r in A for v in r))
        V = array.array("d", (v for r in B for v in r))
        W = array.array("d", (v for r in C for v in r))
        X = numpy.array(A, dtype="float64")
        Y = numpy.array(B, dtype="float64")
        Z = numpy.array(C, dtype="float64")
        t = [[], [], []]
        for _ in range(10):
            t[0].append(dgemm_lists(A, B, C, n))
            t[1].append(dgemm_array(U, V, W, n))
            t[2].append(dgemm_numpy(X, Y, Z, n))
        row = [n]
        row.append("{:5.6f} ± {:5.6f}".format(numpy.mean(t[0]), numpy.std(t[0])))
        row.append("{:5.6f} ± {:5.6f}".format(numpy.mean(t[1]), numpy.std(t[1])))
        row.append("{:5.6f} ± {:5.6f}".format(numpy.mean(t[2]), numpy.std(t[2])))
        table.append(row)
    print("\n" + tabulate.tabulate(table, headers) + "\n")
