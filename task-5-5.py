import math
import numpy
import timeit

def dft_numpy(xr, xi, Xr, Xi, N):
    t0 = timeit.default_timer()
    a  = 2 * math.pi / N
    v  = numpy.arange(float(N))
    Wr = numpy.outer(v, v)
    Wi = numpy.outer(v, v)
    numpy.multiply(a, Wr, Wr)
    numpy.multiply(a, Wi, Wi)
    numpy.cos(Wr, Wr)
    numpy.sin(Wi, Wi)
    numpy.add(Xr, numpy.matmul(Wr,  xr), Xr)
    numpy.add(Xr, numpy.matmul(Wi,  xi), Xr)
    numpy.add(Xi, numpy.matmul(Wi, -xr), Xi)
    numpy.add(Xi, numpy.matmul(Wr,  xi), Xi)
    t1 = timeit.default_timer()
    return t1 - t0

if __name__ == "__main__":
    N  = 16384
    xr = numpy.random.random_sample(N)
    xi = numpy.random.random_sample(N)
    Xr = numpy.full(N, 0.0)
    Xi = numpy.full(N, 0.0)
    print(f"dft_numpy: {N:d} elements, {dft_numpy(xr, xi, Xr, Xi, N):.6f} seconds")
