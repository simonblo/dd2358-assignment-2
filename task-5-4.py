import math
import matplotlib.pyplot as plt
import numpy
import pytest
import random
import timeit

def test_dft_list():
    xr = [1.0,  2.0,  0.0, -1.0]
    xi = [0.0, -1.0, -1.0,  2.0]
    Xr = [0.0,  0.0,  0.0,  0.0]
    Xi = [0.0,  0.0,  0.0,  0.0]
    dft_list(xr, xi, Xr, Xi, 4)
    assert numpy.allclose(Xr, [2.0, -2.0,  0.0, 4.0])
    assert numpy.allclose(Xi, [0.0, -2.0, -2.0, 4.0])

def test_dft_list_opt():
    N  = 128
    xr = [random.random() for _ in range(N)]
    xi = [random.random() for _ in range(N)]
    Xr = [0.0] * N
    Xi = [0.0] * N
    yr = xr.copy()
    yi = xi.copy()
    Yr = Xr.copy()
    Yi = Xi.copy()
    dft_list(xr, xi, Xr, Xi, N)
    dft_list_opt(yr, yi, Yr, Yi, N)
    assert numpy.allclose(Xr, Yr)
    assert numpy.allclose(Xi, Yi)

def test_dft_numpy():
    N  = 128
    xr = [random.random() for _ in range(N)]
    xi = [random.random() for _ in range(N)]
    Xr = [0.0] * N
    Xi = [0.0] * N
    yr = numpy.array(xr)
    yi = numpy.array(xi)
    Yr = numpy.array(Xr)
    Yi = numpy.array(Xi)
    dft_list(xr, xi, Xr, Xi, N)
    dft_numpy(yr, yi, Yr, Yi, N)
    assert numpy.allclose(Xr, Yr)
    assert numpy.allclose(Xi, Yi)

def dft_list(xr, xi, Xr, Xi, N):
    t0 = timeit.default_timer()
    for k in range(N):
        for n in range(N):
            Xr[k] += xr[n] * math.cos(2 * math.pi * k * n / N) + xi[n] * math.sin(2 * math.pi * k * n / N)
            Xi[k] += xi[n] * math.cos(2 * math.pi * k * n / N) - xr[n] * math.sin(2 * math.pi * k * n / N)
    t1 = timeit.default_timer()
    return t1 - t0

def dft_list_opt(xr, xi, Xr, Xi, N):
    t0 = timeit.default_timer()
    alpha = 2 * math.pi / N
    for u in range(N):
        beta = alpha * u
        for v in range(N):
            cos = math.cos(beta * v)
            sin = math.sin(beta * v)
            Xr[u] += xr[v] * cos + xi[v] * sin
            Xi[u] += xi[v] * cos - xr[v] * sin
    t1 = timeit.default_timer()
    return t1 - t0

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
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    fig.supxlabel("Size (N)")
    fig.supylabel("Time (s)")
    xticks = [8, 16, 32, 64, 128, 256, 512, 1024]
    yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
    x = [[], [], []]
    y = [[], [], []]
    for n in xticks:
        xr = [random.random() for _ in range(n)]
        xi = [random.random() for _ in range(n)]
        for _ in range(10):
            x[0].append(n)
            y[0].append(dft_list(xr, xi, [0.0] * n, [0.0] * n, n))
    for n in xticks:
        xr = [random.random() for _ in range(n)]
        xi = [random.random() for _ in range(n)]
        for _ in range(10):
            x[1].append(n)
            y[1].append(dft_list_opt(xr, xi, [0.0] * n, [0.0] * n, n))
    for n in xticks:
        xr = numpy.random.random_sample(n)
        xi = numpy.random.random_sample(n)
        for _ in range(10):
            x[2].append(n)
            y[2].append(dft_numpy(xr, xi, numpy.full(n, 0.0), numpy.full(n, 0.0), n))
    ax.grid()
    ax.scatter(x[0], y[0], alpha=0.25, color="C0", label="dft_list")
    ax.scatter(x[1], y[1], alpha=0.25, color="C1", label="dft_list_opt")
    ax.scatter(x[2], y[2], alpha=0.25, color="C2", label="dft_numpy")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(xticks[0], xticks[-1])
    plt.ylim(yticks[0], yticks[-1])
    plt.xticks(xticks, [f"{v:d}"   for v in xticks])
    plt.yticks(yticks, [f"{v:.5f}" for v in yticks])
    plt.minorticks_off()
    plt.show()
