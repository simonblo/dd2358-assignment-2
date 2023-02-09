import logging
import math
import numpy
import pytest

def test_dft():
    xr = [1.0,  2.0,  0.0, -1.0]
    xi = [0.0, -1.0, -1.0,  2.0]
    Xr = [0.0,  0.0,  0.0,  0.0]
    Xi = [0.0,  0.0,  0.0,  0.0]
    dft(xr, xi, Xr, Xi, 4)
    assert numpy.allclose(Xr, [2.0, -2.0,  0.0, 4.0])
    assert numpy.allclose(Xi, [0.0, -2.0, -2.0, 4.0])

def dft(xr, xi, Xr, Xi, N):
    for k in range(N):
        for n in range(N):
            Xr[k] += xr[n] * math.cos(2 * math.pi * k * n / N) + xi[n] * math.sin(2 * math.pi * k * n / N)
            Xi[k] += xi[n] * math.cos(2 * math.pi * k * n / N) - xr[n] * math.sin(2 * math.pi * k * n / N)
            logging.info("X[{}]: {:8.5f}{:+8.5f}i".format(k, Xr[k], Xi[k]))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    xr = [1.0,  2.0,  0.0, -1.0]
    xi = [0.0, -1.0, -1.0,  2.0]
    Xr = [0.0,  0.0,  0.0,  0.0]
    Xi = [0.0,  0.0,  0.0,  0.0]
    dft(xr, xi, Xr, Xi, 4)
