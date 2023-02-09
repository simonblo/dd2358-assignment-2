import math
import random
import timeit

def dft(xr, xi, Xr, Xi, N):
    """
    Discrete Fourier Transform
    
    Computes the DFT of an input signal based on a sequence of N samples
    
    Parameters:
    xr (float): List of real components for the original input signal
    xi (float): List of imaginary components for the original input signal
    Xr (float): List of real components for the computed DFT frequency
    Xi (float): List of imaginary components for the computed DFT frequency
    N  (int):   The number of samples in the input signal sequence
    
    Returns:
    float: The elapsed time measured in seconds
    """
    t0 = timeit.default_timer()
    for k in range(N):
        for n in range(N):
            Xr[k] += xr[n] * math.cos(2 * math.pi * k * n / N) + xi[n] * math.sin(2 * math.pi * k * n / N)
            Xi[k] += xi[n] * math.cos(2 * math.pi * k * n / N) - xr[n] * math.sin(2 * math.pi * k * n / N)
    t1 = timeit.default_timer()
    return t1 - t0

if __name__ == "__main__":
    N  = 1024
    xr = [random.random() for _ in range(N)]
    xi = [random.random() for _ in range(N)]
    Xr = [0.0] * N
    Xi = [0.0] * N
    print("Computed DFT in {:.6f} seconds".format(dft(xr, xi, Xr, Xi, N)))
