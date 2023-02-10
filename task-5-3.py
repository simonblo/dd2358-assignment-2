"""
Implementation of the Discrete Fourier Transform
"""

import math
import random
import timeit

def dft(in_r, in_i, out_r, out_i, size):
    """
    Discrete Fourier Transform
    
    Computes the DFT of an input signal based on a sequence of N samples
    
    Parameters:
    in_r  (float): List of real components for the original input signal
    in_i  (float): List of imaginary components for the original input signal
    out_r (float): List of real components for the computed DFT frequency
    out_i (float): List of imaginary components for the computed DFT frequency
    size  (int):   The number of samples in the input signal sequence
    
    Returns:
    float: The elapsed time measured in seconds
    """
    t_0 = timeit.default_timer()
    for i in range(size):
        for j in range(size):
            cosv = math.cos(2 * math.pi * i * j / size)
            sinv = math.sin(2 * math.pi * i * j / size)
            out_r[i] += in_r[j] * cosv + in_i[j] * sinv
            out_i[i] += in_i[j] * cosv - in_r[j] * sinv
    t_1 = timeit.default_timer()
    return t_1 - t_0

if __name__ == "__main__":
    N  = 1024
    x_r = [random.random() for _ in range(N)]
    x_i = [random.random() for _ in range(N)]
    X_r = [0.0] * N
    X_i = [0.0] * N
    print(f"Computed DFT in {dft(x_r, x_i, X_r, X_i, N):.6f} seconds")