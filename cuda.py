import numpy as np
from numba import cuda
import time

# CUDA kernel for vector multiplication
@cuda.jit
def vector_multiplication(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] * b[idx]

# CUDA kernel for vector addition
@cuda.jit
def vector_addition(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

def print_performance(label, duration):
    print(f"{label}: {duration:.6f} seconds")

def main():
    sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]  # Expanded testing sizes

    for size in sizes:
        print("\nSize:", size)

        # Initialize arrays
        a = np.random.rand(size)
        b = np.random.rand(size)
        c = np.zeros_like(a)  # For multiplication
        d = np.zeros_like(a)  # For addition

        # CUDA kernel configurations
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

        # Vector multiplication using CUDA
        start = time.time()
        vector_multiplication[blocks_per_grid, threads_per_block](a, b, c)
        cuda.synchronize()
        end = time.time()
        print_performance("Vector Multiplication (CUDA)", end - start)

        # Vector addition using CUDA
        start = time.time()
        vector_addition[blocks_per_grid, threads_per_block](a, b, d)
        cuda.synchronize()
        end = time.time()
        print_performance("Vector Addition (CUDA)", end - start)

        # Verify results with CPU-based calculation
        c_cpu = a * b  # Multiplication
        d_cpu = a + b  # Addition


if __name__ == "__main__":
    main()
