import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import sys
import time
import math
import matplotlib.pyplot as plt

# --- Main Configuration ---
# Define a single, large N to determine the maximum memory needed.
MAX_N_SUPPORTED = 2**20
# List of N values to benchmark, all must be <= MAX_N_SUPPORTED
BENCHMARK_VALUES_N = np.unique(np.logspace(1, 16, 30, base=2, dtype=int)) 
# Since the naive method is slow, we only run it up to a certain N for the benchmark
NAIVE_MAX_N = 30000 

def get_required_limbs(n):
    """Calculates the number of 32-bit integers needed to store F(n)."""
    if n <= 1: return 1
    # n * log2(phi) gives the approximate number of bits
    bits = math.ceil(n * 0.6942419136)
    return max(1, math.ceil(bits / 32))

# Calculate the MAX_LIMBS required for the largest N we will test.
# This value will be hard-coded into the CUDA source for a single, stable compilation.
MAX_LIMBS = get_required_limbs(MAX_N_SUPPORTED)

# The CUDA source code, using our bignum implementation.
cuda_source_template = """
#include <stdio.h>

#define MAX_LIMBS {MAX_LIMBS}

// Bignum struct and essential helper functions
struct BigNum {{
    unsigned int limbs[MAX_LIMBS];
}};
__device__ void set_bignum_to_int(BigNum* n, unsigned int val) {{
    n->limbs[0] = val;
    for (int i = 1; i < MAX_LIMBS; ++i) {{ n->limbs[i] = 0; }}
}}
__device__ void add_bignum(BigNum* c, const BigNum* a, const BigNum* b) {{
    unsigned long long carry = 0;
    for (int i = 0; i < MAX_LIMBS; ++i) {{
        carry += (unsigned long long)a->limbs[i] + b->limbs[i];
        c->limbs[i] = (unsigned int)(carry & 0xFFFFFFFF);
        carry >>= 32;
    }}
}}
__device__ void copy_bignum(BigNum* dest, const BigNum* src) {{
    for (int i = 0; i < MAX_LIMBS; ++i) {{ dest->limbs[i] = src->limbs[i]; }}
}}

// ---- KERNEL ----
extern "C" {{
// This kernel performs a simple iterative addition loop (naive Fibonacci)
__global__ void fib_naive_bignum_kernel(
    BigNum *result, int n,
    BigNum *a_ptr, BigNum *b_ptr, BigNum *c_ptr
) {{
     if (threadIdx.x == 0 && blockIdx.x == 0) {{
        if (n <= 1) {{
            set_bignum_to_int(result, n);
            return;
        }}
        set_bignum_to_int(a_ptr, 0);
        set_bignum_to_int(b_ptr, 1);

        for (int i = 2; i <= n; ++i) {{
            add_bignum(c_ptr, a_ptr, b_ptr);
            copy_bignum(a_ptr, b_ptr);
            copy_bignum(b_ptr, c_ptr);
        }}
        copy_bignum(result, b_ptr);
    }}
}}
}} // extern "C"
"""

def fib_cpu_bignum(n):
    """A simple O(N) iterative addition loop on the CPU using Python's native bignums."""
    if n <= 1: return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

def print_gpu_properties():
    """Queries and prints key properties of the active CUDA device."""
    try:
        device = drv.Device(0)
        print("--- GPU Device Properties ---")
        print(f"Device Name: {device.name()}")
        print("-----------------------------\n")
    except drv.Error as e:
        print(f"Could not query GPU properties. PyCUDA Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_benchmark():
    """Main function to run the benchmark and plot the results."""
    print_gpu_properties()

    # --- Compile CUDA Module ONCE ---
    print(f"Compiling CUDA kernel with fixed MAX_LIMBS = {MAX_LIMBS}...")
    final_cuda_code = cuda_source_template.format(MAX_LIMBS=MAX_LIMBS)
    try:
        module = SourceModule(final_cuda_code, no_extern_c=True)
        fib_naive_gpu = module.get_function("fib_naive_bignum_kernel")
        print("CUDA module compiled successfully.\n")
    except drv.Error as e:
        print(f"\nFATAL: PyCUDA Error during kernel compilation: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    
    print("Starting Arbitrary-Precision Naive Fibonacci Benchmark...")
    print(f"{'N':>8} | {'CPU Time (ms)':>18} | {'GPU Time (ms)':>18}")
    print("-" * 50)

    # --- Pre-allocate memory ONCE ---
    bignum_dt = np.dtype([('limbs', np.uint32, (MAX_LIMBS,))])
    result_gpu = drv.mem_alloc(bignum_dt.itemsize)
    a_p, b_p, c_p = drv.mem_alloc(bignum_dt.itemsize), drv.mem_alloc(bignum_dt.itemsize), drv.mem_alloc(bignum_dt.itemsize)
    start_event, stop_event = drv.Event(), drv.Event()

    for n_numpy in BENCHMARK_VALUES_N:
        n = int(n_numpy)
        if n > NAIVE_MAX_N: break
        
        # --- CPU Benchmark ---
        start_cpu_time = time.perf_counter()
        fib_cpu_bignum(n)
        end_cpu_time = time.perf_counter()
        time_cpu = (end_cpu_time - start_cpu_time) * 1000

        # --- GPU Benchmark ---
        start_event.record()
        fib_naive_gpu(result_gpu, np.int32(n), a_p, b_p, c_p, block=(1, 1, 1), grid=(1, 1))
        stop_event.record()
        stop_event.synchronize()
        time_gpu = stop_event.time_since(start_event)
        
        results.append({
            'n': n,
            'time_cpu': time_cpu,
            'time_gpu': time_gpu
        })
        
        print(f"{n:>8} | {time_cpu:>18.4f} | {time_gpu:>18.4f}")

    # --- Free memory ONCE after the loop ---
    result_gpu.free(); a_p.free(); b_p.free(); c_p.free()

    # --- Plotting ---
    n_vals = np.array([r['n'] for r in results])
    t_cpu = np.array([r['time_cpu'] for r in results])
    t_gpu = np.array([r['time_gpu'] for r in results])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(n_vals, t_cpu, 'o-', label='CPU Naive O(N) (Bignum)', color='red')
    ax.plot(n_vals, t_gpu, 's-', label='CUDA Naive O(N) (Bignum)', color='blue')

    ax.set_xlabel('N-th Fibonacci Number', fontsize=12)
    ax.set_ylabel('Time (milliseconds)', fontsize=12)
    ax.set_title('CPU vs. GPU Naive Fibonacci Performance (Arbitrary Precision)', fontsize=16, fontweight='bold')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()

    plot_filename = 'fibonacci_bignum_naive_benchmark.png'
    plt.savefig(plot_filename)
    print(f"\nBenchmark complete. Plot saved to '{plot_filename}'")

if __name__ == "__main__":
    run_benchmark()
