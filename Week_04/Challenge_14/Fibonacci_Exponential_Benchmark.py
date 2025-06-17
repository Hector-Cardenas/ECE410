import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

# --- Main Configuration ---
# The maximum N for a 32-bit unsigned integer is 47. We'll test up to this limit.
MAX_N = 47
BENCHMARK_VALUES_N = range(1, MAX_N + 1)
# Number of times to run each function to get a stable average time.
# Since these operations are very fast, we need many repetitions.
NUM_REPETITIONS = 10000

# The CUDA source code, simplified to use standard 32-bit unsigned integers.
cuda_source_code = """
#include <stdio.h>

// ---- KERNELS ----

// Kernel 1: Naive O(N) iterative Fibonacci
__global__ void fib_naive_kernel(unsigned int *result, int n)
{
    // This kernel calculates a single F(n), so only one thread works.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n <= 1) {
            *result = n;
            return;
        }

        unsigned int a = 0;
        unsigned int b = 1;
        unsigned int c;

        for (int i = 2; i <= n; ++i) {
            c = a + b;
            a = b;
            b = c;
        }
        *result = b;
    }
}


// Kernel 2: O(log N) Fast Doubling Fibonacci
__global__ void fib_fast_doubling_kernel(unsigned int *F_n_result, int n)
{
     // This kernel calculates a single F(n), so only one thread works.
     if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n == 0) {
            *F_n_result = 0;
            return;
        }

        unsigned int a = 0; // Represents F(k)
        unsigned int b = 1; // Represents F(k+1)

        // Find the most significant bit to determine the number of iterations
        int high_bit_pos = 0;
        for(int i = 31; i >= 0; i--) {
            if ((n >> i) & 1) {
                high_bit_pos = i;
                break;
            }
        }
        
        for (int i = high_bit_pos; i >= 0; i--) {
            // Doubling step:
            // F(2k)   = F(k) * [2*F(k+1) - F(k)]
            // F(2k+1) = F(k+1)^2 + F(k)^2
            unsigned int c = a * (2 * b - a); // F(2k)
            unsigned int d = a * a + b * b;   // F(2k+1)
            
            // Advance based on the current bit of n
            if ((n >> i) & 1) { // If bit is 1, we are at an odd step
                a = d;        // F(2k+1)
                b = c + d;    // F(2k+2)
            } else { // If bit is 0, we are at an even step
                a = c;        // F(2k)
                b = d;        // F(2k+1)
            }
        }
        *F_n_result = a;
    }
}
"""

def fib_cpu_32bit(n):
    """A simple O(N) iterative addition loop on the CPU."""
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
    try:
        module = SourceModule(cuda_source_code, no_extern_c=True)
        fib_naive_gpu = module.get_function("fib_naive_kernel")
        fib_exp_gpu = module.get_function("fib_fast_doubling_kernel")
        print("CUDA module compiled successfully.\n")
    except drv.Error as e:
        print(f"\nFATAL: PyCUDA Error during kernel compilation: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    
    print("Starting 32-bit Fibonacci Benchmark...")
    print(f"{'N':>4} | {'CPU Naive (us)':>16} | {'CUDA Naive (us)':>17} | {'CUDA Exp (us)':>15}")
    print("-" * 60)
    
    # Pre-allocate memory for results
    result_gpu = drv.mem_alloc(np.dtype(np.uint32).itemsize)
    start_event, stop_event = drv.Event(), drv.Event()

    for n in BENCHMARK_VALUES_N:
        
        # --- 1. CPU Naive Benchmark ---
        # Run many times to get a measurable average
        start_cpu_time = time.perf_counter()
        for _ in range(NUM_REPETITIONS):
            fib_cpu_32bit(n)
        end_cpu_time = time.perf_counter()
        # Time in microseconds for a single run
        time_cpu = ((end_cpu_time - start_cpu_time) / NUM_REPETITIONS) * 1e6

        # --- 2. CUDA Naive Benchmark ---
        start_event.record()
        for _ in range(NUM_REPETITIONS):
            fib_naive_gpu(result_gpu, np.int32(n), block=(1,1,1), grid=(1,1))
        stop_event.record()
        stop_event.synchronize()
        # Time in microseconds for a single run
        time_cuda_naive = (stop_event.time_since(start_event) / NUM_REPETITIONS) * 1e3


        # --- 3. CUDA Exponential Benchmark ---
        start_event.record()
        for _ in range(NUM_REPETITIONS):
            fib_exp_gpu(result_gpu, np.int32(n), block=(1,1,1), grid=(1,1))
        stop_event.record()
        stop_event.synchronize()
        # Time in microseconds for a single run
        time_cuda_exp = (stop_event.time_since(start_event) / NUM_REPETITIONS) * 1e3
        
        results.append({
            'n': n,
            'time_cpu': time_cpu,
            'time_cuda_naive': time_cuda_naive,
            'time_cuda_exp': time_cuda_exp
        })
        
        print(f"{n:>4} | {time_cpu:>16.4f} | {time_cuda_naive:>17.4f} | {time_cuda_exp:>15.4f}")

    # Free GPU memory once after the loop
    result_gpu.free()

    # --- Plotting ---
    n_vals = np.array([r['n'] for r in results])
    t_cpu = np.array([r['time_cpu'] for r in results])
    t_cuda_naive = np.array([r['time_cuda_naive'] for r in results])
    t_cuda_exp = np.array([r['time_cuda_exp'] for r in results])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(n_vals, t_cpu, 'o-', label='CPU Naive O(N)', color='red', markersize=4)
    ax.plot(n_vals, t_cuda_naive, 's-', label='CUDA Naive O(N)', color='blue', markersize=4)
    ax.plot(n_vals, t_cuda_exp, '^-', label='CUDA Exponential O(log N)', color='green', markersize=4)

    ax.set_xlabel('N-th Fibonacci Number', fontsize=12)
    ax.set_ylabel('Average Time (microseconds)', fontsize=12)
    ax.set_title('32-Bit Fibonacci Performance Benchmark (CPU vs. CUDA)', fontsize=16, fontweight='bold')
    
    # Use a linear scale for N and a log scale for time to see the differences clearly
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()

    plot_filename = 'fibonacci_32bit_benchmark.png'
    plt.savefig(plot_filename)
    print(f"\nBenchmark complete. Plot saved to '{plot_filename}'")

if __name__ == "__main__":
    run_benchmark()
