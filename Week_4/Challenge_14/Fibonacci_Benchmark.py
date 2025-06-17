import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import sys
import time
import math
import matplotlib.pyplot as plt

# --- Main Configuration ---
# Generate a denser set of N values for benchmarking.
# We use np.linspace on a log scale for even distribution on the log plot.
BENCHMARK_VALUES_N = np.unique(np.logspace(1, 9, 30, dtype=int)) # 30 points from 10 up to ~1470

# CUDA source code now uses 'double' for all calculations.
cuda_source_code = """
#include <stdio.h>

// ------------------------------------------------------------------
// Kernel 1: Naive O(N) Iterative Fibonacci using double (Simplified)
// This now calculates a single F(N) using one thread, for a direct comparison.
// ------------------------------------------------------------------
__global__ void fib_naive_double_kernel(double *result, int n)
{
    // This kernel calculates a single F(n), so only one thread works.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n <= 1) {
            *result = (double)n;
            return;
        }

        double a = 0.0;
        double b = 1.0;
        double c;

        for (int i = 2; i <= n; ++i) {
            c = a + b;
            a = b;
            b = c;
        }
        *result = b;
    }
}


// ------------------------------------------------------------------
// Kernel 2: O(log N) Fast Doubling Fibonacci using double
// ------------------------------------------------------------------
__global__ void fib_fast_doubling_double_kernel(double *F_n_result, int n)
{
     // This kernel calculates a single F(n), so only one thread works.
     if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n == 0) {
            *F_n_result = 0.0;
            return;
        }

        double a = 0.0; // F(k)
        double b = 1.0; // F(k+1)

        int high_bit_pos = 0;
        for(int i = 31; i >= 0; i--) { if ((n >> i) & 1) { high_bit_pos = i; break; } }
        
        for (int i = high_bit_pos; i >= 0; i--) {
            // Doubling step using standard float arithmetic
            double t1 = 2.0 * b - a; // t1 = 2*F(k+1) - F(k)
            double t2 = a * t1;      // t2 = F(k) * [2*F(k+1) - F(k)] = F(2k)
            double a_new = b*b + a*a; // a_new = F(k+1)^2 + F(k)^2 = F(2k+1)
            
            a = t2;
            b = a_new;
            
            if ((n >> i) & 1) { // Advance step for set bits
                double t3 = a + b;
                a = b;
                b = t3;
            }
        }
        *F_n_result = a;
    }
}
"""

def fib_cpu_double(n):
    """Naïve O(N) Fibonacci on CPU using Python floats."""
    if n <= 1:
        return float(n)
    a, b = 0.0, 1.0
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
        print(f"Could not query GPU properties: {e}", file=sys.stderr)
        sys.exit(1)


def run_benchmark():
    """Main function to run all benchmarks and plot the results."""
    print_gpu_properties()
    
    # Compile the CUDA module
    module = SourceModule(cuda_source_code)
    fib_naive_gpu = module.get_function("fib_naive_double_kernel")
    fib_exp_gpu = module.get_function("fib_fast_doubling_double_kernel")

    results = []
    
    print("Starting Fibonacci Benchmark (64-bit Double Precision)...")
    print("This will take a moment as we are gathering more data points.")
    print(f"{'N':>8} | {'CPU Naive (ms)':>18} | {'CUDA Naive (ms)':>18} | {'CUDA Exp (ms)':>16}")
    print("-" * 72)

    for n_numpy in BENCHMARK_VALUES_N:
        n = int(n_numpy) # Convert numpy int to standard python int
        
        # --- 1. CPU Naive Benchmark ---
        start_time = time.perf_counter()
        fib_cpu_double(n) # Simplified: Time a single O(N) call
        end_time = time.perf_counter()
        time_cpu_naive = (end_time - start_time) * 1000

        # --- 2. CUDA Naive Benchmark ---
        result_gpu_naive = drv.mem_alloc(np.dtype(np.float64).itemsize)
        start_event, stop_event = drv.Event(), drv.Event()
        start_event.record()
        # Simplified: Call the single-threaded naive kernel
        # FIX: Pass 'n' as a numpy.int32 to prevent type errors.
        fib_naive_gpu(result_gpu_naive, np.int32(n), block=(1,1,1), grid=(1,1))
        stop_event.record()
        stop_event.synchronize()
        time_cuda_naive = stop_event.time_since(start_event)
        result_gpu_naive.free()

        # --- 3. CUDA Exponential Benchmark ---
        result_gpu_exp = drv.mem_alloc(np.dtype(np.float64).itemsize)
        start_event, stop_event = drv.Event(), drv.Event()
        start_event.record()
        # FIX: Pass 'n' as a numpy.int32 to prevent type errors.
        fib_exp_gpu(result_gpu_exp, np.int32(n), block=(1, 1, 1), grid=(1, 1))
        stop_event.record()
        stop_event.synchronize()
        time_cuda_exp = stop_event.time_since(start_event)
        result_gpu_exp.free()
        
        results.append({
            'n': n,
            'cpu_naive': time_cpu_naive,
            'cuda_naive': time_cuda_naive,
            'cuda_exp': time_cuda_exp
        })
        
        print(f"{n:>8} | {time_cpu_naive:>18.4f} | {time_cuda_naive:>18.4f} | {time_cuda_exp:>16.4f}")

    # --- Data Processing for Plotting ---
    n_vals = np.array([r['n'] for r in results])
    t_cpu_naive = np.array([r['cpu_naive'] for r in results])
    t_cuda_naive = np.array([r['cuda_naive'] for r in results])
    t_cuda_exp = np.array([r['cuda_exp'] for r in results])
    
    # --- Calculate Line of Best Fit on log-log data ---
    log_n = np.log10(n_vals)
    
    # Filter out zero or negative times before taking log
    valid_cpu = t_cpu_naive > 0
    m_cpu, c_cpu = np.polyfit(log_n[valid_cpu], np.log10(t_cpu_naive[valid_cpu]), 1)
    
    valid_cuda_naive = t_cuda_naive > 0
    m_cuda_n, c_cuda_n = np.polyfit(log_n[valid_cuda_naive], np.log10(t_cuda_naive[valid_cuda_naive]), 1)
    
    valid_cuda_exp = t_cuda_exp > 0
    m_cuda_e, c_cuda_e = np.polyfit(log_n[valid_cuda_exp], np.log10(t_cuda_exp[valid_cuda_exp]), 1)
    
    # Generate y-values for the fitted lines
    fit_cpu = 10**(m_cpu * log_n + c_cpu)
    fit_cuda_naive = 10**(m_cuda_n * log_n + c_cuda_n)
    fit_cuda_exp = 10**(m_cuda_e * log_n + c_cuda_e)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the raw data as scattered points
    ax.scatter(n_vals, t_cpu_naive, alpha=0.5, color='red', label='_nolegend_')
    ax.scatter(n_vals, t_cuda_naive, alpha=0.5, color='blue', label='_nolegend_')
    ax.scatter(n_vals, t_cuda_exp, alpha=0.5, color='green', label='_nolegend_')
    
    # Plot the lines of best fit
    ax.plot(n_vals, fit_cpu, label=f'CPU Naive (Best Fit, slope≈{m_cpu:.2f})', color='darkred')
    ax.plot(n_vals, fit_cuda_naive, label=f'CUDA Naive (Best Fit, slope≈{m_cuda_n:.2f})', color='darkblue')
    ax.plot(n_vals, fit_cuda_exp, label=f'CUDA Exponential (Best Fit, slope≈{m_cuda_e:.2f})', color='darkgreen')

    ax.set_xlabel('N-th Fibonacci Number', fontsize=12)
    ax.set_ylabel('Time (milliseconds)', fontsize=12)
    ax.set_title('Fibonacci Benchmark (64-bit Double) - Line of Best Fit', fontsize=16, fontweight='bold')
    
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()

    plot_filename = 'fibonacci_benchmark_double_best_fit.png'
    plt.savefig(plot_filename)
    print(f"\nBenchmark complete. Plot saved to '{plot_filename}'")

if __name__ == "__main__":
    run_benchmark()
