import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import sys
import time
import math
import secrets
import matplotlib.pyplot as plt

# --- Main Configuration ---
# List of N values to benchmark.
BENCHMARK_VALUES_N = np.unique(np.logspace(1, 15, 25, base=2, dtype=int)) # Powers of 2 up to 32768
# Number of times to loop the core operation in micro-benchmarks for stable timing.
MICRO_BENCHMARK_ITERATIONS = 1000

# The CUDA source code is a raw string. We will prepend #define statements later.
cuda_source_code = """
#include <stdio.h>

// Bignum struct and all __device__ helper functions
struct BigNum {
    unsigned int limbs[MAX_LIMBS];
};
__device__ void set_bignum_to_int(BigNum* n, unsigned int val) {
    n->limbs[0] = val;
    for (int i = 1; i < MAX_LIMBS; ++i) { n->limbs[i] = 0; }
}
__device__ void add_bignum(BigNum* c, const BigNum* a, const BigNum* b) {
    unsigned long long carry = 0;
    for (int i = 0; i < MAX_LIMBS; ++i) {
        carry += (unsigned long long)a->limbs[i] + b->limbs[i];
        c->limbs[i] = (unsigned int)(carry & 0xFFFFFFFF);
        carry >>= 32;
    }
}
__device__ void sub_bignum(BigNum* c, const BigNum* a, const BigNum* b) {
    long long borrow = 0;
    for (int i = 0; i < MAX_LIMBS; ++i) {
        long long diff = (long long)a->limbs[i] - b->limbs[i] - borrow;
        c->limbs[i] = (unsigned int)(diff < 0 ? diff + 0x100000000LL : diff);
        borrow = diff < 0 ? 1 : 0;
    }
}
__device__ void mul_bignum(BigNum* c, const BigNum* a, const BigNum* b) {
    BigNum temp_res;
    set_bignum_to_int(&temp_res, 0);
    for (int i = 0; i < MAX_LIMBS; ++i) {
        if (a->limbs[i] == 0) continue;
        unsigned long long carry = 0;
        for (int j = 0; j + i < MAX_LIMBS; ++j) {
            carry += temp_res.limbs[i+j] + (unsigned long long)a->limbs[i] * b->limbs[j];
            temp_res.limbs[i+j] = (unsigned int)(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
    }
    for(int i = 0; i < MAX_LIMBS; ++i) { c->limbs[i] = temp_res.limbs[i]; }
}
__device__ void copy_bignum(BigNum* dest, const BigNum* src) {
    for (int i = 0; i < MAX_LIMBS; ++i) { dest->limbs[i] = src->limbs[i]; }
}

// ---- KERNELS ----

// Kernel to micro-benchmark a single 'add' operation for extrapolation
__global__ void benchmark_add_kernel(BigNum *c, const BigNum *a, const BigNum *b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < BENCH_ITERATIONS; ++i) {
            add_bignum(c, a, b);
        }
    }
}

// Kernel for O(log N) Fast Doubling Fibonacci (for direct timing)
__global__ void fib_fast_doubling_kernel(
    BigNum *F_n, BigNum* F_n_plus_1, int n,
    BigNum *a_ptr, BigNum *b_ptr, BigNum *t1_ptr, BigNum *t2_ptr
) {
     if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n == 0) {
            set_bignum_to_int(F_n, 0);
            set_bignum_to_int(F_n_plus_1, 1);
            return;
        }
        set_bignum_to_int(a_ptr, 0);
        set_bignum_to_int(b_ptr, 1);
        int high_bit_pos = 0;
        // Find the most significant bit to determine the number of iterations
        for(int i = 31; i >= 0; i--) { if ((n >> i) & 1) { high_bit_pos = i; break; } }
        
        for (int i = high_bit_pos; i >= 0; i--) {
            // Doubling step
            add_bignum(t1_ptr, b_ptr, b_ptr);
            sub_bignum(t1_ptr, t1_ptr, a_ptr);
            mul_bignum(t2_ptr, a_ptr, t1_ptr);
            mul_bignum(t1_ptr, a_ptr, a_ptr);
            mul_bignum(a_ptr, b_ptr, b_ptr);
            add_bignum(b_ptr, a_ptr, t1_ptr);
            copy_bignum(a_ptr, t2_ptr);
            // Advance step if the corresponding bit in n is set
            if ((n >> i) & 1) {
                add_bignum(t1_ptr, a_ptr, b_ptr);
                copy_bignum(a_ptr, b_ptr);
                copy_bignum(b_ptr, t1_ptr);
            }
        }
        copy_bignum(F_n, a_ptr);
        copy_bignum(F_n_plus_1, b_ptr);
    }
}
"""

def get_required_limbs(n):
    """Calculates the number of 32-bit integers needed to store F(n)."""
    if n <= 1: return 1
    # n * log2(phi) gives the approximate number of bits
    bits = math.ceil(n * 0.6942419136)
    return max(1, math.ceil(bits / 32))

def benchmark_single_cpu_add(bit_length):
    """Times a single bignum addition on the CPU for extrapolation."""
    # Create two random numbers of the required size
    a = secrets.randbits(bit_length)
    b = secrets.randbits(bit_length)
    
    start_time = time.perf_counter()
    for _ in range(MICRO_BENCHMARK_ITERATIONS):
        _ = a + b # The core operation
    end_time = time.perf_counter()
    
    # Return the average time for a single operation
    return (end_time - start_time) / MICRO_BENCHMARK_ITERATIONS

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
    """Main function to run all benchmarks and plot the results."""
    print_gpu_properties()

    results = []
    
    print("Practical Fibonacci Benchmark (Arbitrary Precision)...")
    print(f"{'N':>8} | {'CPU Naive (ms)':>20} | {'CUDA Naive (ms)':>20} | {'CUDA Exp (ms)':>16}")
    print("-" * 75)

    for n_numpy in BENCHMARK_VALUES_N:
        n = int(n_numpy)
        max_limbs = get_required_limbs(n)
        bit_length = max_limbs * 32

        # --- 1. CPU Naive (Extrapolated) ---
        time_per_cpu_add = benchmark_single_cpu_add(bit_length)
        total_time_cpu_extrapolated = (time_per_cpu_add * n) * 1000 # Convert to ms

        # --- CUDA Benchmarks Setup ---
        # Manually construct the final source code by prepending the #define statements.
        # This is a robust way to avoid compilation errors.
        defines = f"#define MAX_LIMBS {max_limbs}\n#define BENCH_ITERATIONS {MICRO_BENCHMARK_ITERATIONS}\n"
        final_cuda_code = defines + cuda_source_code
        
        try:
            module = SourceModule(final_cuda_code, no_extern_c=True)
            add_bench_kernel = module.get_function("benchmark_add_kernel")
            fib_exp_gpu = module.get_function("fib_fast_doubling_kernel")
        except drv.Error as e:
            print(f"\nFATAL: PyCUDA Error during kernel compilation for N={n}.", file=sys.stderr)
            print(f"This indicates a problem in the CUDA C++ source code.", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        
        bignum_dt = np.dtype([('limbs', np.uint32, (max_limbs,))])
        start_event, stop_event = drv.Event(), drv.Event()

        # --- 2. CUDA Naive (Extrapolated) ---
        a_gpu = drv.mem_alloc(bignum_dt.itemsize)
        b_gpu = drv.mem_alloc(bignum_dt.itemsize)
        c_gpu = drv.mem_alloc(bignum_dt.itemsize)
        
        start_event.record()
        add_bench_kernel(c_gpu, a_gpu, b_gpu, block=(1,1,1), grid=(1,1))
        stop_event.record()
        stop_event.synchronize()
        time_for_many_adds = stop_event.time_since(start_event)
        # Calculate extrapolated time: (total_time / num_ops) * N
        total_time_cuda_extrapolated = (time_for_many_adds / MICRO_BENCHMARK_ITERATIONS) * n
        a_gpu.free(); b_gpu.free(); c_gpu.free()

        # --- 3. CUDA Exponential (Actual Run) ---
        res_n, res_n1 = drv.mem_alloc(bignum_dt.itemsize), drv.mem_alloc(bignum_dt.itemsize)
        a_p, b_p, t1_p, t2_p = drv.mem_alloc(bignum_dt.itemsize), drv.mem_alloc(bignum_dt.itemsize), drv.mem_alloc(bignum_dt.itemsize), drv.mem_alloc(bignum_dt.itemsize)
        
        start_event.record()
        fib_exp_gpu(res_n, res_n1, np.int32(n), a_p, b_p, t1_p, t2_p, block=(1, 1, 1), grid=(1, 1))
        stop_event.record()
        stop_event.synchronize()
        time_cuda_exp = stop_event.time_since(start_event)
        res_n.free(); res_n1.free(); a_p.free(); b_p.free(); t1_p.free(); t2_p.free()
        
        results.append({
            'n': n,
            'cpu_naive_ext': total_time_cpu_extrapolated,
            'cuda_naive_ext': total_time_cuda_extrapolated,
            'cuda_exp': time_cuda_exp
        })
        
        print(f"{n:>8} | {total_time_cpu_extrapolated:>20.4f} | {total_time_cuda_extrapolated:>20.4f} | {time_cuda_exp:>16.4f}")

    # --- Plotting ---
    n_vals = np.array([r['n'] for r in results])
    t_cpu = [r['cpu_naive_ext'] for r in results]
    t_cuda_naive = [r['cuda_naive_ext'] for r in results]
    t_cuda_exp = [r['cuda_exp'] for r in results]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(n_vals, t_cpu, 'o-', label='CPU Naive O(N) (Extrapolated)', color='red')
    ax.plot(n_vals, t_cuda_naive, 's-', label='CUDA Naive O(N) (Extrapolated)', color='blue')
    ax.plot(n_vals, t_cuda_exp, '^-', label='CUDA Exponential O(log N) (Actual)', color='green')

    ax.set_xlabel('N-th Fibonacci Number', fontsize=12)
    ax.set_ylabel('Time (milliseconds)', fontsize=12)
    ax.set_title('Practical Fibonacci Time Complexity Benchmark', fontsize=16, fontweight='bold')
    
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.tight_layout()

    plot_filename = 'fibonacci_practical_benchmark.png'
    plt.savefig(plot_filename)
    print(f"\nBenchmark complete. Plot saved to '{plot_filename}'")

if __name__ == "__main__":
    run_benchmark()
