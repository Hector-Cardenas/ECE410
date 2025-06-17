import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import sys
import time
import math

# --- Main Configuration ---
# How many Fibonacci numbers to calculate in parallel (F(0) to F(N-1))
N_TO_CALCULATE = 2**20 
# Block size for the kernel launch. 256 is a good general-purpose choice.
BLOCK_SIZE = 256

# CUDA source code containing the parallel kernel.
# Note that the bignum functions are now used by many threads simultaneously.
cuda_source_code = """
#include <stdio.h>

// This will be dynamically set by the Python script
// #define MAX_LIMBS 1

// Bignum struct and helper functions (set_bignum, add, sub, mul, copy)
// These are __device__ functions, meaning they can be called by kernels.
struct BigNum {
    unsigned int limbs[MAX_LIMBS];
};

__device__ void set_bignum_to_int(BigNum* n, unsigned int val) {
    n->limbs[0] = val;
    for (int i = 1; i < MAX_LIMBS; ++i) n->limbs[i] = 0;
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
    for(int i = 0; i < MAX_LIMBS; ++i) c->limbs[i] = temp_res.limbs[i];
}
__device__ void copy_bignum(BigNum* dest, const BigNum* src) {
    for (int i = 0; i < MAX_LIMBS; ++i) dest->limbs[i] = src->limbs[i];
}

// ------------------------------------------------------------------
// Parallel Fibonacci Kernel
// Each thread calculates one Fibonacci number independently.
// ------------------------------------------------------------------
__global__ void fib_parallel_kernel(BigNum *results, int n_total)
{
    // --- THIS IS THE CORE OF THE PARALLEL APPROACH ---
    // Calculate the unique global index for this thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread checks if it's within the bounds of the problem.
    // If we launch more threads than N, the extra threads do nothing.
    if (idx >= n_total) {
        return;
    }

    // --- Each thread now runs its OWN sequential Fast Doubling algorithm ---
    // It calculates F(idx) and stores it in results[idx].
    // Note: The 'n' for the algorithm is this thread's unique 'idx'.
    int n = idx;
    
    // The target for this thread's result.
    BigNum *my_result = &results[idx];

    if (n == 0) {
        set_bignum_to_int(my_result, 0);
        return;
    }
    
    // The working variables for this thread's calculation.
    BigNum a, b, t1, t2; 

    set_bignum_to_int(&a, 0); // a = F(k)
    set_bignum_to_int(&b, 1); // b = F(k+1)

    int high_bit_pos = 0;
    for(int i = 31; i >= 0; i--) { if ((n >> i) & 1) { high_bit_pos = i; break; } }
    
    for (int i = high_bit_pos; i >= 0; i--) {
        // Doubling step
        add_bignum(&t1, &b, &b);      
        sub_bignum(&t1, &t1, &a);     
        mul_bignum(&t2, &a, &t1);     
        mul_bignum(&t1, &a, &a);     
        mul_bignum(&a, &b, &b);      
        add_bignum(&b, &a, &t1);     
        copy_bignum(&a, &t2);           
        
        if ((n >> i) & 1) { // Advance step
            add_bignum(&t1, &a, &b);
            copy_bignum(&a, &b);
            copy_bignum(&b, &t1);
        }
    }
    copy_bignum(my_result, &a);
}

"""

def get_required_limbs(n):
    if n <= 1: return 1
    bits = math.ceil(n * 0.6942419136)
    return max(1, math.ceil(bits / 32))

def print_gpu_properties():
    """Queries and prints key properties of the active CUDA device."""
    device = drv.Device(0)
    print("--- GPU Device Properties ---")
    print(f"Device Name: {device.name()}")
    print(f"Total Global Memory: {device.total_memory() / (1024**3):.2f} GB")
    print("-----------------------------\n")

def run_parallel_benchmark():
    """Main function to run the parallel throughput benchmark."""
    print_gpu_properties()

    print(f"Preparing to calculate F(0) through F({N_TO_CALCULATE - 1}) in parallel...")

    # Determine memory requirements for the largest number we need to store
    max_limbs = get_required_limbs(N_TO_CALCULATE - 1)
    print(f"Largest number F({N_TO_CALCULATE - 1}) requires {max_limbs} 32-bit limbs.")
    
    # Compile the CUDA module
    module = SourceModule(f"#define MAX_LIMBS {max_limbs}\n" + cuda_source_code, no_extern_c=True)
    bignum_dt = np.dtype([('limbs', np.uint32, (max_limbs,))])
    
    # Allocate GPU memory for the entire array of results
    # This can be a large allocation.
    try:
        results_gpu = drv.mem_alloc(bignum_dt.itemsize * N_TO_CALCULATE)
    except drv.Error as e:
        print(f"Error: Failed to allocate GPU memory for {N_TO_CALCULATE} results.", file=sys.stderr)
        print(f"Required: {bignum_dt.itemsize * N_TO_CALCULATE / (1024**2):.2f} MB", file=sys.stderr)
        print(f"PyCUDA Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Get the compiled kernel function
    fib_parallel_gpu = module.get_function("fib_parallel_kernel")

    # --- Setup Kernel Launch Configuration ---
    # This is how we use the device's properties to launch effectively.
    grid_size = (N_TO_CALCULATE + BLOCK_SIZE - 1) // BLOCK_SIZE
    print(f"Kernel Launch Configuration:")
    print(f"Block Size: {BLOCK_SIZE} threads")
    print(f"Grid Size: {grid_size} blocks")
    print(f"Total Threads Launched: {grid_size * BLOCK_SIZE}")
    
    # Create CUDA events for timing
    start_event, stop_event = drv.Event(), drv.Event()

    # --- Run and Time the Benchmark ---
    print("\nLaunching parallel kernel...")
    start_event.record()
    
    # Launch the kernel with many threads
    fib_parallel_gpu(
        results_gpu, 
        np.int32(N_TO_CALCULATE), 
        block=(BLOCK_SIZE, 1, 1), 
        grid=(grid_size, 1)
    )
    
    stop_event.record()
    stop_event.synchronize() # Wait for the kernel to finish
    
    # Calculate elapsed time in milliseconds
    time_ms = stop_event.time_since(start_event)
    throughput = N_TO_CALCULATE / (time_ms / 1000)

    # --- Print Results ---
    print("Kernel execution finished.")
    print("\n--- Parallel Throughput Results ---")
    print(f"Total time to calculate {N_TO_CALCULATE} numbers: {time_ms:.4f} ms")
    print(f"Throughput: {throughput:,.2f} Fibonacci numbers/second")

    # Clean up GPU memory
    results_gpu.free()

if __name__ == "__main__":
    run_parallel_benchmark()
