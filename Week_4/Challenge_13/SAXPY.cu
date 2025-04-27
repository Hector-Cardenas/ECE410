#include <stdio.h>
#include <stdlib.h> // For malloc, free, exit
#include <math.h>   // For fabsf, fmaxf (if using error check)

// Include windows.h for QueryPerformanceCounter/Frequency
#ifdef _WIN32 // Guard for Windows compilation
#include <windows.h>
#else
#error "This timing code is specific to Windows. Use POSIX clock_gettime or other timers on non-Windows platforms."
#endif

#include <cuda_runtime.h> // For CUDA API functions and events

// Simple CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        /* Attempt cleanup before exiting */ \
        /* Note: Assumes relevant variables like events and file pointers are declared */ \
        /* cudaEventDestroy(start_event); cudaEventDestroy(stop_event); */ \
        /* if (results_file) fclose(results_file); */ \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Define the CUDA kernel function `saxpy`.
// Performs the actual SAXPY computation on the GPU.
__global__
void saxpy(int n, float a, float *x, float *y)
{
  // Calculate the unique global thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Ensure the index is within the bounds of the vectors
  if (i < n) {
    // Perform the SAXPY operation: y = a*x + y
    y[i] = a * x[i] + y[i];
  }
}

// The main function that runs on the CPU (host).
int main(void)
{
  // --- Configuration ---
  const int NUM_ITERATIONS = 5; // Number of iterations to average over for each size
  const int START_P = 10;       // Starting exponent for N (N=2^p)
  const int END_P = 32;         // Ending exponent for N
  // SAXPY parameters
  float alpha = 2.0f;
  // CUDA kernel launch parameters
  int blockSize = 256; // Number of threads per block
  // ---------------------

  // Host vector pointers (CPU memory)
  float *h_x = NULL, *h_y = NULL;
  // Device vector pointers (GPU memory)
  float *d_x = NULL, *d_y = NULL;

  // CUDA Events for precise kernel timing (created once)
  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  // --- Windows High-Resolution Timer Variables ---
  LARGE_INTEGER cpu_frequency; // Stores the frequency of the performance counter
  LARGE_INTEGER start_time, stop_time; // Generic start/stop for timing sections
  // Get the frequency of the performance counter (ticks per second).
  if (!QueryPerformanceFrequency(&cpu_frequency)) {
      fprintf(stderr, "Error: QueryPerformanceFrequency failed!\n");
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
      return EXIT_FAILURE;
  }
  // ---------------------------------------------

  // Variables to store timing results in milliseconds for a single iteration
  float iter_kernel_ms = 0;
  double iter_total_ms = 0;
  double iter_host_alloc_ms = 0;
  double iter_device_alloc_ms = 0;
  double iter_host_init_ms = 0;
  double iter_h2d_ms = 0;
  double iter_d2h_ms = 0;
  double iter_cleanup_ms = 0;

  // Variables to accumulate timings over NUM_ITERATIONS
  double total_kernel_ms_acc = 0;
  double total_total_ms_acc = 0;
  double total_host_alloc_ms_acc = 0;
  double total_device_alloc_ms_acc = 0;
  double total_host_init_ms_acc = 0;
  double total_h2d_ms_acc = 0;
  double total_d2h_ms_acc = 0;
  double total_cleanup_ms_acc = 0;

  // --- File Handling for CSV Output ---
  FILE *results_file = NULL; // File pointer for the output CSV
  const char *filename = "saxpy_timing_results_avg.csv"; // New filename for averaged results

  // Open the CSV file in write mode ('w')
  results_file = fopen(filename, "w");
  if (results_file == NULL) {
      perror("Error opening output file"); // Print system error message
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
      return EXIT_FAILURE;
  }
  // Write the header row to the CSV file, indicating averaged results
  fprintf(results_file, "Vector Size (log2 N),Avg Kernel Time (ms),Avg Host Alloc Time (ms),Avg Device Alloc Time (ms),Avg Host Init Time (ms),Avg H2D Time (ms),Avg D2H Time (ms),Avg Cleanup Time (ms),Avg Total Time (ms)\n");
  // ------------------------------------

  // Print informational messages to the console
  printf("Running SAXPY for vector sizes from 2^%d to 2^%d\n", START_P, END_P);
  printf("Averaging results over %d iterations for each size.\n", NUM_ITERATIONS);
  printf("Results will also be saved to %s\n", filename);
  printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
  // Print the console output header, indicating averages
  printf("log2 N | Avg Kernel(ms) | Avg HostAlloc(ms)| Avg DevAlloc(ms) | Avg HostInit(ms)| Avg H2D(ms)    | Avg D2H(ms)    | Avg Cleanup(ms)| Avg Total(ms)\n");
  printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

  // Outer loop: Iterate through vector sizes, where N = 2^p
  for (int p = START_P; p <= END_P; ++p)
  {
      // Reset accumulators for this vector size p
      total_kernel_ms_acc = 0;
      total_total_ms_acc = 0;
      total_host_alloc_ms_acc = 0;
      total_device_alloc_ms_acc = 0;
      total_host_init_ms_acc = 0;
      total_h2d_ms_acc = 0;
      total_d2h_ms_acc = 0;
      total_cleanup_ms_acc = 0;

      // Calculate vector size N and total bytes needed for one vector
      int N = 1 << p; // N = 2^p
      size_t sizeBytes = (size_t)N * sizeof(float); // Use size_t for potentially large sizes

      // Inner loop: Repeat operations NUM_ITERATIONS times for averaging
      for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
      {
          // --- Start Total Time Measurement for this single iteration ---
          QueryPerformanceCounter(&start_time); // Use generic start_time here

          // --- Host Memory Allocation Timing ---
          QueryPerformanceCounter(&start_time);
          h_x = (float*)malloc(sizeBytes);
          h_y = (float*)malloc(sizeBytes);
          QueryPerformanceCounter(&stop_time);
          iter_host_alloc_ms = (double)(stop_time.QuadPart - start_time.QuadPart) * 1000.0 / cpu_frequency.QuadPart;
          if (h_x == NULL || h_y == NULL) { // Check after timing
              fprintf(stderr, "Failed to allocate host memory for N = 2^%d (%.2f GB) on iter %d\n", p, (double)sizeBytes * 2 / (1 << 30), iter);
              fclose(results_file); cudaEventDestroy(start_event); cudaEventDestroy(stop_event); exit(EXIT_FAILURE);
          }
          // -------------------------------------

          // --- Device Memory Allocation Timing ---
          QueryPerformanceCounter(&start_time);
          CUDA_CHECK(cudaMalloc(&d_x, sizeBytes));
          CUDA_CHECK(cudaMalloc(&d_y, sizeBytes));
          QueryPerformanceCounter(&stop_time);
          iter_device_alloc_ms = (double)(stop_time.QuadPart - start_time.QuadPart) * 1000.0 / cpu_frequency.QuadPart;
          // ---------------------------------------

          // --- Host Vector Initialization Timing ---
          QueryPerformanceCounter(&start_time);
          for (int i = 0; i < N; i++) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
          }
          QueryPerformanceCounter(&stop_time);
          iter_host_init_ms = (double)(stop_time.QuadPart - start_time.QuadPart) * 1000.0 / cpu_frequency.QuadPart;
          // ---------------------------------------

          // --- Data Transfer (Host to Device) Timing ---
          QueryPerformanceCounter(&start_time);
          CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeBytes, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeBytes, cudaMemcpyHostToDevice));
          QueryPerformanceCounter(&stop_time);
          iter_h2d_ms = (double)(stop_time.QuadPart - start_time.QuadPart) * 1000.0 / cpu_frequency.QuadPart;
          // -------------------------------------------

          // --- Kernel Execution and Timing ---
          int gridSize = (N + blockSize - 1) / blockSize;
          // Ensure previous CUDA operations are finished before starting kernel timing if needed
          // cudaDeviceSynchronize(); // Optional: uncomment if concerned about H2D overlap affecting kernel start event timing
          CUDA_CHECK(cudaEventRecord(start_event, 0));
          saxpy<<<gridSize, blockSize>>>(N, alpha, d_x, d_y);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaEventRecord(stop_event, 0));
          // Need to synchronize *here* to get kernel time for *this* iteration
          CUDA_CHECK(cudaEventSynchronize(stop_event));
          CUDA_CHECK(cudaEventElapsedTime(&iter_kernel_ms, start_event, stop_event));
          // -----------------------------------

          // --- Data Transfer (Device to Host) Timing ---
          QueryPerformanceCounter(&start_time);
          CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeBytes, cudaMemcpyDeviceToHost));
          QueryPerformanceCounter(&stop_time);
          iter_d2h_ms = (double)(stop_time.QuadPart - start_time.QuadPart) * 1000.0 / cpu_frequency.QuadPart;
          // -------------------------------------------

          // --- Cleanup Timing ---
          QueryPerformanceCounter(&start_time);
          CUDA_CHECK(cudaFree(d_x));
          CUDA_CHECK(cudaFree(d_y));
          free(h_x);
          free(h_y);
          QueryPerformanceCounter(&stop_time);
          iter_cleanup_ms = (double)(stop_time.QuadPart - start_time.QuadPart) * 1000.0 / cpu_frequency.QuadPart;
          h_x = NULL; h_y = NULL; d_x = NULL; d_y = NULL; // Nullify pointers
          // -------------------------------------------

          // --- Stop Total Time Measurement for this single iteration ---
          // Need to get the total time *before* accumulating
          QueryPerformanceCounter(&stop_time); // Use the generic stop_time corresponding to the start_time at the beginning of the inner loop
          iter_total_ms = (double)(stop_time.QuadPart - start_time.QuadPart) * 1000.0 / cpu_frequency.QuadPart; // This stop_time corresponds to the start_time at the beginning of the inner loop

          // --- Accumulate timings for averaging ---
          total_kernel_ms_acc += iter_kernel_ms;
          total_host_alloc_ms_acc += iter_host_alloc_ms;
          total_device_alloc_ms_acc += iter_device_alloc_ms;
          total_host_init_ms_acc += iter_host_init_ms;
          total_h2d_ms_acc += iter_h2d_ms;
          total_d2h_ms_acc += iter_d2h_ms;
          total_cleanup_ms_acc += iter_cleanup_ms;
          total_total_ms_acc += iter_total_ms; // Accumulate the total time measured for this iteration
          // --------------------------------------

      } // End of inner loop (iter)

      // --- Calculate Averages ---
      double avg_kernel_ms = total_kernel_ms_acc / NUM_ITERATIONS;
      double avg_host_alloc_ms = total_host_alloc_ms_acc / NUM_ITERATIONS;
      double avg_device_alloc_ms = total_device_alloc_ms_acc / NUM_ITERATIONS;
      double avg_host_init_ms = total_host_init_ms_acc / NUM_ITERATIONS;
      double avg_h2d_ms = total_h2d_ms_acc / NUM_ITERATIONS;
      double avg_d2h_ms = total_d2h_ms_acc / NUM_ITERATIONS;
      double avg_cleanup_ms = total_cleanup_ms_acc / NUM_ITERATIONS;
      double avg_total_ms = total_total_ms_acc / NUM_ITERATIONS;
      // --------------------------

      // --- Output Averaged Results ---
      // Print averaged results to the console
      printf("%-6d | %-14.4f | %-15.4f | %-17.4f | %-14.4f | %-14.4f | %-14.4f | %-14.4f | %-12.4f\n",
             p, avg_kernel_ms, avg_host_alloc_ms, avg_device_alloc_ms, avg_host_init_ms, avg_h2d_ms, avg_d2h_ms, avg_cleanup_ms, avg_total_ms);
      // Write averaged results to the CSV file
      fprintf(results_file, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
              p, avg_kernel_ms, avg_host_alloc_ms, avg_device_alloc_ms, avg_host_init_ms, avg_h2d_ms, avg_d2h_ms, avg_cleanup_ms, avg_total_ms);
      // -----------------------------

  } // End of outer loop (p)

  // Print closing separator line to the console
  printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

  // --- Close Output File ---
  if (fclose(results_file) != 0) { // Close the CSV file
      perror("Error closing output file");
  }
  results_file = NULL; // Set file pointer to NULL
  // -------------------------

  // Destroy CUDA events
  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));

  // Print final completion message
  printf("Timing complete. Averaged results saved to %s\n", filename);

  return 0; // Indicate successful execution
}
