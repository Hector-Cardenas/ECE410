import random
import time # For observing actual execution time, optional

# Attempt to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting will be disabled.")
    print("To enable plotting, please install matplotlib: pip install matplotlib")

# --- Functions for Detailed Simulation (with history logging) ---

def systolic_bubble_sort_simulation(arr_input):
    """
    (For detailed examples)
    Simulates Odd-Even Transposition Sort and logs the history of each step.
    """
    n = len(arr_input)
    hypothetical_clock_cycles = 0
    history = [("Initial", list(arr_input), hypothetical_clock_cycles)]
    if n <= 1:
        if n == 1:
             hypothetical_clock_cycles = 1
             history.append(("Phase 1 (No operations as n=1)", list(arr_input), hypothetical_clock_cycles))
        return list(arr_input), history, hypothetical_clock_cycles
    
    data = list(arr_input)
    swapped_in_previous_phase = True
    for phase_num in range(1, n + 1):
        hypothetical_clock_cycles += 1
        swapped_in_current_phase = False
        if phase_num % 2 == 1:
            phase_description = f"Phase {phase_num} (Even Pairs Compare/Swap)"
            for i in range(0, n - 1, 2):
                if data[i] > data[i+1]:
                    data[i], data[i+1] = data[i+1], data[i]
                    swapped_in_current_phase = True
        else:
            phase_description = f"Phase {phase_num} (Odd Pairs Compare/Swap)"
            for i in range(1, n - 1, 2):
                if data[i] > data[i+1]:
                    data[i], data[i+1] = data[i+1], data[i]
                    swapped_in_current_phase = True
        history.append((phase_description, list(data), hypothetical_clock_cycles))
        if phase_num >= 2 and not swapped_in_current_phase and not swapped_in_previous_phase:
            break
        swapped_in_previous_phase = swapped_in_current_phase
    return data, history, hypothetical_clock_cycles

def naive_bubble_sort_simulation(arr_input):
    """
    (For detailed examples)
    Simulates a naive bubble sort and logs the history of each pass.
    """
    n = len(arr_input)
    sequential_clock_cycles = 0
    history = [("Initial", list(arr_input), sequential_clock_cycles)]
    data = list(arr_input)
    if n <= 1:
        if n == 1:
             history.append(("Pass 1 (No comparisons as n=1)", list(data), 0))
        return data, history, 0
    for i in range(n - 1):
        pass_description = f"Pass {i + 1}"
        swapped_in_pass = False
        for j in range(n - 1 - i):
            sequential_clock_cycles += 1
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                swapped_in_pass = True
        history.append((pass_description, list(data), sequential_clock_cycles))
        if not swapped_in_pass:
            break
    return data, history, sequential_clock_cycles

# --- Lightweight Functions for Benchmarking (no history logging) ---

def systolic_sort_benchmark(arr_input):
    """
    (For benchmarking large N)
    Performs Odd-Even Transposition Sort and returns only the final cycle count.
    This is memory-efficient as it does not store the step-by-step history.
    """
    n = len(arr_input)
    if n <= 1:
        return n # 0 cycles for n=0, 1 cycle for n=1
    data = list(arr_input)
    hypothetical_clock_cycles = 0
    swapped_in_previous_phase = True
    for phase_num in range(1, n + 1):
        hypothetical_clock_cycles += 1
        swapped_in_current_phase = False
        if phase_num % 2 == 1:
            for i in range(0, n - 1, 2):
                if data[i] > data[i+1]:
                    data[i], data[i+1] = data[i+1], data[i]
                    swapped_in_current_phase = True
        else:
            for i in range(1, n - 1, 2):
                if data[i] > data[i+1]:
                    data[i], data[i+1] = data[i+1], data[i]
                    swapped_in_current_phase = True
        if phase_num >= 2 and not swapped_in_current_phase and not swapped_in_previous_phase:
            break
        swapped_in_previous_phase = swapped_in_current_phase
    return hypothetical_clock_cycles

def naive_sort_benchmark(arr_input):
    """
    (For benchmarking large N)
    Performs naive bubble sort and returns only the final cycle (comparison) count.
    Memory-efficient as it does not store history.
    """
    n = len(arr_input)
    if n <= 1:
        return 0
    data = list(arr_input)
    sequential_clock_cycles = 0
    for i in range(n - 1):
        swapped_in_pass = False
        for j in range(n - 1 - i):
            sequential_clock_cycles += 1
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                swapped_in_pass = True
        if not swapped_in_pass:
            break
    return sequential_clock_cycles

def print_sorting_history(algorithm_name, history, total_cycles_label):
    """Helper function to print the detailed sorting history nicely."""
    print(f"\n--- {algorithm_name} Simulation Steps ---")
    total_cycles_value = 0
    if not history:
        print("No history to display.")
    else:
        for description, array_state, clock_cycles in history:
            print(f"- {description}: {array_state} (Cycles: {clock_cycles})")
        total_cycles_value = history[-1][2]
    print(f"{total_cycles_label}: {total_cycles_value}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Section 1: Detailed Example Runs ---
    # This section still uses the original functions with history logging
    print("SECTION 1: DETAILED EXAMPLE RUNS (with history logging)")
    test_arrays = {"Small Mixed": [5, 1, 4, 2], "Already Sorted": [1, 2, 3, 4, 5]}
    for name, arr_input in test_arrays.items():
        print(f"\n================ Test Case: {name} ================")
        _, s_hist, _ = systolic_bubble_sort_simulation(list(arr_input))
        print_sorting_history("Systolic Bubble Sort", s_hist, "Total Hypothetical Parallel Cycles")
        _, n_hist, _ = naive_bubble_sort_simulation(list(arr_input))
        print_sorting_history("Naive Bubble Sort", n_hist, "Total Sequential Cycles (Comparisons)")

    # --- Section 2: Benchmarking Framework ---
    # This section now uses the lightweight, memory-efficient benchmark functions
    print("\n\nSECTION 2: BENCHMARKING FRAMEWORK (memory-efficient)")
    
    n_values = [10, 100, 1000, 10000, 100000] # Increased n, 100k is possible but slow for naive
    max_random_value = 10000

    avg_systolic_cycles_list = []
    avg_naive_cycles_list = []

    print(f"\nRunning benchmarks for n = {n_values}")

    for n_size in n_values:
        # Reduce trials for larger N to keep execution time reasonable
        num_trials = 5 if n_size < 1000 else 2
        
        current_n_systolic_cycles = []
        current_n_naive_cycles = []
        print(f"\n--- Benchmarking for n = {n_size} over {num_trials} trials ---")

        for trial in range(num_trials):
            random_array = [random.randint(0, max_random_value) for _ in range(n_size)]
            
            # Use the benchmark functions that only return cycle counts
            s_cycles = systolic_sort_benchmark(list(random_array))
            current_n_systolic_cycles.append(s_cycles)

            n_cycles = naive_sort_benchmark(list(random_array))
            current_n_naive_cycles.append(n_cycles)

        avg_s = sum(current_n_systolic_cycles) / len(current_n_systolic_cycles)
        avg_systolic_cycles_list.append(avg_s)
            
        avg_n = sum(current_n_naive_cycles) / len(current_n_naive_cycles)
        avg_naive_cycles_list.append(avg_n)
        
        print(f"Average Systolic Cycles for n={n_size}: {avg_s:.2f}")
        print(f"Average Naive Cycles for n={n_size}: {avg_n:.2f}")

    # --- Section 3: Plotting Results ---
    print("\n\nSECTION 3: PLOTTING RESULTS")
    if MATPLOTLIB_AVAILABLE and n_values:
        plt.figure(figsize=(10, 6))
        plt.plot(n_values, avg_systolic_cycles_list, marker='o', linestyle='-', label='Systolic Sort (Parallel Cycles)')
        plt.plot(n_values, avg_naive_cycles_list, marker='x', linestyle='--', label='Naive Bubble Sort (Sequential Cycles)')
        
        plt.xlabel('Array Size (n)')
        plt.ylabel('Average Clock Cycles (Log Scale)')
        plt.title('Performance Comparison: Systolic vs. Naive Bubble Sort')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.yscale('log') 
        plt.xscale('log') # Use log scale for x-axis as well for better visualization

        try:
            plot_filename = "benchmark_plot.png"
            plt.savefig(plot_filename)
            print(f"\nBenchmark plot saved as {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    elif not n_values:
        print("No data to plot.")
    else:
        print("Plotting is disabled because matplotlib is not available.")
    
    print("\n--- End of Program ---")
