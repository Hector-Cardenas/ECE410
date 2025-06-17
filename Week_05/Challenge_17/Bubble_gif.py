import random
import os
import shutil

# --- Dependency Check and User Guidance ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import imageio.v2 as imageio
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("--- Required Libraries Not Found ---")
    print("This script requires 'matplotlib' and 'imageio' to run.")
    print("Please install them by running the following command in your terminal:")
    print("pip install matplotlib imageio")
    print("------------------------------------")


def systolic_sort_step_generator(arr_input):
    """
    A generator that yields the state of the array after each phase of the
    Odd-Even Transposition sort, along with the indices being compared.
    
    Args:
        arr_input (list): The list of numbers to sort.
        
    Yields:
        tuple: A tuple containing (current_array_state, list_of_compared_indices, phase_description).
    """
    n = len(arr_input)
    data = list(arr_input)
    
    # Yield the initial state
    yield list(data), [], "Initial Array"
    
    if n <= 1:
        yield list(data), [], "Already Sorted"
        return

    is_sorted = False
    phase_num = 0
    while not is_sorted:
        phase_num += 1
        is_sorted = True
        compared_indices = []

        # Determine which phase to run (odd or even pairs)
        # This alternates on each outer loop iteration
        is_odd_phase = (phase_num % 2 == 1)
        
        start_index = 1 if is_odd_phase else 0
        phase_description = f"Phase {phase_num}: Odd Pairs" if is_odd_phase else f"Phase {phase_num}: Even Pairs"
        
        # Perform comparisons for the current phase
        for i in range(start_index, n - 1, 2):
            compared_indices.extend([i, i+1])
            if data[i] > data[i+1]:
                data[i], data[i+1] = data[i+1], data[i]
                is_sorted = False # A swap occurred, so not sorted yet

        yield list(data), list(compared_indices), phase_description
        
        # Simple optimization: if two full phases (odd and even) have no swaps, we are done.
        # This is a simplified check for the generator.
        # A more robust check would track swaps over two consecutive phases.
        if is_sorted and phase_num > 1:
            # If a phase had no swaps, run one more to be sure, then break if still no swaps.
            # This logic is simplified for the animation generator. A final check is sufficient.
            pass


def create_animation_frame(array_data, compared_indices, title, frame_path):
    """
    Creates and saves a single frame (a bar chart) for the animation.
    
    Args:
        array_data (list): The current state of the array.
        compared_indices (list): Indices of elements being compared, to be highlighted.
        title (str): The title for the plot frame.
        frame_path (str): The file path to save the generated image.
    """
    n = len(array_data)
    # Assign colors: a default color and a list of highlight colors for pairs.
    colors = ['#4A90E2'] * n  # A nice blue for default bars
    pair_colors = ['#F5A623', '#7ED321', '#E24A4A', '#9013FE', '#4AE2D4'] # orange, green, red, purple, teal
    
    # Iterate through the compared indices in pairs
    for i in range(0, len(compared_indices), 2):
        # Get a color for the pair, cycling through the palette
        color = pair_colors[(i // 2) % len(pair_colors)]
        idx1 = compared_indices[i]
        idx2 = compared_indices[i+1]
        if idx1 < n:
            colors[idx1] = color
        if idx2 < n:
            colors[idx2] = color
            
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the bar chart
    bars = ax.bar(range(n), array_data, color=colors)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Array Index", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_xticks(range(n)) # Ensure all integer ticks are shown for indices
    
    # Ensure y-axis ticks are integers if all values are integers
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set a consistent y-axis limit for all frames for a stable animation
    # Find the maximum value in the initial array to set a constant y-limit
    initial_max_val = max(array_data) if array_data else 10
    ax.set_ylim(0, initial_max_val * 1.1 + 1)
    
    # Add the value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(frame_path)
    plt.close(fig)


def create_sort_gif(array_to_sort, gif_filename="systolic_sort_animation.gif"):
    """
    Main function to generate the frames and compile them into a GIF.
    """
    if not DEPS_AVAILABLE:
        print("Cannot create GIF because required libraries are not installed.")
        return

    # Create a temporary directory to store image frames
    temp_dir = "temp_gif_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"Generating frames for the animation... (Array: {array_to_sort})")
    
    filenames = []
    # Use the generator to get each step of the sorting process
    for i, (data, compared, description) in enumerate(systolic_sort_step_generator(list(array_to_sort))):
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        create_animation_frame(data, compared, description, frame_path)
        filenames.append(frame_path)
    
    # Add a final "Sorted!" frame
    final_data = list(array_to_sort)
    final_data.sort()
    frame_path = os.path.join(temp_dir, f"frame_{len(filenames):03d}.png")
    create_animation_frame(final_data, [], "Sorted!", frame_path)
    filenames.append(frame_path)
    
    print("Compiling frames into a GIF...")
    
    # Read the generated images and compile them into a GIF
    images = [imageio.imread(filename) for filename in filenames]
    # Use a longer duration for the first and last frames
    durations = [1.5] + [0.8] * (len(images) - 2) + [2.0]
    imageio.mimsave(gif_filename, images, duration=durations, loop=0)
    
    print(f"Successfully created animation: {gif_filename}")
    
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define a smallish array to be sorted for the animation
    # You can change this to see how it works with different data
    example_array = random.sample(range(1, 50), 10)
    
    create_sort_gif(example_array)
