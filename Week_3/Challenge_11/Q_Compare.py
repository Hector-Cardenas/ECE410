# -*- coding: utf-8 -*-
"""
Q-Learning Algorithm - Sequential CPU vs. Batch GPU Comparison (Adjusted Params)

Compares CPU (NumPy) vs GPU (CuPy) training duration across varying grid sizes.
Ensures a valid path from start to win exists (BFS check).
Prevents holes immediately adjacent to the start state.
Uses decaying epsilon for exploration.
Runs multiple times per size, averages successful runs.
Includes adjusted Learning Rate and Hole Penalty.
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque # Needed for BFS path check

# --- GPU Setup ---
try:
    import cupy as cp
    from cupy.cuda import runtime as cuda_runtime
    CUPY_AVAILABLE = True
    print("CuPy found. GPU execution option available.")
    try:
        current_device_id = cuda_runtime.getDevice()
        device_properties = cuda_runtime.getDeviceProperties(current_device_id)
        gpu_name = device_properties['name']
        print(f"  Using GPU Device ID: {current_device_id}, Name: {gpu_name}")
    except Exception as e:
        print(f"  Could not retrieve GPU device info: {e}")
except ImportError:
    print("CuPy not found. Using NumPy for all operations (GPU comparison skipped).")
    cp = np # Fallback: assign np to cp alias if cupy not found
    CUPY_AVAILABLE = False
# ------------------------------

# --- Constants ---
ACTIONS = [0, 1, 2, 3] # 0 = up, 1 = down, 2 = left, 3 = right
NUM_ACTIONS = len(ACTIONS)
Q_TABLE_DTYPE_NP = np.float32 # NumPy dtype
Q_TABLE_DTYPE_CP = cp.float32 if CUPY_AVAILABLE else np.float32 # CuPy dtype

# --- Hyperparameters ---
LEARNING_RATE = 0.1   # alpha (Changed from 0.5)
DISCOUNT_FACTOR = 0.9 # gamma
# Decaying Epsilon Parameters
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.99995

NUM_RUNS_PER_SIZE = 5 # Number of runs to average over per grid size

# Training Scaling Parameters
BASE_GRID_SIZE = 5
BASE_EPISODES = 10000
BASE_EPISODES_PER_STATE = BASE_EPISODES / (BASE_GRID_SIZE * BASE_GRID_SIZE) # Approx 400
ENV_STEPS_MULTIPLIER = 50

# --- State Indexing Functions ---
def state_to_index(r, c, cols):
    return r * cols + c

def index_to_state(index, cols):
    return (index // cols, index % cols)

# --- Helper Functions ---
def generate_holes(rows, cols, num_holes, start_pos, win_pos):
    """Generates random hole positions, avoiding start/win states and start neighbors."""
    possible_coords = set((r, c) for r in range(rows) for c in range(cols))
    possible_coords.discard(start_pos)
    possible_coords.discard(win_pos)
    start_r, start_c = start_pos
    neighbors_to_exclude = {
        (start_r + 1, start_c), (start_r - 1, start_c),
        (start_r, start_c + 1), (start_r, start_c - 1)
    }
    for neighbor in neighbors_to_exclude:
        possible_coords.discard(neighbor)
    num_holes = min(num_holes, len(possible_coords))
    if num_holes < 0: num_holes = 0
    return random.sample(list(possible_coords), num_holes)

def is_path_possible(rows, cols, start_pos, win_pos, hole_states_set):
    """Checks if win_pos is reachable from start_pos using BFS, avoiding holes."""
    queue = deque([start_pos])
    visited = {start_pos}
    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        r, c = queue.popleft()
        if (r, c) == win_pos: return True
        for dr, dc in possible_moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor = (nr, nc)
                if neighbor not in visited and neighbor not in hole_states_set:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return False

# --- Environment Class ---
class GridEnvironment():
    def __init__(self, rows, cols, start_pos, win_pos, hole_states):
        self.rows = rows
        self.cols = cols
        self.start_pos = start_pos
        self.win_pos = win_pos
        self.hole_states = set(hole_states)
        self.state = self.start_pos
        self.done = self._check_done()

    def _check_done(self):
        return (self.state == self.win_pos) or (self.state in self.hole_states)

    def get_reward(self):
        if self.state == self.win_pos: return 1.0
        # --- Use stronger penalty ---
        elif self.state in self.hole_states: return -20.0 # Changed from -5.0
        # --------------------------
        else: return -1.0

    def step(self, action):
        if self.done: return self.state, 0.0, self.done
        r, c = self.state
        if action == 0: next_state_rc = (r - 1, c)
        elif action == 1: next_state_rc = (r + 1, c)
        elif action == 2: next_state_rc = (r, c - 1)
        else: next_state_rc = (r, c + 1)
        nr, nc = next_state_rc
        if (nr < 0) or (nr >= self.rows) or (nc < 0) or (nc >= self.cols):
            next_state_rc = self.state
        self.state = next_state_rc
        self.done = self._check_done()
        reward = self.get_reward()
        return self.state, reward, self.done

    def reset(self):
        self.state = self.start_pos
        self.done = self._check_done()
        return self.state

# === Approach 1: Sequential CPU Agent (Optimized - 3D Q-Table) ===
class SequentialAgentCPU():
    def __init__(self, rows, cols, start_pos, win_pos, hole_states):
        self.rows = rows
        self.cols = cols
        self.win_pos = win_pos
        self.env = GridEnvironment(rows, cols, start_pos, win_pos, hole_states)
        self.lr = LEARNING_RATE # Use updated constant
        self.epsilon = EPSILON_START
        self.decay_gamma = DISCOUNT_FACTOR
        self.Q_table = np.zeros((self.rows, self.cols, NUM_ACTIONS), dtype=Q_TABLE_DTYPE_NP)

    def choose_action(self, state_rc, deterministic=False):
        # ... (choose_action logic remains the same) ...
        action = -1
        r, c = state_rc
        if not deterministic and np.random.uniform(0, 1) <= self.epsilon:
            action = np.random.choice(ACTIONS)
        else:
            current_q_values = self.Q_table[r, c, :]
            max_q = np.nanmax(current_q_values)
            if np.isinf(max_q) or np.isnan(max_q):
                 action = np.random.choice(ACTIONS)
            else:
                best_actions_indices = np.where(np.isclose(current_q_values, max_q))[0]
                if best_actions_indices.size == 0:
                     best_actions_indices = [np.argmax(current_q_values)]
                action = best_actions_indices[0] if deterministic else np.random.choice(best_actions_indices)
        action = int(action)
        if not (0 <= action < NUM_ACTIONS): action = 0
        return action


    def learn(self, episodes):
        # ... (learn logic remains the same, uses self.lr) ...
        max_steps_per_episode = max(500, self.rows * self.cols * 2)
        print(f"      Training for {episodes:,} episodes (max steps: {max_steps_per_episode})...")
        for i in range(episodes):
            state_rc = self.env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode:
                action = self.choose_action(state_rc, deterministic=False)
                next_state_rc, reward, done = self.env.step(action)
                r, c = state_rc
                next_r, next_c = next_state_rc
                old_q = self.Q_table[r, c, action]
                next_max_q = 0.0 if done else np.nanmax(self.Q_table[next_r, next_c, :])
                if np.isnan(next_max_q) or np.isinf(next_max_q): next_max_q = 0.0
                new_q = old_q + self.lr * (reward + self.decay_gamma * next_max_q - old_q)
                self.Q_table[r, c, action] = new_q
                state_rc = next_state_rc
                steps += 1
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY_RATE)


    def simulate_policy(self, max_steps=None):
        # ... (simulate_policy remains the same) ...
        if max_steps is None: max_steps = self.rows * self.cols * 2
        print(f"    Simulating policy for {type(self).__name__}...")
        state_rc = self.env.reset()
        steps = 0
        done = False
        while not done and steps < max_steps:
            action = self.choose_action(state_rc, deterministic=True)
            state_rc, _, done = self.env.step(action)
            steps += 1
            if state_rc == self.win_pos and done:
                print(f"      Policy reached WIN state in {steps} steps.")
                return True
            if done:
                 print(f"      Policy hit a HOLE state after {steps} steps.")
                 return False
        if state_rc == self.win_pos:
             print(f"      Policy reached WIN state in {steps} steps.")
             return True
        else:
             print(f"      Policy did NOT reach WIN state within {max_steps} steps.")
             return False


# === Approach 2: Batch GPU Agent ===

# Class to manage parallel environments (simulated on CPU)
class ParallelEnvManager():
    # ... (No changes needed here) ...
    def __init__(self, num_envs, rows, cols, start_pos, win_pos, hole_states):
        self.num_envs = num_envs
        self.rows = rows
        self.cols = cols
        self.envs = [GridEnvironment(rows, cols, start_pos, win_pos, hole_states) for _ in range(num_envs)]
        self.current_states_rc = [env.state for env in self.envs]

    def get_states_indices(self):
        return np.array([state_to_index(r, c, self.cols) for r, c in self.current_states_rc], dtype=np.int32)

    def batch_step(self, actions):
        next_states_rc = [None] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=Q_TABLE_DTYPE_NP)
        dones = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            action_i = int(actions[i])
            if not (0 <= action_i < NUM_ACTIONS): action_i = 0
            next_s_rc, r, d = self.envs[i].step(action_i)
            next_states_rc[i] = next_s_rc
            rewards[i] = r
            dones[i] = d
            self.current_states_rc[i] = self.envs[i].state
            if d: self.current_states_rc[i] = self.envs[i].reset()
        next_states_indices = np.array([state_to_index(r, c, self.cols) for r, c in next_states_rc], dtype=np.int32)
        return next_states_indices, rewards, dones

# Class for the agent logic using batch processing on GPU
class BatchAgentGPU():
    def __init__(self, rows, cols, start_pos, win_pos, hole_states):
        self.rows = rows
        self.cols = cols
        self.num_states = rows * cols
        self.start_pos = start_pos
        self.win_pos = win_pos
        self.hole_states = hole_states
        self.xp = cp if CUPY_AVAILABLE else np
        self.q_dtype = Q_TABLE_DTYPE_CP if CUPY_AVAILABLE else Q_TABLE_DTYPE_NP
        self.lr = LEARNING_RATE # Use updated constant
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START # Use decaying epsilon
        self.Q_table = self.xp.zeros((self.num_states, NUM_ACTIONS), dtype=self.q_dtype)

    def batch_choose_action(self, states_indices_gpu):
        # ... (logic uses self.epsilon, remains the same) ...
        batch_size = states_indices_gpu.shape[0]
        q_values = self.Q_table[states_indices_gpu, :]
        best_actions = self.xp.argmax(q_values, axis=1).astype(self.xp.int32)
        rand_nums = self.xp.random.rand(batch_size)
        random_actions = self.xp.random.randint(0, NUM_ACTIONS, size=batch_size, dtype=self.xp.int32)
        actions = self.xp.where(rand_nums < self.epsilon, random_actions, best_actions)
        return actions

    def batch_update(self, s_gpu, a_gpu, r_gpu, s_prime_gpu, done_gpu):
        # ... (logic uses self.lr and self.gamma, remains the same) ...
        q_s_a = self.Q_table[s_gpu, a_gpu]
        q_s_prime_all = self.Q_table[s_prime_gpu, :]
        next_max_q = self.xp.max(q_s_prime_all, axis=1)
        next_max_q = next_max_q * (1 - done_gpu.astype(self.q_dtype))
        target_q = r_gpu + self.gamma * next_max_q
        new_q_s_a = q_s_a + self.lr * (target_q - q_s_a)
        self.Q_table[s_gpu, a_gpu] = new_q_s_a

    def learn(self, total_env_steps, batch_size):
        # ... (logic remains the same, but epsilon decay added) ...
        env_manager = ParallelEnvManager(batch_size, self.rows, self.cols, self.start_pos, self.win_pos, self.hole_states)
        num_batch_updates = total_env_steps // batch_size
        if num_batch_updates == 0: num_batch_updates = 1
        print(f"      Training for ~{total_env_steps:,} total env steps ({num_batch_updates:,} batch updates)...")
        for i in range(num_batch_updates):
            s_indices_np = env_manager.get_states_indices()
            s_indices_gpu = self.xp.asarray(s_indices_np)
            actions_gpu = self.batch_choose_action(s_indices_gpu)
            actions_np = cp.asnumpy(actions_gpu) if self.xp == cp else actions_gpu
            next_s_indices_np, rewards_np, dones_np = env_manager.batch_step(actions_np)
            s_gpu = s_indices_gpu
            a_gpu = actions_gpu
            r_gpu = self.xp.asarray(rewards_np, dtype=self.q_dtype)
            s_prime_gpu = self.xp.asarray(next_s_indices_np)
            done_gpu = self.xp.asarray(dones_np)
            self.batch_update(s_gpu, a_gpu, r_gpu, s_prime_gpu, done_gpu)
            # Decay epsilon after each batch update
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY_RATE)

    def check_policy(self):
        # ... (check_policy remains the same) ...
        print(f"    Checking policy for {type(self).__name__}...")
        q_table_cpu = cp.asnumpy(self.Q_table) if self.xp == cp else self.Q_table
        temp_agent = SequentialAgentCPU(self.rows, self.cols, self.start_pos, self.win_pos, self.hole_states)
        if q_table_cpu.ndim == 2 and q_table_cpu.shape[0] == self.num_states:
             try:
                 q_table_cpu_3d = q_table_cpu.reshape((self.rows, self.cols, NUM_ACTIONS))
                 temp_agent.Q_table = q_table_cpu_3d
                 return temp_agent.simulate_policy()
             except ValueError as e:
                  print(f"      Error reshaping Q-table for policy check: {e}")
                  return False
        else:
             print(f"      Error: Q-table has unexpected shape {q_table_cpu.shape} for policy check.")
             return False


# --- Main Execution Block ---
if __name__ == '__main__':
    # ... (Setup and loop structure remain the same) ...
    print("--- Starting Q-Learning Sequential CPU vs Batch GPU Benchmark (Averaged Runs, Scaled Training, Safer Holes, Decaying Epsilon, Adjusted Params) ---") # Updated print

    # --- Simulation Parameters ---
    grid_sizes_to_test = [5, 10, 20, 30, 40, 50]
    batch_size_gpu = 128
    num_holes_percent = 0.16
    num_runs = NUM_RUNS_PER_SIZE

    cpu_avg_times = []
    gpu_avg_times = []
    actual_sizes = []

    # --- Loop through Grid Sizes ---
    for size in grid_sizes_to_test:
        print(f"\n--- Testing Grid Size: {size}x{size} ---")
        actual_sizes.append(size)
        rows, cols = size, size
        start_pos = (0, 0)
        win_pos = (rows - 1, cols - 1)
        num_states = rows * cols
        num_holes = int((num_states - 2) * num_holes_percent)

        # Calculate scaled training duration
        episodes_cpu = int(BASE_EPISODES_PER_STATE * num_states)
        episodes_cpu = max(BASE_EPISODES, episodes_cpu) # Ensure minimum episodes
        total_env_steps_gpu = episodes_cpu * ENV_STEPS_MULTIPLIER
        print(f"  Target CPU episodes: {episodes_cpu:,}")
        print(f"  Target GPU env steps: ~{total_env_steps_gpu:,}")

        successful_cpu_times = []
        successful_gpu_times = []

        # --- Loop for Averaging Runs ---
        for run_num in range(num_runs):
            print(f"\n  Run {run_num + 1}/{num_runs} for size {size}x{size}...")
            seed = size * 1000 + run_num
            random.seed(seed)
            np.random.seed(seed)
            if CUPY_AVAILABLE: cp.random.seed(seed)

            # Generate Holes with Path Check & Start Neighbor Check
            holes_generated = False
            regeneration_attempts = 0
            max_regeneration_attempts = 100
            while not holes_generated and regeneration_attempts < max_regeneration_attempts:
                hole_states_list = generate_holes(rows, cols, num_holes, start_pos, win_pos)
                hole_states_set = set(hole_states_list)
                if is_path_possible(rows, cols, start_pos, win_pos, hole_states_set):
                    holes_generated = True
                    print(f"    Generated {len(hole_states_list)} hole states with seed {seed} (Path found, Start neighbors clear).")
                else:
                    regeneration_attempts += 1
                    if regeneration_attempts % 10 == 0:
                         print(f"    Regenerating holes (Attempt {regeneration_attempts})... Path not found.")

            if not holes_generated:
                 print(f"    FAILED to generate valid hole configuration after {max_regeneration_attempts} attempts. Skipping run {run_num + 1}.")
                 continue

            cpu_run_success = False
            gpu_run_success = False

            # --- CPU Run ---
            print("\n    Running Sequential CPU...")
            agent_cpu = SequentialAgentCPU(rows, cols, start_pos, win_pos, hole_states_list)
            print(f"      Q-Table using {type(agent_cpu.Q_table)} with shape {agent_cpu.Q_table.shape}")
            cpu_start_time = time.time()
            agent_cpu.learn(episodes=episodes_cpu)
            cpu_end_time = time.time()
            cpu_duration = cpu_end_time - cpu_start_time
            print(f"    Sequential CPU Duration: {cpu_duration:.4f} seconds")
            cpu_run_success = agent_cpu.simulate_policy()
            if cpu_run_success:
                successful_cpu_times.append(cpu_duration)

            # --- GPU Run ---
            if CUPY_AVAILABLE:
                print("\n    Running Batch GPU...")
                agent_gpu = BatchAgentGPU(rows, cols, start_pos, win_pos, hole_states_list)
                print(f"      Q-Table using {type(agent_gpu.Q_table)} with shape {agent_gpu.Q_table.shape}")
                gpu_start_time = time.time()
                agent_gpu.learn(total_env_steps=total_env_steps_gpu, batch_size=batch_size_gpu)
                gpu_end_time = time.time()
                gpu_duration = gpu_end_time - gpu_start_time
                print(f"    Batch GPU Duration: {gpu_duration:.4f} seconds")
                gpu_run_success = agent_gpu.check_policy()
                if gpu_run_success:
                    successful_gpu_times.append(gpu_duration)

        # --- Averaging ---
        avg_cpu = np.mean(successful_cpu_times) if successful_cpu_times else np.nan
        avg_gpu = np.mean(successful_gpu_times) if successful_gpu_times else np.nan
        cpu_avg_times.append(avg_cpu)
        gpu_avg_times.append(avg_gpu)
        print(f"\n  Average Successful Times for {size}x{size} after {num_runs} runs:")
        print(f"    CPU: {avg_cpu:.4f} sec ({len(successful_cpu_times)}/{num_runs} successful)")
        if CUPY_AVAILABLE:
            print(f"    GPU: {avg_gpu:.4f} sec ({len(successful_gpu_times)}/{num_runs} successful)")
        else:
             print(f"    GPU: Skipped")


    # --- Plotting ---
    # ... (Plotting code remains the same) ...
    print("\n--- Plotting Benchmark Results ---")
    plt.figure(figsize=(10, 6))
    plt.plot(actual_sizes, cpu_avg_times, marker='o', linestyle='-', label='CPU (Sequential Avg)')
    if CUPY_AVAILABLE and any(not np.isnan(t) for t in gpu_avg_times):
        plt.plot(actual_sizes, gpu_avg_times, marker='s', linestyle='--', label='GPU (Batch Update Avg)')

    plt.xlabel("Grid Size (N) for N x N grid")
    plt.ylabel(f"Avg Training Duration (seconds over up to {NUM_RUNS_PER_SIZE} successful runs)")
    plt.title(f"Q-Learning Avg Training Time vs. Grid Size (Scaled Training, Safer Holes, Decaying Epsilon, LR={LEARNING_RATE})") # Updated Title
    plt.xticks(actual_sizes)
    failed_cpu_indices = [i for i, t in enumerate(cpu_avg_times) if np.isnan(t)]
    if failed_cpu_indices:
        plt.plot(np.array(actual_sizes)[failed_cpu_indices], np.zeros(len(failed_cpu_indices)),
                 marker='x', color='red', linestyle='None', markersize=10, label='CPU Failed Policy Check')

    failed_gpu_indices = [i for i, t in enumerate(gpu_avg_times) if np.isnan(t)]
    if failed_gpu_indices and CUPY_AVAILABLE:
         plot_gpu_fails = [i for i in failed_gpu_indices if i not in failed_cpu_indices]
         if plot_gpu_fails:
              plt.plot(np.array(actual_sizes)[plot_gpu_fails], np.zeros(len(plot_gpu_fails)),
                       marker='x', color='orange', linestyle='None', markersize=10, label='GPU Failed Policy Check')

    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    min_val = min([t for t in cpu_avg_times + gpu_avg_times if not np.isnan(t) and t > 0] or [0.01])
    plt.ylim(bottom=min_val * 0.1 if min_val > 0.1 else 0.01)
    plt.show()

    print("\n--- Script Finished ---")
