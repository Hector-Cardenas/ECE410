# -*- coding: utf-8 -*-
"""
Q-Learning Algorithm - GPU Execution Test (5x5 Grid)
with Minor GPU Optimizations and cp.random.choice fix

Runs the Q-Learning algorithm using CuPy (GPU) if available,
otherwise falls back to NumPy (CPU), for a fixed 5x5 grid.
Times the training duration. Includes minor GPU optimizations
(reduced data transfer in chooseAction, float32 dtype).

Note: GPU acceleration is still unlikely to be beneficial for this
      tabular Q-learning algorithm due to overheads.
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import random # Still needed if using np.random.choice

# --- GPU Setup ---
# Import CuPy if available, otherwise fall back to NumPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy found. GPU execution option available.")
except ImportError:
    print("CuPy not found. GPU execution will use NumPy instead.")
    cp = np # Fallback: assign np to cp alias if cupy not found
    CUPY_AVAILABLE = False
# ------------------------------

# --- Constants ---
BOARD_ROWS = 5
BOARD_COLS = 5
START = (0, 0)
WIN_STATE = (4, 4)
# Original hole states (from GitHub version)
HOLE_STATES = [(1,0),(3,1),(4,2),(1,3)]
# Actions
ACTIONS = [0, 1, 2, 3] # 0 = up, 1 = down, 2 = left, 3 = right
NUM_ACTIONS = len(ACTIONS)
# Simulation parameters
EPISODES = 10000
# Data type for Q-table (float32 often better for GPU)
Q_TABLE_DTYPE = np.float32

# --- Environment Class ---
# Simplified to use constants
class State():
    def __init__(self, state=START):
        self.state = state
        self.isEnd = False
        self.checkEnd()

    def giveReward(self):
        if self.state == WIN_STATE: return 1
        elif self.state in HOLE_STATES: return -5
        else: return -1

    def checkEnd(self):
        if (self.state == WIN_STATE) or (self.state in HOLE_STATES):
            self.isEnd = True

    def updateState(self, action):
        r, c = self.state
        if action == 0: next_state = (r - 1, c)
        elif action == 1: next_state = (r + 1, c)
        elif action == 2: next_state = (r, c - 1)
        else: next_state = (r, c + 1) # action == 3

        nr, nc = next_state
        if (nr < 0) or (nr >= BOARD_ROWS) or (nc < 0) or (nc >= BOARD_COLS):
            next_state = self.state # Stay in bounds
        self.state = next_state
        self.checkEnd()

    def Reset(self):
        self.state = START
        self.isEnd = False
        return self.state

# --- Agent Class ---
# Simplified init, determines GPU use internally
class Agent():
    def __init__(self):
        self.actions = ACTIONS
        self.State = State()

        # Learning parameters
        self.lr = 0.5        # alpha
        self.exp_rate = 0.1  # epsilon
        self.decay_gamma = 0.9 # gamma

        # Select array library (NumPy or CuPy)
        self.use_gpu = CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np # Use cp if available, else np
        # Ensure Q_TABLE_DTYPE is compatible with the library
        self.q_dtype = self.xp.float32 if hasattr(self.xp, 'float32') else np.float32

        print(f"Agent using: {'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU fallback)'}")

        # Initialise Q-table using the selected library (xp) and dtype
        self.Q_table = self.xp.zeros((BOARD_ROWS, BOARD_COLS, NUM_ACTIONS), dtype=self.q_dtype)
        print(f"  Q-Table initialized with shape: {self.Q_table.shape}, dtype: {self.Q_table.dtype}, using {type(self.Q_table)}")

    def chooseAction(self):
        action = -1
        current_r, current_c = self.State.state

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions) # Exploration uses CPU random
        else:
            # Exploitation uses xp (np or cp)
            current_q_values = self.Q_table[current_r, current_c, :]
            max_q = self.xp.max(current_q_values)
            best_actions_indices = self.xp.where(current_q_values == max_q)[0]

            if best_actions_indices.size == 0:
                 # If no best action found (e.g., all Q-values are same/invalid), explore
                 action = np.random.choice(self.actions)
            elif self.use_gpu:
                 # OPTIMIZATION: Use cp.random.choice on GPU array, get single scalar back
                 # FIX: Added size=1 argument for CuPy compatibility
                 action = cp.random.choice(best_actions_indices, size=1).item()
            else: # if using NumPy
                 action = np.random.choice(best_actions_indices)
        return action

    def takeAction(self, action):
        self.State.updateState(action)
        return self.State.giveReward()

    def Q_Learning(self, episodes=EPISODES):
        rewards_list = []
        print(f"  Training started ({'GPU' if self.use_gpu else 'CPU fallback'})...")
        q_table_device = self.Q_table # Array is already on correct device

        for i in range(episodes):
            # Optional progress print
            # if i % (episodes // 10) == 0 and i != 0: print(f"    Episode: {i}")

            _ = self.State.Reset()
            total_reward = 0
            steps = 0
            max_steps = max(500, BOARD_ROWS * BOARD_COLS * 2)

            while not self.State.isEnd:
                action = self.chooseAction()
                prev_r, prev_c = self.State.state
                reward = self.takeAction(action)
                total_reward += reward
                next_r, next_c = self.State.state

                # Get values, converting CuPy scalars to Python floats if needed using .item()
                old_q_value = q_table_device[prev_r, prev_c, action]
                if self.use_gpu: old_q_value = old_q_value.item()

                if self.State.isEnd:
                    next_max_q = 0.0 # Use float
                else:
                    next_max_q = self.xp.max(q_table_device[next_r, next_c, :])
                    if self.use_gpu: next_max_q = next_max_q.item()

                # Calculation uses Python floats
                new_q_value = old_q_value + self.lr * (reward + self.decay_gamma * next_max_q - old_q_value)

                # Update Q-table on the device (assigning Python float should cast to table's dtype)
                q_table_device[prev_r, prev_c, action] = new_q_value

                steps += 1
                if steps > max_steps: break

            rewards_list.append(total_reward)

        print(f"  Training finished ({'GPU' if self.use_gpu else 'CPU fallback'}).")
        return rewards_list

    def plot_rewards(self, rewards):
        rewards_np = np.array(rewards)
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_np)
        plt.title(f'Total Reward per Episode ({BOARD_ROWS}x{BOARD_COLS} Grid)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        try:
            if len(rewards_np) >= 100:
                moving_avg = np.convolve(rewards_np, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(len(moving_avg)) + 99, moving_avg, label='100-episode Moving Average', color='red')
                plt.legend()
        except ValueError: pass
        plt.grid(True)
        plt.show()

    def showValues(self):
        device_str = 'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy fallback)'
        print(f"\nMax Q-Value for each State ({BOARD_ROWS}x{BOARD_COLS} Grid - {device_str} Run):")
        # Ensure Q_table is on CPU (NumPy) for printing iteration
        q_table_cpu = cp.asnumpy(self.Q_table) if self.use_gpu else self.Q_table

        for r in range(0, BOARD_ROWS):
            print('-----------------------------------------------')
            out = '| '
            for c in range(0, BOARD_COLS):
                state = (r, c)
                if state == WIN_STATE: mx_nxt_value_str = " R:+1 "
                elif state in HOLE_STATES: mx_nxt_value_str = " R:-5 "
                else:
                    # Use np.max on the CPU array copy
                    max_q_value = np.max(q_table_cpu[r, c, :])
                    mx_nxt_value_str = f"{max_q_value:6.2f}"
                out += mx_nxt_value_str + ' | '
            print(out)
        print('-----------------------------------------------')


# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- Starting Q-Learning GPU Test (5x5 Grid) ---")

    # --- Initialize Agent (will use GPU if available) ---
    agent = Agent() # Simplified init

    # --- Run and Time Training ---
    # This will use GPU if CuPy is installed, otherwise NumPy
    print("\n--- Starting Training ---")
    train_start_time = time.time()
    total_rewards = agent.Q_Learning(episodes=EPISODES)
    train_end_time = time.time()
    # Ensure agent object exists before printing duration
    device_str = 'GPU (CuPy)' if agent.use_gpu else 'CPU (NumPy fallback)'
    print(f"Training Duration ({device_str}): {train_end_time - train_start_time:.4f} seconds")


    # --- Plotting ---
    print("\n--- Plotting Results ---")
    agent.plot_rewards(total_rewards)

    # --- Displaying Max Q-Values ---
    print("\n--- Displaying Max Q-Values per State ---")
    agent.showValues()

    print("\n--- Script Finished ---")

