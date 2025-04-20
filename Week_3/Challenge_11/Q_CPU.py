# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:59:23 2020
Assignment 2 - Agents and Reinforcement Learning
Q-Learning Algorithm - Based on GitHub version, Training Benchmark Only

@author: Ronan Murphy - 15397831 (Original Author)
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import time # Import time module for benchmarking

# --- Constants based on GitHub version ---
BOARD_ROWS = 5
BOARD_COLS = 5
START = (0, 0)
WIN_STATE = (4, 4)
# Hole states from GitHub version
HOLE_STATES = [(1,0),(3,1),(4,2),(1,3)]
# Actions
ACTIONS = [0, 1, 2, 3] # 0 = up, 1 = down, 2 = left, 3 = right

# Create the environment using a class
class State():

    def __init__(self, state=START):
        # Initialise the state
        self.state = state
        # Initialise the determining factor
        self.isEnd = False
        # Check if initial state is end state
        self.checkEnd()

    # Give reward function (based on GitHub version)
    def giveReward(self):
        if self.state == WIN_STATE:
            return 1 # Reward for reaching the win state
        elif self.state in HOLE_STATES:
            return -5 # Penalty for falling into a hole
        else:
            return -1 # Penalty for each step taken

    # Check if the game has ended function (using HOLE_STATES list)
    def checkEnd(self):
        if (self.state == WIN_STATE) or (self.state in HOLE_STATES):
            self.isEnd = True

    # Update the state function (using logic similar to previous working version)
    def updateState(self, action):
        # Action 0 = up, 1 = down, 2 = left, 3 = right
        r, c = self.state
        if action == 0: # Up
            next_state = (r - 1, c)
        elif action == 1: # Down
            next_state = (r + 1, c)
        elif action == 2: # Left
            next_state = (r, c - 1)
        else: # Right (action == 3)
            next_state = (r, c + 1)

        # Check if the next state is off the board
        nr, nc = next_state
        if (nr < 0) or (nr >= BOARD_ROWS) or \
           (nc < 0) or (nc >= BOARD_COLS):
            next_state = self.state # Stay in the current state if move is invalid

        self.state = next_state # Update the current state

        # Check if the new state ends the game
        self.checkEnd()

    # Reset the environment function
    def Reset(self):
        self.state = START # Reset state to start
        self.isEnd = False # Reset end game flag
        return self.state

# Create the agent using a class
class Agent():

    def __init__(self):
        # Initialise the states value estimates
        self.actions = ACTIONS # Use global actions list
        self.State = State() # The environment

        # Initialise the learning parameters (based on GitHub version)
        self.lr = 0.5        # Learning rate (alpha)
        self.exp_rate = 0.1  # Exploration rate (epsilon)
        self.decay_gamma = 0.9 # Discount factor (gamma)

        # Initialise the Q-table (using dictionary mapping state tuple to numpy array of action values)
        # This structure is kept from the benchmarked version for compatibility
        self.Q_values = {} # {(row, col): [q_up, q_down, q_left, q_right]}
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                 self.Q_values[(r,c)] = np.zeros(len(self.actions)) # Initialize Q-values for each state to zeros

    # Choose action function (epsilon-greedy)
    def chooseAction(self):
        # Epsilon-greedy strategy
        action = -1 # Initialize action

        # Exploration vs Exploitation
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions) # Explore: choose a random action
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            current_q_values = self.Q_values[self.State.state]
            # Find the maximum Q-value
            max_q = np.max(current_q_values)
             # Find all actions that have the maximum Q-value
            best_actions = [a for a, q in enumerate(current_q_values) if q == max_q]
            # If multiple actions have the same max value, choose randomly among them
            action = np.random.choice(best_actions)

        return action

    # Take action function
    def takeAction(self, action):
        # Update the state based on the chosen action
        self.State.updateState(action)
        # Return the reward for the new state
        return self.State.giveReward()

    # Q-Learning update function
    def Q_Learning(self, episodes=10000):
        # Run the Q-learning algorithm for a number of episodes
        rewards = [] # List to store total reward per episode
        print("Training started...")
        for i in range(episodes):
            if i % 1000 == 0 and i != 0: # Print progress every 1000 episodes
                print(f"  Episode: {i}")

            current_state_obj = self.State.Reset() # Reset environment for new episode
            total_reward = 0 # Reset total reward for the episode
            steps = 0 # Count steps in the episode

            while not self.State.isEnd:
                # Choose an action based on the current state
                action = self.chooseAction()

                # Store the current state before taking the action
                prev_state = self.State.state # Get tuple (r, c)

                # Take the action and get the reward
                reward = self.takeAction(action)
                total_reward += reward

                # Get the next state
                next_state = self.State.state # Get tuple (r, c)

                # Update the Q-value for the previous state and action
                old_q_value = self.Q_values[prev_state][action]

                # Handle terminal state update - Q(terminal, *) should ideally be 0
                if self.State.isEnd:
                    next_max_q = 0 # No future reward from terminal state
                else:
                    next_max_q = np.max(self.Q_values[next_state]) # Max Q-value for the next state

                # Q-learning formula
                new_q_value = old_q_value + self.lr * (reward + self.decay_gamma * next_max_q - old_q_value)
                self.Q_values[prev_state][action] = new_q_value

                steps += 1
                # Safety break to prevent infinite loops in case of issues
                if steps > 500:
                    # print(f"Warning: Episode {i} exceeded max steps, breaking.")
                    break # Break inner loop, will proceed to next episode

            rewards.append(total_reward) # Store total reward for this episode

        print(f"Training finished after {episodes} episodes.")
        return rewards

    # Function to plot rewards
    def plot_rewards(self, rewards):
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title('Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        # Calculate and plot a moving average to see the trend better
        try: # Add try-except for potential issues with short reward lists
            if len(rewards) >= 100:
                moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(len(moving_avg)) + 99, moving_avg, label='100-episode Moving Average', color='red')
                plt.legend()
            else:
                 print("Not enough episodes to plot 100-episode moving average.")
        except ValueError:
            print("Could not plot moving average (calculation error?).")
        plt.grid(True)
        plt.show()

    # Function to display max Q-values per state (based on GitHub version's showValues)
    def showValues(self):
        print("\nMax Q-Value for each State (V(s) approximation):")
        for r in range(0, BOARD_ROWS):
            print('-----------------------------------------------')
            out = '| '
            for c in range(0, BOARD_COLS):
                state = (r, c)
                # For terminal states, display the reward (or 0)
                if state == WIN_STATE:
                    mx_nxt_value_str = " R:+1 " # Win state reward
                elif state in HOLE_STATES:
                    mx_nxt_value_str = " R:-5 " # Hole state reward
                else:
                    # Find the maximum Q-value among all actions for non-terminal state (r, c)
                    if state in self.Q_values:
                        max_q_value = np.max(self.Q_values[state])
                        mx_nxt_value_str = f"{max_q_value:6.2f}" # Format max Q value
                    else:
                        mx_nxt_value_str = " N/A  " # Should not happen if Q_values initialized correctly

                out += mx_nxt_value_str + ' | '
            print(out)
        print('-----------------------------------------------')


# Main execution block
if __name__ == '__main__':
    print("--- Starting Q-Learning Script (GitHub Version Logic) ---")

    # --- Initialization ---
    # (Removed timing for initialization)
    ag = Agent()

    # --- Training ---
    print("\n--- Starting Training ---")
    train_start_time = time.time() # Keep start time for training
    total_rewards = ag.Q_Learning(episodes=10000) # Train for 10,000 episodes
    train_end_time = time.time() # Keep end time for training
    print(f"Training Duration: {train_end_time - train_start_time:.4f} seconds") # Keep print for training

    # --- Plotting ---
    # (Removed timing for plotting)
    print("\n--- Plotting Results ---")
    ag.plot_rewards(total_rewards)

    # --- Displaying Max Q-Values ---
    # (Removed timing for display)
    print("\n--- Displaying Max Q-Values per State ---")
    ag.showValues() # Use the showValues method

    # --- Removed Policy Simulation ---

    # --- Total Time ---
    # (Removed timing for total execution)
    print("\n--- Script Finished ---")

