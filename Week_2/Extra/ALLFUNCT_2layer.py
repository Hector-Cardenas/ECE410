# Required libraries: numpy, matplotlib, imageio
# Install using: pip install numpy matplotlib imageio
# Warning: This script can be computationally and memory intensive!

import numpy as np
import time
import matplotlib.pyplot as plt
import imageio # Required for saving GIFs
import io      # Required for saving plots to memory buffer
import warnings

# --- MLP Class (Added Momentum) ---
class MLP:
    """
    MLP for single-output binary classification. Includes history storage,
    early stopping, and optional momentum for gradient descent.
    Uses gradients derived from Binary Cross-Entropy.
    """
    def __init__(self, n_input, n_hidden, n_output=1, learning_rate=0.1, n_epochs=10000, random_state=1, momentum=0.0): # Added momentum parameter
        """
        Initializes the MLP.

        Parameters:
        ... (previous parameters) ...
        momentum (float): Momentum factor (0.0 to < 1.0). Default 0 means standard GD.
        """
        if n_output != 1:
            warnings.warn("MLP forced to n_output=1 for binary function task.", UserWarning)
            n_output = 1

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.momentum = momentum # Store momentum parameter
        self.loss_history_ = []
        self.history_ = {'weights_h': [], 'bias_h': [], 'weights_o': [], 'bias_o': [], 'loss': [], 'epoch': []}

        rgen = np.random.RandomState(self.random_state)
        self.weights_h_ = rgen.normal(loc=0.0, scale=0.1, size=(self.n_input, self.n_hidden))
        self.bias_h_ = np.zeros((1, self.n_hidden))
        self.weights_o_ = rgen.normal(loc=0.0, scale=0.1, size=(self.n_hidden, self.n_output))
        self.bias_o_ = np.zeros((1, self.n_output))

    def _sigmoid(self, z):
        """Compute the sigmoid activation function."""
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _sigmoid_derivative(self, sigmoid_output):
        """Compute the derivative of sigmoid given its output."""
        return sigmoid_output * (1.0 - sigmoid_output)

    def _forward(self, X, weights_h, bias_h, weights_o, bias_o):
        """Performs the forward pass using provided weights/biases."""
        Z_h = X @ weights_h + bias_h
        A_h = self._sigmoid(Z_h)
        Z_o = A_h @ weights_o + bias_o
        A_o = self._sigmoid(Z_o)
        return A_h, A_o, Z_h, Z_o

    def _compute_loss(self, y_true, y_pred):
        """Computes the Mean Squared Error loss."""
        y_true_reshaped = y_true.reshape(y_pred.shape)
        loss = 0.5 * np.mean((y_true_reshaped - y_pred)**2)
        return loss

    # --- Updated fit method with momentum ---
    def fit(self, X, y, plot_interval=100, loss_threshold=1e-5, patience=20):
        """
        Trains the MLP using backpropagation and gradient descent with optional momentum.
        Stores history at plot_interval. Includes early stopping based on loss.
        """
        y_col = y.reshape(-1, 1)
        self.loss_history_ = []
        self.history_ = {'weights_h': [], 'bias_h': [], 'weights_o': [], 'bias_o': [], 'loss': [], 'epoch': []}

        weights_h = self.weights_h_.copy(); bias_h = self.bias_h_.copy()
        weights_o = self.weights_o_.copy(); bias_o = self.bias_o_.copy()

        # Initialize previous updates for momentum
        prev_update_wh = np.zeros_like(weights_h)
        prev_update_bh = np.zeros_like(bias_h)
        prev_update_wo = np.zeros_like(weights_o)
        prev_update_bo = np.zeros_like(bias_o)

        # Check if fit is called with non-default patience/threshold
        effective_patience = patience if patience is not None else 20
        effective_threshold = loss_threshold if loss_threshold is not None else 1e-5

        print(f"Starting Training (Momentum={self.momentum}, Max Epochs: {self.n_epochs})...")
        # Only print early stopping info if threshold is set
        if effective_threshold is not None and effective_threshold > 0:
             print(f"Early stopping: Loss < {effective_threshold} for {effective_patience} consecutive epochs.")

        start_time = time.time()
        epochs_below_threshold_count = 0
        final_epoch = self.n_epochs # Assume full run unless stopped early

        # Store initial state
        if plot_interval is not None:
             self.history_['weights_h'].append(weights_h.copy())
             self.history_['bias_h'].append(bias_h.copy())
             self.history_['weights_o'].append(weights_o.copy())
             self.history_['bias_o'].append(bias_o.copy())
             self.history_['loss'].append(None)
             self.history_['epoch'].append(0)

        for epoch in range(self.n_epochs):
            final_epoch = epoch # Keep track of last epoch started
            A_h, A_o, Z_h, Z_o = self._forward(X, weights_h, bias_h, weights_o, bias_o)
            loss = self._compute_loss(y_col, A_o)
            self.loss_history_.append(loss)

            # Backward Pass
            delta_output = A_o - y_col
            error_hidden = delta_output @ weights_o.T
            d_sigmoid_h = self._sigmoid_derivative(A_h)
            delta_hidden = error_hidden * d_sigmoid_h
            grad_weights_o = A_h.T @ delta_output
            grad_bias_o = np.sum(delta_output, axis=0, keepdims=True)
            grad_weights_h = X.T @ delta_hidden
            grad_bias_h = np.sum(delta_hidden, axis=0, keepdims=True)

            # Update Weights with Momentum
            update_wh = (self.momentum * prev_update_wh) - (self.learning_rate * grad_weights_h)
            update_bh = (self.momentum * prev_update_bh) - (self.learning_rate * grad_bias_h)
            update_wo = (self.momentum * prev_update_wo) - (self.learning_rate * grad_weights_o)
            update_bo = (self.momentum * prev_update_bo) - (self.learning_rate * grad_bias_o)
            weights_h += update_wh; bias_h += update_bh
            weights_o += update_wo; bias_o += update_bo
            prev_update_wh = update_wh; prev_update_bh = update_bh
            prev_update_wo = update_wo; prev_update_bo = update_bo

            # Store History at Intervals
            store_this_epoch = (plot_interval is not None) and ((epoch + 1) % plot_interval == 0)
            if store_this_epoch:
                 self.history_['weights_h'].append(weights_h.copy())
                 self.history_['bias_h'].append(bias_h.copy())
                 self.history_['weights_o'].append(weights_o.copy())
                 self.history_['bias_o'].append(bias_o.copy())
                 self.history_['loss'].append(loss)
                 self.history_['epoch'].append(epoch + 1)

            # Early Stopping Check (only if threshold is set)
            if effective_threshold is not None and effective_threshold > 0:
                 if loss < effective_threshold:
                     epochs_below_threshold_count += 1
                 else:
                     epochs_below_threshold_count = 0 # Reset counter
                 if epochs_below_threshold_count >= effective_patience:
                     print(f"Early stopping triggered at epoch {epoch + 1}.")
                     final_epoch = epoch + 1 # Record stopping epoch
                     break # Exit the training loop

        # --- End of Training ---
        self.weights_h_ = weights_h; self.bias_h_ = bias_h
        self.weights_o_ = weights_o; self.bias_o_ = bias_o

        # Ensure the very final state is stored if needed for GIF
        if plot_interval is not None:
            last_hist_epoch = self.history_['epoch'][-1] if self.history_['epoch'] else -1
            # Use final_epoch (actual last epoch run + 1 if stopped early)
            actual_epochs_run = final_epoch + 1 if final_epoch < self.n_epochs else self.n_epochs
            if actual_epochs_run > last_hist_epoch:
                 # Avoid storing duplicate if already stored by interval on last epoch
                 if not (store_this_epoch and actual_epochs_run == epoch + 1) :
                     self.history_['weights_h'].append(weights_h.copy())
                     self.history_['bias_h'].append(bias_h.copy())
                     self.history_['weights_o'].append(weights_o.copy())
                     self.history_['bias_o'].append(bias_o.copy())
                     self.history_['loss'].append(self.loss_history_[-1])
                     self.history_['epoch'].append(actual_epochs_run)

        end_time = time.time()
        actual_epochs_run = final_epoch + 1 if final_epoch < self.n_epochs else self.n_epochs
        print(f"Training finished after {actual_epochs_run} epochs. Final Loss: {self.loss_history_[-1]:.6f}")
        return self

    def predict(self, X):
        """Makes predictions using the final trained weights."""
        _, probabilities, _, _ = self._forward(X, self.weights_h_, self.bias_h_, self.weights_o_, self.bias_o_)
        predictions = np.where(probabilities >= 0.5, 1, 0)
        return predictions.flatten()

# --- Helper Function to Generate Target Vector (Unchanged) ---
def get_target_vector(func_index, n_inputs=2):
    if not 0 <= func_index <= (2**(2**n_inputs) - 1): raise ValueError("func_index out of range")
    num_combinations = 2**n_inputs
    binary_string = format(func_index, f'0{num_combinations}b')
    target_vector = np.array([int(bit) for bit in reversed(binary_string)])
    return target_vector

# --- Plotting Function for Single Output MLP Regions (Unchanged) ---
def plot_mlp_single_output_regions(X, y, mlp_weights_biases, ax, title="MLP Decision Region"):
    ax.clear()
    cmap_points = plt.cm.get_cmap('viridis', 2)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolors='k', s=50, zorder=3)
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    temp_mlp = MLP(mlp_weights_biases['n_input'], mlp_weights_biases['n_hidden'])
    _, Z_probs, _, _ = temp_mlp._forward(grid_input,
                                         mlp_weights_biases['weights_h'], mlp_weights_biases['bias_h'],
                                         mlp_weights_biases['weights_o'], mlp_weights_biases['bias_o'])
    Z = Z_probs.reshape(xx.shape)
    cmap_contour = plt.cm.RdBu
    ax.contourf(xx, yy, Z, cmap=cmap_contour, alpha=0.6, levels=[-0.1, 0.5, 1.1], zorder=1)
    ax.set_xlabel('Input A', fontsize=8); ax.set_ylabel('Input B', fontsize=8)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
    ax.grid(True, linestyle=':', alpha=0.5)


# --- Function to Save 4x4 Grid GIF (Unchanged) ---
def save_16_func_grid_gif(all_histories, X_inputs, target_vectors, func_names,
                          filename="mlp_16_functions.gif", duration_per_frame=0.2):
    if not all_histories or len(all_histories) != 16: print("Error: Need histories for 16 functions."); return
    if not imageio: print("Error: imageio not found."); return
    print(f"\nGenerating frames for 4x4 Grid GIF: {filename}...")
    max_frames = 0
    for history in all_histories:
        if history and history.get('weights_h'): max_frames = max(max_frames, len(history['weights_h']))
    if max_frames == 0: print("Error: No history frames found."); return

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    frames = []
    for frame_idx in range(max_frames):
        print(f"  Generating Frame {frame_idx + 1}/{max_frames}...", end='\r')
        for func_idx in range(16):
            row, col = func_idx // 4, func_idx % 4; ax = axes[row, col]
            history = all_histories[func_idx]; y_target = target_vectors[func_idx]
            # Use min() to handle histories that stopped early
            current_frame_idx = min(frame_idx, len(history['weights_h']) - 1) if history.get('weights_h') else 0

            if current_frame_idx < len(history['weights_h']):
                 mlp_params = { # Package params for plotting function
                     'n_input': X_inputs.shape[1],
                     'n_hidden': history['weights_h'][current_frame_idx].shape[1],
                     'weights_h': history['weights_h'][current_frame_idx],
                     'bias_h': history['bias_h'][current_frame_idx],
                     'weights_o': history['weights_o'][current_frame_idx],
                     'bias_o': history['bias_o'][current_frame_idx]
                 }
                 # Get corresponding loss and epoch number for title
                 loss = history['loss'][current_frame_idx]
                 epoch_num = history['epoch'][current_frame_idx]
                 title_suffix = f"| E:{epoch_num}, L:{loss:.3f}" if loss is not None else f"| E:{epoch_num}"
                 title = f"F{func_idx}: {func_names.get(func_idx, '')}\n{title_suffix}"
                 plot_mlp_single_output_regions(X_inputs, y_target, mlp_params, ax, title=title)
            else: # Should not happen if max_frames logic is correct, but as fallback
                 ax.clear(); ax.set_title(f"F{func_idx}: No History Frame {frame_idx}"); ax.set_xticks([]); ax.set_yticks([])

        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        # Save figure to buffer
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=75); buf.seek(0)
        frame_image = imageio.imread(buf); frames.append(frame_image); buf.close()

    plt.close(fig) # Close plot figure
    print(f"\nSaving 4x4 Grid GIF with {len(frames)} frames...")
    try:
        imageio.mimsave(filename, frames, duration=duration_per_frame)
        print(f"Successfully saved GIF to '{filename}'")
    except Exception as e: print(f"Error saving GIF: {e}")


# === Main Execution Block (Added overall early stop) ===

if __name__ == "__main__":
    # 1. Define base input data
    X_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    num_functions = 16
    func_names = {0: "0", 1: "AND", 2: "A&!B", 3: "A", 4: "!A&B", 5: "B", 6: "XOR", 7: "OR", 8: "NOR", 9: "XNOR", 10: "!B", 11: "A|!B", 12: "!A", 13: "!A|B", 14: "NAND", 15: "1"}
    trained_mlps = {}; target_vectors = {}; all_histories = []

    # 2. Configure MLP parameters
    input_size = X_inputs.shape[1]; hidden_size = 3; output_size = 1
    learn_rate = 0.1
    epochs = 15000 # Max epochs per function
    rand_state_base = 0
    early_stop_threshold = 1e-4
    early_stop_patience = 200
    plot_save_interval = 100
    momentum_value = 0.8

    # 3. Train MLP for each function
    print("\n" + "="*40); print("Training MLPs for all 16 Binary Functions"); print("="*40)
    print(f"Using Momentum: {momentum_value}")
    all_converged = True # Flag to track if all functions converged

    for i in range(num_functions):
        print(f"\n--- Training Function {i}: {func_names.get(i, 'Unknown')} ---")
        y_target = get_target_vector(i, n_inputs=2); target_vectors[i] = y_target
        mlp = MLP(n_input=input_size, n_hidden=hidden_size, n_output=output_size,
                  learning_rate=learn_rate, n_epochs=epochs,
                  random_state=rand_state_base + i,
                  momentum=momentum_value)

        mlp.fit(X_inputs, y_target,
                plot_interval=plot_save_interval,
                loss_threshold=early_stop_threshold,
                patience=early_stop_patience)

        trained_mlps[i] = mlp
        all_histories.append(mlp.history_) # Store history regardless of convergence for potential partial GIF

        # --- Check for convergence failure ---
        predictions = mlp.predict(X_inputs)
        print(f"Target:   {y_target}")
        print(f"Predicted:{predictions}")
        converged = np.array_equal(predictions, y_target)
        if converged:
            print("Result:   SUCCESS")
        else:
            print("Result:   FAILURE")
            all_converged = False # Set flag to false
            print(f"\n*** STOPPING: Function {i} ({func_names.get(i, '')}) failed to converge correctly. ***")
            break # Exit the main training loop

    # 4. Create the 4x4 Grid Animation GIF only if all functions converged
    if all_converged:
        print("\nAll functions converged successfully. Generating GIF...")
        save_16_func_grid_gif(all_histories, X_inputs, target_vectors, func_names,
                              filename="mlp_16_functions_momentum.gif",
                              duration_per_frame=0.1)
    else:
        print("\nGIF not generated because one or more functions failed to converge.")

    print("\n--- All tasks complete ---")

