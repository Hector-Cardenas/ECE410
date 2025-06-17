# Required libraries: numpy, matplotlib, imageio
# Install using: pip install numpy matplotlib imageio

import numpy as np
import time
import matplotlib.pyplot as plt
import imageio # Required for saving GIFs
import io      # Required for saving plots to memory buffer

# --- MLP Class (Handles multiple outputs, includes early stopping) ---
class MLP:
    """
    A simple 2-Layer Multilayer Perceptron (MLP) for multi-output binary classification,
    trained using Backpropagation and Gradient Descent.
    Architecture: Input -> Hidden (Sigmoid) -> Output (Sigmoid)
    Includes history tracking for visualization and early stopping based on loss.
    Uses gradients derived from Binary Cross-Entropy for output layer updates.
    """
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1, n_epochs=10000, random_state=1):
        """
        Initializes the MLP. Handles multiple output neurons.
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output # Now can be > 1
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.loss_history_ = []
        self.history_ = {'weights_h': [], 'bias_h': [], 'weights_o': [], 'bias_o': [], 'loss': []}

        rgen = np.random.RandomState(self.random_state)
        # Weights Input -> Hidden
        self.weights_h_ = rgen.normal(loc=0.0, scale=0.1, size=(self.n_input, self.n_hidden))
        self.bias_h_ = np.zeros((1, self.n_hidden))
        # Weights Hidden -> Output (Shape adapts to n_output)
        self.weights_o_ = rgen.normal(loc=0.0, scale=0.1, size=(self.n_hidden, self.n_output))
        # Bias Output (Shape adapts to n_output)
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
        A_o = self._sigmoid(Z_o) # Output shape: [n_samples, n_output]
        return A_h, A_o, Z_h, Z_o

    def _compute_loss(self, y_true, y_pred):
        """Computes the Mean Squared Error loss across all outputs."""
        y_true_reshaped = y_true.reshape(y_pred.shape)
        loss = 0.5 * np.mean((y_true_reshaped - y_pred)**2)
        return loss

    # --- Updated fit method with early stopping ---
    def fit(self, X, y, plot_interval=100, loss_threshold=1e-5, patience=10):
        """
        Trains the MLP using backpropagation and gradient descent.
        Handles multi-output target y. Includes early stopping based on loss.

        Parameters:
        X (array): Training input data [n_samples, n_input].
        y (array): Target labels [n_samples, n_output].
        plot_interval (int): Store history every 'plot_interval' epochs for GIF generation.
        loss_threshold (float): Stop training if loss is below this value for 'patience' epochs.
        patience (int): Number of consecutive epochs loss must be below threshold to stop.
        """
        # Ensure y has shape [n_samples, n_output]
        if y.ndim == 1:
             if self.n_output == 1: y_col = y.reshape(-1, 1)
             else: raise ValueError(f"Target y has 1 dim but n_output={self.n_output}.")
        elif y.shape[1] != self.n_output:
             raise ValueError(f"Target y shape {y.shape} mismatch n_output={self.n_output}.")
        else: y_col = y

        self.loss_history_ = []
        self.history_ = {'weights_h': [], 'bias_h': [], 'weights_o': [], 'bias_o': [], 'loss': []}

        weights_h = self.weights_h_.copy(); bias_h = self.bias_h_.copy()
        weights_o = self.weights_o_.copy(); bias_o = self.bias_o_.copy()

        print(f"Starting MLP Training (Input={self.n_input}, Hidden={self.n_hidden}, Output={self.n_output})...")
        print(f"Early stopping: Loss < {loss_threshold} for {patience} consecutive epochs.")
        start_time = time.time()

        # --- Early Stopping Initialization ---
        epochs_below_threshold_count = 0

        # Store initial state
        if plot_interval is not None:
             self.history_['weights_h'].append(weights_h.copy())
             self.history_['bias_h'].append(bias_h.copy())
             self.history_['weights_o'].append(weights_o.copy())
             self.history_['bias_o'].append(bias_o.copy())
             self.history_['loss'].append(None)

        # --- Training Loop ---
        final_epoch = self.n_epochs # Track the actual last epoch run
        for epoch in range(self.n_epochs):
            final_epoch = epoch # Update last epoch number
            A_h, A_o, Z_h, Z_o = self._forward(X, weights_h, bias_h, weights_o, bias_o)
            loss = self._compute_loss(y_col, A_o)
            self.loss_history_.append(loss)

            # --- Backward Pass ---
            delta_output = A_o - y_col
            error_hidden = delta_output @ weights_o.T
            d_sigmoid_h = self._sigmoid_derivative(A_h)
            delta_hidden = error_hidden * d_sigmoid_h
            grad_weights_o = A_h.T @ delta_output
            grad_bias_o = np.sum(delta_output, axis=0, keepdims=True)
            grad_weights_h = X.T @ delta_hidden
            grad_bias_h = np.sum(delta_hidden, axis=0, keepdims=True)

            # --- Update Weights ---
            weights_o -= self.learning_rate * grad_weights_o
            bias_o -= self.learning_rate * grad_bias_o
            weights_h -= self.learning_rate * grad_weights_h
            bias_h -= self.learning_rate * grad_bias_h

            # --- Store History ---
            store_this_epoch = (plot_interval is not None) and ((epoch + 1) % plot_interval == 0)
            if store_this_epoch:
                 self.history_['weights_h'].append(weights_h.copy())
                 self.history_['bias_h'].append(bias_h.copy())
                 self.history_['weights_o'].append(weights_o.copy())
                 self.history_['bias_o'].append(bias_o.copy())
                 self.history_['loss'].append(loss)

            # Print progress periodically
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss:.6f}")

            # --- Early Stopping Check ---
            if loss < loss_threshold:
                epochs_below_threshold_count += 1
            else:
                epochs_below_threshold_count = 0 # Reset counter if loss goes up

            if epochs_below_threshold_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}: Loss < {loss_threshold} for {patience} epochs.")
                final_epoch = epoch + 1 # Record the epoch number where we stopped
                break # Exit the training loop

        # --- End of Training ---
        self.weights_h_ = weights_h
        self.bias_h_ = bias_h
        self.weights_o_ = weights_o
        self.bias_o_ = bias_o

        # Ensure the very final state is stored if not caught by interval or early stopping
        # Check if the last stored history corresponds to the final weights
        last_hist_epoch = (len(self.history_['loss']) -1) * plot_interval if plot_interval is not None and len(self.history_['loss']) > 1 else 0
        if plot_interval is not None and final_epoch > last_hist_epoch:
             self.history_['weights_h'].append(weights_h.copy())
             self.history_['bias_h'].append(bias_h.copy())
             self.history_['weights_o'].append(weights_o.copy())
             self.history_['bias_o'].append(bias_o.copy())
             self.history_['loss'].append(self.loss_history_[-1])

        end_time = time.time()
        print("-" * 30)
        print(f"Training finished after {final_epoch} epochs.") # Show actual epochs run
        print(f"Final Loss: {self.loss_history_[-1]:.6f}")
        return self

    def predict(self, X, weights_h=None, bias_h=None, weights_o=None, bias_o=None):
        """
        Makes predictions using specified weights/biases or final trained ones.
        Returns binary predictions [n_samples, n_output].
        """
        w_h = weights_h if weights_h is not None else self.weights_h_
        b_h = bias_h if bias_h is not None else self.bias_h_
        w_o = weights_o if weights_o is not None else self.weights_o_
        b_o = bias_o if bias_o is not None else self.bias_o_
        _, probabilities, _, _ = self._forward(X, w_h, b_h, w_o, b_o)
        predictions = np.where(probabilities >= 0.5, 1, 0)
        return predictions


# --- Plotting Function for MLP Final Prediction Regions (Unchanged) ---
def plot_mlp_multi_output_regions(X, y, mlp, weights_h, bias_h, weights_o, bias_o, epoch_num, fig, axes, title_suffix=""):
    """
    Plots data points and the MLP final prediction regions for TWO outputs (Sum, Carry)
    on the provided figure axes (axes[0] for Sum, axes[1] for Carry).
    """
    axes[0].clear(); axes[1].clear()
    cmap_points = plt.cm.get_cmap('viridis', 2)
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    _, Z_probs_all, _, _ = mlp._forward(grid_input, weights_h, bias_h, weights_o, bias_o)
    Z_probs_sum = Z_probs_all[:, 0].reshape(xx.shape)
    Z_probs_carry = Z_probs_all[:, 1].reshape(xx.shape)
    cmap_contour = plt.cm.RdBu
    # Plot Sum regions
    axes[0].contourf(xx, yy, Z_probs_sum, cmap=cmap_contour, alpha=0.6, levels=[-0.1, 0.5, 1.1], zorder=1)
    scatter_sum = axes[0].scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap=cmap_points, edgecolors='k', s=60, zorder=3)
    axes[0].set_title(f'Sum Prediction - Epoch: {epoch_num} {title_suffix}')
    axes[0].set_ylabel('Input B')
    axes[0].legend(*scatter_sum.legend_elements(), title="Target Sum", loc='upper left')
    # Plot Carry regions
    axes[1].contourf(xx, yy, Z_probs_carry, cmap=cmap_contour, alpha=0.6, levels=[-0.1, 0.5, 1.1], zorder=1)
    scatter_carry = axes[1].scatter(X[:, 0], X[:, 1], c=y[:, 1], cmap=cmap_points, edgecolors='k', s=60, zorder=3)
    axes[1].set_title(f'Carry Prediction - Epoch: {epoch_num} {title_suffix}')
    axes[1].legend(*scatter_carry.legend_elements(), title="Target Carry", loc='upper left')
    # Format axes
    for ax in axes:
        ax.set_xlabel('Input A'); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max())
        ax.grid(True, linestyle=':', alpha=0.7)
    fig.tight_layout(pad=2.0)


# --- Function to Save Multi-Output MLP GIF (Unchanged) ---
def save_mlp_multi_output_gif(mlp, X, y, filename="mlp_multi_output.gif", duration_per_frame=0.1):
    """
    Generates plots of MLP multi-output decision regions and saves as GIF.
    """
    print(f"\nGenerating frames for Multi-Output MLP GIF: {filename}...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    frames = []
    weights_h_hist = mlp.history_['weights_h']; bias_h_hist = mlp.history_['bias_h']
    weights_o_hist = mlp.history_['weights_o']; bias_o_hist = mlp.history_['bias_o']
    loss_hist_frames = mlp.history_['loss']
    num_epochs_stored = len(weights_h_hist)
    # Estimate plot interval based on history length and total epochs run
    actual_epochs_run = len(mlp.loss_history_) # Get actual number of epochs run
    plot_interval = actual_epochs_run // (num_epochs_stored - 1) if num_epochs_stored > 1 else 1

    if num_epochs_stored == 0:
        print("Warning: No history found in MLP. Cannot create GIF.")
        plt.close(fig)
        return

    for i in range(num_epochs_stored):
        # Calculate epoch number for title more accurately
        epoch_num = i * plot_interval if i > 0 else 0
        # Ensure last frame shows the actual final epoch number
        if i == num_epochs_stored - 1:
             epoch_num = actual_epochs_run if actual_epochs_run > 0 else 0 # Use actual epochs run

        weights_h = weights_h_hist[i]; bias_h = bias_h_hist[i]
        weights_o = weights_o_hist[i]; bias_o = bias_o_hist[i]
        loss = loss_hist_frames[i]
        title_suffix = f"| Loss: {loss:.4f}" if loss is not None else "| Initial State"
        plot_mlp_multi_output_regions(X, y, mlp, weights_h, bias_h, weights_o, bias_o, epoch_num, fig, axes, title_suffix)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); buf.seek(0)
        frame_image = imageio.imread(buf); frames.append(frame_image); buf.close()
        print(f"  Frame {i+1}/{num_epochs_stored} (Epoch ~{epoch_num}) generated.", end='\r')

    plt.close(fig)
    print(f"\nSaving Multi-Output MLP GIF with {len(frames)} frames...")
    try:
        imageio.mimsave(filename, frames, duration=duration_per_frame)
        print(f"Successfully saved GIF to '{filename}'")
    except Exception as e: print(f"Error saving GIF: {e}")


# === Main Execution Block ===

if __name__ == "__main__":
    # 1. Define Binary Adder function data
    X_adder = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_adder = np.array([[0, 0], [1, 0], [1, 0], [0, 1]])

    # 2. Configure and Create the MLP instance
    print("\n" + "="*30)
    print("Training MLP on Binary Adder Task with Early Stopping")
    print("="*30)

    input_size = X_adder.shape[1]
    hidden_size = 4
    output_size = y_adder.shape[1]
    learn_rate = 0.1
    epochs = 20000 # Increase max epochs, might stop early
    plot_save_interval = 200
    rand_state = 1
    # Early stopping parameters
    early_stop_threshold = 1e-4 # Stop if loss < 0.0001
    early_stop_patience = 20    # For 20 consecutive epochs

    mlp_adder = MLP(n_input=input_size,
                    n_hidden=hidden_size,
                    n_output=output_size,
                    learning_rate=learn_rate,
                    n_epochs=epochs, # Max epochs
                    random_state=rand_state)

    # 3. Train the network with early stopping parameters
    mlp_adder.fit(X_adder, y_adder,
                  plot_interval=plot_save_interval,
                  loss_threshold=early_stop_threshold,
                  patience=early_stop_patience)

    # 4. Save the MLP OUTPUT REGIONS learning process as a GIF
    save_mlp_multi_output_gif(mlp_adder, X_adder, y_adder,
                              filename="adder_mlp_pred_regions_earlystop.gif", # New filename
                              duration_per_frame=0.15)

    # 5. Test final predictions
    print("\n--- Final Predictions for Binary Adder ---")
    all_correct_samples = 0
    predictions = mlp_adder.predict(X_adder)
    for i in range(len(X_adder)):
        xi = X_adder[i]; target = y_adder[i]; pred = predictions[i]
        correct = np.array_equal(pred, target)
        print(f"Input: {xi} -> Target: {target} -> Predicted: {pred} {'(Correct)' if correct else '(Incorrect)'}")
        if correct: all_correct_samples += 1
    if all_correct_samples == len(X_adder): print("\nSUCCESS: MLP correctly learned the Binary Adder function!")
    else: print(f"\nFAILURE: MLP did not learn all cases correctly ({all_correct_samples}/{len(X_adder)} correct).")

    # 6. Optional: Plot overall training loss curve
    try: import matplotlib.pyplot as plt; PLOT_ENABLED = True
    except ImportError: PLOT_ENABLED = False
    if PLOT_ENABLED and mlp_adder.loss_history_:
        plt.figure()
        plot_step = max(1, len(mlp_adder.loss_history_) // 1000)
        plt.plot(range(1, len(mlp_adder.loss_history_) + 1, plot_step), mlp_adder.loss_history_[::plot_step])
        plt.xlabel(f'Epochs (x{plot_step})'); plt.ylabel('Mean Squared Error Loss')
        plt.title('MLP Training Loss Curve for Binary Adder'); plt.grid(True)
        plt.yscale('log'); plt.show()

