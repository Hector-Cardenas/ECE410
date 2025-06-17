# Required libraries: numpy, matplotlib, imageio
# Install using: pip install numpy matplotlib imageio

import numpy as np
import time
import matplotlib.pyplot as plt
import imageio # Required for saving GIFs
import io      # Required for saving plots to memory buffer

# --- MLP Class (Identical to previous version) ---
class MLP:
    """
    A simple 2-Layer Multilayer Perceptron (MLP) for binary classification,
    trained using Backpropagation and Gradient Descent.
    Architecture: Input -> Hidden (Sigmoid) -> Output (Sigmoid)
    Includes history tracking for visualization.
    Uses gradients derived from Binary Cross-Entropy for output layer updates.
    """
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1, n_epochs=10000, random_state=1):
        """
        Initializes the MLP.

        Parameters:
        n_input (int): Number of input features.
        n_hidden (int): Number of neurons in the hidden layer.
        n_output (int): Number of neurons in the output layer.
        learning_rate (float): Step size for gradient descent.
        n_epochs (int): Number of training epochs.
        random_state (int): Seed for random weight initialization.
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.loss_history_ = [] # Tracks loss value every epoch
        # History stores parameters at specified intervals for visualization
        self.history_ = {'weights_h': [], 'bias_h': [], 'weights_o': [], 'bias_o': [], 'loss': []}

        # Initialize final weights/biases using random seed
        rgen = np.random.RandomState(self.random_state)
        # Weights Input -> Hidden Layer
        self.weights_h_ = rgen.normal(loc=0.0, scale=0.1, size=(self.n_input, self.n_hidden))
        # Bias Hidden Layer
        self.bias_h_ = np.zeros((1, self.n_hidden))
        # Weights Hidden -> Output Layer
        self.weights_o_ = rgen.normal(loc=0.0, scale=0.1, size=(self.n_hidden, self.n_output))
        # Bias Output Layer
        self.bias_o_ = np.zeros((1, self.n_output))

    def _sigmoid(self, z):
        """Compute the sigmoid activation function."""
        # Clip input to avoid overflow/underflow in exp for numerical stability
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _sigmoid_derivative(self, sigmoid_output):
        """Compute the derivative of sigmoid given its output."""
        # sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        return sigmoid_output * (1.0 - sigmoid_output)

    def _forward(self, X, weights_h, bias_h, weights_o, bias_o):
        """
        Performs the forward pass using provided weights/biases.
        Returns hidden activations (A_h), output activations (A_o),
        and intermediate weighted sums (Z_h, Z_o).
        """
        # Input to Hidden Layer
        Z_h = X @ weights_h + bias_h # Net input hidden
        A_h = self._sigmoid(Z_h)     # Activation hidden

        # Hidden to Output Layer
        Z_o = A_h @ weights_o + bias_o # Net input output
        A_o = self._sigmoid(Z_o)      # Activation output (prediction probability)

        return A_h, A_o, Z_h, Z_o

    def _compute_loss(self, y_true, y_pred):
        """
        Computes the Mean Squared Error loss for monitoring purposes.
        Note: Gradients used in backprop correspond to Binary Cross-Entropy.
        """
        # Ensure y_true has the same shape as y_pred for calculation
        y_true_reshaped = y_true.reshape(y_pred.shape)
        # Calculate MSE loss
        loss = 0.5 * np.mean((y_true_reshaped - y_pred)**2)
        return loss

    def fit(self, X, y, plot_interval=100):
        """
        Trains the MLP using backpropagation and gradient descent,
        storing history at specified intervals.

        Parameters:
        X (array): Training input data [n_samples, n_input].
        y (array): Target labels [n_samples] or [n_samples, 1].
        plot_interval (int): Store history every 'plot_interval' epochs for GIF generation.
        """
        # Reshape y to be a column vector [n_samples, 1]
        y_col = y.reshape(-1, 1)
        self.loss_history_ = [] # Track loss every epoch
        # Reset history storage for this training run
        self.history_ = {'weights_h': [], 'bias_h': [], 'weights_o': [], 'bias_o': [], 'loss': []}

        # Use local copies of weights/biases for iterative updates during training
        weights_h = self.weights_h_.copy()
        bias_h = self.bias_h_.copy()
        weights_o = self.weights_o_.copy()
        bias_o = self.bias_o_.copy()

        print("Starting MLP Training (using BCE gradients)...")
        start_time = time.time()

        # Store initial state (Epoch 0) if interval is set
        if plot_interval is not None:
             self.history_['weights_h'].append(weights_h.copy())
             self.history_['bias_h'].append(bias_h.copy())
             self.history_['weights_o'].append(weights_o.copy())
             self.history_['bias_o'].append(bias_o.copy())
             self.history_['loss'].append(None) # No loss calculated yet

        # --- Training Loop ---
        for epoch in range(self.n_epochs):
            # --- Forward Pass ---
            A_h, A_o, Z_h, Z_o = self._forward(X, weights_h, bias_h, weights_o, bias_o)

            # --- Compute Loss (for monitoring) ---
            loss = self._compute_loss(y_col, A_o)
            self.loss_history_.append(loss)

            # --- Backward Pass (Calculate Gradients) ---
            # Output layer error signal (delta) - Corresponds to BCE gradient w.r.t Z_o
            delta_output = A_o - y_col

            # Hidden layer error signal (delta)
            error_hidden = delta_output @ weights_o.T
            d_sigmoid_h = self._sigmoid_derivative(A_h)
            delta_hidden = error_hidden * d_sigmoid_h

            # Gradients for weights and biases
            grad_weights_o = A_h.T @ delta_output
            grad_bias_o = np.sum(delta_output, axis=0, keepdims=True)
            grad_weights_h = X.T @ delta_hidden
            grad_bias_h = np.sum(delta_hidden, axis=0, keepdims=True)

            # --- Update Weights and Biases (Gradient Descent Step) ---
            weights_o -= self.learning_rate * grad_weights_o
            bias_o -= self.learning_rate * grad_bias_o
            weights_h -= self.learning_rate * grad_weights_h
            bias_h -= self.learning_rate * grad_bias_h

            # --- Store History at Intervals ---
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

        # --- End of Training ---
        self.weights_h_ = weights_h
        self.bias_h_ = bias_h
        self.weights_o_ = weights_o
        self.bias_o_ = bias_o
        if plot_interval is not None and (self.n_epochs % plot_interval != 0):
             self.history_['weights_h'].append(weights_h.copy())
             self.history_['bias_h'].append(bias_h.copy())
             self.history_['weights_o'].append(weights_o.copy())
             self.history_['bias_o'].append(bias_o.copy())
             self.history_['loss'].append(self.loss_history_[-1])

        end_time = time.time()
        print("-" * 30)
        print(f"Training finished in {end_time - start_time:.2f} seconds.")
        print(f"Final Loss: {self.loss_history_[-1]:.6f}")
        return self

    def predict(self, X, weights_h=None, bias_h=None, weights_o=None, bias_o=None):
        """Makes predictions using specified weights/biases or final trained ones."""
        w_h = weights_h if weights_h is not None else self.weights_h_
        b_h = bias_h if bias_h is not None else self.bias_h_
        w_o = weights_o if weights_o is not None else self.weights_o_
        b_o = bias_o if bias_o is not None else self.bias_o_
        _, probabilities, _, _ = self._forward(X, w_h, b_h, w_o, b_o)
        predictions = np.where(probabilities >= 0.5, 1, 0)
        return predictions.flatten()


# --- Plotting Function for MLP Final Prediction Regions (Unchanged) ---
def plot_mlp_decision_regions(X, y, mlp, weights_h, bias_h, weights_o, bias_o, epoch_num, ax, title_suffix=""):
    """Plots data points and the MLP final prediction regions with sharp boundaries."""
    ax.clear()
    cmap_points = plt.cm.get_cmap('viridis', 2)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolors='k', s=60, label='Data Points', zorder=3)
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    _, Z_probs, _, _ = mlp._forward(grid_input, weights_h, bias_h, weights_o, bias_o)
    Z = Z_probs.reshape(xx.shape)
    cmap_contour = plt.cm.RdBu
    contour = ax.contourf(xx, yy, Z, cmap=cmap_contour, alpha=0.6, levels=[-0.1, 0.5, 1.1], zorder=1)
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(f'MLP Predicted Regions - Epoch: {epoch_num} {title_suffix}')
    ax.legend(*scatter.legend_elements(), title="Actual Classes", loc='upper left')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(True, linestyle=':', alpha=0.7)


# --- Function to Save MLP OUTPUT REGIONS GIF (Unchanged) ---
def save_mlp_output_regions_gif(mlp, X, y, filename="mlp_output_regions.gif", duration_per_frame=0.1):
    """Generates plots of MLP final prediction regions for each stored epoch state and saves as GIF."""
    print(f"\nGenerating frames for MLP Output Regions GIF: {filename}...")
    fig, ax = plt.subplots()
    frames = []
    weights_h_hist = mlp.history_['weights_h']
    bias_h_hist = mlp.history_['bias_h']
    weights_o_hist = mlp.history_['weights_o']
    bias_o_hist = mlp.history_['bias_o']
    loss_hist_frames = mlp.history_['loss']
    num_epochs_stored = len(weights_h_hist)
    plot_interval = mlp.n_epochs // (num_epochs_stored - 1) if num_epochs_stored > 1 else 1
    if num_epochs_stored == 0:
        print("Warning: No history found in MLP. Cannot create GIF.")
        plt.close(fig)
        return
    for i in range(num_epochs_stored):
        epoch_num = i * plot_interval if i > 0 else 0
        if i == num_epochs_stored - 1: epoch_num = mlp.n_epochs
        weights_h = weights_h_hist[i]; bias_h = bias_h_hist[i]
        weights_o = weights_o_hist[i]; bias_o = bias_o_hist[i]
        loss = loss_hist_frames[i]
        title_suffix = f"| Loss: {loss:.4f}" if loss is not None else "| Initial State"
        plot_mlp_decision_regions(X, y, mlp, weights_h, bias_h, weights_o, bias_o, epoch_num, ax, title_suffix)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); buf.seek(0)
        frame_image = imageio.imread(buf); frames.append(frame_image); buf.close()
        print(f"  Frame {i+1}/{num_epochs_stored} (Epoch ~{epoch_num}) generated.", end='\r')
    plt.close(fig)
    print(f"\nSaving MLP Output Regions GIF with {len(frames)} frames...")
    try:
        imageio.mimsave(filename, frames, duration=duration_per_frame)
        print(f"Successfully saved GIF to '{filename}'")
    except Exception as e: print(f"Error saving GIF: {e}")


# === Main Execution Block ===

if __name__ == "__main__":
    # 1. Define XOR function data
    X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_XOR = np.array([0, 1, 1, 0])

    # 2. Configure and Create the MLP instance
    print("\n" + "="*30)
    print("Training MLP on XOR Gate (Using BCE Gradients) - Attempt 2")
    print("="*30)

    input_size = X_XOR.shape[1]
    # --- Hyperparameter Changes ---
    hidden_size = 2 # TRYING 2 hidden neurons instead of 4
    output_size = 1
    learn_rate = 0.1 # Keep learning rate for now
    epochs = 10000   # Keep epochs for now
    plot_save_interval = 200
    rand_state = 1   # TRYING random_state=1 instead of 42
    # --- End Changes ---

    mlp_xor = MLP(n_input=input_size,
                  n_hidden=hidden_size,
                  n_output=output_size,
                  learning_rate=learn_rate,
                  n_epochs=epochs,
                  random_state=rand_state) # Use updated random state

    # 3. Train the network
    mlp_xor.fit(X_XOR, y_XOR, plot_interval=plot_save_interval)

    # 4. Save the MLP OUTPUT (Prediction) REGIONS learning process as a GIF
    save_mlp_output_regions_gif(mlp_xor, X_XOR, y_XOR,
                                filename="xor_mlp_pred_regions_bce_tuned.gif", # New filename
                                duration_per_frame=0.1)

    # 5. Test final predictions
    print("\n--- Final Predictions for XOR Gate ---")
    all_correct = True
    for xi, target in zip(X_XOR, y_XOR):
        xi_reshaped = xi.reshape(1, -1)
        prediction = mlp_xor.predict(xi_reshaped)[0]
        correct = prediction == target
        print(f"Input: {xi} -> Target: {target} -> Predicted: {prediction} {'(Correct)' if correct else '(Incorrect)'}")
        if not correct: all_correct = False
    if all_correct: print("\nSUCCESS: MLP correctly learned the XOR function!")
    else: print("\nFAILURE: MLP did not converge. Try further tuning (epochs, lr, hidden_units, random_state).")

    # 6. Optional: Plot overall training loss curve
    try: import matplotlib.pyplot as plt; PLOT_ENABLED = True
    except ImportError: PLOT_ENABLED = False
    if PLOT_ENABLED and mlp_xor.loss_history_:
        plt.figure()
        plot_step = max(1, len(mlp_xor.loss_history_) // 1000)
        plt.plot(range(1, len(mlp_xor.loss_history_) + 1, plot_step), mlp_xor.loss_history_[::plot_step])
        plt.xlabel(f'Epochs (x{plot_step})'); plt.ylabel('Mean Squared Error Loss')
        plt.title('MLP Training Loss Curve for XOR (Using BCE Gradients)'); plt.grid(True)
        plt.yscale('log'); plt.show()
