import numpy as np
import time
import matplotlib.pyplot as plt
import imageio # Required for saving GIFs
import io      # Required for saving plots to memory buffer

HOLDLASTFRAME = 5

# --- Original Perceptron Class (Step Function Activation) ---
# (Includes history tracking - identical to the previous version)
class Perceptron:
    """
    A simple Perceptron classifier using a Step Function activation.
    (Includes history tracking for visualization)
    """
    def __init__(self, learning_rate=0.1, n_epochs=50, random_state=1):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None
        self.errors_ = []
        self.history_ = {'weights': [], 'bias': [], 'errors': []}

    def fit(self, X, y, plot_interval=None):
        if X.shape[1] != 2:
            raise ValueError("This implementation expects exactly 2 input features.")
        # Allow 0/1 labels
        # if not np.all(np.unique(y) == [0, 1]):
        #      print(f"Warning: Target labels y should ideally be 0 or 1. Found: {np.unique(y)}")

        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias_ = 0.0
        self.errors_ = []
        self.history_['weights'] = []
        self.history_['bias'] = []
        self.history_['errors'] = []

        self.history_['weights'].append(self.weights_.copy())
        self.history_['bias'].append(self.bias_)
        self.history_['errors'].append(None)

        print(f"Initial weights: {self.weights_}, Initial bias: {self.bias_}")
        print("-" * 30)
        print("Starting Training (using Step Function)...")
        start_time = time.time()

        for epoch in range(self.n_epochs):
            errors_in_epoch = 0
            for xi, target in zip(X, y):
                net_input = self.net_input(xi)
                prediction = np.where(net_input >= 0.0, 1, 0)
                error = target - prediction
                if error != 0:
                    update = self.learning_rate * error
                    self.weights_ += update * xi
                    self.bias_ += update
                    errors_in_epoch += 1
##                  print(f"W={self.weights_}, b={self.bias_:.4f}") # ADD THIS LINE

                    

            self.errors_.append(errors_in_epoch)
            print(f"Epoch {epoch+1}: Errors={errors_in_epoch}, W={self.weights_}, b={self.bias_:.4f}") # ADD THIS LINE

            store_this_epoch = (plot_interval is None) or ((epoch + 1) % plot_interval == 0) or (errors_in_epoch == 0) or (epoch == self.n_epochs - 1)

            if store_this_epoch:
                 self.history_['weights'].append(self.weights_.copy())
                 self.history_['bias'].append(self.bias_)
                 self.history_['errors'].append(errors_in_epoch)

            # Removed periodic print to avoid clutter for potentially short XOR training
            # if epoch % 10 == 0 or epoch == self.n_epochs - 1:
            #      print(f"Epoch {epoch+1}/{self.n_epochs} | Misclassifications: {errors_in_epoch}")

            if errors_in_epoch == 0 and epoch >= 0: # Check convergence from epoch 0
                 print(f"\nConverged after {epoch+1} epochs.")
                 if not store_this_epoch: # Ensure converged state is stored
                     self.history_['weights'].append(self.weights_.copy())
                     self.history_['bias'].append(self.bias_)
                     self.history_['errors'].append(errors_in_epoch)
                 self.errors_.extend([0] * (self.n_epochs - (epoch + 1)))
                 break

        end_time = time.time()
        print("-" * 30)
        # Print final status only if not converged early
        if errors_in_epoch != 0 or epoch == 0:
            print(f"Training finished after {self.n_epochs} epochs (or stopped early). Final Misclassifications: {errors_in_epoch}")
        print(f"Final weights: {self.weights_}")
        print(f"Final bias: {self.bias_}")
        return self

    def net_input(self, X):
        X_array = np.atleast_1d(X)
        return np.dot(X_array, self.weights_) + self.bias_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# --- Plotting Function (Identical) ---
def plot_decision_boundary(X, y, weights, bias, epoch_num, ax, threshold=0.0, title_suffix=""):
    """Plots data points and the decision boundary for a given epoch."""
    ax.clear()
    cmap = plt.cm.get_cmap('viridis', 2)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=50, label='Data Points')

    # Adjust plot limits slightly for 0/1 data
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1 = np.linspace(x1_min, x1_max, 100)
    w1, w2 = weights
    b = bias

    if abs(w2) > 1e-6:
        xx2 = (-w1 * xx1 - b) / w2
        ax.plot(xx1, xx2, 'r--', lw=2, label='Decision Boundary')
    elif abs(w1) > 1e-6:
        x1_boundary = -b / w1
        ax.axvline(x=x1_boundary, color='r', linestyle='--', lw=2, label='Decision Boundary')

    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_xticks([0, 1]) # Set ticks specifically for 0/1 inputs
    ax.set_yticks([0, 1])
    ax.set_title(f'Epoch: {epoch_num} {title_suffix}')
    ax.legend(*scatter.legend_elements(), title="Classes", loc='upper left')
    if 'Decision Boundary' in [l.get_label() for l in ax.lines]:
        ax.legend(loc='lower right') # Place boundary legend better for XOR
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.grid(True)

# --- Function to Save GIF (Identical) ---
def save_learning_gif(perceptron, X, y, filename="perceptron_learning.gif", duration_per_frame=0.03):
    """
    Generates plots for each epoch state and saves them as an animated GIF.
    """
    print(f"\nGenerating frames for GIF: {filename}...")
    fig, ax = plt.subplots()
    frames = []

    weights_history = perceptron.history_['weights']
    bias_history = perceptron.history_['bias']
    errors_history = perceptron.history_['errors']
    num_epochs_stored = len(weights_history)

    if num_epochs_stored == 0:
        print("Warning: No history found in perceptron. Cannot create GIF.")
        plt.close(fig)
        return

    for i in range(num_epochs_stored):
        epoch_num = i
        weights = weights_history[i]
        bias = bias_history[i]
        errors = errors_history[i]

        title_suffix = f"| Misclassified: {errors}" if errors is not None else "| Initial State"

        plot_decision_boundary(X, y, weights, bias, epoch_num, ax, threshold=0.0, title_suffix=title_suffix)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frame_image = imageio.imread(buf)
        frames.append(frame_image)
        buf.close()

        print(f"  Frame {i+1}/{num_epochs_stored} generated.", end='\r')

        # Stop if converged
        if (errors == 0 and epoch_num > 0):
             print(f"\nConvergence detected at epoch {epoch_num}. Adding final frame.")
             for i in range(HOLDLASTFRAME):
                 frames.append(frame_image)
             break

    plt.close(fig)
    print(f"\nSaving GIF with {len(frames)} frames...")

    try:
        imageio.mimsave(filename, frames, duration=duration_per_frame) # Use duration directly
        print(f"Successfully saved GIF to '{filename}'")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("Please ensure the 'imageio' library is installed correctly.")


# === Example Usage for XOR ===

# 1. Define XOR function data
X_XOR = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_XOR = np.array([0, 1, 1, 0]) # XOR outputs

# 2. Create and Train the STEP FUNCTION Perceptron on XOR data
print("\n" + "="*30)
print("Training STEP FUNCTION Perceptron on XOR Gate")
print("="*30)
# Use fewer epochs as XOR usually converges quickly
ppn_XOR = Perceptron(learning_rate=0.001, n_epochs=150, random_state=1)

# Train and store history (plot_interval=1 stores every epoch, needed for GIF)
ppn_XOR.fit(X_XOR, y_XOR, plot_interval=1)

# 3. Save the learning process as a GIF
# Use a slightly longer duration per frame as there might be fewer frames
save_learning_gif(ppn_XOR, X_XOR, y_XOR, filename="XOR_perceptron_learning.gif", duration_per_frame=0.1)

# 4. Optional: Test final predictions
print("\n--- Final Predictions for XOR Gate ---")
for xi, target in zip(X_XOR, y_XOR):
    prediction = ppn_XOR.predict(xi)
    print(f"Input: {xi} -> Target: {target} -> Predicted: {prediction} {'(Correct)' if prediction == target else '(Incorrect)'}")