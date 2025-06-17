# Required libraries: numpy, matplotlib, imageio
# Install using: pip install numpy matplotlib imageio
# Trains 3-input XOR (150) and XNOR (105), generates static loss plot
# and animated ROTATING GIFs showing decision regions via 100 stacked RdBu contour slices.
# Target points plotted as Red (0) and Blue (1) to match prediction regions.
# Warning: GIF generation with many slices/frames is slow and memory intensive!

import numpy as np
import time
import matplotlib.pyplot as plt
# Import the 3D plotting toolkit
from mpl_toolkits.mplot3d import Axes3D
import imageio # Required for saving GIFs
import io      # Required for saving plots to memory buffer
import warnings
import matplotlib.patches as mpatches # For custom legends

# --- MLP Class (with History Storage) ---
class MLP:
    """
    MLP for single-output binary classification. Includes history storage
    for visualization and early stopping based on loss. Optional momentum.
    Uses gradients derived from Binary Cross-Entropy. Assumes n_output=1.
    """
    def __init__(self, n_input, n_hidden, n_output=1, learning_rate=0.1, n_epochs=10000, random_state=1, momentum=0.0):
        """Initializes the MLP."""
        if n_output != 1:
            warnings.warn("MLP forced to n_output=1 for binary function task.", UserWarning)
            n_output = 1
        self.n_input = n_input; self.n_hidden = n_hidden; self.n_output = n_output
        self.learning_rate = learning_rate; self.n_epochs = n_epochs
        self.random_state = random_state; self.momentum = momentum
        self.loss_history_ = []
        # History stores parameters at specified intervals for visualization
        self.history_ = {'weights_h': [], 'bias_h': [], 'weights_o': [], 'bias_o': [], 'loss': [], 'epoch': []}

        rgen = np.random.RandomState(self.random_state)
        self.weights_h_ = rgen.normal(loc=0.0, scale=0.1, size=(self.n_input, self.n_hidden))
        self.bias_h_ = np.zeros((1, self.n_hidden))
        self.weights_o_ = rgen.normal(loc=0.0, scale=0.1, size=(self.n_hidden, self.n_output))
        self.bias_o_ = np.zeros((1, self.n_output))

    def _sigmoid(self, z):
        z_clipped = np.clip(z, -500, 500); return 1.0 / (1.0 + np.exp(-z_clipped))
    def _sigmoid_derivative(self, sigmoid_output):
        return sigmoid_output * (1.0 - sigmoid_output)
    def _forward(self, X, weights_h, bias_h, weights_o, bias_o):
        Z_h = X @ weights_h + bias_h; A_h = self._sigmoid(Z_h)
        Z_o = A_h @ weights_o + bias_o; A_o = self._sigmoid(Z_o)
        return A_h, A_o, Z_h, Z_o
    def _compute_loss(self, y_true, y_pred):
        y_true_reshaped = y_true.reshape(y_pred.shape)
        loss = 0.5 * np.mean((y_true_reshaped - y_pred)**2); return loss

    # --- fit method includes history storage ---
    def fit(self, X, y, plot_interval=100, loss_threshold=1e-5, patience=50):
        """ Trains the MLP. Stores history at plot_interval. Includes early stopping."""
        y_col = y.reshape(-1, 1); self.loss_history_ = []
        self.history_ = {'weights_h': [], 'bias_h': [], 'weights_o': [], 'bias_o': [], 'loss': [], 'epoch': []} # Reset history
        weights_h=self.weights_h_.copy(); bias_h=self.bias_h_.copy()
        weights_o=self.weights_o_.copy(); bias_o=self.bias_o_.copy()
        prev_update_wh=np.zeros_like(weights_h); prev_update_bh=np.zeros_like(bias_h)
        prev_update_wo=np.zeros_like(weights_o); prev_update_bo=np.zeros_like(bias_o)
        effective_patience=patience if patience is not None else 50
        effective_threshold=loss_threshold if loss_threshold is not None else 1e-5
        start_time=time.time(); epochs_below_threshold_count=0; final_epoch=self.n_epochs

        # Store initial state
        if plot_interval is not None and plot_interval > 0: # Check interval > 0
             self.history_['weights_h'].append(weights_h.copy()); self.history_['bias_h'].append(bias_h.copy())
             self.history_['weights_o'].append(weights_o.copy()); self.history_['bias_o'].append(bias_o.copy())
             self.history_['loss'].append(None); self.history_['epoch'].append(0)

        for epoch in range(self.n_epochs):
            final_epoch = epoch
            A_h, A_o, Z_h, Z_o = self._forward(X, weights_h, bias_h, weights_o, bias_o)
            loss = self._compute_loss(y_col, A_o); self.loss_history_.append(loss)
            delta_output=A_o - y_col; error_hidden=delta_output @ weights_o.T
            d_sigmoid_h=self._sigmoid_derivative(A_h); delta_hidden=error_hidden * d_sigmoid_h
            grad_weights_o=A_h.T @ delta_output; grad_bias_o=np.sum(delta_output, axis=0, keepdims=True)
            grad_weights_h=X.T @ delta_hidden; grad_bias_h=np.sum(delta_hidden, axis=0, keepdims=True)
            update_wh=(self.momentum*prev_update_wh)-(self.learning_rate*grad_weights_h)
            update_bh=(self.momentum*prev_update_bh)-(self.learning_rate*grad_bias_h)
            update_wo=(self.momentum*prev_update_wo)-(self.learning_rate*grad_weights_o)
            update_bo=(self.momentum*prev_update_bo)-(self.learning_rate*grad_bias_o)
            weights_h+=update_wh; bias_h+=update_bh; weights_o+=update_wo; bias_o+=update_bo
            prev_update_wh=update_wh; prev_update_bh=update_bh; prev_update_wo=update_wo; prev_update_bo=update_bo

            # Store History at Intervals
            store_this_epoch = (plot_interval is not None and plot_interval > 0) and ((epoch + 1) % plot_interval == 0)
            if store_this_epoch:
                 self.history_['weights_h'].append(weights_h.copy()); self.history_['bias_h'].append(bias_h.copy())
                 self.history_['weights_o'].append(weights_o.copy()); self.history_['bias_o'].append(bias_o.copy())
                 self.history_['loss'].append(loss); self.history_['epoch'].append(epoch + 1)

            # Early Stopping Check
            if effective_threshold is not None and effective_threshold > 0:
                 if loss < effective_threshold: epochs_below_threshold_count += 1
                 else: epochs_below_threshold_count = 0
                 if epochs_below_threshold_count >= effective_patience: final_epoch=epoch+1; break

        self.weights_h_=weights_h; self.bias_h_=bias_h; self.weights_o_=weights_o; self.bias_o_=bias_o
        # Ensure final state is stored
        if plot_interval is not None and plot_interval > 0:
            last_hist_epoch = self.history_['epoch'][-1] if self.history_['epoch'] else -1
            actual_epochs_run = final_epoch + 1 if final_epoch < self.n_epochs else self.n_epochs
            if actual_epochs_run >= last_hist_epoch:
                 if not (store_this_epoch and actual_epochs_run == epoch + 1) :
                     if actual_epochs_run > last_hist_epoch:
                         self.history_['weights_h'].append(weights_h.copy()); self.history_['bias_h'].append(bias_h.copy())
                         self.history_['weights_o'].append(weights_o.copy()); self.history_['bias_o'].append(bias_o.copy())
                         self.history_['loss'].append(self.loss_history_[-1]); self.history_['epoch'].append(actual_epochs_run)
        end_time=time.time(); actual_epochs_run=final_epoch+1 if final_epoch<self.n_epochs else self.n_epochs
        print(f"  Training finished after {actual_epochs_run} epochs. Final Loss: {self.loss_history_[-1]:.6f}")
        return self

    def predict(self, X):
        """Makes predictions using the final trained weights."""
        _, probabilities, _, _ = self._forward(X, self.weights_h_, self.bias_h_, self.weights_o_, self.bias_o_)
        predictions = np.where(probabilities >= 0.5, 1, 0); return predictions.flatten()

# --- Helper Function to Generate Target Vector for N Inputs (Unchanged) ---
def get_target_vector(func_index, n_inputs):
    num_combinations = 2**n_inputs; max_func_index = 2**num_combinations - 1
    if not 0 <= func_index <= max_func_index: raise ValueError(f"func_index out of range")
    binary_string = format(func_index, f'0{num_combinations}b')
    target_vector = np.array([int(bit) for bit in reversed(binary_string)]); return target_vector

# --- Function to generate all input combinations for N inputs (Unchanged) ---
def generate_inputs(n_inputs):
    num_combinations = 2**n_inputs; inputs = np.zeros((num_combinations, n_inputs), dtype=int)
    for i in range(num_combinations):
        binary_repr = format(i, f'0{n_inputs}b'); inputs[i] = [int(bit) for bit in binary_repr]
    return inputs

# --- Plotting Function using Uniform COLOR 3D Contour Slices & Color-Matched Target/Prediction Legend ---
def plot_3d_contour_slices(X, y, mlp_params, ax, title="MLP 3D Decision Slices", num_slices=11):
    """
    Plots data points (Red/Blue) and the MLP final prediction regions
    using contour slices stacked along Z axis with uniform RdBu appearance.
    Includes Red/Blue legend for predicted regions and target points.
    """
    ax.clear(); n_inputs = mlp_params['n_input']
    if n_inputs != 3: ax.text(0.5,0.5,'Plotting only supported for 3 inputs',ha='center',va='center'); ax.set_title(title,fontsize=9); return

    cmap_plot = plt.cm.RdBu # Use RdBu for both points and contours

    # --- Plot Original Data Points ---
    # Color points using the RdBu map based on target y (0 -> Red, 1 -> Blue)
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap_plot,
                         s=150, edgecolors='k', depthshade=True, label='Target Data', zorder=num_slices + 1)

    # --- Create 2D Grid for Slices ---
    h = .05; margin = 0.2
    x_min, x_max = 0 - margin, 1 + margin; y_min, y_max = 0 - margin, 1 + margin
    z_min, z_max = 0 - margin, 1 + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # --- Plot Contour Slices along Z (Input C) axis ---
    slice_values = np.linspace(0, 1, num_slices)
    uniform_alpha = 0.15 # Keep alpha relatively low for many slices

    temp_mlp = MLP(mlp_params['n_input'], mlp_params['n_hidden']) # Temp instance for prediction
    for idx, c_val in enumerate(slice_values):
        grid_input = np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), c_val)]
        _, Z_probs, _, _ = temp_mlp._forward(grid_input, mlp_params['weights_h'], mlp_params['bias_h'], mlp_params['weights_o'], mlp_params['bias_o'])
        Z = Z_probs.reshape(xx.shape)
        # Plot using the SAME RdBu colormap and alpha for all slices
        ax.contourf(xx, yy, Z, cmap=cmap_plot, alpha=uniform_alpha,
                    levels=[-0.1, 0.5, 1.1], # Sharp boundary at 0.5 -> defines 2 regions
                    zdir='z', offset=c_val, zorder=idx+1) # Ensure order

    # --- Create Legend Handles for Predictions (Red & Blue) ---
    pred0_color = cmap_plot(0.25) # Representative color for Pred=0 (Reddish)
    pred1_color = cmap_plot(0.75) # Representative color for Pred=1 (Blueish)
    patch_pred0 = mpatches.Patch(color=pred0_color, label='Pred=0 Region', alpha=0.7)
    patch_pred1 = mpatches.Patch(color=pred1_color, label='Pred=1 Region', alpha=0.7)

    # --- Create Legend Handles for Target Points (Red & Blue) ---
    # Use legend_elements which respects the colormap used in scatter
    try:
        point_handles, point_labels = scatter.legend_elements(prop="colors", alpha=0.9) # Use higher alpha for legend visibility
        target_labels = [f'Target {l}' for l in np.unique(y)] # Assumes y contains 0 and 1
        # Ensure order matches colors if necessary (RdBu: 0 is low/Red, 1 is high/Blue)
        if len(point_handles) == 2 and np.unique(y)[0] == 0: # Standard case 0, 1
             target_handles = point_handles
        else: # Fallback or different label order
             target_handles = [mpatches.Patch(color=cmap_plot(0.25), label='Target 0'),
                               mpatches.Patch(color=cmap_plot(0.75), label='Target 1')]
             target_labels = ['Target 0', 'Target 1']

    except Exception as e:
        print(f"Warning: Could not create target legend elements automatically - {e}")
        # Manual fallback if legend_elements fails
        target_handles = [mpatches.Patch(color=cmap_plot(0.25), label='Target 0'),
                          mpatches.Patch(color=cmap_plot(0.75), label='Target 1')]
        target_labels = ['Target 0', 'Target 1']


    # --- Plot Formatting ---
    ax.set_xlabel('Input A'); ax.set_ylabel('Input B'); ax.set_zlabel('Input C')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.set_title(title, fontsize=12)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    # ax.view_init(elev=25., azim=-55) # Static view angle for plotting function - REMOVED for GIF rotation

    # Combine legends for predictions and targets
    ax.legend(handles=[patch_pred0, patch_pred1] + target_handles,
              labels=['Pred=0 Region', 'Pred=1 Region'] + target_labels,
              loc='upper left', fontsize='small')

    ax.grid(True)


# --- UPDATED Function to Save Single Function GIF with ROTATION ---
def save_single_func_gif(history, X_inputs, y_target, mlp_config, filename="mlp_func_anim.gif", duration_per_frame=0.1, num_slices=11, start_angle=-55, rotation_degrees=180):
    """
    Generates GIF for one function using 3D contour slice plotting,
    rotating the view angle over frames.
    """
    if not history or not history.get('weights_h') or len(history['weights_h']) <= 1:
        print(f"Warning: Insufficient history for {filename} (found {len(history.get('weights_h',[]))} frames). Skipping GIF.")
        return
    if not imageio: print("Error: imageio library not found."); return

    print(f"\nGenerating frames for Rotating GIF: {filename}...")
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    frames = []
    weights_h_hist = history['weights_h']; bias_h_hist = history['bias_h']
    weights_o_hist = history['weights_o']; bias_o_hist = history['bias_o']
    loss_hist_frames = history['loss']; epoch_hist = history['epoch']
    num_frames = len(weights_h_hist)

    delta_azim = rotation_degrees / (num_frames - 1) if num_frames > 1 else 0

    for frame_idx in range(num_frames):
        print(f"  Generating Frame {frame_idx + 1}/{num_frames}...", end='\r')
        mlp_params = {
            'n_input': mlp_config['n_input'], 'n_hidden': mlp_config['n_hidden'],
            'weights_h': weights_h_hist[frame_idx], 'bias_h': bias_h_hist[frame_idx],
            'weights_o': weights_o_hist[frame_idx], 'bias_o': bias_o_hist[frame_idx]
        }
        loss = loss_hist_frames[frame_idx]; epoch_num = epoch_hist[frame_idx]
        title_suffix = f"| E:{epoch_num}, L:{loss:.3f}" if loss is not None else f"| E:{epoch_num}"
        try: func_idx_str = filename.split('_')[2]
        except: func_idx_str = "?"
        title = f"F{func_idx_str} Slices\n{title_suffix}"

        # Call the contour slice plotting function
        plot_3d_contour_slices(X_inputs, y_target, mlp_params, ax, title=title, num_slices=num_slices)

        # Set view angle for this frame
        current_azim = (start_angle + frame_idx * delta_azim)
        ax.view_init(elev=25, azim=current_azim) # Keep elevation constant, change azimuth

        # Save frame to buffer
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=90); buf.seek(0)
        frame_image = imageio.imread(buf); frames.append(frame_image); buf.close()

    plt.close(fig)
    print(f"\nSaving Rotating GIF with {len(frames)} frames...")
    try:
        imageio.mimsave(filename, frames, duration=duration_per_frame)
        print(f"Successfully saved GIF to '{filename}'")
    except Exception as e: print(f"Error saving GIF: {e}")


# === Main Execution Block ===

if __name__ == "__main__":
    # 1. Define Input Parameters
    NUM_INPUTS = 3
    X_inputs = generate_inputs(NUM_INPUTS)

    # Indices for 3-input XOR (150) and XNOR (105)
    functions_to_train = { 150: "3-XOR", 105: "3-XNOR" }
    func_names = functions_to_train

    print(f"Input Combinations (A, B, C order):\n{X_inputs}")
    print(f"\nAttempting to train MLPs for 3-input XOR and XNOR")

    convergence_results = {}

    # 2. Configure MLP parameters
    input_size = NUM_INPUTS; hidden_size = 5; output_size = 1
    learn_rate = 0.1; epochs = 15000; rand_state_base = 0
    early_stop_threshold = 1e-5; early_stop_patience = 300
    momentum_value = 0.85
    # --- GIF / History Parameters ---
    plot_save_interval = 100 # Store history/generate frame every N epochs
    num_plot_slices = 100    # Number of contour slices for visualization

    # 3. Train MLP FOR XOR AND XNOR ONLY
    print("\n" + "="*40)
    print(f"Training specific functions: {list(functions_to_train.keys())}")
    print(f"Using Momentum: {momentum_value}, Hidden Units: {hidden_size}, Max Epochs: {epochs}")
    print(f"Storing history/GIF frames every {plot_save_interval} epochs using {num_plot_slices} color slices.") # Updated print
    print("="*40)
    print(f"WARNING: Generating rotating GIFs with {num_plot_slices} slices and potentially many frames will be very slow!")

    start_bulk_time = time.time()

    for i, name in functions_to_train.items():
        print(f"\n--- Training Function {i}: {name} ---")
        y_target = get_target_vector(i, n_inputs=NUM_INPUTS)
        mlp = MLP(n_input=input_size, n_hidden=hidden_size, n_output=output_size,
                  learning_rate=learn_rate, n_epochs=epochs,
                  random_state=rand_state_base + i, momentum=momentum_value)
        mlp.fit(X_inputs, y_target,
                plot_interval=plot_save_interval,
                loss_threshold=early_stop_threshold, patience=early_stop_patience)
        predictions = mlp.predict(X_inputs); converged = np.array_equal(predictions, y_target)
        convergence_results[i] = converged
        print(f"Target:   {y_target}\nPredicted:{predictions}")
        print(f"Result:   {'SUCCESS' if converged else 'FAILURE'}")

        # --- Generate ROTATING GIF for this specific function ---
        print(f"Generating Rotating GIF for Function {i} (Using {num_plot_slices} slices - this WILL take time)...")
        mlp_config = {'n_input': input_size, 'n_hidden': hidden_size}
        gif_filename = f"mlp_func_{i}_3input_{num_plot_slices}slices_color_rotating.gif" # Changed filename
        gif_duration = 0.15 if len(mlp.history_['epoch']) < 150 else 0.1
        save_single_func_gif(mlp.history_, X_inputs, y_target, mlp_config,
                             filename=gif_filename, duration_per_frame=gif_duration,
                             num_slices=num_plot_slices, start_angle=-70, rotation_degrees=180)

    end_bulk_time = time.time()
    print(f"\n{'='*40}\nFinished training loop in {end_bulk_time - start_bulk_time:.2f} seconds.")

    # 4. Report Results for XOR/XNOR
    print("\n--- Convergence Report ---")
    for i in functions_to_train:
        status = "SUCCESS" if convergence_results.get(i, False) else "FAILURE"
        print(f"Function {i} ({functions_to_train[i]}): {status}")
    if all(convergence_results.get(i, False) for i in functions_to_train): print("\nBoth XOR and XNOR converged successfully!")
    else: print("\nOne or both functions may have failed to converge. Check results and GIFs.")

    print("\n--- All tasks complete ---")

    # Optional: Plot loss curve for the last function trained
    try: import matplotlib.pyplot as plt; PLOT_ENABLED = True
    except ImportError: PLOT_ENABLED = False
    if PLOT_ENABLED and mlp.loss_history_:
        plt.figure()
        plot_step = max(1, len(mlp.loss_history_) // 1000)
        plt.plot(range(1, len(mlp.loss_history_) + 1, plot_step), mlp.loss_history_[::plot_step])
        plt.xlabel(f'Epochs (x{plot_step})'); plt.ylabel('Mean Squared Error Loss')
        plt.title(f'MLP Training Loss Curve for Last Trained Function ({i})'); plt.grid(True)
        plt.yscale('log'); plt.show()

