# Required libraries: numpy, matplotlib, imageio
# Install using: pip install numpy matplotlib imageio
# Trains 4-input XOR (27030) using Sigmoid hidden layer.
# Automatically retries with different random seeds if convergence fails.
# Generates loss plot and animated ROTATING GIF showing side-by-side 3D contour slices,
# LINGERING with rotation on the final frame.
# Warning: Computationally intensive!

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
    Handles variable number of inputs.
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
        if plot_interval is not None and plot_interval > 0:
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
    if not 0 <= func_index <= max_func_index: raise ValueError(f"func_index out of range: {func_index}")
    binary_string = format(func_index, f'0{num_combinations}b')
    target_vector = np.array([int(bit) for bit in reversed(binary_string)]); return target_vector

# --- Function to generate all input combinations for N inputs (Unchanged) ---
def generate_inputs(n_inputs):
    num_combinations = 2**n_inputs; inputs = np.zeros((num_combinations, n_inputs), dtype=int)
    for i in range(num_combinations):
        binary_repr = format(i, f'0{n_inputs}b'); inputs[i] = [int(bit) for bit in binary_repr]
    return inputs

# --- Plotting Function using Uniform Grayscale 3D Contour Slices ---
def plot_4d_slices_in_3d(X, y, mlp_params, fixed_d_value, ax, subplot_title="MLP 3D Slice", num_slices=11):
    """
    Plots data points and MLP prediction regions for 3D slice (A, B, C)
    at a fixed value for the 4th input (D) on the provided axes. Uses Grayscale.
    """
    ax.clear(); n_inputs = mlp_params['n_input']
    if n_inputs != 4: ax.text(0.5,0.5,'Plotting requires 4 inputs',ha='center',va='center'); ax.set_title(subplot_title,fontsize=9); return

    X_slice = X[X[:, 3] == fixed_d_value]; y_slice = y[X[:, 3] == fixed_d_value]
    cmap_points = plt.cm.RdBu
    scatter = ax.scatter(X_slice[:, 0], X_slice[:, 1], X_slice[:, 2], c=y_slice, cmap=cmap_points,
                         s=100, edgecolors='k', depthshade=True, label='Target Data', zorder=num_slices + 1)

    h = .1; margin = 0.2
    x_min, x_max = 0 - margin, 1 + margin; y_min, y_max = 0 - margin, 1 + margin
    z_min, z_max = 0 - margin, 1 + margin
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h), indexing='ij')
    grid_input = np.c_[xx.ravel(), yy.ravel(), zz.ravel(), np.full_like(xx.ravel(), fixed_d_value)]

    temp_mlp = MLP(mlp_params['n_input'], mlp_params['n_hidden'])
    _, Z_probs, _, _ = temp_mlp._forward(grid_input, mlp_params['weights_h'], mlp_params['bias_h'], mlp_params['weights_o'], mlp_params['bias_o'])
    Z_grid_shape = (xx.shape[0], xx.shape[1], xx.shape[2])
    Z_probs_grid = Z_probs.reshape(Z_grid_shape)

    cmap_contour = plt.cm.Greys; slice_values_c = np.linspace(0, 1, num_slices); uniform_alpha = 0.15
    xx_2d, yy_2d = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    for idx, c_val in enumerate(slice_values_c):
        c_idx = np.argmin(np.abs(np.arange(z_min, z_max, h) - c_val))
        if c_idx < Z_probs_grid.shape[2]:
             Z_slice = Z_probs_grid[:, :, c_idx]
             ax.contourf(xx_2d, yy_2d, Z_slice, cmap=cmap_contour, alpha=uniform_alpha,
                         levels=[-0.1, 0.5, 1.1], zdir='z', offset=c_val, zorder=idx+1)

    pred0_color = 'white'; pred1_color = 'black'
    patch_pred0 = mpatches.Patch(facecolor=pred0_color, label='Pred=0 Region', edgecolor='grey')
    patch_pred1 = mpatches.Patch(facecolor=pred1_color, label='Pred=1 Region', edgecolor='grey')

    ax.set_xlabel('Input A'); ax.set_ylabel('Input B'); ax.set_zlabel('Input C')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.set_title(subplot_title, fontsize=10)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
    # View angle set dynamically in GIF loop
    try:
        point_handles, point_labels = scatter.legend_elements(prop="colors", alpha=0.8)
        ax.legend(handles=[patch_pred0, patch_pred1] + point_handles,
                  labels=['Pred=0 Region', 'Pred=1 Region'] + [f'Target {l}' for l in np.unique(y_slice)],
                  loc='upper left', fontsize='small')
    except Exception as e: print(f"Warning: Could not create full legend - {e}"); ax.legend(handles=[scatter])
    ax.grid(True)


# --- UPDATED Function to Save Side-by-Side 4D GIF with Rotation & Linger ---
def save_4d_side_by_side_gif(history, X_inputs, y_target, mlp_config, func_index, func_name,
                            filename="mlp_4d_sbs_anim.gif", duration_per_frame=0.15, num_slices=15,
                            start_angle=-70, rotation_degrees=180, linger_frames=15): # Added linger_frames
    """
    Generates GIF for one 4-input function using side-by-side 3D contour slices
    (for D=0 and D=1) and rotating the view angle over frames.
    Includes extra frames at the end showing the final state rotating.
    """
    if not history or not history.get('weights_h') or len(history['weights_h']) <= 1:
        print(f"Warning: Insufficient history for {filename} (found {len(history.get('weights_h',[]))} frames). Skipping GIF.")
        return
    if not imageio: print("Error: imageio library not found."); return

    print(f"\nGenerating frames for Side-by-Side Rotating GIF: {filename}...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    frames = []
    weights_h_hist = history['weights_h']; bias_h_hist = history['bias_h']
    weights_o_hist = history['weights_o']; bias_o_hist = history['bias_o']
    loss_hist_frames = history['loss']; epoch_hist = history['epoch']
    num_frames_history = len(weights_h_hist) # Number of saved states during training

    # --- Calculate rotation delta based on total frames (training + linger) ---
    total_frames_for_rotation = num_frames_history + linger_frames
    delta_azim = rotation_degrees / (total_frames_for_rotation - 1) if total_frames_for_rotation > 1 else 0

    # --- Generate frames for training history ---
    for frame_idx in range(num_frames_history):
        print(f"  Generating Training Frame {frame_idx + 1}/{num_frames_history}...", end='\r')
        mlp_params = { # Package params for plotting function
            'n_input': mlp_config['n_input'], 'n_hidden': mlp_config['n_hidden'],
            'weights_h': weights_h_hist[frame_idx], 'bias_h': bias_h_hist[frame_idx],
            'weights_o': weights_o_hist[frame_idx], 'bias_o': bias_o_hist[frame_idx]
        }
        loss = loss_hist_frames[frame_idx]; epoch_num = epoch_hist[frame_idx]
        title_suffix = f"| E:{epoch_num}, L:{loss:.3f}" if loss is not None else f"| E:{epoch_num}"
        current_azim = (start_angle + frame_idx * delta_azim)

        # Plot D=0 and D=1 for this history point
        plot_4d_slices_in_3d(X_inputs, y_target, mlp_params, 0, axes[0], subplot_title=f"D=0 {title_suffix}", num_slices=num_slices)
        axes[0].view_init(elev=25, azim=current_azim)
        plot_4d_slices_in_3d(X_inputs, y_target, mlp_params, 1, axes[1], subplot_title=f"D=1 {title_suffix}", num_slices=num_slices)
        axes[1].view_init(elev=25, azim=current_azim)

        fig.suptitle(f"MLP Decision Regions for F{func_index}: {func_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Save frame
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=90); buf.seek(0)
        frame_image = imageio.imread(buf); frames.append(frame_image); buf.close()

    # --- Generate LINGER frames using FINAL state ---
    if num_frames_history > 0: # Ensure there's a final state
        print("\n  Generating Linger Frames...", end='\r')
        # Get final parameters
        final_mlp_params = {
            'n_input': mlp_config['n_input'], 'n_hidden': mlp_config['n_hidden'],
            'weights_h': weights_h_hist[-1], 'bias_h': bias_h_hist[-1],
            'weights_o': weights_o_hist[-1], 'bias_o': bias_o_hist[-1]
        }
        final_loss = loss_hist_frames[-1]; final_epoch_num = epoch_hist[-1]
        title_suffix = f"| E:{final_epoch_num}, L:{final_loss:.3f} (Final State)"

        for linger_idx in range(linger_frames):
            print(f"  Generating Linger Frame {linger_idx + 1}/{linger_frames}...", end='\r')
            # Continue rotation from the last training frame's angle
            frame_num_overall = num_frames_history + linger_idx
            current_azim = (start_angle + frame_num_overall * delta_azim)

            # Plot D=0 and D=1 using FINAL parameters
            plot_4d_slices_in_3d(X_inputs, y_target, final_mlp_params, 0, axes[0], subplot_title=f"D=0 {title_suffix}", num_slices=num_slices)
            axes[0].view_init(elev=25, azim=current_azim)
            plot_4d_slices_in_3d(X_inputs, y_target, final_mlp_params, 1, axes[1], subplot_title=f"D=1 {title_suffix}", num_slices=num_slices)
            axes[1].view_init(elev=25, azim=current_azim)

            fig.suptitle(f"MLP Decision Regions for F{func_index}: {func_name}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            # Save frame
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=90); buf.seek(0)
            frame_image = imageio.imread(buf); frames.append(frame_image); buf.close()

    plt.close(fig) # Close plot figure AFTER all loops finish
    print(f"\nSaving Side-by-Side Rotating GIF with {len(frames)} frames...")
    try:
        imageio.mimsave(filename, frames, duration=duration_per_frame)
        print(f"Successfully saved GIF to '{filename}'")
    except Exception as e: print(f"Error saving GIF: {e}")


# === Main Execution Block ===

if __name__ == "__main__":
    # 1. Define Input Parameters
    NUM_INPUTS = 4
    X_inputs = generate_inputs(NUM_INPUTS)

    # Index for 4-input XOR: 27030 decimal
    function_index = 27030
    function_name = "4-XOR"
    functions_to_train = { function_index: function_name } # Train only this one

    print(f"Input Combinations ({NUM_INPUTS} Inputs):\n{X_inputs}")
    print(f"\nAttempting to train MLP for {function_name} ({function_index}) with retries...")

    # 2. Configure MLP parameters
    input_size = NUM_INPUTS
    hidden_size = 8
    output_size = 1
    learn_rate = 0.1
    epochs = 25000 # Max epochs per attempt
    rand_state_base = 0 # Starting seed
    early_stop_threshold = 1e-5
    early_stop_patience = 500
    momentum_value = 0.9
    # --- GIF / History Parameters ---
    plot_save_interval = 250 # Store history less often
    num_plot_slices = 50    # Reduced slices for performance
    num_linger_frames = 20  # Number of extra frames to linger at the end

    # 3. Train MLP FOR 4-XOR ONLY with Retries
    print("\n" + "="*40)
    print(f"Training Function {function_index}: {function_name}")
    print(f"Using Sigmoid Hidden Layer, Momentum: {momentum_value}, Hidden Units: {hidden_size}, Max Epochs: {epochs}")
    print(f"Will retry up to max_retries times with different seeds if convergence fails.")
    print(f"Storing history/GIF frames every {plot_save_interval} epochs using {num_plot_slices} grayscale slices.")
    print("="*40)
    print(f"WARNING: Generating rotating 4D GIF will be VERY slow!")

    start_bulk_time = time.time()
    converged = False
    attempt = 0
    max_retries = 5 # Maximum number of attempts with different seeds
    final_mlp = None # To store the successfully converged MLP

    while not converged and attempt < max_retries:
        attempt += 1
        current_random_state = rand_state_base + attempt - 1
        print(f"\n--- Attempt {attempt}/{max_retries} (Seed: {current_random_state}) ---")

        y_target = get_target_vector(function_index, n_inputs=NUM_INPUTS)

        mlp = MLP(n_input=input_size, n_hidden=hidden_size, n_output=output_size,
                  learning_rate=learn_rate, n_epochs=epochs,
                  random_state=current_random_state,
                  momentum=momentum_value)

        mlp.fit(X_inputs, y_target,
                plot_interval=plot_save_interval,
                loss_threshold=early_stop_threshold,
                patience=early_stop_patience)

        predictions = mlp.predict(X_inputs)
        converged = np.array_equal(predictions, y_target)

        print(f"Attempt {attempt} Target:   {y_target}")
        print(f"Attempt {attempt} Predicted:{predictions}")
        print(f"Attempt {attempt} Result:   {'SUCCESS' if converged else 'FAILURE'}")

        if converged:
            final_mlp = mlp; break

    end_bulk_time = time.time()
    print(f"\n{'='*40}\nFinished training attempts in {end_bulk_time - start_bulk_time:.2f} seconds.")

    # 4. Generate Side-by-Side Rotating 4D GIF with Linger *only if converged*
    if converged and final_mlp is not None:
        print("\n--- SUCCESS: Generating Side-by-Side Rotating 4D Decision Slice GIF with Linger ---")
        mlp_config = {'n_input': input_size, 'n_hidden': hidden_size}
        gif_filename = f"mlp_func_{function_index}_4input_{num_plot_slices}slices_sbs_rot_linger_gray_seed{final_mlp.random_state}.gif"
        num_hist_frames = len(final_mlp.history_['epoch'])
        gif_duration = 0.2 if num_hist_frames < 75 else 0.15

        save_4d_side_by_side_gif(final_mlp.history_, X_inputs, y_target, mlp_config, function_index, function_name,
                                 filename=gif_filename,
                                 duration_per_frame=gif_duration,
                                 num_slices=num_plot_slices,
                                 start_angle=-70,
                                 rotation_degrees=180, # Total rotation degrees
                                 linger_frames=num_linger_frames) # Add linger frames
    elif not converged:
        print(f"\nFAILURE: MLP failed to converge for {function_name} after {max_retries} attempts.")
        print("Skipping GIF generation.")
        print("Consider increasing epochs further or tuning other hyperparameters (lr, hidden units, momentum).")

    print("\n--- All tasks complete ---")

    # Optional: Plot loss curve of the last attempt
    try: import matplotlib.pyplot as plt; PLOT_ENABLED = True
    except ImportError: PLOT_ENABLED = False
    if PLOT_ENABLED and 'mlp' in locals() and mlp.loss_history_:
        plt.figure()
        plot_step = max(1, len(mlp.loss_history_) // 1000)
        plt.plot(range(1, len(mlp.loss_history_) + 1, plot_step), mlp.loss_history_[::plot_step])
        plt.xlabel(f'Epochs (x{plot_step})'); plt.ylabel('Mean Squared Error Loss')
        plt.title(f'MLP Training Loss Curve for {function_name} (Last Attempt, Seed {mlp.random_state})'); plt.grid(True)
        plt.yscale('log'); plt.show()

