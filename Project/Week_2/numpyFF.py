import numpy as np
import math
import matplotlib.pyplot as plt
import cProfile # For cProfile
import pstats   # For cProfile
import io       # For cProfile output formatting

try:
    profile # NoQA
except NameError:
    def profile(func):
        return func

# --- Activation Functions ---
# @profile # Example for line_profiler
def tanh(x):
    return np.tanh(x)

# @profile # Example for line_profiler
def sigmoid(x_input):
    """Numerically stable sigmoid function."""
    dtype = x_input.dtype
    if dtype == np.float32:
        exp_limit = 80.0
    elif dtype == np.float64:
        exp_limit = 700.0
    else:
        exp_limit = 30.0

    x_clipped = np.clip(x_input, -exp_limit, exp_limit)
    return np.where(x_clipped >= 0,
                    1 / (1 + np.exp(-x_clipped)),
                    np.exp(x_clipped) / (1 + np.exp(x_clipped)))

# @profile # Example for line_profiler
def relu(x):
    return np.maximum(0, x)

# --- NumPy Convolution Implementations ---

class Conv1d1x1_numpy:
    """1x1 Conv1d in NumPy."""
    def __init__(self, in_channels, out_channels, bias=True, dtype=np.float32):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = np.random.randn(out_channels, in_channels, 1).astype(dtype)
        if bias:
            self.bias = np.random.randn(out_channels).astype(dtype)
        else:
            self.bias = np.zeros(out_channels, dtype=dtype)
        self.has_bias = bias
        self.dtype = dtype

    # @profile # Example for line_profiler
    def __call__(self, x_input):
        B, C_in, T = x_input.shape
        assert C_in == self.in_channels, f"Input channel mismatch. Expected {self.in_channels}, got {C_in}"
        w_reshaped = self.weights[:, :, 0]
        output = np.einsum('bct,oc->bot', x_input, w_reshaped)
        output += self.bias[np.newaxis, :, np.newaxis]
        return output.astype(self.dtype)

class Conv1d_numpy:
    """General 1D Conv in NumPy with dilation and padding."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, bias=True, dtype=np.float32):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size).astype(dtype)
        if bias:
            self.bias = np.random.randn(out_channels).astype(dtype)
        else:
            self.bias = np.zeros(out_channels, dtype=dtype)
        self.has_bias = bias
        self.dtype = dtype

    @profile # Example for line_profiler
    def __call__(self, x_input):
        B, C_in, T_in = x_input.shape
        assert C_in == self.in_channels, f"Input channel mismatch. Expected {self.in_channels}, got {C_in}"

        if self.padding > 0:
            padded_input = np.pad(x_input, ((0,0), (0,0), (self.padding, self.padding)), mode='constant', constant_values=0)
        else:
            padded_input = x_input
        
        _, _, T_padded = padded_input.shape
        T_out = T_in # Assuming "same" convolution output length
        
        output = np.zeros((B, self.out_channels, T_out), dtype=self.dtype)

        for b_idx in range(B):
            for oc_idx in range(self.out_channels):
                for t_idx in range(T_out):
                    current_sum = 0.0
                    for ic_idx in range(self.in_channels):
                        for k_idx in range(self.kernel_size):
                            idx_in_padded = t_idx + k_idx * self.dilation
                            if 0 <= idx_in_padded < T_padded:
                                current_sum += padded_input[b_idx, ic_idx, idx_in_padded] * \
                                               self.weights[oc_idx, ic_idx, k_idx]
                    output[b_idx, oc_idx, t_idx] = current_sum + self.bias[oc_idx]
        
        if output.shape[2] != T_in: # Adjust if "same" padding calculation isn't perfect
             if output.shape[2] > T_in: output = output[:,:,:T_in]
             elif output.shape[2] < T_in: output = np.pad(output, ((0,0),(0,0),(0, T_in - output.shape[2])), mode='constant')
        return output.astype(self.dtype)

# --- NumPy ResidualBlock Implementation ---
class ResidualBlock_numpy:
    def __init__(self, kernel_size=3, residual_channels=64, gate_channels=128,
                 skip_channels=64, aux_channels=80, dilation=1, bias=True,
                 causal=False, dtype=np.float32):
        self.residual_channels = residual_channels
        self.gate_channels = gate_channels
        self.skip_channels = skip_channels
        self.aux_channels = aux_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.causal = causal

        assert (kernel_size - 1) % 2 == 0, "Kernel size must be odd for non-causal same padding"
        self.padding = (kernel_size - 1) // 2 * dilation

        self.conv = Conv1d_numpy(residual_channels, gate_channels, kernel_size,
                                 dilation=dilation, padding=self.padding, bias=bias, dtype=dtype)
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1_numpy(aux_channels, gate_channels, bias=False, dtype=dtype)
        else:
            self.conv1x1_aux = None

        self.gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1_numpy(self.gate_out_channels, residual_channels, bias=bias, dtype=dtype)
        self.conv1x1_skip = Conv1d1x1_numpy(self.gate_out_channels, skip_channels, bias=bias, dtype=dtype)
        self.dtype = dtype

    # @profile # Example for line_profiler
    def __call__(self, x, c):
        B, C_res, T = x.shape
        residual = x
        x_conv = self.conv(x)
        
        if x_conv.shape[2] != T: # Ensure "same" length
            if x_conv.shape[2] > T: x_conv = x_conv[:, :, :T]
            else: x_conv = np.pad(x_conv, ((0,0),(0,0),(0, T - x_conv.shape[2])), mode='constant')

        xa = x_conv[:, :self.gate_out_channels, :]
        xb = x_conv[:, self.gate_out_channels:, :]

        if c is not None and self.conv1x1_aux is not None:
            assert c.shape[2] == T, f"Time dimension mismatch between x ({T}) and c ({c.shape[2]}) in ResidualBlock"
            assert c.shape[1] == self.aux_channels, "Aux channel mismatch for c"
            c_processed = self.conv1x1_aux(c)
            if c_processed.shape[2] != T: # Ensure "same" length
                 if c_processed.shape[2] > T: c_processed = c_processed[:,:,:T]
                 else: c_processed = np.pad(c_processed, ((0,0),(0,0),(0, T - c_processed.shape[2])), mode='constant')
            ca = c_processed[:, :self.gate_out_channels, :]
            cb = c_processed[:, self.gate_out_channels:, :]
            xa = xa + ca
            xb = xb + cb

        gated_activation = tanh(xa) * sigmoid(xb)
        s = self.conv1x1_skip(gated_activation)
        x_out_resid = self.conv1x1_out(gated_activation)
        x_final = (x_out_resid + residual) * math.sqrt(0.5)
        return x_final.astype(self.dtype), s.astype(self.dtype)

# --- NumPy ParallelWaveGANGenerator Implementation ---
class ParallelWaveGANGenerator_numpy:
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3,
                 layers=30, stacks=3, residual_channels=64,
                 gate_channels=128, skip_channels=64, aux_channels=80,
                 dtype=np.float32):
        self.layers = layers
        self.stacks = stacks
        assert layers % stacks == 0
        self.layers_per_stack = layers // stacks

        self.first_conv = Conv1d1x1_numpy(in_channels, residual_channels, bias=True, dtype=dtype)
        self.upsample_net = None

        self.conv_layers = []
        for layer_idx in range(layers):
            dilation = 2**(layer_idx % self.layers_per_stack)
            conv_block = ResidualBlock_numpy(
                kernel_size=kernel_size, residual_channels=residual_channels,
                gate_channels=gate_channels, skip_channels=skip_channels,
                aux_channels=aux_channels, dilation=dilation, bias=True,
                causal=False, dtype=dtype)
            self.conv_layers.append(conv_block)

        self.last_conv_relu1 = relu
        self.last_conv1 = Conv1d1x1_numpy(skip_channels, skip_channels, bias=True, dtype=dtype)
        self.last_conv_relu2 = relu
        self.last_conv2 = Conv1d1x1_numpy(skip_channels, out_channels, bias=True, dtype=dtype)
        self.final_activation = tanh
        self.dtype = dtype

    # @profile # Example for line_profiler
    def __call__(self, x, c=None):
        B, _, T_x = x.shape
        
        if c is not None:
            assert c.shape[0] == B, "Batch size mismatch between x and c"
            assert c.shape[2] == T_x, \
                f"Time dimension mismatch: c.shape[2]={c.shape[2]}, x.shape[2]={T_x}."

        x = self.first_conv(x)

        if not self.conv_layers:
            current_skip_channels = self.last_conv1.in_channels
        else:
            current_skip_channels = self.conv_layers[0].skip_channels
        skips_sum = np.zeros((B, current_skip_channels, T_x), dtype=self.dtype)

        for res_block in self.conv_layers:
            x, h_skip = res_block(x, c)
            skips_sum += h_skip

        if len(self.conv_layers) > 0:
            skips_sum *= math.sqrt(1.0 / len(self.conv_layers))
        
        x_out = skips_sum
        x_out = self.last_conv_relu1(x_out)
        x_out = self.last_conv1(x_out)
        x_out = self.last_conv_relu2(x_out)
        x_out = self.last_conv2(x_out)
        
        if self.final_activation:
            x_out = self.final_activation(x_out)
        return x_out.astype(self.dtype)

# --- Example Usage ---
if __name__ == '__main__':
    print("Setting up NumPy Parallel WaveGAN Generator...")

    BATCH_SIZE = 1; IN_CHANNELS = 1; OUT_CHANNELS = 1; KERNEL_SIZE = 3
    LAYERS = 30; STACKS = 3; RESIDUAL_CHANNELS = 64; GATE_CHANNELS = 128
    SKIP_CHANNELS = 64; AUX_CHANNELS = 80; TIME_STEPS = 256
    DTYPE = np.float32

    generator = ParallelWaveGANGenerator_numpy(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE,
        layers=LAYERS, stacks=STACKS, residual_channels=RESIDUAL_CHANNELS,
        gate_channels=GATE_CHANNELS, skip_channels=SKIP_CHANNELS,
        aux_channels=AUX_CHANNELS, dtype=DTYPE)
    print("Generator initialized.")

    dummy_x = np.random.randn(BATCH_SIZE, IN_CHANNELS, TIME_STEPS).astype(DTYPE)
    dummy_c = np.random.randn(BATCH_SIZE, AUX_CHANNELS, TIME_STEPS).astype(DTYPE)
    print(f"Input noise x shape: {dummy_x.shape}")
    print(f"Input conditioning c shape: {dummy_c.shape}")
    
    # --- cProfile Integration ---
    print("\nPerforming inference with cProfile...")
    profiler = cProfile.Profile()
    profiler.enable()

    with np.errstate(all='warn'):
        output_audio = generator(dummy_x, dummy_c)

    profiler.disable()
    print(f"Output audio shape: {output_audio.shape}")
    print("\nNumPy inference stage emulation complete.")

    s = io.StringIO()
    sortby = 'cumulative' # Can be 'cumtime', 'tottime', 'ncalls'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(20) # Print top 20 lines
    print("\n--- cProfile Results (Top 20) ---")
    print(s.getvalue())
    # To save full stats: ps.dump_stats('profiling_results.prof')
    # You can then visualize with snakeviz: snakeviz profiling_results.prof
    # --- End cProfile Integration ---

    # --- Plotting ---
    # (Plotting code remains the same)
    if BATCH_SIZE == 1 and OUT_CHANNELS == 1:
        plt.figure(figsize=(12, 4))
        time_axis = np.arange(TIME_STEPS)
        plt.plot(time_axis, output_audio[0, 0, :])
        plt.title("Generated Audio Waveform (NumPy Emulation)")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        try:
            plt.savefig("numpy_generated_audio.png")
            print("Plot saved to numpy_generated_audio.png")
        except Exception as e:
            print(f"Could not save plot: {e}")
    else:
        print("Plotting skipped for BATCH_SIZE or OUT_CHANNELS not equal to 1.")

    # --- Instructions for line_profiler ---
    # To use line_profiler for more detailed, line-by-line analysis:
    # 1. Install it: `pip install line_profiler`
    # 2. In your script, import it if you want to avoid NameError when not using kernprof:
    #    try:
    #        profile # Checks if 'profile' is defined (it will be by kernprof)
    #    except NameError:
    #        def profile(func): return func # Dummy decorator
    # 3. Add the `@profile` decorator above the functions/methods you want to inspect
    #    (e.g., above `Conv1d_numpy.__call__`, `ResidualBlock_numpy.__call__`,
    #     `ParallelWaveGANGenerator_numpy.__call__`, `sigmoid`, `tanh`, `relu`).
    # 4. Run from the command line: `kernprof -l your_script_name.py`
    # 5. View the results: `python -m line_profiler your_script_name.py.lprof`
