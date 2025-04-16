// ============================================================================
// Module: lif_neuron
// Description: Implements a digital Leaky Integrate-and-Fire (LIF) neuron
//              model with refractory period and overflow/underflow handling.
//              (Version 2.4: Added clamping for delta_v calculation)
// Author: AI Assistant (Gemini)
// Date: 2025-04-02
// ============================================================================
module lif_neuron #(
    // --- Parameters ---
    parameter int DATA_WIDTH = 16,
    parameter int LEAKAGE_SHIFT = 4,
    parameter logic signed [DATA_WIDTH-1:0] V_THRESHOLD = 'sd1000,
    parameter logic signed [DATA_WIDTH-1:0] V_RESET = 'sd0,
    parameter int REFRACTORY_PERIOD_CYCLES = 5
) (
    // --- Ports ---
    input  logic clk,
    input  logic rst_n,
    input  logic signed [DATA_WIDTH-1:0] i_inj,
    output logic o_spike
);

    // --- Constants for Saturation Arithmetic & Checks ---
    localparam logic signed [DATA_WIDTH-1:0] MAX_POTENTIAL = $signed({1'b0, {(DATA_WIDTH-1){1'b1}}});
    localparam logic signed [DATA_WIDTH-1:0] MIN_POTENTIAL = $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});
    localparam logic signed [DATA_WIDTH-1:0] MAX_POTENTIAL_CHECK = $signed({1'b0, {(DATA_WIDTH-1){1'b1}}});

    // --- Parameter Validation (Simulation Only) ---
    initial begin
        // (Parameter validation code remains the same as v2.1)
        if (DATA_WIDTH <= 0) begin $error("FATAL [lif_neuron %m]: DATA_WIDTH must be positive (value=%0d)", DATA_WIDTH); end
        if (LEAKAGE_SHIFT < 0) begin $warning("WARN [lif_neuron %m]: LEAKAGE_SHIFT is negative (value=%0d), behavior undefined.", LEAKAGE_SHIFT); end
        if (REFRACTORY_PERIOD_CYCLES < 0) begin $error("FATAL [lif_neuron %m]: REFRACTORY_PERIOD_CYCLES cannot be negative (value=%0d)", REFRACTORY_PERIOD_CYCLES); end
        if (V_THRESHOLD <= V_RESET) begin $warning("WARN [lif_neuron %m]: V_THRESHOLD (%0d) is not greater than V_RESET (%0d). Neuron might not behave as expected.", V_THRESHOLD, V_RESET); end
        if (V_THRESHOLD > MAX_POTENTIAL_CHECK) begin $error("FATAL [lif_neuron %m]: V_THRESHOLD (%0d) exceeds maximum possible potential (%0d) for DATA_WIDTH=%0d.", V_THRESHOLD, MAX_POTENTIAL_CHECK, DATA_WIDTH); end
        if (V_RESET < MIN_POTENTIAL) begin $warning("WARN [lif_neuron %m]: V_RESET (%0d) is below minimum possible potential (%0d) for DATA_WIDTH=%0d.", V_RESET, MIN_POTENTIAL, DATA_WIDTH); end
    end


    // --- Internal State Registers ---
    logic signed [DATA_WIDTH-1:0] v_mem;
    logic [$clog2(REFRACTORY_PERIOD_CYCLES+1)-1:0] refractory_counter;

    // --- Internal Combinational Signals (Next State Logic) ---
    logic signed [DATA_WIDTH-1:0] v_mem_next;
    logic [$clog2(REFRACTORY_PERIOD_CYCLES+1)-1:0] refractory_counter_next;
    logic                         spike_next;

    // ============================================================================
    //  Sequential Logic: Register state updates on clock edge or reset
    // ============================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem              <= V_RESET;
            refractory_counter <= '0;
            o_spike            <= 1'b0;
        end else begin
            v_mem              <= v_mem_next;
            refractory_counter <= refractory_counter_next;
            o_spike            <= spike_next;
        end
    end

    // ============================================================================
    //  Combinational Logic: Calculate the next state based on current state and inputs
    // ============================================================================
    always_comb begin
        // --- Intermediate variables for clarity ---
        logic signed [DATA_WIDTH-1:0] temp_v_mem_next;
        logic                         temp_spike_next;
        logic [$clog2(REFRACTORY_PERIOD_CYCLES+1)-1:0] temp_ref_ctr_next;

        // --- Default assignments for temporary variables ---
        temp_v_mem_next = v_mem;
        temp_ref_ctr_next = refractory_counter;
        temp_spike_next = 1'b0;

        // --- Refractory Period Handling ---
        if (refractory_counter > 0) begin
            temp_v_mem_next = V_RESET;
            temp_ref_ctr_next = refractory_counter - 1;
            temp_spike_next = 1'b0;
        end else begin
            // --- Normal Operation (Not Refractory) ---
            logic signed [DATA_WIDTH-1:0] leak_value;
            // ** Intermediate delta_v calculation with clamping **
            logic signed [DATA_WIDTH:0]   delta_v_intermediate; // Wider intermediate result
            logic signed [DATA_WIDTH-1:0] delta_v_clamped;      // Clamped delta_v
            // ** --------------------------------------------- **
            logic signed [DATA_WIDTH-1:0] v_mem_sum_unclamped;
            logic                         positive_overflow_detected;
            logic                         negative_overflow_detected;
            logic signed [DATA_WIDTH-1:0] v_mem_after_saturation;

            // 1. Calculate Leakage
            leak_value = v_mem >>> LEAKAGE_SHIFT;

            // 2. Calculate Net Change (delta_v) using wider intermediate result
            delta_v_intermediate = {i_inj[DATA_WIDTH-1], i_inj} - {leak_value[DATA_WIDTH-1], leak_value};

            // 2a. Clamp delta_v_intermediate to DATA_WIDTH range to prevent overflow in delta_v itself
            if (delta_v_intermediate > {MAX_POTENTIAL[DATA_WIDTH-1], MAX_POTENTIAL}) begin
                delta_v_clamped = MAX_POTENTIAL;
            // Check against MIN_POTENTIAL for underflow clamping
            end else if (delta_v_intermediate < {MIN_POTENTIAL[DATA_WIDTH-1], MIN_POTENTIAL}) begin
                delta_v_clamped = MIN_POTENTIAL;
            end else begin
                // Value is within representable range for DATA_WIDTH
                delta_v_clamped = delta_v_intermediate[DATA_WIDTH-1:0];
            end

            // 3. Integrate Potential & Check for Overflow (using clamped delta_v)
            v_mem_sum_unclamped = v_mem + delta_v_clamped; // Use the clamped delta_v

            // Detect positive overflow for the final sum: (+A) + (+B) = (-Result)
            positive_overflow_detected = (v_mem[DATA_WIDTH-1] == 1'b0) &&
                                         (delta_v_clamped[DATA_WIDTH-1] == 1'b0) && // Check clamped delta_v sign
                                         (v_mem_sum_unclamped[DATA_WIDTH-1] == 1'b1);

            // Detect negative overflow for the final sum: (-A) + (-B) = (+Result)
            negative_overflow_detected = (v_mem[DATA_WIDTH-1] == 1'b1) &&
                                         (delta_v_clamped[DATA_WIDTH-1] == 1'b1) && // Check clamped delta_v sign
                                         (v_mem_sum_unclamped[DATA_WIDTH-1] == 1'b0);

            // 4. Apply Saturation for the final sum (result stored in v_mem_after_saturation)
            if (positive_overflow_detected) begin
                v_mem_after_saturation = MAX_POTENTIAL;
            end else if (negative_overflow_detected) begin
                 // Negative overflow occurred (wrapped past MIN_POTENTIAL)
                 // Clamp to the defined minimum allowed value, V_RESET
                v_mem_after_saturation = V_RESET;
            end else begin
                // No overflow during addition. Check if result is below V_RESET.
                if (v_mem_sum_unclamped < V_RESET) begin
                    // Clamp to V_RESET if below it, even if no overflow occurred
                    v_mem_after_saturation = V_RESET;
                end else begin
                    // Result is within valid range [V_RESET, MAX_POTENTIAL]
                    v_mem_after_saturation = v_mem_sum_unclamped;
                end
            end

            // 5. Firing Threshold Check (uses the saturated value)
            if (v_mem_after_saturation >= V_THRESHOLD) begin
                // Spike occurs
                temp_spike_next = 1'b1;
                temp_v_mem_next = V_RESET; // Final potential for next cycle is V_RESET
                if (REFRACTORY_PERIOD_CYCLES > 0) begin
                   temp_ref_ctr_next = REFRACTORY_PERIOD_CYCLES; // Start refractory period
                end else begin
                   temp_ref_ctr_next = 0;
                end
            end else begin
                // No spike occurs
                temp_spike_next = 1'b0;
                temp_v_mem_next = v_mem_after_saturation; // Final potential is the saturated value
                temp_ref_ctr_next = 0; // Stay out of refractory
            end
        end

        // --- Assign final calculated values to the outputs of the combinational block ---
        v_mem_next = temp_v_mem_next;
        spike_next = temp_spike_next;
        refractory_counter_next = temp_ref_ctr_next;
    end

endmodule : lif_neuron

