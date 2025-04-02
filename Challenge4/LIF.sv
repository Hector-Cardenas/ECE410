module LIF_Neuron #(
    parameter int WIDTH = 16,          // Bit-width of inputs and internal variables
    parameter int V_THRESH = 32768,    // Spiking threshold
    parameter int LEAK_SHIFT = 4,      // Leak shift (determines decay rate, should be within 1 to WIDTH-1)
    parameter int REFRACTORY = 10,     // Refractory period in clock cycles
    parameter int REFRACTORY_WIDTH = 8 // Bit-width for refractory counter
)(
    input  logic                clk,
    input  logic                rst,
    input  logic signed [WIDTH-1:0] I_in,  // Input current
    output logic                spike     // Output spike signal
);

    logic signed [WIDTH-1:0] V_m;          // Membrane potential
    logic [REFRACTORY_WIDTH-1:0] refrac_count; // Refractory period counter
    logic refractory_state;                // Refractory state flag

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            V_m <= 0;
            spike <= 0;
            refrac_count <= 0;
            refractory_state <= 0;
        end else begin
            spike <= 0; // Default no spike
            
            if (refractory_state) begin
                // In refractory period: Count down and reset after expiry
                if (refrac_count > 0)
                    refrac_count <= refrac_count - 1;
                else
                    refractory_state <= 0;  // Exit refractory state
            end else begin
                // Normal operation: update membrane potential with overflow protection
                logic signed [WIDTH:0] V_m_next;
                int safe_leak_shift = (LEAK_SHIFT < 1) ? 1 : (LEAK_SHIFT >= WIDTH ? WIDTH-1 : LEAK_SHIFT);
                V_m_next = V_m + I_in - (V_m >>> safe_leak_shift);
                
                // Overflow and underflow protection
                if (V_m > 0 && I_in > 0 && V_m_next < 0) begin
                    V_m <= {1'b0, {WIDTH-1{1'b1}}}; // Clamp to max positive value
                end else if (V_m < 0 && I_in < 0 && V_m_next > 0) begin
                    V_m <= {1'b1, {WIDTH-1{1'b0}}}; // Clamp to min negative value
                end else if (V_m_next >= V_THRESH) begin
                    spike <= 1;            // Fire a spike
                    V_m <= 0;              // Reset potential
                    refrac_count <= REFRACTORY; // Set refractory counter
                    refractory_state <= 1; // Enter refractory state
                end else begin
                    V_m <= V_m_next[WIDTH-1:0];
                end
            end
        end
    end
endmodule

