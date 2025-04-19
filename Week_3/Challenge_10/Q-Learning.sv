//-----------------------------------------------------------------------------
// Module: q_learning_update_unit
//-----------------------------------------------------------------------------
// Description:
// Performs a single Q-learning update step in hardware.
// Assumes fixed-point representation for Q-values and parameters.
// Addresses the bottleneck of software dictionary copying by using
// dedicated on-chip memory (SRAM) and in-place updates.
//
// Q-Update Rule:
// Q(s, a) <= Q(s, a) + alpha * (R + gamma * max_a'(Q(s', a')) - Q(s, a))
//-----------------------------------------------------------------------------

module q_learning_update_unit #(
    //--- Parameters ---
    parameter GRID_ROWS      = 5,
    parameter GRID_COLS      = 5,
    parameter NUM_ACTIONS    = 4,
    // Fixed-point representation: Q<INT_BITS>.<FRAC_BITS>
    parameter Q_VALUE_WIDTH  = 16, // Total bits for Q-values and Reward (e.g., Q8.8)
    parameter PARAM_WIDTH    = 16, // Total bits for alpha and gamma (e.g., Q0.16)
    parameter FRAC_BITS      = 8   // Number of fractional bits
) (
    //--- Ports ---
    // Control Signals
    input logic clk,          // Clock
    input logic rst_n,        // Asynchronous Reset (active low)
    input logic start,        // Start signal for one update cycle

    // Input Data (Registered externally or valid with 'start')
    input logic [$clog2(GRID_ROWS)-1:0] current_state_row, // s_row
    input logic [$clog2(GRID_COLS)-1:0] current_state_col, // s_col
    input logic [$clog2(NUM_ACTIONS)-1:0] action,          // a
    input logic signed [Q_VALUE_WIDTH-1:0] reward,        // R (fixed-point Q8.8)
    input logic [$clog2(GRID_ROWS)-1:0] next_state_row,    // s'_row
    input logic [$clog2(GRID_COLS)-1:0] next_state_col,    // s'_col

    // Learning Parameters (Assumed constant during operation or loaded separately)
    input logic signed [PARAM_WIDTH-1:0] alpha,           // Learning rate (fixed-point Q0.16)
    input logic signed [PARAM_WIDTH-1:0] gamma,           // Discount factor (fixed-point Q0.16)

    // Status Signals
    output logic busy,         // Module is processing an update
    output logic done          // Update cycle completed
);

    //-------------------------------------------------------------------------
    // Internal Parameters and Types
    //-------------------------------------------------------------------------
    localparam Q_TABLE_DEPTH = GRID_ROWS * GRID_COLS * NUM_ACTIONS;
    localparam ADDR_WIDTH    = $clog2(Q_TABLE_DEPTH);
    // Define parameter fractional bits based on assumption (e.g., Q0.PARAM_WIDTH)
    localparam PARAM_FRAC_BITS = PARAM_WIDTH;

    // State machine states
    typedef enum logic [3:0] {
        IDLE,
        CALC_ADDR_SA,      // Calculate address for Q(s, a)
        READ_Q_SA,         // Read Q(s, a) from memory
        CALC_ADDR_SP_A0,   // Calculate address for Q(s', 0)
        READ_Q_SP_A0,      // Read Q(s', 0)
        CALC_ADDR_SP_A1,   // Calculate address for Q(s', 1)
        READ_Q_SP_A1,      // Read Q(s', 1)
        CALC_ADDR_SP_A2,   // Calculate address for Q(s', 2)
        READ_Q_SP_A2,      // Read Q(s', 2)
        CALC_ADDR_SP_A3,   // Calculate address for Q(s', 3)
        READ_Q_SP_A3,      // Read Q(s', 3)
        FIND_MAX_Q_SP,     // Find max(Q(s', a'))
        CALCULATE_UPDATE,  // Perform the Q-update arithmetic
        WRITE_Q_SA,        // Write the new Q(s, a) back to memory
        FINISH             // Signal completion
    } state_t;

    //-------------------------------------------------------------------------
    // Q-Table Memory (Simulated as behavioral memory)
    //-------------------------------------------------------------------------
    logic signed [Q_VALUE_WIDTH-1:0] q_table [Q_TABLE_DEPTH];
    logic [ADDR_WIDTH-1:0]           mem_addr;
    logic                            mem_write_en;
    logic signed [Q_VALUE_WIDTH-1:0] mem_write_data;
    logic signed [Q_VALUE_WIDTH-1:0] mem_read_data;

    // Behavioral Memory Model - UPDATED WITH RESET
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < Q_TABLE_DEPTH; i++) begin
                q_table[i] <= '0;
            end
        end else begin
            if (mem_write_en) begin
                q_table[mem_addr] <= mem_write_data;
            end
        end
    end
    assign mem_read_data = q_table[mem_addr];

    //-------------------------------------------------------------------------
    // Internal Registers and Wires
    //-------------------------------------------------------------------------
    state_t current_state, next_state;

    // Registers
    logic [ADDR_WIDTH-1:0]           addr_sa_reg;
    logic [ADDR_WIDTH-1:0]           addr_sp_base_reg;
    logic signed [Q_VALUE_WIDTH-1:0] q_sa_reg;
    logic signed [Q_VALUE_WIDTH-1:0] q_sp_a0_reg, q_sp_a1_reg, q_sp_a2_reg, q_sp_a3_reg;
    logic signed [Q_VALUE_WIDTH-1:0] max_q_sp_reg;
    logic signed [Q_VALUE_WIDTH-1:0] q_update_term_reg;
    logic signed [Q_VALUE_WIDTH-1:0] q_delta_reg;
    logic signed [Q_VALUE_WIDTH-1:0] q_scaled_delta_reg;
    logic signed [Q_VALUE_WIDTH-1:0] new_q_sa_reg;

    // Wires
    logic [ADDR_WIDTH-1:0]           addr_sa_comb;
    logic [ADDR_WIDTH-1:0]           addr_sp_base_comb;
    logic signed [Q_VALUE_WIDTH-1:0] max_q_sp_comb;
    logic signed [Q_VALUE_WIDTH-1:0] q_update_term_comb;
    logic signed [Q_VALUE_WIDTH-1:0] q_delta_comb;
    logic signed [Q_VALUE_WIDTH-1:0] q_scaled_delta_comb;
    logic signed [Q_VALUE_WIDTH-1:0] new_q_sa_comb;

    // Intermediate wider values for multiplication results
    localparam INTERMEDIATE_FRAC_BITS = PARAM_FRAC_BITS + FRAC_BITS; // 16 + 8 = 24
    localparam INTERMEDIATE_WIDTH   = Q_VALUE_WIDTH + PARAM_FRAC_BITS; // 16 + 16 = 32 bits is sufficient
    logic signed [INTERMEDIATE_WIDTH-1:0] gamma_x_max_q_sp;
    logic signed [INTERMEDIATE_WIDTH-1:0] alpha_x_q_delta;
    // Wires to hold shifted intermediate results before slicing (Reintroduced)
    logic signed [INTERMEDIATE_WIDTH-1:0] shifted_gamma_x_max_q_sp;
    logic signed [INTERMEDIATE_WIDTH-1:0] shifted_alpha_x_q_delta;

    // Correct scaling shift amount
    localparam SCALE_SHIFT_AMOUNT = PARAM_FRAC_BITS; // = 16

    //-------------------------------------------------------------------------
    // Address Calculation Logic
    //-------------------------------------------------------------------------
    function automatic [ADDR_WIDTH-1:0] calculate_address (
        input logic [$clog2(GRID_ROWS)-1:0] row,
        input logic [$clog2(GRID_COLS)-1:0] col,
        input logic [$clog2(NUM_ACTIONS)-1:0] act
    );
        return row * (GRID_COLS * NUM_ACTIONS) + col * NUM_ACTIONS + act;
    endfunction

    assign addr_sa_comb = calculate_address(current_state_row, current_state_col, action);
    assign addr_sp_base_comb = calculate_address(next_state_row, next_state_col, 0);

    //-------------------------------------------------------------------------
    // State Machine Logic
    //-------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            // Reset registers
            q_sa_reg <= '0; q_sp_a0_reg <= '0; q_sp_a1_reg <= '0; q_sp_a2_reg <= '0; q_sp_a3_reg <= '0;
            max_q_sp_reg <= '0; q_update_term_reg <= '0; q_delta_reg <= '0;
            q_scaled_delta_reg <= '0; new_q_sa_reg <= '0; addr_sa_reg <= '0; addr_sp_base_reg <= '0;
        end else begin
            current_state <= next_state;
            // Latch intermediate values
             case (current_state)
                CALC_ADDR_SA:    addr_sa_reg <= addr_sa_comb;
                READ_Q_SA:       q_sa_reg <= mem_read_data;
                CALC_ADDR_SP_A0: addr_sp_base_reg <= addr_sp_base_comb;
                READ_Q_SP_A0:    q_sp_a0_reg <= mem_read_data;
                READ_Q_SP_A1:    q_sp_a1_reg <= mem_read_data;
                READ_Q_SP_A2:    q_sp_a2_reg <= mem_read_data;
                READ_Q_SP_A3:    q_sp_a3_reg <= mem_read_data;
                FIND_MAX_Q_SP:   max_q_sp_reg <= max_q_sp_comb;
                CALCULATE_UPDATE: begin
                    q_update_term_reg <= q_update_term_comb;
                    q_delta_reg       <= q_delta_comb;
                    q_scaled_delta_reg<= q_scaled_delta_comb;
                    new_q_sa_reg      <= new_q_sa_comb;
                end
                default: ;
            endcase
        end
    end

    // Combinational logic
    always_comb begin
        // Defaults
        next_state     = current_state;
        busy           = (current_state != IDLE);
        done           = 1'b0;
        mem_write_en   = 1'b0;
        mem_addr       = '0;
        mem_write_data = '0;
        max_q_sp_comb      = '0; q_update_term_comb = '0; q_delta_comb       = '0;
        q_scaled_delta_comb= '0; new_q_sa_comb      = '0; gamma_x_max_q_sp   = '0;
        alpha_x_q_delta    = '0;
        shifted_gamma_x_max_q_sp = '0; // Default intermediate signals
        shifted_alpha_x_q_delta  = '0;

        case (current_state)
            IDLE: begin
                busy = 1'b0;
                if (start) next_state = CALC_ADDR_SA;
            end
            CALC_ADDR_SA:    next_state = READ_Q_SA;
            READ_Q_SA:       next_state = CALC_ADDR_SP_A0;
            CALC_ADDR_SP_A0: next_state = READ_Q_SP_A0;
            READ_Q_SP_A0:    next_state = READ_Q_SP_A1;
            READ_Q_SP_A1:    next_state = READ_Q_SP_A2;
            READ_Q_SP_A2:    next_state = READ_Q_SP_A3;
            READ_Q_SP_A3:    next_state = FIND_MAX_Q_SP;
            FIND_MAX_Q_SP: begin
                // Find max Q(s', a')
                max_q_sp_comb = q_sp_a0_reg;
                if (q_sp_a1_reg > max_q_sp_comb) max_q_sp_comb = q_sp_a1_reg;
                if (q_sp_a2_reg > max_q_sp_comb) max_q_sp_comb = q_sp_a2_reg;
                if (q_sp_a3_reg > max_q_sp_comb) max_q_sp_comb = q_sp_a3_reg;
                next_state = CALCULATE_UPDATE;
            end
            CALCULATE_UPDATE: begin
                // Perform fixed-point arithmetic combinatorially

                // gamma * max_q_sp -> Intermediate result Q8.24
                gamma_x_max_q_sp = gamma * max_q_sp_reg;
                // Arithmetically shift right to get intermediate Q8.16 (still 32 bits wide)
                shifted_gamma_x_max_q_sp = $signed(gamma_x_max_q_sp) >>> SCALE_SHIFT_AMOUNT;
                // Add reward, using explicitly sliced bits from shifted result
                q_update_term_comb = reward + shifted_gamma_x_max_q_sp[Q_VALUE_WIDTH-1:0]; // Slice intermediate wire

                // q_update_term - q_sa
                q_delta_comb = q_update_term_comb - q_sa_reg;

                // alpha * q_delta -> Intermediate result Q8.24
                alpha_x_q_delta = alpha * q_delta_comb;
                // Arithmetically shift right to get intermediate Q8.16 (still 32 bits wide)
                shifted_alpha_x_q_delta = $signed(alpha_x_q_delta) >>> SCALE_SHIFT_AMOUNT;
                // Explicitly take lower bits of shifted result for Q8.8 format
                q_scaled_delta_comb = shifted_alpha_x_q_delta[Q_VALUE_WIDTH-1:0]; // Slice intermediate wire

                // new_q = q_sa + scaled_delta
                // Reverted to simple addition - explicit cast failed compilation and $signed() is redundant
                new_q_sa_comb = q_sa_reg + q_scaled_delta_comb;

                next_state = WRITE_Q_SA;
            end
            WRITE_Q_SA: begin
                // Prepare write for next clock edge
                mem_addr       = addr_sa_reg;
                mem_write_data = new_q_sa_reg; // Use latched result
                mem_write_en   = 1'b1;
                next_state     = FINISH;
            end
            FINISH: begin
                done = 1'b1;
                next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

endmodule : q_learning_update_unit

