//-----------------------------------------------------------------------------
// Module: q_learning_tb
//-----------------------------------------------------------------------------
// Description:
// Testbench for the q_learning_update_unit module.
// - Instantiates the DUT.
// - Generates clock and reset.
// - Applies a basic stimulus sequence for one update.
// - Assumes Q-table starts at 0 (due to DUT reset init).
// - Calculates the expected Q-value update based on 0 init.
// - Verifies the Q-value written by the DUT by reading its internal memory.
//-----------------------------------------------------------------------------
`timescale 1ns/1ps

module q_learning_tb;

    //--- Parameters ---
    // Match DUT parameters
    parameter GRID_ROWS      = 5;
    parameter GRID_COLS      = 5;
    parameter NUM_ACTIONS    = 4;
    parameter Q_VALUE_WIDTH  = 16; // Total bits for Q-values and Reward (e.g., Q8.8)
    parameter PARAM_WIDTH    = 16; // Total bits for alpha and gamma (e.g., Q0.16 or Q1.15)
    parameter FRAC_BITS      = 8;  // Number of fractional bits

    // Testbench Parameters
    parameter CLK_PERIOD     = 10; // Clock period in ns
    parameter RESET_DURATION = CLK_PERIOD * 5;

    // Calculate address width locally (Fixes scope resolution issue)
    localparam Q_TABLE_DEPTH = GRID_ROWS * GRID_COLS * NUM_ACTIONS;
    localparam ADDR_WIDTH    = $clog2(Q_TABLE_DEPTH);

    //-------------------------------------------------------------------------
    // Signal Declarations
    //-------------------------------------------------------------------------
    logic clk;
    logic rst_n;
    logic start;

    logic [$clog2(GRID_ROWS)-1:0] current_state_row;
    logic [$clog2(GRID_COLS)-1:0] current_state_col;
    logic [$clog2(NUM_ACTIONS)-1:0] action;
    logic signed [Q_VALUE_WIDTH-1:0] reward;
    logic [$clog2(GRID_ROWS)-1:0] next_state_row;
    logic [$clog2(GRID_COLS)-1:0] next_state_col;

    logic signed [PARAM_WIDTH-1:0] alpha;
    logic signed [PARAM_WIDTH-1:0] gamma;

    logic busy;
    logic done;

    // Internal TB variables
    integer errors = 0;
    logic signed [Q_VALUE_WIDTH-1:0] expected_q_value;
    logic signed [Q_VALUE_WIDTH-1:0] actual_q_value;
    // Use locally calculated ADDR_WIDTH for target_addr declaration
    logic [ADDR_WIDTH-1:0] target_addr;

    //-------------------------------------------------------------------------
    // DUT Instantiation
    //-------------------------------------------------------------------------
    // Note: Ensure q_learning_update_unit module is compiled before this testbench
    q_learning_update_unit #(
        .GRID_ROWS      (GRID_ROWS),
        .GRID_COLS      (GRID_COLS),
        .NUM_ACTIONS    (NUM_ACTIONS),
        .Q_VALUE_WIDTH  (Q_VALUE_WIDTH),
        .PARAM_WIDTH    (PARAM_WIDTH),
        .FRAC_BITS      (FRAC_BITS)
    ) dut (
        .clk                (clk),
        .rst_n              (rst_n),
        .start              (start),
        .current_state_row  (current_state_row),
        .current_state_col  (current_state_col),
        .action             (action),
        .reward             (reward),
        .next_state_row     (next_state_row),
        .next_state_col     (next_state_col),
        .alpha              (alpha),
        .gamma              (gamma),
        .busy               (busy),
        .done               (done)
    );

    //-------------------------------------------------------------------------
    // Clock Generation
    //-------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    //-------------------------------------------------------------------------
    // Reset Generation
    //-------------------------------------------------------------------------
    initial begin
        rst_n = 1'b0; // Assert reset
        #RESET_DURATION;
        rst_n = 1'b1; // Deassert reset
    end

    //-------------------------------------------------------------------------
    // Stimulus and Checking
    //-------------------------------------------------------------------------
    initial begin
        $display("Starting Testbench...");

        // --- Test Case 1: Basic Update ---
        $display("Test Case 1: Basic Update");

        // 1. Assume Q-Table starts at 0 (due to DUT reset init)
        //    Removed direct assignments to dut.q_table to avoid multiple driver errors.
        $display("Assuming relevant Q-Table entries start at 0 (initialized by DUT reset)...");

        // 2. Set Parameters and Inputs (Fixed-point Q8.8 for Q/R, Q0.16 for params)
        // Example: alpha = 0.5, gamma = 0.9
        alpha = signed'(16'h8000); // 0.5 in Q0.16 (assuming PARAM_WIDTH=16)
        gamma = signed'(16'hE666); // ~0.9 in Q0.16 (0.9 * 2^16)

        current_state_row = 1;
        current_state_col = 1;
        action            = 2; // Left
        reward            = 16'hFF00; // R = -1.0 (Q8.8, two's complement of 1.0 = 0x0100)
        next_state_row    = 1;
        next_state_col    = 0;

        // 3. Wait for reset to finish
        wait (rst_n === 1'b1);
        @(posedge clk);

        // 4. Start the DUT
        $display("Applying start signal...");
        start = 1'b1;
        @(posedge clk);
        start = 1'b0; // Deassert start after one cycle

        // 5. Wait for DUT to finish (monitor done signal)
        $display("Waiting for DUT completion (done signal)...");
        wait (done === 1'b1);
        $display("DUT finished operation.");
        @(posedge clk); // Allow signals to settle after done

        // 6. Calculate Expected Result (Assuming relevant Q-values started at 0)
        // Q(s,a)_old = 0.0
        // R = -1.0
        // gamma = 0.9
        // Q(s',*) = 0.0 (due to assumption)
        // max_a' Q(s', a') = 0.0
        // Target = R + gamma * max_a' Q(s', a') = -1.0 + 0.9 * 0.0 = -1.0
        // Delta = Target - Q(s,a)_old = -1.0 - 0.0 = -1.0
        // New Q(s,a) = Q(s,a)_old + alpha * Delta = 0.0 + 0.5 * (-1.0) = -0.5
        // Expected Q8.8 value for -0.5: two's complement of 0.5 (0x0080) -> 0xFF80
        expected_q_value = 16'hFF80; // Calculated expected value for -0.5

        // 7. Verify the result by reading DUT internal memory
        // Use hierarchical reference to call DUT's function (ensure compiler supports this)
        target_addr = dut.calculate_address(current_state_row, current_state_col, action);
        actual_q_value = dut.q_table[target_addr]; // Read directly from DUT memory

        $display("Checking Q-value at address %h...", target_addr);
        $display("  Expected Q(s,a) = %h (%f)", expected_q_value, $itor(-128.0) / (2.0**FRAC_BITS)); // Approximate float display for -0.5
        $display("  Actual   Q(s,a) = %h (%f)", actual_q_value, $signed(actual_q_value) / (2.0**FRAC_BITS));

        if (actual_q_value !== expected_q_value) begin
            $error("Mismatch! Q-value verification failed.");
            errors = errors + 1;
        end else begin
            $display("Match! Q-value verification passed.");
        end

        // --- Add More Test Cases Here ---

        // 8. Final Report
        if (errors == 0) begin
            $display("All tests passed!");
        end else begin
            $display("%0d errors found.", errors);
        end

        $finish; // End simulation
    end

    //-------------------------------------------------------------------------
    // Monitoring (Optional - Uncomment to trace signals)
    //-------------------------------------------------------------------------
    initial begin
        // Monitor key DUT signals for debugging
        $monitor("Time=%0t State=%s busy=%b done=%b addr=%h wr_en=%b wr_data=%h rd_data=%h new_q=%h",
                 $time, dut.current_state.name(), busy, done, dut.mem_addr, dut.mem_write_en, dut.mem_write_data, dut.mem_read_data, dut.new_q_sa_reg);
    end

endmodule : q_learning_tb

