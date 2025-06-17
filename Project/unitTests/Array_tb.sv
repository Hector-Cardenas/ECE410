`timescale 1ns / 1ps

module SystolicArray_tb;

    // Parameters for the Systolic Array and Matrices
    // M: Rows in A & C (maps to ARRAY_ROWS)
    // N: Cols in B & C (maps to ARRAY_COLUMNS)
    // K: Common Dim (Cols A, Rows B) - This is a testbench concept
    localparam TB_M_ROWS     = 2;
    localparam TB_N_COLS     = 2;
    localparam TB_K_COMMON   = 2;
    localparam TB_INPUTS_N   = 8;
    localparam TB_ACCUM_OUT_N = 32;

    // Clock and Control Signals
    reg Clock;
    reg Reset;
    // Changed to unpacked arrays to match DUT port style more closely
    reg Clear_Row_tb [TB_M_ROWS-1:0];
    reg Clear_Column_tb [TB_N_COLS-1:0];

    // DUT Inputs (using 'reg' for testbench drivers)
    reg signed [TB_INPUTS_N-1:0] Acts_In_Top_tb             [TB_N_COLS-1:0];
    reg                             Act_Valids_In_Top_tb       [TB_N_COLS-1:0];
    reg signed [TB_INPUTS_N-1:0] Weights_In_Left_tb         [TB_M_ROWS-1:0];
    reg                             Weight_Valids_In_Left_tb   [TB_M_ROWS-1:0];

    // DUT Outputs (using 'wire' to connect to DUT outputs)
    wire signed [TB_INPUTS_N-1:0] Acts_Out_Bottom_dut         [TB_N_COLS-1:0];
    wire                             Act_Valids_Out_Bottom_dut   [TB_N_COLS-1:0];
    wire signed [TB_INPUTS_N-1:0] Weights_Out_Right_dut       [TB_M_ROWS-1:0];
    wire                             Weight_Valids_Out_Right_dut [TB_M_ROWS-1:0];
    wire signed [TB_ACCUM_OUT_N-1:0] Accs_Out_dut               [TB_M_ROWS-1:0][TB_N_COLS-1:0];

    // Instantiate the DUT
    SystolicArray #(
        .ARRAY_INPUTS_N(TB_INPUTS_N),
        .ARRAY_OUTPUTS_N(TB_ACCUM_OUT_N),
        .ARRAY_ROWS(TB_M_ROWS),
        .ARRAY_COLUMNS(TB_N_COLS)
    ) dut (
        .Clock(Clock),
        .Reset(Reset),
        .Clear_Row(Clear_Row_tb),
        .Clear_Column(Clear_Column_tb),
        .Acts_In_Top(Acts_In_Top_tb),
        .Act_Valids_In_Top(Act_Valids_In_Top_tb),
        .Weights_In_Left(Weights_In_Left_tb),
        .Weight_Valids_In_Left(Weight_Valids_In_Left_tb),
        .Acts_Out_Bottom(Acts_Out_Bottom_dut),
        .Act_Valids_Out_Bottom(Act_Valids_Out_Bottom_dut),
        .Weights_Out_Right(Weights_Out_Right_dut),
        .Weight_Valids_Out_Right(Weight_Valids_Out_Right_dut),
        .Accs_Out(Accs_Out_dut)
    );

    // Clock Generation
    localparam CLOCK_PERIOD = 10;
    always # (CLOCK_PERIOD / 2) Clock = ~Clock;

    // Testbench Data Storage
    reg signed [TB_INPUTS_N-1:0]   matrix_A_weights [TB_M_ROWS-1:0][TB_K_COMMON-1:0];
    reg signed [TB_INPUTS_N-1:0]   matrix_B_acts    [TB_K_COMMON-1:0][TB_N_COLS-1:0];
    reg signed [TB_ACCUM_OUT_N-1:0] expected_C_accs  [TB_M_ROWS-1:0][TB_N_COLS-1:0];

    // Initialization and Test Sequence
    initial begin
        // Declare all loop variables and temp variables at the start of the block
        integer t, r, c, k;
        integer k_val_for_A, k_val_for_B;
        integer max_feed_timestep, cycles_for_last_result;
        integer cycles_already_passed, cycles_to_wait;
        reg error_found;

        $display("[%0t] Starting SystolicArray Testbench (Compatibility Mode)...", $time);
        Clock = 1'b0;
        Reset = 1'b1;

        // Initialize Clear signals using loops for unpacked arrays
        for (r = 0; r < TB_M_ROWS; r = r + 1) begin
            Clear_Row_tb[r] = 1'b0;
        end
        for (c = 0; c < TB_N_COLS; c = c + 1) begin
            Clear_Column_tb[c] = 1'b0;
        end

        // Removed problematic aggregate assignments to unpacked arrays:
        // Act_Valids_In_Top_tb   = '0;
        // Acts_In_Top_tb         = '0;
        // Weight_Valids_In_Left_tb = '0;
        // Weights_In_Left_tb     = '0;

        // Initialize unpacked arrays of valid/data signals element-wise
        for (c = 0; c < TB_N_COLS; c = c + 1) begin
            Act_Valids_In_Top_tb[c] = 1'b0;
            Acts_In_Top_tb[c] = '0; // Assign '0' to each packed vector element
        end
        for (r = 0; r < TB_M_ROWS; r = r + 1) begin
            Weight_Valids_In_Left_tb[r] = 1'b0;
            Weights_In_Left_tb[r] = '0; // Assign '0' to each packed vector element
        end


        // --- 1. Initialize Matrices ---
        matrix_A_weights[0][0] = 1; matrix_A_weights[0][1] = 2;
        matrix_A_weights[1][0] = 3; matrix_A_weights[1][1] = 4;
        matrix_B_acts[0][0] = 5; matrix_B_acts[0][1] = 6;
        matrix_B_acts[1][0] = 7; matrix_B_acts[1][1] = 8;

        // --- 2. Calculate Expected Result ---
        for (r = 0; r < TB_M_ROWS; r = r + 1) begin
            for (c = 0; c < TB_N_COLS; c = c + 1) begin
                expected_C_accs[r][c] = 0;
                for (k = 0; k < TB_K_COMMON; k = k + 1) begin
                    expected_C_accs[r][c] = expected_C_accs[r][c] + (matrix_A_weights[r][k] * matrix_B_acts[k][c]);
                end
            end
        end
        $display("Expected Result Matrix calculated.");

        // --- 3. Apply Reset ---
        repeat (2) @(posedge Clock);
        Reset = 1'b0;
        @(posedge Clock);
        @(negedge Clock);
        $display("[%0t] Reset de-asserted.", $time);

        // --- 4. Apply Clear ---
        // Assign to unpacked arrays using loops
        for (r = 0; r < TB_M_ROWS; r = r + 1) begin
            Clear_Row_tb[r] = 1'b1;
        end
        for (c = 0; c < TB_N_COLS; c = c + 1) begin
            Clear_Column_tb[c] = 1'b1;
        end
        @(posedge Clock);
        for (r = 0; r < TB_M_ROWS; r = r + 1) begin
            Clear_Row_tb[r] = 1'b0;
        end
        for (c = 0; c < TB_N_COLS; c = c + 1) begin
            Clear_Column_tb[c] = 1'b0;
        end
        @(negedge Clock);
        $display("[%0t] Global Clear applied.", $time);
        for (r = 0; r < TB_M_ROWS; r = r + 1) begin
            for (c = 0; c < TB_N_COLS; c = c + 1) begin
                if (Accs_Out_dut[r][c] !== 0) begin
                    $error("[%0t] ERROR: Accs_Out_dut[%0d][%0d] is %d after Clear, expected 0.", $time, r, c, Accs_Out_dut[r][c]);
                end
            end
        end

        // --- 5. Feed Data with Skewing ---
        max_feed_timestep = (TB_K_COMMON - 1) + ((TB_M_ROWS > TB_N_COLS) ? (TB_M_ROWS - 1) : (TB_N_COLS - 1));
        $display("[%0t] Starting data feed for %0d timesteps...", $time, max_feed_timestep + 1);
        for (t = 0; t <= max_feed_timestep; t = t + 1) begin
            for (r = 0; r < TB_M_ROWS; r = r + 1) begin
                k_val_for_A = t - r;
                if (k_val_for_A >= 0 && k_val_for_A < TB_K_COMMON) begin
                    Weights_In_Left_tb[r] = matrix_A_weights[r][k_val_for_A];
                    Weight_Valids_In_Left_tb[r] = 1'b1;
                end else begin
                    // Weights_In_Left_tb[r] = '0; // Default value if needed, already initialized
                    Weight_Valids_In_Left_tb[r] = 1'b0;
                end
            end
            for (c = 0; c < TB_N_COLS; c = c + 1) begin
                k_val_for_B = t - c;
                if (k_val_for_B >= 0 && k_val_for_B < TB_K_COMMON) begin
                    Acts_In_Top_tb[c] = matrix_B_acts[k_val_for_B][c];
                    Act_Valids_In_Top_tb[c] = 1'b1;
                end else begin
                    // Acts_In_Top_tb[c] = '0; // Default value if needed, already initialized
                    Act_Valids_In_Top_tb[c] = 1'b0;
                end
            end
            @(posedge Clock);
        end
        // De-assert valid signals after feeding
        for (r = 0; r < TB_M_ROWS; r = r + 1) Weight_Valids_In_Left_tb[r] = 1'b0;
        for (c = 0; c < TB_N_COLS; c = c + 1) Act_Valids_In_Top_tb[c] = 1'b0;
        $display("[%0t] Data feed completed.", $time);

        // --- 6. Wait for Computation to Complete ---
        cycles_for_last_result = (TB_K_COMMON - 1) + (TB_M_ROWS - 1) + (TB_N_COLS - 1) + 2;
        cycles_already_passed = max_feed_timestep + 1;
        cycles_to_wait = cycles_for_last_result - cycles_already_passed;
        if (cycles_to_wait < 0) cycles_to_wait = 0;
        cycles_to_wait = cycles_to_wait + 2; // Safety margin
        $display("[%0t] Waiting %0d additional cycles for results.", $time, cycles_to_wait);
        repeat (cycles_to_wait) @(posedge Clock);
        @(negedge Clock);

        // --- 7. Verify Output Matrix ---
        $display("[%0t] Verifying output Accs_Out_dut...", $time);
        error_found = 1'b0;
        for (r = 0; r < TB_M_ROWS; r = r + 1) begin
            for (c = 0; c < TB_N_COLS; c = c + 1) begin
                if (Accs_Out_dut[r][c] !== expected_C_accs[r][c]) begin
                    $error("[%0t] ERROR: Accs_Out[%0d][%0d] = %d, Expected = %d", $time, r, c, Accs_Out_dut[r][c], expected_C_accs[r][c]);
                    error_found = 1'b1;
                end else begin
                     $display("[%0t] INFO: Accs_Out[%0d][%0d] = %d -- MATCH", $time, r, c, Accs_Out_dut[r][c]);
                end
            end
        end

        if (!error_found) $display("[%0t] SUCCESS: Output Matrix matches expected values!", $time);
        else $display("[%0t] FAILURE: Output Matrix has mismatches.", $time);

        $display("[%0t] Testbench finished.", $time);
        $finish;
    end

endmodule

