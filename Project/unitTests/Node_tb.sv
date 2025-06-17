// Testbench for SystolicNode module
`timescale 1ns / 1ps

module SystolicNode_tb;

    // Parameters for the DUT (Device Under Test)
    localparam INPUTS_N    = 8;
    localparam ACCUM_OUT_N = 32;
    localparam PRODUCT_N   = INPUTS_N * 2;

    // Testbench signals to connect to the DUT
    logic                           Clock;
    logic                           Reset;
    logic                           Clear;
    logic                           Act_Valid_In;
    logic signed [INPUTS_N-1:0]     Act_In;
    logic                           Weight_Valid_In;
    logic signed [INPUTS_N-1:0]     Weight_In;

    logic                           Act_Valid_Out;
    logic signed [INPUTS_N-1:0]     Act_Out;
    logic                           Weight_Valid_Out;
    logic signed [INPUTS_N-1:0]     Weight_Out;
    logic signed [ACCUM_OUT_N-1:0]  Accum_Out;

    // Instantiate the DUT
    SystolicNode #(
        .INPUTS_N(INPUTS_N),
        .ACCUM_OUT_N(ACCUM_OUT_N)
    ) dut (
        .Clock(Clock),
        .Reset(Reset),
        .Clear(Clear),
        .Act_Valid_In(Act_Valid_In),
        .Act_In(Act_In),
        .Weight_Valid_In(Weight_Valid_In),
        .Weight_In(Weight_In),
        .Act_Valid_Out(Act_Valid_Out),
        .Act_Out(Act_Out),
        .Weight_Valid_Out(Weight_Valid_Out),
        .Weight_Out(Weight_Out),
        .Accum_Out(Accum_Out)
    );

    // Clock generation
    localparam CLOCK_PERIOD = 10; // 10 ns period -> 100 MHz
    always begin
        Clock = 1'b0;
        #(CLOCK_PERIOD / 2);
        Clock = 1'b1;
        #(CLOCK_PERIOD / 2);
    end

    // Test sequence
    initial begin
        $display("Starting Testbench for SystolicNode (Negedge Sampling - Corrected Timing)");

        // Initialize inputs
        Reset           = 1'b1; // Assert Reset initially
        Clear           = 1'b0;
        Act_Valid_In    = 1'b0;
        Act_In          = '0;
        Weight_Valid_In = 1'b0;
        Weight_In       = '0;

        // Apply Reset for a few cycles
        $display("[%0t] Asserting Reset", $time);
        repeat (2) @(posedge Clock);
        Reset = 1'b0; // De-assert Reset
        @(posedge Clock); // Posedge where Reset is low for the first time
        @(negedge Clock); // Sample after Reset takes effect
        $display("[%0t] Reset De-asserted. Sampled at negedge. Accum_Out = %d", $time, Accum_Out);
        if (Accum_Out !== 0) begin
            $error("[%0t] ERROR: Accum_Out is not 0 after reset. Value: %d", $time, Accum_Out);
        end

        // Test Case 1: Single valid input (5x3 = 15)
        $display("[%0t] Test Case 1: Single valid input (5x3=15)", $time);
        Act_In          = 8'd5;
        Act_Valid_In    = 1'b1;
        Weight_In       = 8'd3;
        Weight_Valid_In = 1'b1;
        // Initial Accum_Out is 0.

        // Cycle 1 (P1/N1): Apply inputs. DUT registers inputs, calculates Product_reg=15, mac_enable=1 (for P2's acc).
        // Accumulator itself uses previous (e.g., reset) mac_enable, so Accum_Out doesn't change yet.
        @(posedge Clock); // P1
        @(negedge Clock); // N1: Sample outputs
        $display("[%0t] TC1 Post-P1 (Sampled Negedge): Act_Out=%d, W_Out=%d, AV_Out=%b, WV_Out=%b, Accum_Out=%d (Expected Accum: 0)",
                 $time, Act_Out, Weight_Out, Act_Valid_Out, Weight_Valid_Out, Accum_Out);
        if (Act_Out !== 8'd5 || Weight_Out !== 8'd3 || Act_Valid_Out !== 1'b1 || Weight_Valid_Out !== 1'b1) begin
             $error("[%0t] ERROR TC1.1: Input pass-through incorrect. Act_Out=%d, W_Out=%d, AV_Out=%b, WV_Out=%b",
                    $time, Act_Out, Weight_Out, Act_Valid_Out, Weight_Valid_Out);
        end
        if (Accum_Out !== 0) begin
            $error("[%0t] ERROR TC1.2: Accum_Out expected 0 (pre-accumulation of 5*3), got %d", $time, Accum_Out);
        end

        // De-assert valid inputs NOW so that on P2, the next Product_reg/mac_enable are calculated based on invalid inputs.
        Act_Valid_In    = 1'b0;
        Weight_Valid_In = 1'b0;

        // Cycle 2 (P2/N2): Accumulator uses Product_reg=15 and mac_enable=1 (from P1). Accum_Out becomes 15.
        // SystolicNode calculates new Product_reg=0, mac_enable=0 (because inputs are now invalid for P2's Product_reg/mac_enable calc).
        @(posedge Clock); // P2
        @(negedge Clock); // N2: Sample outputs
        $display("[%0t] TC1 Post-P2 (Sampled Negedge): Accum_Out = %d (Expected: 15). Act_Valid_Out=%b, Weight_Valid_Out=%b", $time, Accum_Out, Act_Valid_Out, Weight_Valid_Out);
        if (Accum_Out !== 15) begin
            $error("[%0t] ERROR TC1.3: Accum_Out expected 15, got %d", $time, Accum_Out);
        end
        if (Act_Valid_Out !== 1'b0 || Weight_Valid_Out !== 1'b0) begin // Check that pass-through valids are now low
             $error("[%0t] ERROR TC1.4: Valid outputs not de-asserted as expected. AV_Out=%b, WV_Out=%b", $time, Act_Valid_Out, Weight_Valid_Out);
        end

        // Cycle 3 (P3/N3): Accumulator uses Product_reg=0 and mac_enable=0 (from P2). Accum_Out should hold at 15.
        // This cycle is to ensure the de-assertion of valids has propagated and prevents further accumulation.
        @(posedge Clock); // P3
        @(negedge Clock); // N3: Sample outputs
        $display("[%0t] TC1 Post-P3 (Sampled Negedge): Accum_Out = %d (Expected to hold: 15)", $time, Accum_Out);
        if (Accum_Out !== 15) begin
            $error("[%0t] ERROR TC1.5: Accum_Out expected to hold 15, got %d", $time, Accum_Out);
        end

        // Test Case 2: Accumulate another value (2x4=8, on top of 15. Expected total 23)
        $display("[%0t] Test Case 2: Accumulate another value (2x4=8 on 15 -> 23)", $time);
        // At start of TC2: Accum_Out=15. Act_Valid_In/Weight_Valid_In are 0 from end of TC1.
        // Internal mac_enable for accumulator (for P4's acc) was set to 0 on P3.

        Act_In          = 8'd2;
        Act_Valid_In    = 1'b1;
        Weight_In       = 8'd4;
        Weight_Valid_In = 1'b1;

        // Cycle 4 (P4/N4 of overall test, P2.1/N2.1 for TC2): DUT computes Product_reg=8, mac_enable=1 (for P5's acc).
        // Accumulator uses P3's (PR=0, ME=0). Accum_Out remains 15.
        @(posedge Clock); // P4
        @(negedge Clock); // N4: Sample.
        $display("[%0t] TC2 Post-P4 (Sampled Negedge): Act_Out=%d, W_Out=%d, Accum_Out=%d (Expected Accum: 15)",
                 $time, Act_Out, Weight_Out, Accum_Out);
        if (Act_Out !== 8'd2 || Weight_Out !== 8'd4 || Act_Valid_Out !== 1'b1 || Weight_Valid_Out !== 1'b1 ) begin
             $error("[%0t] ERROR TC2.1: Input pass-through incorrect.", $time);
        end
        if (Accum_Out !== 15) begin
            $error("[%0t] ERROR TC2.2: Accum_Out expected 15 (old value), got %d", $time, Accum_Out);
        end

        // De-assert valids for P5's Product_reg/mac_enable computation
        Act_Valid_In    = 1'b0;
        Weight_Valid_In = 1'b0;

        // Cycle 5 (P5/N5, P2.2/N2.2 for TC2): DUT computes Product_reg=0, mac_enable=0 (for P6's acc).
        // Accumulator uses P4's (PR=8, ME=1). Accum_Out = 15 + 8 = 23.
        @(posedge Clock); // P5
        @(negedge Clock); // N5: Sample.
        $display("[%0t] TC2 Post-P5 (Sampled Negedge): Accum_Out = %d (Expected: 23). AV_Out=%b, WV_Out=%b", $time, Accum_Out, Act_Valid_Out, Weight_Valid_Out);
        if (Accum_Out !== 23) begin
            $error("[%0t] ERROR TC2.3: Accum_Out expected 23, got %d", $time, Accum_Out);
        end
         if (Act_Valid_Out !== 1'b0 || Weight_Valid_Out !== 1'b0) begin
             $error("[%0t] ERROR TC2.4: Valid outputs not de-asserted as expected. AV_Out=%b, WV_Out=%b", $time, Act_Valid_Out, Weight_Valid_Out);
        end

        // Cycle 6 (P6/N6, P2.3/N2.3 for TC2): DUT computes Product_reg=0, mac_enable=0 (for P7's acc).
        // Accumulator uses P5's (PR=0, ME=0). Accum_Out remains 23 (holds).
        @(posedge Clock); // P6
        @(negedge Clock); // N6: Sample.
        $display("[%0t] TC2 Post-P6 (Sampled Negedge): Accum_Out = %d (Expected to hold: 23)", $time, Accum_Out);
        if (Accum_Out !== 23) begin
            $error("[%0t] ERROR TC2.5: Accum_Out expected to hold 23, got %d", $time, Accum_Out);
        end

        // Test Case 3: Clear accumulator (remains largely the same, as Clear is synchronous)
        $display("[%0t] Test Case 3: Clear accumulator", $time);
        Clear = 1'b1;
        @(posedge Clock); // P7: Clear is asserted. Accumulator is cleared to 0.
        @(negedge Clock); // N7: Sample outputs.
        $display("[%0t] TC3 Clear Asserted (Sampled Negedge): Accum_Out = %d (Expected: 0)", $time, Accum_Out);
        if (Accum_Out !== 0) $error("[%0t] ERROR TC3.1: Accum_Out not 0 after clear. Value: %d", $time, Accum_Out);
        Clear = 1'b0;

        @(posedge Clock); // P8: Clear is low. Accumulator uses mac_enable from P7 (which would be based on P6's invalid inputs). Accum_Out remains 0.
        @(negedge Clock); // N8: Sample outputs.
        $display("[%0t] TC3 Clear De-asserted (Sampled Negedge): Accum_Out = %d (Expected: 0)", $time, Accum_Out);
        if (Accum_Out !== 0) $error("[%0t] ERROR TC3.2: Accum_Out not 0 after clear de-asserted. Value: %d", $time, Accum_Out);

        // Test Case 4: Inputs not valid (remains largely the same)
        $display("[%0t] Test Case 4: Inputs not valid", $time);
        Act_In = 8'd10; Act_Valid_In = 1'b0; Weight_In = 8'd10; Weight_Valid_In = 1'b1;
        @(posedge Clock); @(negedge Clock); // P9/N9
        @(posedge Clock); @(negedge Clock); // P10/N10
        $display("[%0t] TC4.2 Inval_Act (Sampled Negedge): Accum_Out = %d (Expected: 0)", $time, Accum_Out);
        if (Accum_Out !== 0) $error("[%0t] ERROR TC4.2: Accum_Out changed. Value: %d", $time, Accum_Out);

        Act_In = 8'd12; Act_Valid_In = 1'b1; Weight_In = 8'd12; Weight_Valid_In = 1'b0;
        @(posedge Clock); @(negedge Clock); // P11/N11
        @(posedge Clock); @(negedge Clock); // P12/N12
        $display("[%0t] TC4.4 Inval_Weight (Sampled Negedge): Accum_Out = %d (Expected: 0)", $time, Accum_Out);
        if (Accum_Out !== 0) $error("[%0t] ERROR TC4.4: Accum_Out changed. Value: %d", $time, Accum_Out);

        // Test Case 5: Back-to-back valid inputs (3x3=9, then 4x4=16. Expected total 9+16=25)
        $display("[%0t] Test Case 5: Back-to-back valid inputs (3x3, then 4x4)", $time);
        // Ensure Accum_Out is 0 (e.g. after clear or previous tests ending with invalid inputs and hold cycles)
        // For this test, let's explicitly clear first to ensure a known start.
        Clear = 1'b1; @(posedge Clock); Clear = 1'b0; @(negedge Clock); // Clear and sample
        @(posedge Clock); @(negedge Clock); // Hold cycle for clear to ensure mac_enable is also 0 for accumulator.
        $display("[%0t] TC5 Pre-B2B Clear. Accum_Out=%d", $time, Accum_Out);


        // --- First B2B input: 3x3 ---
        Act_In          = 8'd3; Act_Valid_In    = 1'b1; Weight_In       = 8'd3; Weight_Valid_In = 1'b1;
        @(posedge Clock); // P(B2B.1): DUT: PR_A=9, ME_A=1 (for P(B2B.2) acc). Acc: uses pre-state -> Accum_Out=0.
        @(negedge Clock); // N(B2B.1): Sample. Accum_Out=0.
        $display("[%0t] TC5 Post-P(B2B.1) (Sampled Negedge): Accum_Out=%d (Expected Accum: 0)", $time, Accum_Out);
        if (Accum_Out !== 0) $error("[%0t] ERROR TC5.1: B2B Test - Accum_Out expected 0, got %d", $time, Accum_Out);

        // --- Second B2B input: 4x4 ---
        // Inputs for 3x3 are active for one cycle of Product_reg/mac_enable calculation.
        // Now apply inputs for 4x4. These will be used for the *next* Product_reg/mac_enable calculation.
        // Act_Valid_In and Weight_Valid_In remain 1'b1 for this cycle.
        Act_In          = 8'd4; /* Act_Valid_In is still 1'b1 */ Weight_In  = 8'd4; /* Weight_Valid_In is still 1'b1 */
        @(posedge Clock); // P(B2B.2): DUT: PR_B=16, ME_B=1 (for P(B2B.3) acc). Acc: uses PR_A=9, ME_A=1 -> Accum_Out=0+9=9.
        @(negedge Clock); // N(B2B.2): Sample. Accum_Out=9.
        $display("[%0t] TC5 Post-P(B2B.2) (Sampled Negedge, After 1st B2B product accumulated): Act_Out=%d, W_Out=%d, Accum_Out=%d (Expected Accum: 9)",
                 $time, Act_Out, Weight_Out, Accum_Out);
        if (Accum_Out !== 9) $error("[%0t] ERROR TC5.2: B2B Test - Accum_Out expected 9, got %d", $time, Accum_Out);

        // De-assert valids so the 4x4 product is not re-processed for the *next* Product_reg/mac_enable
        Act_Valid_In    = 1'b0;
        Weight_Valid_In = 1'b0;

        @(posedge Clock); // P(B2B.3): DUT: PR_C=0, ME_C=0 (for P(B2B.4) acc). Acc: uses PR_B=16, ME_B=1 -> Accum_Out=9+16=25.
        @(negedge Clock); // N(B2B.3): Sample. Accum_Out=25.
        $display("[%0t] TC5 Post-P(B2B.3) (Sampled Negedge, After 2nd B2B product accumulated): Accum_Out=%d (Expected Accum: 25)",
                 $time, Accum_Out);
        if (Accum_Out !== 25) $error("[%0t] ERROR TC5.3: B2B Test - Accum_Out expected 25, got %d", $time, Accum_Out);

        @(posedge Clock); // P(B2B.4): DUT: PR_D=0, ME_D=0. Acc: uses PR_C=0, ME_C=0 -> Accum_Out=25 (holds).
        @(negedge Clock); // N(B2B.4): Sample. Accum_Out=25.
        $display("[%0t] TC5 Post-P(B2B.4) (Sampled Negedge, Accum_Out should hold): Accum_Out=%d (Expected Accum: 25)",
                 $time, Accum_Out);
        if (Accum_Out !== 25) $error("[%0t] ERROR TC5.4: B2B Test - Accum_Out expected to hold 25, got %d", $time, Accum_Out);


        $display("[%0t] All basic test cases completed.", $time);
        $finish;
    end

endmodule
