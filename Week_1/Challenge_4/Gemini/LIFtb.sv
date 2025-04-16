// ============================================================================
// Module: lif_neuron_tb
// Description: Testbench for the lif_neuron module.
//              Applies various stimuli to verify functionality, including
//              edge cases and saturation behavior. (Version 8: Added wait for refractory in Scen 6/7)
// Author: AI Assistant (Gemini)
// Date: 2025-04-02
// ============================================================================
`timescale 1ns/1ps // Define simulation timescale

module lif_neuron_tb;

    // --- Parameters (Match DUT or modify for specific tests) ---
    localparam int DATA_WIDTH = 16;
    localparam int LEAKAGE_SHIFT = 4;
    localparam logic signed [DATA_WIDTH-1:0] V_THRESHOLD = 'sd1000;
    localparam logic signed [DATA_WIDTH-1:0] V_RESET = 'sd0;
    localparam int REFRACTORY_PERIOD_CYCLES = 5;
    localparam logic signed [DATA_WIDTH-1:0] MAX_POTENTIAL = $signed({1'b0, {(DATA_WIDTH-1){1'b1}}});

    // --- Testbench Signals ---
    logic clk;
    logic rst_n;
    logic signed [DATA_WIDTH-1:0] i_inj;
    logic o_spike;

    // --- Instantiate the Device Under Test (DUT) ---
    // Instantiate V2.3 of the DUT (which includes explicit comb logic)
    lif_neuron #(
        .DATA_WIDTH(DATA_WIDTH),
        .LEAKAGE_SHIFT(LEAKAGE_SHIFT),
        .V_THRESHOLD(V_THRESHOLD),
        .V_RESET(V_RESET),
        .REFRACTORY_PERIOD_CYCLES(REFRACTORY_PERIOD_CYCLES)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .i_inj(i_inj),
        .o_spike(o_spike)
    );

    // --- Clock Generation ---
    localparam CLK_PERIOD = 10; // Clock period in ns
    localparam int SIM_TIMEOUT_CYCLES = 5000; // Timeout for basic firing wait
    initial begin
        clk = 0;
        forever #(CLK_PERIOD / 2) clk = ~clk;
    end

    // --- Monitor (Optional: Display signals on change) ---
    initial begin
        // Re-enable monitor to help debug
         $monitor("T=%0t: clk=%b rst_n=%b i_inj=%d v_mem=%d refractory=%d o_spike=%b",
                  $time, clk, rst_n, i_inj, dut.v_mem, dut.refractory_counter, o_spike);
    end

    // --- Stimulus Generation ---
    initial begin
        // --- Declare variables used in test scenarios ---
        logic signed [DATA_WIDTH-1:0] target_v;
        logic signed [DATA_WIDTH-1:0] leak_at_target;
        logic signed [DATA_WIDTH-1:0] target_v_exact;
        logic signed [DATA_WIDTH-1:0] leak_at_target_exact;
        int                            overflow_cycle_count;
        bit                            spike_detected_flag;
        bit                            spiked_from_max;
        bit                            needs_precharge_scen9;
        bit                            spike_occurred_scen7;


        $display("--- Testbench Started ---");

        // 1. Initialize and Reset
        rst_n = 1'b1;
        i_inj = 'sd0;
        #5;
        rst_n = 1'b0;
        $display("T=%0t: Asserting Reset (rst_n = 0)", $time);
        #(CLK_PERIOD * 2);
        rst_n = 1'b1;
        $display("T=%0t: Deasserting Reset (rst_n = 1)", $time);
        #(CLK_PERIOD);

        // --- Test Scenarios ---

        // 2. Basic Firing Test
        $display("\n--- Scenario: Basic Firing ---");
        i_inj = V_THRESHOLD / 2;
        $display("T=%0t: Applying positive current i_inj = %d", $time, i_inj);
        spike_detected_flag = 0;
        fork
            begin : wait_for_spike
                wait (o_spike == 1'b1);
                spike_detected_flag = 1;
            end
            begin : wait_timeout
                #(CLK_PERIOD * SIM_TIMEOUT_CYCLES);
            end
        join_any
        if (!spike_detected_flag) begin
            $error("TIMEOUT: Failed to detect spike in Scenario 2 within %0d cycles.", SIM_TIMEOUT_CYCLES);
        end else begin
             $display("T=%0t: ** Spike Detected (o_spike = 1)! ** v_mem=%d ref_ctr=%d", $time, dut.v_mem, dut.refractory_counter);
            @(posedge clk);
            $display("T=%0t: After spike: v_mem=%d ref_ctr=%d o_spike=%b", $time, dut.v_mem, dut.refractory_counter, o_spike);
            assert(dut.v_mem == V_RESET) else $error("FAILED: Potential did not reset to V_RESET after spike.");
            if (REFRACTORY_PERIOD_CYCLES > 0) begin
                assert(dut.refractory_counter == REFRACTORY_PERIOD_CYCLES) else $error("FAILED: Refractory counter not set correctly after spike.");
            end else begin
                 assert(dut.refractory_counter == 0) else $error("FAILED: Refractory counter not 0 when period is 0.");
            end
        end
        i_inj = 'sd0; // Remove current
        $display("T=%0t: Delaying before next scenario...", $time);
        #(CLK_PERIOD * 5);

        // 3. Refractory Period Test
        $display("\n--- Scenario: Refractory Period ---");
        if (spike_detected_flag && REFRACTORY_PERIOD_CYCLES > 0) begin
            $display("T=%0t: Waiting for refractory period (%d cycles)...", $time, REFRACTORY_PERIOD_CYCLES);
            if (dut.refractory_counter > 0) begin
                repeat (REFRACTORY_PERIOD_CYCLES - 1) begin
                    @(posedge clk);
                    i_inj = V_THRESHOLD;
                    $display("T=%0t: During refractory: i_inj=%d v_mem=%d ref_ctr=%d o_spike=%b", $time, i_inj, dut.v_mem, dut.refractory_counter, o_spike);
                    assert(o_spike == 1'b0) else $error("FAILED: Neuron spiked during refractory period!");
                    assert(dut.v_mem == V_RESET) else $error("FAILED: Potential did not stay clamped at V_RESET during refractory period.");
                end
                @(posedge clk);
                $display("T=%0t: End of refractory: i_inj=%d v_mem=%d ref_ctr=%d o_spike=%b", $time, i_inj, dut.v_mem, dut.refractory_counter, o_spike);
                assert(dut.refractory_counter == 0) else $error("FAILED: Refractory counter did not reach 0.");
            end else begin
                 $warning("WARN: Skipping refractory test checks as refractory counter was not active when expected.");
            end
        end else if (!spike_detected_flag) begin
             begin
                 $warning("WARN: Skipping refractory test as initial spike was not detected.");
             end
        end else begin
             begin
                 $display("T=%0t: Skipping refractory test as REFRACTORY_PERIOD_CYCLES is 0.", $time);
             end
        end
        i_inj = 'sd0;
        $display("T=%0t: Delaying before next scenario...", $time);
        #(CLK_PERIOD * 5);

        // 4. Sub-threshold Input
        $display("\n--- Scenario: Sub-threshold Input ---");
        i_inj = (V_THRESHOLD >>> (LEAKAGE_SHIFT + 1));
        if (i_inj == 0 && V_THRESHOLD > 0) i_inj = 1;
        $display("T=%0t: Applying weak positive current i_inj = %d", $time, i_inj);
        repeat (20) begin
            @(posedge clk);
             $display("T=%0t: Sub-threshold: v_mem=%d o_spike=%b", $time, dut.v_mem, o_spike);
            assert(o_spike == 1'b0) else $error("FAILED: Neuron spiked on sub-threshold input!");
            if (dut.v_mem >= V_THRESHOLD) $error("FAILED: Potential reached threshold on sub-threshold input!");
        end
        i_inj = 'sd0;
        $display("T=%0t: Delaying before next scenario...", $time);
        #(CLK_PERIOD * 5);

        // 5. Inhibitory Input
        $display("\n--- Scenario: Inhibitory Input ---");
        i_inj = V_THRESHOLD / 4; // Charge up slightly
        @(posedge clk); @(posedge clk); @(posedge clk);
        $display("T=%0t: Potential before inhibition: v_mem=%d", $time, dut.v_mem);
        i_inj = - (V_THRESHOLD / 2); // Apply negative current
        $display("T=%0t: Applying negative current i_inj = %d", $time, i_inj);
        repeat (5) begin
             @(posedge clk);
             $display("T=%0t: Inhibitory: v_mem=%d o_spike=%b", $time, dut.v_mem, o_spike);
             assert(o_spike == 1'b0) else $error("FAILED: Neuron spiked on inhibitory input!");
             if (dut.v_mem < V_RESET) $error("FAILED: Potential went below V_RESET (%0d)", V_RESET);
        end
        i_inj = 'sd0;
        $display("T=%0t: Delaying before next scenario...", $time);
        #(CLK_PERIOD * 5);

        // 6. Threshold Edge Case (Just Below)
        $display("\n--- Scenario: Threshold Edge Case (Just Below) ---");
        // NOTE: Precisely hitting V_THRESHOLD-1 is difficult...
        i_inj = 'sd0; @(posedge clk);
        // ** FIX: Use lower charging current **
        i_inj = V_THRESHOLD / 4; // Charge up less aggressively
        $display("T=%0t: Charging potential for Edge Case (Just Below) with i_inj = %d", $time, i_inj);
        repeat(10) @(posedge clk); // Charge for longer if needed with lower current
        // Check if spike occurred during charge-up (shouldn't with lower current)
        if (o_spike == 1'b1) begin
            $error("ERROR: Unexpected spike during charge-up for Scenario 6!");
            // Need to wait out refractory before proceeding
             if (REFRACTORY_PERIOD_CYCLES > 0) begin repeat (REFRACTORY_PERIOD_CYCLES) @(posedge clk); end
        end
        i_inj = 'sd0; // Let it leak
        $display("T=%0t: Letting potential leak towards threshold... Current v_mem=%d", $time, dut.v_mem);
        // Adjust wait condition if needed based on new charge level
        while (dut.v_mem > V_THRESHOLD + (V_THRESHOLD >> 2) && dut.v_mem > V_RESET) @(posedge clk); // Wait until closer

        // ** FIX: Wait for refractory period to end before applying fine-tuned current **
        if (dut.refractory_counter > 0) begin
            $display("T=%0t: Waiting for existing refractory period to end before edge test...", $time);
            wait (dut.refractory_counter == 0);
            @(posedge clk); // Add one cycle delay after refractory ends
            $display("T=%0t: Refractory ended.", $time);
        end

        $display("T=%0t: Applying current to target V_THRESHOLD-1. Current v_mem=%d", $time, dut.v_mem);
        target_v = V_THRESHOLD - 1;
        leak_at_target = (dut.v_mem > V_RESET) ? (dut.v_mem >>> LEAKAGE_SHIFT) : 'sd0;
        if (target_v > dut.v_mem) begin
            i_inj = leak_at_target + (target_v - dut.v_mem);
        end else begin
            i_inj = leak_at_target;
        end
        // Ensure i_inj is not excessively large if v_mem is far below target
        if (i_inj > V_THRESHOLD) i_inj = V_THRESHOLD;
        $display("T=%0t: Fine-tuned i_inj = %d", $time, i_inj);
        @(posedge clk);
        $display("T=%0t: Potential near threshold: v_mem=%d o_spike=%b", $time, dut.v_mem, o_spike);
        assert(o_spike == 1'b0) else $error("FAILED: Spiked just below threshold!");
        if (dut.v_mem >= V_THRESHOLD) $warning("WARN: Potential reached threshold unexpectedly in edge case test.");
        i_inj = 'sd0;
        $display("T=%0t: Delaying before next scenario...", $time);
        #(CLK_PERIOD * 5);

        // 7. Threshold Edge Case (Exactly At)
        $display("\n--- Scenario: Threshold Edge Case (Exactly At) ---");
        // NOTE: Precisely hitting V_THRESHOLD is difficult...
        i_inj = 'sd0; @(posedge clk);
        // ** FIX: Use lower charging current **
        i_inj = V_THRESHOLD / 4; // Charge up less aggressively
        $display("T=%0t: Charging potential for Edge Case (Exactly At) with i_inj = %d", $time, i_inj);
        repeat(10) @(posedge clk); // Charge for longer if needed
        // Check if spike occurred during charge-up (shouldn't with lower current)
         if (o_spike == 1'b1) begin
            $error("ERROR: Unexpected spike during charge-up for Scenario 7!");
            // Need to wait out refractory before proceeding
             if (REFRACTORY_PERIOD_CYCLES > 0) begin repeat (REFRACTORY_PERIOD_CYCLES) @(posedge clk); end
        end
        i_inj = 'sd0; // Let it leak
        $display("T=%0t: Letting potential leak towards threshold... Current v_mem=%d", $time, dut.v_mem);
        // Adjust wait condition if needed based on new charge level
        while (dut.v_mem > V_THRESHOLD + (V_THRESHOLD >> 2) && dut.v_mem > V_RESET) @(posedge clk); // Wait until closer

        // ** FIX: Wait for refractory period to end before applying fine-tuned current **
        if (dut.refractory_counter > 0) begin
            $display("T=%0t: Waiting for existing refractory period to end before edge test...", $time);
            wait (dut.refractory_counter == 0);
            @(posedge clk); // Add one cycle delay after refractory ends
            $display("T=%0t: Refractory ended.", $time);
        end

        $display("T=%0t: Applying current to target V_THRESHOLD. Current v_mem=%d", $time, dut.v_mem);
        target_v_exact = V_THRESHOLD;
        leak_at_target_exact = (dut.v_mem > V_RESET) ? (dut.v_mem >>> LEAKAGE_SHIFT) : 'sd0;
        if (target_v_exact > dut.v_mem) begin
             i_inj = leak_at_target_exact + (target_v_exact - dut.v_mem);
        end else begin
             i_inj = leak_at_target_exact;
        end
        // Ensure i_inj is not excessively large if v_mem is far below target
        if (i_inj > V_THRESHOLD) i_inj = V_THRESHOLD;
        $display("T=%0t: Fine-tuned i_inj = %d", $time, i_inj);
        @(posedge clk); // Apply current
        $display("T=%0t: Potential at threshold: v_mem=%d o_spike=%b", $time, dut.v_mem, o_spike);
        @(posedge clk); // Check spike on next cycle
        $display("T=%0t: Cycle after reaching threshold: v_mem=%d o_spike=%b", $time, dut.v_mem, o_spike);
        assert(o_spike == 1'b1) else $error("FAILED: Did not spike exactly at threshold!");
        i_inj = 'sd0;
        spike_occurred_scen7 = o_spike;
        if (spike_occurred_scen7 == 1'b1) begin
             if (REFRACTORY_PERIOD_CYCLES > 0) begin
                 repeat (REFRACTORY_PERIOD_CYCLES) @(posedge clk);
             end
        end
        $display("T=%0t: Delaying before next scenario...", $time);
        #(CLK_PERIOD * 5);

        // 8. Saturation Test (Overflow)
        $display("\n--- Scenario: Saturation Test (Overflow) ---");
        i_inj = MAX_POTENTIAL;
        $display("T=%0t: Applying max positive current i_inj = %d", $time, i_inj);
        overflow_cycle_count = 0;
        spiked_from_max = 0;
        repeat (10) begin
            @(posedge clk);
            overflow_cycle_count++;
            $display("T=%0t: Overflow test (cycle %0d): v_mem=%d o_spike=%b ref_ctr=%d", $time, overflow_cycle_count, dut.v_mem, o_spike, dut.refractory_counter);
            if (!spiked_from_max && dut.v_mem == MAX_POTENTIAL && dut.refractory_counter == 0) begin
                 @(posedge clk);
                 overflow_cycle_count++;
                 $display("T=%0t: Checking spike from MAX_POTENTIAL (cycle %0d): v_mem=%d o_spike=%b", $time, overflow_cycle_count, dut.v_mem, o_spike);
                 assert(o_spike == 1'b1) else $error("FAILED: Did not spike when clamped at MAX_POTENTIAL!");
                 spiked_from_max = 1;
                 if (REFRACTORY_PERIOD_CYCLES > 0) begin
                     repeat(REFRACTORY_PERIOD_CYCLES) begin
                        @(posedge clk);
                        overflow_cycle_count++;
                     end
                 end
                 // break; // Optional
            end
            if (dut.refractory_counter == 0 && o_spike == 1'b0) begin
                 if (overflow_cycle_count > 2 && dut.v_mem != MAX_POTENTIAL) begin
                     $warning("WARN: Potential (%0d) did not remain clamped at MAX_POTENTIAL (%0d) during overflow test.", dut.v_mem, MAX_POTENTIAL);
                 end
            end
        end
         i_inj = 'sd0;
        $display("T=%0t: Delaying before next scenario...", $time);
        #(CLK_PERIOD * 5);

        // 9. Saturation Test (Underflow/Reset Clamp)
        $display("\n--- Scenario: Saturation Test (Underflow/Reset Clamp) ---");
        @(posedge clk);
        needs_precharge_scen9 = (dut.v_mem == V_RESET);
        if (needs_precharge_scen9) begin
            i_inj = V_THRESHOLD / 4;
            @(posedge clk); @(posedge clk);
            $display("T=%0t: Charged potential slightly to %d before negative clamp test", $time, dut.v_mem);
        end
        i_inj = $signed({1'b1, {(DATA_WIDTH-1){1'b0}}});
        $display("T=%0t: Applying max negative current i_inj = %d", $time, i_inj);
        // #1; // Optional delay
        repeat (10) begin
            @(posedge clk);
            $display("T=%0t: Underflow test: v_mem=%d o_spike=%b", $time, dut.v_mem, o_spike);
            assert(o_spike == 1'b0) else $error("FAILED: Spiked during max negative current test!");
            assert(dut.v_mem == V_RESET) else $error("FAILED: Potential did not clamp at V_RESET (%0d) during underflow test, value=%d", V_RESET, dut.v_mem);
        end
        i_inj = 'sd0;
        $display("T=%0t: Delaying before final finish...", $time);
        #(CLK_PERIOD * 5);


        // --- Test End ---
        $display("\n--- Testbench Finished ---");
        #(CLK_PERIOD * 5);
        $finish;

    end

endmodule : lif_neuron_tb

