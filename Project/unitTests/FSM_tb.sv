// Testbench for Conv1D_Control_FSM - Strict Port-Level Verification (Simplified v2)
`timescale 1ns/1ps

module Conv1D_Control_FSM_tb_port_level_v2;

    // Parameters for FSM and Test
    localparam SYSTOLIC_ARRAY_PIPELINE_DEPTH_TB = 5;
    localparam NUM_CYCLES_TO_PROCESS_INPUTS_TB = 10;
    
    localparam FSM_MAX_OP_LENGTH_FOR_TB_SIZING = Conv1D_Control_FSM#()::MAX_OP_LENGTH; 
    localparam TB_OP_LENGTH_WIDTH = $clog2(FSM_MAX_OP_LENGTH_FOR_TB_SIZING); 
    
    localparam MAX_SIM_DURATION_TC1 = NUM_CYCLES_TO_PROCESS_INPUTS_TB + SYSTOLIC_ARRAY_PIPELINE_DEPTH_TB + 20;
    localparam GENERAL_TIMEOUT_CYCLES = 200; 

    // Testbench Signals
    logic clk;
    logic rst;

    logic start_conv_op;
    logic [TB_OP_LENGTH_WIDTH-1:0] num_cycles_to_process_inputs_tb; 

    logic act_fifo_empty_tb;
    logic weight_fifo_empty_tb;
    logic result_fifo_full_tb;

    logic accelerator_busy;
    logic conv_op_done;
    logic act_fifo_rd_en;
    logic weight_fifo_rd_en;
    logic sys_array_clear_accumulators;
    logic result_fifo_wr_en;

    Conv1D_Control_FSM #(
        .SYSTOLIC_ARRAY_PIPELINE_DEPTH(SYSTOLIC_ARRAY_PIPELINE_DEPTH_TB)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start_conv_op(start_conv_op),
        .num_cycles_to_process_inputs(num_cycles_to_process_inputs_tb), 
        .accelerator_busy(accelerator_busy),
        .conv_op_done(conv_op_done),
        .act_fifo_rd_en(act_fifo_rd_en),
        .act_fifo_empty(act_fifo_empty_tb),
        .weight_fifo_rd_en(weight_fifo_rd_en),
        .weight_fifo_empty(weight_fifo_empty_tb),
        .sys_array_clear_accumulators(sys_array_clear_accumulators),
        .result_fifo_wr_en(result_fifo_wr_en),
        .result_fifo_full(result_fifo_full_tb)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task apply_reset;
        rst = 1'b1;
        start_conv_op = 1'b0;
        num_cycles_to_process_inputs_tb = '0;
        act_fifo_empty_tb   = 1'b1; 
        weight_fifo_empty_tb = 1'b1; 
        result_fifo_full_tb  = 1'b0; 
        repeat(2) @(posedge clk);
        rst = 1'b0;
        @(posedge clk); 
        $display("[%0t] Reset released.", $time);
    endtask

    initial begin
        integer cycle_count;        
        integer i;                    
        integer inputs_sent_monitor;  
        integer outputs_sent_monitor; 

        $display("Starting Conv1D FSM Testbench (Port-Level, Simplified v2)...");

        apply_reset();
        @(negedge clk); 
        if (accelerator_busy || conv_op_done) begin $error("TC1.0 Error: FSM not idle after reset. Busy: %b, Done: %b", accelerator_busy, conv_op_done); end

        $display("[%0t] TC1: Basic Operation Start. Num Cycles: %0d, Pipeline Depth: %0d", $time, NUM_CYCLES_TO_PROCESS_INPUTS_TB, SYSTOLIC_ARRAY_PIPELINE_DEPTH_TB);
        num_cycles_to_process_inputs_tb = NUM_CYCLES_TO_PROCESS_INPUTS_TB;
        act_fifo_empty_tb   = 1'b0; 
        weight_fifo_empty_tb = 1'b0;
        result_fifo_full_tb  = 1'b0;

        start_conv_op = 1'b1; 
        @(posedge clk);
        start_conv_op = 1'b0;

        @(negedge clk);
        if (!accelerator_busy) begin $error("TC1.1 Error: accelerator_busy should be high. Is: %b", accelerator_busy); end
        
        @(posedge clk); @(negedge clk); 
        if (!sys_array_clear_accumulators) begin $error("TC1.1b Error: sys_array_clear_accumulators should be high in S_CLEAR. Is: %b", sys_array_clear_accumulators); end

        @(posedge clk); @(negedge clk); 
        if (sys_array_clear_accumulators) begin $error("TC1.2 Error: sys_array_clear_accumulators should be low in S_PROCESS. Is: %b", sys_array_clear_accumulators); end
        
        inputs_sent_monitor = 0;
        outputs_sent_monitor = 0;

        for (cycle_count = 0; cycle_count < MAX_SIM_DURATION_TC1; cycle_count = cycle_count + 1) begin
            @(negedge clk); 

            if (conv_op_done && cycle_count > 0) begin
                $display("[%0t] TC1: conv_op_done asserted. Cycle %0d in this loop.", $time, cycle_count);
                break; 
            end
            if (!accelerator_busy && !conv_op_done) begin $error("TC1 Error: accelerator_busy became low prematurely. Cycle %0d", cycle_count); end

            if (inputs_sent_monitor < NUM_CYCLES_TO_PROCESS_INPUTS_TB) begin
                if (!act_fifo_rd_en || !weight_fifo_rd_en) begin
                    $error("TC1 Error: Input FIFO read enables not asserted. Cycle %0d, inputs_sent %0d. act_en:%b, w_en:%b", cycle_count, inputs_sent_monitor, act_fifo_rd_en, weight_fifo_rd_en);
                end
            end else begin 
                if (act_fifo_rd_en || weight_fifo_rd_en) begin
                    $error("TC1 Error: Input FIFO read enables asserted after all inputs sent. Cycle %0d. act_en:%b, w_en:%b", cycle_count, act_fifo_rd_en, weight_fifo_rd_en);
                end
            end

            if (cycle_count >= SYSTOLIC_ARRAY_PIPELINE_DEPTH_TB && outputs_sent_monitor < NUM_CYCLES_TO_PROCESS_INPUTS_TB) begin
                if (!result_fifo_wr_en) begin
                    $error("TC1 Error: Result FIFO write enable not asserted. Cycle %0d, outputs_sent %0d. res_wr_en:%b", cycle_count, outputs_sent_monitor, result_fifo_wr_en);
                end
            end else if (outputs_sent_monitor >= NUM_CYCLES_TO_PROCESS_INPUTS_TB) begin
                if (result_fifo_wr_en && !conv_op_done) begin
                    $error("TC1 Error: Result FIFO write enable asserted after all outputs sent. Cycle %0d. res_wr_en:%b", cycle_count, result_fifo_wr_en);
                end
            end

            @(posedge clk); // Semicolon added here if it was missing in the user's version

            if (act_fifo_rd_en && weight_fifo_rd_en) begin inputs_sent_monitor = inputs_sent_monitor + 1; end
            if (result_fifo_wr_en) begin outputs_sent_monitor = outputs_sent_monitor + 1; end
        end

        if (!(conv_op_done === 1'b1)) begin $error("TC1 Error: conv_op_done not high at end of processing. inputs_sent: %0d, outputs_sent: %0d", inputs_sent_monitor, outputs_sent_monitor); end
        if (!(inputs_sent_monitor == NUM_CYCLES_TO_PROCESS_INPUTS_TB)) begin $error("TC1 Error: Wrong number of input cycles. Expected %0d, Got %0d", NUM_CYCLES_TO_PROCESS_INPUTS_TB, inputs_sent_monitor); end
        if (!(outputs_sent_monitor == NUM_CYCLES_TO_PROCESS_INPUTS_TB)) begin $error("TC1 Error: Wrong number of output cycles. Expected %0d, Got %0d", NUM_CYCLES_TO_PROCESS_INPUTS_TB, outputs_sent_monitor); end
        
        @(negedge clk); 
        if (!(accelerator_busy === 1'b0)) begin $error("TC1 Error: accelerator_busy not low during S_DONE. busy: %b", accelerator_busy); end
        if (!(conv_op_done === 1'b1)) begin $error("TC1 Error: conv_op_done was not 1 during S_DONE negedge. done: %b", conv_op_done); end

        @(posedge clk); 
        @(negedge clk);
        if (conv_op_done) begin $error("TC1 Error: conv_op_done not de-asserted (should be a pulse). done: %b", conv_op_done); end
        if (accelerator_busy) begin $error("TC1 Error: accelerator_busy not low in S_IDLE after op. busy: %b", accelerator_busy); end

        $display("[%0t] TC2: Input FIFO (act) becomes empty...", $time); // Example for line 169 (user's error line)
        apply_reset(); @(negedge clk);
        num_cycles_to_process_inputs_tb = 5; // Example for line 171 (user's error line)
        act_fifo_empty_tb = 1'b0; weight_fifo_empty_tb = 1'b0; result_fifo_full_tb = 1'b0;
        start_conv_op = 1'b1; @(posedge clk); start_conv_op = 1'b0;
        repeat(2) @(posedge clk); 

        inputs_sent_monitor = 0;
        for(i=0; i<2; i=i+1) begin 
            @(negedge clk); if(!(act_fifo_rd_en && weight_fifo_rd_en)) begin $error("TC2 Error pre-empty"); end
            @(posedge clk);
            if(act_fifo_rd_en && weight_fifo_rd_en) begin inputs_sent_monitor = inputs_sent_monitor + 1; end
        end
        act_fifo_empty_tb = 1'b1; 
        @(posedge clk); @(negedge clk);
        if(act_fifo_rd_en) begin $error("TC2 Error: act_fifo_rd_en should be low when empty."); end
        if(!weight_fifo_empty_tb && (inputs_sent_monitor < num_cycles_to_process_inputs_tb) && !weight_fifo_rd_en) begin 
            $error("TC2 Error: weight_fifo_rd_en should be high if weight FIFO not empty and inputs not done.");
        end
        act_fifo_empty_tb = 1'b0; 
        for(cycle_count=0; cycle_count < GENERAL_TIMEOUT_CYCLES && !conv_op_done; cycle_count = cycle_count+1) begin @(posedge clk); end // Semicolon added
        @(negedge clk); if(!conv_op_done) begin $error("TC2 Error: Did not complete after FIFO active."); end


        $display("[%0t] TC3: Output FIFO becomes full...", $time);
        apply_reset(); @(negedge clk);
        num_cycles_to_process_inputs_tb = SYSTOLIC_ARRAY_PIPELINE_DEPTH_TB + 3; 
        act_fifo_empty_tb = 1'b0; weight_fifo_empty_tb = 1'b0; result_fifo_full_tb = 1'b0;
        start_conv_op = 1'b1; @(posedge clk); start_conv_op = 1'b0;
        
        repeat(1 + SYSTOLIC_ARRAY_PIPELINE_DEPTH_TB) @(posedge clk); 
        @(negedge clk);
        if(!result_fifo_wr_en) begin $error("TC3 Error: result_fifo_wr_en should be high before full."); end
        result_fifo_full_tb = 1'b1; 
        @(posedge clk); @(negedge clk);
        if(result_fifo_wr_en) begin $error("TC3 Error: result_fifo_wr_en should be low when full."); end
        result_fifo_full_tb = 1'b0; 
        for(cycle_count=0; cycle_count < GENERAL_TIMEOUT_CYCLES && !conv_op_done; cycle_count = cycle_count+1) begin @(posedge clk); end // Semicolon added
        @(negedge clk); if(!conv_op_done) begin $error("TC3 Error: Did not complete after FIFO not full."); end


        $display("[%0t] All FSM test cases completed.", $time);
        $finish;
    end

endmodule

