// Testbench for the simplified FIFO module
// Samples outputs on negedge clk after applying stimulus on posedge clk.
`timescale 1ns/1ps

module FIFO_tb;

    // Parameters for the FIFO instance
    localparam DATA_WIDTH = 8;  // For easier testing
    localparam DEPTH      = 16; // And a smaller depth

    // Testbench signals
    logic                           clk;
    logic                           rst;
    logic                           wr_en;
    logic [DATA_WIDTH-1:0]          data_in;
    logic                           rd_en;
    logic [DATA_WIDTH-1:0]          data_out;
    logic                           full;
    logic                           empty;

    // Instantiate the FIFO
    FIFO #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .wr_en(wr_en),
        .data_in(data_in),
        .rd_en(rd_en),
        .data_out(data_out),
        .full(full),
        .empty(empty)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period, 100MHz clock
    end

    // Test sequence
    initial begin
        integer i; // Declare loop variable
        logic [DATA_WIDTH-1:0] expected_val_tb;
        logic [DATA_WIDTH-1:0] temp_data_to_write_tb;
        logic [DATA_WIDTH-1:0] temp_expected_read_tb;
        logic [DATA_WIDTH-1:0] initial_fill_data_tb [DEPTH/2];

        $display("Starting FIFO Testbench (with negedge sampling)...");

        // 1. Initialize and Reset
        rst = 1'b1;
        wr_en = 1'b0;
        rd_en = 1'b0;
        data_in = '0;
        @(posedge clk); // Apply reset
        rst = 1'b0;     // De-assert reset
        @(posedge clk); // Release reset from DUT's perspective
        
        @(negedge clk); // Sample after reset settles
        $display("[%0t] Reset released. FIFO should be empty.", $time);
        if (!(empty === 1'b1)) begin $error("Test Failed: FIFO not empty after reset. Empty: %b", empty); end
        if (!(full === 1'b0)) begin $error("Test Failed: FIFO full after reset. Full: %b", full); end

        // 2. Write until full
        $display("[%0t] Test Case 2: Writing until FIFO is full...", $time);
        wr_en = 1'b1;
        for (i = 0; i < DEPTH; i = i + 1) begin
            data_in = i; 
            @(posedge clk); 
            
            @(negedge clk); 
            if (i == DEPTH - 1) begin 
                $display("[%0t] After writing item %0d (value %h). Full: %b, Empty: %b", $time, i, data_in, full, empty);
                if (!(full === 1'b1)) begin $error("Test Failed: FIFO not full after writing DEPTH items. (i=%0d)", i); end
                if (!(empty === 1'b0)) begin $error("Test Failed: FIFO empty after writing DEPTH items. (i=%0d)", i); end
            end else begin
                if (full) begin $error("Test Failed: FIFO reported full prematurely at item %0d.", i); end
            end
        end
        @(negedge clk); 
        if (!(full === 1'b1)) begin $error("Test Failed: FIFO not full after writing DEPTH items (final check)."); end
        wr_en = 1'b0; 
        @(posedge clk); 

        // 3. Attempt to write to full FIFO
        $display("[%0t] Test Case 3: Attempting to write to full FIFO...", $time);
        data_in = 'hFF; 
        wr_en = 1'b1;
        @(posedge clk); 
        
        @(negedge clk); 
        if (!(full === 1'b1)) begin $error("Test Failed: FIFO not reporting full after write attempt to full FIFO."); end
        wr_en = 1'b0;
        @(posedge clk);

        // 4. Read until empty
        $display("[%0t] Test Case 4: Reading until FIFO is empty...", $time);
        rd_en = 1'b1;
        for (i = 0; i < DEPTH; i = i + 1) begin
            expected_val_tb = i; 
            @(posedge clk); 
            
            @(negedge clk); 
            $display("[%0t] Read data: %h (expected: %h). Empty: %b, Full: %b", $time, data_out, expected_val_tb, empty, full);
            if (!(data_out === expected_val_tb)) begin $error("Test Failed: Read data mismatch. Expected %h, got %h", expected_val_tb, data_out); end
            
            if (i == DEPTH - 1) begin 
                if (!(empty === 1'b1)) begin $error("Test Failed: FIFO not empty after reading DEPTH items."); end
            end else begin
                if (empty) begin $error("Test Failed: FIFO reported empty prematurely at item %0d.", i); end
            end
        end
        @(negedge clk);
        if (!(empty === 1'b1)) begin $error("Test Failed: FIFO not empty after reading DEPTH items (final check)."); end
        rd_en = 1'b0; 
        @(posedge clk);

        // 5. Attempt to read from empty FIFO
        $display("[%0t] Test Case 5: Attempting to read from empty FIFO...", $time);
        rd_en = 1'b1;
        @(posedge clk); 
        
        @(negedge clk); 
        $display("[%0t] data_out after read attempt from empty: %h. Empty: %b", $time, data_out, empty);
        if (!(empty === 1'b1)) begin $error("Test Failed: FIFO not reporting empty after read attempt from empty FIFO."); end
        if (!(data_out === '0)) begin $warning("Warning: data_out was not '0' on read from empty. Current value: %h", data_out); end
        rd_en = 1'b0;
        @(posedge clk);

        // 6. Simultaneous Write and Read
        $display("[%0t] Test Case 6: Simultaneous Write and Read...", $time);
        wr_en = 1'b1;
        for (i = 0; i < DEPTH / 2; i = i + 1) begin
            initial_fill_data_tb[i] = i + 'hA0; 
            data_in = initial_fill_data_tb[i];
            @(posedge clk);
        end
        wr_en = 1'b0;
        @(posedge clk); @(negedge clk); 
        $display("[%0t] FIFO half-filled. Empty: %b, Full: %b", $time, empty, full);
        if (!(!empty && !full)) begin $error("FIFO not half-filled as expected for simultaneous test."); end

        for (i = 0; i < DEPTH; i = i + 1) begin 
            temp_data_to_write_tb = i + 'hC0; 
            if (i < DEPTH/2) begin
                temp_expected_read_tb = initial_fill_data_tb[i];
            end else begin
                temp_expected_read_tb = (i - DEPTH/2) + 'hC0; 
            end

            data_in = temp_data_to_write_tb;
            wr_en = 1'b1;
            rd_en = 1'b1;
            @(posedge clk); 
            
            @(negedge clk); 
            $display("[%0t] Sim R/W (iter %0d): Read: %h (Exp: %h), Written: %h. Empty: %b, Full: %b",
                     $time, i, data_out, temp_expected_read_tb, temp_data_to_write_tb, empty, full);
            if (!(data_out === temp_expected_read_tb)) begin $error("Sim R/W Failed: Read data mismatch on iter %0d. Got %h, Exp %h", i, data_out, temp_expected_read_tb); end
        end
        wr_en = 1'b0;
        rd_en = 1'b0;
        @(posedge clk);

        $display("[%0t] Reading out remaining items from Sim R/W test...", $time);
        rd_en = 1'b1;
        for (i = 0; i < DEPTH / 2; i = i + 1) begin 
            temp_expected_read_tb = (i + DEPTH/2) + 'hC0; 
            @(posedge clk);
            @(negedge clk);
            $display("[%0t] Sim R/W Post: Read: %h (Exp: %h). Empty: %b", $time, data_out, temp_expected_read_tb, empty);
            if(!(data_out === temp_expected_read_tb)) begin $error("Sim R/W Post: Read data mismatch on iter %0d. Got %h, Exp %h", i, data_out, temp_expected_read_tb); end
        end
        @(negedge clk);
        if(!(empty === 1'b1)) begin $error("Sim R/W Post: FIFO not empty after reading remaining items."); end
        rd_en = 1'b0;
        @(posedge clk);

        // 7. Test Reset during operation
        $display("[%0t] Test Case 7: Reset during operation...", $time);
        wr_en = 1'b1;
        data_in = 'hDD; @(posedge clk);
        data_in = 'hEE; @(posedge clk);
        wr_en = 1'b0;
        @(posedge clk); @(negedge clk); 
        if (!(empty === 1'b0)) begin $error("Test Failed: FIFO empty before reset test."); end

        rst = 1'b1; 
        @(posedge clk);
        @(posedge clk); 
        rst = 1'b0; 
        @(posedge clk); 
        
        @(negedge clk); 
        $display("[%0t] Reset applied and released. FIFO should be empty.", $time);
        if (!(empty === 1'b1)) begin $error("Test Failed: FIFO not empty after mid-operation reset."); end
        if (!(full === 1'b0)) begin $error("Test Failed: FIFO full after mid-operation reset."); end

        $display("[%0t] All FIFO test cases completed.", $time);
        $finish;
    end

endmodule

