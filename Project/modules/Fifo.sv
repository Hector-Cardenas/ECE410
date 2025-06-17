// FIFO (First-In, First-Out) Buffer Module - Simplified with Registered Output
//
// Parameters:
//   DATA_WIDTH: Bit width of the data words stored in the FIFO.
//   DEPTH:      Number of DATA_WIDTH-bit words the FIFO can store.
//
// Ports:
//   clk:          Clock signal.
//   rst:          Synchronous reset (active high).
//   wr_en:        Write enable. Data is written on posedge clk if wr_en is high and FIFO is not full.
//   data_in:      Data to be written into the FIFO.
//   rd_en:        Read enable. Data is read on posedge clk if rd_en is high and FIFO is not empty.
//   data_out:     Registered data read from the FIFO. Valid on the cycle AFTER rd_en if not empty.
//   full:         Asserted when the FIFO is full.
//   empty:        Asserted when the FIFO is empty.

module FIFO #(
    parameter DATA_WIDTH = 256, // Default for ASIC target, wide bus
    parameter DEPTH      = 1024 // Default for ASIC target, reasonable buffering
) (
    // Clock and Reset
    input  logic                           clk,
    input  logic                           rst,

    // Write Port
    input  logic                           wr_en,
    input  logic [DATA_WIDTH-1:0]          data_in,

    // Read Port
    input  logic                           rd_en,
    output logic [DATA_WIDTH-1:0]          data_out, // Will be registered

    // Status Signals
    output logic                           full,
    output logic                           empty
);

    localparam ADDR_WIDTH = (DEPTH == 0) ? 1 : ((DEPTH == 1) ? 0 : $clog2(DEPTH));

    logic [DATA_WIDTH-1:0] mem_array [0:DEPTH-1];
    logic [ADDR_WIDTH-1:0] wr_ptr_reg;
    logic [ADDR_WIDTH-1:0] rd_ptr_reg;
    logic [$clog2(DEPTH+1)-1:0] item_count_reg;

    logic do_write;
    logic do_read;

    // Registered output for data_out
    logic [DATA_WIDTH-1:0] data_out_reg;
    assign data_out = data_out_reg;

    //--------------------------------------------------------------------------
    // Write Logic
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            wr_ptr_reg <= '0;
        end else if (do_write) begin // do_write is wr_en && !full
            mem_array[wr_ptr_reg] <= data_in;
            wr_ptr_reg            <= wr_ptr_reg + 1'b1;
        end
    end

    //--------------------------------------------------------------------------
    // Read Logic (Pointer and Registered Output Data)
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            rd_ptr_reg   <= '0;
            data_out_reg <= '0; // Reset registered output
        end else begin
            if (do_read) begin // do_read is rd_en && !empty
                data_out_reg <= mem_array[rd_ptr_reg]; // Latch current head data
                rd_ptr_reg   <= rd_ptr_reg + 1'b1;   // Advance pointer for next read
            end else if (rd_en && empty) begin // Attempt to read from empty
                data_out_reg <= '0; // Or hold: data_out_reg <= data_out_reg; (implicit)
                                    // For simplicity and defined behavior, output '0'
            end
            // If !do_read and !(rd_en && empty), data_out_reg holds its previous value.
        end
    end

    //--------------------------------------------------------------------------
    // Status Logic (Full, Empty, Item Count)
    //--------------------------------------------------------------------------
    assign empty = (item_count_reg == 0);
    assign full  = (item_count_reg == DEPTH);

    assign do_write = wr_en && !full;
    assign do_read  = rd_en && !empty;

    always_ff @(posedge clk) begin
        if (rst) begin
            item_count_reg <= '0;
        end else begin
            if (do_write && !do_read) begin
                item_count_reg <= item_count_reg + 1'b1;
            end else if (!do_write && do_read) begin
                item_count_reg <= item_count_reg - 1'b1;
            end
            // If (do_write && do_read), item_count_reg remains unchanged.
            // If (!do_write && !do_read), item_count_reg remains unchanged.
        end
    end

endmodule
