// Control FSM for the Conv1D Accelerator - Simplified Version 2
// Assumes pre-skewed and pre-formatted data is fed via FIFOs.
// CPU/cocotb provides num_cycles_to_process_inputs which dictates how many
// cycles valid data is streamed into the systolic array.

module Conv1D_Control_FSM #(
    parameter SYSTOLIC_ARRAY_PIPELINE_DEPTH = 5,    // Cycles from array input until first valid array output
    parameter MAX_OP_LENGTH                 = 1024 * 8 // Max value for num_cycles_to_process_inputs
) (
    // Clock and Reset
    input  logic clk,
    input  logic rst,

    // CPU/Cocotb Interface
    input  logic start_conv_op,                         // Start signal for a Conv1D operation
    input  logic [$clog2(MAX_OP_LENGTH)-1:0] num_cycles_to_process_inputs, // How many input sets to stream
    output logic accelerator_busy,
    output logic conv_op_done,                          // Pulsed high for one cycle when done

    // Input FIFO Control/Status (for Activations)
    output logic act_fifo_rd_en,
    input  logic act_fifo_empty,

    // Input FIFO Control/Status (for Weights)
    output logic weight_fifo_rd_en,
    input  logic weight_fifo_empty,

    // Systolic Array Control
    output logic sys_array_clear_accumulators,          // Drives Clear_Row & Clear_Column for all PEs

    // Output FIFO Control/Status (for Results)
    output logic result_fifo_wr_en,
    input  logic result_fifo_full
);

    // FSM States
    typedef enum logic [1:0] { // Reduced states
        S_IDLE,
        S_CLEAR,    // Latch params, reset counters, clear accumulators
        S_PROCESS,  // Stream data in and out
        S_DONE      // Signal completion
    } state_t;

    state_t current_state, next_state;

    // Internal Registers for operation parameters and progress tracking
    logic [$clog2(MAX_OP_LENGTH)-1:0] op_length_reg;             // Stores num_cycles_to_process_inputs
    logic [$clog2(MAX_OP_LENGTH)-1:0] input_cycles_done_reg;     // Counts input sets sent to systolic array
    logic [$clog2(MAX_OP_LENGTH)-1:0] output_cycles_done_reg;    // Counts output sets received from systolic array
    logic [$clog2(SYSTOLIC_ARRAY_PIPELINE_DEPTH)-1:0] pipeline_fill_count_reg; // Tracks initial pipeline fill

    // Intermediate signals for enabling actions
    logic can_read_inputs;
    logic can_write_output;
    logic inputs_fully_streamed;
    logic outputs_fully_streamed;
    logic pipeline_is_filled;

    //------------------------------------------------------
    // Next State Logic (Combinational)
    //------------------------------------------------------
    always_comb begin
        next_state = current_state; // Default: stay in current state
        case (current_state)
            S_IDLE: begin
                if (start_conv_op) begin
                    next_state = S_CLEAR; 
                end
            end
            S_CLEAR: begin
                next_state = S_PROCESS;
            end
            S_PROCESS: begin
                // Transition to S_DONE once all inputs have been streamed AND all corresponding outputs have been written
                if (inputs_fully_streamed && outputs_fully_streamed) begin
                    next_state = S_DONE;
                end
            end
            S_DONE: begin
                next_state = S_IDLE;
            end
            default: begin
                next_state = S_IDLE;
            end
        endcase
    end

    //------------------------------------------------------
    // State Register and Counters (Sequential)
    //------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state             <= S_IDLE;
            op_length_reg             <= '0;
            input_cycles_done_reg     <= '0;
            output_cycles_done_reg    <= '0;
            pipeline_fill_count_reg   <= '0;
        end else begin
            current_state <= next_state;

            if (current_state == S_IDLE && next_state == S_CLEAR) begin // Transitioning from IDLE to CLEAR
                op_length_reg             <= num_cycles_to_process_inputs;
                input_cycles_done_reg     <= '0; // Reset counters here
                output_cycles_done_reg    <= '0;
                pipeline_fill_count_reg   <= '0;
            end

            if (current_state == S_PROCESS) begin
                if (can_read_inputs) begin
                    input_cycles_done_reg <= input_cycles_done_reg + 1'b1;
                end

                if (!pipeline_is_filled) begin // Only increment until full
                    pipeline_fill_count_reg <= pipeline_fill_count_reg + 1'b1;
                end

                if (can_write_output) begin
                    output_cycles_done_reg <= output_cycles_done_reg + 1'b1;
                end
            end
        end
    end

    //------------------------------------------------------
    // Output Logic & Intermediate Signals (Combinational)
    //------------------------------------------------------

    assign inputs_fully_streamed = (input_cycles_done_reg == op_length_reg);
    assign outputs_fully_streamed = (output_cycles_done_reg == op_length_reg);
    assign pipeline_is_filled = (pipeline_fill_count_reg == SYSTOLIC_ARRAY_PIPELINE_DEPTH);

    assign can_read_inputs = (current_state == S_PROCESS) && !inputs_fully_streamed && !act_fifo_empty && !weight_fifo_empty;
    assign act_fifo_rd_en    = can_read_inputs;
    assign weight_fifo_rd_en = can_read_inputs;

    assign sys_array_clear_accumulators = (current_state == S_CLEAR);

    // Corrected: S_FLUSH_PIPELINE was removed from state machine logic.
    // Output can happen in S_PROCESS once pipeline is filled.
    assign can_write_output = (current_state == S_PROCESS) && 
                              pipeline_is_filled && !outputs_fully_streamed && !result_fifo_full;
    assign result_fifo_wr_en = can_write_output;

    assign conv_op_done     = (current_state == S_DONE);
    assign accelerator_busy = (current_state != S_IDLE) || (start_conv_op && current_state == S_IDLE);

endmodule

