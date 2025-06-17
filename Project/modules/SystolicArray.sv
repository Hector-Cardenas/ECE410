// Systolic Array Module
// This module instantiates a 2D array of SystolicNode PEs and connects them.
// It takes activation inputs from the top, weight inputs from the left,
// and outputs accumulated values from each PE, along with propagated
// activations (bottom) and weights (right).

module SystolicArray #(
    parameter ARRAY_INPUTS_N  = 8,  // Bit width for Activation and Weight inputs to each PE
    parameter ARRAY_OUTPUTS_N = 32, // Bit width for the Accumulator output from each PE
    parameter ARRAY_ROWS      = 64, // Number of rows in the systolic array (Default: 64)
    parameter ARRAY_COLUMNS   = 64  // Number of columns in the systolic array (Default: 64)
) (
    // Inputs for the top row of PEs
    input  logic signed [ARRAY_INPUTS_N-1:0] Acts_In_Top             [ARRAY_COLUMNS-1:0],
    input  logic                             Act_Valids_In_Top       [ARRAY_COLUMNS-1:0],

    // Inputs for the leftmost column of PEs
    input  logic signed [ARRAY_INPUTS_N-1:0] Weights_In_Left         [ARRAY_ROWS-1:0],
    input  logic                             Weight_Valids_In_Left   [ARRAY_ROWS-1:0],

    // Control Inputs
    input  logic                             Clear_Row               [ARRAY_ROWS-1:0],    // Per-row clear enable for accumulators
    input  logic                             Clear_Column            [ARRAY_COLUMNS-1:0], // Per-column clear enable for accumulators
    input  logic                             Reset,                                       // Global reset
    input  logic                             Clock,                                       // Global clock

    // Outputs from the bottom row of PEs
    output logic signed [ARRAY_INPUTS_N-1:0] Acts_Out_Bottom         [ARRAY_COLUMNS-1:0],
    output logic                             Act_Valids_Out_Bottom   [ARRAY_COLUMNS-1:0],

    // Outputs from the rightmost column of PEs
    output logic signed [ARRAY_INPUTS_N-1:0] Weights_Out_Right       [ARRAY_ROWS-1:0],
    output logic                             Weight_Valids_Out_Right [ARRAY_ROWS-1:0],

    // Accumulated outputs from all PEs
    output logic signed [ARRAY_OUTPUTS_N-1:0] Accs_Out               [ARRAY_ROWS-1:0][ARRAY_COLUMNS-1:0]
);

    // Internal wires for connecting PEs.
    // The packed width includes 1 bit for the valid signal + ARRAY_INPUTS_N bits for data.
    localparam PACKED_WIDTH = ARRAY_INPUTS_N + 1;

    // Activation wires: ARRAY_ROWS+1 rows to connect inputs to top row and outputs from bottom row.
    // Each element is [PACKED_WIDTH-1:0] wide.
    logic [PACKED_WIDTH-1:0] Act_Wires [ARRAY_ROWS:0][ARRAY_COLUMNS-1:0];

    // Weight wires: ARRAY_COLUMNS+1 columns to connect inputs to left col and outputs from right col.
    // Each element is [PACKED_WIDTH-1:0] wide.
    logic [PACKED_WIDTH-1:0] Weight_Wires [ARRAY_ROWS-1:0][ARRAY_COLUMNS:0];

    // genvar for generate loops
    genvar row, col;

    // Generate block for creating the array of PEs and their connections
    generate
        // Connect top-level activation inputs to the first row of Act_Wires (row 0)
        // The MSB of the packed wire is the valid signal.
        for (col = 0; col < ARRAY_COLUMNS; col++) begin : connect_top_act_inputs_loop
            assign Act_Wires[0][col] = {Act_Valids_In_Top[col], Acts_In_Top[col]};
        end

        // Connect top-level weight inputs to the first column of Weight_Wires (column 0)
        // The MSB of the packed wire is the valid signal.
        for (row = 0; row < ARRAY_ROWS; row++) begin : connect_left_weight_inputs_loop
            assign Weight_Wires[row][0] = {Weight_Valids_In_Left[row], Weights_In_Left[row]};
        end

        // Instantiate the 2D array of SystolicNode PEs
        for (row = 0; row < ARRAY_ROWS; row++) begin : gen_rows_loop
            for (col = 0; col < ARRAY_COLUMNS; col++) begin : gen_cols_systolic_node_loop
                // Instantiate SystolicNode.
                // Parameters are passed by name for clarity.
                // Assumes SystolicNode module is defined elsewhere and has parameters:
                // INPUTS_N (for Act_In, Weight_In width)
                // ACCUM_OUT_N (for Accum_Out width)
                SystolicNode #(
                    .INPUTS_N(ARRAY_INPUTS_N),
                    .ACCUM_OUT_N(ARRAY_OUTPUTS_N)
                ) node_instance (
                    .Clock(Clock),
                    .Reset(Reset),
                    .Clear(Clear_Row[row] & Clear_Column[col]), // PE accumulator clear is AND of row and column clear

                    // Activation path (data flows from top to bottom)
                    // Unpack valid bit (MSB) and data from Act_Wires
                    .Act_Valid_In(Act_Wires[row][col][ARRAY_INPUTS_N]),
                    .Act_In(Act_Wires[row][col][ARRAY_INPUTS_N-1:0]),
                    // Connect outputs to the next row in Act_Wires
                    .Act_Valid_Out(Act_Wires[row+1][col][ARRAY_INPUTS_N]),
                    .Act_Out(Act_Wires[row+1][col][ARRAY_INPUTS_N-1:0]),

                    // Weight path (data flows from left to right)
                    // Unpack valid bit (MSB) and data from Weight_Wires
                    .Weight_Valid_In(Weight_Wires[row][col][ARRAY_INPUTS_N]),
                    .Weight_In(Weight_Wires[row][col][ARRAY_INPUTS_N-1:0]),
                    // Connect outputs to the next column in Weight_Wires
                    .Weight_Valid_Out(Weight_Wires[row][col+1][ARRAY_INPUTS_N]),
                    .Weight_Out(Weight_Wires[row][col+1][ARRAY_INPUTS_N-1:0]),

                    // Accumulator output
                    .Accum_Out(Accs_Out[row][col])
                );
            end
        end

        // Connect last row of Act_Wires (row ARRAY_ROWS) to the module's activation outputs
        for (col = 0; col < ARRAY_COLUMNS; col++) begin : connect_bottom_act_outputs_loop
            // Unpack valid bit (MSB) and data to respective output ports
            assign {Act_Valids_Out_Bottom[col], Acts_Out_Bottom[col]} = Act_Wires[ARRAY_ROWS][col];
        end

        // Connect last column of Weight_Wires (column ARRAY_COLUMNS) to the module's weight outputs
        for (row = 0; row < ARRAY_ROWS; row++) begin : connect_right_weight_outputs_loop
            // Unpack valid bit (MSB) and data to respective output ports
            assign {Weight_Valids_Out_Right[row], Weights_Out_Right[row]} = Weight_Wires[row][ARRAY_COLUMNS];
        end

    endgenerate

endmodule

