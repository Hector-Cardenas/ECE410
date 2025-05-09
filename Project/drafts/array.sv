module SystolicArray(Acts_In, Act_Valids_In, Weights_In, Weight_Valids_In, 
                     Clear_Row, Clear_Col, Reset, Clock, 
                     Acts_Out, Act_Valids_Out, Weights_Out, Weight_Valids_Out, Accs_Out);
parameter ARRAY_INPUTS_N;
parameter ARRAY_OUTPUTS_N;
parameter ARRAY_ROWS;
parameter ARRAY_COLUMNS;

input logic signed Acts_In [ARRAY_COLUMNS-1:0] [ARRAY_INPUTS_N-1:0];
input Act_Valids_In [ARRAY_COLUMNS-1:0];
input logic signed Weights_In [ARRAY_ROWS-1:0] [ARRAY_INPUTS_N-1:0];
input Weight_Valids_In [ARRAY_ROWS-1:0];
input Clear_Row [ARRAY_ROWS-1:0];
input Clear_Collumn [ARRAY_COLUMNS-1:0];
input Clock;

output logic Acts_Out [ARRAY_COLUMNS-1:0] [ARRAY_INPUTS_N-1:0];
output Act_Valids_Out [ARRAY_COLUMNS-1:0];
output logic Weights_Out [ARRAY_ROWS-1:0] [ARRAY_INPUTS_N-1:0];
output Weight_Valids_Out [ARRAY_ROWS-1:0];
output logic Accs_Out [ARRAY_ROWS-1:0] [ARRAY_COLLUMS-1:0] [ARRAY_OUTPUTS_N-1:0];

wire Act_Wires [ARRAY_ROWS:0] [ARRAY_COLUMNS-1:0] [ARRAY_INPUTS_N:0];
wire Weight_Wires [ARRAY_ROWS-1:0] [ARRAY_COLUMNS:0] [ARRAY_INPUTS_N:0];

genvar row, column;

generate
    for (column=0; column < ARRAY_COLUMNS; collumn++)
        assign Act_wires[0][column] = {Act_Valids_In, Acts_In[column]};

    for (row=0; row < ARRAY_ROWS; row++)
        assign Weight_Wires[row][0] = {Weight_Valids_In, Weights_In[row]};

    for (row=0; row < ARRAY_ROWS; row++)
        for (column=0; column < ARRAY_COLUMNS; collumn++) begin: systolic
            SystolicNode #(
                .INPUTS_N(ARRAY_INPUTS_N), 
                .ACCUM_OUT_N(ARRAY_OUTPUTS_N)
            ) node(
                .Clock(Clock),
                .Reset(Reset),
                .Clear(Clear_Row[row]&&Clear_Collumn[column]),
                .Act_Valid_In(Act_Wires[row][column][ARRAY_INPUTS_N]),
                .Act_In(Act_Wires[row][column][ARRAY_INPUTS_N-1:0]),
                .Weight_Valids_In(Weight_Wires[row][column][ARRAY_INPUTS_N]),
                .Weights_In(Weight_Wires[row][column][ARRAY_INPUTS_N-1:0]),
                .Act_Valid_Out(Act_Wires[row+1][column][ARRAY_INPUTS_N]),
                .Act_Out(Act_Wires[row+1][column][ARRAY_INPUTS_N-1:0]),
                .Weight_Valid_Out(Weight_Wires[row][column+1][ARRAY_INPUTS_N]),
                .Weight_Out(Weight_Wires[row][column+1][ARRAY_INPUTS_N-1:0]),
                .Accum_Out(Accs_Out[row][column])
            );
            end
 
    for (column=0; column < ARRAY_COLUMS; column++)
        assign {Act_Valids_Out[column], Acts_Out[column]} = Act_Wires[ARRAY_ROWS][column];

    for (row=0; row < ARRAY_ROWS; row++)
        assign {Weight_Valids_Out[row], Weights_Out[row]} = Weight_Wires[row][ARRAY_COLUMNS];

endgenerate      

endmodule  
