module Accumulator(In, Valid, Clear, Reset, Clock, Out);

parameter INPUT_N       = 16;
parameter ACCUMULATOR_N = 32;

input signed [INPUT_N-1:0] In;
input Valid, Clear, Reset, Clock;

output signed [ACCUMULATOR_N-1:0] Out;

logic signed [ACCUMULATOR_N-1:0] Accumulated;

always_ff @(posedge Clock)
    if (Reset || Clear) Accumulated <= '0;
    else if (Valid)     Accumulated <= Accumulated + In;

assign Out = Accumulated;

endmodule

module SystolicNode(Act_In, Weight_In, Act_Valid_In, Weight_Valid_In, Clear, Reset, Clock,
                    Act_Out, Weight_Out, Act_Valid_Out, Weight_Valid_Out, Accum_Out);

parameter INPUTS_N = 8;
parameter ACCUM_OUT_N = 32;

input signed [INPUTS_N-1:0] Act_In, Weight_In;
input Act_Valid_In, Weight_Valid_In, Clear, Reset, Clock;

output logic signed [INPUTS_N-1:0] Act_Out, Weight_Out;
output logic signed [ACCUM_OUT_N-1:0] Accum_Out;
output logic Act_Valid_Out, Weight_Valid_Out;

logic signed [INPUTS_N*2-1:0] Product;

Accumulator #(INPUTS_N*2, ACCUM_OUT_N) acc0(.In(Product),
                                            .Valid(Act_Valid_Out & Weight_Valid_Out),
                                            .Clear(Clear), .Reset(Reset), .Clock(Clock),
                                            .Out(Accum_Out));

always_FF @(posedge Clock)
    if (Reset) begin
        Act_Out <= '0;
        Act_Valid_Out <= '0;
        Weight_Out <= '0;
        Weight_Valid_Out <= '0;
        Product <= '0;
    end
    else begin
        Act_Valid_Out <= Act_Valid_In;
        Weight_Valid_Out <= Weight_Valid_In;       
        Act_Out <= Act_In;
        Weight_Out <= Weight_In;
        if (Act_Valid_In && Weight_Valid_In) Product <= Act_In * Weight_In;
        else Product <= '0;
    end

endmodule
