// Accumulator Module
// This module performs accumulation of input values.
// It can be cleared or reset, and only accumulates when 'Valid' is asserted.

module Accumulator #(
    parameter INPUT_N       = 16, // Bit width of the input to be accumulated
    parameter ACCUMULATOR_N = 32  // Bit width of the internal accumulator and output
) (
    // Inputs
    input   logic                           Clock, // Clock signal
    input   logic                           Reset, // When asserted, accumulator is synchronously reset to 0
    input   logic                           Clear, // When asserted, accumulator is synchronously cleared to 0
    input   logic                           Valid, // When asserted, 'In' is added to the accumulator
    input   logic signed [INPUT_N-1:0]      In,    // Input value to be added to the accumulator

    // Outputs
    output  logic signed [ACCUMULATOR_N-1:0] Out    // Output of the accumulator
);

    // Internal register for accumulation
    logic signed [ACCUMULATOR_N-1:0] Accumulated;

    // Synchronous logic for accumulation
    // Reset or Clear takes precedence.
    // Accumulates 'In' if 'Valid' is asserted and not Reset or Clear.
    always_ff @(posedge Clock) begin
        if (Reset || Clear) begin
            Accumulated <= '0;
        end else if (Valid) begin
            Accumulated <= Accumulated + In;
        end
        // If not Reset, Clear, or Valid, 'Accumulated' holds its value implicitly
    end

    // Assign accumulated value to output
    assign Out = Accumulated;

endmodule

// Systolic Node (Processing Element - PE) Module
// This module represents a single node in a systolic array.
// It performs a MAC operation (Product + Accumulation) and passes
// activations and weights systolically.
module SystolicNode #(
    parameter INPUTS_N    = 8,  // Bit width for Activation and Weight inputs
    parameter ACCUM_OUT_N = 32  // Bit width for the Accumulator output
) (
    // Inputs
    input   logic                           Clock,            // Clock signal
    input   logic                           Reset,            // Global reset signal
    input   logic                           Clear,            // Clear signal for the internal accumulator
    input   logic                           Act_Valid_In,     // Valid signal for Activation input
    input   logic signed [INPUTS_N-1:0]     Act_In,           // Activation input from top/previous PE
    input   logic                           Weight_Valid_In,  // Valid signal for Weight input
    input   logic signed [INPUTS_N-1:0]     Weight_In,        // Weight input from left/previous PE

    // Outputs
    output  logic                           Act_Valid_Out,    // Registered Valid signal for Activation output
    output  logic signed [INPUTS_N-1:0]     Act_Out,          // Registered Activation output to bottom/next PE
    output  logic                           Weight_Valid_Out, // Registered Valid signal for Weight output
    output  logic signed [INPUTS_N-1:0]     Weight_Out,       // Registered Weight output to right/next PE
    output  logic signed [ACCUM_OUT_N-1:0]  Accum_Out         // Output from the internal accumulator
);

    // Define product width locally for clarity
    localparam PRODUCT_N = INPUTS_N * 2;

    // Internal signal for the product of Activation and Weight
    logic signed [PRODUCT_N-1:0] Product_reg; // Registered product

    // Internal signal for enabling the accumulator, derived from registered valid signals
    logic                        mac_enable;

    // Instantiate the Accumulator module
    // The accumulator's input width is PRODUCT_N (for the product)
    // The accumulator's valid signal (mac_enable) is derived from the
    // registered valid outputs (Act_Valid_Out, Weight_Valid_Out) of this PE,
    // ensuring it aligns with the Product_reg.
    Accumulator #(
        .INPUT_N(PRODUCT_N),
        .ACCUMULATOR_N(ACCUM_OUT_N)
    ) acc0 (
        .Clock(Clock),
        .Reset(Reset),
        .Clear(Clear),
        .Valid(mac_enable), // Use the pipelined mac_enable
        .In(Product_reg),
        .Out(Accum_Out)
    );

    // Synchronous logic for registering inputs, calculating product, and passing data
    always_ff @(posedge Clock) begin
        if (Reset) begin
            // Reset all registered outputs and internal product to '0'
            Act_Out          <= '0;
            Act_Valid_Out    <= 1'b0;
            Weight_Out       <= '0;
            Weight_Valid_Out <= 1'b0;
            Product_reg      <= '0;
            mac_enable       <= 1'b0; // Reset the accumulator enable signal
        end
        else begin
            // Stage 1: Register inputs and their valid signals
            Act_Out          <= Act_In;         // Systolic pass-through of activation
            Act_Valid_Out    <= Act_Valid_In;   // Systolic pass-through of activation valid
            Weight_Out       <= Weight_In;      // Systolic pass-through of weight
            Weight_Valid_Out <= Weight_Valid_In;// Systolic pass-through of weight valid

            // Stage 2: Calculate product based on current inputs and determine mac_enable for next cycle
            // The product and its enable signal are registered to be used by the accumulator in the next cycle.
            if (Act_Valid_In && Weight_Valid_In) begin
                Product_reg <= Act_In * Weight_In;
                mac_enable  <= 1'b1; // Enable accumulator for the product calculated this cycle
            end else begin
                Product_reg <= '0;   // Product is '0' if inputs are not valid
                mac_enable  <= 1'b0; // Do not enable accumulator
            end
        end
    end

endmodule
