module tb_LIF_Neuron;

    // Testbench Parameters
    parameter int WIDTH = 16;
    parameter int V_THRESH = 32768;
    parameter int LEAK_SHIFT = 4;
    parameter int REFRACTORY = 10;
    parameter int REFRACTORY_WIDTH = 8;

    // Signals
    logic clk;
    logic rst;
    logic signed [WIDTH-1:0] I_in;
    logic spike;
    logic signed [WIDTH-1:0] V_m;  // Added V_m as an output for checking

    // Instantiate the LIF_Neuron
    LIF_Neuron #(
        .WIDTH(WIDTH),
        .V_THRESH(V_THRESH),
        .LEAK_SHIFT(LEAK_SHIFT),
        .REFRACTORY(REFRACTORY),
        .REFRACTORY_WIDTH(REFRACTORY_WIDTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .I_in(I_in),
        .spike(spike),
        .V_m(V_m)  // Pass V_m to testbench
    );

    // Clock Generation
    always begin
        #5 clk = ~clk;  // Generate a clock with a period of 10 time units
    end

    // Stimulus generation
    initial begin
        // Initialize signals
        clk = 0;
        rst = 0;
        I_in = 0;

        // Apply reset
        rst = 1;
        #10;
        rst = 0;
        
        // Test Case 1: Apply input current to cause a spike
        I_in = 5000; // Apply a small positive current
        #20;
        assert(spike == 0) else $fatal("Error: Neuron should not spike yet");
        
        I_in = 30000; // Apply a large positive current to exceed threshold
        #20;
        #1;  // Delay to allow spike signal to update
        assert(spike == 1) else $fatal("Error: Neuron should have spiked");
        
        // Wait for the spike to be deasserted after one clock cycle
        #10;
        assert(spike == 0) else $fatal("Error: Neuron should have deasserted spike after one clock cycle");
        
        // Test Case 2: Check for refractory period
        #20;
        #1;  // Delay to ensure spike signal is properly checked
        assert(spike == 0) else $fatal("Error: Neuron should not spike during refractory period");
        
        // Test Case 3: Check decay of membrane potential
        I_in = -1000; // Apply negative current to see the decay
        #100;
        assert(V_m <= 0) else $fatal("Error: Neuron potential should decay towards zero");
        
        // Test Case 4: Reset neuron and reapply input
        rst = 1; // Reset the neuron
        #10;
        rst = 0;
        I_in = 10000; // Apply a moderate positive current
        #20;
        #1;  // Delay to allow spike signal to update
        assert(spike == 0) else $fatal("Error: Neuron should not spike yet");
        
        I_in = 35000; // Apply another large positive current to cause a spike
        #20;
        #1;  // Delay to ensure spike signal is properly checked
        assert(spike == 1) else $fatal("Error: Neuron should spike again");

        // Wait for the spike to be deasserted after one clock cycle
        #10;
        assert(spike == 0) else $fatal("Error: Neuron should have deasserted spike after one clock cycle");

        $finish; // End the simulation
    end

endmodule

