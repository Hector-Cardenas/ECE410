* 4x4 Resistive Crossbar MVM - Modified Resistor Network
* Y' = G'_physical * V_in
* Where G'_physical_rc = 1 / (W^T)_rc
* So, R'_physical_rc = (W^T)_rc
*
* V_in = [1V, 2V, 3V, 4V]^T
* Original W = [ 1  2  3  4;
* 5  6  7  8;
* 9 10 11 12;
* 13 14 15 16 ]
* W^T = [ 1  5  9 13;
* 2  6 10 14;
* 3  7 11 15;
* 4  8 12 16 ]
* The elements of W^T are now the resistance values in Ohms.

* --- Input Voltage Sources (Applied to Rows) ---
Vin1 ROW1 0 DC 1V  ; Input V1 to ROW1
Vin2 ROW2 0 DC 2V  ; Input V2 to ROW2
Vin3 ROW3 0 DC 3V  ; Input V3 to ROW3
Vin4 ROW4 0 DC 4V  ; Input V4 to ROW4

* --- Resistive Crossbar Array (Modified Resistances) ---
* Format: R<name> <node1> <node2> <resistance_value>
* Resistances R'_rc are numerically equal to (W^T)_rc

* Resistors connected to ROW1
R11 ROW1 COL1_SENSE 1   ; (W^T)_11 = 1 Ohm
R12 ROW1 COL2_SENSE 5   ; (W^T)_12 = 5 Ohms
R13 ROW1 COL3_SENSE 9   ; (W^T)_13 = 9 Ohms
R14 ROW1 COL4_SENSE 13  ; (W^T)_14 = 13 Ohms

* Resistors connected to ROW2
R21 ROW2 COL1_SENSE 2   ; (W^T)_21 = 2 Ohms
R22 ROW2 COL2_SENSE 6   ; (W^T)_22 = 6 Ohms
R23 ROW2 COL3_SENSE 10  ; (W^T)_23 = 10 Ohms
R24 ROW2 COL4_SENSE 14  ; (W^T)_24 = 14 Ohms

* Resistors connected to ROW3
R31 ROW3 COL1_SENSE 3   ; (W^T)_31 = 3 Ohms
R32 ROW3 COL2_SENSE 7   ; (W^T)_32 = 7 Ohms
R33 ROW3 COL3_SENSE 11  ; (W^T)_33 = 11 Ohms
R34 ROW3 COL4_SENSE 15  ; (W^T)_34 = 15 Ohms

* Resistors connected to ROW4
R41 ROW4 COL1_SENSE 4   ; (W^T)_41 = 4 Ohms
R42 ROW4 COL2_SENSE 8   ; (W^T)_42 = 8 Ohms
R43 ROW4 COL3_SENSE 12  ; (W^T)_43 = 12 Ohms
R44 ROW4 COL4_SENSE 16  ; (W^T)_44 = 16 Ohms

* --- Output Current Measurement (Columns grounded via 0V sources acting as ammeters) ---
Vamm1 COL1_SENSE 0 DC 0V ; Ammeter for Column 1 output current I(Y'1)
Vamm2 COL2_SENSE 0 DC 0V ; Ammeter for Column 2 output current I(Y'2)
Vamm3 COL3_SENSE 0 DC 0V ; Ammeter for Column 3 output current I(Y'3)
Vamm4 COL4_SENSE 0 DC 0V ; Ammeter for Column 4 output current I(Y'4)

* --- Analysis ---
.OP ; DC Operating Point Analysis

* --- Output ---
.PRINT DC I(Vamm1) I(Vamm2) I(Vamm3) I(Vamm4)

* Expected approximate values:
* I(Vamm1) = 4.0 A
* I(Vamm2) = 1.461904 A
* I(Vamm3) = 0.917171 A
* I(Vamm4) = 0.669780 A

.END
