OPENQASM 3.0;
include "stdgates.inc";

// --- Basic registers ---
qubit[3] q;
bit[3] c;

// --- Standard gates ---
h q[0];
x q[1];
rz(1.5707963) q[2];

// --- 2-qubit gates ---
cx q[0], q[1];
rzz(0.5) q[1], q[2];

// --- Barrier ---
barrier q[0], q[1], q[2];

// --- ctrl modifier ---
ctrl @ x q[0], q[2];

// --- inv modifier ---
inv @ t q[1];

// --- Nested ctrl @ inv ---
ctrl @ inv @ s q[0], q[2];

// --- User-defined compound gate ---
gate mybell a, b {
    h a;
    cx a, b;
}
mybell q[0], q[1];

// --- Measure and if/else ---
c[0] = measure q[0];
c[1] = measure q[1];

if (c[0]) {
    z q[2];
}

if (c[1]) {
    x q[2];
} else {
    h q[2];
}

c[2] = measure q[2];
