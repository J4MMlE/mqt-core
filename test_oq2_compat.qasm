OPENQASM 2.0;
include "qelib1.inc";

// --- Basic registers ---
qreg q[3];
creg c[3];

// --- Standard gates ---
h q[0];
x q[1];
rz(1.5707963) q[2];

// --- OQ2-style controlled gates (c-prefix compat) ---
cx q[0], q[1];
ccx q[0], q[1], q[2];

// --- Barrier ---
barrier q[0], q[1], q[2];

// --- Measure and if ---
measure q[0] -> c[0];
measure q[1] -> c[1];

if (c[0] == 1) x q[2];

measure q[2] -> c[2];
