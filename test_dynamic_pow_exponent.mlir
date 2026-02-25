// RUN: mqt-opt %s -verify-diagnostics

// Test that PowOp rejects a dynamic (non-constant) exponent.
// The exponent must be produced by an arith.constant op.

func.func @dynamic_pow_exponent_qc(%exp: f64) {
  %q0 = qc.alloc : !qc.qubit
  // expected-error @below {{exponent must be a constant}}
  qc.pow(%exp) {
    qc.s %q0 : !qc.qubit
  }
  qc.dealloc %q0 : !qc.qubit
  return
}
