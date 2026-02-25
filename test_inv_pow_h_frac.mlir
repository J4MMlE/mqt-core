// Test: inv(pow(0.5){H}) should canonicalize to pow(-0.5){H}
// Run:  just run file=test_inv_pow_h_frac.mlir

module {
  func.func @inv_pow_h_frac() {
    %q0 = qco.alloc : !qco.qubit

    %q0_1 = qco.inv (%a = %q0) {
      %exp = arith.constant 5.000000e-01 : f64
      %a_1 = qco.pow (%exp) (%b = %a) {
        %b_1 = qco.h %b : !qco.qubit -> !qco.qubit
        qco.yield %b_1 : !qco.qubit
      } : {!qco.qubit} -> {!qco.qubit}
      qco.yield %a_1 : !qco.qubit
    } : {!qco.qubit} -> {!qco.qubit}

    qco.sink %q0_1 : !qco.qubit
    return
  }
}
