// Test: ctrl { pow(1/3) { sx } } should survive (pow blocked inside ctrl)
// Run:  just run file=test_ctrl_pow_sx.mlir

module {
  func.func @ctrl_pow_sx() {
    %ctrl = qco.alloc : !qco.qubit
    %tgt  = qco.alloc : !qco.qubit

    %ctrl_1, %tgt_1 = qco.ctrl(%ctrl) targets(%t = %tgt) {
      %exp = arith.constant 3.333333e-01 : f64
      %t_1 = qco.pow (%exp) (%a = %t) {
        %a_1 = qco.sx %a : !qco.qubit -> !qco.qubit
        qco.yield %a_1 : !qco.qubit
      } : {!qco.qubit} -> {!qco.qubit}
      qco.yield %t_1 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    qco.sink %ctrl_1 : !qco.qubit
    qco.sink %tgt_1  : !qco.qubit
    return
  }
}
