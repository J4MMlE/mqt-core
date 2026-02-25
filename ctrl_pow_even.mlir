module {
  func.func @ctrlPowEvenH() {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %exp = arith.constant 2.000000e+00 : f64

    %q0_1, %q1_1 = qco.ctrl(%q0) targets(%t = %q1) {
      %t_1 = qco.pow (%exp) (%a = %t) {
        %a_1 = qco.h %a : !qco.qubit -> !qco.qubit
        qco.yield %a_1
      } : {!qco.qubit} -> {!qco.qubit}
      qco.yield %t_1
    } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})

    qco.sink %q0_1 : !qco.qubit
    qco.sink %q1_1 : !qco.qubit
    return
  }
}
