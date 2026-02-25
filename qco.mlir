module {
    func.func @testPow() {
      %q0_0 = qco.alloc : !qco.qubit
      %c_0 = arith.constant 1.000000e+00 : f64

      %q0_1 = qco.pow (2.000000e+00) (%a = %q0_0) {
        %a_1 = qco.rz(%c_0) %a : !qco.qubit -> !qco.qubit
        qco.yield %a_1
      } : {!qco.qubit} -> {!qco.qubit}

      qco.dealloc %q0_1 : !qco.qubit
      return
    }
  }

module {
  func.func @testInv(){
    %q_in = qco.alloc : !qco.qubit

    %q_out = qco.inv (%q = %q_in) {
      %q_1 = qco.s %q : !qco.qubit -> !qco.qubit
      qco.yield %q_1
    } : {!qco.qubit} -> {!qco.qubit}

    qco.dealloc %q_out : !qco.qubit
    return
  }
}

module {
  func.func @testPowCtrl() {
    %q0_0 = qco.alloc : !qco.qubit
    %q1_0 = qco.alloc : !qco.qubit

    %q0_1, %q1_1 = qco.pow (5.000000e-01) (%a0 = %q0_0, %a1 = %q1_0) {
      %a0_1, %a1_1 = qco.ctrl(%a0) targets(%t = %a1) {
        %t_1 = qco.x %t : !qco.qubit -> !qco.qubit
        qco.yield %t_1
      } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
      qco.yield %a0_1, %a1_1
    } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}

    qco.dealloc %q0_1 : !qco.qubit
    qco.dealloc %q1_1 : !qco.qubit
    return
  }
}

module {
  func.func @testNestedPow() {
    %q0_0 = qco.alloc : !qco.qubit
    %c_0 = arith.constant 1.000000e+00 : f64

    %q0_1 = qco.pow (3.000000e+00) (%a = %q0_0) {
      %a_1 = qco.pow (2.000000e+00) (%b = %a) {
        %b_1 = qco.rz(%c_0) %b : !qco.qubit -> !qco.qubit
        qco.yield %b_1
      } : {!qco.qubit} -> {!qco.qubit}
      qco.yield %a_1
    } : {!qco.qubit} -> {!qco.qubit}

    qco.dealloc %q0_1 : !qco.qubit
    return
  }
}
