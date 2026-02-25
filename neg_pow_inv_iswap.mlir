module {
  func.func @negPowInvIswap() {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %exp = arith.constant -2.000000e+00 : f64

    %q0_1, %q1_1 = qco.pow (%exp) (%a0 = %q0, %a1 = %q1) {
      %a0_1, %a1_1 = qco.inv (%b0 = %a0, %b1 = %a1) {
        %b0_1, %b1_1 = qco.iswap %b0, %b1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        qco.yield %b0_1, %b1_1
      } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
      qco.yield %a0_1, %a1_1
    } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}

    qco.sink %q0_1 : !qco.qubit
    qco.sink %q1_1 : !qco.qubit
    return
  }
}

// Expected output:
//
// module {
//   func.func @negPowInvIswapRef() {
//     %q0 = qco.alloc : !qco.qubit
//     %q1 = qco.alloc : !qco.qubit
//     %theta = arith.constant -6.283185307179586 : f64
//     %beta = arith.constant 0.000000e+00 : f64
//
//     %q0_1, %q1_1 = qco.xx_plus_yy(%theta, %beta) %q0, %q1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
//
//     qco.sink %q0_1 : !qco.qubit
//     qco.sink %q1_1 : !qco.qubit
//     return
//   }
// }
