// // Test: long chain of repeated RX -> RY -> RZ (18 gates)
// // All SU(2), so no GPhaseOp expected — should merge into a single UOp.
// module {
//   func.func @testLongChain() {
//     %q0 = qco.alloc : !qco.qubit
//     %c = arith.constant 1.000000e+00 : f64
//     %q1  = qco.rx(%c) %q0  : !qco.qubit -> !qco.qubit
//     %q2  = qco.ry(%c) %q1  : !qco.qubit -> !qco.qubit
//     %q3  = qco.rz(%c) %q2  : !qco.qubit -> !qco.qubit
//     %q4  = qco.rx(%c) %q3  : !qco.qubit -> !qco.qubit
//     %q5  = qco.ry(%c) %q4  : !qco.qubit -> !qco.qubit
//     %q6  = qco.rz(%c) %q5  : !qco.qubit -> !qco.qubit
//     %q7  = qco.rx(%c) %q6  : !qco.qubit -> !qco.qubit
//     %q8  = qco.ry(%c) %q7  : !qco.qubit -> !qco.qubit
//     %q9  = qco.rz(%c) %q8  : !qco.qubit -> !qco.qubit
//     %q10 = qco.rx(%c) %q9  : !qco.qubit -> !qco.qubit
//     %q11 = qco.ry(%c) %q10 : !qco.qubit -> !qco.qubit
//     %q12 = qco.rz(%c) %q11 : !qco.qubit -> !qco.qubit
//     %q13 = qco.rx(%c) %q12 : !qco.qubit -> !qco.qubit
//     %q14 = qco.ry(%c) %q13 : !qco.qubit -> !qco.qubit
//     %q15 = qco.rz(%c) %q14 : !qco.qubit -> !qco.qubit
//     %q16 = qco.rx(%c) %q15 : !qco.qubit -> !qco.qubit
//     %q17 = qco.ry(%c) %q16 : !qco.qubit -> !qco.qubit
//     %q18 = qco.rz(%c) %q17 : !qco.qubit -> !qco.qubit
//     qco.dealloc %q18 : !qco.qubit
//     return
//   }
// }
//
// // -----
// // Test: chain with a P gate (non-SU(2)) -> should emit GPhaseOp
// module {
//   func.func @testChainWithPhase() {
//     %q0 = qco.alloc : !qco.qubit
//     %c = arith.constant 1.000000e+00 : f64
//     %q1 = qco.rx(%c) %q0 : !qco.qubit -> !qco.qubit
//     %q2 = qco.p(%c) %q1 : !qco.qubit -> !qco.qubit
//     qco.dealloc %q2 : !qco.qubit
//     return
//   }
// }
//
// // -----
// // Test: same-type single-param (RX + RX) -> should NOT be merged by this pass
// module {
//   func.func @testSameTypeNoMerge() {
//     %q0 = qco.alloc : !qco.qubit
//     %c = arith.constant 1.000000e+00 : f64
//     %q1 = qco.rx(%c) %q0 : !qco.qubit -> !qco.qubit
//     %q2 = qco.rx(%c) %q1 : !qco.qubit -> !qco.qubit
//     qco.dealloc %q2 : !qco.qubit
//     return
//   }
// }

module {
    func.func @testConstantBetweenChainOps() {
      %q0 = qco.alloc : !qco.qubit
      %c1 = arith.constant 1.000000e+00 : f64
      %q1 = qco.rx(%c1) %q0 : !qco.qubit -> !qco.qubit
      %c2 = arith.constant 2.000000e+00 : f64
      %q2 = qco.ry(%c2) %q1 : !qco.qubit -> !qco.qubit
      qco.dealloc %q2 : !qco.qubit
      return
    }
  }
