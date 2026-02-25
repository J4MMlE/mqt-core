// module {
//   func.func @testPow() {
//     %q0_0 = qco.alloc : !qco.qubit
//
//     %q0_1 = qco.inv (%a = %q0_0){
//       %a_1 = qco.pow (2.000000e+00) (%b = %a) {
//         %c_0 = arith.constant 0.123 : f64
//         %b_1 = qco.rx(%c_0) %b : !qco.qubit -> !qco.qubit
//         qco.yield %b_1
//       } : {!qco.qubit} -> {!qco.qubit}
//       qco.yield %a_1
//     } : {!qco.qubit} -> {!qco.qubit}
//     qco.sink %q0_1 : !qco.qubit
//     return
//   }
// }

// module {
//     func.func @testPow() {
//       %q0_0 = qco.alloc : !qco.qubit
//
//       %q0_1 = qco.inv (%a = %q0_0){
//         %a_1 = qco.pow (2.000000e+00) (%b = %a) {
//           %c_0 = arith.constant 0.123 : f64
//           %b_1 = qco.rx(%c_0) %b : !qco.qubit -> !qco.qubit
//           qco.yield %b_1
//         } : {!qco.qubit} -> {!qco.qubit}
//         qco.yield %a_1
//       } : {!qco.qubit} -> {!qco.qubit}
//       qco.sink %q0_1 : !qco.qubit
//       return
//     }
//   }

module {
    func.func @testPow() {
      %q0_0 = qco.alloc : !qco.qubit

      %q0_1 = qco.inv (%a = %q0_0){
        %a_1 = qco.pow (2.000000e+00) (%b = %a) {
          %c_0 = arith.constant 0.123 : f64
          %b_1 = qco.rx(%c_0) %b : !qco.qubit -> !qco.qubit
          qco.yield %b_1
        } : {!qco.qubit} -> {!qco.qubit}
        qco.yield %a_1
      } : {!qco.qubit} -> {!qco.qubit}
      qco.sink %q0_1 : !qco.qubit
      return
    }
  }

// module {
//     func.func @bad_case() {
//       %ctrl = qco.alloc : !qco.qubit
//       %tgt  = qco.alloc : !qco.qubit
//
//       %ctrl_1, %tgt_1 = qco.ctrl(%ctrl) targets(%t =
//   %tgt) {
//         %t_1 = qco.pow (1.5e+00) (%a = %t) {
//           %a_1 = qco.sx %a : !qco.qubit -> !qco.qubit
//           qco.yield %a_1
//         } : {!qco.qubit} -> {!qco.qubit}
//         qco.yield %t_1
//       } : ({!qco.qubit}, {!qco.qubit}) ->
//   ({!qco.qubit}, {!qco.qubit})
//
//       qco.sink %ctrl_1 : !qco.qubit
//       qco.sink %tgt_1  : !qco.qubit
//       return
//     }
//   }
//
