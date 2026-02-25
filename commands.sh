diff <(grep -oP "MQT_NAMED_BUILDER\(\K[^)]*" ./mlir/unittests/Dialect/QCO/IR/test_qco_ir.cpp | grep -i "pow")
diff <(grep -oP "MQT_NAMED_BUILDER\(\K[^)]*" ./mlir/unittests/Dialect/QCO/IR/test_qco_ir.cpp | grep -i "pow") ./qc_programs.cpp.functions
grep -oP 'void \K\w+(?=\(QCOProgramBuilder)' /home/anatol/git/core/pow-modifier/mlir/unittests/programs/qco_programs.h | grep pow
diff <(grep -oP "MQT_NAMED_BUILDER\(\K[^)]*" ./mlir/unittests/Dialect/QCO/IR/test_qco_ir.cpp | grep -i "pow") <(grep -oP 'void \K\w+(?=\(QCOProgramBuilder)' /home/anatol/git/core/pow-modifier/mlir/unittests/programs/qco_programs.h | grep -i "pow")
diff <(grep -oP "MQT_NAMED_BUILDER\(\K[^)]*" ./mlir/unittests/Dialect/QCO/IR/test_qco_ir.cpp | grep -i "pow") <(grep -oP "MQT_NAMED_BUILDER\(\K[^)]*" ./mlir/unittests/Dialect/QC/IR/test_qc_ir.cpp | grep -i "pow")
