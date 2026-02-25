# Variables
build_base := "build"
parallel := "8"
build_type := "Debug"

# Default recipe
default: build

# Configure with optional build type and coverage
# Usage: just configure
#        just configure Debug
#        just configure Debug true
configure config=build_type coverage="false":
    cmake \
        -G Ninja \
        -S . \
        -B {{build_base}}/{{config}} \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DBUILD_MQT_CORE_TESTS=ON \
        -DBUILD_MQT_CORE_MLIR=ON \
        -DLLVM_EXTERNAL_LIT=$(which lit) \
        -DENABLE_COVERAGE={{coverage}} \
        -DCMAKE_BUILD_TYPE={{config}}
    # ln -sf {{build_base}}/{{config}}/compile_commands.json compile_commands.json

# Generic build helper
[private]
_build target config=build_type:
    cmake --build {{build_base}}/{{config}} --target {{target}} --config {{config}} --parallel {{parallel}}

# Build targets
build: (configure build_type) (_build "mqt-cc" build_type)
mqt-cc: (configure build_type) (_build "mqt-cc" build_type)
mlir-doc: (configure build_type) (_build "mlir-doc" build_type)

# Testing
test config="Debug": (configure config)
    cmake --build {{build_base}}/{{config}} --config {{config}} --parallel {{parallel}}
    ctest --test-dir {{build_base}}/{{config}} -C {{config}} -j {{parallel}} --output-on-failure

test-qco config="Debug": (configure config) (_build "mqt-core-mlir-unittest-optimizations" config)
    {{build_base}}/{{config}}/mlir/unittests/Dialect/QCO/Transforms/Optimizations/mqt-core-mlir-unittest-optimizations

# Coverage
clean-coverage config="Debug":
    find {{build_base}}/{{config}} -name "*.gcda" -delete 2>/dev/null || true

coverage: (configure "Debug" "true") (clean-coverage "Debug") (_build "mqt-core-mlir-unittests" "Debug")
    -ctest --test-dir {{build_base}}/Debug -C Debug -j {{parallel}} --output-on-failure
    gcovr --root . \
        --exclude '.*/test/.*' \
        --exclude './test/.*' \
        --exclude '.*/_deps/.*' \
        --gcov-executable "llvm-cov gcov" \
        --html --html-details \
        -o {{build_base}}/Debug/coverage.html \
        {{build_base}}/Debug/
    @echo "Coverage report: {{build_base}}/Debug/coverage.html"

# Run quantum-opt (uses Release by default)
run file="test3.mlir" config=build_type:
    {{build_base}}/{{config}}/mlir/tools/quantum-opt/quantum-opt \
        --merge-rotation-gates {{file}}

run-quat file="test.mlir" config=build_type:
    {{build_base}}/{{config}}/mlir/tools/quantum-opt/quantum-opt \
        --pass-pipeline='builtin.module(merge-rotation-gates{quaternion-folding})' {{file}}

run-test config=build_type:
    {{build_base}}/{{config}}/mlir/tools/quantum-opt/quantum-opt \
        --pass-pipeline='builtin.module(merge-rotation-gates{quaternion-folding})' \
        mlir/test/Dialect/MQTOpt/Transforms/quantum-merge-rotation-gates.mlir

# mlir/lib/Compiler/CompilerPipeline.cpp \
# mlir/tools/mqt-cc/mqt-cc.cpp \
# mlir/unittests/Compiler/test_compiler_pipeline.cpp \
# mlir/lib/Dialect/QCO/Transforms/Optimizations/QuaternionMergeRotationGates.cpp
lint config=build_type:
    clang-tidy -p {{build_base}}/{{config}} --extra-arg=-std=c++20 \
        --header-filter='mlir/(lib|tools|unittests)/.*' \
        mlir/unittests/Dialect/QCO/Transforms/Optimizations/test_qco_quaternion_merge.cpp


# Cleanup
clean:
    rm -rf {{build_base}}
