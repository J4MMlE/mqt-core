# MQT Core — development task runner
#
# Quick start:
#   1. Install prerequisites: cmake, clang, ninja, lit, LLVM 22.1+ (for MLIR)
#   2. Run `just build` to configure and build the compiler
#   3. Run `just test` to run the full test suite
#
# Common recipe arguments (passed positionally):
#   config   — CMake build type: "Debug" (default) or "Release". Example: `just test Release`
#   coverage — Enable coverage instrumentation: "true" or "false" (default). Only used by `configure`.
#   file     — Path to an MLIR file for quantum-opt. Example: `just run my_circuit.mlir`
#
# Run `just --list` to see all available recipes.

# Output directory for cmake builds (each config gets a subdirectory, e.g. build/Debug)
build_dir := "build"
# Number of parallel build and test jobs (passed to cmake --jobs and ctest -j)
jobs := "8"
# Default CMake build type when not specified per-recipe ("Debug" or "Release")
default_config := "Debug"

default:
    @just --list

# Configure cmake (auto-detects clang, ninja, lit)
# Debug builds enable tests, compile_commands.json, and IDE symlink.
# Release builds produce a minimal compiler without dev tooling.
configure config=default_config coverage="false" dir=config:
    cmake \
        -G Ninja \
        -S . \
        -B {{build_dir}}/{{dir}} \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
        -DCMAKE_BUILD_TYPE={{config}} \
        -DBUILD_MQT_CORE_MLIR=ON \
        -DLLVM_EXTERNAL_LIT=$(which lit) \
        -DGIT_EXECUTABLE=$(which git) \
        {{ if config == "Debug" { \
            "-DBUILD_MQT_CORE_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" \
        } else { \
            "-DBUILD_MQT_CORE_TESTS=OFF" \
        } }} \
        {{ if coverage == "true" { \
            "-DENABLE_COVERAGE=ON -DBUILD_MQT_CORE_TESTS=ON" \
        } else { \
            "-DENABLE_COVERAGE=OFF" \
        } }}
    {{ if config == "Debug" { \
        "ln -sf " + build_dir + "/" + dir + "/compile_commands.json compile_commands.json" \
    } else { \
        "" \
    } }}

# Build the mqt-cc compiler
build: (_ensure_configured default_config) (_build "mqt-cc")

# Alias for build
mqt-cc: (_ensure_configured default_config) (_build "mqt-cc")

# Generate MLIR dialect documentation
mlir-doc: (_ensure_configured default_config) (_build "mlir-doc")

# Run the full test suite (configure if needed + build all + ctest)
test config=default_config: (_ensure_configured config) (_build "all" config)
    ctest --test-dir {{build_dir}}/{{config}} -C {{config}} -j {{jobs}} --output-on-failure

# Run only the QCO quaternion merge unit tests (faster iteration)
test-qco config=default_config: (_ensure_configured config) (_build "mqt-core-mlir-unittest-optimizations" config)
    {{build_dir}}/{{config}}/mlir/unittests/Dialect/QCO/Transforms/Optimizations/mqt-core-mlir-unittest-optimizations

# Delete coverage data (.gcda files) from a previous run
clean-coverage config="Debug":
    find {{build_dir}}/{{config}} -name "*.gcda" -delete 2>/dev/null || true

# Generate an HTML coverage report at build/Coverage/coverage.html (requires gcovr and llvm-cov)
coverage: (configure "Debug" "true" "Coverage") (clean-coverage "Coverage") (_build "all" "Debug" "Coverage")
    # Ignore test failures so gcovr still generates the report
    -ctest --test-dir {{build_dir}}/Coverage -C Debug -j {{jobs}} --output-on-failure
    gcovr --root . \
        --exclude '.*/test/.*' \
        --exclude './test/.*' \
        --exclude '.*/_deps/.*' \
        --gcov-executable "llvm-cov gcov" \
        --html --html-details \
        -o {{build_dir}}/Coverage/coverage.html \
        {{build_dir}}/Coverage/
    @echo "Coverage report: {{build_dir}}/Coverage/coverage.html"

# Run quantum-opt with the merge-rotation-gates pass on an MLIR file
run file="test3.mlir" config=default_config:
    {{build_dir}}/{{config}}/mlir/tools/quantum-opt/quantum-opt \
        --merge-rotation-gates {{file}}

# Run quantum-opt with quaternion folding enabled
run-quat file="test.mlir" config=default_config:
    {{build_dir}}/{{config}}/mlir/tools/quantum-opt/quantum-opt \
        --pass-pipeline='builtin.module(merge-rotation-gates{quaternion-folding})' {{file}}

# Run quaternion folding on the in-tree test file
run-test config=default_config:
    {{build_dir}}/{{config}}/mlir/tools/quantum-opt/quantum-opt \
        --pass-pipeline='builtin.module(merge-rotation-gates{quaternion-folding})' \
        mlir/test/Dialect/MQTOpt/Transforms/quantum-merge-rotation-gates.mlir

# Check formatting of MLIR sources (uses .clang-format config)
# Example: `just format mlir/lib/Dialect/QCO/IR/QCOOps.cpp mlir/lib/Dialect/QC/IR/QCOps.cpp`
format *files:
    clang-format --dry-run --Werror \
        {{ if files == "" { "$(find mlir/lib mlir/tools mlir/unittests -name '*.cpp' -o -name '*.h')" } else { files } }}

# Fix formatting of MLIR sources in-place
format-fix *files:
    clang-format -i \
        {{ if files == "" { "$(find mlir/lib mlir/tools mlir/unittests -name '*.cpp' -o -name '*.h')" } else { files } }}

# Run clang-tidy on MLIR sources (requires a configured Debug build)
# Example: `just lint Debug mlir/lib/Dialect/QCO/IR/QCOOps.cpp mlir/lib/Compiler/CompilerPipeline.cpp`
lint config=default_config *files:
    clang-tidy -p {{build_dir}}/{{config}} --extra-arg=-std=c++20 \
        --header-filter='mlir/(lib|tools|unittests)/.*' \
        {{ if files == "" { "$(find mlir/lib mlir/tools mlir/unittests -name '*.cpp')" } else { files } }}

# Remove all build artifacts
clean:
    rm -rf {{build_dir}}

# --- Internal helpers ---

# Only run configure if the build directory doesn't exist yet
[private]
_ensure_configured config=default_config:
    {{ if path_exists(build_dir / config) != "true" { \
        "just configure " + config \
    } else { \
        "" \
    } }}

[private]
_build target config=default_config dir=config:
    cmake --build {{build_dir}}/{{dir}} --target {{target}} --config {{config}} --parallel {{jobs}}
