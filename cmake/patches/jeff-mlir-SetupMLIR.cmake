find_package(MLIR REQUIRED CONFIG)

# Workaround for nixpkgs where MLIR_TABLEGEN_EXE is set to a bare name
# instead of an absolute path due to package splitting.
# See: https://github.com/NixOS/nixpkgs/issues/XXXXX
if(NOT IS_ABSOLUTE "${MLIR_TABLEGEN_EXE}")
  unset(MLIR_TABLEGEN_EXE)
  unset(MLIR_TABLEGEN_EXE CACHE)
  find_program(MLIR_TABLEGEN_EXE mlir-tblgen REQUIRED)
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
