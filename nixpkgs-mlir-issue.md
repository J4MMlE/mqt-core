`MLIRConfig.cmake`: `mlir-tblgen` imported target missing due to package splitting

When using the nix-packaged MLIR in a CMake project via `find_package(MLIR)`, any call to `mlir_tablegen()`, `add_mlir_dialect()`, or `add_mlir_interface()` fails at build time with errors like:

```
ninja: error: '_deps/jeff-mlir-build/include/jeff/IR/mlir-tblgen', needed by
'_deps/jeff-mlir-build/include/jeff/IR/JeffInterfaces.h.inc', missing and no known rule to make it
```

The build system looks for `mlir-tblgen` at the wrong location because it resolves the bare name relative to the working directory instead of using the actual binary path.

The installed `MLIRConfig.cmake` sets:

```cmake
set(MLIR_TABLEGEN_EXE "mlir-tblgen")
```

This is intended to be a CMake imported executable target (it's listed in `MLIR_EXPORTED_TARGETS` on line 10 of the same file), which CMake would resolve to the actual binary path when used in `add_custom_command(COMMAND ... DEPENDS ...)`. However, the `mlir-tblgen` target is **not defined** in `MLIRTargets.cmake` or `MLIRTargets-release.cmake` because it lives in the separate `llvmPackages_22.tblgen` package.

In a non-split LLVM install, the imported target definition for `mlir-tblgen` would be present in `MLIRTargets.cmake` with a valid `IMPORTED_LOCATION`, and everything resolves correctly.

**Affected nixpkgs package:** `mlir` (from `pkgs/development/compilers/llvm/common/mlir/default.nix`)

**Affected version:** `22.1.0-rc3`

**Root cause:** The upstream template `mlir/cmake/modules/MLIRConfig.cmake.in` substitutes `@MLIR_TABLEGEN_EXE@` as a target name, but the nix output splitting puts the actual `mlir-tblgen` binary into a separate derivation (`llvmPackages_22.tblgen`) without carrying over the imported target definition into `mlir-dev/lib/cmake/mlir/MLIRTargets.cmake`.

**Relevant files:**

- `mlir-dev/lib/cmake/mlir/MLIRConfig.cmake:10,13` — declares `mlir-tblgen` as exported target and sets `MLIR_TABLEGEN_EXE`
- `mlir-dev/lib/cmake/mlir/MLIRTargets.cmake` — missing `mlir-tblgen` target definition
- `mlir-dev/lib/cmake/mlir/MLIRTargets-release.cmake` — also missing
- `llvmPackages_22.tblgen/bin/mlir-tblgen` — where the actual binary lives
