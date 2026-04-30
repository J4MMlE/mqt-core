# Plan: Direct Bidirectional QASM3 ↔ MLIR Translation

## Context

The current QASM3↔MLIR path goes through `qc::QuantumComputation` (the legacy core IR):
- Import: `QASM3 → qasm3::Importer → QuantumComputation → translateQuantumComputationToQC() → QC dialect`
- Export: **does not exist**

`QuantumComputation` is no longer actively developed and cannot represent advanced QASM3 features. This plan implements direct translation between QASM3 text and the QC MLIR dialect, bypassing `QuantumComputation` entirely.

**Target dialect: QC** — its reference semantics (mutable qubits, in-place gate application, ctrl/inv modifier regions) map naturally to QASM3's programming model. Classical constructs use standard MLIR dialects (`arith`, `scf`, `func`).

**Parser: Keep and extend MQT's existing hand-written parser** (`src/qasm3/`).

**Reference implementation:** qe-compiler (`~/git/qe-compiler`) — IBM's MLIR-based quantum compiler with full QASM3 support. Uses a similar architecture (external parser → visitor → MLIR ops via OpBuilder). Their `QUIRGenQASM3Visitor` (~2500 lines) is a useful reference for the translation patterns.

---

## Scope

### Tier 1 — Full support (this implementation)
Qubit/bit declarations, all standard gates with ctrl/inv/negctrl modifiers, measure, reset, barrier, if/else, compound gates (user-defined), broadcasting, initial layout / output permutation.

### Tier 2 basics — Included in this implementation
- Classical types: `int`, `float`, `bool` declarations and basic arithmetic
- Control flow: `for` loops, `while` loops
- Subroutines: `def` functions with parameters and return values
- These map to standard MLIR dialects: `arith` (arithmetic), `scf` (for/while/if), `func` (subroutines)

### Deferred (future work)
- `angle`, `complex`, `duration`, `stretch` types
- `switch`/`case`, `break`/`continue`
- Arrays, aliases (`let`)
- Casting expressions
- `pow` modifier
- Timing (`delay`, `box`, `durationof`)
- Calibration (`cal`, `defcal`, `defcalgrammar`)
- `input`/`output` declarations, pragmas, annotations
- `extern` functions

---

## Part A: Parser & AST Extensions

The existing parser (`src/qasm3/`) handles Tier 1 but needs extensions for Tier 2 basics. The AST currently has no `ForStatement`, `WhileStatement`, or `DefStatement` nodes.

### New AST nodes needed (in `Statement.hpp`)

```cpp
class ForStatement final : public Statement { ... };
  // scalarType, identifier, iterable (range or set), body statements

class WhileStatement final : public Statement { ... };
  // condition expression, body statements

class DefStatement final : public Statement { ... };
  // name, parameters (typed), return type, body statements

class ReturnStatement final : public Statement { ... };
  // optional expression

class ClassicalDeclarationStatement final : public Statement { ... };
  // scalar type (int/float/bool + width), identifier, optional initializer
  // (or extend existing DeclarationStatement to handle classical types)
```

### New InstVisitor methods

```cpp
virtual void visitForStatement(std::shared_ptr<ForStatement>) = 0;
virtual void visitWhileStatement(std::shared_ptr<WhileStatement>) = 0;
virtual void visitDefStatement(std::shared_ptr<DefStatement>) = 0;
virtual void visitReturnStatement(std::shared_ptr<ReturnStatement>) = 0;
```

These also need default implementations in `DefaultInstVisitor`.

### Parser extensions needed (in `Parser.cpp`)

- Parse `for` statement: `FOR scalarType Identifier IN (range|set|expr) body`
- Parse `while` statement: `WHILE LPAREN expr RPAREN body`
- Parse `def` statement: `DEF Identifier LPAREN argList RPAREN returnSig? scope`
- Parse `return` statement: `RETURN expr? SEMICOLON`
- Parse classical declarations: `int[32] x = 5;`, `float[64] y = 3.14;`, `bool flag = true;`
- Extend expression parser for full arithmetic (currently only const-evaluates; need runtime expressions)

### Scanner extensions (in `Scanner.cpp`)

New keywords: `for`, `while`, `in`, `def`, `return`, `int`, `uint`, `float`, `bool` (some may already exist as tokens).

### Files to modify
- `include/mqt-core/qasm3/Statement.hpp` — new AST nodes
- `include/mqt-core/qasm3/Statement_fwd.hpp` — forward declarations
- `include/mqt-core/qasm3/InstVisitor.hpp` — new visitor methods
- `src/qasm3/Parser.cpp` — parse new constructs
- `src/qasm3/Scanner.cpp` — new token kinds (if needed)
- `include/mqt-core/qasm3/Token.hpp` — new token kinds (if needed)
- `src/qasm3/Statement.cpp` — accept() for new nodes

**Important:** These AST/parser extensions must not break the existing `qasm3::Importer` targeting `QuantumComputation`. The existing importer can add no-op implementations for the new visitor methods (via `DefaultInstVisitor`).

---

## Part B: QASM3 → QC Direct Importer

### New files
- `mlir/include/mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h`
- `mlir/lib/Dialect/QC/Translation/TranslateQASM3ToQC.cpp`

### Public API
```cpp
namespace mlir::qc {
OwningOpRef<ModuleOp> translateQASM3ToQC(MLIRContext *context,
                                          const std::string &filename);
OwningOpRef<ModuleOp> translateQASM3ToQC(MLIRContext *context,
                                          std::istream &input);
}
```

### Class design

`MLIRQasmImporter` implements `qasm3::InstVisitor` and uses `QCProgramBuilder` + raw `mlir::OpBuilder` for classical constructs.

**State:**
- `QCProgramBuilder builder` — constructs QC ops (gates, measure, reset, barrier, alloc)
- `mlir::OpBuilder &rawBuilder` — for arith/scf/func ops (classical)
- `NestedEnvironment<DeclarationStatement> declarations` — symbol table
- `llvm::StringMap<llvm::SmallVector<mlir::Value>> qubitRegisters` — qubit name → values
- `llvm::StringMap<QCProgramBuilder::ClassicalRegister> classicalRegisters` — classical reg name → values
- `llvm::DenseMap<llvm::StringRef, mlir::Value> classicalVariables` — classical var name → SSA value
- `std::unordered_map<std::string, std::shared_ptr<Gate>> gates` — gate library
- `ConstEvalPass constEvalPass`, `TypeCheckPass typeCheckPass`

### Visitor method implementations

**Tier 1 (quantum core):**

| Visitor method | MLIR output |
|---|---|
| `visitDeclarationStatement` (qubit) | `builder.allocQubitRegister(n, name)` |
| `visitDeclarationStatement` (bit) | `builder.allocClassicalBitRegister(n, name)` |
| `visitGateCallStatement` → standard | `builder.h(target)`, `builder.rz(param, target)`, etc. |
| `visitGateCallStatement` → controlled | `builder.ctrl(controls, [&]{ builder.h(target); })` |
| `visitGateCallStatement` → inverse | `builder.inv([&]{ builder.h(target); })` |
| `visitGateCallStatement` → negctrl | X-bracket: `builder.x(ctrl); builder.ctrl(ctrl, ...); builder.x(ctrl)` |
| `visitGateCallStatement` → compound | Recursive visit with scoped qubit/param bindings |
| `visitBarrierStatement` | `builder.barrier(targets)` |
| `visitResetStatement` | `builder.reset(target)` |
| `visitAssignmentStatement` (measure) | `builder.measure(qubit, bit)` |
| `visitIfStatement` | `scf::IfOp` with condition from `arith::CmpIOp` |
| `visitGateStatement` | Store CompoundGate in gates map (no MLIR emission) |
| `visitInitialLayout` | Module attribute `qc.initial_layout` |
| `visitOutputPermutation` | Module attribute `qc.output_permutation` |

**Tier 2 basics (classical):**

| Visitor method | MLIR output |
|---|---|
| `visitClassicalDeclaration` (int) | `arith::ConstantIntOp` or variable allocation |
| `visitClassicalDeclaration` (float) | `arith::ConstantFloatOp` or variable allocation |
| `visitClassicalDeclaration` (bool) | `arith::ConstantIntOp` (i1) |
| `visitForStatement` | `scf::ForOp` (convert inclusive range → half-open) |
| `visitWhileStatement` | `scf::WhileOp` |
| `visitDefStatement` | `func::FuncOp` with typed parameters |
| `visitReturnStatement` | `func::ReturnOp` |
| Function calls | `func::CallOp` |
| Arithmetic expressions | `arith::AddIOp`, `arith::MulIOp`, `arith::CmpIOp`, etc. |
| Assignments | Update SSA value in `classicalVariables` map |

**Gate dispatch:** `llvm::StringSwitch` or dispatch table mapping gate names from `StdGates.hpp` to `QCProgramBuilder` methods.

**Expression evaluation:** For Tier 2, expressions can't always be const-evaluated — they may involve runtime variables. The importer needs an expression visitor that produces `mlir::Value` (via `arith` ops) rather than `double`. `ConstEvalPass` is still used for compile-time constants; runtime expressions use `arith` ops.

---

## Part C: QC → QASM3 Direct Exporter

### New files
- `mlir/include/mlir/Dialect/QC/Translation/TranslateQCToQASM3.h`
- `mlir/lib/Dialect/QC/Translation/TranslateQCToQASM3.cpp`

### Public API
```cpp
namespace mlir::qc {
LogicalResult translateQCToQASM3(ModuleOp module, llvm::raw_ostream &os);
}
```

### Algorithm

1. **Preamble:** Emit `OPENQASM 3.0;` and `include "stdgates.inc";`

2. **Register discovery:** Walk `qc.alloc` ops, group by `register_name` attribute → emit `qubit[N] name;`. Build `DenseMap<Value, std::string>` mapping each qubit Value to its QASM3 name (e.g. `q[0]`). Allocs without metadata get synthetic names.

3. **Classical register discovery:** Walk `qc.measure` ops, group by `register_name` → emit `bit[N] name;`. Map measure result values to `c[i]` names.

4. **Classical variable discovery:** Walk `arith.constant` and variable-related ops → emit classical declarations.

5. **Operation emission:** Walk function body sequentially. For each op:
   - **Gate ops** (UnitaryOpInterface): `getBaseSymbol()` → QASM3 gate name. Format params, look up qubit names. Emit `gatename(params) targets;`
   - **CtrlOp:** Recursively unwrap — accumulate `ctrl(N) @ ` prefix, collect control qubit names, recurse into body
   - **InvOp:** Accumulate `inv @ ` prefix, recurse into body
   - **MeasureOp:** Emit `c[i] = measure q[j];`
   - **ResetOp:** Emit `reset q[j];`
   - **BarrierOp:** Emit `barrier q[0], q[1], ...;`
   - **GPhaseOp:** Emit `gphase(theta);`
   - **scf.for:** Emit `for int i in [start:step:end] { ... }`
   - **scf.while:** Emit `while (cond) { ... }`
   - **scf.if:** Emit `if (cond) { ... } else { ... }`
   - **func.func:** Emit `def name(params) -> rettype { ... }`
   - **func.call:** Emit `name(args)`
   - **arith ops:** Emit infix expressions (`a + b`, `a * b`, etc.)

6. **Layout metadata:** If module attributes exist, emit as `// i ...` / `// o ...` comments.

**Ctrl/Inv unwinding helper (recursive):**
```
emitOperation(op, prefix, controls):
  if op is CtrlOp → append "ctrl(N) @ ", collect controls, recurse on body
  if op is InvOp → append "inv @ ", recurse on body
  else → emit "{prefix}{gateName}({params}) {controls}, {targets};"
```

**Parameter formatting:** Trace `Value` back to `arith.constant` to extract double. Emit with full precision. Non-constant → diagnostic error.

---

## Part D: Wire into mqt-cc

### Modify: `mlir/tools/mqt-cc/mqt-cc.cpp`

Add flags:
```cpp
static cl::opt<bool> emitQASM3("emit-qasm3",
    cl::desc("Emit QASM3 output instead of MLIR"), cl::init(false));
static cl::opt<bool> directImport("direct-import",
    cl::desc("Use direct QASM3->QC import (bypass QuantumComputation)"),
    cl::init(false));
```

Update `loadQASMFile()` to offer both paths. Update output section to call `translateQCToQASM3()` when `--emit-qasm3` is set.

---

## Part E: Build System

### Modify: `mlir/lib/Dialect/QC/Translation/CMakeLists.txt`

```cmake
add_mlir_library(
  MLIRQCTranslation
  TranslateQuantumComputationToQC.cpp
  TranslateQASM3ToQC.cpp        # NEW
  TranslateQCToQASM3.cpp         # NEW
  LINK_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRQCDialect
  MLIRQCProgramBuilder
  MQT::CoreIR
  MQT::CoreQASM)                 # NEW — for Scanner/Parser/AST
```

### Modify: `mlir/unittests/Dialect/QC/Translation/CMakeLists.txt`
Add new test files.

---

## Part F: Tests

### New files
- `mlir/unittests/Dialect/QC/Translation/test_qasm3_to_qc.cpp`
- `mlir/unittests/Dialect/QC/Translation/test_qc_to_qasm3.cpp`
- `mlir/unittests/Dialect/QC/Translation/test_qasm3_roundtrip.cpp`

### Test cases

**Tier 1 tests:**
- All individual standard gates (X, H, RZ, SWAP, etc.)
- Controlled gates (CX, CCX, multi-controlled)
- Modifiers (ctrl @, inv @, nested ctrl @ inv @)
- Negctrl (via X-bracket)
- Compound gates (user-defined)
- Broadcasting (register-width gate calls)
- Measure, reset, barrier
- If/else
- Multi-register circuits
- Bell state, GHZ, QFT end-to-end

**Tier 2 tests:**
- Classical variable declarations (int, float, bool)
- Arithmetic expressions in gate parameters
- For loops with qubit operations
- While loops
- Subroutine definitions and calls
- Mixed classical/quantum circuits

**Round-trip tests:**
- QASM3 → import → export → re-import → compare module structure
- QCProgramBuilder → export → import → compare
- Verify register metadata survives round-trip
- Verify parameter precision maintained

---

## Implementation Order

```
1. Parser/AST extensions        — for/while/def/return + classical decl AST nodes
                                  (extend parser, keep existing Importer working)
2. Exporter (QC → QASM3)       — smaller scope, testable with existing import path
3. Export tests                  — validate with QCProgramBuilder-built modules
4. Importer (QASM3 → QC)       — Tier 1 first, then Tier 2 visitor methods
5. Import tests                  — validate against reference modules
6. Round-trip tests              — both directions
7. mqt-cc wiring                — --emit-qasm3 and --direct-import flags
```

Starting with the exporter gives us a working QASM3 output quickly (using existing import path for input), then we build the importer and get full round-trip validation.

---

## Key Reference Files

| File | Role |
|---|---|
| `src/qasm3/Importer.cpp` (965 lines) | Blueprint for MLIRQasmImporter — port backend |
| `mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h` (948 lines) | Target builder API for importer |
| `mlir/lib/Dialect/QC/Translation/TranslateQuantumComputationToQC.cpp` (646 lines) | Shows QCProgramBuilder usage patterns |
| `mlir/include/mlir/Dialect/QC/IR/QCOps.td` | All QC ops; `getBaseSymbol()` for exporter |
| `include/mqt-core/qasm3/StdGates.hpp` | 40+ gate name mappings with aliases |
| `include/mqt-core/qasm3/InstVisitor.hpp` | Visitor interface (10 methods, will grow) |
| `include/mqt-core/qasm3/Statement.hpp` | AST nodes (needs extensions) |
| `src/qasm3/Parser.cpp` | Parser (needs extensions) |
| `mlir/tools/mqt-cc/mqt-cc.cpp` | Compiler driver to wire into |
| `~/git/qe-compiler/lib/Frontend/OpenQASM3/QUIRGenQASM3Visitor.cpp` | Reference: full QASM3→MLIR visitor (~2500 lines) |
