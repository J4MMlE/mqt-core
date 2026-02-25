# CLAUDE.md — Project Instructions for Claude Code

## Workflow
When asked about refactoring or design changes, present a 3-5 bullet proposed approach first. Do NOT edit any files until explicitly confirmed. Do not default to full rewrites — a hybrid approach preserving existing patterns may be preferred.

## Communication Style
When asked for a recommendation, commit to a clear position. Do not flip-flop between opposing suggestions across messages. If there are genuine tradeoffs, state them once and give a final recommendation.

## Project
MQT Core — Munich Quantum Toolkit core library.
Current branch: `quaternion-rotation-merging` — refactoring the quaternion merge MLIR pass.
Git layout: bare repo with worktrees (bare at `/home/anatol/git/core`, this worktree at `core/quaternion-rotation-merging`).

## Build & Test (justfile — user runs these, not Claude)
- `just configure` — cmake + Ninja, clang, compile_commands, MLIR enabled
- `just build` / `just mqt-cc` — build the compiler
- `just test` — full test suite
- `just test-qco` — build + run quaternion merge unit tests only
- `just run-quat file=test.mlir` — run quaternion folding pass via quantum-opt
- `just coverage` — gcovr HTML coverage report
- Build dir: `build/Debug`, uses Ninja with `-j8`
- Lint (MUST pass): `uvx nox -s lint`
- LLVM 22.1+ required for MLIR

## Key Files
- `mlir/lib/Dialect/QCO/Transforms/Optimizations/QuaternionMergeRotationGates.cpp` — main file being refactored (only file in Optimizations/)
- `mlir/unittests/Dialect/QCO/Transforms/Optimizations/test_qco_quaternion_merge.cpp` — unit tests for the pass
- `mlir/include/mlir/Dialect/QCO/Transforms/Passes.td` — pass tablegen definitions
- `mlir/include/mlir/Dialect/QCO/Transforms/Passes.h` — pass header
- `mlir/include/mlir/Dialect/QCO/IR/QCOOps.td` — QCO op definitions
- `mlir/include/mlir/Dialect/QCO/IR/QCOInterfaces.td` — QCO interfaces (UnitaryOpInterface etc.)
- `PLAN.md` — detailed refactoring plan (chain-based merging with global phase tracking)
- `AGENTS.md` — full conventions reference (template-managed, do NOT edit)

## MLIR Dialect Architecture
- **QC** — quantum computation (high-level gates: GPhaseOp, POp, standard gates)
- **QCO** — quantum circuit optimization (IR, transforms, builder, utils)
- **QIR** — quantum IR (low-level target)
- **Jeff** — intermediate dialect with bidirectional QCO conversions
- Conversion paths: QC↔QCO, QCO↔Jeff, QC→QIR

## QCO Transforms
- `Optimizations/QuaternionMergeRotationGates.cpp` — quaternion merge pass
- `Mapping/` — Architecture.cpp, Mapping.cpp (qubit mapping)

## Conventions
- C++20; prefer LLVM data structures in `mlir/` (`llvm::SmallVector`, `llvm::function_ref`, etc.)
- Doxygen-style comments, `#pragma once` for header guards
- Always add/update tests for every code change
- Commit footer: `Assisted-by: Claude Opus 4.6 via Claude Code`
- Update `CHANGELOG.md` / `UPGRADING.md` for user-facing changes
- Never modify template-generated files (check file header)

## Workflow Preferences
- Prefer parallel sub-agents (Task tool) for independent work: multi-file research, concurrent build+lint, exploring separate parts of the codebase. Only sequentialize when one result feeds into the next.
- Use context-mode MCP tools (`ctx_batch_execute`, `ctx_execute`, `ctx_execute_file`, `ctx_search`) for any command producing >20 lines of output. Do NOT use raw Bash for exploration or analysis.
- Use `Read` only when intending to `Edit` a file afterward. For analysis, use `ctx_execute_file`.
- Do not re-explore the repo structure every session — rely on this file and MEMORY.md for context.
- Prefer targeted tests over full test suite during development.
