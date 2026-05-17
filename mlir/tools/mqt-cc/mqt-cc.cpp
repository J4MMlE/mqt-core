/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "qasm3/Exception.hpp"
#include "qasm3/Importer.hpp"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace mlir;

// Command-line options
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir/.qasm file>"),
                                          cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> convertToQIR("emit-qir",
                                  cl::desc("Convert to QIR at the end"),
                                  cl::init(false));

static cl::opt<bool> recordIntermediates(
    "record-intermediates",
    cl::desc("Record intermediate IR after each compiler stage"),
    cl::init(false));

static cl::opt<bool> enableTiming("mlir-timing",
                                  cl::desc("Enable pass timing statistics"),
                                  cl::init(false));

static cl::opt<bool> enableStatistics("mlir-statistics",
                                      cl::desc("Enable pass statistics"),
                                      cl::init(false));

static cl::opt<bool>
    printIRAfterAllStages("mlir-print-ir-after-all-stages",
                          cl::desc("Print IR after each compiler stage"),
                          cl::init(false));

static cl::opt<bool> emitPipelineTime(
    "emit-pipeline-time",
    cl::desc("Print pipeline-only wall time in seconds to stderr"),
    cl::init(false));

static cl::opt<unsigned> pipelineIterations(
    "iterations",
    cl::desc("Run the compilation pipeline N times and report the average "
             "pipeline time (parse/IO done once; module cloned per iteration)"),
    cl::value_desc("N"),
    cl::init(1));

static cl::opt<bool> directImport(
    "direct-import",
    cl::desc("Use direct QASM3 → QC import (bypasses QuantumComputation)"),
    cl::init(false));

static cl::opt<bool> disableMergeSingleQubitRotationGates(
    "disable-merge-single-qubit-rotation-gates",
    cl::desc("Disable quaternion-based single-qubit rotation gate merging"),
    cl::init(false));

/**
 * @brief Load and parse a .qasm file via the legacy QuantumComputation path.
 */
static OwningOpRef<ModuleOp> loadQASMFileLegacy(StringRef filename,
                                                MLIRContext* context) {
  try {
    const ::qc::QuantumComputation qc =
        qasm3::Importer::importf(filename.str());
    return translateQuantumComputationToQC(context, qc);
  } catch (const qasm3::CompilerError& exception) {
    errs() << "Failed to parse QASM file '" << filename << "': '"
           << exception.what() << "'\n";
  } catch (const std::exception& exception) {
    errs() << "Failed to load QASM file '" << filename << "': '"
           << exception.what() << "'\n";
  }
  return nullptr;
}

/**
 * @brief Load and parse a .qasm file, dispatching to the chosen import path.
 */
static OwningOpRef<ModuleOp> loadQASMFile(StringRef filename,
                                          MLIRContext* context) {
  if (directImport) {
    return mlir::qc::translateQASM3ToQC(context, filename.str());
  }
  return loadQASMFileLegacy(filename, context);
}

/**
 * @brief Load and parse a .mlir file
 */
static OwningOpRef<ModuleOp> loadMLIRFile(StringRef filename,
                                          MLIRContext* context) {
  // Set up the input file
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    errs() << "Failed to load file '" << filename << "': '" << errorMessage
           << "'\n";
    return nullptr;
  }

  // Parse the input MLIR
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, context);
}

/**
 * @brief Write the module to the output file
 */
static mlir::LogicalResult writeOutput(ModuleOp module, StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    errs() << errorMessage << "\n";
    return mlir::failure();
  }

  module.print(output->os());
  output->keep();
  return mlir::success();
}

int main(int argc, char** argv) {
  const InitLLVM y(argc, argv);

  // Parse command-line options; exit on error and print to stderr
  cl::ParseCommandLineOptions(argc, argv, "MQT Core Compiler Driver\n");

  // Set up MLIR context with all required dialects
  DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect>();
  registry.insert<qco::QCODialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<LLVM::LLVMDialect>();

  // Configure the compiler pipeline
  QuantumCompilerConfig config;
  config.convertToQIR = convertToQIR;
  config.recordIntermediates = recordIntermediates;
  config.enableTiming = enableTiming;
  config.enableStatistics = enableStatistics;
  config.printIRAfterAllStages = printIRAfterAllStages;
  config.disableMergeSingleQubitRotationGates =
      disableMergeSingleQubitRotationGates;

  // Parse the input once, then clone the module for each benchmark iteration.
  // All dialects are loaded upfront, so the context state is stable across runs.
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> templateModule;
  if (inputFilename.getValue().ends_with(".qasm")) {
    templateModule = loadQASMFile(inputFilename, &context);
  } else {
    templateModule = loadMLIRFile(inputFilename, &context);
  }
  if (!templateModule) {
    return 1;
  }

  // Run the compilation pipeline (optionally N times for benchmarking).
  const unsigned iterations = pipelineIterations;
  std::vector<double> iterationTimes;
  iterationTimes.reserve(iterations);
  OwningOpRef<ModuleOp> lastResult;

  for (unsigned i = 0; i < iterations; ++i) {
    OwningOpRef<ModuleOp> current(templateModule.get().clone());

    const QuantumCompilerPipeline pipeline(config);

    CompilationRecord record;
    const auto pipelineStart = std::chrono::steady_clock::now();
    if (pipeline.runPipeline(current.get(), recordIntermediates ? &record : nullptr).failed()) {
      errs() << "Compilation pipeline failed\n";
      return 1;
    }
    const auto pipelineEnd = std::chrono::steady_clock::now();
    iterationTimes.push_back(
        std::chrono::duration<double>(pipelineEnd - pipelineStart).count());

    if (i == iterations - 1) {
      lastResult = std::move(current);

      if (recordIntermediates) {
        outs() << "=== Compilation Record ===\n";
        outs() << "After QC Import:\n" << record.afterQCImport << "\n";
        outs() << "After Initial QC Canonicalization:\n"
               << record.afterInitialCanon << "\n";
        outs() << "After QC-to-QCO Conversion:\n"
               << record.afterQCOConversion << "\n";
        outs() << "After Initial QCO Canonicalization:\n"
               << record.afterQCOCanon << "\n";
        outs() << "After Optimization:\n" << record.afterOptimization << "\n";
        outs() << "After Final QCO Canonicalization:\n"
               << record.afterOptimizationCanon << "\n";
        outs() << "After QCO-to-QC Conversion:\n"
               << record.afterQCConversion << "\n";
        outs() << "After Final QC Canonicalization:\n"
               << record.afterQCCanon << "\n";
        outs() << "After QC-to-QIR Conversion:\n"
               << record.afterQIRConversion << "\n";
        outs() << "After QIR Canonicalization:\n"
               << record.afterQIRCanon << "\n";
      }
    }
  }

  if (emitPipelineTime) {
    errs() << "pipeline-times: ";
    for (unsigned i = 0; i < iterationTimes.size(); ++i) {
      if (i > 0) {
        errs() << ",";
      }
      errs() << iterationTimes[i];
    }
    errs() << "\n";
  }

  // Write the output
  if (writeOutput(lastResult.get(), outputFilename).failed()) {
    errs() << "Failed to write output file: " << outputFilename << "\n";
    return 1;
  }

  return 0;
}
