//===-- X86CtSelectAnalysis.h - Analyze CT Select Instructions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86CtSelectAnalysis pass, which analyzes constant-time
// select instructions in X86 machine code for security and optimization purposes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86CTSELECTANALYSIS_H
#define LLVM_LIB_TARGET_X86_X86CTSELECTANALYSIS_H

#include "X86.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class MachineFunction;
class MachineBasicBlock;

/// Analysis results for CT select instructions
struct CtSelectStats {
  unsigned TotalCtSelectInstructions = 0;
  unsigned CtSelectByWidth[4] = {0, 0, 0, 0}; // 128, 256, 512, other
  unsigned CtSelectByType[3] = {0, 0, 0};     // int, float, mixed
  
  void clear() {
    TotalCtSelectInstructions = 0;
    std::fill(std::begin(CtSelectByWidth), std::end(CtSelectByWidth), 0);
    std::fill(std::begin(CtSelectByType), std::end(CtSelectByType), 0);
  }
  
  void print(raw_ostream &OS) const;
  void dump() const;
};

/// X86 Constant-Time Select Analysis Pass
///
/// This pass analyzes the usage of constant-time select instructions in X86
/// machine code to provide insights for security analysis and optimization.
class X86CtSelectAnalysis : public MachineFunctionPass {
private:
  /// Statistics for the current function
  CtSelectStats FunctionStats;
  
  /// Statistics aggregated across all functions
  static CtSelectStats GlobalStats;
  
  /// Map from basic blocks to their CT select counts
  DenseMap<const MachineBasicBlock*, unsigned> BBCtSelectCounts;
  
  /// List of all CT select instructions found
  SmallVector<const MachineInstr*, 16> CtSelectInstructions;

public:
  static char ID;
  
  X86CtSelectAnalysis() : MachineFunctionPass(ID) {}
  
  bool runOnMachineFunction(MachineFunction &MF) override;
  
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  
  StringRef getPassName() const override {
    return "X86 Constant-Time Select Analysis";
  }
  
  /// Get statistics for the last analyzed function
  const CtSelectStats &getFunctionStats() const { return FunctionStats; }
  
  /// Get global statistics across all functions
  static const CtSelectStats &getGlobalStats() { return GlobalStats; }
  
  /// Get CT select count for a specific basic block
  unsigned getCtSelectCount(const MachineBasicBlock *MBB) const {
    auto It = BBCtSelectCounts.find(MBB);
    return It != BBCtSelectCounts.end() ? It->second : 0;
  }
  
  /// Get all CT select instructions found in the last function
  ArrayRef<const MachineInstr*> getCtSelectInstructions() const {
    return CtSelectInstructions;
  }

private:
  /// Analyze a single machine instruction
  void analyzeInstruction(const MachineInstr &MI);
  
  /// Determine the vector width category from opcode
  unsigned getVectorWidthCategory(unsigned Opcode) const;
  
  /// Determine the data type category from opcode  
  unsigned getDataTypeCategory(unsigned Opcode) const;
  
  /// Check if instruction is a CT select variant
  bool isCtSelectInstruction(const MachineInstr &MI) const;
  
  /// Print analysis results for the current function
  void printFunctionResults(const MachineFunction &MF) const;
};

/// Create the X86 CT Select Analysis pass
MachineFunctionPass *createX86CtSelectAnalysisPass();

} // end namespace llvm

#endif // LLVM_LIB_TARGET_X86_X86CTSELECTANALYSIS_H