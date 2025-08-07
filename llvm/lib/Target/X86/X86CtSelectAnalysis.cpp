//===-- X86CtSelectAnalysis.cpp - Analyze CT Select Instructions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86CtSelectAnalysis pass, which analyzes 
// constant-time select instructions in X86 machine code.
//
//===----------------------------------------------------------------------===//

#include "X86CtSelectAnalysis.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "x86-ctselect-analysis"

using namespace llvm;

// Initialize static member
CtSelectStats X86CtSelectAnalysis::GlobalStats;

char X86CtSelectAnalysis::ID = 0;

// Register the pass
INITIALIZE_PASS(X86CtSelectAnalysis, "x86-ctselect-analysis", 
                "X86 Constant-Time Select Analysis", false, true)

void CtSelectStats::print(raw_ostream &OS) const {
  OS << "CT Select Statistics:\n";
  OS << "  Total CT Select Instructions: " << TotalCtSelectInstructions << "\n";
  OS << "  By Vector Width:\n";
  OS << "    128-bit: " << CtSelectByWidth[0] << "\n";
  OS << "    256-bit: " << CtSelectByWidth[1] << "\n"; 
  OS << "    512-bit: " << CtSelectByWidth[2] << "\n";
  OS << "    Other:   " << CtSelectByWidth[3] << "\n";
  OS << "  By Data Type:\n";
  OS << "    Integer: " << CtSelectByType[0] << "\n";
  OS << "    Float:   " << CtSelectByType[1] << "\n";
  OS << "    Mixed:   " << CtSelectByType[2] << "\n";
}

void CtSelectStats::dump() const {
  print(dbgs());
}

bool X86CtSelectAnalysis::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Running X86CtSelectAnalysis on function: " 
                    << MF.getName() << "\n");
  
  // Clear per-function data structures
  FunctionStats.clear();
  BBCtSelectCounts.clear();
  CtSelectInstructions.clear();
  
  // Analyze each basic block
  for (const auto &MBB : MF) {
    unsigned BBCtSelectCount = 0;
    
    // Analyze each instruction in the basic block
    for (const auto &MI : MBB) {
      if (isCtSelectInstruction(MI)) {
        analyzeInstruction(MI);
        CtSelectInstructions.push_back(&MI);
        BBCtSelectCount++;
      }
    }
    
    // Record basic block statistics
    if (BBCtSelectCount > 0) {
      BBCtSelectCounts[&MBB] = BBCtSelectCount;
    }
  }
  
  // Update global statistics
  GlobalStats.TotalCtSelectInstructions += FunctionStats.TotalCtSelectInstructions;
  for (int i = 0; i < 4; i++) {
    GlobalStats.CtSelectByWidth[i] += FunctionStats.CtSelectByWidth[i];
  }
  for (int i = 0; i < 3; i++) {
    GlobalStats.CtSelectByType[i] += FunctionStats.CtSelectByType[i];
  }
  
  // Print results if debug is enabled
  LLVM_DEBUG(printFunctionResults(MF));
  
  // This is an analysis pass, so it doesn't modify the function
  return false;
}

void X86CtSelectAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void X86CtSelectAnalysis::analyzeInstruction(const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  
  // Update total count
  FunctionStats.TotalCtSelectInstructions++;
  
  // Categorize by vector width
  unsigned WidthCategory = getVectorWidthCategory(Opcode);
  if (WidthCategory < 4) {
    FunctionStats.CtSelectByWidth[WidthCategory]++;
  }
  
  // Categorize by data type
  unsigned TypeCategory = getDataTypeCategory(Opcode);
  if (TypeCategory < 3) {
    FunctionStats.CtSelectByType[TypeCategory]++;
  }
  
  LLVM_DEBUG(dbgs() << "Found CT Select: " << MI.getOpcode() 
                    << " (width=" << WidthCategory 
                    << ", type=" << TypeCategory << ")\n");
}

unsigned X86CtSelectAnalysis::getVectorWidthCategory(unsigned Opcode) const {
  switch (Opcode) {
  // 128-bit variants
  case X86::CTSELECT_V2F64:
  case X86::CTSELECT_V4F32: 
  case X86::CTSELECT_V4I32:
  case X86::CTSELECT_V2I64:
  case X86::CTSELECT_V8I16:
  case X86::CTSELECT_V16I8:
  case X86::CTSELECT_V8F16:
  case X86::CTSELECT_V4F32X:
  case X86::CTSELECT_V4I32X:
  case X86::CTSELECT_V2F64X:
  case X86::CTSELECT_V2I64X:
  case X86::CTSELECT_V8I16X:
  case X86::CTSELECT_V16I8X:
  case X86::CTSELECT_V8F16X:
    return 0; // 128-bit
    
  // 256-bit variants  
  case X86::CTSELECT_V8F32:
  case X86::CTSELECT_V8I32:
  case X86::CTSELECT_V4F64:
  case X86::CTSELECT_V4I64:
  case X86::CTSELECT_V16I16:
  case X86::CTSELECT_V32I8:
  case X86::CTSELECT_V16F16:
    return 1; // 256-bit
    
  // 512-bit variants
  case X86::CTSELECT_V8F64:
  case X86::CTSELECT_V16F32:
  case X86::CTSELECT_V16I32:
  case X86::CTSELECT_V8I64:
  case X86::CTSELECT_V32I16:
  case X86::CTSELECT_V64I8:
  case X86::CTSELECT_V32F16:
    return 2; // 512-bit
    
  default:
    return 3; // Other/unknown
  }
}

unsigned X86CtSelectAnalysis::getDataTypeCategory(unsigned Opcode) const {
  switch (Opcode) {
  // Integer types
  case X86::CTSELECT_V4I32:
  case X86::CTSELECT_V2I64:
  case X86::CTSELECT_V8I16:
  case X86::CTSELECT_V16I8:
  case X86::CTSELECT_V4I32X:
  case X86::CTSELECT_V2I64X:
  case X86::CTSELECT_V8I16X:
  case X86::CTSELECT_V16I8X:
  case X86::CTSELECT_V8I32:
  case X86::CTSELECT_V4I64:
  case X86::CTSELECT_V16I16:
  case X86::CTSELECT_V32I8:
  case X86::CTSELECT_V16I32:
  case X86::CTSELECT_V8I64:
  case X86::CTSELECT_V32I16:
  case X86::CTSELECT_V64I8:
    return 0; // Integer
    
  // Floating point types
  case X86::CTSELECT_V2F64:
  case X86::CTSELECT_V4F32:
  case X86::CTSELECT_V8F16:
  case X86::CTSELECT_V4F32X:
  case X86::CTSELECT_V2F64X:
  case X86::CTSELECT_V8F16X:
  case X86::CTSELECT_V8F32:
  case X86::CTSELECT_V4F64:
  case X86::CTSELECT_V16F16:
  case X86::CTSELECT_V8F64:
  case X86::CTSELECT_V16F32:
  case X86::CTSELECT_V32F16:
    return 1; // Float
    
  default:
    return 2; // Mixed/other
  }
}

bool X86CtSelectAnalysis::isCtSelectInstruction(const MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();
  
  // Check if this is any of the CTSELECT opcodes
  switch (Opcode) {
  case X86::CTSELECT_V2F64:
  case X86::CTSELECT_V4F32:
  case X86::CTSELECT_V4I32:
  case X86::CTSELECT_V2I64:
  case X86::CTSELECT_V8I16:
  case X86::CTSELECT_V16I8:
  case X86::CTSELECT_V8F16:
  case X86::CTSELECT_V4F32X:
  case X86::CTSELECT_V4I32X:
  case X86::CTSELECT_V2F64X:
  case X86::CTSELECT_V2I64X:
  case X86::CTSELECT_V8I16X:
  case X86::CTSELECT_V16I8X:
  case X86::CTSELECT_V8F16X:
  case X86::CTSELECT_V8F32:
  case X86::CTSELECT_V8I32:
  case X86::CTSELECT_V4F64:
  case X86::CTSELECT_V4I64:
  case X86::CTSELECT_V16I16:
  case X86::CTSELECT_V32I8:
  case X86::CTSELECT_V16F16:
  case X86::CTSELECT_V8F64:
  case X86::CTSELECT_V16F32:
  case X86::CTSELECT_V16I32:
  case X86::CTSELECT_V8I64:
  case X86::CTSELECT_V32I16:
  case X86::CTSELECT_V64I8:
  case X86::CTSELECT_V32F16:
    return true;
  default:
    return false;
  }
}

void X86CtSelectAnalysis::printFunctionResults(const MachineFunction &MF) const {
  if (FunctionStats.TotalCtSelectInstructions == 0)
    return;
    
  dbgs() << "=== CT Select Analysis Results for " << MF.getName() << " ===\n";
  FunctionStats.print(dbgs());
  
  dbgs() << "  Basic Block Distribution:\n";
  for (const auto &Entry : BBCtSelectCounts) {
    dbgs() << "    " << Entry.first->getName() 
           << ": " << Entry.second << " instructions\n";
  }
  
  dbgs() << "  Instruction Details:\n";
  for (const auto *MI : CtSelectInstructions) {
    dbgs() << "    ";
    MI->print(dbgs());
  }
  
  dbgs() << "=== End CT Select Analysis ===\n";
}

// Factory function to create the pass
MachineFunctionPass *llvm::createX86CtSelectAnalysisPass() {
  return new X86CtSelectAnalysis();
}