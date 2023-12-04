//===- MC/TargetRegistry.h - Target Registration ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the TargetRegistry interface, which tools can use to access
// the appropriate target specific classes (TargetMachine, AsmPrinter, etc.)
// which have been registered.
//
// Target specific class implementations should register themselves using the
// appropriate TargetRegistry interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_CCREGISTRY_H
#define LLVM_MC_CCREGISTRY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include <map>
#include <memory>
namespace llvm {

class CCObj {
    public:
    virtual ~CCObj() = default;
    virtual std::function<CCAssignFn> CCAssignFnForNode(CallingConv::ID CC,
                                                       bool Return,
                                                       bool isVarArg) = 0;
                                                    
    virtual std::optional<const MCPhysReg*> getCalleeSaves(const MachineFunction *M) = 0;     

    virtual std::optional< const uint32_t *> getCallPreservedMask(const MachineFunction& M,  CallingConv::ID) = 0;     

    virtual bool isTailCallEquiv(CallingConv::ID CC) = 0;
};


class CCRegistry {

public:
    using override_map = std::multimap<std::string, std::unique_ptr<CCObj>> ;
    using it_range = iterator_range<override_map::const_iterator>;

    static void registerCCOverrride(const char* TargetName, std::unique_ptr<CCObj> Handler);

    static it_range findCCOverrides(llvm::StringRef TargetName);
};

} // namespace llvm

#endif