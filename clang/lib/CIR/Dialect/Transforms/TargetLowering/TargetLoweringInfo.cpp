//===- TargetLoweringInfo.cpp - Per-target transform-pass info base -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetLoweringInfo.h"

namespace cir {

TargetLoweringInfo::TargetLoweringInfo(std::unique_ptr<ABIInfo> info)
    : info(std::move(info)) {}

TargetLoweringInfo::~TargetLoweringInfo() = default;

} // namespace cir
