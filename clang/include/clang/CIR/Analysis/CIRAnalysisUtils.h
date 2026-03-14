//===- CIRAnalysisUtils.h - CIR Analysis Utilities ---------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for CIR analyses that run before CIR-to-CIR passes.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_ANALYSIS_CIRANALYSISUTILS_H
#define CLANG_CIR_ANALYSIS_CIRANALYSISUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "clang/Basic/SourceManager.h"

namespace cir {

/// Convert an MLIR Location (as created by CIR codegen) back to a
/// Clang SourceLocation for diagnostic emission from CIR analyses.
///
/// Handles the common location kinds produced by CIRGen:
///  - FileLineColLoc: the most common, from CIRGenFunction::getLoc()
///  - FusedLoc: from CIRGenFunction::getLoc(SourceRange), extracts first
///  sub-loc
///  - Unknown/other: returns an invalid SourceLocation (graceful fallback)
inline clang::SourceLocation mlirLocToClangLoc(mlir::Location Loc,
                                               clang::SourceManager &SM) {
  // Handle FileLineColLoc (most common from CIR codegen).
  if (auto FileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(Loc)) {
    auto FileRef = SM.getFileManager().getOptionalFileRef(
        FileLoc.getFilename().getValue());
    if (FileRef)
      return SM.translateFileLineCol(&FileRef->getFileEntry(),
                                     FileLoc.getLine(), FileLoc.getColumn());
  }
  // Handle FusedLoc (from CIRGenFunction::getLoc(SourceRange)).
  if (auto Fused = mlir::dyn_cast<mlir::FusedLoc>(Loc)) {
    if (!Fused.getLocations().empty())
      return mlirLocToClangLoc(Fused.getLocations().front(), SM);
  }
  // Unknown location fallback -- return invalid SourceLocation.
  return clang::SourceLocation();
}

} // namespace cir

#endif // CLANG_CIR_ANALYSIS_CIRANALYSISUTILS_H
