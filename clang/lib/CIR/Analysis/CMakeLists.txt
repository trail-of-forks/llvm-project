add_clang_library(clangCIRAnalysis
  # Analysis source files added here.
  # Concrete analysis passes will be added in Phase 11+.
  CIRAnalysis.cpp

  DEPENDS
  MLIRCIRPassIncGen
  MLIRCIROpsIncGen

  LINK_LIBS PUBLIC
  clangBasic

  MLIRAnalysis
  MLIRIR
  MLIRPass

  MLIRCIR
  MLIRCIRInterfaces
)
