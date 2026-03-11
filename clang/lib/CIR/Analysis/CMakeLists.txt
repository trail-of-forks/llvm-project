add_clang_library(clangCIRAnalysis
  CIRAnalysis.cpp
  Gap7Verify.cpp

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
