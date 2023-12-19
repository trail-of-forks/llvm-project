// Copyright (c) 2023-present, Trail of Bits, Inc.
// All rights reserved.
//
// This source code is licensed in accordance with the terms specified in
// the LICENSE.TXT file found in the root directory of this source tree.

#pragma once

#include "PPCallbacks.h"

namespace clang {

enum PPCallbacks::EventKind : int {
  // Tell us just after one of the lexers has lexed a token.
  //
  // `Tok` is the token generated from one of the underlying lexers.
  // `Data` is zero or it is a raw source location for where the lexer was
  // invoked.
  TokenFromLexer,
  TokenFromTokenLexer,
  TokenFromCachingLexer,
  TokenFromAfterModuleImportLexer,

  // Tell the listener that the parser has split a token. This happens in C++
  // code for templates, e.g. `constant<bar<A>>>0`, where the `>>>` is first
  // treated as one token, but then where the parser realizes that it is
  // really `constant<bar<A>> > 0`.
  SplitToken,

  // Tell the listener that we've just lexed the hash token that should start
  // off a directive.
  //
  // `Tok` is the `#`.
  BeginDirective,

  // Ends with an `EndDirective`.
  //
  // `Tok` is the `#`.
  BeginSkippedArea,

  // Tell the listener that we're in a named directive, e.g. `if` or `define`.
  //
  // `Tok` is the `#`.
  // `Data` is a `Token *` of the token lexed after the `#`.
  SetNamedDirective,

  // Tell the listener that we're in an unnamed directive, e.g. GNU line
  // numbers, such as `# 1`.
  //
  // `Tok` is the `#`.
  // `Data` is a `const Token *` of the token lexed after the `#`.
  SetUnnamedDirective,

  // End a directive.
  //
  // `Tok` is the `tok::eod` token.
  EndDirective,

  // We thought something was a directive, but it wasn't, e.g. due to us
  // parsing a .S file.
  EndNonDirective,

  // `Tok` is the name of the macro being expanded.
  // `Data` is the `MacroInfo *`. For built-in macros, this may be `nullptr`.
  BeginMacroExpansion,
  SwitchToExpansion,
  BeginPreArgumentExpansion,
  EndPreArgumentExpansion,
  PrepareToCancelExpansion,  // E.g. `_Pragma` in macro argument pre-exansion.
  CancelExpansion,  // E.g. `_Pragma` in a macro parameter.
  EndMacroExpansion,

  // `Tok` is the name of the macro being expanded.
  // `Data` is a `MacroInfo *`.
  BeginMacroCallArgumentList,

  // `Tok` is the token that terminated the argument list, i.e. a `)`.
  // `Data` is a `MacroArgs *`. For built-in macros, this may be `nullptr`.
  EndMacroCallArgumentList,

  // `Tok` is the token just before the first token of the argument, e.g.
  // `(` or `,`.
  // `Data` is a `Token *` of the macro name.
  BeginMacroCallArgument,

  // `Tok` is the token just before the first token of the argument.
  // `Data` is a `Token *` just after the last token of the argument, e.g. a
  // `)` or a `,`.
  EndMacroCallArgument,

  // `Tok` is the token just before the first token of the variadic arguments,
  // e.g. a `(` or a `,`.
  // `Data` is a `Token *` of the macro name.
  BeginVariadicCallArgumentList,

  // `Tok` is the token just before the first token of the variadic arguments
  // `Data` is a `Token *` just after the last token of the arguments, e.g. a
  // `)` or a `,`.
  EndVariadicCallArgumentList,

  // `Tok` is the token which begins the substitution.
  // `Data` is `nullptr`.
  BeginSubstitution,

  // `Tok` is the token (previously visible via another event) which we want
  // to say begins the substitution.
  // `Data` is `nullptr`.
  BeginDelayedSubstitution,

  // `Tok` is the last token before the substituted tokens will begin being
  // outputted.
  // `Data` is a `Token *` of the first token of the substitution.
  SwitchToSubstitution,

  // `Tok` is the last substituted token.
  // `Data` is a `Token *` of the first token of the substitution.
  EndSubstitution,

  // `Tok` is the token (possibly previously visible via another event) which
  // is about to be pasted with something else.
  // `Data` is a `Token *` of the `##` token.
  BeginConcatenation,
  EndConcatenation,

  // `Tok` is a `##` for concatenation.
  // `Data` is a `Token *` of the token into which the right hand side of the
  // `##` is being concatenated.
  ConcatenationOperatorToken,

  // `Tok` is a right hand side for concatenation.
  // `Data` is a `Token *` of the token into which `Tok` is being concatenated.
  ConcatenationAccumulationToken,

  // `Tok` is the first output token in the intermediate results buffer.
  // `Data` is the number of tokens in the buffer.
  //
  // NOTE(pag): Before/After concatenation don't actually show us the resulting
  //            pasted token. Instead, that is done in `TokenLexer::Lex`, as it
  //            is reading from the resulting tokens. To observe the
  //            concatenation, use `BeginConcatenation` and `EndConcatenation`.
  BeforeParameterSubstitutions,
  AfterParameterSubstitutions,
  BeforeVAOpt,
  AfterVAOpt,
  BeforeStringify,
  AfterStringify,
  BeforeConcatenation,
  AfterConcatenation,
  BeforeMacroParameterUse,
  AfterMacroParameterUse,
  BeforeRemoveCommas,
  AfterRemoveCommas,

  // `Tok` is the token inside of a `__VA_OPT__` that is being skipped if
  // `__VA_ARGS__` is empty. `Data` is zero.
  SkippedVAOptToken,

  // // `Tok` is a special token elided from a macro definition body, e.g.
  // // `#` before a parameter name.
  // // `Data` is a `MacroInfo *`.
  // ElidedTokenInDefinitionBody,

  // // `Tok` is a token in the body.
  // TokenInDefinitionBody,
};

}  // namespace clang
