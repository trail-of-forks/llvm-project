// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// Exercises the union and non-trivial-dtor-field branches that
// visitCXXParenListOrInitListExpr now handles. Constant-foldable cases
// short-circuit to a const_record store before reaching the visitor; the
// tests below use runtime calls to keep the visitor path live.

extern int makeInt();
extern float makeFloat();

union U {
  int i;
  float f;
};

[[maybe_unused]] void union_init_list_runtime_named() {
  U u{makeInt()};
}
// CIR-LABEL: cir.func {{.*}}union_init_list_runtime_named
// CIR: %[[U:.*]] = cir.alloca !rec_U
// CIR: %[[I:.*]] = cir.get_member %[[U]][0] {name = "i"}
// CIR: %[[V:.*]] = cir.call @{{.*}}makeInt
// CIR: cir.store{{.*}} %[[V]], %[[I]]

[[maybe_unused]] void union_init_list_runtime_designated() {
  U u{.f = makeFloat()};
}
// CIR-LABEL: cir.func {{.*}}union_init_list_runtime_designated
// CIR: %[[U:.*]] = cir.alloca !rec_U
// CIR: %[[F:.*]] = cir.get_member %[[U]][1] {name = "f"}
// CIR: %[[V:.*]] = cir.call @{{.*}}makeFloat
// CIR: cir.store{{.*}} %[[V]], %[[F]]


// ---- Struct with non-trivial-dtor field via paren-list (C++20) ----
//
// The visitor pushes a deferred destructor cleanup for `nt`. The
// CleanupDeactivationScope deactivates it once aggregate init succeeds; the
// local-variable cleanup then runs `WithDtorField::~WithDtorField` (which
// destroys `nt` transitively) on scope exit.

struct NonTrivial {
  ~NonTrivial();
  int x;
};

struct WithDtorField {
  NonTrivial nt;
  int extra;
};

[[maybe_unused]] void struct_with_dtor_field_paren_list() {
  WithDtorField s(NonTrivial{}, makeInt());
}
// CIR-LABEL: cir.func {{.*}}struct_with_dtor_field_paren_list
// CIR: %[[S:.*]] = cir.alloca !rec_WithDtorField
// CIR: %[[NT:.*]] = cir.get_member %[[S]][0] {name = "nt"}
// CIR: cir.get_member %[[NT]][0] {name = "x"}
// CIR: %[[E:.*]] = cir.get_member %[[S]][1] {name = "extra"}
// CIR: cir.call @{{.*}}makeInt
// CIR: cir.call @{{.*}}WithDtorFieldD1Ev(%[[S]])
// CIR: cir.return
