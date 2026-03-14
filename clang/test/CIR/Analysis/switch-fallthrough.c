// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fclangir-analysis=switch-fallthrough -Wimplicit-fallthrough -verify %s -o /dev/null

// End-to-end test for A2 CIR switch fall-through analysis.
// Verifies that -fclangir-analysis=switch-fallthrough produces the same
// -Wimplicit-fallthrough diagnostics as the CFG-based analysis by walking
// cir.switch/cir.case region structure directly.

void f(void);
void g(void);
void a(void);
void b(void);
void c(void);

// ANLYS-01a: Basic fall-through detection.
void test_basic(int x) {
  switch (x) {
  case 1:
    f();
  case 2: // expected-warning {{unannotated fall-through between switch labels}} \
           // expected-note {{insert 'break;' to avoid fall-through}}
    g();
    break;
  }
}

// ANLYS-01b: Empty case body -- no warning.
// Consecutive case labels with no statements should NOT trigger a warning.
void test_empty_cases(int x) {
  switch (x) {
  case 1:
  case 2:
    f();
    break;
  }
}

// ANLYS-01c: Break suppresses warning.
void test_break(int x) {
  switch (x) {
  case 1:
    f();
    break;
  case 2:
    g();
    break;
  }
}

// ANLYS-01d: Return suppresses warning.
int test_return(int x) {
  switch (x) {
  case 1:
    return 1;
  case 2:
    return 2;
  }
  return 0;
}

// Multiple fall-throughs in one switch.
void test_multiple(int x) {
  switch (x) {
  case 1:
    a();
  case 2: // expected-warning {{unannotated fall-through between switch labels}} \
           // expected-note {{insert 'break;' to avoid fall-through}}
    b();
  case 3: // expected-warning {{unannotated fall-through between switch labels}} \
           // expected-note {{insert 'break;' to avoid fall-through}}
    c();
    break;
  }
}

// Default case handling.
void test_default(int x) {
  switch (x) {
  case 1:
    a();
  default: // expected-warning {{unannotated fall-through between switch labels}} \
            // expected-note {{insert 'break;' to avoid fall-through}}
    b();
    break;
  }
}

// No switch -- no crash, no warnings.
void test_no_switch(int x) {
  if (x)
    f();
}
