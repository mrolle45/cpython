// macro_helpers.h

// Some macros to aid in generation of code in pegen.

#ifndef MACRO_HELPERS_H
#define MACRO_HELPERS_H

#define MACRO_HELPERS_TEST 0x0

#if MACRO_HELPERS_TEST
static void macro_helpers_test() {
#endif

// Stringizing
// _PyPegen_STR(x) = "x", after expanding x.
#define _PyPegen_STR2(...) #__VA_ARGS__
#define _PyPegen_STR(x) _PyPegen_STR2(x)

// String concatenating
// _PyPegen_CONCAT(x, y) expands then concatenates x and y.
#define _PyPegen_CONCAT2(x, y) x ## y
#define _PyPegen_CONCAT(x, y) _PyPegen_CONCAT2(x, y)

// Call a macro with arguments.
// _PyPegen_CALL_MACRO(macro, arg, ...) expands to
//      macro(arg, ...)

#define _PyPegen_CALL_MACRO(macro, ...) macro(__VA_ARGS__)

// 
// Token expansion
// _PyPegen_EXPAND(x) expands x.
#define _PyPegen_EXPAND(x) _PyPegen_EXPAND2(x)
#define _PyPegen_EXPAND2(x) x
#define D_PyPegen_EXPAND2(x) <Expand 2 #x> x
//#define _PyPegen_EXPAND(x) x
#define D_PyPegen_EXPAND(x) <Expand 1 #x> D_PyPegen_EXPAND2(x)
//#define _PyPegen_EXPAND(...) <Expand 1 #__VA_ARGS__> _PyPegen_EXPAND2(__VA_ARGS__)
//
    _PyPegen_EXPAND(printf("%d", 2));
// _PyPegen_IF_ARGS tests for one or more arguments being present.
//  A single comma is same as no arguments.
// _PyPegen_IF_ARGS (if_args, ...) expands to `if_args` if any arguments, or nothing otherwise.
#define _PyPegen_IF_ARGS(if_args, ...) __VA_OPT__(if_args)

// _PyPegen_NAME_IF_ARGS tests for one or more arguments being present.
//  A single comma is same as no arguments.
// _PyPegen_IF_ARGS (name, suffix, ...) expands to
//      `suffix` if any arguments,
//      or just `name` otherwise.
#define _PyPegen_NAME_IF_ARGS(name, suffix, ...) \
    _PyPegen_CONCAT(name, __VA_OPT__(suffix))

// _PyPegen_HAS_PARAMS tests for a parameter sequence in parens.
// The argument is either the sequence or empty.  It is followed by a trailing comma.
// _PyPegen_HAS_PARAMS ((param, ...),) expands to 1.
// _PyPegen_HAS_PARAMS (,) expands to 0.
// This can be appended to the name of another macro to select alternate macros.
#define _PyPegen_PARAMS_SUFFIX_P2 1         // Called if HAS_PARAMS is called with params
#define _PyPegen_CHOOSE_PARAMS_SUFFIX2 0    // Called if HAS_PARAMS is called with no params
#define _PyPegen_CHOOSE_PARAMS_SUFFIX(...) _PyPegen_PARAMS_SUFFIX_P
#define _PyPegen_HAS_PARAMS(p, ...)  _PyPegen_CONCAT(_PyPegen_CHOOSE_PARAMS_SUFFIX p, 2)

// Parens as a macro, in order to follow another macro name without making a call to it.
#define _PyPegen_PARENS ()

#if MACRO_HELPERS_TEST + 0x01
// Tests for _PyPegen_HAS_PARAMS.
// _PyPegen_HAS_PARAMS (,) -> 0
    _PyPegen_HAS_PARAMS(,);
    _PyPegen_STR(_PyPegen_HAS_PARAMS(,)));
    // _PyPegen_HAS_PARAMS((p),) -> 1
   _PyPegen_STR(_PyPegen_HAS_PARAMS((p),));
#endif // MACRO_HELPERS_TEST

   // Add a _0 or _1 suffix, based on whether parameters are present.
   // The parameters are either an empty argument, or a sequence of parameters in parentheses.
   // A comma is required after the parmeters argument, anything after the comma is ignored.
   // _PyPegen_NAME_IF_PARAMS (N, (param, ...),)    expands to N_1.
   // _PyPegen_NAME_IF_PARAMS (N, (),)              expands to N_1.
   // _PyPegen_NAME_IF_PARAMS (N,)                  expands to N_0.

#define _PyPegen_NAME_IF_PARAMS(n, p, ...) \
    _PyPegen_CONCAT(n ## _, _PyPegen_HAS_PARAMS (p,)) \

   // Add an optional suffix, based on whether parameters are present.
   // The parameters are either an empty argument, or a sequence of parameters in parentheses.
   // A comma is required after the parmeters argument, anything after the comma is ignored.
   // _PyPegen_ADD_SFX_IF_PARAMS (N, S, (param, ...),)    expands to N ## s.
   // _PyPegen_ADD_SFX_IF_PARAMS (N, S, (),)              expands to N ## s.
   // _PyPegen_ADD_SFX_IF_PARAMS (N, S,)                  expands to N.

#define _PyPegen_ADD_SFX_0(N, S) N
#define _PyPegen_ADD_SFX_1(N, S) N ## S
#define _PyPegen_ADD_SFX_IF_PARAMS(N, S, p, ...) \
    _PyPegen_NAME_IF_PARAMS(_PyPegen_ADD_SFX, p,) (N, S) \

#if MACRO_HELPERS_TEST + 0x01
   // Tests for _PyPegen_NAME_IF_PARAMS.
   // _PyPegen_NAME_IF_PARAMS (N, ,) -> N_0
   _PyPegen_NAME_IF_PARAMS(N, ,);
   _PyPegen_STR(_PyPegen_NAME_IF_PARAMS(N, ,)));
   // _PyPegen_NAME_IF_PARAMS(N, (p),) -> N_1
   _PyPegen_STR(_PyPegen_NAME_IF_PARAMS(N, (p),));

   // Tests for _PyPegen_ADD_SFX_IF_PARAMS.
   // _PyPegen_ADD_SFX_IF_PARAMS (N, P, ,) -> N
   _PyPegen_ADD_SFX_IF_PARAMS(N, P, ,);
   _PyPegen_STR(_PyPegen_ADD_SFX_IF_PARAMS(N, P, ,)));
   // _PyPegen_NAME_IF_PARAMS(N, P, (p),) -> NP
   _PyPegen_ADD_SFX_IF_PARAMS(N, P, (p),);
   _PyPegen_STR(_PyPegen_ADD_SFX_IF_PARAMS(N, P, (p),));
#endif // MACRO_HELPERS_TEST

   // Apply a macro cumulatively to a starting value.  Similar to itertools.accumulate().
// Given a starting value, a combining macro, and zero or more combining elements.
//    With no combining elements, a trailing comma is required.
// The macro has two arguments: the accumulated result so far, and the next element,
//  and returns the new accumulated result.
// For example, _PyPegen_APPLY_CUMULATIVE(M, S, E1, E2) expands to
//      M(M(S, E1), E2)
// And _PyPegen_APPLY_CUMULATIVE(M, S,) expands to
//      S

#define _PyPegen_APPLY_MORE(macro, start, elem, ...) \
    _PyPegen_EXPAND(_PyPegen_APPLY_AGAIN _PyPegen_PARENS (macro, macro(start, elem), __VA_ARGS__)); \

    //"<< APPLY_MORE macro, start, elem, __VA_ARGS__ -> _PyPegen_EXPAND(_PyPegen_APPLY_AGAIN _PyPegen_PARENS (macro, macro(start, elem), __VA_ARGS__)) >>"; \

#define _PyPegen_APPLY_CUMULATIVE(macro, start, ...) \
    _PyPegen_NAME_IF_ARGS(_PyPegen_APPLY, _MORE, __VA_ARGS__) (macro, start, __VA_ARGS__); \

    //"<< APPLY_CUMULATIVE macro, start, __VA_ARGS__ >>"; \

#define _PyPegen_APPLY_AGAIN() \
    _PyPegen_EXPAND(_PyPegen_APPLY_HELPER); \

    //"<< APPLY_AGAIN -> _PyPegen_EXPAND(_PyPegen_APPLY_HELPER>>"; \

_PyPegen_APPLY_AGAIN _PyPegen_PARENS (macro, macro(start, elem), __VA_ARGS__));
#define _PyPegen_APPLY_HELPER(macro, start, elem, ...) \
    __VA_OPT__(_PyPegen_APPLY_CUMULATIVE(macro, start, elem, __VA_ARGS__)) \

   //<< APPLY_HELPER macro, start, elem, __VA_ARGS__ >> \

#define _PyPegen_APPLY(macro, start, ...) start

////#define M(S, E) (*(S)) E

{
   int i = __LINE__;

   _PyPegen_APPLY_CUMULATIVE(M, S, E1, E2);

   _PyPegen_APPLY_MORE(M, S, E1, E2);

   _PyPegen_EXPAND(_PyPegen_APPLY_AGAIN _PyPegen_PARENS (M, M(S, E1), E2));

    //typedef char M(M(foo_type, (int)), (void));
    typedef char M(M(foo_type, (int)), (void));
}
   _PyPegen_APPLY_CUMULATIVE(M, S,)

//
//#define _PyPegen_APPLY_CUMULATIVE_AGAIN(macro, start, elem, ...) \
//   << APPLY_AGAIN macro, start, elem, __VA_ARGS__ >> \
//    _PyPegen_APPLY_HELPER
//
//#define _PyPegen_APPLY_CUMULATIVE_AGAIN(...) \
//   _PyPegen_EXPAND(_PyPegen_APPLY_CUMULATIVE(__VA_ARGS__))

#if MACRO_HELPERS_TEST * 0x0
// Tests for _PyPegen_NAME_IF_PARAMS, all OK.
// _PyPegen_NAME_IF_PARAMS (N) -> N_0
   _PyPegen_STR(_PyPegen_NAME_IF_PARAMS (N));
// _PyPegen_NAME_IF_PARAMS (N, (p)) -> N_1
   _PyPegen_STR(_PyPegen_NAME_IF_PARAMS (N, (p)));

// _PyPegen_NAME_IF_PARAMS(X) -> X_0
   _PyPegen_STR(_PyPegen_NAME_IF_PARAMS(X));
// _PyPegen_NAME_IF_PARAMS(X, (p)) -> X_1
   _PyPegen_STR(_PyPegen_NAME_IF_PARAMS(X, (p)));
#endif // MACRO_HELPERS_TEST

#define X_0(t, n, p) "X_0" (#t, #n, #p)
//#define X_0(tnp) "X_0" (#tnp)
#define X_1(t, n, p) "X_0" (#t, #n, #p)
//#define X_1(...) "X_1" (__VA_ARGS__) #__VA_ARGS__

// Ungroup a sequence of items.  The argument is the sequence, wrapped in parentheses.
// _PyPegen_UNGROUP((item1, item2, ...)) expands to
//      item1, item2, ...

#define _PyPegen_UNGROUP(...) __VA_ARGS__
// _PyPegen_UNGROUP (1, 2)
   char * _foo = _PyPegen_STR((_PyPegen_UNGROUP (1, 2) ));

// Ungroup a sequence of items, with additional optional items.
// The argument is the sequence, wrapped in parentheses, followed by the optional additional items.
// If no additional items, the result has a blank at the end.
// _PyPegen_EXTEND((item1, item2, ...), more, ...) expands to
//      item1, item2, ..., more, ...
// _PyPegen_EXTEND((item1, item2, ...) or _PyPegen_EXTEND((item1, item2, ..., ) expands to
//      item1, item2, ...,

#define _PyPegen_EXTEND(args, params) \
    _PyPegen_UNGROUP args, params

#if MACRO_HELPERS_TEST * 0x0
// _PyPegen_EXTEND((1, 2), )                -> 1, 2,
   _PyPegen_STR((_PyPegen_EXTEND((1, 2), ) )); 
// _PyPegen_EXTEND((1, 2))                  -> 1, 2,
   _PyPegen_STR((_PyPegen_EXTEND((1, 2), ) ));
// _PyPegen_EXTEND((1, 2), (p))             -> 1, 2, (p)
   _PyPegen_STR((_PyPegen_EXTEND((1, 2), (p)) ));
#endif // MACRO_HELPERS_TEST

 /* Expand using one of two alternative macros, depending on whether there is a parameter list
     following the arguments.  Either _0 (no params) or _1 (params) is appended to given macro name.
    The arguments are in a single argument, wrapped in parentheses, but given individually to the chosen macro.
    The parameters are in a single argument, wrapped in parentheses and given still wrapped
     to the chosen macro, or else a blank or missing argument.
    _PyPegen_SEL_IF_PARAMS(name, (arg, ...), (param, ...))   -> name_1(arg, ..., (param, ...))
    _PyPegen_SEL_IF_PARAMS(name, (arg, ...), ())             -> name_1(arg, ..., ())
    _PyPegen_SEL_IF_PARAMS(name, (arg, ...))                 -> name_0(arg, ...)
    _PyPegen_SEL_IF_PARAMS(name, (arg, ...), )               -> name_0(arg, ...)
*/

#define _PyPegen_SEL_IF_PARAMS(name, args, ...) \
    _PyPegen_CALL_MACRO(_PyPegen_NAME_IF_PARAMS(name, __VA_ARGS__), _PyPegen_EXPAND(_PyPegen_EXTEND (args, __VA_ARGS__))) \

#define D_PyPegen_SEL_IF_PARAMS(name, args, ...) \
    <SEL __LINE__ #name, #args, #__VA_ARGS__> \
    <Calling _PyPegen_NAME_IF_PARAMS(name, __VA_ARGS__) with (_PyPegen_EXPAND(_PyPegen_UNGROUP args), __VA_ARGS__)> \
    _PyPegen_CALL_MACRO(_PyPegen_NAME_IF_PARAMS(name, __VA_ARGS__), _PyPegen_EXPAND(_PyPegen_EXTEND (args, __VA_ARGS__))) \

// Skip to here.
#if MACRO_HELPERS_TEST * 0x0
// Test _PyPegen_SEL_IF_PARAMS macro.
// N_0 and N_1 are not defined.
// _PyPegen_SEL_IF_PARAMS (N, (1, 2))           -> N_0(1, 2, )
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (N, (1, 2)) ));
// _PyPegen_SEL_IF_PARAMS (N, (1, 2), )         -> N_0 (1, 2, )
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (N, (1, 2), ) ));
// _PyPegen_SEL_IF_PARAMS (N, (1, 2), (p))      -> N_1 (1, 2, (p))
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (N, (1, 2), (p)) ));
// _PyPegen_SEL_IF_PARAMS (N, (1, 2), ())       -> N_1 (1, 2, ())
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (N, (1, 2), ()) ));

// X_0 and X_1 show their arguments.
// _PyPegen_SEL_IF_PARAMS (X, (1, 2))           -> X_0 (1, 2, )
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (X, (1, 2)) ));
// _PyPegen_SEL_IF_PARAMS (X, (1, 2), )         -> X_0 (1, 2, )
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (X, (1, 2), ) ));
// _PyPegen_SEL_IF_PARAMS (X, (1, 2), (p))      -> X_1 (1, 2, (p))
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (X, (1, 2), (p)) ));
// _PyPegen_SEL_IF_PARAMS (X, (1, 2), ())       -> X_1 (1, 2, ())
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (X, (1, 2), ()) ));
#endif // MACRO_HELPERS_TEST

#if MACRO_HELPERS_TEST
}
#endif

#endif // MACRO_HELPERS_H
