#ifndef PEGEN_H
#define PEGEN_H
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <token.h>
#include <pycore_ast.h>
#include "stdbool.h"
#define MACRO_HELPERS_TEST 0
#include "macro_helpers.h"


#if 0
#define PyPARSE_YIELD_IS_KEYWORD        0x0001
#endif

#define PyPARSE_DONT_IMPLY_DEDENT       0x0002

#if 0
#define PyPARSE_WITH_IS_KEYWORD         0x0003
#define PyPARSE_PRINT_IS_FUNCTION       0x0004
#define PyPARSE_UNICODE_LITERALS        0x0008
#endif

#define PyPARSE_IGNORE_COOKIE 0x0010
#define PyPARSE_BARRY_AS_BDFL 0x0020
#define PyPARSE_TYPE_COMMENTS 0x0040
#define PyPARSE_ASYNC_HACKS   0x0080
#define PyPARSE_ALLOW_INCOMPLETE_INPUT 0x0100

#define CURRENT_POS (-5)

typedef struct _memo {
    int type;               // Indicates the Rule which was parsed.
    int mark;               // Token index following the parsed rule, if successful,
                            // -1 if parse failed.
    struct _memo *next;     // Chain of Memos leading from the starting Token of the parse.
    // Variable length area for the cached result value, if successful.
    // It is a copy of the parse result.
    char result[1];
} Memo;

typedef struct _token {
    int type;
    PyObject *bytes;
    int level;
    int lineno, col_offset, end_lineno, end_col_offset;
    Memo *memo;
} Token;

typedef struct {
    char *str;
    int type;
} KeywordToken;


typedef struct {
    struct {
        int lineno;
        char *comment;  // The " <tag>" in "# type: ignore <tag>"
    } *items;
    size_t size;
    size_t num_items;
} growable_comment_array;

typedef struct RuleDescr RuleDescr;
typedef struct RuleInfo RuleInfo;

typedef struct {
    struct tok_state *tok;
    Token **tokens;
    int mark;
    int fill, size;
    PyArena *arena;
    KeywordToken **keywords;
    char **soft_keywords;
    int n_keyword_lists;
    // Pointers to local values that are currently in scope.
    // The array is the largest size that is needed.
    // It consists of addresses of:
    //  Return value of the current rule.
    //  Parameters of the current rule.
    //  Named variables (i.e. NAME = item in the grammar)
    //      in each rule alternative, outermost first.
    //      Only those variables that have already been parsed.
    void * (*local_values);

    int start_rule;
    // The rule, if any, currently being parsed.
    const RuleInfo * rule;
    int *errcode;
    int parsing_started;
    PyObject* normalize;
    int starting_lineno;
    int starting_col_offset;
    int error_indicator;
    // Indicator for a Cut being parsed.
    // It is cleared at the start of parsing any Rule alternative.
    // It is set parsing any Rule alternative when it parses a Cut.
    // If set after parsing a Rule alternative which fails, then
    //  any remaining alternatives are skipped.
    int cut_indicator;

    int flags;
    int feature_version;
    growable_comment_array type_ignore_comments;
    Token *known_err_token;
    int level;
    int call_invalid_rules;
} Parser;

// Success or failure of a parse attempt.
typedef _Bool ParseStatus;

// Pointer to sized result of successful parse.
// The result could be any type.
typedef struct {
    void * addr;
    size_t size;
} ParseResultPtr;

// Macros to create / initialize a ParseResultPtr and access its value.

#define _PyPegen_PARSERESULTPTR(data) \
    {&(data), sizeof (data)}

Py_LOCAL_INLINE(ParseResultPtr)
_PyPegen_make_result_ptr(void * data) {
    ParseResultPtr ptr = _PyPegen_PARSERESULTPTR(data);
        return ptr;
}

#define _PyPegen_PARSE_REF(ppRes, type) \
    *(type *)(ppRes).addr

#define _PyPegen_PARSE_REF_UNTYPED(ppRes) \
    _PyPegen_PARSE_REF(ppRes, void *)

#define _PyPegen_PARSE_COPY_FROM(dst, ppRes) \
    memcpy(*(dst), (ppRes)->addr, (ppRes)->size);

#define _PyPegen_PARSE_COPY_TO(ppRes, src) \
    memcpy((ppRes)->addr, *(src), (ppRes)->size);

// Return a successful parse result via a result pointer, and return true as ParseStatus.
// The result pointer is optional.  If NULL, the type and src are not evaluated.
#define _PyPegen_RETURN_RESULT(ppRes, type, src) {              \
    if (ppRes) {                                                \
        /*assert ((ppRes)->size == sizeof(type)); */                \
        _PyPegen_PARSE_REF(*(ppRes), type) = src;    \
    }                                                           \
    return true;                                                \
}

#if MACRO_HELPERS_TEST
static void pegen_test() {
#endif
    // Macros to declare a parse result variable and corresponding ParseResultPtr.
// Optionally add it to or copy it from the Parser's variables.
// The variable has a name and a type, and possibly a set of parameters.
// If it has parameters, it is a callable, and they are in a parenthesized sequence.
// Otherwise, it is an ordinary object, and the parameters argument is empty.

// ... for anonymous VarItem.
#define _PyPegen_DECL_LOCAL_PARSE_RESULT(type, name, params) \
    _PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (type, name), params); \
    ParseResultPtr _ptr_ ## name = _PyPegen_PARSERESULTPTR(name) \

// ... for global variable inherited from calling function.
#define _PyPegen_GET_GLOBAL_PARSE_RESULT(type, name, params, index) \
    _PyPegen_DECL_LOCAL_PARSE_RESULT(type, name, params); \
    name = * (_PyPegen_SEL_IF_PARAMS(_PyPegen_TYPE, (type), params)) _p->local_values[index] \

// ... for new global variable.
#define _PyPegen_ADD_GLOBAL_PARSE_RESULT(type, name, params, index) \
    _PyPegen_DECL_LOCAL_PARSE_RESULT(type, name, params); \
    _p->local_values[index] = (_PyPegen_SEL_IF_PARAMS(_PyPegen_TYPE, (type), params) *) &name \

// ... for a pointer to the variable, to use as a ppRes argument to a parse function.
#define _PyPegen_PARSE_RESULT_PTR(name) \
    (& _ptr_ ## name)

// ... type expression for pointer to any object type.
#define _PyPegen_TYPE_0(type, _unused) \
    type *

// ... type expression for pointer to any function type.
#define _PyPegen_TYPE_1(type, params) \
    type (*) params

// ... declare any object type variable.
#define _PyPegen_DECL_VAR_0(type, name, _unused) \
    type name

// ... declare any function type variable.
#define _PyPegen_DECL_VAR_1(type, name, params) \
    type (*name) params

#if MACRO_HELPERS_TEST
// Test the above macros...

// Skip to here.
#define XX_1(n, t, p) "XX_1"(n, t, p)

__LINE__;
// _PyPegen_SEL_IF_PARAMS(XX, (int, name), (int *, const int *));
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS(XX, (int, name), (int *, const int *)) ));

__LINE__;
// _PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (int, name), (int *, const int *));
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (int, name), (int *, const int *)) ));

__LINE__;
// _PyPegen_DECL_LOCAL_PARSE_RESULT(int, name, (int *, const int *));
   _PyPegen_STR((_PyPegen_DECL_LOCAL_PARSE_RESULT(int, name, (int *, const int *)) ));

__LINE__;
// _PyPegen_SEL_IF_PARAMS(XX, (int, name), );
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS(XX, (int, name), ) ));

__LINE__;
// _PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (int, name), );
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (int, name), ) ));

__LINE__;
// _PyPegen_DECL_LOCAL_PARSE_RESULT(int, name, );
   _PyPegen_STR((_PyPegen_DECL_LOCAL_PARSE_RESULT(int, name, ) ));

// _PyPegen_GET_GLOBAL_PARSE_RESULT(int, name, (int *, const int *), 42);
   _PyPegen_STR(_PyPegen_GET_GLOBAL_PARSE_RESULT(int, name, (int *, const int *), 42) );

// _PyPegen_GET_GLOBAL_PARSE_RESULT(int, name, , 42);
   _PyPegen_STR(_PyPegen_GET_GLOBAL_PARSE_RESULT(int, name, , 42) );

// _PyPegen_ADD_GLOBAL_PARSE_RESULT(int, name, (int *, const int *), 42);
   _PyPegen_STR(_PyPegen_ADD_GLOBAL_PARSE_RESULT(int, name, (int *, const int *), 42) );

// _PyPegen_ADD_GLOBAL_PARSE_RESULT(int, name, , 42);
   _PyPegen_STR(_PyPegen_ADD_GLOBAL_PARSE_RESULT(int, name, , 42) );

// Test some variable declaration macros...

    _PyPegen_STR((_PyPegen_DECL_LOCAL_PARSE_RESULT(int, name, (int *, const int *)) ));

    _PyPegen_STR((_PyPegen_DECL_LOCAL_PARSE_RESULT(int, name, ) ));

    _PyPegen_STR((_PyPegen_GET_GLOBAL_PARSE_RESULT(int, name, (int *, const int *), 42) ));

    _PyPegen_STR((_PyPegen_GET_GLOBAL_PARSE_RESULT(int, name, , 42) ));

    _PyPegen_STR((_PyPegen_ADD_GLOBAL_PARSE_RESULT(int, name, (int *, const int *), 42) ));

    _PyPegen_STR((_PyPegen_ADD_GLOBAL_PARSE_RESULT(int, name, , 42) ));

    _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (_PyPegen_TYPE, (int), ) ));

    _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (_PyPegen_TYPE, (int), (int *, const int *)) ));

#endif // MACRO_HELPERS_TEST
#if MACRO_HELPERS_TEST
}
#endif

// Typedefs for callback parsing functions...
//  ... Returns status, and result via pointer.
typedef ParseStatus (ParseFunc)(Parser * p, ParseResultPtr * _ppRes);
//  ... Returns just result via pointer.
typedef void (ParseTrue)(Parser * p, ParseResultPtr * _ppRes);
//  ... Returns only status.
typedef ParseStatus (ParseTest)(Parser * p);

typedef struct {
    cmpop_ty cmpop;
    expr_ty expr;
} CmpopExprPair;

typedef struct {
    expr_ty key;
    expr_ty value;
} KeyValuePair;

typedef struct {
    expr_ty key;
    pattern_ty pattern;
} KeyPatternPair;

typedef struct {
    arg_ty arg;
    expr_ty value;
} NameDefaultPair;

typedef struct {
    asdl_arg_seq *plain_names;
    asdl_seq *names_with_defaults; // asdl_seq* of NameDefaultsPair's
} SlashWithDefault;

typedef struct {
    arg_ty vararg;
    asdl_seq *kwonlyargs; // asdl_seq* of NameDefaultsPair's
    arg_ty kwarg;
} StarEtc;

typedef struct { operator_ty kind; } AugOperator;

typedef struct {
    void* element;
    int is_keyword;
} KeywordOrStarred;

// Static information about a Rule being parsed.
struct RuleDescr {
    ParseFunc * rhs;        // Parses the ordered choice of Alts.
    const char* name;       // For debugging.
    const char* expr;       // For debugging.
    // Used if the rule is memoized, default initialization otherwise ...
    int memo;               // This is the key corresponding to the rule.
    size_t result_size;     // Size of the result to store in the Memo.
};

#define DECL_RULE_DESCR(name, rule, result_type, expr)  \
    static RuleDescr name = {                           \
        _ ## rule ## __rhs, # rule, expr,               \
        rule ## _type, sizeof(result_type)              \
    };

// Dynamic information about the Rule being parsed.
// This is kept in the Parser and saved/restored on entering/leaving the rule.
struct RuleInfo {
    const RuleDescr * descr;        // Which Rule.
    void ** local_vars;             // Local variables visible within the rule.
    int level;                      // Depth of Rule parse calls.
};

typedef struct RuleAltDescr {
    ParseFunc * parse;
    const char* name;
    const char* expr;
} RuleAltDescr;


// Internal parser functions
#if defined(Py_DEBUG)
void _PyPegen_clear_memo_statistics(void);
PyObject *_PyPegen_get_memo_statistics(void);
#endif

// Helper functions / macros to parse some grammar node, returning the success / failure result.
// Some functions return the parsed value via a result pointer, which may be NULL.
// If the parse fails, the mark is left unchanged.
// If the parser's error_indicator is true, parsing fails immediately.  It can also be
// set by the parse (such as being out of memory).

// Parsers for a Rule.  Actually, parse the ordered choice of alternatives.  The Rule itself is called
//  with some parameters, whose addresses are copied into the local_vars[] array.
//  There are variations depending on left recursion and memoization...

//  ... Plain Rule, no memo, not recursive.
//      Or recursive but not the leader (this doesn't need to be memoized).
ParseStatus _PyPegen_parse_rule(Parser* p, ParseResultPtr * ppRes, RuleDescr* rule, void * local_vars[]);

//  ... Memoized Rule, not a left-recursive group leader.
ParseStatus _PyPegen_parse_memo_rule(Parser* p, ParseResultPtr * ppRes, RuleDescr* rule, void* local_vars[]);

//  ... Left-recursive group leader Rule.
ParseStatus _PyPegen_parse_recursive_rule(Parser* p, ParseResultPtr * ppRes, RuleDescr* rule, void* local_vars[]);

// Helper function to parse a rule alternative, returning the parse result.
// If the parse fails, false is returned and the mark is left unchanged.
ParseStatus _PyPegen_parse_alt(Parser *p, ParseResultPtr * ppRes, const RuleAltDescr *alt);

// Helper function to parse several rule alternatives, returning the first successful parse result.
// However, if an alternative sets p->cut_indicator, no more alternatives are tried.
// If the parse fails, false is returned and the mark is left unchanged.
// All the alternatives have to agree on their result type.
ParseStatus _PyPegen_parse_alts(Parser *p, ParseResultPtr * ppRes, const RuleAltDescr alts[], int count);

// Macro to parse several rule alternatives, given an array of alternative descriptors.
#define _PyPegen_PARSE_ALT_ARRAY(p, ppres, alts) \
    _PyPegen_parse_alts(p, ppres, alts, sizeof alts / sizeof alts[0]);

// Parse an item some number of times, returning an asdl_seq * with the parsed results as elements.
// Given a minimum and maximum number of items.  If the minimum number is not reached, the parse fails.
// If the maximum number is reached, no more items are attempted.  max = 0 means no limit.
// Optionally provide a separator item, which must be parsed between consecutive items.
ParseStatus _PyPegen_parse_seq(Parser *p, ParseResultPtr * ppRes,
    ParseFunc item, size_t item_size, ParseFunc * sep, int min, int max);

// Variations of the above, for different combinations of parameters.
//
//  ... Optionally parse an item, always succeeds.  Return value has either 0 or 1 elements.
Py_LOCAL_INLINE(ParseStatus)
_PyPegen_parse_opt(Parser * p, ParseResultPtr * ppRes, ParseFunc item, size_t item_size) {
    _PyPegen_parse_seq(p, ppRes, item, item_size, NULL, 0, 1);
}

//  ... Parse an item as many times as possible.  Optionally require at least one item.
// Parse an item several times, returning an asdl_seq * with the parsed results as elements.
Py_LOCAL_INLINE(ParseStatus)
_PyPegen_parse_repeat(Parser * p, ParseResultPtr * ppRes, ParseFunc item, size_t item_size, int repeat1) {
    _PyPegen_parse_seq(p, ppRes, item, item_size, NULL, repeat1, 0);
}

//  ... Parse an item as many times as possible, with a given separator item, which must be parsed between consecutive items.
//      Optionally require at least one item.
Py_LOCAL_INLINE(ParseStatus)
_PyPegen_parse_gather(Parser * p, ParseResultPtr * ppRes, ParseFunc item, size_t item_size, ParseFunc sep, int repeat1) {
    _PyPegen_parse_seq(p, ppRes, item, item_size, sep, repeat1, 0);
}

// Test parsing an item without consuming any input or computing any value.  Does not return any value.
// The parse succeeds if the item parse success matches the given success value.
ParseStatus _PyPegen_parse_lookahead(Parser *, ParseStatus positive, ParseTest item);

// Parse an item, requiring it to succeed.  If the item parse fails, raise a SyntaxError.
void _PyPegen_parse_forced(Parser *p, ParseResultPtr * ppRes, ParseFunc item, const char* expected);

// Parse a NAME Token.  Result value is an AST expr_ty object with Name_kind.
ParseStatus _PyPegen_parse_name(Parser *p, ParseResultPtr * ppRes);

// Parse any soft keyword or just a specific soft keyword.
// Same as _PyPegen_parse_name(), but fails if the name is not an actual soft keyword, or the given keyword. 
ParseStatus _PyPegen_parse_any_soft_keyword(Parser *p, ParseResultPtr * ppRes);
ParseStatus _PyPegen_parse_soft_keyword(Parser *p, ParseResultPtr * ppRes, const char * keyword);

// Parse a NUMBER Token.  Result value is an AST expr_ty object with Constant_kind.  Its value is
//  the appropriate numeric Python object.
ParseStatus _PyPegen_parse_number(Parser *p, ParseResultPtr * ppRes);

// Parse a STRING Token.  Result value is a Token *.
ParseStatus _PyPegen_parse_string(Parser *p, ParseResultPtr * ppRes);

// Parse a given type of Token.  Result value is a Token *.
ParseStatus _PyPegen_parse_token(Parser *p, ParseResultPtr * ppRes, int type);

// Parse an OP Token.  Result value is a Token *.
ParseStatus _PyPegen_parse_op(Parser *p, ParseResultPtr * ppRes);

// Parse an OP Token with given character as its string.  Result value is a Token *.
ParseStatus _PyPegen_parse_char(Parser *p, ParseResultPtr * ppRes, char c);


void _PyPegen_parse_cut(Parser *p);

int _PyPegen_lookahead_with_name(int, expr_ty (func)(Parser *), Parser *);
int _PyPegen_lookahead_with_int(int, Token *(func)(Parser *, int), Parser *, int);
int _PyPegen_lookahead_with_string(int , expr_ty (func)(Parser *, const char*), Parser *, const char*);
int _PyPegen_lookahead(int, void *(func)(Parser *), Parser *);


Token * _PyPegen_expect_token(Parser *p, int type);
Token *_PyPegen_expect_char(Parser *p, char c);
void* _PyPegen_expect_forced_result(Parser *p, void* result, const char* expected);
Token *_PyPegen_expect_forced_token(Parser *p, int type, const char* expected);
expr_ty _PyPegen_expect_soft_keyword(Parser *p, const char *keyword);
ParseStatus _PyPegen_group(Parser *p, ParseResultPtr * ppRes, ParseFunc rhs);
expr_ty _PyPegen_soft_keyword_token(Parser *p);
Token *_PyPegen_get_last_nonnwhitespace_token(Parser *p);
int _PyPegen_fill_token(Parser *p);
expr_ty _PyPegen_name_token(Parser *p);
expr_ty _PyPegen_number_token(Parser *p);
Token *_PyPegen_string_token(Parser *p);
Token * _PyPegen_type_comment_token(Parser *p);
Py_ssize_t _PyPegen_byte_offset_to_character_offset(PyObject *line, Py_ssize_t col_offset);
int _PyPegen_location_start(Parser* p, int * lineno, int * col_offset);   // Current location in input.
int _PyPegen_location_end(Parser* p, int * lineno, int * col_offset);     // Current location in input, but backs up over whitespace.

// Error handling functions and APIs
typedef enum {
    STAR_TARGETS,
    DEL_TARGETS,
    FOR_TARGETS
} TARGETS_TYPE;

int _Pypegen_raise_decode_error(Parser *p);
void _PyPegen_raise_tokenizer_init_error(PyObject *filename);
int _Pypegen_tokenizer_error(Parser *p);
void *_PyPegen_raise_error(Parser *p, PyObject *errtype, const char *errmsg, ...);
void *_PyPegen_raise_error_known_location(Parser *p, PyObject *errtype,
                                          Py_ssize_t lineno, Py_ssize_t col_offset,
                                          Py_ssize_t end_lineno, Py_ssize_t end_col_offset,
                                          const char *errmsg, va_list va);
void _Pypegen_set_syntax_error(Parser* p, Token* last_token);
Py_LOCAL_INLINE(void *)
RAISE_ERROR_KNOWN_LOCATION(Parser *p, PyObject *errtype,
                           Py_ssize_t lineno, Py_ssize_t col_offset,
                           Py_ssize_t end_lineno, Py_ssize_t end_col_offset,
                           const char *errmsg, ...)
{
    va_list va;
    va_start(va, errmsg);
    Py_ssize_t _col_offset = (col_offset == CURRENT_POS ? CURRENT_POS : col_offset + 1);
    Py_ssize_t _end_col_offset = (end_col_offset == CURRENT_POS ? CURRENT_POS : end_col_offset + 1);
    _PyPegen_raise_error_known_location(p, errtype, lineno, _col_offset, end_lineno, _end_col_offset, errmsg, va);
    va_end(va);
    return NULL;
}
#define RAISE_SYNTAX_ERROR(msg, ...) _PyPegen_raise_error(p, PyExc_SyntaxError, msg, ##__VA_ARGS__)
#define RAISE_INDENTATION_ERROR(msg, ...) _PyPegen_raise_error(p, PyExc_IndentationError, msg, ##__VA_ARGS__)
#define RAISE_SYNTAX_ERROR_KNOWN_RANGE(a, b, msg, ...) \
    RAISE_ERROR_KNOWN_LOCATION(p, PyExc_SyntaxError, (a)->lineno, (a)->col_offset, (b)->end_lineno, (b)->end_col_offset, msg, ##__VA_ARGS__)
#define RAISE_SYNTAX_ERROR_KNOWN_LOCATION(a, msg, ...) \
    RAISE_ERROR_KNOWN_LOCATION(p, PyExc_SyntaxError, (a)->lineno, (a)->col_offset, (a)->end_lineno, (a)->end_col_offset, msg, ##__VA_ARGS__)
#define RAISE_SYNTAX_ERROR_STARTING_FROM(a, msg, ...) \
    RAISE_ERROR_KNOWN_LOCATION(p, PyExc_SyntaxError, (a)->lineno, (a)->col_offset, CURRENT_POS, CURRENT_POS, msg, ##__VA_ARGS__)
#define RAISE_SYNTAX_ERROR_INVALID_TARGET(type, e) _RAISE_SYNTAX_ERROR_INVALID_TARGET(p, type, e)

Py_LOCAL_INLINE(void *)
CHECK_CALL(Parser *p, void *result)
{
    if (result == NULL) {
        assert(PyErr_Occurred());
        p->error_indicator = 1;
    }
    return result;
}

/* This is needed for helper functions that are allowed to
   return NULL without an error. Example: _PyPegen_seq_extract_starred_exprs */
Py_LOCAL_INLINE(void *)
CHECK_CALL_NULL_ALLOWED(Parser *p, void *result)
{
    if (result == NULL && PyErr_Occurred()) {
        p->error_indicator = 1;
    }
    return result;
}

#define CHECK(type, result) ((type) CHECK_CALL(p, result))
#define CHECK_NULL_ALLOWED(type, result) ((type) CHECK_CALL_NULL_ALLOWED(p, result))

expr_ty _PyPegen_get_invalid_target(expr_ty e, TARGETS_TYPE targets_type);
const char *_PyPegen_get_expr_name(expr_ty);
Py_LOCAL_INLINE(void *)
_RAISE_SYNTAX_ERROR_INVALID_TARGET(Parser *p, TARGETS_TYPE type, void *e)
{
    expr_ty invalid_target = CHECK_NULL_ALLOWED(expr_ty, _PyPegen_get_invalid_target(e, type));
    if (invalid_target != NULL) {
        const char *msg;
        if (type == STAR_TARGETS || type == FOR_TARGETS) {
            msg = "cannot assign to %s";
        }
        else {
            msg = "cannot delete %s";
        }
        return RAISE_SYNTAX_ERROR_KNOWN_LOCATION(
            invalid_target,
            msg,
            _PyPegen_get_expr_name(invalid_target)
        );
        return RAISE_SYNTAX_ERROR_KNOWN_LOCATION(invalid_target, "invalid syntax");
    }
    return NULL;
}

// Action utility functions

expr_ty _PyPegen_dummy_name(Parser *p, ...);
void * _PyPegen_seq_last_item(asdl_seq *seq);
#define PyPegen_last_item(seq, type) ((type)_PyPegen_seq_last_item((asdl_seq*)seq))
void * _PyPegen_seq_first_item(asdl_seq *seq);
#define PyPegen_first_item(seq, type) ((type)_PyPegen_seq_first_item((asdl_seq*)seq))
#define UNUSED(expr) do { (void)(expr); } while (0)
#define EXTRA_EXPR(head, tail) head->lineno, (head)->col_offset, (tail)->end_lineno, (tail)->end_col_offset, _p->arena
#define EXTRA _start_lineno, _start_col_offset, _end_lineno, _end_col_offset, _p->arena
PyObject *_PyPegen_new_type_comment(Parser *, const char *);

Py_LOCAL_INLINE(PyObject *)
NEW_TYPE_COMMENT(Parser *p, Token *tc)
{
    if (tc == NULL) {
        return NULL;
    }
    const char *bytes = PyBytes_AsString(tc->bytes);
    if (bytes == NULL) {
        goto error;
    }
    PyObject *tco = _PyPegen_new_type_comment(p, bytes);
    if (tco == NULL) {
        goto error;
    }
    return tco;
 error:
    p->error_indicator = 1;  // Inline CHECK_CALL
    return NULL;
}

Py_LOCAL_INLINE(void *)
INVALID_VERSION_CHECK(Parser *p, int version, char *msg, void *node)
{
    if (node == NULL) {
        p->error_indicator = 1;  // Inline CHECK_CALL
        return NULL;
    }
    if (p->feature_version < version) {
        p->error_indicator = 1;
        return RAISE_SYNTAX_ERROR("%s only supported in Python 3.%i and greater",
                                  msg, version);
    }
    return node;
}

#define CHECK_VERSION(type, version, msg, node) ((type) INVALID_VERSION_CHECK(_p, version, msg, node))

arg_ty _PyPegen_add_type_comment_to_arg(Parser *, arg_ty, Token *);
PyObject *_PyPegen_new_identifier(Parser *, const char *);
asdl_seq *_PyPegen_singleton_seq(Parser *, void *);
asdl_seq *_PyPegen_seq_insert_in_front(Parser *, void *, asdl_seq *);
asdl_seq *_PyPegen_seq_append_to_end(Parser *, asdl_seq *, void *);
asdl_seq *_PyPegen_seq_flatten(Parser *, asdl_seq *);
expr_ty _PyPegen_join_names_with_dot(Parser *, expr_ty, expr_ty);
int _PyPegen_seq_count_dots(asdl_seq *);
alias_ty _PyPegen_alias_for_star(Parser *, int, int, int, int, PyArena *);
asdl_identifier_seq *_PyPegen_map_names_to_ids(Parser *, asdl_expr_seq *);
CmpopExprPair *_PyPegen_cmpop_expr_pair(Parser *, cmpop_ty, expr_ty);
asdl_int_seq *_PyPegen_get_cmpops(Parser *p, asdl_seq *);
asdl_expr_seq *_PyPegen_get_exprs(Parser *, asdl_seq *);
expr_ty _PyPegen_set_expr_context(Parser *, expr_ty, expr_context_ty);
KeyValuePair *_PyPegen_key_value_pair(Parser *, expr_ty, expr_ty);
asdl_expr_seq *_PyPegen_get_keys(Parser *, asdl_seq *);
asdl_expr_seq *_PyPegen_get_values(Parser *, asdl_seq *);
KeyPatternPair *_PyPegen_key_pattern_pair(Parser *, expr_ty, pattern_ty);
asdl_expr_seq *_PyPegen_get_pattern_keys(Parser *, asdl_seq *);
asdl_pattern_seq *_PyPegen_get_patterns(Parser *, asdl_seq *);
NameDefaultPair *_PyPegen_name_default_pair(Parser *, arg_ty, expr_ty, Token *);
SlashWithDefault *_PyPegen_slash_with_default(Parser *, asdl_arg_seq *, asdl_seq *);
StarEtc *_PyPegen_star_etc(Parser *, arg_ty, asdl_seq *, arg_ty);
arguments_ty _PyPegen_make_arguments(Parser *, asdl_arg_seq *, SlashWithDefault *,
                                     asdl_arg_seq *, asdl_seq *, StarEtc *);
arguments_ty _PyPegen_empty_arguments(Parser *);
AugOperator *_PyPegen_augoperator(Parser*, operator_ty type);
stmt_ty _PyPegen_function_def_decorators(Parser *, asdl_expr_seq *, stmt_ty);
stmt_ty _PyPegen_class_def_decorators(Parser *, asdl_expr_seq *, stmt_ty);
KeywordOrStarred *_PyPegen_keyword_or_starred(Parser *, void *, int);
asdl_expr_seq *_PyPegen_seq_extract_starred_exprs(Parser *, asdl_seq *);
asdl_keyword_seq *_PyPegen_seq_delete_starred_exprs(Parser *, asdl_seq *);
expr_ty _PyPegen_collect_call_seqs(Parser *, asdl_expr_seq *, asdl_seq *,
                     int lineno, int col_offset, int end_lineno,
                     int end_col_offset, PyArena *arena);
expr_ty _PyPegen_concatenate_strings(Parser *p, asdl_seq *);
expr_ty _PyPegen_ensure_imaginary(Parser *p, expr_ty);
expr_ty _PyPegen_ensure_real(Parser *p, expr_ty);
asdl_seq *_PyPegen_join_sequences(Parser *, asdl_seq *, asdl_seq *);
int _PyPegen_check_barry_as_flufl(Parser *, Token *);
int _PyPegen_check_legacy_stmt(Parser *p, expr_ty t);
mod_ty _PyPegen_make_module(Parser *, asdl_stmt_seq *);
void *_PyPegen_arguments_parsing_error(Parser *, expr_ty);
expr_ty _PyPegen_get_last_comprehension_item(comprehension_ty comprehension);
void *_PyPegen_nonparen_genexp_in_call(Parser *p, expr_ty args, asdl_comprehension_seq *comprehensions);

// Parser API

Parser *_PyPegen_Parser_New(struct tok_state *, int, int, int, int *, PyArena *);
void _PyPegen_Parser_Free(Parser *);
mod_ty _PyPegen_run_parser_from_file_pointer(FILE *, int, PyObject *, const char *,
                                    const char *, const char *, PyCompilerFlags *, int *, PyArena *);
void *_PyPegen_run_parser(Parser *);
mod_ty _PyPegen_run_parser_from_string(const char *, int, PyObject *, PyCompilerFlags *, PyArena *);
asdl_stmt_seq *_PyPegen_interactive_exit(Parser *);

// Generated function in parse.c - function definition in python.gram
void *_PyPegen_parse(Parser *);

#endif
