#ifndef PEGEN_H
#define PEGEN_H
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <token.h>
#include <pycore_ast.h>
#include "stdbool.h"
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
    _Bool cut_indicator;

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

#define _PyPegen_PARSE_REF(_ppRes, type) \
        *(type *) (*(_ppRes)).addr

#define _PyPegen_PARSE_REF_UNTYPED(ppRes) \
    _PyPegen_PARSE_REF(ppRes, void *)

#define _PyPegen_PARSE_COPY_FROM(dst, ppRes) \
    memcpy(*(dst), (ppRes)->addr, (ppRes)->size);

#define _PyPegen_PARSE_COPY_TO(ppRes, src) \
    memcpy((ppRes)->addr, *(src), (ppRes)->size);

// Return a successful parse result via a result pointer, and return true as ParseStatus.
// The result pointer is optional.  If NULL, the type and src are not evaluated.
//  'type' is a declarator for "_type__return".

#define _PyPegen_RETURN(_ppRes, src, type) {                    \
    if (_ppRes) {                                                \
        _PyPegen_TYPEDEF(type); \
        /*assert ((_ppRes)->size == sizeof(_type__return)); */                \
        _PyPegen_PARSE_REF((_ppRes), _type__return) = (_type__return) (src);    \
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

// ... for any variable, either object or function type.
//      'name' is the name of the variable.
//      'type' is a declarator for the type of the variable, and
//      must be named "_type_{name}"
#define _PyPegen_DECL_VAR(name, type) \
    _PyPegen_TYPEDEF(type); \
    _type_ ## name name; \
    ParseResultPtr _ptr_ ## name = _PyPegen_PARSERESULTPTR(name) \

// ... for anonymous VarItem.  Args same as _PyPegen_DECL_VAR, above.
//      
#define _PyPegen_DECL_LOCAL(name, type) \
    _PyPegen_DECL_VAR(name, type) \

// ... for global variable inherited from calling function.
//      Loads the value from the local environment.
//      Args same as _PyPegen_DECL_VAR, above.

#define _PyPegen_GET_GLOBAL(index, name, type) \
    _PyPegen_DECL_VAR(name, type); \
    name = * (_type_ ## name *) _p->local_values[index] \

// ... for new global variable.
//      Puts pointer to variable in the local environment.
//      Args same as _PyPegen_DECL_VAR, above.
#define _PyPegen_ADD_GLOBAL(index, name, type) \
    _PyPegen_DECL_VAR(name, type); \
    /** (_type_ ## name *) _p->local_values[index] = (_type_ ## name) &name*/ \
    * (_type_ ## name *) _p->local_values[index] = name \


// ... for a pointer to the variable, to use as a ppRes argument to a parse function.
#define _PyPegen_PARSE_RESULT_PTR(name) \
    (& _ptr_ ## name)

// ... type expression for pointer to any object type.
#define _PyPegen_TYPE_0(type, ...) \
    type *

// ... type expression for pointer to any function type.
//      `params` is list of parameters in parens.
#define _PyPegen_TYPE_1(type, params) \
    type (**) params

#define _PyPegen_TYPE(type, params) \
    _PyPegen_CONCAT(_PyPegen_TYPE_, _PyPegen_HAS_PARAMS(params)) (type, params)

// ... declare a typedef.
//      'decl' is any declarator, using the type name as the variable name.
//      The type name can then be used as a 'type' argument to other macros.
#define _PyPegen_TYPEDEF(decl) \
    typedef decl


#if MACRO_HELPERS_TEST
// Test the above macros...

// Skip to here.
#define XX_1(n, t, _p) "XX_1"(n, t, _p)

__LINE__;
// _PyPegen_SEL_IF_PARAMS(XX, (int, name), (int *, const int *));
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS(XX, (int, name), (int *, const int *)) ));
   (_PyPegen_SEL_IF_PARAMS(XX, (int, name), (int *, const int *)) );

__LINE__;
// _PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (name, int), (int *, const int *));
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (name, int), (int *, const int *)) ));

__LINE__;
// _PyPegen_DECL_LOCAL_PARSE_RESULT(name, int, (int *, const int *));
   _PyPegen_STR((_PyPegen_DECL_LOCAL_PARSE_RESULT(name, int, (int *, const int *)) ));

__LINE__;
// _PyPegen_SEL_IF_PARAMS(XX, (int, name), );
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS(XX, (int, name), ) ));

__LINE__;
// _PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (name, int), );
   _PyPegen_STR((_PyPegen_SEL_IF_PARAMS(_PyPegen_DECL_VAR, (name, int), ) ));

__LINE__;
// _PyPegen_DECL_LOCAL_PARSE_RESULT(name, int, );
   _PyPegen_STR((_PyPegen_DECL_LOCAL_PARSE_RESULT(name, int, ) ));

// _PyPegen_GET_GLOBAL_PARSE_RESULT(name, int, (int *, const int *), 42);
   _PyPegen_STR(_PyPegen_GET_GLOBAL_PARSE_RESULT(name, int, (int *, const int *), 42) );

// _PyPegen_GET_GLOBAL_PARSE_RESULT(name, int, , 42);
   _PyPegen_STR(_PyPegen_GET_GLOBAL_PARSE_RESULT(name, int, , 42) );

// _PyPegen_ADD_GLOBAL_PARSE_RESULT(42, name, int, (int *, const int *));
   _PyPegen_STR(_PyPegen_ADD_GLOBAL_PARSE_RESULT(42, name, int, (int *, const int *)) );

// _PyPegen_ADD_GLOBAL_PARSE_RESULT(42, name, int, );
   _PyPegen_STR(_PyPegen_ADD_GLOBAL_PARSE_RESULT(42, name, int, ) );

// Test some variable declaration macros...

    _PyPegen_STR((_PyPegen_DECL_LOCAL_PARSE_RESULT(name, int, (int *, const int *)) ));

    _PyPegen_STR((_PyPegen_DECL_LOCAL_PARSE_RESULT(name, int, ) ));

    _PyPegen_STR((_PyPegen_GET_GLOBAL_PARSE_RESULT(name, int, (int *, const int *), 42) ));

    _PyPegen_STR((_PyPegen_GET_GLOBAL_PARSE_RESULT(name, int, , 42) ));

    _PyPegen_STR((_PyPegen_ADD_GLOBAL_PARSE_RESULT(42, name, int, (int *, const int *)) ));

    _PyPegen_STR((_PyPegen_ADD_GLOBAL_PARSE_RESULT(42, name, int, ) ));

    _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (_PyPegen_TYPE, (int), ) ));

    _PyPegen_STR((_PyPegen_SEL_IF_PARAMS (_PyPegen_TYPE, (int), (int *, const int *)) ));

#endif // MACRO_HELPERS_TEST
#if MACRO_HELPERS_TEST
}
#endif

// Typedefs for callback parsing functions...
//  ... Returns status, and result via pointer.
typedef ParseStatus (ParseFunc)(Parser * _p, ParseResultPtr * _ppRes);
//  ... Returns just result via pointer.
typedef void (ParseTrue)(Parser * _p, ParseResultPtr * _ppRes);
//  ... Returns only status.
typedef ParseStatus (ParseTest)(Parser * _p);
//  ... Returns neither, but still uses the parser.
typedef void (ParseVoid)(Parser * _p);

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

typedef struct RuleAltDescr {
    ParseFunc * parse;
    const char* name;
    const char* expr;
} RuleAltDescr;

// Static information about a Rule being parsed.
struct RuleDescr {
    ParseFunc * rhs;        // Parses the ordered choice of Alts.
    const char* name;       // For debugging.
    const char* expr;       // For debugging.
    // Used if the rule is memoized, default initialization otherwise ...
    int memo;               // This is the key corresponding to the rule.
    size_t result_size;     // Size of the result to store in the Memo.
    // Specify the Alts ...
    size_t num_alts;
    const RuleAltDescr * alts;
};

#define _PyPegen_RULE_DESCR(name, rule, result_type, expr)  \
    static RuleDescr name = {                           \
        _ ## rule ## __rhs, # rule, expr,               \
        rule ## _type, sizeof(result_type)              \
    };

// Macros to forward declare and define a Rule.
// The definition is only the name and parameters, and none of
// the function body.
// The rule parameters, if any, are optional and are enclosed in parentheses as ...

#define _PyPegen_RULE(rule, ...)  \
    _PyPegen_NAME_IF_ARGS(_PyPegen_RULE2, _FUN, __VA_ARGS__) (rule, __VA_ARGS__) \

#define _PyPegen_RULE2(rule, ...)  \
    ParseStatus _ ## rule ## _rule(Parser* _p, ParseResultPtr* _ppRes) \

#define _PyPegen_RULE2_FUN(rule, params)  \
    ParseStatus _ ## rule ## _rule(Parser* _p, ParseResultPtr* _ppRes, \
        _PyPegen_UNGROUP params) \

#define _PyPegen_DECL_RULE(rule, ...)  \
    static _PyPegen_NAME_IF_ARGS(_PyPegen_DECL_RULE2, _FUN, __VA_ARGS__) (rule, __VA_ARGS__); \

#define _PyPegen_DECL_RULE2(rule, ...)  \
    ParseStatus _ ## rule ## _rule(Parser* _p, ParseResultPtr* _ppRes) \

#define _PyPegen_DECL_RULE2_FUN(rule, params)  \
    ParseStatus _ ## rule ## _rule(Parser* _p, ParseResultPtr* _ppRes, \
        _PyPegen_UNGROUP params) \

// Dynamic information about the Rule being parsed.
// This is kept in the Parser and saved/restored on entering/leaving the rule.
struct RuleInfo {
    const RuleDescr * descr;        // Which Rule.
    void ** local_vars;             // Local variables visible within the rule.
};

#define _PyPegen_ALT_DESCR(name, expr) {name, #name, expr}

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
ParseStatus _PyPegen_parse_rule(Parser* _p, ParseResultPtr * ppRes, RuleDescr* rule, void * local_vars[]);

//  ... Memoized Rule, not a left-recursive group leader.
ParseStatus _PyPegen_parse_memo_rule(Parser* _p, ParseResultPtr * ppRes, RuleDescr* rule, void* local_vars[]);

//  ... Left-recursive group leader Rule.
ParseStatus _PyPegen_parse_recursive_rule(Parser* _p, ParseResultPtr * ppRes, RuleDescr* rule, void* local_vars[]);

// Helper function to parse a rule alternative, returning the parse result.
// If the parse fails, false is returned and the mark is left unchanged.
ParseStatus _PyPegen_parse_alt(Parser *_p, ParseResultPtr * ppRes, const RuleAltDescr *alt);

#define _PyPegen_RETURN_PARSE_ALTS(...) \
    RuleAltDescr _alts[] = { __VA_ARGS__ }; \
    return _PyPegen_parse_alts(_p, _ppRes, _alts, sizeof _alts / sizeof (RuleAltDescr)); \


// Helper function to parse several rule alternatives, returning the first successful parse result.
// However, if an alternative sets _p->cut_indicator, no more alternatives are tried.
// If the parse fails, false is returned and the mark is left unchanged.
// All the alternatives have to agree on their result type.
ParseStatus _PyPegen_parse_alts(Parser *_p, ParseResultPtr * ppRes, const RuleAltDescr alts[], int count);

#define _PyPegen_PARSE_ALTS() \


// Macro to parse several rule alternatives, given an array of alternative descriptors.
#define _PyPegen_PARSE_ALT_ARRAY(_p, ppres, alts) \
    _PyPegen_parse_alts(_p, ppres, alts, sizeof alts / sizeof alts[0]);

// Parse an item some number of times, returning an asdl_seq * with the parsed results as elements.
// Given a minimum and maximum number of items.  If the minimum number is not reached, the parse fails.
// If the maximum number is reached, no more items are attempted.  max = 0 means no limit.
// Optionally provide a separator item, which must be parsed between consecutive items.
ParseStatus _PyPegen_parse_seq(Parser *_p, ParseResultPtr * ppRes,
    ParseFunc item, ParseFunc * sep, int min, int max);

// Variations of the above, for different combinations of parameters.
//
//  ... Optionally parse an item, always succeeds.  Return value has either 0 or 1 elements.
Py_LOCAL_INLINE(ParseStatus)
_PyPegen_parse_opt(Parser * _p, ParseResultPtr * ppRes, ParseFunc item) {
    _PyPegen_parse_seq(_p, ppRes, item, NULL, 0, 1);
}

//  ... Parse an item as many times as possible.  Optionally require at least one item.
// Parse an item several times, returning an asdl_seq * with the parsed results as elements.
Py_LOCAL_INLINE(ParseStatus)
_PyPegen_parse_repeat(Parser * _p, ParseResultPtr * ppRes, ParseFunc item, int repeat1) {
    _PyPegen_parse_seq(_p, ppRes, item, NULL, repeat1, 0);
}

//  ... Parse an item as many times as possible, with a given separator item, which must be parsed between consecutive items.
//      Optionally require at least one item.
Py_LOCAL_INLINE(ParseStatus)
_PyPegen_parse_gather(Parser * _p, ParseResultPtr * ppRes, ParseFunc item, ParseFunc sep, int repeat1) {
    _PyPegen_parse_seq(_p, ppRes, item, sep, repeat1, 0);
}

// Test parsing an item without consuming any input or computing any value.  Does not return any value.
// The parse succeeds if the item parse success matches the given success value.
ParseStatus _PyPegen_parse_lookahead(Parser *, ParseStatus positive, ParseFunc item);

// Parse an item, requiring it to succeed.  If the item parse fails, raise a SyntaxError.
// If the function returns, then the parse succeeded.  No status is returned.
void _PyPegen_parse_forced(Parser *_p, ParseResultPtr * ppRes, ParseFunc item, const char* expected);

// Parse a NAME Token.  Result value is an AST expr_ty object with Name_kind.
ParseStatus _PyPegen_parse_name(Parser *_p, ParseResultPtr * ppRes);

// Parse a specific keyword.
ParseStatus _PyPegen_parse_keyword(Parser *_p, ParseResultPtr * ppRes, int type);

// Parse any soft keyword or just a specific soft keyword.
// Same as _PyPegen_parse_name(), but fails if the name is not an actual soft keyword, or the given keyword. 
ParseStatus _PyPegen_parse_any_soft_keyword(Parser *_p, ParseResultPtr * ppRes);
ParseStatus _PyPegen_parse_soft_keyword(Parser *_p, ParseResultPtr * ppRes, const char * keyword);

// Parse a NUMBER Token.  Result value is an AST expr_ty object with Constant_kind.  Its value is
//  the appropriate numeric Python object.
ParseStatus _PyPegen_parse_number(Parser *_p, ParseResultPtr * ppRes);

// Parse a STRING Token.  Result value is a Token *.
ParseStatus _PyPegen_parse_string(Parser *_p, ParseResultPtr * ppRes);

// Parse a given type of Token.  Result value is a Token *.
ParseStatus _PyPegen_parse_token(Parser *_p, ParseResultPtr * ppRes, int type);

// Parse an OP Token.  Result value is a Token *.
ParseStatus _PyPegen_parse_op(Parser *_p, ParseResultPtr * ppRes);

// Parse a CHAR Token.  Result value is a Token *.
ParseStatus _PyPegen_parse_char(Parser *_p, ParseResultPtr * ppRes);

// Parse an OP Token with given character as its string.  Result value is a Token *.
ParseStatus _PyPegen_parse_specific_char(Parser *_p, ParseResultPtr * ppRes, char c);

// Parse a Call expression.  Given a node (as a ParseFunc) and arguments.
// The node function returns a function, which is called with the args.
// The type of this function is given by a type (which is also the result type of the Call),
// and parameters (enclosed in parentheses).
// Fails if the node parse fails.
#define _PyPegen_CALL(_p, _ppRes, func, type, params, ...) \
    _PyPegen_DECL_VAR(_func, type (*_type__func) params); \
    if (!func(_p, &_ptr__func)) return false; \
    _PyPegen_RETURN(_ppRes, _func(__VA_ARGS__), type _type__return) \


// Parse a Cut expression.  This doesn't parse anything of the input, but just
//  sets an indication that if the current Alt faile, then no more Alts will be tried.
void _PyPegen_parse_cut(Parser *_p);

int _PyPegen_lookahead_with_name(int, expr_ty (func)(Parser *), Parser *);
int _PyPegen_lookahead_with_int(int, Token *(func)(Parser *, int), Parser *, int);
int _PyPegen_lookahead_with_string(int , expr_ty (func)(Parser *, const char*), Parser *, const char*);
int _PyPegen_lookahead(int, void *(func)(Parser *), Parser *);


Token * _PyPegen_expect_token(Parser *_p, int type);
Token *_PyPegen_expect_char(Parser *_p, char c);
void* _PyPegen_expect_forced_result(Parser *_p, void* result, const char* expected);
Token *_PyPegen_expect_forced_token(Parser *_p, int type, const char* expected);
expr_ty _PyPegen_expect_soft_keyword(Parser *_p, const char *keyword);
ParseStatus _PyPegen_group(Parser *_p, ParseResultPtr * ppRes, ParseFunc rhs);
expr_ty _PyPegen_soft_keyword_token(Parser *_p);
Token *_PyPegen_get_last_nonnwhitespace_token(Parser *_p);
int _PyPegen_fill_token(Parser *_p);
expr_ty _PyPegen_name_token(Parser *_p);
expr_ty _PyPegen_number_token(Parser *_p);
Token *_PyPegen_string_token(Parser *_p);
ParseFunc _PyPegen_parse_type_comment;
Py_ssize_t _PyPegen_byte_offset_to_character_offset(PyObject *line, Py_ssize_t col_offset);

// Getting input file locations, as part of having EXTRA appear in action...
// At start of the Alt, declare local variables and set the starting location.
#define _PyPegen_EXTRA_START(_p) \
    int _start_lineno, _start_col_offset, _end_lineno, _end_col_offset; \
    _PyPegen_location_start(_p, &_start_lineno, &_start_col_offset); \

// At end of the Alt, if successful, set th ending location.
#define _PyPegen_EXTRA_END(_p) \
    _PyPegen_location_end(_p, &_end_lineno, &_end_col_offset); \

// Helper functions used by the macros.
int _PyPegen_location_start(Parser* _p, int * lineno, int * col_offset);   // Current location in input.
int _PyPegen_location_end(Parser* _p, int * lineno, int * col_offset);     // Current location in input, but backs up over whitespace.

// Error handling functions and APIs
typedef enum {
    STAR_TARGETS,
    DEL_TARGETS,
    FOR_TARGETS
} TARGETS_TYPE;

int _Pypegen_raise_decode_error(Parser *_p);
void _PyPegen_raise_tokenizer_init_error(PyObject *filename);
int _Pypegen_tokenizer_error(Parser *_p);
void *_PyPegen_raise_error(Parser *_p, PyObject *errtype, const char *errmsg, ...);
void *_PyPegen_raise_error_known_location(Parser *_p, PyObject *errtype,
                                          Py_ssize_t lineno, Py_ssize_t col_offset,
                                          Py_ssize_t end_lineno, Py_ssize_t end_col_offset,
                                          const char *errmsg, va_list va);
void _Pypegen_set_syntax_error(Parser* _p, Token* last_token);
Py_LOCAL_INLINE(void *)
RAISE_ERROR_KNOWN_LOCATION(Parser *_p, PyObject *errtype,
                           Py_ssize_t lineno, Py_ssize_t col_offset,
                           Py_ssize_t end_lineno, Py_ssize_t end_col_offset,
                           const char *errmsg, ...)
{
    va_list va;
    va_start(va, errmsg);
    Py_ssize_t _col_offset = (col_offset == CURRENT_POS ? CURRENT_POS : col_offset + 1);
    Py_ssize_t _end_col_offset = (end_col_offset == CURRENT_POS ? CURRENT_POS : end_col_offset + 1);
    _PyPegen_raise_error_known_location(_p, errtype, lineno, _col_offset, end_lineno, _end_col_offset, errmsg, va);
    va_end(va);
    return NULL;
}
#define RAISE_SYNTAX_ERROR(msg, ...) _PyPegen_raise_error(_p, PyExc_SyntaxError, msg, ##__VA_ARGS__)
#define RAISE_INDENTATION_ERROR(msg, ...) _PyPegen_raise_error(_p, PyExc_IndentationError, msg, ##__VA_ARGS__)
#define RAISE_SYNTAX_ERROR_KNOWN_RANGE(a, b, msg, ...) \
    RAISE_ERROR_KNOWN_LOCATION(_p, PyExc_SyntaxError, (a)->lineno, (a)->col_offset, (b)->end_lineno, (b)->end_col_offset, msg, ##__VA_ARGS__)
#define RAISE_SYNTAX_ERROR_KNOWN_LOCATION(a, msg, ...) \
    RAISE_ERROR_KNOWN_LOCATION(_p, PyExc_SyntaxError, (a)->lineno, (a)->col_offset, (a)->end_lineno, (a)->end_col_offset, msg, ##__VA_ARGS__)
#define RAISE_SYNTAX_ERROR_STARTING_FROM(a, msg, ...) \
    RAISE_ERROR_KNOWN_LOCATION(_p, PyExc_SyntaxError, (a)->lineno, (a)->col_offset, CURRENT_POS, CURRENT_POS, msg, ##__VA_ARGS__)
#define RAISE_SYNTAX_ERROR_INVALID_TARGET(type, e) _RAISE_SYNTAX_ERROR_INVALID_TARGET(_p, type, e)

Py_LOCAL_INLINE(void)
CHECK_CALL(Parser *_p, void *result)
{
    if (result == NULL) {
        assert(PyErr_Occurred());
        _p->error_indicator = 1;
    }
}

/* This is needed for helper functions that are allowed to
   return NULL without an error. Example: _PyPegen_seq_extract_starred_exprs */
Py_LOCAL_INLINE(void *)
CHECK_CALL_NULL_ALLOWED(Parser *_p, void *result)
{
    if (result == NULL && PyErr_Occurred()) {
        _p->error_indicator = 1;
    }
    return result;
}

#define CHECK(type_unused, result) (CHECK_CALL(_p, result), result)
#define _PyPegen_CHECK(result) (CHECK_CALL(_p, result), result)
#define CHECK_NULL_ALLOWED(type, result) ((type) CHECK_CALL_NULL_ALLOWED(_p, result))

expr_ty _PyPegen_get_invalid_target(expr_ty e, TARGETS_TYPE targets_type);
const char *_PyPegen_get_expr_name(expr_ty);
Py_LOCAL_INLINE(void *)
_RAISE_SYNTAX_ERROR_INVALID_TARGET(Parser *_p, TARGETS_TYPE type, void *e)
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

typedef expr_ty dummy_ty;
typedef void * circ_ty;
dummy_ty _PyPegen_dummy_name(Parser *_p, ...);
void *  _PyPegen_seq_last_item(asdl_seq *seq);
#define PyPegen_last_item(seq, type) ((type)_PyPegen_seq_last_item((asdl_seq*)seq))
void *  _PyPegen_seq_first_item(asdl_seq *seq);
#define PyPegen_first_item(seq, type) ((type)_PyPegen_seq_first_item((asdl_seq*)seq))
#define UNUSED(expr) do { (void)(expr); } while (0)
#define EXTRA_EXPR(head, tail) head->lineno, (head)->col_offset, (tail)->end_lineno, (tail)->end_col_offset, _p->arena
#define EXTRA _start_lineno, _start_col_offset, _end_lineno, _end_col_offset, _p->arena
PyObject *_PyPegen_new_type_comment(Parser *, const char *);

Py_LOCAL_INLINE(PyObject *)
NEW_TYPE_COMMENT(Parser *_p, Token *tc)
{
    if (tc == NULL) {
        return NULL;
    }
    const char *bytes = PyBytes_AsString(tc->bytes);
    if (bytes == NULL) {
        goto error;
    }
    PyObject *tco = _PyPegen_new_type_comment(_p, bytes);
    if (tco == NULL) {
        goto error;
    }
    return tco;
 error:
    _p->error_indicator = 1;  // Inline CHECK_CALL
    return NULL;
}

Py_LOCAL_INLINE(void *)
INVALID_VERSION_CHECK(Parser *_p, int version, char *msg, void *node)
{
    if (node == NULL) {
        _p->error_indicator = 1;  // Inline CHECK_CALL
        return NULL;
    }
    if (_p->feature_version < version) {
        _p->error_indicator = 1;
        return RAISE_SYNTAX_ERROR("%s only supported in Python 3.%i and greater",
                                  msg, version);
    }
    return node;
}

#define CHECK_VERSION(type, version, msg, node) ((type) INVALID_VERSION_CHECK(_p, version, msg, node))

arg_ty _PyPegen_add_type_comment_to_arg(Parser *, arg_ty, Token *);
PyObject *_PyPegen_new_identifier(Parser *, const char *);
asdl_seq *_PyPegen_singleton_seq(Parser *, void *);
#define _PyPegen_SINGLETON_SEQ(type, element) \
    ((asdl_##type##_seq *) _PyPegen_singleton_seq (_p, (type##_ty) (element)))

asdl_seq *_PyPegen_seq_insert_in_front(Parser *, void *, asdl_seq *);
#define _PyPegen_SEQ_INSERT_IN_FRONT(type, element, seq) \
    ((asdl_##type##_seq *) _PyPegen_seq_insert_in_front (_p, (type##_ty) (element), (asdl_seq *) (seq)))
asdl_seq *_PyPegen_seq_append_to_end(Parser *, asdl_seq *, void *);
asdl_seq *_PyPegen_seq_flatten(Parser *, asdl_seq *);
expr_ty _PyPegen_join_names_with_dot(Parser *, expr_ty, expr_ty);
int _PyPegen_seq_count_dots(asdl_seq *);
alias_ty _PyPegen_alias_for_star(Parser *, int, int, int, int, PyArena *);
asdl_identifier_seq *_PyPegen_map_names_to_ids(Parser *, asdl_expr_seq *);
CmpopExprPair *_PyPegen_cmpop_expr_pair(Parser *, cmpop_ty, expr_ty);
asdl_int_seq *_PyPegen_get_cmpops(Parser *_p, asdl_seq *);
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
expr_ty _PyPegen_concatenate_strings(Parser *_p, asdl_seq *);
expr_ty _PyPegen_ensure_imaginary(Parser *_p, expr_ty);
expr_ty _PyPegen_ensure_real(Parser *_p, expr_ty);
asdl_seq *_PyPegen_join_sequences(Parser *, asdl_seq *, asdl_seq *);
int _PyPegen_check_barry_as_flufl(Parser *, Token *);
int _PyPegen_check_legacy_stmt(Parser *_p, expr_ty t);
mod_ty _PyPegen_make_module(Parser *, asdl_stmt_seq *);
void *_PyPegen_arguments_parsing_error(Parser *, expr_ty);
expr_ty _PyPegen_get_last_comprehension_item(comprehension_ty comprehension);
void *_PyPegen_nonparen_genexp_in_call(Parser *_p, expr_ty args, asdl_comprehension_seq *comprehensions);

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
