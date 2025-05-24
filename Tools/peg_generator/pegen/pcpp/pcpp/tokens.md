# Preprocessor Tokens

A **Preprocessor Token** is the embodiment of the entity `preprocessing-token`
in the Standard documents.  It is the "the minimal lexical element of the language in translation
phases 3 through 6".  
In phases 1 and 2, the contents of a source file are converted to the source
character set, trigraphs are substituted, and line splices are performed.  
Tokens may be separated by whitespace, including comments.  
General categories of tokens are:
- header-name.  Only in the context of an `#include` directive.
- identifier.
- pp-number.  This includes integer and floating point constants and other
  forms that are neither of these.
- character-constant.
- string-literal.
- punctuator.
- a universal character name that cannot be part of the above.
- a non-whitespace character that cannot be part of the above.

Tokens are produced by:
- Lexing the source file after phases 1 and 2.
- Performing a `##` operator while expanding a macro.  This will be a token
  which matches a string as though it is the entire content of a file.
- Performing a #` operator while expanding a macro.  It is a string-literal.
- A placemarker.  This is a special token produced when expanding a macro
  parameter whose value contains no tokens.  These are dropped at the end of
  phase 4.
## Data streams
### The `Source` class and source data
`Source` manages a single `#include` of a source file.  There is also a '< 
top level >' Source object which is built from the command line.
The term **source data** is a string which is the entire contents of the
source file.  If the actual file has an encoding other than the default
'UTF-8', the file data is decoded to produce the source data.
### The `PpLex` class and lexer data
This class performs lexical analysis of the contents of a Source.  Every
Source has its own PpLex object, referred to as its **lexer**.
The lexer takes the source data and makes certain **replacements** to produce the
**lexer data**, in the following order:
1. All trigraph sequences are replaced.  For example, '??>' is replaced with
   '}'.
2. All line splces are performed.  Any sequence '\' '\n' is removed.
3. Some **universal character name** sequences are replaced by a single
   character for the corresponding codepoint.  For example, '\u03b4' is
   replaced with 'δ'.

The lexer keeps track of all the replacements performed.  This enables it to:
- Determine the offset in the source data corresponding to a lexed token.
- Revert some or all of the replacements within a token's value.  Most
  notably, raw string tokens have the original source data as their value.
Lexical analysis is performed on the resulting lexer data.

## The `PpTok` class

This object is created by `pcpp` for each Preprocessor Token.  It is created
by:
- The lexer for a source file.
- A `#` or `##` operator during a macro expansion.
- An empty parameter during a macro expansion.
### Lifetime
Generally, token objects are short-lived.  If a token is not a macro name, it
is written directly to the output file.  If produced by a lexer, it will be
deleted the next time the lexer is asked for the next token.  All references
to the token are gone and the token is reclaimed.

If the token is in the replacement list for a macro (in a `#define`
directive), it is kept in the macro definition for the remainder of `pcpp`, or
until the macro is undefined.  
If the token is in the argument list of a function-like macro call, it is kept
as part of the corresponding argument.  This lasts for as long as that macro
call is being expanded, and then the call is dropped, along with all the
argument tokens.

### Type
Every token has a **type**.  This is stored in `tok.type`.  It is a member of
the enumeration `TokType`.
A `TokType` has several properties which can be tested while analyzing the
token.
Refer to [tokentype.py](tokentype.py)

*In the future, this will be incorporated into a particular subclass of `PpTok`
to reduce the memory footprint of the token.*
### Value
**Token.value** is a string associated with the token.  It is used when
writing it to the output file, and when part of a `#` or `##` operator.
The value is a computed property.  Hence it does not take up memory, and since
it is infrequently used, it doesn't have to be highly optimized.
#### Lexed value
For tokens which don't have a single fixed spelling, the token stores the
actual string (substring of the lexer data) which is matched.  
*In the future, the value may be computed from the range of offsets in the
lexer data.*
#### Reverted value
In some cases, some or all of the lexer's replacements of the source data will
be reverted.  In this case, the reverted value will be stored in the token.
Or if it corresponds to a substring of the source data, the range of offsets
can be stored in order to save on memory footprint.
#### Literals
A Literal Token is one whose value is fixed by its type.  `token.value` is a @property
whose value is `token.type.lit`.  All punctuators fall into this category.  *In
the future, whitespace types may have `type.lit` = " ".*
### Location
For a lexed token, the token identifies where it came from in its source file.
It is composed of:
- The source file.
- The offset in the source data.
- The physical line number.  This counts all newlines in the source data,
  including line splices.
- The logical line number.  This is the last physical line which was not
  spliced.
- The column number in the physical line.
- The presumed line numbers (physical and logical).
- The presumed filename.

The Location of a token is mainly used in writing the token to the output
file.  This affects `#line` directives and newlines that are written, and
leading whitespace at the start of a line.
#### Mover
A **Mover** (class `**TokLocMove**`) object tracks any changes in presumed line numbers and
presumed filename, within the source file.  It provides:
- The shift between line numbers and presumed line numbers.  Initially, this is 0.
- The presumed filename.  Initially, this isthe same as the source's filename.
When a `#line` directive is seen while lexing the source, a new Mover is
created.  The line shift is the line number in the directive minus the line
number following the directive.  The filename is the name given in the
directive, if any, or else the name in the current MoveR.

#### Position
The location is encapsulated in a *Position* value, which is simply an `int`
value.  This is a unique value for every *lexed* token (other tokens have a
Position of 0).  
Each source file owns a range of Position values, corresponding to the length
of the source data.  None of these ranges
overlap.  The preprocessor maintains a lookup table which quickly identifies
the source file for any given Position, and the offset into the source's data.  
The source file has two lookup tables:
- The positions of all the newlines in the source data.  This is used to
  determine the physical line number.  The logical line number is determined
  by finding line splices at or preceding the position.
- The positions of all `#line` directives and their Mover objects.  This is used to find which
  Mover is currently in effect.  It provides presumed line numbers (physical
  or logical) for the token line numbers, and the presumed filenme.

All of these Position lookups are optimized on the assumption that a lookup
will most likely give the same result object, or in the case of the newline
table, a neighboring line number.
These are checked first, before resorting to a binary search.

From a Position, the Preprocessor can compute:
- The Source object.
- The offset of the token's value in the source data.
- The physical line number and the offset of the newline in the source data.
- The logical line number (given the number of line splices preceding the
  token, which is usually 0).
- The column number.
- The presumed line numbers and filename.
- The original spelling of the token, given the Position of the end of the token.

### Output file spacing
The preprocessor has to decide what whitespace, if any, to write to the output file preceding the
value of any token.
Possibilities are:
- A newline or a `#line` directive followed possibly by one or more spaces
    (as indicated by the token's column number)
- A single space.
- Nothing.

The attribute **`token.spacing`** indictes where the token is located
relative to the preceding token.  Possible values are:
- Adjacent.  It is immediately after the preceding token.  This is the
  `PpTok.spacing` class variable.
- Line.  It is the first token in a logical line, except if it is lexed while
  scanning the argument list of a function-like macro.  The token may have a column number.
- Space.  Preceded by one or more whitespace characters but not a Line
  separation.

When expanding a macro or substituting a macro parameter:

- The token for the macro or parameter name is known as the **original
  token**, and is set in **`token.orig`**.
- The Preprocessor will use `token(.orig)+.spacing` instead of
  `token.spacing`, for as many levels of `.orig` that are present.

The Preprocessor uses these rules to write whitespace before a token:
- If `token.spacing` is Line, it writes newline characters, or a `#line`
  directive to bring the output file to the desired presumed line number.
  Then it writes space characters if the column number is > 1.
- If `token.spacing` is Space, it writes a single space. 
- If `token.spacing` is Adjacent, then:
    - Designate LHS and RHS as the previous written token and the token, *resp*.
    - The lexer will have set `RHS.prev` = the previous token in the source
      file.
    - If LHS.prev is RHS, then do nothing.  No spacing is needed to
      distinguisn the separate tokens.
    - Examine LHS.type.sep_from[RHS.type], if it exists.  If it does not
      exist, then do nothing.
    - If it is false, then write a space.
    - Otherwise it is a function, which is called with LHS, RHS, and the token
      written before LHS.  If this returns true, then write a space, else do
      nothing.
    - Note, the reason for the token before LHS is that if we have the tokens
      '.', '.', '.', then the output will already have '..', and following it
      with another '.' will produce an Ellipsis '...', so a space is required.

When a token has no spacing flag, then the preprocessor uses these rules:
- If the token is the next token in the same source file as the previous token
  written, then no spacing is needed, since they appear next to each other in
  the source file.
- Otherwise, the previous one or two tokens written are concatenated with the
  new token to see if lexing this string will produce the same tokens.  If so,
  a space is written before the new token.  
  Note, going back two tokens is needed in the case of writing the tokens `.`
  `.` `.`.  Lexing '..' would be no problem, but lexing '...' would be an
  Ellipsis token.

As an optimization, the paste avoidance uses information in the types of the
tokens involved.  This can indicate that spacing is or is not required, in
which case the tokens don't have to be concatenated and lexed.  Otherwise, the
slower concatenation test is performed.

### Hide set
A token has a **hide set**.  This comes from D. Prosser's [algorithm](https://www.spinellis.gr/blog/20060626/cpp.algo.pdf) for macro
expansion, and used to prevent expansion of the token as a macro in certain
recursive situations.
This is a `PpTok` class variable, whose value is an empty set.  
If the token came from an expansion of a macro, it is an instance variable.
It is a set with the name of that macro, plus the hide set associated with the
macro name token itself.
While expanding any macro, if a token is seen whose hide set contains the name
of that macro, then the macro is not expanded.
