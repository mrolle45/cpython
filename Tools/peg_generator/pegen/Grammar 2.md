# Grammars and their Parsing

A **Grammar** is a Python object which describes the structure of an 
**input file**, in order to translate (*parse*) it into a result.

The Grammar is created by a *metaparser* from a *grammar file*.  This is given to a *parser generator* which produces a parser.

There are two kinds of parsing.

- The parser is a C program.  The Grammar comes from a *python grammar* file.  The parser produces a syntax tree of AST objects described in [pycore_ast.h](../../../Include/internal/pycore_ast.h).  How this syntax tree is further used is beyond the scope of this document.

- The parser is a Python program which is generated from the Grammar.  The Grammar comes from a *metagrammar* file.  It produces a Grammar, which is then used to create a new metaparser.  If the process is repeated, using the new metaparser, the result is unchanged.

## Steps in the Parse Process

1. Optionally create the [metaparser](grammar_parser.py) from the [metagrammar file](metagrammar.gram).

    This happens once, while building the Python release.  The metagrammar and the metaparser both part of the Python repo.

    The metaparser is not created in a normal Python build.  However, if the metagrammar file is modified, then the metaparser is regenerated to match it.  This reads the metagrammar file and writes the new metaparser.

    However, it uses the *existing* metaparser in the process, to create a Grammar which represents the metagrammar file.  As a double check, the generator is run twice.

    1. It creates the new metaparser as an internal string.  This uses the *existing* metaparser to create a Grammar,  Then the generator creates the metaparser from the Grammar..
    2. It creates another metaparser string, but this time using the new metaparser string instead of the existing metaparser module.

    If the two strings match, then the new metaparser is written to the repo.

    Otherwise, an error message is written and the metaparser is *not* written back to the repo.

    Each of the above steps consists of:
    - Running the existing or new metaparser on the metagrammar, to produce a Grammar object.

    - Running the [Python generator](python_generator.py) with this Grammar and producing a string for the metaparser.


2. Optionally create the [C parser](../../../Parser/parser.c) from the [Python grammar](../../../Grammar/python.gram).

    This happens once, while building the Python release.  The grammar and the parser are both part of the Python repo.

    The parser is not created in a normal Python build.  However, if the grammar file is modified, then the parser is regenerated to match it.

    The regeneration consists of:
    - Running the metaparser on the grammar, to produce a Grammar object.

    - Running the [C generator](c_generator.py) with this Grammar and producing a string for the parser.

    - Writing this string to the parser file.

3. Run the C parser.  This reads the input file and creates a syntax tree.

## Grammar File

A **grammar file** is a text file which defines the structure (*i.e.*, the grammar) of an input file which is to be parsed.

The format for the file is based on PEG (Parse Expression Grammar) grammar format, described generally on this [Wikipedia page](https://en.wikipedia.org/wiki/Parsing_expression_grammar).  There are several changes and enhancements, notably:
  - A Rule is defined using a **`:`** rather than a backarrow **`←`** symbol.
  - Standard Python token structure.  Comments are ignored.  Any matching `()` or `[]` or `{}` (not within a string or a comment) can span multiple lines and newlines and indentation are ignored.  Other than that, newlines and indentation are significant (resulting in NEWLINE, INDENT, and DEDENT tokens).
  - Formatting of Rules with newlines and indents to allow them to carry over several lines in the same manner as compound Python statements.
  - The expression ordered choice operator is changed from **`/`** to **`|`**.
  - Several new parse expresions.
  - Optional calling parameters for nonterminals.  

## Tokens

A parse input file is **tokenized** using the standard Python [tokenizer](https://docs.python.org/3/library/tokenize.html), which turns it into a sequence of **Token**s.  The index of each Token in the sequence is called the **mark**.

This applies equally to a grammar file because it is parsed by the metaparser.

However, the Token sequence is filtered, to remove comments, blank lines, and line breaks (within parenthesized text only).  Unmatched quote characters raise a SyntaxError.

The remaining Tokens are as follows:

 Type       | Description
 ----       | ---- 
 NAME       | Python identifier
 STRING     | Python string, single- or triple-quoted
 NUMBER     | Python number, real or imaginary, not complex
 NEWLINE    | End of logical line
 INDENT     | Longer whitespace at start of logical line
 DEDENT     | Shorter whitespace at start of logical line
 OP         | Any 1-, 2-, or 3-character operator
 CHAR       | Any single character otherwise
 ENDMARKER  | End of the file

## Parse Expressions

A **parse expression** is a contiguous portion of a grammar file which designates a parse operation.  The parse operation occurs while parsing an input file and occurs at the current location (or **mark**).


Type        | Syntax | Class | Notes
----        | ---- | ----
1 rule      | name memo ? ":" alts | Rule | memo is optional
2 alts      | alt "\|" alt ... | Rhs | 1 or more
3 alt       | item item ... action ? | Alt | 0 or more, action optional
4 item      | atom          |
...         | name "=" item | VarItem
...         | "&" "&" item | Forced | Takes precedence over "&" item
...         | "&" item | PositiveLookahead
...         | "!" item | NegativeLookahead
...         | "~" | Cut
5 atom      | primary
...         | atom "?" | Opt
...         | atom "*" | Repeat0
...         | atom "+" | Repeat1
...         | atom "." atom "*" | Gather0
...         | atom "." atom "+" | Gather1
6 primary   | "(" alts ")" type ? | Group       | type is optional
...         | "[" alts "]" type ? | OptGroup    | type is optional
...         | terminal          | NameLeaf      | 
...         | nonterminal arglist ?    | NameLeaf | name of rule, rule param, or variable<br>arglist is optional [^args-params-opt]
...         | strliteral    | StringLeaf
**Non-parsing...** |
grammar   | meta * rule + ENDMARKER | Grammar   | Entire grammar file<br>0 or more metas<br>1 or more rules
memo        | "(" "memo" ")" ?    | str | generates memoization for the rule
name        | nonterminal type ?   | TypedName | type is optional
type        | annotation ? params ? | ValueType | annotation and params are optional[^args-params-opt]
annotation  | "[" target string[^targ-str] "]" | str | name of a target *object* type
params      | "(" param ... ")" | Params | 0 or more
param       | nonterminal type   | Param | when part of rule name
..          | NAME ? type   | Param | elsewhere, NAME is optional
arglist     | "(" arg "," ... ")" | Args | 0 or more[^targ-str]
arg         | target string[^targ-str] | Arg | target language value expression
strliteral  | "'" identifier "'"   | StringLeaf | keyword
...         | "\"" identifier "\""   | StringLeaf | soft keyword
...         | other token in quotes   | StringLeaf
nonterminal | identifier    | str       | no leading "_"<br>not uppercase()
terminal    | token type    | NameLeaf | [^terminal]
action      | "{" target string[^targ-str] "}" | str | target language value expression
meta        | "@" metavalue | Meta  | On a separate logical line
metavalue   | NAME          | str   | Value is the name.
...         | STRING        | str

[^args-params-opt]:Optional arglist and params.  
  For a rule, missing arglist and/or params is the same as '()'.  Otherwise, missing params *is not* the same as '()'.  
Presence and number of args must agree with presence and number of parameters of the nonterminal.

[^terminal]:Terminal.  
Name (unquoted) of a specific type of [Token](#tokens).  It matches any Token of that type.  
May also be `SOFT_KEYWORD`.  This matches a NAME Token, except it matches only soft keywords.  A soft keyword is any name appearing in "identifier" in a strliteral elsewhere in the grammar file.
All Python identifiers which satisfy `identifier.isupper()` are reserved as terminals.

[^targ-str]:Target strings.  
These are strings in the target language.  They have a limited syntax
  - NUMBER or NAME or STRING or CHAR.
  - target string within matching groups - "(...)", "[...]", "{...}".  These can be nested.  Unmatched group is a SyntaxError.
  - OP, other than a grouping token.  
Special syntax for arglist and arg:
  - A single arg requires a trailing comma.  Multiple args have optional trailing comma.
  - In an arg, OP cannot be ",".  Commas separate args from each other in an arglist.
