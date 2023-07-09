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
  - Standard Python indentation structure.  This means that any expression with matching `()` or `[]` or `{}` can span multiple lines, ignoring indentation.
  - Formatting of Rules with newlines and indents to allow them to carry over several lines.
  - The expression ordered choice operator is changed from **`/`** to **`|`**.
  - A Gather expression, similar to a repeat expression, but with another expression for a separator between successive repetitions.
  - A Forced expression `&&expr`, which is the same as `expr` except that if it fails to parse, it raises a `SyntaxError`.
  - An optional group, `[alts]`, which is the same as `((alts)?)`.
  - A Cut item, **`~`**, which consumes no input, and always succeeds.  However, if all further items in the alt fail, then all following alts in the ordered choice are ignored and the ordered choice itself fails.
  - An end of file item, **`$`**, which can appear as the *last* item in an alt.
  - An optional variable name, `var`**`=`**`item` for most alt items.  `var` becomes a nonterminal name within all following items in the same alt.
  - Optional calling parameters for nonterminals.  
  - :heavy_check_mark:
  - 🕑✔️✔
- xxx
  - xxx
  - 
  - 
  - xxx
- 

