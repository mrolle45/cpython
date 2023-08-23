# Parameters and Arguments for Grammar Rules and Other Enhancements

This is a guide to a (*proposed*) feature for `cpython`.  It is implemented in the [feature-rule-params](https://github.com/mrolle45/cpython/tree/feature-rule-params) branch in my fork of the python/cpython repo.

It consists of:

- An [enhancement](#metagrammar-enhancements) to the metagrammar.  
It adds an optional parameter list to certain names.  
It adds an optional argument list to such a name appearing in a rule alternative item.
It allows a rule alternative with no items, called `<always>`.  It consumes no input and returns a true value or whatever action is given.

- Use of these features in the generated Python [parser](#python-parser-generator).

- Use of these features in the generated C [parser](#c-parser-generator).

All of these changes are in the `Tools/peg_generator/pegen` directory, except if indicated otherwise.

## Metagrammar Enhancements

These appear in the file `metagrammar.gram`.  They enhance what can appear in any PEG grammar file (including `metagrammar.gram` itself)

Included are some purely cosmetic changes, such as replacing `rules` with `rule*` and combining alts with and without an item via `item?`.

### Guide to Grammars

The structure of any grammar file, including the metagrammar itself, is specified in the metagrammar.gram file.  The grammar file uses the same basic tokens and tokenizer (the [`tokenize`](https://docs.python.org/3/library/tokenize.html) library module) as Python source, including continuation of a logical line to match parentheses and trailing backslashes.

The hierarchy of elements in the grammar is:

- *meta*.  This is information which is used by a generator which converts the grammar to a parser file (in the target language).

- *rule*.  This is something which consumes some text from the file being parsed and produces some result (such as a Python object or a C value).

    - alternative (or *alt*).  One way of parsing the input to satisfy the rule.  It produces the result for the rule if it is successful, or a false value (such as `None` in Python or `NULL` in C) if not.  
If the alt fails, then the parser will try the next alt.  If all alts fail, then the rule also fails.

      - *item*.  Something which must be parsed from the input to satisfy the alt.  
        Items are parsed in the order they appear in the alt, each one consuming more of the input, as long as they all succeed.  If any item fails, then the entire alt fails.  
        *New:* Items are optional.  If there are no Items, the alt succeeds.  If an action is not given, the default action is a true value (such as `True` in Python or `true` in C).

        - *variable name*, which is set to the result of the item in the parser.  It is optionally *designated* by `name =` at the beginning of the item.  If there is no name specified, the parser generator creates a *default* name.  
            This name is visible everywhere in the alt following this item.

        - *leaf*.  This is the basic unit that is consumed from the input.  This may be either a NAME or a STRING token produced from the tokenizer.  
            The item consists of a leaf or a more complex expression involving various leaves.  The expression may include more alts.  For details, see the `item` and `atom` rules in metagrammar.py.

      - *action*.  Appears optionally at the end of the alt, after all the items.  It is a target enclosed in `{...}`.  It is copied into the generated parser verbatim to specify the result value of the alt.

- *typed name*.  This is used as the name of a rule, or a designated variable name, or (recursively) a parameter name.  
*New.* The name may be followed by zero or more parameters.  Each parameter is another *typed name*.
  
    - *annotation*.  An optional target enclosed in `[...]` following the name itself.  It may appear in the generated parser as the type of the rule or variable.

- *target*.  An expression in the target language.  It is copied verbatim into the generated parser.  
    In the metagrammar, a target is limited to basic elements possibly enclosed (recursively) in matching parens (`()`, `[]`, or `{}`).  The basic elements are names, literal strings, operators, and some other punctuation characters.  


### Rule and Variable Parameters

The syntax for a typed name is extended to include an optional parameter list.  The new syntax is
`name [params] [annotation]`.

This is a new metagrammar rule called `typed_name`

`params` has the form:  
`( typed_name [, typed_name]... [,])` or `()`.  
Note the optional trailing comma, and the recursive use of `typed_name`.

The name may be used within its scope in:
- a target.  Here it is copied verbatim to the generated file.
- an alternative item.

If the name is a rule, and params is not given, then it is considered to be `()`.  In generated code, the rule is always called as a function or a method.

By the way, `(memo)` following a rule name, as in `r (memo): ...`, is NOT interpreted as a parameter list, but rather as a `memoflag`.  A rule may have both params and a memoflag (in that order).

### Name Leaf Arguments and Extensions

A name leaf can include a parameter or variable name, in addition to a rule name.  This is implemented by the parser generators.

A name leaf now includes optional arguments.  The new syntax is:

`NAME [arguments]`.

`arguments` has the same form as a parenthesized tuple:  
`( argument , argument [, argument]... [,])`  
or `( argument , )`  
or `()`.  
That is, zero or more `argument`s separated by commas, and a trailing comma required with one argument and optional with two or more arguments.

An argument is any target, except that it does not contain a comma that is not nested in `()`, `[]`, or `{}`.  
For example, `(x, y, z)` is three arguments, while `(x, (y, z))` is two arguments.

The arguments must agree in presence and number with the params of the corresponding typed name.
If the typed name has no parameters, then the arguments must be omitted.    
If the typed name has parameters (including `()`), then the arguments must be provided, and contain the same number of arguments as parameters.  However if the name is for a rule, then arguments may be omitted.


### Null Alternatives, or `<always>`

A rule alternative may now omit any Items.  This always matches the input without consuming any of it.  It must be the last alternative in a rule or (alt ...) or [alt ...].  It will be shown as `<always>` in comments in generated code.

It is useful to supply a default value in its action when all other alts fail.  If no action is specified, then a true value is returned.

Here is an example, which is now in the metagrammr:

```
rule[Rule]:
    | n=typed_name m=memoflag? ":" a=maybe_alts? NEWLINE aa=more_alts? {
          Rule(n, Rhs(a + aa), memo=m) }

#   After ':' in a rule on same line.  Zero or more with '|' separator.
maybe_alts[Rhs]:
    | !NEWLINE a="|".alt+  { Rhs(a) }
    | { Rhs() }

#   Indented block following a rule, each line introduced by '|', or nothing.
more_alts[Rhs]:
    | INDENT a=( "|" b=alts NEWLINE { b } )+ DEDENT { Rhs (chain (*a)) }
    | { Rhs() }
```
The `maybe_alts` and `more_alts` both return an Rhs object (which is a list except that an empty list tests as true).  So these results may be used directly in the `rule` rule.

Another useful possibility is to insert a test condition in a series of alternative items, so that it must be true in order to proceed to the next item.  The test will be an expression in the target language, and could involve a rule parameter or an earlier designated variable.  The condition must be enclosed in a group.

For example, a rule `foo(n)` can test whether `n` > 1 and then parse a NAME:

```
foo(n[int]):
    | ({n > 1}) NAME
    | something_else
```

## Python Parser Generator

- Parameter names and types are added to the signature of the `def` statement for a rule.  The type defaults to `Any`.  An example would be:  
`r(x[int], y)` generates `def r(self, x: int, y: Any):`  
If the parameter name has its own parameters, then these are included in the generated parameter's type.

- Arguments are added to the name of a name leaf in the generated parser.  
For a rule `r` , they are inserted inside the `()` calling the rule's method, as in `(var := self.r(arg1, arg2))`  
For a non-rule name, they are appended to the name, including the `(...)`, as in `(var := name(arg1, arg2))` 

- The parser object can be used in a non-rule name, but it requires adding `self` as both a parameter of the name and an argument.


## C Parser Generator

This is the same behavior as for the Python generator, above, except that:

- The parameter names for rules apply in both the declaration and the definition of the rule, as well as any generated `r_raw` rule generated for rule `r`.  The types of the parameters (including those with their own parameters) are included in the generated code.

- For a rule `r` invoked with parameters, the generated parser contains `r_rule(p, arg1, arg2)`  
For a non-rule name, same as the Python parser generator.

- For the parser object in a *callable* non-rule name, it requires `Parser * p` as a parameter and `p` as an argument.

- Also, a single-character literal can be used in an alternative item, even if it is neither an identifier nor a Python operator.  The tokenizer makes a "unrecognized character" token for this character.

## Note on Duplicate Names of Rules, Parameters and Variables

Every Item in an alt is associated with a variable name, either designated or default.

Every Rule and Rule parameter is also associated by a name, specifically given in the Rule definition.

Since these names can appear later in the grammar file, the parser generators prevent duplication of these names.  

The list of names that are visible at any point in the grammar file consists of, in this order:
1. All rule names in the grammar file.
2. The parameter names of the current rule.
3. The variable names (both designated and default) introduced so far in the current alt.

If a name appears more than once in this list, in a later string, the name refers only to the *first* appearance.  The names of subsequent appearances are modified in an *unspecified* manner; the modified names can be used, but *this is discouraged*.

In order to avoid confusion, it is recommended that:
- parameter names and designated variable names are different from any rule names in the entire grammar.
- Designated variable names are different from each other

