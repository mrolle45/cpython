""" tokentype.py

Dynamically creates an enumeration class called TokType, which varies
according to a given Preprocessor parameter.

Each TokType member's value is a TokAttrs object, which has various attributes
pertinent to the token type.  For example, TokType.CPP_ID.id = True.

"""

from __future__ import annotations

import enum
from itertools import product

from pcpp.common import *
from pcpp.replacements import ReplStage

__all__ = 'type_names make_TokType'.split()

def make_TokType(prep: Preprocessor,
                 REs: RegExes,
                 # Maps each punctuator string to one or more type names.
                 punc_values: Mapping[str, list[str]],
                 ) -> type:
    """
    Create a class called TokType, which is an enumeration class.  This is
    customized according to attributes of the given preprocessor.

    TokType['token'], or TokType.token, is the member whose name is `token`.
    """

    lang: prep.Language = prep.lang

    # Create the TokAttrs class for the enum members.
    class TokAttrs(TokAttrsBase):
        pass

    TokAttrs.class_init(prep.lang, REs)

    TokAttrs.set_punctuators(punc_values)

    TokType = enum.Enum('TokType', {tok:tok for tok in type_names},
                        type=TokAttrs)

    TokType.__repr__ = lambda obj: obj.value

    # Alias names.  Alias name -> one or more token names, as a string.
    # These are added to TokType.tok_alias_table.

    TokType.tok_alias_table = dict()
    def alias(name: str, tok: str) -> None:
        TokType.tok_alias_table[name] = tok

    alias('ANYNUM', 'CPP_INTEGER CPP_FLOAT CPP_DOT_FLOAT CPP_NUMBER')
    alias('ANYCHAR', 'CPP_CHAR CPP_EXPRCHAR')
    alias('ANYSTR', 'CPP_STRING CPP_RSTRING')
    alias('ANYQUOTED', 'ANYSTR ANYCHAR')
    alias('LITERAL', 'ANYNUM ANYCHAR ANYSTR')

    if lang.clang:
        _add_seps_clang(TokType)
    else:
        _add_seps(TokType)

    return TokType

def _add_sep(
    cls, left_tokens: str, right_tokens: str, *,
    sep: Callable[PpTok, PpTok, PpTok] = None,
    ) -> None:
    """
    Set attrs.sep_from for the token types.  Each types argument can be:

    * An object of type Self.
    * A token name.
    * Token names separated by space, expanded recursively.
    * An alias in tok_alias_table, expanded recursively.
    * A punctuator value, representing the corresponding token name(s) in
        cls.punc_values.  In right_tokens, it is a single character, and
        expands to all values starting with that character.

    After expanding left and right to a sequence of token names, the left and
    right types are declared as separated.  If `sep` is provided, then this is
    a callable(before left token, left token, right token) which returns True
    for separation of the actual left and right tokens.  The default always
    returns True.
    """
    # Helper generator to expand token names.
    def expand(tokens: str,
                # Match all punct values starting with value character.
                # True for right tokens, False for left tokens.
                value_char: bool
                ) -> Iterator[Self]:
        """ Generates expanded sequence of token names. """
        def expand2(tokens: str | Self) -> Iterator[Self]:
            if isinstance(tokens, cls):
                yield tokens
            elif ' ' in tokens:
                for t in tokens.split():
                    yield from expand2(t)
                    isinstance(tokens, typing.Sequence)
            elif tokens in cls.tok_alias_table:
                yield from expand2(cls.tok_alias_table[tokens])
            elif value_char and len(tokens) == 1:
                for t in cls.punc_values:
                    if t.startswith(tokens):
                        for token in cls.punc_values[t]:
                            yield cls[token]
            elif not value_char and tokens in cls.punc_values:
                for token in cls.punc_values[tokens]:
                    yield cls[token]
            else:
                yield cls[tokens]
        yield from expand2(tokens)

    lefts = expand(left_tokens, value_char=False)
    rights = expand(right_tokens, value_char=True)

    for left, right in itertools.product(lefts, rights):
        left.sep_from[right] = sep

def _add_seps(cls) -> None:
    """ Set token separations for default use. """

    prep = cls.prep
    REs = cls.REs

    add_sep = lambda *args, **kwds: _add_sep(cls, *args, **kwds)

    # Any punctuator which becomes another punctuator by adding 
    # a single character...
    for p1, p2 in product(cls.punc_values, repeat=2):
        if p2.startswith(p1) and len(p2) == len(p1) + 1:
            add_sep(cls.punc_values[p1], p2[-1])
            x = 0
    add_sep('ANYNUM', '+ -')
    add_sep('CPP_DOT', 'ANYNUM CPP_ELLIPSIS')
    add_sep('ANYNUM', '. + -')
    add_sep('ANYNUM', 'CPP_ID ANYCHAR ANYNUM')

def _add_seps_clang(cls) -> None:
    """ Set token separations for clang simulation. """

    # Reference AvoidConcat() in clang/lib/Lex/TokenConcatenation.cpp from
    # clang 20.0 repository.

    # Separation is solely a function of the left and right types.  In a few
    # cases, the token preceding the left token is involved in the test.  
    #
    # When the left and right types are possibly separated, there is a
    # corresponding callback function.  If this is None, the result is always
    # true.  Otherwise the callback is called with the actual tokens as well
    # as the token written before the left token, to make the decision.

    lang = cls.lang
    REs = cls.REs

    add_sep = lambda *args, **kwds: _add_sep(cls, *args, **kwds)

    # These are the cases implemented in the AvoidConcat() function.
    #   left        right       right[0]    conditions
    #   ----        -----       --------    ----------

    #   punct x                 =           when 'x=' is also a punct.

    for p in cls.punc_values:
        if f'{p}=' in cls.punc_values:
            add_sep(p, '=')

    if lang.cplus_ver:
        # C++ has ud-suffixes which look like identifiers.
    #   quoted      ...                     C++.

    #   ...         ident
        add_sep('ANYQUOTED', 'CPP_ID')

    #   ...         number      not .       left has UD suffix.
        sep = (lambda _before_left, left, right:
               left.match.group('ud_sfx')
               )
        add_sep('ANYQUOTED', 'CPP_INTEGER CPP_FLOAT', sep=sep)

    #               quoted                  left has UD suffix
    #                                       and right has size prefix.
        def sep_quoted_quoted(_before_left, left, right) -> bool:
            return (left.match.group('ud_sfx')
                    and right.match.group('pfx')
                   )
        add_sep('ANYQUOTED', 'ANYQUOTED', sep=sep_quoted_quoted)

    #   ident       number                  [0] is not '.'
    add_sep('CPP_ID', 'CPP_INTEGER CPP_FLOAT')

    #   ident       ident
    add_sep('CPP_ID', 'CPP_ID')

    #   ident       quoted                  right has size prefix
    #   ident       quoted                  left is a size prefix and
    #                                         right has no size prefix
    # Size prefixes are different for C and C++.
    # clang gets it wrong for C.

    if lang.cplus_ver:
        size_prefixes = 'L u u8 U R LR uR u8R UR'
    else:
        size_prefixes = 'L'         # according to clang.
                                    # should be 'L u u8 U'.
    pfxs = set(size_prefixes.split())
    def sep(_before_left, left: PpTok, right: PpTok) -> bool:
        if right.type.patt.match(right.value).group('pfx'):
            return True
        elif left.value in pfxs:
            return True
        return False

    add_sep('CPP_ID', 'ANYQUOTED', sep=sep)

    #   quoted      ident                   C++
    if lang.cplus_ver:
        add_sep('ANYQUOTED', 'CPP_ID')

    #   quoted      any                     C++
    #                                       and ident + right is separated 
    #                                       and left has UD-suffix
    for right_type, id_sep in cls.CPP_ID.sep_from.items():
        def addsep(right_type, id_sep) -> None:
            if right_type is cls.CPP_ID:
                sep = None
            elif id_sep is None:
                sep = lambda before_left, left, right: (
                    left.match.group('ud_sfx')
                    )
            else:
                sep = lambda before_left, left, right: (
                    id_sep(before_left, left, right)
                    and left.match.group('ud_sfx')
                    )
            #if prep.cplus_ver:
            #    add_sep('ANYQUOTED', right_type, sep=sep)
        addsep(right_type, id_sep)

    #   number                  . + - a-z A-Z 0-9 _
    add_sep('ANYNUM', 'ANYNUM CPP_ID . + -')
    def sep_num_quote(_before_left, left: PpTok, right: PpTok) -> bool:
        return bool(right.match.group('pfx'))
    add_sep('ANYNUM', 'ANYQUOTED', sep=sep_num_quote)
    #   .                       .           left preceded by . also.
    sep = (lambda before_left, left, right:
            before_left and before_left.type is cls.CPP_DOT)
    add_sep('.', '.', sep=sep)
        
    #   .               number  0-9
    digit_re = re.compile(REs.digit)
    sep = (lambda before_left, left, right:
            digit_re.match(right.value[:1]))
    add_sep('.', 'ANYNUM', sep=sep)

    pairs = '&& ++ -- // << >> || ## -> /* <: <% %> %: :> #@ #%'
    if lang.cplus_ver:
        pairs += ' .* :: ->* '
        if lang.cplus_ver > 2020:
            pairs += ' <=>'

    for pair in pairs.split():
        add_sep(pair[:-1], pair[-1])


type_names: list[str] = [
    # Tokens in C are named CPP_type.
    # Tokens only in C++ are named CXX_type.  (C++11 2.7)
    # Digraphs for CPP_type are named CPP_ALT_type.

    # Basic C language tokens, in translation phases 7 and 8.  (C99 6.4).

    'CPP_ID',           # Any identifier, including C or +C++ keywords.
                        # Keywords are recognized after the preprocessor.
    'CPP_INTEGER',      # Integer constant (C99 6.4.4.1).
    'CPP_DOT_FLOAT',    # Floating constant (C99 6.4.4.2) with leading '.'
    'CPP_FLOAT',        # Floating constant (C99 6.4.4.2), otherwise
    'CPP_STRING',       # String literal (C99 6.4.5).
    'CPP_RSTRING',      # Raw string literal (C++23 5.13.5).
    'CPP_CHAR',         # Character constant (C99 6.4.4.4).

    # Punctuator tokens (C99 6.4.6) ...
    # C++ (C++23 5.12) has four punctuators not found in C:
    #   "::", ".*", "->*", "<=>",
    #   as well as named operators such as "bitor".  
    #   Lexing these tokens is enabled with --c++ on the command line.

    # Arithmetic operators
    'CPP_PLUS',             # +
    'CPP_PLUSPLUS',         # ++
    'CPP_MINUS',            # -
    'CPP_MINUSMINUS',       # --
    'CPP_STAR',             # *
    'CPP_FSLASH',           # /
    'CPP_PERCENT',          # %
    'CPP_LSHIFT',           # <<
    'CPP_RSHIFT',           # >>

    # Logical operators
    'CPP_LOGICALAND',       # &&
    'CXX_AND',              # and
    'CPP_LOGICALOR',        # ||
    'CXX_OR',               # or
    'CPP_EXCLAMATION',      # !
    'CXX_NOT',              # not

    # bitwise operators

    'CPP_AMPERSAND',        # &
    'CXX_BITAND',           # bitand
    'CPP_BAR',              # |
    'CXX_BITOR',            # bitor
    'CPP_HAT',              # ^
    'CXX_XOR',              # xor
    'CPP_TILDE',            # ~
    'CXX_COMPL',            # compl

    # Comparison operators
    'CPP_EQUALITY',         # ==
    'CPP_INEQUALITY',       # !=
    'CXX_NOT_EQ',           # not_eq
    'CPP_GREATEREQUAL',     # >=
    'CPP_GREATER',          # >
    'CPP_LESS',             # <
    'CPP_LESSEQUAL',        # <=
    'CXX_SPACESHIP',        # <=>

    # Conditional expression operators
    'CPP_QUESTION',         # '?
    'CPP_COLON',            # ':

    # Member access operators
    'CPP_DOT',              # .
    'CPP_DEREFERENCE',      # ->
    'CXX_DOTPTR',           # .*
    'CXX_DEREFPTR',         # ->*
    'CXX_DCOLON',           # ::

    # Assignment operators
    'CPP_EQUAL',            # =
    'CPP_XOREQUAL',         # ^=
    'CXX_XOR_EQ',           # xor_eq
    'CPP_MULTIPLYEQUAL',    # *=
    'CPP_DIVIDEEQUAL',      # /=
    'CPP_PLUSEQUAL',        # +=
    'CPP_MINUSEQUAL',       # -=
    'CPP_OREQUAL',          # |=
    'CXX_OR_EQ',            # or_eq
    'CPP_ANDEQUAL',         # &=
    'CXX_AND_EQ',           # and_eq
    'CPP_PERCENTEQUAL',     # %=
    'CPP_LSHIFTEQUAL',      # <<=
    'CPP_RSHIFTEQUAL',      # >>=

    # Grouping and separators
    'CPP_LPAREN',           # (
    'CPP_RPAREN',           # )
    'CPP_LBRACKET',         # [
    'CPP_ALT_LBRACKET',     # <:
    'CPP_RBRACKET',         # ]
    'CPP_ALT_RBRACKET',     # :>
    'CPP_LCURLY',           # {
    'CPP_ALT_LCURLY',       # <%
    'CPP_RCURLY',           # }
    'CPP_ALT_RCURLY',       # %>

    'CPP_COMMA',            # ,
    'CPP_SEMICOLON',        # ;

    # Other
    'CPP_POUND',            # #
    'CPP_ALT_POUND',        # %:
    'CPP_DPOUND',           # ##
    'CPP_ALT_DPOUND',       # %:%:
    'CPP_ELLIPSIS',         # ...

    # Not in source character set, but valid in GCC.
    'CPP_DOLLAR',           # $
    'CPP_AT',               # @
    'CPP_GRAVE',            # `

    'CPP_DELTA',            # greek small delta char for testing.

    # Remaining token types are used only within the preprocessor, and not
    #   passed on to translation phase 5 (except where otherwise noted).
    # However, when writing an output file, some of these tokens 
    #   may be shown.

    'CPP_WS',
    'CPP_COMMENT1',
    'CPP_COMMENT2',
    'CPP_NEWLINE',

    # Character constant (C99 6.4.4.4) in #if(def) directive.  Same as
    # CPP_CHAR, but unicode escapes aren't reverted.
    ## NOTE: a CPP_CHAR could appear in an #if(def) if the char is part of
    ## macro expansion of def.  So we can't use this type.
    'CPP_EXPRCHAR',

    # A directive name 
    'CPP_DIRECTIVE',

    # A macro name following a "#define" (state = 'DEFINE').  
    # Possibly incomplete identifier.  This names the token function.
    'CPP_MACRO',
    # The token function gets the complete identifier and sets one of these
    # types.
    'CPP_OBJ_MACRO', 'CPP_FUNC_MACRO',

    # A header name following a "#include" (state = 'INCLUDE').
    'CPP_H_HDR_NAME', 'CPP_Q_HDR_NAME',

    # Markers for a change in source file and/or line number...
    # Created by entering an included source.
    'CPP_LOCENTER',
    # Created by leaving an included source.
    'CPP_LOCLEAVE',
    # Created by processing a '# line' directive.
    'CPP_LOCMOVE',

    # A placemarker.  Value is empty.  Type.marker is True.  Used for a
    # parameter or __VA_OPT__ with no tokens.  Denoted by the symbol φ.
    'CPP_MARKER',

    # A stringize operator, '#' in a macro replacement list.  A '#' token
    # anywhere else is just a normal punctuator.
    'CPP_MKSTR',
    'CPP_ALT_MKSTR',

    # A paste operator, '##' in a macro replacement list.  A '##' token
    # anywhere else is just a normal punctuator.
    'CPP_PASTE',
    'CPP_ALT_PASTE',

    # A value to be passed through to the output file without further
    # interpretation.  This is written on a separate line.
    'CPP_GROUP',

    # Preprocessing number that is not a CPP_INTEGER or CPP_FLOAT or
    # CPP_DOT_FLOAT.  It will be skipped, and type.err is True.
    'CPP_NUMBER',

    # Anything that does not match any other tokens.
    # Value is the next single character in most cases.
    #   Or the rest of the line for unmatched quotes.
    #   Or a number that is not a valid integer or floating value.
    'error',

    ]


class TokAttrsBase(str):
    """
    A dynamically created class, dependent on some given parameters.  The
    actual created class will be called TokAttrs, and will be a subclass.

    Each instance describes a token with a given type name.  It is the value
    of the TokType enumeration having that name.  For example, TokType.CPP_ID
    is an object TokAttrs('CPP_ID'), and it has attributes described in the
    TokAttrs class.
    """

    # Attributes for a subclass as a whole..
    prep: Preprocessor
    REs: RegExes

    # Various attributes of the TokType.  Class attributes are the default.
    # Different values are set in an instance by the class_init().
    ws: bool = False        # Any kind of whitespace.
    space: bool = False     # Any kind of whitespace other than
                            # newline.
    comment: bool = False   # Either type of comment.
    nl: bool = False        # A newline
    chk_nl: bool = False    # Value may contain newlines.
    revert: str = ''        # Undo some or all value replacements.
                            #   Each replacement is reverted if the type
                            #   corresponds to a word in the str.
                            # ... spliced: line splices.
                            # ... unicode: unicode chars.
                            # ... trigraph: trigraphs.
                            # ... *: all of the above.
    norm: bool = True       # Anything which is a regular token.
    str: bool = False       # "..."
    chr: bool = False       # '...'
    quoted: bool = False    # either str or chr
    id: bool = False        # An indentifier.
    newpos: bool = False    # A marker for change in token position.
    enter: bool = False     # ... Entering included file.
    leave: bool = False     # ... Leaving included file.
    move: bool = False      # ... Changing line/filename.
    dir: bool = False       # A directive.
    hhdr: bool = False      # <...> in a #include directive.
    qhdr: bool = False      # "..." in a #include directive.
    hdr: bool = False       # Either hhdr or qhdr.
    int: bool = False       # An integer.
    float: bool = False     # A floating point number.
    number: bool = False    # CPP_NUMBER
    hash: bool = False      # CPP_POUND (or '#').
    dhash: bool = False     # CPP_DPOUND (or '##').
    stringize: bool = False # CPP_MKSTR (or 'Σ').
    paste: bool = False     # CPP_PASTE (or 'π').
    val: str = ''           # Default value for new token.
    patt: re.Pattern = None # Pattern used to match the value.
    group: bool = False     # Pass the value directly to output.
    iter: bool = False      # Pass values in an iterator to output.
    marker: bool = False    # Empty placemarker (or 'φ').
    err: bool = False       # An error token (single character).
    lit: str = None         # Token (a punctuator) always has this value.
    sep_from: TokSpaceDict  # Any types for RHS token which might
                            # need a space for separation from LHS token.

    # Table of non-default attributes for each token type name.  Used to set
    # the attributes of a TokType member.  Set on the subclass by the
    # class_init() class method.
    attrs_table: Mapping[str, TokAttrs]

    # Associates a punctuator's value with its token name(s).
    punc_values: Mapping[str, list[str]]

    @classmethod
    def class_init(cls, lang: PreProcessor.Language, REs: RegExes) -> None:
        """ Initializes the subclass and creates cls.attrs_table. """
        cls.lang = lang
        cls.REs = REs

        # Make table of attributes for various TokType's.  Note, types
        # 'CPP_ALT_xxx' will have same as 'CPP_xxx'.
        cls.attrs_table = collections.defaultdict(dict)
        cls.attrs_table.update(dict(
            CPP_NEWLINE=    dict(nl=True, chk_nl=True, ws=True, val='\n'),
            CPP_WS=         dict(ws=True, val=' '),
            CPP_COMMENT1=   dict(comment=True,
                                 revert=ReplStage.SPLICE,
                                 ws=not(lang.comments and lang.emulate)),
            CPP_COMMENT2=   dict(comment=True,
                                 revert=ReplStage.SPLICE,
                                 ws=not(lang.comments and lang.emulate)),
            CPP_CHAR=       dict(chr=True, quoted=True,
                                 revert=ReplStage.ESCAPE,
                                 patt=re.compile(REs.char)),
            CPP_EXPRCHAR=   dict(chr=True, quoted=True,
                                 revert=ReplStage.ESCAPE,
                                 patt=re.compile(REs.char)),
            CPP_STRING=     dict(str=True, quoted=True,
                                 revert=ReplStage.ESCAPE,
                                 patt=re.compile(REs.string)),
            CPP_RSTRING=    dict(str=True, quoted=True,
                                 revert=ReplStage.ORIG,
                                 chk_nl=True,
                                 patt=re.compile(REs.rstring)),
            CPP_ID=         dict(id=True,
                                 #revert=ReplStage.ESCAPE,
                                 patt=re.compile(REs.ident)),
            CPP_OBJ_MACRO=  dict(copy='CPP_ID'),
            CPP_FUNC_MACRO= dict(copy='CPP_ID'),
            CPP_LOCENTER=   dict(newpos=True, enter=True, norm=False),
            CPP_LOCLEAVE=   dict(newpos=True, leave=True, norm=False),
            CPP_LOCMOVE=    dict(newpos=True, move=True, norm=False),
            CPP_GROUP=      dict(group=True, norm=False),
            CPP_INTEGER=    dict(int=True, revert=ReplStage.ESCAPE,
                                 patt=re.compile(REs.int)),
            CPP_FLOAT=      dict(float=True, revert=ReplStage.ESCAPE),
            CPP_DOT_FLOAT=  dict(float=True, revert=ReplStage.ESCAPE),
            CPP_NUMBER=     dict(number=True, revert=ReplStage.ESCAPE),
            CPP_POUND=      dict(hash=True),        # '#' not in macro.
            CPP_DPOUND=     dict(dhash =True),      # '##' not in macro
            CPP_MKSTR=      dict(stringize=True),   # '#' in a macro defn.
            CPP_PASTE=      dict(paste=True),       # '##' in a macro defn.
            CPP_DIRECTIVE=  dict(dir=True, norm=False),
            CPP_H_HDR_NAME= dict(hhdr=True, hdr=True,
                                 revert=ReplStage.ESCAPE),
            CPP_Q_HDR_NAME= dict(qhdr=True, hdr=True,
                                 revert=ReplStage.ESCAPE),
            CPP_MARKER=     dict(marker=True),      # 'φ'.
            error=          dict(err=True,
                                 revert=ReplStage.ESCAPE,
                                 ),
            ))

    @classmethod
    def set_punctuators(cls, puncs: Mapping[str, list[str]]):
        """
        Given mapping from punctuator values -> token names, set attrs.lit in
        the corresponding type.  Stores the mapping for use in add_sep().
        """
        cls.punc_values = puncs
        for value, tokens in puncs.items():
            for token in tokens:
                cls.attrs_table[token].update(lit=value)

    def __init__(self, token: str) -> None:
        """ Sets attributes from corresponding member of attrs_table. """

        # The instance already has the attributes given in the class
        # attributes, so we only need to override them from attrs_table.
        self.sep_from = TokSpaceDict()
        attrs = self.attrs_table[token]
        if '_ALT' in token:
            altattrs = self.attrs_table[token.replace('CPP_ALT', 'CPP')]
            altattrs.update(attrs)
            attrs = altattrs
        if attrs:
            copy = attrs.pop('copy', '')
            for copyfrom in copy.split():
                attrs.update(**self.attrs_table[copyfrom])
            self.__dict__.update(**attrs)

    #def __str__(self) -> str:
    #    return self.name

class TokSpaceDict(collections.UserDict):
    pass
