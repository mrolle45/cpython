""" lexer.py
Defines a lexer, to use with ply.lex, to turn input string into tokens.
"""

from __future__ import annotations

import dataclasses
import enum
import re
import copy

from ply import lex, yacc
from ply.lex import LexToken, Lexer, LexError, TOKEN
from pcpp.parser import in_production

# -----------------------------------------------------------------------------
# Default preprocessor lexer definitions.   These tokens are enough to get
# a basic preprocessor working.   Other modules may import these if they want
# -----------------------------------------------------------------------------

# Special tokenizing for preprocessing directives
#   A '#' after possible whitespace, introduces a directive.  Token type is CPP_DIRECTIVE,
#       and new state is DIRECTIVE.
#   Following whitespace is skipped.  The next token should be the directive name, a CPP_ID.
#       If it is 'include', type is CPP_INCLUDE and new state is INCLUDE.
#       If it is 'define', type is CPP_DEFINE and new state is DEFINE.
#       Any other token of any type goes back to the INITIAL state.
#   Reaching a newline goes back to the INITIAL state.

states = [
    ('DIRECTIVE', 'inclusive'),
    ('INCLUDE', 'inclusive'),
    ('DEFINE', 'inclusive')
    ]

tokens = (
   'CPP_ID','CPP_INTEGER', 'CPP_FLOAT', 'CPP_STRING', 'CPP_CHAR', 'CPP_WS', 'CPP_COMMENT1', 'CPP_COMMENT2',
   'CPP_NEWLINE', 'CPP_DIRECTIVE', 'CPP_OBJ_MACRO', 'CPP_FUNC_MACRO',
   'CPP_POUND','CPP_DPOUND', 'CPP_PLUS', 'CPP_MINUS', 'CPP_STAR', 'CPP_FSLASH', 'CPP_PERCENT', 'CPP_BAR',
   'CPP_AMPERSAND', 'CPP_TILDE', 'CPP_HAT', 'CPP_LESS', 'CPP_GREATER', 'CPP_EQUAL', 'CPP_EXCLAMATION',
   'CPP_QUESTION', 'CPP_LPAREN', 'CPP_RPAREN', 'CPP_LBRACKET', 'CPP_RBRACKET', 'CPP_LCURLY', 'CPP_RCURLY',
   'CPP_DOT', 'CPP_COMMA', 'CPP_SEMICOLON', 'CPP_COLON', 'CPP_BSLASH', 'CPP_SQUOTE', 'CPP_DQUOTE',

   'CPP_DEREFERENCE', 'CPP_MINUSEQUAL', 'CPP_MINUSMINUS', 'CPP_LSHIFT', 'CPP_LESSEQUAL', 'CPP_RSHIFT',
   'CPP_GREATEREQUAL', 'CPP_LOGICALOR', 'CPP_OREQUAL', 'CPP_LOGICALAND', 'CPP_ANDEQUAL', 'CPP_EQUALITY',
   'CPP_INEQUALITY', 'CPP_XOREQUAL', 'CPP_MULTIPLYEQUAL', 'CPP_DIVIDEEQUAL', 'CPP_PLUSEQUAL', 'CPP_PLUSPLUS',
   'CPP_PERCENTEQUAL', 'CPP_LSHIFTEQUAL', 'CPP_RSHIFTEQUAL', 'CPP_ELLIPSIS',
   'CPP_H_HEADER_NAME', 'CPP_Q_HEADER_NAME',
   'CPP_PLACEMARKER', 'CPP_NULL', 'CPP_PASTED', 'error',
)

comment_tokens = ('CPP_COMMENT1', 'CPP_COMMENT2', )
space_tokens = ('CPP_WS', 'CPP_NEWLINE', 'CPP_NULL', )
ws_tokens = space_tokens + comment_tokens

# Note, unmatched quote characters are an error, not a literal
literals = "+-*/%|&~^<>=!?()[]{}.,;:\\"
#literals = "+-*/%|&~^<>=!?()[]{}.,;:\\\'\""

# Whitespace, but don't match past the end of a line
def t_ANY_CPP_WS(t):
    r'([ \t]+)'
    return t.lexer.whitespace(t)

r_newline = r'(\n)'

# Special newline in a directive.  Returns to INITIAL state.
@TOKEN(r_newline)
def t_DIRECTIVE_INCLUDE_DEFINE_CPP_NEWLINE(t):
    t.lexer.begin('INITIAL')
    return t_ANY_CPP_NEWLINE(t)

# Newline.  Type CPP_WS, advances line number.
@TOKEN(r_newline)
def t_ANY_CPP_NEWLINE(t):
    t.lexer.newline()
    #t.type = 'CPP_WS'
    return t

def t_DIRECTIVE_CPP_ID(t):
    r'[A-Za-z_][\w_]*'
    if t.value == 'include':
        t.lexer.begin('INCLUDE')
    elif t.value == 'define':
        t.lexer.begin('DEFINE')
    return t
_string_literal_linecont_pat = re.compile(r'\\[ \t]*\n')

def t_CPP_DPOUND(t: LexToken) -> LexToken:
    r'\#\#|%:%:'
    return t

# A '#', the first non-whitespace in a line, is a directive.
def t_CPP_POUND(t: LexToken) -> LexToken:
    r'\#|%:'
    if t.lexer.all_ws:
        t.lexer.begin('DIRECTIVE')
        t.type = 'CPP_DIRECTIVE'
    return t

t_CPP_PLUS = r'\+'
t_CPP_MINUS = r'-'
t_CPP_STAR = r'\*'
t_CPP_FSLASH = r'/'
t_CPP_PERCENT = r'%'
t_CPP_BAR = r'\|'
t_CPP_AMPERSAND = r'&'
t_CPP_TILDE = r'~'
t_CPP_HAT = r'\^'
t_CPP_LESS = r'<'
t_CPP_GREATER = r'>'
t_CPP_EQUAL = r'='
t_CPP_EXCLAMATION = r'!'
t_CPP_QUESTION = r'\?'
t_CPP_LPAREN = r'\('
t_CPP_RPAREN = r'\)'

# Include digraphs, which are different spellings of single characters.
t_CPP_LBRACKET = r'\[|<:'
t_CPP_RBRACKET = r'\]|:>'
t_CPP_LCURLY = r'{|<%'
t_CPP_RCURLY = r'}|%>'

t_CPP_DOT = r'\.'
t_CPP_COMMA = r','
t_CPP_SEMICOLON = r';'
t_CPP_COLON = r':'
t_CPP_BSLASH = r'\\'

# Matched quotes on the same line are CPP_CHAR or CPP_STRING.
# Unmatched quotes will be errors, so the following are commented out.
#t_CPP_SQUOTE = r"'"
#t_CPP_DQUOTE = r'"'

t_CPP_DEREFERENCE = r'->'
t_CPP_MINUSEQUAL = r'-='
t_CPP_MINUSMINUS = r'--'
t_CPP_LSHIFT = r'<<'
t_CPP_LESSEQUAL = r'<='
t_CPP_RSHIFT = r'>>'
t_CPP_GREATEREQUAL = r'>='
t_CPP_LOGICALOR = r'\|\|'
t_CPP_OREQUAL = r'\|='
t_CPP_LOGICALAND = r'&&'
t_CPP_ANDEQUAL = r'&='
t_CPP_EQUALITY = r'=='
t_CPP_INEQUALITY = r'!='
t_CPP_XOREQUAL = r'^='
t_CPP_MULTIPLYEQUAL = r'\*='
t_CPP_DIVIDEEQUAL = r'/='
t_CPP_PLUSEQUAL = r'\+='
t_CPP_PLUSPLUS = r'\+\+'
t_CPP_PERCENTEQUAL = r'%='
t_CPP_LSHIFTEQUAL = r'<<='
t_CPP_RSHIFTEQUAL = r'>>='
t_CPP_ELLIPSIS = r'\.\.\.'


# Identifier
t_CPP_ID = r'[A-Za-z_][\w_]*'

# Object and function macro identifiers.
# CPP_OBJ_MACRO is the macro name, skipping the following whitespace
@TOKEN(rf'{t_CPP_ID}(?=\()')
def t_DEFINE_CPP_FUNC_MACRO(t):
    t.type = 'CPP_FUNC_MACRO'
    t.lexer.begin('INITIAL')
    return t
@TOKEN(rf'{t_CPP_ID}')
def t_DEFINE_CPP_OBJ_MACRO(t):
    t.type = 'CPP_OBJ_MACRO'
    t.lexer.begin('INITIAL')
    return t

# Floating literal
def t_CPP_FLOAT(t):
    r'((\d+)(\.\d+)(e(\+|-)?(\d+))?|(\d+)e(\+|-)?(\d+))([lL]|[fF])?'
    return t

# Integer literal
def t_CPP_INTEGER(t):
    r'(((((0x)|(0X))[0-9a-fA-F]+)|(\d+))([uU][lL]|[lL][uU]|[uU]|[lL])?)'
    return t

# String literal
# Terminating " required.
r_schar = r'[^"\n]'
@TOKEN(rf'(L|U|u|u8)?\"{r_schar}*\"')
def t_CPP_STRING(t):
    return t

# h-char.  Part of a <...> header name
#   Anything other than newline or >.
r_hchar = r'[^>\n]'
r_hname = re.compile(f'<({r_hchar})*>')

# q-char.  Part of a "..." header name
#   Anything other than newline or ".
r_qchar = r'[^"\n]'
r_qname = re.compile(f'<({r_qchar})*>')

# h-type and q-type header names.  Only used in INCLUDE state.
# Note, some things in these names are undefined behavior (C99 Section 6.4.7),
#   and this is checked in the preprocessor.include() method.

t_INCLUDE_CPP_H_HEADER_NAME = rf'<({r_hchar})*>'
#t_INCLUDE_CPP_Q_HEADER_NAME = rf'\"({r_qchar})*\"'

# Character constant 'c+' or L'c+' or (U|u|u8)'c'.
# Terminating ' required.
r_cchar = r"[^'\n]"
@TOKEN(rf"(L?\'{r_cchar}*|(U|u|u8)\'{r_cchar})\'")
#@TOKEN(rf"(L?\'{r_cchar}*|(U|u|u8)\'{r_cchar})\'?")
def t_CPP_CHAR(t):
    return t

# VBlock comment, possibly spanning multiple lines
def t_CPP_COMMENT1(t):
    r'(/\*(.|\n)*?\*/)'
    ncr = t.value.count("\n")
    t.lexer.lineno += ncr
    return t.lexer.whitespace(t)

# Line comment
def t_CPP_COMMENT2(t):
    r'(//[^\n]*)'
    return t.lexer.whitespace(t)
    
def t_ANY_error(t):
    # Check for unmatched quote character.
    if t.value[0] in '\'\"':
        endline = t.value.find('\n')
        t.value = t.value[:endline]
        message = f"Unmatched quote {t.value}\\n"
    else:
        t.value = t.value[0]
        message = f"Illegal character {t.value!r}"

    t.type = TokType.error
    t.lexer.prep.on_error_token(t, message)
    t.lexer.skip(len(t.value))
    return t

class Tokens(list[LexToken]):
    """ A list of tokens with some extra capabilities.
    Can make copies of another list.
    Can mark a span within the list, which can be replaced by some other tokens.
    Indicates if changes have been made.
    Can iterate the mark forward.
    """
    def __init__(self, input: Iterable[LexToken] = ()):
        self[:] = map(copy.copy, input)
        self.start, self.end = 0, len(self)

    def moveto(self, start: int) -> None:
        self.start = self.end = start
        self.changed = False

    def mark(self, end: int) -> None:
        self.end = end
        self.changed = False

    @property
    def span(self) -> Tokens:
        return Tokens(self[self.start : self.end])

    def replace(self, rep: Tokens) -> None:
        self[self.start : self.end] = rep
        self.end = self.start + len(rep)
        self.changed = True

    def __str__(self) -> str:
        return ''.join(tok.value for tok in self)

    def __repr__(self) -> str:
        s = str(self)
        more = "'..." if len(s) > 20 else ''
        return f"<Tokens {s!r:.20}{more}>"

# TokType
# An enum class for the .type of a token, where the members are TokAttrs objects.
# The enum member is a str subclass with its name as the value.
# Thus TokType[name] can be used like name, such as a key in another dict.
# The members have other attributes to get properties of a token from its type.  Such as TokType.CPP_WS.ws is True.

class TokAttrs(str):
    def __new__(cls, value: str):
        return super().__new__(cls, value)

    def __init__(self, value):
        self.ws = self in ws_tokens
        self.space = self in space_tokens
        self.comment = self in comment_tokens
        self.nl = self == 'CPP_NEWLINE'
        self.err = self == 'error'
        super().__init__()

TokType = enum.Enum('TokType', {tok:tok for tok in tokens}, type=TokAttrs)

class Hide(frozenset[str]):

    def __new__(cls, names: Set = frozenset()):
        return super().__new__(cls, names)

    def __or__(self, other: Set) -> Hide:
        return Hide(super().__or__(other))

    def __and__(self, other: Set) -> Hide:
        return Hide(super().__and__(other))

    def __repr__(self) -> str:
        if self:
            return f"{{{', '.join(sorted(self))}}}"
        else:
            return "{}"

class PpTok(LexToken):
    hide: set[str] = Hide()            # Macro names to suppress expansion.

    def add_hide(self, name: str) -> None:
        """ Add the name to the hide set """
        self.hide |= {name}

    def __repr__(self) -> str:
        rep = f"{self.value!r} @ {self.lineno}:{self.colno}"
        if self.hide:
            rep = f"{rep} - {self.hide!r}"
        return rep

class PpLex(Lexer):
    """ This is a subclass of lex.Lexer.
    It performs translation phases 1 and 2 (C99 Section 5.1.1.2) on the input data.
    It does tokenizing (phase 3) using global variables to define the states, rules, etc.
    It maintains the current line number (the base class has a line number but relies on
        rule actions to advance it).  Also provides column position within the current line.
    Note that digraph sequences are not replaced.  Rather, they are treated as punctuator tokens
        in a context where a punctuator is possible.  Within a character constant or a string,
        a digraph is just two characters.  See C99 Section 6.4.6, Para 3.
    """
    all_ws: bool                # The current line has only whitespace so far.
                                # Useful for detecting the first non-whitespace token.
    is_ws: bool                # Current token is whitespace.
    cuts: _Cuts                 # Places where the original data was shortened.

    def __new__(cls, prep: Preprocessor = None, from_lexer: Lexer = None, *args, source: str = '', **kwds) -> Self:
        if from_lexer:
            lexer = from_lexer.clone()
            lexer.__class__ = cls
            return lexer
        else:
            return super().__new__(cls)

    def __init__(self, prep: Preprocessor, *args, **kwds):
        self.prep = prep
        self.input('', source=prep.topsource)
        pass

    def __call__(self, text: str) -> None:
        self.input(text)
        return iter(self)

    def input(self, data: str, source: Source = None):
        self.cuts = self._Cuts(self)
        super().input(self._do_cuts(data))
        self.linepos = 0
        if source: self.source = source
        self.all_ws = True

    def parse(self, data: str) -> Iterable[LexToken]:
        lex = self.clone()

    def token(self) -> PpTok | None:
        self.is_ws = False      # Will be set to True by whitespace().
        colno = self.colno      # Get this before the token is parsed.
        tok: LexToken | None = super().token()
        if tok:
            tok.__class__ = PpTok
            tok.lexer = self 
            try: tok.type = TokType[tok.type]
            except KeyError: pass
            tok.colno = colno

        if self.is_ws:
            self.is_ws = False
        elif not tok or tok.value != '\n':
            self.all_ws = False
        if tok:
            newline = tok.value == '\n'
            self.cuts.moveto(self.lexpos, newline)
            tok.lineno = self.adjust_line() - newline
            tok.source = self.source
            if self.source.filename == '< top level >': print('----', tok.value)
        return tok

    def null(self) -> PpTok:
        lex = self.clone()
        lex.input(' ')
        tok = lex.token()
        tok.type = TokType.CPP_NULL
        tok.value = ''
        return tok

    def newline(self):
        """ A new line ('\n') is tokenized. """
        self.lineno += 1
        self.all_ws = True

    def whitespace(self, t: LexToken) -> LexToken:
        """ A whitespace token is tokenized. """
        self.is_ws = True
        return t

    def skip(self, n):
        """ Skip ahead n characters and note any newlines and cuts passed.
        Mainly used to skip over a character that is not the start of a token.
        """
        assert n >= 0, f"Trying to use skip({n}) to move backward."
        pos = self.lexpos
        super().skip(n)
        while pos < self.lexpos:
            if self.lexdata[pos] == '\n':
                self.newline()
            pos += 1
        self.cuts.moveto(pos, False)

    def erase(self, tok: LexToken) -> None:
        """ Change token into an empty whitespace, which will eventually be discarded,
        without altering the token list it is a part of.
        """
        tok.type = TokType.CPP_WS
        tok.value = ''

    def fix_paste(self, tok: PpTok) -> Iterable[PpTok]:
        """ Handle token resulting from a ## operator.
        If value is for a valid token, set its type.
        Otherwise, make two or more tokens parsed from the value.
        Generates the token(s).
        """
        self.input(tok.value)
        toks = list(self)

        if len(toks) != 1:
            # Insert spaces between the tokens.
            for j in reversed(range(1, len(toks))):
                self.input(' ')
                toks[j : j] = list(self)
            self.prep.on_error_token(tok,
                f"Pasting result {tok.value!r} is not a valid token.")
        else:
            tok.type = toks[0].type

        yield from toks

    def separate(self, left: LexToken, right: LexToken) -> bool:
        """ True if pasting the two tokens will parse differently from
        the tokens, thus requiring whitespace between them.
        See https://github.com/ned14/pcpp/issues/29, which is fixed by this.
        """

    def adjust_pos(self) -> int:
        return self.lexpos + self.cuts.addpos

    def adjust_line(self) -> int:
        return self.lineno + self.cuts.addline

    @property
    def colno(self) -> int:
        """ Column number (starting at 1) for current position.
        A line splice sets the colno back to 1.
        """
        return self.lexpos - self.linepos + 1

    def _do_cuts(self, input: str) -> str:
        """ Find all the shortenings of the input for translation phases 1 and 2.
        These are:
            - the 9 trigraphs, which are 3 characters and are replaced with one character.
                This was a way of encoding source characters which were not in in a 6-bit
                character set.
            - line splice, which is a '\\\n' sequence which is simply deleted.
            - A combination, which uses a trigraph for the backslash, as '??/\n'.
                All 4 characters are deleted.
        """
        if not self._cuts_pat.search(input):
            return input
        cuts: list[Self._Cut] = []

        def replace(m: re.Match) -> str:
            """ Replace the matched cut sequence with single character or nothing.
            Also, note where the replacement took place in the data before replacement.
            """
            old = m.group()
            new = self._cuts_rep[old]
            cuts.append(self._Cut(m.start(), len(old) - len(new), not new))
            return new

        shorter = self._cuts_pat.sub(replace, input)
        cuts.append(self._Cut(len(input) + 1, 0, False))
        self.cuts.set_cuts(cuts)
        return shorter

    _cuts_pat = re.compile(r'\?\?([=\'\(\)\!<>\-]|/\\?\n)|\\\n')
    _cuts_rep = {
        '??=':'#',      '??/':'\\',     "??'":'^',
        '??(':'[',      '??)':']',      '??!':'|',
        '??<':'{',      '??>':'}',      '??-':'~',
        '\\\n':'',      '??/\n':'',
    }

    class _Cuts(list):
        """ Manages places in the data where it was cut by replacing a substring
        with a shorter string.
        The purpose is to adjust position and line number based on the current position.
        Tracks forward movement in the current position.
        Maintains position of the current line of the original text.  If lines were
        spliced together, then the position after the splice is a new line start.
        Optimized when there are no cuts.
        """
        def __init__(self, lexer: Lexer):
            # Set pass-through routines for the case where there are no cuts.
            # Delete these if there are cuts, to revert to the class method.
            self.lexer = lexer
            self.addpos = self.addline = 0
            self.adjust_pos = lambda : self.lexer.lexpos
            self.adjust_line = lambda : self.lexer.lineno
            pass

        def set_cuts(self, cuts: list[_Cut]) -> None:
            """ Record one or more Cuts made.  The last cut marks the end of the data plus 1. """
            del self.adjust_pos, self.adjust_line
            reduction = 0
            for i, cut in enumerate(cuts):
                cut.where -= reduction
                reduction += cut.reduction
            self.cuts_passed = 0
            self.lexer.linepos = 0
            self.nextpos = cuts[0].where
            self.nextcut = cuts[0]
            self[:] = cuts
            pass

        def moveto(self, pos: int, newline: bool):
            """ Change current position in reduced data.
            Update self.addpos and self.addline and lexer.linepos.
            Note the last Cut which has been passed.
            """
            if newline:
                self.lexer.linepos = pos
            if not self: return
            while self.nextpos <= pos:
                cut = self.nextcut
                if cut.joined:
                    self.addline += 1
                    self.lexer.linepos = cut.where
                self.addpos += cut.reduction
                self.cuts_passed += 1
                cut = self.nextcut = self[self.cuts_passed]
                self.nextpos = cut.where

    @dataclasses.dataclass
    class _Cut:
        where: int              # Position in original data before the cut.
                                # After adjustment, position in shortened data.
        reduction: int          # How much shorter the data was made.
                                # 2 for most things, 4 for ??/_n.
        joined: bool            # If this was a line join --- \\\n or ??/\n.

def default_lexer():
    return lex.lex(optimize=in_production)

""" Handling whitespace...
A whitespace token is any of:
    Space or tab characters.
    Comment.
    Newline.
A backslash-newline sequence in the input has been removed already.
When doing macro expansion:
    Leading and trailing whitespace in a function argument is ignored.
    However, this applies only to variable args as a group.
    Thus, ( a , b ) matching ... becomes 'a , b'.
    This is only relevant with # arg.  Initial and final whitespace is
    deleted and consecutive whitespace elsewhere is replaced by a single ' '.
    Otherwise, including the ## operator, all whitespace is deleted from the args.
In phase 3, whitespace is condensed.  Newlines are kept.
Each comment is replaced by a single space.  However, the passthru_comments flag
will keep comments as they are.
"""
def no_ws(tokens: Iterable[LexToken]) -> Iterable[LexToken]:
    """ Filters out all whitespace tokens from given tokens, inclusing newlines.
    Consumes the input if it is an iterator.
    """
    # Get the iterator.
    it: Iterable[LexToken] = iter(tokens)

    tok: LexToken
    for tok in it:
        if not tok.type.ws:
            yield tok

def reduce_ws(tokens: Iterable[LexToken], prep: Preprocessor) -> Iterable[LexToken]:
    """ Compresses whitespace, as in translation phase 3.
    However, comments will be preserved if the Preprocessor says so.
    Consumes the input if it is an iterator.
    """
    # Get the iterator.
    it: Iterable[LexToken] = iter(tokens)

    tok: LexToken
    have_ws: bool = False
    for tok in it:
        if (tok.type.ws and not tok.type.nl
            and (not tok.type.comment
                 or not prep.on_comment(tok)
                 )
            ):
            # This is whitespace to compress
            if have_ws: continue
            # First whitespace token.
            have_ws = True
            tok.value = ' '
        else:
            have_ws = False

        yield tok

def merge_ws(tokens: Iterable[LexToken]) -> Iterable[LexToken]:
    """ Reduces all consecutive whitespace, including newlines and comments,
    to a single space.
    """
    # Get the iterator.
    it: Iterable[LexToken] = iter(tokens)

    tok: LexToken
    pend_ws: LexToken = None

    for tok in it:
        if tok.type.ws:
            if not pend_ws:
                pend_ws = tok
        else:
            if pend_ws:
                t = copy.copy(pend_ws)
                t.value = ''
                yield t
                pend_ws = False
            yield tok
    if pend_ws:
        t = copy.copy(pend_ws)
        t.value = ''
        yield t

