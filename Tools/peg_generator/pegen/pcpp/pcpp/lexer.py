""" lexer.py
Defines a lexer, to use with ply.lex, to turn input string into tokens.
"""

# TODO: Treat comments as non-whitespace if passthru comments and gcc.
#   This hides a following directive and the # is an ordinary token.
#   Otherwise treat comments as whitespace.

from __future__ import annotations

import codecs
import dataclasses
from dataclasses import replace
import enum
import re
import os.path
from functools import cached_property
from itertools import count, product, zip_longest
import operator
from typing import Callable, NewType, TypeVar, Union

from pcpp.ply import lex
from pcpp.ply.lex import LexToken, Lexer, LexError, TOKEN

from pcpp.common import *
#from pcpp.dfltlexer import *
from pcpp.regexes import *
from pcpp.replacements import *
from pcpp.tokens import (PpTok, RawTok, Tokens, TokIter, TokLoc, TokenSep)
from pcpp.directive import Directive
from pcpp.writer import OutPosFlag, OutPosChange, OutPosMove, OutPosEnter


class Lines:
    """
    A mixin class for the PpLex lexer, which tracks lines as tokens are lexed
    from the data.  Attributes lexdata, lexlen, lexpos, and lineno are
    contained in the lex.Lexer class, the base class of PpLex.

    When a token is lexed by lex.Lexer, self.lexpos and self.lineno are copied
    to the token, and self.lexpos is bumped up by the length of the
    token.value.  It is up to the Lines class to update self.lineno and any
    other variables.  This is done by the update_lines(token) method.
    """

    # Position of first character in current physical line.  Beginning of
    # file, or after last line splice or newline character.
    linepos: int

    # Change of position after replacements in current line.
    repl_delta: int

    # Position after next newline char at or after linepos.  Will become
    # linepos when lexpos reaches this point, starting a new logical line.
    next_linepos: int

    # Number of physical lines within the logical line.
    phys_lines: int

    # The current logical line has only whitespace so far.  Useful for
    # detecting the first non-whitespace token.  Such as the '#' in a
    # directive.
    only_ws_this_line: bool = True

    @property
    def colno(self) -> int:
        """
        Column number (starting at 1) for current lexer position in current
        physical line.  A line splice sets the colno back to 1 by setting
        linepos = splice position.
        """
        return self.lex.lexpos - self.linepos + self.repl_delta + 1

    def input(self, data: str) -> None:
        """ Set the input data here and in the lex.Lexer class.  Translation
        phase 1 and 2 replacements have already been made in the data, and
        self.repls records all of them.
        """
        # Initialize in lex.Lexer class.  Sets lexdata, lexlen, and lexpos.
        data = super().input(data)
        self.lineno = 1
        self.linepos = 0
        self.repl_delta = 0
        self.next_linepos = data.find('\n') + 1
        self.only_ws_this_line = True
        self.phys_lines = 1

    def update_lines(self, tok: PpTok) -> None:
        """
        Bring internal state up to date after given token, and also set some
        token attributes.  lexpos is now at the end of the value.
        """
        def find_nl() -> int:
            """
            Position after next newline character after self.linepos, or
            lexlen if not found.
            """
            return self.data.find('\n', self.linepos) + 1 or len(self.data) + 1

        # Did the token cross a newline?
        if self.lex.lexpos >= self.next_linepos:
            # Token could be a newline token, or it could be some other type
            # with newlines embedded within it.
            if tok.type.nl:
                self.lineno += self.phys_lines
                self.linepos = self.next_linepos
                self.repl_delta = 0
                self.next_linepos = find_nl()
                self.phys_lines = 1
                self.first_ws = None
                self.only_ws_this_line = True
            else:
                # Newline(s) embedded in token value.
                pos = self.next_linepos
                while self.lex.lexpos >= pos:
                    self.phys_lines += 1
                    self.linepos = pos
                    pos = find_nl()
                self.repl_delta = 0
                self.next_linepos = pos

    def newphys(self, nextpos: int) -> None:
        """
        Start of a new physical line, same logical line.
        """
        self.phys_lines += 1
        self.linepos = nextpos
        self.repl_delta = 0


class RawLexer:
    """
    Low level interface to the lex.Lexer class.  It accepts a data string,
    which might be the contents of a source file, or otherwise artificially
    constructed.

    It performs translation phases 1 and 2 on the input data.  The input data
    is provided separately by the input() method.

    Produces raw tokens serially from the data.  Also provides an iterator to
    do this.  The raw token does not have complete PpTok information.  It does
    provide enough information to analyze the tokens contained in an arbitrary
    string.
    """
    # Enable error messages during lexing.  Can be set by subclass.
    errors: bool = False

    # Places where the original data was altered in phase 1 or 2.
    repls: ReplMgr = EmptyReplMgr()

    def __init__(self, lex: Lexer, *args, **kwds):
        self.prep = prep = lex.prep
        self.REs = lex.REs
        self.TokType = lex.TokType
        #self.clones = []
        self.source = prep.currsource

    def input(self, data: str) -> str:
        """ Begin lexing.  Do transforms on the data, return new data. """
        repls = ReplMgr(self)
        data = repls.do_repls(self, data)
        if repls: self.repls = repls
        self.data = data
        self.lex.input(data)
        return data

    def parse(self, data: str, raw: bool = True) -> Iterable[RawTok]:
        """
        Iterate over the tokens in given data, using only the superclass
        methods and a clone of self.
        """
        lex: RawLexer
        with self.cloned(errors=False, cls=RawLexer) as lex:
            lex.input(data)
            tok: RawTok = next(lex.raw_tokens())
            yield tok

    def raw_token(self) -> RawTok | None:
        """
        Next raw token, if any, in entire data string without any other
        intervention.  Using superclass methods.
        """
        t: LexToken | None = self.lex.token()
        if not t: return None
        return RawTok(self, t)

    def raw_tokens(self) -> Iterator[RawTok]:
        """
        All raw tokens in entire data string without any other intervention.
        Using superclass methods.
        """
        while True:
            t: LexToken | None = self.lex.token()
            if not t: return
            tok = RawTok(self, t)
            yield tok

    def skip(self, n):
        """ Skip ahead n characters and note any newlines and cuts passed.
        Mainly used to skip over a character that is not the start of a token.
        """
        assert n >= 0, f"Trying to use skip({n}) to move backward."
        pos = self.lex.lexpos
        self.lex.skip(n)
        while pos < self.lex.lexpos:
            if self.data[pos] == '\n':
                self.newline(pos)
            pos += 1

    def spelling(self, start: int, stop: int) -> str:
        return self.data[start : stop]


class PpLex(Lines, RawLexer):
    """ This is a subclass of lex.Lexer.
    It performs translation phases 1 and 2 on the input data.  It
    does tokenizing (phase 3) using global variables to define the states,
    rules, etc.

    It maintains the current line number (the base class has a line number but
    relies on rule actions to advance it).  Also provides column position
    within the current line.

    Note that digraph sequences are not replaced.  Rather, they are treated as
    punctuator tokens in a context where a punctuator is possible.  Within a
    character constant or a string, a digraph is just two characters.
    """
    # Object representing the source file inclusion.
    source: Source = None

    lineno: int                 # Line number.

    move: MoveTok = None        # Last change in output position.

    errors: bool                # Enable error messages during tokens().

    lex: Lexer                  # Original lexer this came from.
    clones: list[PpLex] = []    # Any available clones of self.lex.

    def __new__(cls, *, from_lexer: Lexer = None,
                **kwds
                ) -> Self:
        lexer = super().__new__(cls)
        if from_lexer:
            lex = from_lexer.clone()
            lexer.lex = lex
            lexer.clones = []
        return lexer

    def __init__(self, *, lex: Lexer = None, **kwds):
        if not lex:
            lex = self.lex
        super().__init__(lex)

    def input(self, data: str, source: Source = None):
        """
        Set the data to be lexed.  Performs Translation Phase 1 and 2
        replacements.  Sets the source object, if given.
        """
        self.source = source
        super().input(data)
        self.linepos = 0
        #if source:
        #    self.source = source
        #    self.filename = source.filename
        self.only_ws_this_line = True

    @contextlib.contextmanager
    def cloned(self, cls: Type = None, errors: bool = True) -> PpLex:
        """
        A new lexer of same class, or given class.  Uses a clone of self.lex.

        Can turn off error reporting during the context.
        """
        try:
            clone = self.clone(cls)
            clone.errors = errors
            yield clone
        finally:
            clone.errors = True 
            self.clones.append(clone.lex)

    def clone(self, cls: type = None) -> Self:
        """ Make a clone of the prep's PpLex.  It won't be reusable. """
        lex = self.lex
        clones = self.clones
        if clones:
            lex = clones.pop()
        else:
            lex = lex.clone()
        clone = (cls or type(self))(lex=lex)
        clone.lex = lex
        lex.owner = clone
        clone.clones = clones
        return clone

    @contextlib.contextmanager
    def setstate(self, state: str) -> None:
        """ Set the lexing state during the context, then restore it. """
        old = self.lexstate
        self.begin(state)
        try: yield
        finally:
            self.begin(old)

    @contextlib.contextmanager
    def seterrors(self, errors: bool = True) -> None:
        """
        Set the error message enabling during the context, then restore it.
        """
        old = self.errors
        self.errors = errors
        try: yield
        finally:
            self.errors = old

    def make_token(self, type: TokType, *, value: str = None,
                   loc: TokLoc = None, cls: type = PpTok,
                   **attrs
                   ) -> PpTok:
        """
        Make a new token with given type and optional value, and any other
        attributes desired.  The default value varies with the type.  The
        token is at the current location of self, by default, or `loc` if not
        None.
        """
        if value is None: value = type.val
        if loc is None: loc = self.loc
        return cls(self, type=type, value=value, loc=loc, **attrs)

    @property
    def loc(self) -> TokLoc:
        """ Current location, which will be stored in a new token. """
        loc = TokLoc(lineno=self.lineno, colno=self.colno, source=self.source,
                     datapos=self.lex.lexpos, move=self.move)

        phys_offset = self.phys_lines - 1
        if phys_offset:
            loc = replace(loc, phys_offset=phys_offset)
        return loc

    def nexttok(self) -> PpTok | None:
        """ Get the next token from lexer, or None.  Skip whitespace and
        newlines, but these are reflected in later tokens."""

        loc = self.loc
        t: LexToken | None = self.lex.token()
        if not t: return None
        tok: PpTok = PpTok(self, t)
        tok.loc = loc
        repls = self.repls.movetotoken(tok)
        self.update_lines(tok)
        newline = tok.type.nl
        if tok.type.ws:
            if newline:
                self.only_ws_this_line = True
        else:
            self.only_ws_this_line = False

        if repls:
            if tok.type.revert:
                tok = tok.revert(tok.type)

        if newline:
            self.phys_lines = 1
        elif '\n' in tok.value:
            pass
            # Multiline token.  
            # GCC will write leading space if it is at the start of the
            # line.
            #adjust_line: bool = not (tok.type.comment and self.prep.emulate)
            #self.newlines_in_token(tok, adjust_line=adjust_line)
        #if tok.brk():
        #    print(f'--- {tok.source} {tok!r} pos = '
        #          f'{tok.datapos} .. {self.lex.lexpos}')
        return tok

    def parse_tokens(self, data: str) -> Iterable[PpTok]:
        """
        Iterate over the tokens in given data, using a clone of self.  
        """
        lex: PpLexer
        with self.cloned(errors=False) as lex:
            lex.input(data)
            yield from lex.tokens()

    #def parse(self, input: str) -> TokIter:
    #    """ Get iterator of tokens from input string. """
    #    lex: PpLex
    #    with self.cloned(errors=False, cls=PpLex) as lex:
    #        lex.input(input)
    #        yield from lex.tokens()

    @TokIter.from_generator
    def tokens(self, errors: bool = True
               ) -> Iterator[PpTok]:
        """
        Generate all tokens for the entire data string.  Includes indent tokens
        for logical lines.  Translation Phase 1 and 2 replacements have been
        done already.
        """
        tok: PpTok = None
        prev: PpTok = None
        #toks: Iterable[PpTok] = self.raw_tokens(errors)

        self.errors = errors
        ws = False              # Whitespace seen since last token or newline.
        lineno: int = 0         # Line # of last token seen.

        def nexttok() -> PpTok | None:
            """
            Get the next token from lexer, or None.  Skip whitespace other than
            newlines, but these are reflected in next returned token.
            """

            nonlocal prev, ws

            while True:
                tok: PpTok = self.nexttok()
                if tok:
                    newline = tok.type.nl
                    if tok.type.ws:
                        if newline:
                            # newline: forget previous ws, and return token.
                            ws = False
                            prev = None
                            tok.sep = TokenSep.create()
                        else:
                            # other whitespace: remember and go to next token.
                            ws = True
                            continue
                    else:
                        # Normal token.
                        self.only_ws_this_line = False
                        tok.sep = TokenSep.create(spacing=ws)
                        ws = False
                return tok
            # end of nexttok()

        while True:
            tok = nexttok()
            if not tok: break
            tok.prev = prev
            prev = tok
            if tok.type.dir:
                # Special handling for a directive.
                dir = tok
                dir.line = Tokens([tok.copy(type=self.TokType.CPP_POUND)])
                ws = False
                tok = nexttok()
                while not tok.type.nl:
                    dir.line.append(tok)
                    tok = nexttok()
                if self.prep.clang and self.source.in_macro:
                    if self.source.in_macro.args is None:
                        yield dir.make_null()
                dir.dir = Directive(dir)
                yield dir
                continue
            # Not directive.
            # Check for change in line number.
            if tok.lineno != lineno:
                tok.sep = TokenSep.create(indent=tok.loc)
                lineno = tok.lineno
            if not tok.type.nl:
                yield tok

    def make_passthru(self, toks: Iterable[PpTok]) -> PpTok:
        """ A CPP_PASSTHRU token at current location. """
        return self.make_token(self.TokType.CPP_PASSTHRU, toks=toks, )

    def newlines_in_token(
            self, tok: PpTok, *,
            #adjust_line: bool = True,           # Add to self.phys_lines.
            nl_pat=re.compile(r'(?<![\\])\n'),  # Unescaped newline regex.
            ) -> None:
        """ Account for all (unescaped) newline characters in token. """
        for m in re.finditer(nl_pat, tok.value):
            self.newphys(tok.datapos + m.start() + 1)

    def try_paste(self, lhs: PpTok, rhs:PpTok) -> PpTok | None:
        """
        Try to paste two tokens together.  Returns new token at the location
        of the left token (or the right token if left is a placemarker).
        Merges the two hide sets.  Returns None if not a valid token result.
        """
        lex: PpLex
        if lhs.type.null: return rhs
        if rhs.type.null: return lhs

        value = lhs.value + rhs.value
        with self.cloned(errors=False) as lex:
            lex.input(value)
            # Using this string as input data, lex the first token using base
            # class lexer.  This should return a LexToken with exactly the
            # same value, and not an error type.  Otherwise the input data is
            # not valid.  Could call lex.token(), but this way is faster.
            t: LexToken | None = lex.lex.token()
            typ = self.TokType[t.type]
            if not t or len(value) != lex.lex.lexpos or typ.err:
                # Failure to match the value.
                self.prep.on_error_token(lhs,
                    f"Pasting result {value!r} is not a valid token.")
                return None

            return lhs.copy(value=value, type=typ,
                            hide=lhs.hide and rhs.hide
                            and lhs.hide & rhs.hide,
                            )

    def fix_paste(self, tok: PpTok) -> PpTok | None:
        """ Handle token resulting from a ## operator.
        If entire value is for a valid token, set its type and return it,
        otherwise return None.
        """
        lex: PpLex
        with self.cloned(errors=False) as lex:
            lex.input(tok.value)
            # Using given string as input data, lex the first token 
            #   using self.lex.  This should return a LexToken with exactly
            #   the same value, and not an error type.
            #   Otherwise the input data is not valid.
            #   Could call lex.token(), but this way is faster.
            t: LexToken | None = lex.lex.token()
            typ: self.TokType = t and self.TokType[t.type] or tok.type
            if not t or len(tok.value) != lex.lex.lexpos or typ.err:
                # Failure.
                self.prep.on_error_token(tok,
                    f"Pasting result {tok.value!r} is not a valid token.")
                return None
            tok.type = typ
            return tok

    def separate(self, left: str, right: str) -> bool:
        """ True if two tokens with given values need a separating space. """
        lex: PpLex
        with self.cloned(errors=False) as lex:
            lex.input(left + right)
            tok: LexToken           # Might be a PpTok subclass.
            if lex.repls:
                tok = next(lex.tokens()).revert()
                return tok.value != left
            else:
                # Could call lex.token(), but without input() replacements, 
                # this way is faster and has same result
                lex.lex.token()
                # First token, value should match left.
                return len(left) != lex.lex.lexpos

    def brk(self) -> bool:
        """ Break condition for debugging. """
        return break_match(line=self.lineno, col=self.colno,
                           pos=self.lex.lexpos, file=self.source.filename)

    def __repr__(self) -> str:
        return f"<PpLex {self.loc}>"


def default_lexer(prep: Preprocessor) -> PpLex:
    """ Creates a single PpLex lexer that handles everything,
    along with clones.  Each clone handles a single input string, and some
    clones may be used (serially) for different inputs.

    The details of the PpLex vary with attributes of the prep provided.  This
    function can be called repeatedly to make different lexers.
    """
    from pcpp.dfltlexer import default_lexer as dfltlex
    newlex: PpLex = dfltlex(prep)
    return newlex
    # -------------------------------------------------------------------------
    # Default preprocessor lexer definitions.   These tokens are enough to get
    # a basic preprocessor working.   
    # Other modules may import these if they want.
    # -------------------------------------------------------------------------

    # Special tokenizing for preprocessing directives, using lexer states.
    #   A '#' after possible whitespace, at the start of a line, introduces a
    #       directive.  Token type is CPP_DIRECTIVE, and new state = DIRECTIVE.
    #   Following whitespace is skipped.  The next token should be the
    #       directive name, a CPP_ID.
    #       If it is 'include', type = CPP_INCLUDE and new state is INCLUDE.
    #           Then looks for CPP_H_HDR_NAME or CPP_Q_HDR_NAME.
    #       If it is 'define', type = CPP_DEFINE and new state is DEFINE.
    #           Then looks for a CPP_OBJ_MACRO or CPP_FUNC_MACRO.
    #           After that, it enters the MACREPL state, where any ## tokens
    #           become CPP_PASTE, rather than CPP_DPOUND.
    #       If it is 'if' or 'elif', new state is CONTROL.  Same as INITIAL
    #           except that CHAR tokens will not revert \U or \u escapes.
    #       Any other token of any type goes back to the INITIAL state.
    #   Reaching a newline goes back to the INITIAL state.
    #   All tokens from the '#' up to, but not including, the next newline
    #       are put into a Tokens and set as (the # token).line.

    #states = [
    #    ('DIRECTIVE', 'inclusive'),
    #    ('INCLUDE', 'inclusive'),
    #    ('DEFINE', 'inclusive'),
    #    ('CONTROL', 'inclusive'),
    #    ]

    #tokens = PpTok.type_names

    ## Some common REs that are incorporated into other REs...
    #REs = RegExes(prep)

    ## Note, unmatched quote characters and misplaced backslashes are a token,
    ##  not a literal.  GCC also recognizes $, @, and ` as source characters.
    #literals = "+-*/%|&~^<>=!?()[]{}.,;:"
    #if prep.emulate: literals += REs.ascii_not_source

    #""" Lex Rules ...
    #A lex rule is given to lex.lex() by means of creating an variable in the
    #closure (i.e., the body of default_lexer().  The name of the variable is
    #the name of the rule, and has the form 't_...'.  The value of the variable
    #is either a string denoting the regex to be matched, or a function f(t:
    #LexToken) -> LexToken, where f.regex is the regex.

    #Rules are stored in the dictionary `rules`, where rules[name] = rule.
    #`name` is the rule name without the leading 't_'.

    #The Lexer tries to match function rules in the order they appear in rules,
    #then it tries to match string rules.
    #"""
    #StrRule = NewType('StrRule', str)
    #FuncRule = Callable[[LexToken], LexToken]
    #Rule = Union[StrRule, FuncRule]

    #rules: Mapping[str, Rule] = {}

    #def funcrule(regex: str, name: str = None) -> Callable[[FuncRule], FuncRule]:
    #    """
    #    Decorator for a function f(t: LexToken) -> Lextoken.
    #    """
    #    def func(f: FuncRule) -> FuncRule:
    #        f.regex = regex
    #        if name:
    #            fname = f.__name__ = f't_{name}'
    #        else:
    #            fname = f.__name__
    #        rules[fname] = f
    #    return func

    #def puncrule(*values: str) -> Callable[[FuncRule], FuncRule]:
    #    """
    #    Decorator for a function f(t: LexToken) -> LexToken.

    #    A proxy for f is added to rules[name of f].  Name is modified for
    #    extra values.  punc_values[value] = name.
    #    """
    #    def func(f: FuncRule) -> FuncRule:
    #        fname = f.__name__
    #        punc(fname[2:], *values, proxy=f)
    #    return func

    ## These rules can be looked up by any matching string.
    #punct_values: Mapping[str, str] = {}    # value -> name for each value.
    
    #funcs: list[Callable[[LexToken], LexToken]] = []

    ## Token rules other than for punctuators...

    ## Whitespace, one or more consecutive whitespace character(s) 
    ## other than newline.
    #t_ANY_CPP_WS = r'((?!\n)\s)+'

    ## Special newline in a directive.  Returns to INITIAL state.

    ## Place before the newline rule below!
    #@TOKEN(REs.newline)
    #def t_DIRECTIVE_INCLUDE_DEFINE_CONTROL_CPP_NEWLINE(t):
    #    t.lexer.begin('INITIAL')
    #    return t_ANY_CPP_NEWLINE(t)

    ## Newline, other than in a directive.  Advances line number eventually.
    #@TOKEN(REs.newline)
    #def t_ANY_CPP_NEWLINE(t):
    #    return t

    #def t_DIRECTIVE_CPP_ID(t):
    #    r'[A-Za-z_][\w_]*'
    #    if t.value == 'include':
    #        t.lexer.begin('INCLUDE')
    #    elif t.value == 'define':
    #        t.lexer.begin('DEFINE')
    #    elif t.value.endswith('if'):
    #        t.lexer.begin('CONTROL')
    #    else:
    #        t.lexer.begin('INITIAL')
    #    return t
    #_string_literal_linecont_pat = re.compile(r'\\[ \t]*\n')

    ## A '##' rule has to come before the '#' rule.
    #punc('CPP_DPOUND', '##', '%:%:',       func=True)

    ## A '#', if the first non-whitespace in a line, is a directive.
    ##@puncrule('#', '%:')
    #def t_CPP_POUND(t: PpTok) -> PpTok:
    #    r'\#|%:'
    #    # A PpLex indicates if at the start of a line, RawLexer does not.
    #    try:
    #        if t.lexer.owner.only_ws_this_line:
    #            t.lexer.begin('DIRECTIVE')
    #            t.type = 'CPP_DIRECTIVE'
    #    except AttributeError: pass
    #    return t

    ## Identifier 
    #t_CPP_ID = REs.ident

    ## Object and function macro identifiers.  
    ## CPP_FUNC_MACRO is the macro name, if followed immediately by '('.
    ## CPP_OBJ_MACRO is the macro name, otherwise.
    #@TOKEN(rf'{REs.ident}(?=\()')
    #def t_DEFINE_CPP_FUNC_MACRO(t):
    #    t.type = 'CPP_FUNC_MACRO'
    #    t.lexer.begin('INITIAL')
    #    return t
    ## Place this AFTER FUNC_MACRO!
    #@TOKEN(REs.ident)
    #def t_DEFINE_CPP_OBJ_MACRO(t):      
    #    t.type = 'CPP_OBJ_MACRO'
    #    t.lexer.begin('INITIAL')
    #    return t

    ## Floating literal.  Put these before integer.
    #makefunc('CPP_FLOAT', REs.float)
    #makefunc('CPP_DOT_FLOAT', REs.dotfloat)

    ## Integer constant 
    #makefunc('CPP_INTEGER', REs.int)

    ## General pp-number, other than integer or float constant.  (C99 6.4.8,
    ## C++14 5.9).  Put this after integer and float.
    #makefunc('CPP_NUMBER', REs.ppnum)

    ## String literal.  # Terminating " required on same logical line.
    #t_CPP_STRING = REs.string

    ## Raw string literal.  
    ## Terminating matching delimiter required, possibly on later logical line.
    ## Only tokenized if C++ or (C with GNU extensions).

    ##if prep.cplus_ver or prep.emulate:
    ##    @TOKEN(REs.rstring)
    ##    def t_CPP_RSTRING(t):
    ##        # Special handling for raw strings.  (C++14 5.4 (3.1)).

    ##        # The transformations in phases 1 and 2 have been made already,
    ##        # but they must be reverted.  Since GCC also replaces trigraphs,
    ##        # these are also reverted.  
    ##        return t

    ## h-type and q-type header names.  Only used in INCLUDE state.  
    ## Note, some things in these names are undefined behavior (C99 6.4.7), and
    ## this is checked in the preprocessor.include() method.

    #t_INCLUDE_CPP_H_HDR_NAME = REs.hhdrname
    #t_INCLUDE_CPP_Q_HDR_NAME = REs.qhdrname

    ## Character constant (L|U|u|u8)?'cchar*'.  # Terminating ' required.
    #@TOKEN(REs.char)
    #def t_CPP_CHAR(t):
    #    return t

    ## Same, within a CONTROL expression.  yacc evaluates this differently.
    #t_CONTROL_CPP_EXPRCHAR = REs.char

    ## Block comment (C), possibly spanning multiple lines.  
    #t_CPP_COMMENT1 = r'(/\*(.|\n)*?\*/)'

    ## Line comment (C++).  PCCP accepts them in C files also.  
    #t_CPP_COMMENT2 = r'(//[^\n]*)'
    
    #def t_ANY_error(t):
    #    # Check for unmatched quote character.  
    #    if t.value[0] in '\'\"':
    #        endline = t.value.find('\n')
    #        t.value = t.value[:endline]
    #        message = f"Unmatched quote character {t.value}"
    #    else:
    #        t.value = t.value[0]
    #        message = f"Illegal character {t.value!r}"
    #    t.lexer.skip(len(t.value))
    #    return error(t, message)

    #def error(t: PpTok, msg: str, keep_type: bool = False) -> PpTok:
    #    if not keep_type:
    #        t.type = TokType.error
    #    if t.lexer.owner.errors:
    #        t.lexer.prep.on_error_token(t, msg)
    #    return t

    ## PUNCTUATORS ...

    ## Punctuator lexer rule has one or more fixed strings which it matches.
    #def set_punctuators() -> None:
    #    """ Add rules for most punctuator tokens. """

    #    # Arithmetic operators
    #    punc('CPP_PLUS',          '+')
    #    punc('CPP_PLUSPLUS',      '++')
    #    punc('CPP_MINUS',         '-')
    #    punc('CPP_MINUSMINUS',    '--')
    #    punc('CPP_STAR',          '*')
    #    punc('CPP_FSLASH',        '/')
    #    punc('CPP_PERCENT',       '%')
    #    punc('CPP_LSHIFT',        '<<')
    #    punc('CPP_RSHIFT',        '>>')

    #    # Logical operators
    #    #   Place && and || before & and |
    #    punc('CPP_LOGICALAND',    '&&',   'and',    func=True)
    #    punc('CPP_LOGICALOR',     '||',   'or',     func=True)
    #    punc('CPP_EXCLAMATION',   '!',    'not')

    #    # bitwise operators

    #    punc('CPP_AMPERSAND',     '&',   'bitand')
    #    punc('CPP_BAR',           '|',   'bitor')
    #    punc('CPP_HAT',           '^',   'xor')
    #    punc('CPP_TILDE',         '~',   'compl')

    #    # Comparison operators
    #    punc('CPP_EQUALITY',      '==')
    #    punc('CPP_INEQUALITY',    '!=',   'not_eq')
    #    punc('CPP_GREATEREQUAL',  '>=')
    #    punc('CPP_GREATER',       '>')
    #    punc('CPP_LESS',          '<')
    #    punc('CPP_LESSEQUAL',     '<=')
    #    punc('CPP_SPACESHIP',     '<=>')              # C++

    #    # Conditional expression operators
    #    punc('CPP_QUESTION',      '?')
    #    punc('CPP_COLON',         ':')

    #    # Member access operators
    #    punc('CPP_DOT',           '.')
    #    punc('CPP_DOTPTR',        '.*')               # C++
    #    punc('CPP_DEREFERENCE',   '->')
    #    punc('CPP_DEREFPTR',      '->*')              # C++
    #    punc('CPP_DCOLON',        '::')               # C++

    #    # Assignment operators
    #    punc('CPP_EQUAL',         '=')
    #    punc('CPP_XOREQUAL',      '^=',   'xor_eq')
    #    punc('CPP_MULTIPLYEQUAL', '*=')
    #    punc('CPP_DIVIDEEQUAL',   '/=')
    #    punc('CPP_PLUSEQUAL',     '+=')
    #    punc('CPP_MINUSEQUAL',    '-=')
    #    punc('CPP_OREQUAL',       '|=',   'or_eq')
    #    punc('CPP_ANDEQUAL',      '&=',   'and_eq')
    #    punc('CPP_PERCENTEQUAL',  '%=')
    #    punc('CPP_LSHIFTEQUAL',   '<<=')
    #    punc('CPP_RSHIFTEQUAL',   '>>=')

    #    # Grouping and separators
    #    punc('CPP_LPAREN',        '(')
    #    punc('CPP_RPAREN',        ')')
    #    punc('CPP_LBRACKET',      '[',    '<:')
    #    punc('CPP_RBRACKET',      ']',    ':>')
    #    punc('CPP_LCURLY',        '{',    '<%')
    #    punc('CPP_RCURLY',        '}',    '%>')

    #    # Unmatched quotes, and backslashes not before newline, will be
    #    # errors.  Matched quotes on the same line are part of CPP_CHAR or
    #    # CPP_STRING.  Backslash before newline is removed from input before
    #    #lexing.  punc('CPP_SQUOTE',       '\'') punc('CPP_DQUOTE',       '"')
    #    punc('CPP_BSLASH',        '\\')

    #    # Single-characters not in the source character set (valid in GCC).
    #    if prep.emulate:
    #        punc('CPP_DOLLAR',    '$')
    #        punc('CPP_AT',        '@')
    #        punc('CPP_GRAVE',     '`')

    #    # Miscellaneous

    #    punc('CPP_COMMA',         ',')
    #    punc('CPP_SEMICOLON',     ';')
    #    punc('CPP_ELLIPSIS',      '...')

    #set_punctuators()

    ## Add punctuator rules and lexer functions to local context, so that
    ## lex.lex() will find them.
    #locals().update(rules)
    #def addfunc(name: str, func: FuncRule) -> FuncRule:
    #    def proxy(t: LexToken) -> LexToken:
    #        return func(t)
    #    proxy.__name__ = name
    #    proxy.regex = func.regex
    #    return proxy

    #for name, f in rules.items():
    #    if hasattr(f, '__call__'):
    #        try: del locals()[name]
    #        except: pass
    #        locals()[name] = addfunc(name, f)
    #        locals()[name] = addfunc(name, f)
    #        x=0
    ## TokType. 
    ## Enumeration for each type of token.
    ## The enum member is a str subclass (TokAttrs) with its name as the value.
    ## Thus TokType[name] can be used like name, such as a key in another dict.
    ## The members have other attributes to get properties of a token 
    ##   from its type, such as TokType.CPP_WS.ws == True.

    #PpTok.TokAttrs.class_init(prep, REs)
    #TokType = enum.Enum('TokType', {tok:tok for tok in tokens},
    #                    type=PpTok.TokAttrs)

    #TokType.__repr__ = lambda obj: obj.value

    #prep.TokType = TokType

    ## Token type map name to the type, or a name of a type.
    ## Lexer token maps its own name to the TokType.
    ## Aliases map a name to another name or several names separated by spaces.
    #toktypedict = {name : TokType[name] for name in tokens}

    ## Now that TokType exists, set the typ.lit properties for all punctuator
    ## types.
    #for value, typ in punct_values.items():
    #    TokType[typ].lit = value

    ## Alias names.  Alias name -> one or more token names, as a string.
    ## These are added to toktypedict.

    #def alias(name: str, tok: str) -> None:
    #    toktypedict[name] = tok

    #alias('ANYNUM', 'CPP_INTEGER CPP_FLOAT CPP_DOT_FLOAT CPP_NUMBER')
    #alias('ANYCHAR', 'CPP_CHAR CPP_EXPRCHAR')
    #alias('ANYSTR', 'CPP_STRING CPP_RSTRING')
    #alias('ANYQUOTED', 'ANYSTR ANYCHAR')
    #alias('LITERAL', 'ANYNUM ANYCHAR ANYSTR')

    #'''
    #AVOIDING PASTE.
    #Here are rules for pairs of tokens which require whitespace separation,
    #to avoid creating text which would lex differently if written adjacently.
    #These mimic the rules found in GCC in avoid_paste() function.

    #Most cases are accomplished by looking at types, and not the values, 
    #of the tokens.
    #If left.type.sep_from contains right.type, then the result is true.

    #The rules for separating tokens are taken from the gcc file gcclib/lex.cc
    #  in function cpp_avoid_paste().  
    #  Each rule specifies the types of the two tokens.
    #In some rules, the type of the right side can be any type whose spelling 
    #  starts with a given character, 'x', and is here denoted as 'x'.
    #  This is only for punctuators.  
    #Otherwise, the rule uses the CPP_xxx name or the token value 
    #for left and right tokens.
    #Some punctuators, like '|', have alternate spellings in ISO 646.
    #  For a C file, the input file must #include <iso646.h>, which defines 
    #  them as macros for the normal spelling.  
    #  Thus, the preprocessor is not involved.
    #  For a C++ file, --c++ on the command line includes these spellings.

    #Summary of the pairs of tokens:
    #  Any of ( = ! < > + - * / % | & ^ << >> ), =
    #  Digraphs: %: %:%: <: :> <% %>, using 'x' for the second character.
    #  Repeats: > < + - * / : | & . # ID ANYNUM, using 'x' for punctuators.
    #  - '>'
    #  / '*'
    #  . '%'
    #  . ANYNUM
    #  # '%'
    #  ID ANYNUM (if ANYNUM is all [A-Za-z0-9_])
    #  ID CHAR 
    #  ID STRING
    #  ANYNUM any of (ID CHAR '.' '+' '-')
    #  '\' ID
    #  <= '>'
    #  STRING ID
    #  STRING (anything other than a punctuator whose value starts with 
    #          [A-Za-z0-9_])
    #'''

    #def add_sep(type1: str | TokType, type2: str | TokType, *,
    #            failed_paste_only: bool = False,
    #            sep: Callable[PpTok, PpTok, PpTok] = None,
    #            ) -> None:
    #    """ Set type1 to be separated from type2.
    #    Either type can be several other type names separated by spaces,
    #        or a TokType object.  A type name can be an alias.
    #    type1 can be the value of a punctuator.

    #    If type2 is a single character, this matches all punctuators
    #        having any value that starts with that character.

    #    If failed_paste_only is true, then separation applies
    #        only to failed paste, and not to other cases of adjacent tokens.

    #    If `sep` is provided, then this is a callable(before left token, left
    #    token, right token) which returns True for separation.  The default
    #    always returns True.
    #    """
    #    def inner1(type1: str | TokType, type2: str | TokType) -> None:
    #        def inner2(type2: str | TokType) -> None:
    #            if type(type2) is str:
    #                if ' ' in type2:
    #                    for t in type2.split():
    #                        inner2(t)
    #                elif len(type2) == 1:
    #                    for value, typ in punct_values.items():
    #                        if value.startswith(type2):
    #                            inner2(typ)
    #                else:
    #                    inner2(toktypedict[type2])
    #            else:
    #                t1 = TokType[type1]
    #                t2 = TokType[type2]
    #                if not failed_paste_only:
    #                    TokType[type1].sep_from[t2] = sep
    #                TokType[type1].sep_from_paste[t2] = sep

    #        if type(type1) is str:
    #            if ' ' in type1:
    #                for t in type1.split():
    #                    inner1(t, type2)
    #            elif type1 in punct_values:
    #                inner1(punct_values[type1], type2)
    #            else:
    #                t = toktypedict[type1]
    #                if type(t) is str:
    #                    inner1(t, type2)
    #                else:
    #                    inner2(type2)
    #        else:
    #            inner2(type2)
    #    inner1(type1, type2)

    #def add_seps() -> None:
    #    """ Set token separations for default use. """

    #    # Any punctuator which becomes another punctuator by adding 
    #    # a single character...
    #    for p1, p2 in product(punct_values, repeat=2):
    #        if p2.startswith(p1) and len(p2) == len(p1) + 1:
    #            add_sep(punct_values[p1], p2[-1])
    #            x = 0
    #    add_sep('ANYNUM', '+ -')
    #    if prep.gcc:
    #        add_sep('CPP_ID', 'ANYCHAR CPP_STRING', failed_paste_only=True)
    #    if prep.clang:
    #        add_sep('= / $', 'CPP_ID', failed_paste_only=True)
    #        add_sep('= / $', 'CPP_ID', failed_paste_only=True)
    #    add_sep('CPP_DOT', 'ANYNUM CPP_ELLIPSIS')
    #    add_sep('ANYNUM', '. + -')
    #    add_sep('ANYNUM', 'CPP_ID ANYCHAR ANYNUM')

    #def add_seps_clang() -> None:
    #    """ Set token separations for clang simulation. """

    #    # Reference AvoidConcat() in clang/lib/Lex/TokenConcatenation.cpp from
    #    # clang 20.0 repository.

    #    # Separation of `left` and `right` tokens is NOT tested if both came
    #    # from the same source and were adjacent there.  In that case,
    #    # separation is neither added nor removed.

    #    # Separation is solely a function of the left and right types.  The
    #    # right type of a punctuator could be identified by its first
    #    # character, meaning all such punctuators.  
    #    #
    #    # When the left and right types are possibly separated, a
    #    # corresponding callback is called with the actual tokens to make the
    #    # decision.  In most cases, this is always True, but special cases are
    #    # handled by custom callbacks.

    #    # These are the cases implemented in the AvoidConcat() function.
    #    #   left        right       right[0]    conditions
    #    #   ----        -----       --------    ----------
    #    #   punct x                 =           when 'x=' is a punct.
    #    for p in punct_values:
    #        if f'{p}=' in punct_values:
    #            add_sep(p, '=')

    #    #   quoted      ident                   C++ >= 11.
    #    if prep.cplus_ver >= 2011:
    #        add_sep('ANYQUOTED', 'CPP_ID')

    #    #   quoted   same as for ident      C++ >= 11 and
    #    #            (see below)            left has UD suffix.
    #        #add_sep('ANYQUOTED', 'ANYNUM',
    #        #        sep=SepLeftUD() & SepRightNoPeriod())

    #    #   ident       number                  [0] is not '.'
    #    add_sep('CPP_ID', 'CPP_INTEGER CPP_FLOAT')

    #    #   ident       ident
    #    add_sep('CPP_ID', 'CPP_ID')

    #    #   ident       quoted                  right has size prefix
    #    #   ident       quoted                  left is a size prefix and
    #    #                                         right has no size prefix
    #    # Size prefixes are different for C and C++.
    #    # clang gets it wrong for C.

    #    if prep.cplus_ver:
    #        size_prefixes = 'L u u8 U R LR uR u8R UR'
    #    else:
    #        size_prefixes = 'L u8 u8R'  # according to clang.
    #                                    # should be 'L u u8 U'.
    #    pfxs = set(size_prefixes.split())
    #    sep = lambda before_left, left, right: (
    #        right.match.group('pfx') or left.value in pfxs)
    #    ## TEMP: to avoid right.match being undefined
    #    sep = lambda *args: True
    #    add_sep('CPP_ID', 'ANYQUOTED', sep=sep)

    #    #   quoted      any                 ident + right is separated and
    #    #                                   left has UD-suffix
    #    for right_type, id_sep in TokType.CPP_ID.sep_from.items():
    #        sep = lambda before_left, left, right: (
    #            id_sep(before_left, left, right)
    #            and left.match.group('sfx')
    #            )
    #        ## TEMP: to avoid left.match being undefined
    #        sep = lambda *args: True
    #        add_sep('ANYQUOTED', right_type, sep=sep)

    #    #   number                  . + - 
    #    #   number          ident
    #    #                   number
    #    add_sep('ANYNUM', 'ANYNUM CPP_ID . + -')

    #    #   .                       .           left preceded by . also.
    #    sep = (lambda before_left, left, right:
    #           before_left and before_left.type is TokType.CPP_DOT)
    #    add_sep('.', '.', sep=sep)
        
    #    #   .               number  0-9
    #    digit_re = re.compile(REs.digit)
    #    sep = (lambda before_left, left, right:
    #           digit_re.match(right.value[:1]))
    #    add_sep('.', 'ANYNUM', sep=sep)

    #    pairs = '&& ++ -- // << >> || ## -> /* <: <% %> %: :> #@ #%'
    #    if prep.cplus_ver:
    #        pairs += ' .* :: ->* '
    #        if prep.cplus_ver > 2020:
    #            pairs += ' <=>'

    #    for pair in pairs.split():
    #        add_sep(pair[:-1], pair[-1])

    #if prep.clang:
    #    add_seps_clang()
    #else:
    #    add_seps()

    ## We need to have the lextab module be specific to the same parameters
    ## that govern the content of the lexer, i.e.,
    ## 
    ## prep.cplus_ver selects C or C++ as the language, standard version
    ## doesn't matter.  C++ enables the extra punctuators
    ##
    ## prep.emulate includes ` @ and $ as literals..
    ##
    ## prep.gnu enables R-strings for all languages.
    #lextab = f"""\
    #    lextab\
    #    {'-c -cplusplus'.split()[bool(prep.cplus_ver)]}\
    #    {'-gcc' * bool(prep.emulate)}\
    #    {'-gnu' * bool(prep.gnu)}\
    #    """
    #lextab = lextab.replace(' ', '')

    ## Build the lexer from my environment and return it.
    #lexer = lex.lex(lextab=lextab)
    ##lexer = PpLex(prep, lex.lex(lextab=lextab))
    #lexer.prep = prep
    #lexer.TokType = TokType
        
    #lexer.REs = REs
    #return PpLex(from_lexer=lexer)
