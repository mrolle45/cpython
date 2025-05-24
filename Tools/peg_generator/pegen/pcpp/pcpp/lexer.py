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
from pcpp.regexes import *
from pcpp.replacements import *
from pcpp.tokens import (PpTok, RawTok, Tokens, TokIter, TokLoc, TokenSep,
                         TokenSepSpace, MoveTok, TokLocMoveBase)
from pcpp.directive import Directive
from pcpp.writer import OutPosFlag, OutPosChange, OutPosEnter


class Lines:
    """
    A mixin class for the PpLex lexer, which tracks lines as tokens are lexed
    from the lex_data.  Line information is advanced by the update_lines()
    method to apply to the just-lexed token.  Line information is, however,
    for lines in src_data.

    Logical and physical lines are tracked separately.  A physical line starts
    at the beginning of the src_data, or immediately after any newline
    character.  A logical line excludes newline characters that are part of a
    token other than a CPP_NEWLINE token.

    As an optimization, it holds the src_data position after the next newline
    character after the current physical line, if any, else the end of the
    src_data.  This makes for fast determination of whether a token contains
    any newlines.

    It counts physical lines up to the end of the current token, as
    self.src_lineno.  The line number of the current logical line is kept in
    self.src_log_lineno.

    """

    # These attributes are valid for the current lexed token, after
    # self.update_lines(token) is called...

    # Position in src_data of first character in current physical line.
    # Beginning of file, or after last newline character before current lexer
    # position.
    src_linepos: int

    # Position after next newline char after src_linepos.  Will become
    # src_linepos when pos reaches this point, starting a new physical line.
    src_next_linepos: int

    # Line number for current physical or logical line, starting at 1.
    src_lineno: int
    src_log_lineno: int

    # The current logical line has only whitespace so far.  Useful for
    # detecting the first non-whitespace token.  Such as the '#' in a
    # directive.
    only_ws_this_line: bool = True

    def input(self, data: str) -> None:
        """ Set the input data here and in the lex.Lexer class.  Translation
        phase 1 and 2 replacements have already been made in the data, and
        self.repls records all of them.
        """
        # Initialize in lex.Lexer class.  Sets lexdata, lexlen, and lexpos.
        data = super().input(data)
        self.src_lineno = 0
        self.src_next_linepos = 0
        self.advance_line()

    def update_lines(self, tok: PpTok) -> None:
        """
        Bring internal state up to date after given token, and also set some
        token attributes.  lexpos is now at the end of the value.
        """

        # Did the token cross a newline?
        while self.src_endpos >= self.src_next_linepos:
            # Token could be a newline token, or it could be some other type
            # with newlines embedded within it.  A newline token might have
            # some line splices also.
            logical: bool = tok.type.nl and self.src_endpos == self.src_next_linepos
            self.advance_line(logical)

    def advance_line(self, logical: bool = True) -> int:
        """
        Move to next newline, as self.src_linepos.  Find
        self.src_next_linepos.  Increment and return self.src_lineno.

        For logical line, set variables for new logical line.  For physical
        line, increment physical line count.
        """

        self.src_lineno += 1
        if logical:
            self.src_log_lineno = self.src_lineno
            self.only_ws_this_line = True

        self.src_linepos = self.src_next_linepos
        next_linepos = (self.src_data.find('\n', self.src_linepos) + 1)
        if next_linepos:
            self.src_next_linepos = next_linepos
        else:
            self.src_next_linepos = self.src_linepos + 1

        return self.src_lineno

    def advance_physical_line(self) -> int:
        return self.advance_line(logical=False)

    def find_src_nl(self) -> int:
        """
        Position after next newline character after self.src_linepos, or
        srclen if not found.
        """
        return (self.src_data.find('\n', self.src_linepos) + 1
                or len(self.src_data) + 1)

    @property
    def colno(self) -> int:
        """
        Column number (starting at 1) for current source position in current
        physical line.  A line splice sets the colno back to 1 by setting
        src_linepos = splice position.

        """
        return self.src_pos - self.src_linepos + 1


class RawLexer:
    """
    Low level interface to the lex.Lexer class.  It accepts a data string,
    which might be the contents of a source file, or otherwise artificially
    constructed.

    The input data is provided separately by the input() method.

    Produces raw tokens serially from the data.  Also provides an iterator to
    do this.  The raw token does not have complete PpTok information.  It does
    provide enough information to analyze the tokens contained in an arbitrary
    string.
    """
    # Enable error messages during lexing.  Can be set by subclass.
    errors: bool = False

    # Original data, usually from a SourceFile via input() method.
    src_data: str

    # Original data with replacements, which is what is actually lexed.
    lex_data: str

    def __init__(self, lex: Lexer, *args, **kwds):
        self.prep = prep = lex.prep
        self.REs = lex.REs
        self.TokType = lex.TokType
        self.source = prep.currsource
        try: lex.orig
        except AttributeError:
            lex.orig = self

    def input(self, data: str) -> str:
        """ Begin lexing. """
        self.lex_data = data
        self.lex.input(data)
        return data

    def parse(self, data: str, raw: bool = True) -> Iterator[RawTok]:
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
        """
        Skip ahead n characters and note any newlines and cuts passed.

        Mainly used to skip over a character that is not the start of a token.
        """
        assert n >= 0, f"Trying to use skip({n}) to move backward."
        pos = self.lex.lexpos
        self.lex.skip(n)
        while pos < self.lex.lexpos:
            if self.lex_data[pos] == '\n':
                self.newline(pos)
            pos += 1

    def spelling(self, start: int, stop: int) -> str:
        return self.lex_data[start : stop]

    def brk(self) -> bool: return False


class PpLex(Lines, RawLexer):
    """
    Lexer for inclusion of a single source file.  It takes the data from the
    file, then it performs translation phases 1 and 2 and unicode replacement
    on the input data.  Then it does tokenizing of this result (phase 3) using
    a lex.Lexer object.

    It maintains the current logical and physical line number.  Also provides
    column position within the current physical line.

    Note that digraph sequences are not replaced.  Rather, they are treated as
    punctuator tokens in a context where a punctuator is possible.  Within a
    character constant or a string, a digraph is just two characters.
    """
    # Object representing the source file inclusion.
    source: Source = None

    lineno: int                 # Line number.

    # Last change in output position (if any).
    move = TokLocMoveBase()

    errors: bool                # Enable error messages during tokens().

    lex: Lexer                  # Original ply.py lexer this came from.
    clones: list[Lexer] = []    # Any available clones of self.lex.

    # Look ahead token.  Use this instead of getting a new one.
    lookahead: PpTok = None

    # Original data, before replacements.  The replaced data is in the base
    # RawLexer class.
    src_data: str

    # Range in source data of current token.  This is set after the token is
    # lexed and its replacements are located.
    src_range: Range[str]

    # Places where the original src_data was altered in phase 1 or 2 or
    # unicode escapes.
    repls: Replacer = EmptyReplacer()

    # Set while scanning macro argument list.
    _in_macro: MacroArgs = None

    # Previous token lexed.
    prev: PpTok = None

    def __new__(cls, *, from_lexer: Lexer = None,
                **kwds
                ) -> Self:
        lexer = super().__new__(cls)
        if from_lexer:
            lex = from_lexer.clone()
            lex.owner = lexer
            lexer.lex = lex
            lexer.clones = []
        return lexer

    def __init__(self, *, lex: Lexer = None, **kwds):
        if not lex:
            lex = self.lex
        super().__init__(lex)

    def input(self, data: str, source: Source = None):
        """
        Set the data to be lexed.  Performs Translation Phase 1 and 2 and
        unicode replacements.  Sets the source object, if given.
        """
        self.source = source
        repls = Replacer(self)
        self.src_data = data
        self.src_range = Range(0)
        data = repls.find_repls(data)
        if repls: self.repls = repls
        super().input(data)
        self.linepos = 0

    @property
    def src_pos(self) -> int:
        """ Location in source data of last lexed token. """
        return self.src_range.start

    @property
    def src_endpos(self) -> int:
        """ Location in source data after last lexed token. """
        return self.src_range.stop

    @contextlib.contextmanager
    def cloned(self, cls: Type = None, errors: bool = True
               ) -> ContextManager[PpLex]:
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
        """ Make a clone of the prep's PpLex, or reuse an earlier clone. """
        lex: Lexer = self.lex
        clones = self.clones
        if clones:
            lex = clones.pop()
        else:
            orig: PpLex = lex.orig
            lex = lex.clone()
            lex.orig = orig
        clone = (cls or type(self))(lex=lex)
        clone.lex = lex
        lex.owner = clone
        clone.clones = clones
        return clone

    @contextlib.contextmanager
    def seterrors(self, errors: bool = True) -> ContextManager[None]:
        """
        Set the error message enabling during the context, then restore it.
        """
        old = self.errors
        self.errors = errors
        try: yield
        finally:
            self.errors = old

    def make_token(self, typ: TokType, *, value: str = None,
                   loc: TokLoc = None, cls: type = PpTok,
                   **attrs
                   ) -> PpTok:
        """
        Make a new token with given type and optional value, and any other
        attributes desired.  The default value varies with the type.  The
        token is at the current location of self, by default, or `loc` if not
        None.
        """
        if value is None: value = typ.val
        if loc is None: loc = self.loc
        return cls(self, type=typ, value=value, loc=loc, **attrs)

    @property
    def loc(self) -> TokLoc:
        """ Current location, which will be stored in a new token. """
        loc = TokLoc(lineno=self.src_lineno, colno=self.colno,
                     source=self.source, datapos=self.lex.lexpos,
                     move=self.move)

        phys_offset = self.src_lineno - self.src_log_lineno
        if phys_offset:
            loc = replace(loc, phys_offset=phys_offset)
        return loc

    def nexttok(self, nl: bool = False) -> PpTok | None:
        """
        Get the next token from lexer, or None.  Skip whitespace and maybe
        also newlines.
        """

        tok: PpTok = self.lookahead
        if tok:
            del self.lookahead
            return tok

        ws: bool = False
        #self.brk()
        while True:
            self.src_range = Range(self.src_range.stop, 0)
            loc = self.loc
            #if self.brk():
            #    re.match(self.REs.char, self.lex_data[loc.datapos:])
            t: LexToken | None = self.lex.token()
            if not t: return None
            tok = PpTok(self, t)
            tok.loc = loc
            repls = self.repls.movetotoken(tok)
            self.src_range = tok.src_range
            tok.pos = loc.source.position_at(self.src_range.start)
            self.update_lines(tok)
            if repls:
                if tok.type.revert:
                    tok = tok.revert()
            if tok.type.ws:
                if tok.type.nl:
                    self.only_ws_this_line = True
                    ws = False
                    if nl:
                        break
                else:
                    ws = True
            else:
                self.only_ws_this_line = False
                break

        tok.sep = TokenSep.create(spacing=ws)
        if not ws:
            tok.prev = self.prev
        self.prev = tok.pos
        return tok

    @TokIter.from_generator
    def tokens(self, errors: bool = True
               ) -> Iterator[PpTok]:
        """
        Generate all tokens for the entire lex_data string.  Includes indent
        tokens for logical lines.  Translation Phase 1 and 2 replacements have
        been done already.
        """
        tok: PpTok = None
        prev: PpTok = None

        self.errors = errors
        lineno: int = 0         # Line # of last token seen.

        while True:
            tok = self.nexttok()
            if not tok: break
            #if not tok.sep:
            #    tok.prev = prev
            prev = tok
            #if self._in_macro:
            #    tok.in_macro = True

            if tok.type.dir:
                # Special handling for a directive.
                dir = tok
                tok.sep = TokenSep.create(indent=tok.loc)
                dir.line = Tokens([tok.copy(type=self.TokType.CPP_POUND)])
                tok = self.nexttok(nl = True)
                while not tok.type.nl:
                    dir.line.append(tok)
                    tok = self.nexttok(nl = True)
                dir.dir = Directive(dir)
                yield dir
                # 
                continue
            # Not directive.  Check for change in line number.
            if tok.log_lineno != lineno:
                if not self.in_macro():
                    # But not while scanning within a macro arg list.  N.B.:
                    # if this is the opening '(', it will get the indent.
                    tok.sep = TokenSep.create(indent=tok.loc)
                else:
                    # The previous newline is still whitespace.
                    tok.sep = TokenSepSpace.instance
                lineno = tok.log_lineno
            if not tok.type.nl:
                self.prep.log.msg(f"Lex token {tok!r}", tok)
                yield tok

    def make_passthru(self, toks: Iterable[PpTok]) -> PpTok:
        """ A CPP_GROUP token at current location. """
        return self.make_token(self.TokType.CPP_GROUP, toks=toks, )

    def try_paste(self, lhs: PpTok, rhs:PpTok) -> PpTok | None:
        """
        Try to paste two tokens together.  Returns new token at the location
        of the left token (or the right token if left is a placemarker).
        Merges the two hide sets.  Returns None if not a valid token result.
        """
        lex: PpLex
        if lhs.type.marker: return rhs
        if not lhs.value: return rhs
        if rhs.type.marker: return lhs

        value = lhs.value + rhs.value
        #with self.cloned(errors=False) as lex:
        lex = self.lex.orig.pasting
        if True:
            # Using this string as input data, lex the first token using base
            # class lexer.  This should return a LexToken with exactly the
            # same value, and not an error type.  Otherwise the input data is
            # not valid.  Could call lex.token(), but this way is faster.
            lex.input(value)
            t: LexToken | None = lex.lex.token()
            if t:
                typ = self.TokType[t.type]
                if len(value) != lex.lex.lexpos or typ.err:
                    # Failure to match the value.
                    self.prep.on_error_token(lhs,
                        f"Pasting result ‘{value}’ is not a valid token.")
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

    @contextlib.contextmanager
    def inmacro(self, args: MacroArgs) -> ContextManager[None]:
        """
        Declare that the consumer of the next tokens is in or preceding a
        function macro argument list, during the context.  Some lexing,
        notably certain directives and indents, is handled differently.
        """
        old = self._in_macro
        self._in_macro = args
        self.prep.log.msg("Enter inmacro()", args.call.nametok)
        try: yield
        finally:
            if not old:
                del self._in_macro
            else:
                self._in_macro = old
            self.prep.log.msg("Leave inmacro()", args.call.nametok)
            self._in_macro

    def in_macro(self, in_args: bool = True) -> MacroArgs | None:
        """
        Is a function macro being scanned for the argument list?  If `in_args`
        is true, then only if the opening '(' has been seen.  Otherwise if it
        has NOT been seen.
        """
        args: MacroArgs | None = self._in_macro
        if args:
            if in_args == (args.args is None):
                # Scanning for args but not in desired place.
                return None
        return args

    def brk(self) -> bool:
        """ Break condition for debugging. """
        return break_match(line=self.src_lineno, col=self.colno,
                           pos=self.lex.lexpos,
                           file=self.source and self.source.filename or "")

    def __repr__(self) -> str:
        return f"<PpLex {self.loc}>"
