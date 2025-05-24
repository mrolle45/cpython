""" Module token.py.
PpTok class and related definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect

from pcpp.common import *
chain = itertools.chain

from pcpp.debugging import *
from pcpp.tokentype import *

__all__ = ('PpTok Tokens TokIter tokenstrip'
           ' TokenSep TokenSepNone TokenSepSpace TokenSepIndent'
           #' TokenSep TokenSepNone TokenSepSpace TokenSepIndent TokenSepPad'
           ' TokLocMove'
           ).split()

#class Hide(frozenset[str]):
#    """ The "hide set" of a token, in Prosser's algorithm.
#    Any ID token whose name is in its hide set won't be macro expanded.
#    """
#    ## TODO: Keep unique Hide's in the prep.  Do | by lookup.

#    def __new__(cls, prep: Preprocessor, names: Set = frozenset()):
#        return super().__new__(cls, names)

#    def __or__(self, other: Set) -> Hide:
#        return Hide(super().__or__(other))

#    def __and__(self, other: Set) -> Hide:
#        return Hide(super().__and__(other))

#    def __repr__(self) -> str:
#        if self:
#            return f"{{{', '.join(sorted(self))}}}"
#        else:
#            return "{}"


# Debugging version of Hide.  Shows names in (last) insertion order.

HideNames = tuple[str]
HideDict = collections.UserDict[str, None]

class Hide(HideDict):
    """
    The "hide set" of a token, in Prosser's algorithm.  Any ID token whose
    name is in its hide set won't be macro expanded.  Iteration will yield the
    names in the order they were last added.
    """
    names: frozenset[str] = frozenset()

    def __new__(cls, prep: Preprocessor, *names: str):
        try: return prep.hides[names]
        except KeyError:
            self = super().__new__(cls)
            self.data = HideDict(((name, None) for name in names))
            prep.hides[names] = self
            self.prep = prep
            self.names = frozenset(names)
            return self

    def __init__(self, prep: Preprocessor, *names: str):
        pass

    def __or__(self, other: Hide) -> Hide:
        """ Add names to end of self, preserving order. """
        names = self.data.copy()
        for name in other:
            if name in names: del names[name]
            names[name] = None
        return Hide(self.prep, *names)

    def __ior__(self, other: Hide):
        return NotImplemented

    def __and__(self, other: Hide | None) -> Hide | None:
        """ Intersection of hide sets, preserving order in self. """
        if not other: return other
        names = self.data.copy()
        for name in self:
            if name not in other: del names[name]
        return Hide(self.prep, *names)

    def add(self, name: str) -> Hide:
        names = self.data.copy()
        if name in names: del names[name]
        names[name] = None
        return Hide(self.prep, *names)

    def __repr__(self) -> str:
        if self:
            return f"{{{', '.join(self)}}}"
        else:
            return "{}"


class P:
    def __init__(self): self.hides = dict()

p = P()

class I(int):
    def __new__(cls, i = 42):
        return super().__new__(cls, i)

#i = I(4)

#h = Hide(p, 1, 2, 3)
#hh = Hide(p, 1, 2, 3)

class RawTok:
    """
    Result of lexing the next token, with minimal information.  Includes
    whitespace and newline tokens.
    """
    lexer: PpLex
    type: TokType
    len: int = None             # Value length if not specified by type.
    datapos: int                # Offset of value in lexer.lexdata.
    lineno: ClassVar[int] = 1
    colno: ClassVar[int] = 1
    source: ClassVar[Source] = None

    def __init__(self, lexer: PpLex, lextoken: LexToken):
        """ Construct from token delivered by lex.Lexer. """
        self.lexer = lexer
        typ = self.type = lexer.TokType[lextoken.type]
        if typ.lit is None:
            self.len = len(lextoken.value)
        self.datapos = lextoken.lexpos

    @property
    def value(self) -> str:
        """ Spelling of the token in the lexer's data. """
        lit = self.type.lit
        if lit is not None:
            return lit
        else:
            start = self.datapos
            return self.lexer.spelling(start, start + self.len)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.value!r}"


class TokenSep(abc.ABC):
    """
    Represents the separation properties between two PpTok's (called lhs and
    rhs) in the output stream.

    Two TokenSep's can be merged using the operator (left + right).  An indent
    takes precedence over a spacing.
    """

    # rhs is preceded by a move to this location (line and column).  In
    # TokenSepIndent class, this is an instance variable.
    indent: ClassVar[TokLoc] = None
    # rhs is preceded by a single space character.  In TokenSepSpace this is a
    # class variable = True.
    spacing: ClassVar[bool] = False

    # rhs is preceded by a single space character if required to avoid making
    # a different token if lhs and rhs are written adjacently.  True as a
    # class variable in TokenSepPad class.
    #padding: ClassVar[bool] = False     # True for TokenSepPad

    # Used to return singleton instance of the class, for some subclasses.
    instance: ClassVar[TokenSep]

    # Used to return singleton instance with no indent and given spacing.
    space_tab: ClassVar[tuple[TokenSep, TokenSepSpace]]

    def __new__(cls, spacing: bool = False) -> TokenSep:
        """ Constructor for no indent, possible spacing. """
        return cls.space_tab[bool(spacing)]

    @abc.abstractmethod
    def __bool__ (self) -> bool:
        """ Any kind of separation, either space or indent or padding. """
        ...

    @abc.abstractmethod
    def moveto(self, writer: OutLoc) -> None:
        """ Called by the Writer after writing lhs and before writing rhs. """
        ...

    @abc.abstractmethod
    def __add__(self, rhs: TokenSep) -> TokenSep:
        """ Merge with rhs. """
        ...

    @classmethod
    def create(cls, *, indent: TokLoc = None, spacing: bool = False) -> Self:
        if indent:
            return TokenSepIndent(indent)
        return cls.space_tab[bool(spacing)]


class TokenSepNone(TokenSep):
    """
    Writes nothing between lhs and rhs, which are adjacent in the same source.
    """
    instance: ClassVar[Self]

    def __new__(cls, ) -> Self:
        return cls.instance

    def __bool__ (self) -> bool:
        return False

    def __add__(self, rhs: TokenSep) -> TokenSep:
        """ Merge with rhs. """
        return rhs

    def moveto(self, writer: OutLoc) -> None:
        if writer.avoid_paste(): writer.spacing()
        pass

    def __repr__(self) -> str:
        return PpTok.Reprs.nospace


class TokenSepIndent(TokenSep):
    """ Moves output to the logical line and column of a token. """

    def __new__(cls, indent: TokLoc) -> Self:
        self = object.__new__(cls)
        self.indent = indent
        return self

    def __bool__ (self) -> bool: return True

    def __add__(self, rhs: TokenSep) -> TokenSep:
        """ Merge with rhs. """
        if rhs.indent:
            return rhs
        else:
            return self

    def moveto(self, writer: OutLoc) -> None:
        writer.indent(self.indent)

    def __repr__(self) -> str:
        ind = self.indent
        rep = f"{PpTok.Reprs.toline}{ind.lineno}"
        if ind.colno > 1:
            rep = f"{rep}:{ind.colno}"
        return rep


class TokenSepSpace(TokenSep):
    """ WrItes a single space between lhs and rhs. """
    spacing: ClassVar[bool] = True
    instance: ClassVar[Self]

    def __new__(cls, ) -> Self:
        return cls.instance

    def __bool__ (self) -> bool: return True

    def __add__(self, rhs: TokenSep) -> TokenSep:
        """ Merge with rhs. """
        if rhs.indent: return rhs
        else: return self

    def moveto(self, writer: OutLoc) -> None:
        writer.spacing()

    def __repr__(self) -> str:
        return PpTok.Reprs.spacing


TokenSepNone.instance = object.__new__(TokenSepNone)
TokenSepSpace.instance = object.__new__(TokenSepSpace)
TokenSep.space_tab = (TokenSepNone.instance, TokenSepSpace.instance)


class PpTok:
    """
    A preprocessor token, with a type and value.  Also has a location, hide
    set and separation from previous token.  Can make copy of self, with some
    attributes changed.

    Since there are so many PpToks in existence, PpTok is designed to take as
    little space as possible.  Class attributes are used for values which are
    the same in most cases, with instance attributes for the exceptional
    cases.  Many attributes are implemented as properties.
    """

    # Basic attributes of all tokens...

    # The Lexer which created this token.  It could also be decoded from the
    # Position by the Preprocessor.
    lexer: Lexer

    # Globally unique number, which can be decoded into a Source and an offset
    # within the Source's data.  0 for a non-lexed token.
    pos: Position = 0

    type: TokType

    # New output location mover.
    newpos: TokMove = None

    # For types with varying values, this is the value from the lexer.
    _value: str = None

    # Number of chars removed from the lexed value.  Added to the value length
    # to get the lexpos for the end of the lexed value.  This only applies if
    # there is an invalid codepoint which is to be ignored.
    skip_len: int = 0

    hide: Hide = None         # Macro names to suppress expansion.

    # True if the token is part of a macro replacement list, or part of a
    # macroargument list.
    in_macro: bool = False

    # Instance variable = replacements made in this token, if any.
    repls: Repls = None

    # True if any Repl has repl.err.
    repl_err: bool = False

    # Where in the Source the start of the value is located.  TODO: this can
    # be derived from self.pos.
    loc: TokLoc = None

    # Separation from adjacent token in output.

    sep: TokenSep = TokenSepNone.instance
    sep_after: TokenSep = TokenSepNone.instance

    # Previous adjacent token, if there is no space between it and self.
    prev: PpTok = None

    def __init__(self, lexer: PpLex, lextoken: LexToken = None,
                 token: PpTok = None, **attrs):
        """
        Construct from a LexToken produced by lex.lex(), or from another
        PpTok, or from scratch.  Keywords are attributes to set on the token.
        """
        if lextoken:
            self.__dict__.update(lextoken.__dict__)
            try: self.type = lexer.TokType[self.type]
            except KeyError: pass
            if not self.type.lit:
                self._value = lextoken.value
            del self.__dict__['value']
            try: self.skip_len = lextoken.skip_len
            except AttributeError: pass

        elif token:
            self.loc = token.loc
        else:
            self.loc = lexer.loc
        self.__dict__.update(**attrs)
        self.lexer = lexer

    def add_hide(self, hide: Hide) -> PpTok:
        """ New token with given hide set added to the current hide set """
        if self.hide:
            hide = self.hide | hide
        return self.copy(hide=hide)

    def copy(self, value: str = None, **attrs) -> PpTok:
        """ Make a copy, and update attributes using keywords. """
        tok: PpTok = copy.copy(self)
        tok.__dict__.update(attrs)
        if value is not None:
            tok.value = value
        return tok

    @property
    def value(self) -> str:
        """ The value of the token. """
        return self.type.lit or self._value

    @value.setter
    def value(self, val: str) -> None:
        if not self.type.lit:
            self._value = val

    @property
    def source(self) -> Source:
        return self.loc.source

    @property
    def lineno(self) -> int:
        """ The physical line number. """
        return self.loc.lineno

    @property
    def colno(self) -> int:
        return self.loc.colno

    @property
    def log_lineno(self) -> int:
        return self.loc.log_lineno

    @property
    def datapos(self) -> int:
        return self.loc.datapos

    @property
    def dataendpos(self) -> int:
        return self.loc.datapos + len(self.value) + self.skip_len

    @property
    def datarange(self) -> Range:
        """ (start, end) of offsets into the replaced source data. """
        start = self.loc.datapos
        return Range(start, len=len(self.value) + self.skip_len)

    @property
    def move(self) -> MoveTok:
        """
        A change of output location which is in effect at this token's
        location in the source file.  Will get value of any __LINE__ or
        __FILE__ macro.
        """
        return self.loc.move

    @property
    def presumed_lineno(self) -> int:
        """ Presumed line number in the output file for this token. """
        if self.move:
            return self.move[self.lineno]
        else:
            return self.lineno

    @property
    def indent(self) -> TokLoc | None:
        return self.sep.indent

    @property
    def spacing(self) -> bool:
        return self.sep.spacing

    def __eq__(self, rhs: PpTok) -> bool:
        """ Tokens are equal if they have the same location. """
        return rhs and self.loc is rhs.loc

    def spacing_after(self) -> None:
        """ Copy of self, with sep_after set. """
        if not self.sep_after:
            return self.copy(sep_after=TokenSepSpace.instance)
        return self
    
    def make_space(self) -> PpTok:
        """ A new token at same location, with a space characer value. """
        return self.copy(value=' ', type=self.type.CPP_WS)

    def make_newline(self) -> PpTok:
        """ A new CPP_NEWLINE token at same location. """
        return self.copy(value='\n', type=self.type.CPP_NEWLINE)

    def make_pos(self, poscls: Type[OutPosChange] = None,
                 tokcls: Type[PpTok] = None,
                 _typenames = dict(
                     OutPosEnter='CPP_LOCENTER',
                     OutPosLeave='CPP_LOCLEAVE',
                     TokLocMove='CPP_LOCMOVE',
                     ),
                 **kwds) -> PpTok:
        """
        Create a position-change token at my location.  This will be for a
        different location from my current location, though in the same
        Source.  It will later be used to set a new output location.
        """
        source = self.source
        
        newpos: OutPosChange = poscls(**kwds)

        tok = self.lexer.make_token(
            self.lexer.TokType(_typenames[poscls.__name__]),
            loc=self.loc, newpos=newpos,
            cls=tokcls or PpTok)
        return tok

    def make_marker(self, sep: TokenSep = None, **attrs) -> PpTok:
        """ A CPP_MARKER token at same location, optional separation. """
        tok: Self = self.copy(value='', type=self.type.CPP_MARKER, **attrs)
        if sep is not None:
            tok.sep = sep
        return tok

    def make_string(self, string: str, **kwds) -> str:
        """ Make CPP_STRING token. """
        return self.copy(value=f'"{string}"', type=self.type.CPP_STRING,
                         **kwds)

    def make_passthru(self, toks: Iterable[PpTok]) -> PpTok:
        """ A CPP_GROUP token at same location. """
        return self.copy(value='', type=self.type.CPP_GROUP,
                         toks=toks, spacing=False)

    def with_sep(self, sep: TokenSep) -> PpTok:
        """ Same token, or copy, with sep = given value. """
        return sep is not self.sep and self.copy(sep=sep) or self

    def add_sep(self, sep: TokenSep) -> PpTok:
        """ Same token, or copy, with sep | given value. """
        return (sep is not self.sep and self.copy(sep=self.sep + sep)
                or self)

    def without_spacing(self) -> PpTok:
        """ Same token, or copy, with spacing = False. """
        if self.spacing:
            self = self.copy()
            self.sep = TokenSepNone()
        return self

    def add_spacing(self) -> PpTok:
        """
        If given value is true (the default), return self, or copy, with
        spacing = True.  Otherwise self unchanged.
        """
        if not self.spacing:
            self = self.add_sep(TokenSepSpace())
        return self

    def with_indent(self, indent: TokLoc) -> PpTok:
        return self.copy(sep=TokenSepIndent(indent))

    #def revert(self, stage: ReplStage = None, *, force: bool = True) -> PpTok:
    def revert(self, stage: ReplStage = None, *, force: bool = False) -> PpTok:
        """
        Reconstruct the original text for the token value and store it in the
        value.  Some replacements won't be reverted, based on the token type.
        Returns new token if changed, else self.
        """
        # TODO: Use reverted Repls and value to make new token.
        if not self.repls:
            return self
        newtok: PpTok = self.repls.revert(self, stage or self.type.revert,
                                          force=force)
        return newtok

    @property
    def match(self) -> re.Match | None:
        """ The Match object from self.value, if any.
        Only for token types which provide a re.Pattern.
        This is a property so that in most cases, it takes no memory.
        """
        patt: re.Pattern | None = self.type.patt
        if not patt: return None
        return patt.match(self.value)

    def brk(self) -> bool:
        """ Break condition for debugging. """
        return break_match(line=self.loc and self.lineno,
                           col=self.loc and self.colno,
                           pos=self.datapos,
                           file=self.loc and self.source
                           and self.source.filename,
                           )

    def wsval(self) -> str:
        """ The token value with preceding whitespace as a space character.
        """
        return f"{self.spacing and ' ' or ''}{self.value}"

    # Special characters used in __repr__()...
    class Reprs:
        # For tokens...
        marker = 'φ'        # '' Empty param or VaOpt or macro expansion.
        paste = 'π'         # '##' concatenate in macro
        tostr = 'σ'         # '#' stringize in macro
        nl = '↩'            # newline
        # For separators...
        nospace = '⦾'       # TokSepNone
        spacing = '•'       # TokSepSpace
        maybe = '‖'         # TokSepPad
        toline = '🠙'        # TokSepIndent, with {line}(:{col})?

    def show_pads(self) -> str:
        """ A display of self which shows any pads. """
        try: pads = ''.join(str(pad) for pad in self.pads)
        except AttributeError: pads = ''
        if pads: pads += ' '
        return f'{pads}{self!r}'

    def __str__(self) -> str:
        val = self.value
        if not self.type:
            rep = f"{val!r}"
        elif self.type.str:
            rep = f'‘{val}’'
        elif self.type.dir and hasattr(self, 'line'):
            rep = f"‘{self.line}’"
        elif self.type.group:
            rep = f"<group {self.source}>"
        elif self.type.paste:
            rep = self.Reprs.paste
        elif self.type.stringize:
            rep = self.Reprs.tostr
        elif self.type.marker:
            rep = self.Reprs.marker
        elif self.type.nl:
            rep = self.Reprs.nl
        elif val:
            rep = f"‘{val}’"
        elif self.type.newpos: rep = f'<newpos {type(self.newpos).__name__}>'
        else: rep = "''"
        return rep

    def __repr__(self) -> str:
        rep = str(self)
        seps: str
        if self.sep:
            rep = f"{self.sep} {rep}"
        if self.sep_after:
            rep = f"{rep} {self.Reprs.spacing}"
        rep = f"{rep} @{self.loc.showpos}"
        if self.hide:
            rep = f"{rep} - {self.hide!r}"

        return rep


class MoveTok(PpTok):
    """
    Specialized PpTok which carries an OutPosMove as self.newpos to indicate a
    change in output location which takes effect at this token's location.
    This is reused by all tokens until a new OutPosMove is seen.
    """
    def __init__(self, lexer: PpLex, newpos: TokMove, **attrs):
        """ Constructed from another token and an OutPosChange.  Copies the
        token's location and other attributes, but with type CPP_NEWPOS.
        """
        super().__init__(lexer, **attrs)
        self.newpos = newpos


@dataclass(frozen=True)
class TokLoc:
    """
    The location within a Source of the start of data for a PpTok.  This is
    kept in the PpLex, then stored in the PpTok when the token is lexed.

    It does not change even if other token attributes change or the token is
    copied.  Useful to compare tokens.
    """
    # Physical Line number, starting at 1.
    lineno: int
    # The Source it was lexed from.
    source: Source
    # Offset of the value in the Source data
    datapos: int
    # Column number of start of value, starting at 1.
    colno: int
    # Current out location change.  Comes from a #line directive executed.
    move: PresumeMover
    # Physical line number - self.lineno.
    phys_offset: ClassVar[int] = 0

    def copy(self, **attrs) -> Self:
        return dataclasses.replace(self, **attrs)

    @property
    def log_lineno(self) -> int:
        """
        Logical line where token is located, rather than lineno, which is the
        physical line.
        """
        return self.lineno - self.phys_offset

    @property
    def output_lineno(self) -> int:
        """ Presumed line number for writing a token. """
        return self.move.out_lineno(self.lineno)

    @property
    def output_filename(self) -> str:
        """
        Presumed file name for writing a token.  Used for __FILE__ macro at
        this location. """
        return self.move.out_filename(self.source)

    @property
    def showpos(self) -> str:
        """ String for just the line and column, not the source. """
        if self.phys_offset:
            line = f"{self.log_lineno}/{self.lineno}"
        else:
            line = f"{self.lineno}"
        return f"{line}:{self.colno}"

    def brk(self) -> bool:
        """ Break condition for debugging. """
        return break_match(line=self.lineno,
                           col=self.colno,
                           pos=self.datapos,
                           file=self.source and self.source.filename,
                           )

    def __str__(self) -> str:
        if self.source:
            filename = self.output_filename
            filename = (f'/{filename!r}'
                        * bool(filename != self.source.filename))
            filename = f"{self.source}{filename}"
        else:
            filename = '<no file>'
        return f"{filename} @{self.showpos}"

    def __repr__(self) -> str:
        return f"{self}"

class TokLocRange(TokLoc):
    """ Location range within a Source file data. """
    src_stop: int                       # End of the range.

    def __init__(self, src_stop: int = None, *args, **kwds):
        super().__init__(*args, **kwds)
        self.src_stop = src_stop

    @property
    def src_range(self) -> Range[str]:
        return Range[str](self.datapos, self.src_stop)


class TokLocMoveBase:
    """
    An object which represents a (possible) #line directive executed in a
    Source file.  At any point in the file, it will provide the presumed line
    number and filename.  These are used when writing a #line directive and
    when expanding __LINE__ and __FILE__ macros.

    As an optimization, the base class has no #line associated with it, and is
    used for the entire source file, or until the first #line is executed.
    The methods for presumed line number and file name are trivial.

    """
    filename: str = None

    def out_lineno(self, src_lineno: int) -> int:
        """
        The output line number for given source line number.  This class
        returns the given line number.
        """
        return src_lineno

    def out_filename(self, source: Source) -> str:
        """
        The output file name for given Source.  This class returns the Source's actual filename.
        """
        return source.filename


class TokLocMove(TokLocMoveBase):
    """
    A shift in line number and/or filename within a Source.  Used to get
    output line number and filename, and __LINE__ and __FILE__ macro values.

    It is associated with a sub-range of the source data, starting after an
    executed #line directive and extending to the next one, or to the end of
    the data.
    """
    delta_lineno: int           # Add to src_lineno to get output.
    filename: str = ""          # Output filename, if overriding Source.

    def __init__(self,
                 # Token for the #line directive, which takes effect at the
                 # following line.
                 dir: PpTok,
                 # New line number.
                 lineno: int,
                 # New file name, if any.  Otherwise use the current lexer's
                 # move's file name.
                 filename: str,
                 ):
        self.delta_lineno = lineno - dir.lineno - 1
        filename = filename or dir.lexer.move.filename
        if filename:
            self.filename = filename

    def out_lineno(self, src_lineno: int) -> int:
        """ Presumed line number for actual source line number. """
        return src_lineno + self.delta_lineno

    def out_filename(self, source: Source) -> str:
        return self.filename or source.filename


class Tokens(collections.UserList[PpTok]):
    """
    An iterable of PpTok tokens, constructed from an iterable of these tokens.

    It behaves like a list.  Tokens.data is the list of stored tokens.  List
    methods work as usual, except that those which produce a new object will
    return a Tokens.  str(tokens) shows the concatenation of all the
    tokens.data values with their spacing (if any).
    """

    def __init__(self, input: Iterable[PpTok] = None):
        super().__init__(input)

    @staticmethod
    def join(*tokens) -> Tokens:
        """ New Tokens object from given token objects. """
        return Tokens(tokens)

    def __str__(self) -> str:
        """ The token values together, with preceding whitespace. """
        return ''.join(map(operator.methodcaller('wsval'), self.data))

    def __repr__(self) -> str:
        if not self:
            return "<No tokens>"
        s = str(self)
        more = "..." if len(s) > 20 else ""
        return (f"<Tokens {s!r:.20}{more}>")

# The TTokens type is anything which can iterate PpTok objects.
TTokens = typing.NewType('TTokens', typing.Iterator[PpTok])

class TokIter(typing.Iterator[PpTok]):
    """
    A specialized Iterator of PpTok objects.

    It represents a chain of 0 or more iterable Items, each of which is either
    a single PpTok or an Iterable[PpTok].  A single token is the same as an
    Iterator yielding just that token.  Iteration goes through the Items, in
    order.  When an Item is exhausted, it is removed from the internal
    implementation as an optimization.

    Built from these operations:
        1. Make the TokIter object, in one of these ways, depending on what
           the tokens will be:
            - Another Iterable, use the constructor TokIter(Iterable).
            - Several Iterables, use TokIter.join(*iterable).
            - A generator function, with arguments, use the decorator
                @TokIter.from_generator.  This results in a function, called
                with the same arguments, which returns the desired TokIter.
            - Single token, use TokIter.from_token(token).
            - Nothing, use TokIter.empty().
        2. Augment the iteration by placing an Item in front of the remaining
           iteration (possibly after it has been partly or fully iterated.
           Call self.putback(token) or self.prepend(Iterable).

    The peek() method gets the first token (if any), without removing it from
    the iteration order.

    The get_tokens() method runs the iterator completely and returns a Tokens
    object which contains the resulting tokens.  This ends iteration on self
    (unless more tokens are later added).

    The copy_tokens() method runs the iterator completely, makes a Tokens
    object with copies of the tokens, and puts the tokens back into the
    iterator, thus preserving future iteration result.
    """

    # The generator of Tokens.  next(self.gen) returns next token or
    # StopIteration.  Will change after the first Item is exhausted, leaving
    # only the remainder.  Will be an empty iterator after the iteration is
    # exhausted.
    gen: Iterable[PpTok]

    # The current iterator.  Changes when its head runs out to point to its
    # tail.
    #iter: Iterator[PpTok]

    Empty: ClassVar[Gen]

    # Set while consuming a defined-macro expression.  This tells the expander
    # to return the macro name identifier verbatim.
    in_defined_expr: ClassVar[bool] = False

    if __debug__:
        _serial = itertools.count(1)
        ser: int

    def __init__(self, gen: Iterable[PpTok] = None):
        """
        Constructor for the last iterable in the chain.  When given `gen` is
        exhausted, self.gen reverts to self.Empty and StopIteration is raised.
        """
        if gen:
            if isinstance(gen, typing.Sequence):
                self.set_gen(self.SeqGen(self, gen, self.Empty))
            else:
                self.set_gen(self.PairGen(self, gen, self.Empty))
        else:
            self.set_gen(self.Empty)
        if __debug__: self.ser = next(self._serial)

    def __iter__(self):
        while True:
            try: yield next(self.gen)
            except StopIteration: return

    def __next__(self) -> PpTok:
        return next(iter(self.gen))

    def __bool__(self) -> bool: return bool(self.gen.peek())

    @classmethod
    def class_init(cls):
        cls.Empty = cls.EmptyGen()

    def set_gen(self, gen: Iterable[PpTok]) -> None:
        self.gen = gen

    @classmethod
    def empty(cls) -> TokIter:
        """ New TokIter which yields nothing. """
        return cls()

    @classmethod
    def join(cls, *iterables: Iterable[PpTok]) -> TokIter:
        """ New TokIter chains several iterables. """
        return cls(chain(*iterables))

    @classmethod
    def from_token(cls, token: PpTok) -> TokIter:
        """ New TokIter which yields only the given token. """
        self = cls()
        self.putback(token)
        return self

    @classmethod
    def from_tokens(cls, tokens: typing.Sequence[PpTok]) -> TokIter:
        """ New TokIter which yields only the given tokens. """
        self = cls()
        self.prepend(self.SeqGen(self, tokens, self.Empty))
        return self

    # Decorate a token generator function to produce a TokIter.
    def from_generator(gen: Iterable[PpTok]) -> Callable[..., TokIter]:
        """
        @TokIter.from_generator

        def gen(self, *args, **kwds) -> Iterable[PpTok]
            ...
        Creates method gen(self, *args, **kwds) -> TokIter
        """
        def tokens(*args, **kwds) -> TokIter:
            return TokIter(gen(*args, **kwds))

        return tokens

    def peek(self) -> PpTok | None:
        return self.gen.peek()

    def putback(self, token: PpTok) -> None:
        """ Puts a given token in front of the existing iteration. """

        self.set_gen(self.LookaheadGen(self, token, self.gen))

    def prepend(self, gen: Iterable[PpTok]) -> None:
        """
        Puts a given token iterable in front of the existing iteration.
        """
        if isinstance(gen, typing.Sequence):
            self.set_gen(self.SeqGen(self, gen, self.gen))
        else:
            self.set_gen(self.PairGen(self, gen, self.gen))

    def get_tokens(self, max: int = None) -> Tokens:
        """
        Runs the iteration, then returns a Tokens containing the iterated
        tokens.  This exhausts iteration of self.

        Optional `max` argument if not None limits the number of tokens.
        """
        if max is None:
            return Tokens(self)
        else:
            def toks() -> Iterator[PpTok]:
                for _ in range(max):
                    tok = next(self, None)
                    if tok is None: break
                    yield tok
            return Tokens(toks())

    def gen_until(self, pred: Callable[[PpTok], bool]) -> Iterator[PpTok]:
        """
        Generate all initial tokens which DO NOT satisfy pred(tok).  These are
        consumed but the next token, if any, remains in the iteration.
        """
        tok: PpTok = self.peek()
        if not tok or pred(tok):
            return
        for tok in self:
            if pred(tok):
                self.putback(tok)
                break
            yield tok

    def gen_while(self, pred: Callable[[PpTok], bool]) -> Iterator[PpTok]:
        """
        Generate all initial tokens which satisfy pred(tok).  These are
        consumed but the next token, if any, remains in the iteration.
        """
        tok: PpTok = self.peek()
        if not (tok and pred(tok)):
            return
        for tok in self:
            if not pred(tok):
                self.putback(tok)
                break
            yield tok

    def copy_tokens(self, max: int = None) -> Tokens:
        """
        Runs the iteration, puts the iterated tokens back into self, and
        returns a Tokens with copies of the tokens.  Iteration of self is
        preserved.

        Optional `max` argument if not None limits the number of tokens.
        """
        toks: Tokens = self.get_tokens(max)
        self.prepend(toks)
        return toks

    @from_generator
    def apply(self, func: Callable[[PpTok], PpTok]) -> Iterator[PpTok]:
        """ New iterator with given function called on the tokens. """
        yield from (func(tok) for tok in self)

    def apply_first(self, func: Callable[[PpTok], PpTok]) -> TokIter:
        """
        Same iterator with the first token (if any) changed by calling the
        given function, and remaining tokens unchanged.
        """
        for tok in self:
            self.putback(func(tok))
            break
        return self

    @from_generator
    def frame_pads(self, ref: PpTok) -> Iterator[PpTok]:
        """
        Generate the tokens, preceded and followed by pads.  A φ is supplied
        for an empty iterator.
        """
        tok: PpTok = next(self, None)
        if not tok:
            tok = ref.make_marker()
        # First token, or new marker.  Gets sep added.
        tok = tok.add_sep(ref.sep)
        prev = tok
        for tok in self:
            yield prev
            prev = tok
        # Last token.  Gets sep after itself.
        tok = tok.add_seps_after()
        yield tok

    @from_generator
    def skip_pads(self) -> list[PpTok]:
        """
        Consume and return pad tokens at the beginning.  Depending on what the
        next non-pad token is, the caller may want to output these pads.
        """
        if not self.peek().pad:
            return []
        result: list[PpTok] = []
        for tok in self:
            if tok.pad:
                result.append(tok)
            else:
                self.putback(tok)
                break
        return result

    @from_generator
    def strip(self) -> Iterator[PpTok]:
        """
        Remove leading/trailing whitespace.  Only the first token can have
        whitespace, so that is removed.  Remaining tokens are unchanged.
        Returns self.
        """
        tok = next(self, None)
        if tok:
            yield tok.without_spacing()
        l = list(self)
        yield from l

    @staticmethod
    def check_type(obj: TokIter, descr: str) -> None:
        """ Assertion that given object is a TokIter, else AssertionError """
        assert isinstance(obj, TokIter), (
                f"{descr} requires TokIter, got {type(obj).__name__}.")

    if __debug__:
        def print(self, indent: str = '') -> None:
            """ Hierarchical dump of the objects. """
            print(f"{self.ser} {type(self).__name__}")
            self.gen.print(indent + '  ')


    class Gen(collections.abc.Iterator):
        """ Base class for the iterable stored in TokIter.gen. """
        ti: TokIter                     # The TokIter this belongs to

        def __init__(self, ti: TokIter):
            self.ti = ti
            if __debug__: self.ser = next(ti._serial)

        def __iter__(self): return self

        if __debug__:
            def print(self, indent: str = '  ') -> None:
                """ Hierarchical dump of the objects. """
                print(f"{indent}{self.ser} {type(self).__name__}")
                self.printitems(indent + '  ')

            def printitems(self, indent: str) -> None:
                pass

        def __repr__(self) -> str:
            if __debug__: ser = f"{self.ser} "
            else: ser = ""
            return f"<{ser}{type(self).__name__}>"


    class LookaheadGen(Gen):
        """ Manages a lookahead token followed by a tail iterable. """
        def __init__(self, ti: TokIter, tok: PpTok, tail: Iterable[PpTok]):
            super().__init__(ti)
            self.tok = tok
            self.tail = tail

        def __next__(self) -> PpTok:
            """ Get the next token.  Reset ti to point to tail. """
            tail = self.tail
            self.ti.gen = iter(tail)
            return self.tok

        def peek(self) -> PpTok | None:
            return self.tok

        if __debug__:
            def printitems(self, indent: str) -> None:
                print(f"{indent}{self.tok}")
                self.tail.print(indent)


    class PairGen(Gen):
        """
        Manages a TokIter with head and tail iterables.  Head may be a
        generator function or a TokIter.
        """
        def __init__(self, ti: TokIter, head: Iterable[PpTok],
                     tail: Iterable[PpTok]):
            super().__init__(ti)
            self.head = iter(head)
            self.tail = tail

        def __next__(self) -> Iterator[PpTok]:
            tok = next(self.head, None)
            if tok: return tok
            self.ti.gen = iter(self.tail)
            try: return next(self.ti)
            except StopIteration:
                raise
            except:
                traceback.print_exc()
                raise

        def peek(self) -> PpTok | None:
            tok = next(self.ti, None)
            if tok:
                self.ti.putback(tok)
            return tok

        if __debug__:
            def printitems(self, indent: str) -> None:
                if inspect.isgenerator(self.head):
                    print(f"{indent}{self.head.__qualname__}")
                else:
                    self.head.print(indent)
                self.tail.print(indent)


    class SeqGen(Gen):
        """ Manages a TokIter with token sequence head and iterable tail. """
        def __init__(self, ti: TokIter, head: Sequence[PpTok],
                     tail: Iterable[PpTok]):
            super().__init__(ti)
            self.head = iter(head)
            self.toks = head
            self.i = 0
            self.len = len(head)
            self.tail = tail

        def __next__(self) -> PpTok:
            """
            Get next token, or raise StopIteration.  If head is empty, reset
            ti to point to tail and try again.
            """
            tok: PpTok = next(self.head, None)
            self.i += 1
            if tok: return tok

            self.ti.gen = iter(self.tail)
            return next(self.ti)

        def peek(self) -> PpTok | None:
            if self.i < self.len:
                return self.toks[self.i]
            self.ti.gen = iter(self.tail)
            return self.ti.peek()


        if __debug__:
            def printitems(self, indent: str) -> None:
                for i in range(self.i, self.len):
                    print(f"{indent}[{i}] {self.toks[i]}")
                self.tail.print(indent)


    class EmptyGen(Gen):
        """ Manages a TokIter with no tokens at all. """
        def __init__(self):
            if __debug__: self.ser = 0
        def __next__(self) -> PpTok:
            raise StopIteration
        def peek(self) -> PpTok | None:
            return None

        if __debug__:
            def printitems(self, indent: str) -> None:
                pass


    def __repr__(self) -> str:
        if __debug__: ser = f"{self.ser} "
        else: ser = ""
        return f"<{ser}TokIter>"


TokIter.class_init()


# ----------------------------------------------------------------------
# tokenstrip()
# 
# Remove leading/trailing whitespace tokens from a token list
# ----------------------------------------------------------------------

def tokenstrip(tokens: Tokens) -> Tokens:
    """ Remove leading/trailing whitespace tokens from a token list.
    Return the same, but modified, list.
    """
    # The leading whitespace if any belongs to the first token.  There is no
    # trailing whitespace.
    if tokens:
        tokens[0] = tokens[0].without_spacing()
    return tokens

