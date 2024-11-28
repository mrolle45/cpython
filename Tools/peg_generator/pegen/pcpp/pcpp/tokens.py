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

__all__ = ('PpTok, Tokens, reduce_ws, split_lines'.split()
           + 'tokenstrip, tokenstripiter'.split()
           )

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
    #    self.data = HideDict(((name, None) for name in names))
    #    prep.hides[names] = self
    #    prep = prep
    #    self.names = frozenset(names)


    #def __init__(self, data: Mapping[str, None] = {}):
    #    super().__init__(data)
    #    self.names = frozenset(data)

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

i = I(4)

h = Hide(p, 1, 2, 3)
hh = Hide(p, 1, 2, 3)
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


class TokenSep:
    """
    Represents the separation properties between two PpTok's (called lhs and
    rhs) in the output stream.  The base class is a singleton object which
    states that the two tokens are written adjacently.

    Two TokenSep's can be merged using the operator (rhs |= lhs).  An indent
    takes precedence over a spacing.
    """

    def __bool__ (self) -> bool: return False
    indent: ClassVar[TokLoc] = None
    spacing: ClassVar[bool] = False

    instance: ClassVar[TokenSep]

    def moveto(self, writer: OutLoc) -> None:
        """ Called by the Writer after writing lhs and before writing rhs. """
        pass

    def __ior__(self, lhs: TokenSep):
        """ Merge with lhs.  Self is not an indent. """
        if lhs: return lhs
        else: return self

    @classmethod
    def create(cls, *, indent: TokLoc = None, spacing: bool = False) -> Self:
        if indent:
            return TokenSepIndent(indent)
        if spacing:
            return TokenSepSpace.instance
        return TokenSep.instance

    def __repr__(self) -> str:
        return "<>"

TokenSep.instance = TokenSep()


class TokenSepIndent(TokenSep):
    """ Moves output to the logical line and column of lhs.
    """
    indent: PpTok

    def __bool__ (self) -> bool: return True

    def __init__(self, indent: TokLoc):
        self.indent = indent

    def __ior__(self, lhs: TokenSep):
        return self

    def moveto(self, writer: OutLoc) -> None:
        writer.indent(self.indent)

    def __repr__(self) -> str:
        ind = self.indent
        rep = f"{PpTok.Reprs.toline}{ind.phys_lineno}"
        if ind.colno > 1:
            rep = f"{rep} {PpTok.Reprs.tocol}{ind.colno - 1}"
        return rep


class TokenSepSpace(TokenSep):
    """ WrItes a single space between lhs and rhs. """
    def __bool__ (self) -> bool: return True
    spacing: ClassVar[bool] = True
    instance: ClassVar[TokenSepSpace]

    def moveto(self, writer: OutLoc) -> None:
        writer.spacing()

    def __repr__(self) -> str:
        return PpTok.Reprs.spacing

TokenSepSpace.instance = TokenSepSpace()


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
    # within the Source's data.
    pos: Position

    type: TokType

    # Length of the value, for types with varying values.  For types with a
    # fixed value, use the length of that value.
    len: int = 0

    hide: Hide = None         # Macro names to suppress expansion.

    # Instance variable = replacements made in this token, if any.
    repls: list[Repl] = []
    # Instance variable = the token before reverting if revert() called and
    # repls != [].
    reverted_from: PpTok = None
    # Where in the Source the start of the value is located.
    loc: TokLoc = None

    # Token spacing...
    '''
    Indicates if a padding space is written between this token and previous
    token, or indentation at the start of logical line.

    1. The lexer establishes a logical line number for each token.  It counts
      line splices seen in any tokens since the last newline token, and then
      subtracts this from the actual line number.
    2. If a token is not whitespace and is preceded by whitespace token(s),
      the lexer sets token.spacing = the first of these whitespace tokens.
    3. The current Source groups these tokens into logical lines, separated
      by newlines.
    4. Macro expansion processes either a directive line or a maximal 
      sequence of non-directive logical lines.  If a function macro call
      extends over multiple lines, then the next logical line is considered to
      begin after the macro call.
    5. The first token, `t`, in a logical line ignores t.spacing and sets
      t.spacing = t itself.  This tells the output Writer that it starts a new
      line and where the starting column is.  If the line is not empty, the
      Writer outputs spaces up to the starting column.
    6. If any token, `t`, is a macro that is replaced, then t.spacing is
      moved to the first replacement token, or if no replacement tokens, then
      to the following token (if any).  If t is the start of a line, this
      tells the Writer that the line was originally non-empty.  When emulating
      GCC, it treats the line as non-empty even if there are no more tokens in
      the line.
    7. In some circumstances connected with macro expansion, a token `t`
      may follow a token `t0` that it did not follow originally.  If there is
      no t.spacing already, then the Writer makes a fast check to see if a
      separating space is necessary.  This check uses the types, not the
      values, of t0 and t, and so may indicate a spacing required when it is
      not actually needed.  In GCC emulation, there are some cases where it
      requires spacing but no possible values for the tokens do.
    '''


    sep: PpTok = TokenSep.instance
    spacing: bool = False   # Possible space before in output.
    indent: PpTok = None    # self.orig if first token in logical line.
                            # Use this token for output location.


    exp_from: Macro = None  # Macro whose definition contains self.

    # Macro call which generated this token from macro replacement list.
    call_from: MacroCall = None

    prev: PpTok = None      # Previous token on same line, skipping whitespace.
                            # None if first token in logical line. 

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
        elif token:
            self.loc = token.loc
        else:
            self.loc = lexer.loc
        self.__dict__.update(**attrs)
        self.lexer = lexer

    def add_hide(self, hide: Hide) -> None:
        """ Add the hide set to the current hide set """
        if self.hide:
            self.hide |= hide
        else:
            self.hide = hide

    def copy(self, **attrs) -> PpTok:
        """ Make a copy, and update attributes using keywords. """
        tok: PpTok = copy.copy(self)
        tok.__dict__.update(attrs)
        return tok

    @property
    def source(self) -> Source:
        return self.loc.source

    @property
    def lineno(self) -> int:
        return self.loc.lineno

    @property
    def colno(self) -> int:
        return self.loc.colno

    @property
    def phys_lineno(self) -> int:
        return self.loc.phys_lineno

    @property
    def datapos(self) -> int:
        return self.loc.datapos

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

    def make_space(self) -> PpTok:
        """ A new token at same location, with a space characer value. """
        return self.copy(value=' ', type=self.type.CPP_WS)

    def make_newline(self) -> PpTok:
        """ A new CPP_NEWLINE token at same location. """
        return self.copy(value='\n', type=self.type.CPP_NEWLINE)

    def make_pos(self, cls: Type[OutPosChange], **attrs) -> PpTok:
        """
        Create a CPP_NEWPOS token at my location.  This will be for a
        different location from my current location, though in the same
        Source.  It will later be used to set a new output location.
        """
        source = self.source
        
        pos: OutPosChange = cls(**attrs)

        if pos.move:
            tok = self.lexer.make_token(self.lexer.TokType.CPP_NEWPOS,
                                        cls=MoveTok, loc=self.loc, pos=pos)
            self.lexer.move = tok
        else:
            tok = self.lexer.make_token(self.lexer.TokType.CPP_NEWPOS,
                                        loc=self.loc, pos=pos)
        return tok

    def make_null(self, **attrs) -> PpTok:
        """ A CPP_NULL token at same location. """
        return self.copy(value='', type=self.type.CPP_NULL, **attrs)

    def make_passthru(self, toks: Iterable[PpTok]) -> PpTok:
        """ A CPP_PASSTHRU token at same location. """
        return self.copy(value='', type=self.type.CPP_PASSTHRU,
                         toks=toks, spacing=False)

    def with_sep(self, sep: TokenSep) -> PpTok:
        return sep is not self.sep and self.copy(sep=sep) or self

    def set_spacing(self, spacing: bool) -> PpTok:
        """ Same token, or copy, with spacing = given value. """
        return (self.spacing != spacing
                and self.copy(spacing=spacing,
                              sep=spacing and TokenSep.instance
                              or TokenSepSpace.instance)
                or self)

    #def with_indent(self, indent: PpTok) -> PpTok:
    #    """ Same token, or copy, with indent = given value. """
    #    return (self.indent is not indent
    #            and self.copy(indent=indent, spacing=False,
    #                          sep=indent and TokenSepIndent(indent)
    #                          or TokenSep.instance)
    #            or self)

    def without_spacing(self) -> PpTok:
        """ Same token, or copy, with spacing = False. """
        return (self.spacing
                and self.copy(spacing=False, sep=TokenSep.instance)
                or self)

    def with_spacing(self, spacing: bool = True) -> PpTok:
        """
        If given value is true (the default), return self, or copy, with
        spacing = True.  Otherwise self unchanged.
        """
        return (not self.spacing and spacing
                and self.copy(spacing=True, sep=TokenSepSpace.instance)
                or self)

    def revert(self, typ: TokType = None, *, unicode: bool = False,
               spliced: bool = False, trigraph: bool = False,
               orig_spelling: bool = False # don't do special clang stuff.
               ) -> PpTok:
        """
        Reconstruct the original text for the token value and store it in the
        value.  Some replacements won't be reverted, based on the token type.
        Returns new token if changed, else self.
        """
        if not self.repls or self.reverted_from:
            return self
        # Get the options, from the token type or from the keywords.
        if typ:
            spliced, trigraph, unicode = (
                typ.spliced, typ.trigraph, typ.unicode)
        # Special case.  clang reverts to the unicode character even
        # if it was a UCN in the source.
        codepoint = (unicode and not orig_spelling
                     and typ and not typ.quoted and self.lexer.prep.clang
                    )
        newvalue = self.lexer.repls.revert(
            self, spliced, trigraph, unicode, codepoint)
        return self.copy(reverted_from=self, value=newvalue,
                         value_orig=self.value)

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
        return break_match(line=self.loc and self.phys_lineno,
                           col=self.loc and self.colno,
                           pos=self.datapos,
                           file=self.loc and self.source
                           and self.source.filename,
                           )

    def wsval(self) -> str:
        """ The token value with preceding whitespace as a space character.
        """
        return f"{self.spacing and ' ' or ''}{self.value}"

    # Special characters used in repr()...
    class Reprs:
        null = 'φ'
        spacing = '•'
        toline = '↑'
        tocol = '→'
        paste = 'π'        # '##' concatenate in macro
        tostr = 'Σ'        # '#' stringize in macro

    def __str__(self) -> str:
        val = self.value
        if not self.type:
            rep = f"{val!r}"
        elif self.type.str:
            rep = f'"{val}"'
        elif self.type.dir and hasattr(self, 'line'):
            rep = f"\'{self.line}\'"
        elif self.type.passthru:
            rep = "<passthru>"
        elif self.type.paste:
            rep = self.Reprs.paste
        elif self.type.stringize:
            rep = self.Reprs.tostr
        elif val:
            rep = f"{val}"
        elif self.type.pos: rep = f'<pos {self.pos}>'
        else: rep = self.Reprs.null
        return rep

    def __repr__(self) -> str:
        rep = str(self)
        indent = self.indent
        if indent:
            # Sets output position, maybe indirectly.
            if indent.colno > 1:
                rep = f"{self.Reprs.tocol}{indent.colno - 1} {rep}"
            rep = f"{self.Reprs.toline}{indent.phys_lineno} {rep}"
        rep = f"{rep} @{self.loc.showpos}"
        if self.hide:
            rep = f"{rep} - {self.hide!r}"
        if self.spacing:
            rep = f"{self.Reprs.spacing} {rep}"

        return rep


class MoveTok(PpTok):
    """
    Specialized PpTok which carries an OutPosMove as self.pos to indicate a
    change in output location which takes effect at this token's location.
    This is reused by all tokens until a new OutPosMove is seen.
    """
    def __init__(self, lexer: PpLex, pos: OutPosMove, **attrs):
        """ Constructed from another token and an OutPosChange.  Copies the
        token's location and other attributes, but with type CPP_NEWPOS.
        """
        super().__init__(lexer, **attrs)
        self.pos = pos
        self.addline = pos.lineno - self.lineno - 1

    def __getitem__(self, src_lineno: int) -> int:
        """ Maps a source line to a presumed output line. """
        return src_lineno + self.addline

    def linemacro(self, m: PpTok) -> int:
        """
        Evaluate a __LINE__ macro token where self is the move currently in
        effect.
        """
        return m.phys_lineno + self.addline

    def filemacro(self) -> str:
        """ Evaluate a __FILE__ macro token. """
        return self.pos.filename or self.loc.output_filename

@dataclass(frozen=True)
class TokLoc:
    """
    The location within a Source of a PpTok.  It does not change even if other
    token attributes change or the token is copied.  Useful to compare tokens.
    """
    # Logical Line number, starting at 1.
    lineno: int
    # The Source it was lexed from.
    source: Source
    # Offset of the value in the Source data
    datapos: int
    # Column number of start of value, starting at 1.
    colno: int
    # Current out location change.
    move: PresumeMover
    # Physical line number - self.lineno.
    phys_offset: ClassVar[int] = 0

    def copy(self, **attrs) -> Self:
        return dataclasses.replace(self, **attrs)

    @property
    def phys_lineno(self) -> int:
        """
        Physical line where token is located, rather than lineno, which is the
        logical line.
        """
        return self.lineno + self.phys_offset

    @property
    def output_lineno(self) -> int:
        """ Presumed line number for writing a token. """
        return self.move.lineno(self.phys_lineno)

    @property
    def output_filename(self) -> str:
        """
        Presumed file name for writing a token.  Used for __FILE__ macro at
        this location. """
        return self.move.filename

    def linemacro(self) -> int:
        """
        Evaluate this token as a __LINE__ macro.  self.move maps phys line
        numbers to output line numbers.  Without self.move, the mapping is an
        identity.
        """
        if self.move:
            return self.move.linemacro(self)
        else:
            return self.phys_lineno

    def filemacro(self) -> str:
        """ Evaluate this token as a __FILE__ macro. """
        if self.move:
            return self.move.filemacro()
        else:
            return self.source.filename

    @property
    def showpos(self) -> str:
        """ String for just the line and column, not the source. """
        if self.phys_offset:
            line = f"{self.lineno}/{self.lineno + self.phys_offset}"
        else:
            line = f"{self.lineno}"
        return f"{line}:{self.colno}"

    def brk(self) -> bool:
        """ Break condition for debugging. """
        return break_match(line=self.phys_lineno,
                           col=self.colno,
                           pos=self.datapos,
                           file=self.source and self.source.filename,
                           )

    def __str__(self) -> str:
        filename = self.move.filename
        filename = (f'/{filename!r}'
                    * bool(filename != self.source.filename))
        return f"{self.source}{filename} @{self.showpos}"

    def __repr__(self) -> str:
        return f"{self}"

s = TokLoc(1, 2, 3, 4, 5)
    
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

# New and improved TokIter class.
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

    # The generator of Tokens.  self.gen() returns next token or
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

    if __debug__: _serial = itertools.count(1)

    def __init__(self, gen: Iterable[PpTok] = None):
        """
        Constructor for the last iterable in the chain.  When given `gen` is
        exhausted, self.gen reverts to self.Empty and StopIteration is raised.
        """
        if gen:
            if isinstance(gen, typing.Sequence):
                self.gen = (self.SeqGen(self, gen, self.Empty))
            else:
                self.gen = (self.PairGen(self, gen, self.Empty))
        else:
            self.gen = self.Empty
        if __debug__: self.ser = next(self._serial)

    def __iter__(self): return self
    def __next__(self) -> PpTok:
        return next(iter(self.gen))

    def __bool__(self) -> bool: return bool(self.gen.peek())

    def peek(self) -> PpTok | None:
        return self.gen.peek()

    @classmethod
    def class_init(cls):
        cls.Empty = cls.EmptyGen()

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

    def putback(self, token: PpTok) -> None:
        """ Puts a given token in front of the existing iteration. """

        self.gen = self.LookaheadGen(self, token, self.gen)

    def prepend(self, gen: Iterable[PpTok]) -> None:
        """
        Puts a given token iterable in front of the existing iteration.
        """
        if isinstance(gen, typing.Sequence):
            self.gen = self.SeqGen(self, gen, self.gen)
        else:
            self.gen = self.PairGen(self, gen, self.gen)

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
    def strip(self) -> Iterator[PpTok]:
        """ Remove leading/trailing whitespace. """
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

        def gen(self) -> Iterator[PpTok]:
            tail = self.tail
            self.ti.gen = tail
            yield self.tok

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
            if __debug__: self.i += 1
            if tok: return tok

            self.ti.gen = iter(self.tail)
            return next(self.ti)

        def gen(self) -> Iterator[PpTok]:
            for self.i in range(self.len):
                yield self.head[self.i]
            self.ti.gen = iter(self.tail)
            tok = next(self.ti, None)
            if tok: yield tok

        def peek(self) -> PpTok | None:
            return self.toks[self.i]

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

# This will iterate with 42, 43, 44.

ti = TokIter.empty()
ti.putback(42)

ti = TokIter([42, 43])
t = next(ti)
t = next(ti)
t = next(ti, None)
ti = TokIter.from_tokens([42, 43])
next(ti)
next(ti)
next(ti, None)
ti = TokIter.from_token(44)
ti.prepend([42, 43])
next(ti)    # 42
next(ti)    # 43
next(ti)    # 44
next(ti, None)
for t in ti:
    pass
# This will iterate with 41, then 42, 43, 42, 43, ... forever.
ti = TokIter(itertools.cycle([43, 44]))
ti.putback(42)
next(ti)    # 42
next(ti)    # 43
next(ti)    # 44
next(ti)    # 43
next(ti)    # 44

# Empty iterator
ti = TokIter.empty()
bool(ti)    # False
ti.peek()   # None
next(ti, None)  # None

def reduce_ws(toks: Iterable[PpTok], prep: Preprocessor) -> Iterable[PpTok]:
    """ Compresses whitespace, as in translation phase 3.
    However, comments will be preserved if the Preprocessor says so.  Tabs
    have been expanded, if specified by command line arguments.

    Zero or more whitespace characters before the first non-whitespace
    character are replaced by an indentation token

    One or more other whitespace characters are replaced by a single space.

    Consumes the input if it is an iterator.
    """

    tok: PpTok
    have_ws: bool = False
    all_ws: bool = True

    for tok in toks:
        if tok.type.nl:
            have_ws = False
            all_ws = True
        elif (tok.type.ws
              and (not tok.type.comment
                   or not prep.on_comment(tok)
                  )
             ):
            # This is whitespace to compress
            if have_ws or all_ws: continue
            # First consecutive whitespace token.
            have_ws = True
            tok = tok.make_space()
        else:
            # Non-whitespace.
            if all_ws:
                #yield tok.make_indent()
                all_ws = False
            have_ws = False

        yield tok

@TokIter.from_generator
def filt_line(toks: TokIter, lineno: int) -> Iterator[Pptok]:
    """
    Generates all incoming tokens having the given line number.  Any following
    token is put back into the input.
    """
    for tok in toks:
        if tok.lineno == lineno:
            yield tok
        else:
            toks.putback(tok)
            break

def split_lines(toks: Tokens) -> Iterator[Tokens]:
    """ Breaks the Tokens into groups separated by newlines.
    Each line is a Tokens object (without the newline),
        which is reused each time.
    """
    line = Tokens()
    for tok in toks:
        if tok.type.nl:
            yield line
            line.clear()
        else:
            line.append(tok)
    if line:
        yield line

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

