""" escape.py  
Handles escape sequences in source file data.

Regexes for lexing.

Replacement strings.

Numeric value.
"""

from __future__ import annotations

import codecs

from pcpp.common import *
from pcpp.regexes import RegExes

# Regular expressions.  Varies with language options.  
# Before lexing the source file, and after performing translation phases 1 and
# 2, the lexer replaces each escape with a single character, if possible, or
# else a canonical error escape.

#@functools.cache
#def escape_regex(lang: Preprocessor.Language):
#    REs = RegExes(lang)

#    def body_alts() -> Iterator[str]:
#        """ Generate the alternative regexes for the escape body. """
#        # Reference C 6.4.4.4. or C++ 5.13.3.
#        hex = '[0-9a-fA-F]'
#        oct = '[0-7]'
#        # simple-escape-sequence.
#        yield r'[\'"?\\abfnrtv]'
#        # octal-escape=seuence
#        yield rf'{oct}{{1,3}}'
#        # hexadecimal-escape=seuence
#        yield rf'x{hex}+'
#        # universal-character-name (C 6.4.4.4, C++ 5.3)
#        yield rf'u{hex}{{4}}'
#        yield rf'U{hex}{{8}}'
#        # New in C++23, also clang for all languages;
#        if lang.cplus_ver >= 2023 or lang.clang:
#            # universal-character-name
#            yield rf'u{{{hex}+}}'
#            # named-universal-character
#            n_char = '[^\\n}}]'         # any but '}' or '\n'.
#            yield rf'N{{{n_char}+}}'
#            # octal-escape=seuence
#            yield rf'o{{{oct}+}}'
#            # hexadecimal-escape=seuence
#            yield rf'x{{{hex}+}}'
#        # conditional-escape-sequence.  Any single char not in the above.
#        yield '.'

#    body = REs.joinalts(*body_alts())
#    regex = REs.wrap(rf'\\{body}')
#    return regex


class EscapeDiag(Exception):
    """
    Diagnostic, if any, for an Escape.  The message, if any, is str(self).

    The severity depends on the context where the escape appears.
    """

    class Severity(enum.Enum):
        Ignore = enum.auto()
        Warn = enum.auto()
        Err = enum.auto()
        def __bool__(self) -> bool:
            return self is not self.Ignore
        @property
        def warn(self): return self is self.Warn
        @property
        def err(self): return self is self.Err
        @classmethod
        def get(cls, sev: bool | None) -> Self:
            return (cls.Ignore if sev is None
                    else cls.Err if sev
                    else cls.Warn)

    sev: Severity = Severity.Ignore

    _quoted: Severity = None    # Severity in quoted token, if different.
    _ctrlexpr: Severity = None  # Severity in control expr, if different.

    def __init__(self, msg: str, /,
                 ctrlexpr: bool = None, quoted: bool = None, **kwds):
        super().__init__(msg)
        if kwds: self.__dict__.update(**kwds)
        if ctrlexpr is not None:
            self._ctrlexpr = self.Severity.get(ctrlexpr)
        if quoted is not None:
            self._quoted = self.Severity.get(quoted)

    def __bool__(self) -> bool:
        return bool(self.sev)

    @property
    def msg(self) -> str | None:
        return str(self)

    @property
    def quoted(self) -> Severity:
        return self.sev if self._quoted is None else self._quoted

    @property
    def ctrlexpr(self) -> Severity:
        return self.sev if self._ctrlexpr is None else self._ctrlexpr


class EscapeNoDiag(EscapeDiag):
    """ Default Escape.diag to indicate absence of error or warning. """
    def __init__(self):
        super().__init__(None)


class EscapeError(EscapeDiag):
    sev = EscapeDiag.Severity.Err


class EscapeWarning(EscapeDiag):
    sev = EscapeDiag.Severity.Warn


class Escape:
    """
    Describes an escape sequence in source data, given a Match object.
    Contains the replacement string and possibly a message.

    All escapes may appear in any quoted token.  The lexer matches the
    replacement data, but the token's value contains the original escape text.

    Universal character names (UCNs) are allowed elsewhere and are lexed as
    part of an identifier.  Any other escapes are lexed as errors.
    """
    # The actual escape sequence
    esc: str
    # The chars which determine the value, from the Match capturing group.
    body: str
    # The numeric value.  May be a class attribute or a property.  -1 if there
    # is no value.
    @functools.cached_property
    def value(self) -> int:
        return self.getvalue()
    # The replacement string.  None means no replacement.
    repl: str | None
    # Is it a UCN escape?  True for EscUni and subclasses.
    unicode: typing.ClassVar[bool] = False
    # Error/warning (if any).
    diag: EscapeDiag = EscapeNoDiag()
    # Error/warning message
    msg: str = None
    # True if msg and msg is an error.
    err: bool = False

    def __init__(self, mgr: EscapeMgr, old: str, body: str,
                 *, ctrlexpr: bool = False):
        self.mgr = mgr
        self.esc = old
        self.body = body
        self.ctrlexpr = ctrlexpr

    @property
    def repl(self) -> str:
        """
        Replacement for the escape in the ESCAPE stage of the Lexer
        replacement pass.  There is always some replacement string, even for
        invalid escapes.  In most cases, it is a single unicode character
        designated by the escape.  Otherwise it is a placeholder string which
        will let it be lexed as part of a quoted or an identifier token.
        """
        val = self.value
        if val < 0:
            # Placeholder for escape which has no defined value.
            return r'\{}'
        # NEW: Don't change anything.
        return None
        try:
            if val >= 0x20:
                return chr(val)
        except ... as e:
            # Value is out of range.
            pass
        # Value is out of range, or an ASCII control character.
        return rf'\{{{val:x}}}'

    @property
    def err(self) -> bool: return self.msg and not self.warn

    @property
    def msg(self) -> str | None:
        return self.diag.msg

    @property
    def err(self) -> bool:
        return self.diag.sev.err

    @property
    def warn(self) -> bool:
        return self.diag.sev.warn

    def set_diag(self, msg: str, warn: bool = False, **kwds) -> None:
        """ Set a message and keep going. """
        cls: Type[EscapeDiag] = (EscapeError, EscapeWarning)[warn]
        self.diag = cls(f"{msg}: {self.esc!r}", **kwds)

# Simple escapes...
class EscSimple(Escape):
    """ Simple escape '\' + single character with corresponding value. """
    pass

def make_simple_escape(body: str):
    """
    Create global class EscXX, with given body and corresponding body and
    replacement.  Class will be a singleton.
    """
    val: int = ord(body)
    name: str = f'{val:02X}'
    clsdict = dict(value=val,
                   body=body,
                   _repl=rf'\{{{name}}}',
                   __new__=lambda cls: cls.instance,
                   )
    cls: Type[Escape] = type(f'Esc{name}', (Escape, ), clsdict)
    cls.instance = object.__new__(cls)
    globals()[cls.__name__] = cls


'\'"?abfnrtv'
class Esc27(EscSimple): value = 0x27; _repl = r'\{27}'; body = '\''
class Esc22(EscSimple): value = 0x22; _repl = r'\{22}'; body = '"'
class Esc3F(EscSimple): value = 0x3F; _repl = r'\{3f}'; body = '?'
class Esc07(EscSimple): value = 0x07; _repl = r'\{07}'; body = 'a'
class Esc08(EscSimple): value = 0x08; _repl = r'\{08}'; body = 'b'
class Esc0C(EscSimple): value = 0x0C; _repl = r'\{0c}'; body = 'f'
class Esc0A(EscSimple): value = 0x0A; _repl = r'\{0a}'; body = 'n'
class Esc0D(EscSimple): value = 0x0D; _repl = r'\{0d}'; body = 'r'
class Esc09(EscSimple): value = 0x09; _repl = r'\{09}'; body = 't'
class Esc0B(EscSimple): value = 0x0B; _repl = r'\{0b}'; body = 'v'
class Esc5C(EscSimple): value = 0x5C; _repl = r'\{5c}'; body = '\\'


class EscBody(Escape):
    """ Any escape '\' + optional prefix + (body or { body }). """
    # The body characters, as a property.
    body: str

    # Full length of body, if > 0.
    len: int = 0


class EscBodyLen(EscBody):
    """ An escape with a designated full length of the body. """

    def __init__(self, mgr: EscapeMgr, old: str, body: str, **kwds):
        super().__init__(mgr, old, body, **kwds)
        # Make a warning if body is shorter than full length.
        if len(body) < self.len:
            self.set_diag(rf'Truncated escape sequence',
                       #warn=True,
                       )

class EscDelim(EscBody):
    """ Escape of the form '\' <prefix> { <body> }. """
    def __init__(self, mgr: EscapeMgr, old: str, body: str, **kwds):
        super().__init__(mgr, old, body, **kwds)
        # Make an error if escape doesn't end with <body> }, or if body is
        # empty.
        if not body:
            self.set_diag(rf'Delimited escape cannot be empty')
        if self.esc[len(body) + 3 : ] != '}':
            self.set_diag(rf'Malformed escape sequence')

class EscOctVal(Escape):
    """ Escape with octal digits. """
    def __init__(self, mgr: EscapeMgr, old: str, body: str, **kwds):
        super().__init__(mgr, old, body, **kwds)
        if not body:
            self.set_diag(rf'Escape requires 1 - 3 octal digits')

    def getvalue(self) -> int:
        return int(self.body or '0', 8)


class EscHexVal(Escape):
    """ Escape with hexadecimal digits. """
    def __init__(self, mgr: EscapeMgr, old: str, body: str, **kwds):
        super().__init__(mgr, old, body, **kwds)
        if not body:
            self.set_diag(rf'Escape requires 1 or more hex digits',
                          ctrlexpr=False)

    def getvalue(self) -> int:
        return int(self.body or '0', 16)


# Unicode escapes...
class EscUnicode(Escape):
    """
    Any universal char name (UCN) escape: '\' [uUN] ... .  Certain values are
    errors.
    """
    unicode = True

    def __init__(self, mgr: EscapeMgr, old: str, body: str, **kwds):
        super().__init__(mgr, old, body, **kwds)
        self.getvalue()

    def getvalue(self) -> int | None:
        #if self.err:
        #    return None
        codepoint: int = super().getvalue()
        # Check for invalid codepoint.
        if codepoint is not None:
            handler = self.mgr.invalid_codepoints.get(codepoint)
            if handler:
                handler(self, codepoint)
        return codepoint

    @property
    def repl(self) -> str | None:
        """
        Replacement for the escape in the ESCAPE stage of the Lexer
        replacement pass.  There is always some replacement string, even for
        invalid escapes.  In most cases, it is a single unicode character
        designated by the escape.  Otherwise it is a placeholder string which
        will let it be lexed as part of a quoted or an identifier token.
        """
        if self.err:
            return None
        val = self.value
        if val and val < 0:
            # Placeholder for escape which has no defined value.
            return None
            return r'\{}'
        try:
            if val and val >= 0x20:
                return chr(val)
        except ValueError:
            # Value is out of range.
            pass
        # Value is out of range, or an ASCII control character.
        return None
        return rf'\{{{val:x}}}'

    # Handlers for an invalid codepoint ...
    def diag_name(self, codepoint: int) -> int:
        """ Handle an unknown universal character name.  Return new value. """
        if not self.ctrlexpr:
            self.set_diag(f"Unknown universal character name")
        return codepoint

    def diag_control(self, codepoint: int) -> None:
        """ Handle a codepoint which is a control code. """
        if not self.ctrlexpr:
            self.set_diag(
                f"Universal character name refers to a control character",
                quoted=False,
                )

    def diag_ascii(self, codepoint: int) -> None:
        """ Handle a codepoint which is an ASCII char. """
        if not self.ctrlexpr:
            char = chr(codepoint)
            self.set_diag(
                f"ASCII character {char!r} cannot be "
                f"a universal character name",
                quoted=False,
                )

    def diag_surrogate(self, codepoint: int) -> None:
        """ Handle a codepoint which is a surrogate. """
        self.set_diag(f"Unicode escape specifies a surrogate codepoint")

    def diag_range(self, codepoint: int) -> None:
        """ Handle a codepoint over maximum range. """
        self.set_diag(f"Unicode escape value out of range")

    def diag_range_clang(self, codepoint: int) -> None:
        """ Handle a codepoint over maximum range.  Used in clang mode. """
        self.set_diag(f"Unicode escape value out of range", skip=True)

    def codepoint_escape(self, codepoint: int) -> str:
        """ Make a unicode escape string for the codepoint. """
        return rf'\{{{codepoint:x}}}'


class EscUni4(EscUnicode, EscBodyLen, EscHexVal, ): len = 4
class EscUni8(EscUnicode, EscBodyLen, EscHexVal, ): len = 8
class EscUniDelim(EscUnicode, EscDelim, EscHexVal, ): pass


class EscNucVal(EscDelim):

    def getvalue(self) -> int:
        """ Value for \\N{name}. """
        name = self.body.strip().replace('_', '-')
        try:
            new = ord(codecs.decode(
                f'\\N{{{name}}}', 'unicode-escape'))
            # Check for lowercase letters and spaces with
            # clang.
            if self.mgr.lang.clang and name.upper() != name:
                # With lowercase letters, clang reports an
                # error, but makes the replacement anyway.  However, in a
                # control expression, it is a real error.
                self.set_diag(
                    f"clang requires an exact match, "
                    f"including case and whitespace, in unicode name",
                    warn=not self.ctrlexpr,
                    )
        except UnicodeDecodeError:
            self.set_diag(f"Unknown unicode name {name!r}")
            new = None
        return new

class EscNuc(EscUnicode, EscNucVal): pass


# Numeric escapes...
class EscOct(EscOctVal): pass
class EscHex(EscHexVal): pass
class EscOctDelim(EscDelim, EscOctVal): pass
class EscHexDelim(EscDelim, EscHexVal): pass


# Undefined escape.
class EscUndef(Escape):

    repl = None

    def __init__(self, mgr: EscapeMgr, old: str, body: str, **kwds):
        super().__init__(mgr, old, body, **kwds)
        self.set_diag(f"Unknown escape, "
                   f'replacing with {self.esc[1]!r}',
                   warn=self.ctrlexpr,
                   )

    def getvalue(self) -> int:
        return ord(self.esc[1])


class EscapeMgr:
    """
    Object which will evaluate an escape sequence and produce an Escape
    result.  Also holds the RE for lexing an escape from source data after
    translation phases 1 and 2.

    """
    lang: Preprocessor.Language

    # Regular expression which matches any escape, or any unicode escape.
    regex: str
    uni_regex: str

    ## Type of Escape initializer.  Returns replacement string.
    #Call = typing.Callable[[Escape], str]

    ## Type of entry in self.disp dispatch table.
    #class Disp(typing.NamedTuple):
    #    key: str
    #    zero: int
    #    group: int
    #    cls: Type[Escape]
    #    #call: Call
    #    #args: tuple

    # Dispatch table, which translates an escape to a subclass of Escape.
    # Entries are indexed by matching group index.  There is a separte table
    # for escape which is limited to unicode.
    classes: list[typing.Type[Escape]]
    uni_classes: list[typing.Type[Escape]]

    # Dispatch table to handle invalid codepoints in a unicode escape.
    invalid_codepoints: RangeMap

    def __init__(self, lang: Preprocessor.Language):
        self.lang = lang

        # Type of escape information for each escape pattern.  Consists of a
        # string formatted as "key, pattern, classname".  
        # The `key` is a string which is <= any escape (after the initial '\')
        # with the pattern and separates this pattern from all the others.
        # `pattern` is what is matched by the lexer.  It contains exactly one
        # capturing group.  

        PatternInfo = 'tuple[str, str | typing.Type[Escape]]'

        hex = '[0-9a-fA-F]'
        oct = '[0-7]'

        REs = lang.REs

        def genpatterns() -> Iterator[PatternInfo]:
            """
            Generate the alternative pattern information for each escape body
            pattern.  The class can be either a class or the name of one.
            """
            # Reference C 6.4.4.4. or C++ 5.13.3.
            uni: RegExes.UniInfo = REs.uni_info

            # simple-escape-sequence.
            for c in '\'"?abfnrtv':
                arg = codecs.decode(rf'\{c}', 'unicode-escape')
                yield rf'([{c}])', rf'Esc{ord(arg[-1]):02X}'
            arg = codecs.decode('\\\\', 'unicode-escape')
            yield r'([\\])', rf'Esc{ord(arg[-1]):02X}'
            # octal-escape-sequence.
            yield rf'({oct}{{1,3}})', EscOct
            # hexadecimal-escape-sequence.
            yield rf'x({hex}+)', EscHex
            # universal-character-name (C 6.4.4.4, C++ 5.3)
            yield uni.ucn4, EscUni4
            yield uni.ucn8, EscUni8
            # New in C++23, also clang for all languages:
            if lang.cplus_ver >= 2023 or lang.clang:
                # universal-character-name
                yield uni.ucndelim, EscUniDelim
                # named-universal-character
                yield uni.nuc, EscNuc
                # octal-escape=sequence, including bare \o
                yield rf'o{REs.delimited(oct)}', EscOctDelim
                yield rf'o()', EscOctDelim
                # hexadecimal-escape=sequence
                yield rf'x{REs.delimited(hex)}', EscHexDelim
            # hexadecimal-escape-sequence, with no digits.
            yield rf'x()', EscHex
            # conditional-escape-sequence.  Any single char not in the above.
            yield '(.)', EscUndef 

        infos: list[PatternInfo] = list(genpatterns())

        # Build the regexes and dispatch tables from the pattern infos.
        patts: list[str] = []
        uni_patts: list[str] = []
        classes = self.classes = [None]
        uni_classes = self.uni_classes = [None]
        for info in infos:
            patt, cls = info
            if isinstance(cls, str):
                cls = eval(cls)
            if issubclass(cls, EscUnicode):
                uni_patts.append(patt)
                uni_classes.append(cls)
            patts.append(patt)
            classes.append(cls)
        body = REs.joinalts(*patts)
        self.regex = REs.wrap(rf'\\{body}')
        # Same for each plain char or escape sequence in char constant.
        self.char_regex = re.compile(REs.wrap(rf'\\{body}|.'))
        body = REs.joinalts(*uni_patts)
        self.uni_regex = re.compile(REs.wrap(rf'\\{body}'))

        self.invalid_codepoints = RangeMap(
            (Range(-1, 0x00), EscUnicode.diag_name),
            (Range(0x00, 0x20), EscUnicode.diag_control),
            (Range(0x20, 0x7F), EscUnicode.diag_ascii),
            (Range(0x7F, 0xA0), EscUnicode.diag_control),
            (Range(0xD800, 0xE000), EscUnicode.diag_surrogate),
            (Range(
                sys.maxunicode + 1, 0x_1_0000_0000),
                lang.clang and EscUnicode.diag_range_clang
                or EscUnicode.diag_range),
            )

    def __call__(self, esc: str, m: re.Match, uni: bool = False, **kwds
                 ) -> Escape:
        """
        An Escape object which contains the replacement string for the escape
        and some other information.  Optionally is only for unicode.
        """
        i: int = m.lastindex
        cls: Type[Escape] = (self.uni_classes if uni else self.classes)[i]
        return cls(self, esc, m.group(i), **kwds)

    def char_eval_iter(self, val: str, tok: PpTok, exc: bool = True
                       ) -> Iterator[int]:
        """
        Generates the numerical value of each escape sequence or other
        character, within the string in character constant token.  Errors in
        escapes will post diagnostics but use a value of 0, or else raise the
        diagnostic as an exception.
        """
        for m in re.finditer(self.char_regex, val):
            val = m.group()
            if val[0] == '\\':
                esc: Escape = self(val, m, ctrlexpr=True)
                if esc.diag.ctrlexpr.err:
                    if tok:
                        tok.lexer.prep.on_error_token(tok, esc.msg, esc.warn)
                    if exc:
                        raise esc.diag
                yield esc.value    # This may be 0 if an error.
            else:
                # Must be single character
                yield ord(val[0])

        return

    def char_eval(self, val: str, max: int = None, exc: bool = True) -> int:
        """ The numerical value of the content of a character constant. """
        vals = list(self.char_eval_iter(val, None, exc=exc))
        if max:
            vals = vals[- max : ]
        n = 0
        for v in vals:
            if v is None:
                return None
            n = (n << 8) + v
        return n

@functools.cache
def escapes(lang: preprocessor.Language) -> EscapeMgr:
    return EscapeMgr(lang)
