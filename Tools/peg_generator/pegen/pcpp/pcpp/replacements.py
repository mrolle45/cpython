""" Module replacements.py

Manages translation phases 1 and 2 replacements to a source data string, and
reverting replacements in a token value.

ReplMgr class performs replacements on original data, and keeps a record of
individual replacements.  It walks through tokens lexed from the replaced data
and provides information about replacements made within the token and
corresponding positions within the original data.

Repl class describes an individual replacement, giving original and replaced strings and positions, as well as what type(s) of replacements were made.  The types are:

* Line splice
* Trigraph
* Unicode character

"""

from __future__ import annotations

import codecs
from itertools import count, takewhile

from pcpp.common import *

class ReplMgr(list):
    """ Manages alterations to original source file contents as prescribed
    by translation phases 1 and 2 (C99 5.2pp1-2).  These consist of:
        Trigraph sequences.
        Line splices (including the backslash coming from a trigraph).
        Unicode codepoints.
        Universal char names.  For GCC emulation, these are replaced by
            GCC's spelling of them in the output .i file.

    First, it takes the original data and makes the changes, keeping
    a list of the individual changes.

    Then, it supports lexing of individual tokens by providing a list of
    the changes occurring within that token, and also makes adjustments
    to the lexer's line number and start of current line for line splices.

    Some tokens should not have certain replacements performed, for example,
    raw strings.  The replacements have been already done before the type of
    the token is known, and so the token has a method to revert the
    replacement if needed.

    Optimized when there are no repls.
    """
    repl_pat: re.Pattern        # Matches any string that gets replaced.

    # Map trigraphs and line splices to their replacements.
    repl_lookup = {
        '??=':'#',      '??/':'\\',     "??'":'^',
        '??(':'[',      '??)':']',      '??!':'|',
        '??<':'{',      '??>':'}',      '??-':'~',
        '\\\n':'',      '??/\n':'',
    }

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.REs = lexer.REs
        self.repl_pat = re.compile(self.REs.repls)

    # These methods are used in setting up the replacement list...

    def do_repls(self, lexer: Lexer, input: str) -> str:
        """
        Find all the changes of the input for translation phases 1 and 2.
        These are:
            - the 9 trigraphs, which are 3 characters and are replaced with
                one character.  This was a way of encoding source
                characters which were not in a 6-bit character set.
            - line splice, a '\\\n' sequence which is simply deleted.
            - A combination, which uses a trigraph for the backslash,
                as '??/\n'.  All 4 characters are deleted.
            - A Unicode codepoint, a single character.
                Replaced by its UCN spelling.
            - A Unicode escape (GCC only).  Changed to 8 digits lower case.
        """
        if not self.repl_pat.search(input):
            return input

        self.repls: list[Self.Repl] = []
        # Matches that couldn't be replaced.
        self.errors: list[re.Match] = []
        # Changes in position accumulated as replacements are made.
        self.delta_pos: int = 0

        prep = self.prep = self.lexer.prep

        def replace(m: re.Match) -> str:
            """
            Callback for re.sub() for each matched pattern, returning
            replacement character(s).

            Also, add a Repl object for this replacement to the repls[] list.
            """
            old = m.group()
            new = self.repl_lookup.get(old)
            codepoint: int = None
            offset: int
            offset = 0

            def tri_back() -> None:
                nonlocal old, offset
                # using ??/ trigraph as the \.  This is actually
                # two replacements - the trigraph then something else.
                self.add('??/', '\\', m)
                old = '\\' + old[3:]
                offset += 2

            if new is None:
                # No replacement value, so old is a unicode codepoint, or a
                #   UCN or NUC escape sequence, or a splice with whitespace.
                if m.group('splice'):
                    new = ''
                    unicode = False
                    if old.startswith('?'):
                        # using ??/ trigraph as the \.  This is actually
                        # two replacements - the trigraph then the splice.
                        tri_back()          # Replaces ??/ in old.
                    if len(old) > 2:
                        # Extra whitespace.
                        self.note(old, m,
                                  "Backslash and newline separated by space.")
                else:
                    if prep.emulate and m.group('ucn'):
                        if old.startswith('?'):
                            # using ??/ trigraph as the \.  This is actually
                            # two replacements - the trigraph then the UCN.
                            codepoint = int(old[4:], 16)
                            tri_back()
                        else:
                            # using plain \.
                            codepoint = int(old[2:], 16)
                    elif ((prep.cplus_ver >= 2023 or prep.clang)
                          and m.group('nuc')
                          ):
                        # \N{name}
                        name = m.group('nuc').strip().replace('_', '-')
                        try: codepoint = ord(codecs.decode(
                            f'\\N{{{name}}}', 'unicode-escape'))
                        except:
                            self.note(old, m,
                                      f"Unknown unicode name {name!r}",
                                      err=True)
                            return old
                        # Check for lowercase letters with clang.
                        if prep.clang and name.upper() != name:
                            self.note(old, m,
                                      f"clang does not support lowercase "
                                      f"letters in unicode name {name!r}",
                                      err=True)
                            #return old

                    else:
                        codepoint = ord(old[0])
                    if codepoint >= 0x10000 or prep.gcc:
                        new = f'\\U{codepoint:0>8x}'
                    else:
                        new = f'\\u{codepoint:0>4x}'
                    unicode = True
            else:
                unicode = False
            self.add(old, new, m, unicode, off=offset,
                     codepoint=codepoint)
            return new

        replaced = self.repl_pat.sub(replace, input)
        self.orig = input
        self.lexer.linepos = 0
        self.lastpos = 0
        self[:] = self.repls
        self.count = len(self)
        self.range = range(0, self.count)
        return replaced

    def add(self, old: str, new: str, m: re.Match, unicode: bool = False,
            *, codepoint: int = None, off: int = 0,
            ) -> None:
        """ Create a replacement record and update position delta. """
        orig_pos = m.start() + off
        delta_len = len(new) - len(old)
        repl_pos = orig_pos + self.delta_pos
        repl = Repl(
            orig_pos=orig_pos, orig=old,
            repl_pos=repl_pos, repl_end=repl_pos + len(new), repl=new,
            )
        if old.startswith('??'): repl.trigraph = True
        if new == '': repl.spliced = True
        if unicode: repl.unicode = True
        self.repls.append(repl)
        self.delta_pos += delta_len
        if self.prep.clang and codepoint is not None:
            repl.codepoint = codepoint

    def note(self, old: str, m: re.Match, msg: str, err: bool = False
             ) -> None:
        """ Record a message in a Repl.  No change to the text. """
        orig_pos = m.start()
        repl_pos = orig_pos + self.delta_pos
        repl = Repl(
            orig_pos=orig_pos, orig=old,
            #repl_pos=repl_pos, repl_end=repl_pos + len(old), repl=old,
            repl_pos=repl_pos, repl_end=repl_pos, repl=old,
            msg=msg, err=err,
            )
        self.repls.append(repl)

    # These methods are called to track replacements in tokens...

    def movetotoken(self, tok: PpTok) -> list[Repl]:
        """ Change current position in reduced data to the token end.
        Splices update lexer.phys_lines and lexer.linepos.
        Return list of any Repls passed, and store it in tok.repls.
        """
        repls : list[Repl] = self.movetopos(tok, tok.lexpos + len(tok.value))
        if repls:
            tok.repls = repls
        return repls

    def movetopos(self, tok: PpTok, pos: int) -> list[Repl]:
        """
        Change current position in reduced data, and examine replacements
        passed.  Replacements update lexer.repl_delta.  Splices update
        lexer.phys_lines and lexer.linepos.  Return list of any Repls passed.
        """
        if not self: return []
        repls: list[Repl] = self.get_repls(pos)
        for repl in repls:
            if repl.spliced:
                self.lexer.newphys(repl.repl_end)
            else:
                pass
                self.lexer.repl_delta -= repl.delta_pos
            if repl.msg:
                self.lexer.prep.on_error_token(tok, repl.msg,
                                               warn=not repl.err)
        return repls

    def get_repls(self, pos: int, *,
                  _getpos: Callable = operator.attrgetter('repl_end'),
                  ) -> list[repl]:
        """
        All Repl's within current index range, with end position <= pos.
        Moves current index range past the Repl's.
        """

        # First, a quick test for empty results.
        r = self.range
        if not r:
            return []
        getrepl = self.__getitem__
        test = pos.__ge__
        if not test(getrepl(r.start).repl_end):
            return []

        # Note, all iteration is done within library functions.  There are no
        # loops here.

        positer = map(_getpos, map(getrepl, r))
        x = takewhile(test, positer)
        c = count()
        z = zip(x, c)           # Advance c for elements of x.
        all(z)                  # Runs iterator z without making a list.
        i = r.start
        i2 = i + next(c)
        self.range = range(i2, self.count)
        return self[i:i2]

    def revert(self, tok: PpTok,
               spliced: bool,
               trigraph: bool,
               unicode: bool,
               codepoint: bool,
               ) -> str:
        """
        Revert replacements of specified types in given token.  Return new
        value.
        """
        startpos = tok.datapos
        val = tok.value
        endpos = startpos + len(val)
        parts: list[str] = []
        pos = 0
        for repl in tok.repls:
            # Repl replaced old with new at val[repl.repl_pos - startpos].
            # Check replacements not requested.
            if repl.msg:
                #tok.lexer.prep.on_error_token(tok, repl.msg)

                continue
            if not spliced and repl.spliced: continue
            if not trigraph and repl.trigraph: continue
            if not unicode and repl.unicode: continue
            # Now make the opposite replacement.
            old = repl.orig
            if codepoint and repl.codepoint:
                old = chr(repl.codepoint)
            new = repl.repl
            newpos = repl.repl_pos - startpos
            parts.append(val[pos:newpos])
            parts.append(old)
            pos = newpos + len(new)
        parts.append(val[pos:])
        return ''.join(parts)

    def __repr__(self) -> str:
        return f"< {len(self)} replacement(s) >"


@dataclasses.dataclass
class Repl:
    """ A single replacement in original data. """
    orig_pos: int           # Position in original data before change.
    orig: str = ''          # The original data, e.g. '??='
    repl_pos: int = 0       # Position in replaced data.
    repl_end: int = 0       # Position of data end in replaced data.
    repl: str = ''          # The replaced data, e.g. '#'
    spliced: bool = False   # orig is a line splice --- \\\n.
    trigraph: bool = False  # orig is a trigraph.
    unicode: bool = False   # orig is a unicode codepoint character.
    msg: str = None         # Error or warning message.
    err: bool = False       # If msg is for an error.
    codepoint: ClassVar[int] = None   # ord(char) for unicode char.

    @property
    def delta_pos(self) -> int:
        return len(self.repl) - len(self.orig)

    def __repr__(self) -> str:
        return (f"<Repl {'? ' * bool(self.msg)}"
                f"{self.orig_pos}:{self.orig!r}"
                f" -> {self.repl_pos}:{self.repl!r}>")


class EmptyReplMgr(list):
    """ Specialized replacement manager having no replacements. """

    def movetotoken(*args) -> None: pass
