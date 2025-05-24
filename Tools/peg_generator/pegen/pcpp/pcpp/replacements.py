""" Module replacements.py

Manages translation phases 1 and 2 replacements to a source data string, and
reverting replacements in a token value.

ReplMgr class performs replacements on original data, and keeps a record of
individual replacements.  It walks through tokens lexed from the replaced data
and provides information about replacements made within the token and
corresponding positions within the original data.

Repl class describes an individual replacement, giving original (old) and
replaced (new) strings and positions, old/new status of the token value,
and what type(s) of replacements were made.  The types are:

* Trigraph, e.g., '??=' -> '#'.
* Line splice, e.g., '\\\n' -> ''.  The '\\' could be from a trigraph '??/'.
* Unicode character, possibly no change.
* Note.  No change to text.
"""

from __future__ import annotations

import codecs
from collections import UserList
from itertools import count, takewhile

from pcpp.common import *
from pcpp.escape import Escape, escapes
from pcpp.regexes import RegExes


class ReplStage(enum.IntEnum):
    """
    Which stage of replacements applies to a replacement or other object.

    Basic stages are ones which can be in a Repl object.

    In a ReplMgr, the manager only makes replacements of that stage.

    In a ReplGroup, all Repl's come from that stage.

    In a Repls, each ReplGroup holds Repl's in its stage which were made in
    the token's value.

    If a token is reverted for some stage, a new token is created with a new
    value and a new Repls.  All ReplGroups at that stage or later are removed
    from the Repls and the removed replacements are reverted in the token
    value.  The removed groups are removed in reverse order.  The result is
    the same as though the reverted Repl's were never applied to the original
    data.
    """

    # In Repls, indicating that token reverted back to original value.
    ORIG = enum.auto()
    # In Repl of a trigraph, e.g., ??= -> #
    TRIGRAPH = enum.auto()
    # In Repl of a line splice, '\\\n' -> ''.  The '\\' could have come from a
    # trigraph.
    SPLICE = enum.auto()
    # In Repl of a unicode escape, e.g., \u03b4 -> δ.
    ESCAPE = enum.auto()

    def __repr__(self) -> str:
        return self.name

'''
Step 1: Making replacements,
------
There are two strings of data belonging to a Lexer:
  - Source, or 'lexer.src_data', which is the content of the source file.
    Diagnostics and log messages have a source location, a slice into the
    source data.  These usually belong to a particular token.
  - Lex data, or 'lexer.lex_data', which is the src_data after going through
    the various replacements.  This is the data which the lexer parses to
    break it up into tokens.  The token is associated with a lex location, a
    Range in lexer.lex_data which the lexer consumed to make the token.

The lexer.lex_data is derived from the lexer.src_data by applying replacements
one stage at a time.

Each stage is handled by a ReplMgr object.  The ReplMgr has two strings of
data:
  - Original, or 'mgr.old_data'.  This is the data for which the replacements
    are performed at this stage.
  - Replaced, or 'mgr.new_data'.  This is old_data after replacements.
The first stage ReplMgr takes the src_data as its old_data.  The new_data
from each mgr is the old_data for the next stage mgr.  Finally, the new_data
from the last mgr is the lex_data.

The lexer has a Replacer object, which manages replacements.  It holds only
those mgrs that made at least one replacement.

In the lexer and the ReplMgr, a Range object can be used to extract a val
substring from one of its data strings, as in data[range].  The range and val
are named similarly to the data.  For example, src_data[src_range] is called
'src_val'.

Each ReplMgr keeps a record of the replacements it made.  These are in the
form of a Repl object.

A Repl contains:
  - The substring, or old_val, within old_data which was replaced.
  - The Range, or old_range, of old_val within old_data.
  - The string, or new_val, which it was replaced with, which becomes part of
    new_data.
  - The Range, or new_range, of new_val within new_data.
  - Sometimes it contains a warning or error message.  It has zero-length
    Ranges.  It is same as replacing "" with "".  There might or might not be
    another Repl starting at the same new_range.start.
  - Note: Some of the above are not actually stored in the Repl, but rather
    are available as properties.

As each replacement is made, the remainder of the old_data gets shifted.  The
mgr keeps track of this shift, and so it is able to calculate the new_range
from the old_range by adding the shift.  Replacements are at nondecreasing
ranges and do not overlap. 

Step 2: Tokenizing
------
After all of this replacing of src_data into lex_data, the lexer divides the
lex_data into tokens.  The token has a lex_slice of lex_data.  These tokens
are adjacent.  The token's data val may be the result of some replacements at
some stages.  However, the result of a replacement will never overlap multiple
tokens.

As each token is lexed, all the ReplMgrs will locate any Repl's which lie
within that token.  This is done in reverse stage order.  So the last mgr gets
the new_range of its new_data (which is lex_data) from the lexer.  It finds
the Repl's which lie within that Range.  It then translates that to the
corresponding old_range of old_val in the old_data.  It accounts for shifts
in position resulting from replacements in earlier tokens.  This old_range is
then given to the mgr for the preceding stage as its new_range, and so on.
The old_range produced by the first stage mgr is then stored in the token as
its src_range.

Each ReplMgr, if it finds any Repl's, produces a ReplGroup.  If there are any
groups, then the token gets a Repls object, which holds all the groups in
ascending stage order.

A ReplGroup contains:
  - The ReplMgr which produced it.
  - All the Repl's found, in left-to-right order.
'''

class ReplMgr(list, abc.ABC):
    """
    Manages replacements to original data for a given ReplStage.

    Operates in two steps:
     1. find_repls(orig) searches given old_data for strings to be replaced.
        This data is either the new_data from the previous stage, or else the
        src_data of the lexer.  Makes the replacements and stores Repl objects
        for them.  Returns replaced data as its new_data.
     2. movetotok(tok, new_range).  Associates Repl's with a given token,
        called as the lexer finds each token in its lex_data.  The given range
        is the range in new_data of the of the token value.  The tok argument
        is used to report any warnings or errors to the preprocessor.

        Returns the range for the token in self.old_data (which is the
        new_data for the previous stage or the src_data for the lexer).

        Stores the Repl's as a ReplGroup in tok.repls, if it is non-empty.

        This method is called for the same tok in reverse order of stages.

        The new_ranges are non-overlapping in increasing order, and generally
        adjacent.  If there is a gap between a range and the previous range,
        any Repl's found within the gap are noted, since they affect the
        returned old_range, but not included in the returned ReplGroup.

    Some tokens should not have certain replacements performed, for example,
    raw strings.  The replacements have been already done before the type of
    the token is known, and so the token has a method to revert the
    replacements if needed.  There are two options: (1) revert only the Repl's
    from a given stage and later, as though no Repl's existed at those stages,
    or (2) revert everything and restore the original spelling.

    """

    repls: Replacer

    # The next earlier-stage non-empty manager, if any.
    prev: ReplMgr = None

    # Instance variable is recursive caller to self.prev.movetopos(), if
    # self.prev exists.  Otherwise, this is a method.
    prev_movetopos: Callable[[Range[str], PpTok],
                             Range[str]]

    # Shift from old_pos to corresponding new_pos from current position on.
    delta_pos: int

    # Indices of Repl's not yet examined.  Updated by movetopos().
    repl_index_range: range

    def __init__(self, repls: Replacer, prev_mgr: ReplMgr = None):
        self.repls = repls
        if prev_mgr:
            self.prev = prev_mgr
            def prev_movetopos(new_range: Range, tok: PpTok
                              ) -> list[ReplGroup]:
                return prev_mgr.movetopos(new_range, tok)
            self.prev_movetopos = prev_movetopos

    # These methods are used in setting up the replacement list...

    @functools.cached_property
    def REs(self) -> RegExes:
        return self.repls.lang.REs

    @functools.cached_property
    def repl_pat(self) -> re.Pattern:
        """ Matches any string in old_data that gets replaced. """
        return re.compile(self.REs.repls)

    @property
    def lexer(self) -> PpLex:
        return self.repls.lexer

    def sub_cb(self, m: re.Match) -> str:
        """
        Callback function for re.sub() applied to old_data.  Called with a
        Match for the replacement pattern, and returns a string to replace it
        with.  It may add some Repl objects to the self[:] list.
        """

        return self.get_new(m.group(), m)

    @abc.abstractmethod
    def get_new(self, old: str, m: re.Match) -> str:
        """
        Return replacement string for old string.  Called from re.sub() as a
        callback.  It may add some Repl objects to the self[:] list.
        """
        old = old
        return old

    def find_repls(self, input: str) -> str:
        """
        Find all the changes of the input for this stage
        """
        if not self.repl_pat.search(input):
            return input

        # Changes in position accumulated as replacements are made.
        self.delta_pos: int = 0

        replaced = self.repl_pat.sub(self.sub_cb, input)
        self.old_data = input
        self.new_data = replaced
        self.count = len(self)
        self.repl_index_range = range(0, self.count)
        # Reset delta_pos for accumulation by movetotoken().
        self.delta_pos = 0

        return replaced

    def add(self, old: str, new: str, m: re.Match, stage: ReplStage = None,
            **kwds,
            ) -> str:
        """
        Create a replacement record and update position delta.  Return new
        value.
        """
        old_pos = m.start()
        new_pos = old_pos + self.delta_pos
        repl = Repl(
            self.stage or stage,
            old_val=old, old_pos=old_pos, new_val=new, new_pos=new_pos,
            **kwds,
            )
        if not repl.msg:
            self.delta_pos += repl.delta_len
        self.append(repl)
        return new

    # Test function to compare a token position to a Repl position.
    @staticmethod
    def _testpos(pos: int) -> Callable[[int], bool]:
       return pos.__gt__

    def note(self, m: re.Match, msg: str, quoted: bool = True, **attrs
             ) -> None:
        self.add('', '', m, msg=ReplMsg(msg, quoted, **attrs))

    # These methods are called to track replacements in tokens...

    def src_tok_old_pos(self, old_pos: int) -> int:
        """
        The source position for given position in old_data.  This is relative
        to the current token.
        """
        if self.prev:
            # This is also the position within the previous mgr's new_data.
            return self.prev.src_tok_new_pos(old_pos)
        else:
            # Translate this to the lexer src_data.
            old_pos += self.old_start_pos
            return old_pos

    def src_tok_new_pos(self, new_pos: int) -> int:
        """
        The source position for given position in new_data.  This is relative
        to the current token.  Uses the repls to get corresponding position in
        old_data.
        """
        pos = new_pos
        for repl in self.repls:
            if new_pos < repl.old_pos:
                break
            pos -= repl.delta_len

        return self.src_tok_old_pos(pos)

    @property
    def src_delta_pos(self) -> int:
        """ Shift from lexer.src_pos to self.old_pos at current position. """
        return (self.prev.src_delta_pos + self.delta_pos if self.prev
                else 0)

    def repls_in_range(self, new_range: Range) -> list[Repl]:
        """
        All Repl objects which lie within given range in new_data.  These are
        not yet relocated to be relative to current token.
        """
        # First, a quick test for empty results.
        if not self.repl_index_range:
            return []
        repl_index = self.repl_index_range.start
        nextpos: int = self[repl_index].new_end
        endpos: int = new_range.stop
        if nextpos > endpos:
            return []

        new_repl_index: int = self.get_repls(endpos)
        if new_repl_index == repl_index:
            return []
        return self[repl_index : new_repl_index]

    def movetopos(self, new_range: Range, tok: PpTok) -> list[ReplGroup]:
        """
        Find any replacements, a.k.a. repls, within the given range of offsets
        in self.new_data corresponding to given token.  The lexer is taken
        from the token.  The range is translated to a range in lexer's
        src_data and stored in given token.src_range.

        This method is recursive, in reverse stage order of mgrs.  This allows
        translation of positions in new_data or old_data, for this and all
        later stage mgrs, to be translated to positions in src_data.

        Replacements update lexer.new_delta.  Splices update lexer.phys_lines
        and lexer.linepos.  Returns Range for the token in self.old_data.

        If there are any repls found, they are gathered into a ReplGroup.  The
        return value is list of all such groups for this and earlier stage
        mgrs, in stage order.

        The position ranges are increasing and non-overlapping over all calls
        to this method.  Usually, they are adjacent, but if there are any
        gaps, repls in those gaps are not part of any returned group, but
        otherwise processed.

        Any messages in repls are reported to the preprocessor.
        """
        repls = self.repls_in_range(new_range)

        # Translate to range in old_data.  May need extending later.
        old_range: Range = new_range - self.delta_pos

        old_start_pos = old_range.start
        if not self.prev:
            self.old_start_pos = old_start_pos
        if repls:
            group = ReplGroup(tok, self, repls)
            self.delta_pos += group.delta_len
            new_start_pos = new_range.start
            for repl in group:
                old_pos = repl.old_pos
                repl.old_pos = old_pos - old_start_pos
                #repl.old_pos -= old_range.start
                new_pos = repl.new_pos
                repl.new_pos = new_pos - new_start_pos
            old_range = old_range.extend(- group.delta_len)

        groups: list[ReplGroup] = self.prev_movetopos(
            old_range, tok)

        if repls:
            for repl in repls:
                msg = repl.msg
                if msg:
                    # Usually an error.  In a CHAR token with ASCII codepoint,
                    # the error is ignored if it is in a control expression
                    # (including macro expansion), but reported if it is in
                    # any other context.  More serious errors are always
                    # reported.
                    if tok.type.revert and tok.type.revert <= self.stage:
                        pass
                    elif not (not msg.quoted and tok.type.quoted):
                        self.repls.prep.on_error_token(tok, msg,
                                                       warn=not msg.err)
                        if msg.err:
                            tok.repl_err = True
            if tok.type.id and self.stage is ReplStage.ESCAPE:
                if self.repls.lang.clang and not tok.repl_err:
                    group.skip_id = True

        if repls: groups.append(group)
        return groups

    def get_repls(self, pos: int, *,
                  _getpos: Callable = operator.attrgetter('new_pos'),
                  ) -> int:
        """
        All Repl's within current repl_index_range, with end position <= pos.
        Moves current repl_index_range past the Repl's.  Returns next repl
        index.
        """

        # Note, all iteration is done within library functions.  There are no
        # loops here.

        r = self.repl_index_range
        getrepl = self.__getitem__
        #test = pos.__ge__
        test = pos.__gt__
        test = self._testpos(pos)
        positer = map(_getpos, map(getrepl, r))
        x = takewhile(test, positer)
        c = count()
        z = zip(x, c)           # Advance c for elements of x.
        all(z)                  # Runs iterator z without making a list.
        i: int = r.start
        num_repls: int = next(c)
        if num_repls:
            i += num_repls
            self.repl_index_range = range(i, self.count)
        return i

    def prev_movetopos(self, new_range: Range, tok: PpTok) -> list[ReplGroup]:
        """
        Recursive call to movetopos() when there is no previous mgr.  Given
        range is the range in lexer src_data and is stored in the given token.
        """
        tok.src_range = new_range
        return []

    def special_clang_id(self, tok: PpTok) -> bool:
        """ True if special handling of revert() by clang. """
        return False

    def brk(self) -> bool: return brk(self)

    def __repr__(self) -> str:
        return f"<{self.stage.name} {len(self)}>"


class ReplMgrTrigraph(ReplMgr):
    """ Makes trigraph substitutions in original data. """
    stage: ReplStage = ReplStage.TRIGRAPH

    @functools.cached_property
    def repl_pat(self) -> re.Pattern:
        return re.compile(self.REs.repl_trigraphs)

    def find_repls(self, input: str) -> str:
        """ Find all the changes of the input for this stage. """
        if not self.repls.lang.trigraphs:
            return input
        return super().find_repls(input)

    def get_new(self, old: str, m: re.Match,
                # Map trigraphs to their replacements.
                _repl_lookup = {
                    '??=':'#',      '??/':'\\',     "??'":'^',
                    '??(':'[',      '??)':']',      '??!':'|',
                    '??<':'{',      '??>':'}',      '??-':'~',
                    }
                ) -> str:
        """
        Callback for re.sub() for each matched pattern, returning
        replacement string.

        Also, add a Repl object for this replacement to the repls[] list.
        """
        new = _repl_lookup[old]
        return self.add(old, new, m)


class ReplMgrSplice(ReplMgr):
    """ Makes line splice substitutions in original data. """
    stage: ReplStage = ReplStage.SPLICE

    # Test function to compare a token position to a Repl position.
    @staticmethod
    def _testpos(pos: int) -> Callable[[int], bool]:
       return pos.__ge__

    @functools.cached_property
    def repl_pat(self) -> re.Pattern:
        return re.compile(self.REs.repl_splice)

    def get_new(self, old: str, m: re.Match) -> str:
        """
        Callback for re.sub() for each matched pattern, returning
        replacement string.

        Also, add a Repl object for this replacement to the repls[] list.
        """
        if len(old) > 2:
            # Extra whitespace.
            self.note(m, "Backslash and newline separated by whitespace.")
        return self.add(old, '', m)


class ReplMgrEscape(ReplMgr):
    """ Makes any escape substitutions in original data. """
    stage: ReplStage = ReplStage.ESCAPE

    esc_eval: EscapeFactory

    @functools.cached_property
    def repl_pat(self) -> re.Pattern:
        return re.compile(self.escapes.regex)

    def __init__(self, repls: Replacer, prev_mgr: ReplMgr = None):
        super().__init__(repls, prev_mgr)
        self.escapes = escapes(repls.lang)
        self.escape_pat = self.escapes.regex

    def get_new(self, old: str, m: re.Match) -> str:
        """
        Callback for re.sub() for each matched pattern, returning
        replacement string.

        Also, add a Repl object for this replacement to the repls[] list.
        """
        # Clang changes unicode escapes to their codepoints, but
        # only in identifiers.  They will be reverted in quoted
        # tokens.
        #
        # GCC changes \u escapes to lowercase \U escapes, but only in
        # identifiers.  It does not recognize \N escapes.
        e = self.escapes(old, m)
        new = e.repl
        if e.diag:
            self.note(m, e.msg, e.diag.quoted, err=e.err)
        if new is None:
            new = old
        else:
            self.add(old, new, m)
        return new


class Replacer(list[ReplMgr]):

    """ Manages replacements of original data for a single PpLex. """

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.prep = prep = lexer.prep
        self.lang = prep.lang

    def find_repls(self, input: str) -> str:
        """
        Transforms input data and returns the result.  Sets self[:] = list of
        all ReplMgr's which made change to the data, in stage order.

        input is lexer.src_data.  Returned value will be stored in
        lexer.lex_data.
        """

        '''
        Diagram of replacements.

        ReplMgr changes data:

        old_data        |                                       |

        repl changes    |   old_pos ->  | ...old_val... |       |
            into        |   new_pos ->  | .....new_val..... |       | 
                            delta_pos   |-->                    |
        Note, delta_pos = sum (len(new_val) - len(old_val) over previous
        repls.  Thus, new_pos can be computed from old_pos by adding
        delta_pos.  So the Repl will just store the old_pos, old_val, and
        new_val.

        repeat for each repl.

        result new_data |                                               |

        Replacer combines these:

        lexer.src_data  |   |
                         vvv
        mgr.old_data    |   |
            replace
        mgr.new_data    |       |

        Repeat for each mgr.

                         vvvvvvv
        lexer.lex_data  |       |

        '''

        prev: ReplMgr = None

        cls: Type[ReplMgr]

        for cls in ReplMgrTrigraph, ReplMgrSplice, ReplMgrEscape:
            mgr = cls(self, prev)
            input = mgr.find_repls(input)
            if mgr:
                prev = mgr
                self.append(mgr)

        return input

    def movetotoken(self, tok: PpTok) -> Repls:
        """ Change current position in new_data to the token end.
        Splices update lexer.phys_lines and lexer.linepos.

        Searches each ReplMgr for Repls within the range of the token, and
        gets a ReplGroup with the Repls (if any) found.  The result is a list
        of all the non-empty ReplGroups.  This is stored in tok.repls.  The
        range in the lexer.src_data is stored in tok.src_range.
        """

        '''
        As the lexer finds tokens in its lex_data, this method finds all Repls
        in all mgrs which fall within the range of the token's data.

        To simplify a later revert() method, all Repl positions are translated
        so that they are relative to the current token's range.  This way, the
        token's value can be un-replaced by the Repl's in reverse stage order
        until reaching a desired new stage.

        '''
        # Range of index in a ReplMgr's new_data corresponding to the
        # token.  For the first mgr, this comes from the token's dataspan.
        # Each mgr will adjust this range for its old_data, which is the
        # next mgr's new_data.
        tokrange: Range = tok.datarange

        # Recursive movetopos(), in reverse stage order.
        groups: list[ReplGroup] = self[-1].movetopos(tokrange, tok)

        repls = Repls(tok, *groups)
        if repls: tok.repls = repls
        return repls

    def revert(self, tok: PpTok, stage: ReplStage = None) -> PpTok:
        """
        Revert the token back to given stage.  Return new token if changed, or
        self.
        """
        stage = stage or tok.type.revert
        groups = iter(reversed(tok.repls))

        # All mgrs >= stage take part, even if they didn't actually replace
        # anything, because they still shifted the data range.
        nextgroup = next(groups)
        for mgr in reversed(self):
            if mgr.stage < stage:
                ...

        mgr = next(mgrs)

    def brk(self) -> bool: return brk(self)


class EmptyReplacer(Replacer):
    """ A Replacer with no replacements. """
    def __init__(self) -> None:
        pass

    def movetotoken(self, tok: PpTok) -> None:
        tok.src_range = tok.datarange
        pass


class Repl:
    """
    A single replacement in original data.

    It is created by the ReplMgr when it makes a change to old data, resulting
    in new data.  This is before the lexer's data is tokenized.

    While the lex_data is tokenized, if a token contains a Repl, then the
    repl.tok is set to include it.
    """
    tok: PpTok              # The token affected by self.
    # Position in old_data before change.  When belonging to a token, this is
    # relative to the start of the token old_range.
    old_pos: int
    old_val: str = ''       # The original data, e.g. '??='
    # Accumulated (len(new_val) - len(old_val) over earlier Repls in the mgr.
    delta_pos: int
    # Position in new after change.  When belonging to a token, this is
    # relative to the start of the token new_range.
    new_pos: int
    new_val: str = ''       # The replacement data, e.g. '#'
    mgr: ReplMgr            # Creator of self.
    msg: ReplMsg = None     # Error or warning message.

    def __init__(self, stage: ReplStage, **kwds):
        self.stage = stage
        self.__dict__.update(**kwds)

    @property
    def delta_len(self) -> int:
        """ The shift in data length from old to new. """
        return len(self.new_val) - len(self.old_val)

    @property
    def delta_pos(self) -> int:
        """
        Accumulated (len(new_val) - len(old_val) over earlier Repls in the
        mgr.
        """
        return self.new_pos - self.old_pos

    @property
    def new_end(self) -> int:
        """
        position of end of new_val after replacement, relative to
        mgr.new_data.
        """
        return self.new_pos + len(self.new_val)

    @property
    def change(self) -> bool:
        return self.new_val != self.old_val

    def revert(self, replval: str) -> str:
        """
        Convert new value to old value.  This can be repeated for
        all Repls in a ReplGroup, in right-to-left order, to revert the entire
        value.
        """
        start: int = self.new_pos
        end: int = start + len(self.new_val)
        return f"{replval[:start]}{self.old_val}{replval[end:]}"

    def __repr__(self) -> str:
        if self.msg:
            return (f"<Repl {self.new_pos}: {self.msg!r}")
        else:
            return (f"<Repl {self.new_pos}:{self.new_val!r}"
                    f" <- {self.old_pos}:{self.old_val!r}>")

class ReplMsg:
    """
    An error or warning message, possibly restricted to non-quoted tokens.
    """
    # The message text.
    msg: str

    # Error, as opposed to warning.
    err: bool = False

    quoted: bool = True

    def __init__(self, msg: str, quoted: bool = True, **attrs: bool):
        self.msg = msg
        self.__dict__.update(**attrs)
        if not quoted: self.quoted = quoted

    def __str__(self) -> str:
        return self.msg

    def __repr__(self) -> str:
        p = ''
        if self.err: p = '*'
        if not self.quoted: p += 'Q'
        return f'{p and p + " " or p}{self.msg!r}'


class ReplGroup(tuple[Repl, ...]):
    """ All the Repl's (at least one) belonging to a particular PpTok. """

    tok: PpTok

    # Manager that created self.
    mgr: ReplMgr

    # Shift in position from mgr.old_data to mgr.new_data.
    # Taken from the mgr.delta_pos at constructor time.
    delta_pos: int

    # Shift in position within the data from lexer.src_data to mgr.new_data.
    lex_delta_pos: int

    # Special handling in a CPP_ID token, to emulate clang.  Only the first
    # character is reverted, or if there was a replacement error.  This is
    # overridden if the token is being stringized.
    skip_id: bool = False

    def __new__ (cls, tok: PpTok, mgr: ReplMgr,
                 items: list[Repl]):
        return super().__new__(cls, items)

    def __init__ (self, tok: PpTok, mgr: ReplMgr,
                  items: list[Repl]):
        assert items
        for item in items:
            item.tok = tok
        self.mgr = mgr
        self.tok = tok
        self.delta_pos = mgr.delta_pos
        self.lex_delta_pos = self.delta_pos

    @property
    def delta_len(self) -> int:
        """ 
        Total change in length of value from repl.old_val to repl.new_val when
        the replacements are made.
        """
        return sum(map(operator.attrgetter('delta_len'), self))

    def revert(self, replval: str, force: bool = False) -> str:
        """
        Convert replaced value to original value.
        """
        
        for repl in reversed(self):
            # Check for bypass revert with clang
            if not force and self.skip_id:
            #if not force and self.first_char_only and valpos == repl.new_pos:
                continue
            replval = repl.revert(replval)
        return replval

    def revertall(self, tok: PpTok) -> str:
        """ Revert all replacements in given token.  Return new value. """
        data = tok.source.file.data
        pos = tok.datapos - self.delta_pos
        return data[pos : pos + len(tok.value) - self.delta_len]


class Repls(tuple[ReplGroup, ...]):
    """ All groups of replacements for a particular token. """
    def __new__(cls, tok: PpTok, *groups: replGroup):
        return super().__new__(cls, groups)

    def __init__(self, tok: PpTok, *groups: replGroup):
        self.tok = tok

    def revert(self, tok: PpTok, stage: ReplStage,
               # No special case for clang and CPP_ID.
               force: bool = False
               ) -> PpTok:
        """ New token with reverts up to given stage. """
        val: str = tok.value
        groups = list(self)
        for group in reversed(self):
            if group.mgr.stage >= stage:
                ## Revert this group, unless ignored by clang.
                if not force and group.skip_id:
                    foo = ... #; break
                val = group.revert(val,
                                   force=force)
                                   #force=force or tok.new_err)
                groups.pop()
            else:
                break
        return tok.copy(value=val, repls=Repls(tok, *groups))

# For setting breakpoints in the debugger...
break_lens: list[int] = []      # If not empty, break when len(obj) in here.

def brk(obj) -> bool:
    if break_lens and len(obj) not in break_lens:
        return False
    return break_match(
        file=obj.lexer and obj.lexer.source and obj.lexer.source.filename
             or "")
