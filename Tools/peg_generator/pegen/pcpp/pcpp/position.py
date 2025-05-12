""" position.py
Manages unique locations of data contained in source files.  Position is
identified by a single integer, called Position.

From the Position, one can obtain
    - the Source
    - logical and physical line numbers
    - column number
    - spelling
    - presumed line number
    - presumed file name

Every lexed token has a different position, and so having this be an int
reduces the memory footprint from lots of tokens.  For a token following a
line splice, the token also carries the number of skipped splices.

Positions are allocated in a range to each Source object, with an interval of
1 separating them.  The first Source starts at Position 1.  Any token which
does not come from a Source has a Position of 0.

Note that the Position identifies where the token's value is located in the
source data AFTER phase 1 and phase 2 translations have been performed.  The
Lexer.repls manager can be used to get the corresponding location in the
original data.

"""

from __future__ import annotations

from array import array
from bisect import bisect

from pcpp.common import *
from pcpp.tokens import TokLocMoveBase, TokLocMove

# Position is a number representing a position in a SourceFile's (replaced)
# data.  Each file has its own range of Position values, which are unique over
# all files in the prep.  Values are limited to those in an unsigned long C
# type.

Position = int

@dataclasses.dataclass
class PosRange:
    """
    A range of Position's, equivalent to range(self.start, self.stop).  Also
    behaves like an int with self.start, as in int(self) or self + offset.
    """
    start: Position
    stop: Position

    def __int__(self) -> Position:
        return self.start

    def __add__(self, offset: int) -> Position:
        return self.start + offset

    @property
    def len(self) -> int:
        return self.stop - self.start

    def __repr__(self) -> str:
        return repr(range(self.start, self.stop))


class PosLineTab(array):
    """
    A lookup table which maps any Position to an index of the last entry which
    is <= the position.  The Position must be in range(self[0], self[-1].  For
    speedup, it remembers the last lookup and tries that result next, or the
    one next to that, before going to a binary search.
    """
    last_index: int
    last_range: range

    def __new__(cls, values: Iterable[int] = []) -> Self:
        self = super().__new__(cls, 'Q', values)
        self.last_index = 0
        try: self.last_range = range(self[0], self[1])
        except IndexError:
            if len(self): self.last_range = ()
            else: raise ValueError("At least one value is required.")
        return self

    def find(self, pos: Position) -> int:
        """ Index of entry where self[index] <= pos < self[index + 1]. """
        i = self.last_index
        if pos in self.last_range:
            # Got it already.  [i] <= pos < [i+1]
            return i

        if pos < self[i]:
            # Look in earlier items.
            if not i:
                raise ValueError(f"find({pos}) below first value {self[0]}")
            i = bisect(self, pos, hi=i) - 1
            # [i] <= pos < [i+1]
        else:
            # Look in next or later items.
            i += 1
            # [i-1] <= pos
            if self[i] <= pos:
                i = bisect(self, pos, lo=i)
            else:
                pass
                # otherwise [i-1] <= pos < [i]
            i -= 1
        try: self.last_range = range(self[i], self[i+1])
        except IndexError:
            raise ValueError(f"find({pos}) >= last value {self[i]}")

        self.last_index = i
        return i


class PosTab(collections.UserList[tuple[PosRange, T]]):
    """
    Mapping from a Position to a result [T] object.  Each result has a range
    of Position's.  Ranges are increasing and non-overlapping.

    Used for:
      - PosSrcMgr's in global position space.
      - Ranges covered by a Move in Source position space.
    """
    last_index: int = 0

    @dataclasses.dataclass
    class Entry:
        pos_range: PosRange
        obj: T
        def __repr__(self) -> str:
            return f"{self.pos_range!r} : {self.obj}"

    starts: list[Position]

    def __init__(self) -> None:
        super().__init__()
        self.starts = []

    def add(self, pos_range: PosRange, obj: T,
            overlap: bool = False
            ) -> None:
        """
        Add another T value with a range of Positions.  Must be
        non-overlapping and in increasing order.  Optionally, the new range
        may steal from the current last range.
        """
        if overlap and self:
            last: Entry = self[-1]
            if pos_range.start >= last.pos_range.start:
                self[-1].pos_range.stop = pos_range.start

        assert not self or self[-1].pos_range.stop <= pos_range.start, (
            f"Add at position {pos_range.start} "
            f"< last {self[-1].pos_range.stop }")
        self.append(self.Entry(pos_range, obj))
        self.starts.append(pos_range.start)

    def find(self, pos: Position) -> T:
        """ Get the T object with given position in its range, or None. """
        if not self: return None
        entry: Entry = last_entry
        if entry:
            if entry.pos_range.start <= pos < entry.pos_range.stop:
                return entry.obj
        i : int = bisect(self.starts, pos) - 1
        if i < 0: return None
        self.last_index = i
        self.last_entry = self[i]
        if pos < entry.pos_range.stop: return entry.obj
        return None


class PosMgr:
    """ Central manager for all Positions in existence. """
    prep: Preprocessor
    # Lookup for a Position -> PosSrcMgr
    source_tab: PosTab[PosSrcMgr]
    # First Position to allocate to a new Source.
    next_pos: Position

    def __init__(self, prep: Preprocessor):
        self.prep = prep
        self.source_tab = PosTab()
        self.next_pos = 1

    def add_source(self, source: Source) -> PosSrcMgr:
        datalen = len(source.data)
        start = self.next_pos
        stop = start + datalen
        self.next_pos = stop + 1
        pos_range = PosRange(start, stop)
        mgr = PosSrcMgr(source, pos_range, self)
        self.source_tab.add(pos_range, mgr)
        return mgr

    def source(self, pos: Position) -> Source:
        """ Source which allocated this position. """
        return self.source_tab.find(pos).source


class PosSrcMgr:
    """ Manages all Positions allocated by a Source. """
    owner: PosMgr
    source: Source
    pos_range: PosRange
    line_tab: PosLineTab
    move_tab: PosTab[TokLocMove]

    def __init__(self, source: Source, pos_range: PosRange, owner: PosMgr):
        self.source = source
        self.owner = owner
        self.pos_range = pos_range

        self.line_tab = PosLineTab(source.iterlines(pos_range.start))
        self.move_tab = PosTab[TokLocMove]()
        self.add_move(TokLocMoveBase())

    def add_move(self, move: TokLocMove, offset: int = 0):
        """
        Assign Position range, from given offset from the beginning of the
        Source, to the end of the Source.
        """
        self.move_tab.add(PosRange(self.pos_range.start + offset,
                                   self.pos_range.stop),
                          move, overlap=True)


def find_mover(prep: Preprocessor, pos: Position) -> TokLocMove:
    """ Get the move object which owns the position. """
    source: Source = prep.source_pos_tab.find(pos)
    move: TokLocMove = source.move_pos_tab.find(pos)
    return move

PosMgr(None)
