""" Module common.py.
Various things generally useful to many other modules. 
"""

from __future__ import annotations

import abc
import copy
import collections
import contextlib
import dataclasses
from dataclasses import dataclass
import enum
import functools
import inspect
import io
import itertools
import operator
import os
import re
import sys
import textwrap
import traceback
import typing

import pcpp
from pcpp.debugging import *

T = typing.TypeVar('T')
IndexType = typing.TypeVar('IndexType', bound=int)

def show_quoted(s: str) -> str:
    """ Given string surrounded by fancy quote chars, suitable for output. """
    return f'‘{s}’'

@dataclasses.dataclass
class ValueRef(typing.Generic[T]):
    """
    Reference to a stored value, in self.data.  This allows a value to be seen
    in several places and changes will be seen everywhere.
    """
    data: T

    def __bool__(self) -> bool:
        """ Test for true stored value. """
        return bool(self.data)

    @contextlib.contextmanager
    def tempvalue(self, value: T) -> ContextManager[None]:
        save = self.data
        self.data = value
        yield
        self.data = save

    def __str__(self) -> bool:
        return str(self.data)

    def __repr__(self) -> str:
        return f"<Ref {self.data!r}>"


class Stack(collections.UserList[T]):
    """
    Maintains a stack of T objects.  It's a list with extra frills.
    """
    def __init__(self, *elements, lead: str = ''):
        super().__init__(*elements)
        self.lead = lead

    def indent(self, more: int = 0) -> str:
        """
        String useful as a prefix to other information, based on current depth.
        """
        return self.lead * (len(self) + more)

    @property
    def depth(self) -> int:
        return len(self)

    def top(self, default: T = None) -> T | None:
        return self and self[-1] or default

    @contextlib.contextmanager
    def nest(self, item: T = None) -> ContextManager[None]:
        """
        Push given item onto the stack for the duration of the context.
        """
        self.append(item)
        try: yield
        finally: self.pop()


class Offset(int, typing.Generic[T]):
    """ An integer that is used as an index into a Sequence[T] target. """
    pass


class RangeTuple(typing.NamedTuple):
    """
    Standin for a range object with extra frills, but no step.  It's a range
    of indices of type IndexType, into a container.

    Construct similarly to range(), i.e., Range(stop) or Range(start, stop).
    Also from Range(start, len=len).

    Add or subtract an offset to translate start and stop.

    Used as:
      - val: Sequence[T] = range.get_from(seq: Sequence[T])
      - val: Sequence[T] = get_range(seq: Sequence[T], range)
      - for i in range:
            val: T = seq[i]

    Can be compared as tuples (start, stop).
    """
    start: IndexType
    stop: IndexType

    def __add__(self, rhs: IndexType) -> Self:
        return RangeTuple(self.start + rhs, self .stop + rhs)

    def __sub__(self, rhs: IndexType) -> Self:
        return RangeTuple(self.start - rhs, self.stop - rhs)

    def __contains__(self, i: int) -> bool:
        return self.start <= i < self.stop

    @property
    def range(self) -> range:
        return range(self.start, self.stop)

    def extend(self, delta: IndexType) -> Self:
        """ Extend self.stop by given amount """
        return RangeTuple(self.start, self.stop + delta)

    def get_from(self, seq: Sequence[T]) -> Sequence[T]:
        return seq[self.start : self.stop]

    def __repr__(self) -> str:
        r = self.range
        res = f'[{r.start or ""}:{r.stop or ""}'
        if r.step != 1: res = f'{res}:{r.step}'
        return f'{res}]'

def Range(*args: IndexType, len: int = None) -> RangeTuple:
        from builtins import len as l
        if len is None:
            assert 1 <= l(args) <= 2, f"Range() got {l(args)} arguments"
            if l(args) == 1:
                start, stop = 0, args[0]
            else:
                start, stop = args
        else:
            assert 1 == l(args), f"Range() got {l(args)} arguments"
            start = args[0]
            start, stop = start, start + len

        self = RangeTuple(start, stop)
        return self

def get_range(seq: Sequence[T], range: Range[T, IndexType]
              ) -> Sequence[T]:
    return seq[range.start : range.stop]

RangeType = RangeTuple[IndexType]

class Ranges(typing.Generic[IndexType]):
    """
    Container of disjoint and increasing Range[IndexType] objects, acting as a
    container of all their individual indices.

    May be constructed from an iterable of Range's, and/or they may be
    appended separately.  Caller must ensure that the increasing ordering is
    maintained.  This enables a bisect method to determine whether an index is
    in one of the ranges.
    """
    ranges: list[RangeType] = []

    def __init__(self, *ranges: RangeType):
        self.ranges = list(ranges)

    def __contains__(self, index: IndexType,
                     _key=operator.attrgetter('start'),
                     ) -> bool:
        i: int = bisect(self.ranges, (index + 1,))
        #i: int = bisect_key(self.ranges, index, key=_key)
        return i and index in self.ranges[i - 1]


class RangeMap(Ranges[RangeType], typing.Generic[T, IndexType]):
    """
    A mapping from an index to a result value, where each T is
    associated with a Range.  The lookup of an index returns the T associated
    with the Range which contains the index.  The Ranges are in increasing
    order and non-overlapping.

    May be constructed from an iterable of (range, value) pairs, and/or they
    may be appended separately.  An appended Range may steal indices from the
    end of the last Range already stored.
    """
    Item = tuple[RangeType, T]

    items: list[Item] = []

    def __init__(self, *items: Item):
        if items:
            self.items = list(items)
            ranges, values = zip(*items)

    def __getitem__(self, index: IndexType,
                    ) -> T:
        items = self.items

        i: int = bisect(items, ((index + 1,),))
        if not i:
            raise ValueError(index)
        range, value = self.items[i - 1]
        if index not in range:
            raise ValueError(index)
        return value

    def get(self, index: IndexType, default = None) -> T:
        """ Same as self[index] with a default value if index not found. """
        items = self.items

        i: int = bisect(items, ((index + 1,),))
        if not i:
            return default
        range, value = self.items[i - 1]
        if index not in range:
            return default
        return value

    def append(self, range: RangeType, value: T):
        items = self.items
        assert not items or items[-1][0].stop <= range.start
        items.append((range, value))

    def steal(self, start: IndexType, value: T):
        """
        Split the last range at given start and attach given value to the
        second part.
        """
        last, lastvalue = self.items[-1]
        assert(last.stop >= start)
        self.items[-1] = (Range(last.start, start), lastvalue)
        self.items.append((Range(start, last.stop), value))


try: from itertools import pairwise
except ImportError:
    # Version 3.9 or earlier.  Copy the definition from the library doc.
    def pairwise(iterable):
        # pairwise('ABCDEFG') -> AB BC CD DE EF FG
        iterator = iter(iterable)
        a = next(iterator, None)
        for b in iterator:
            yield a, b
            a = b

# bisect_key().  Same as bisect.bisect_right(), except that with Python 3.9 or
# earlier, this is copied below from the 3.10 library module, with the key=
# argument being required.
if sys.version_info < (3, 10):
    # Version 3.9 or earlier.  Copy the function from library
    # bisect.bisect_right() function.  This does not use the faster C version.

    def bisect_key(a, x, lo=0, hi=None, *, key: Callable = None):
        """
        Return the index where to insert item x in list a, assuming a is
        sorted.

        The return value i is such that all e in a[:i] have e <= x, and all e
        in a[i:] have e > x.  So if x already appears in the list, a.insert(i,
        x) will insert just after the rightmost x already there.

        Optional args lo (default 0) and hi (default len(a)) bound the slice
        of a to be searched.
        """

        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        # Note, the comparison uses "<" to match the
        # __lt__() logic in list.sort() and in heapq.
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
        return lo
else:
    # Version 3.10 or later.
    from bisect import bisect_right as bisect_key

from bisect import bisect

in_production = 0  # Set to 0 if editing pcpp implementation!
