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
    def tempvalue(self, value: T) -> None:
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
    def nest(self, item: T = None) -> Iterable:
        """ Push given item onto the stack for the duration of the context.
        Call without an item to simply maintain a nesting level.
        """
        self.append(item)
        try: yield
        finally: self.pop()


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

in_production = 0  # Set to 0 if editing pcpp implementation!
