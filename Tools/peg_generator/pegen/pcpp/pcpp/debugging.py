""" Module debugging.py.
Definitions useful for debugging this project.
Some of these disappear in an optimized build (with -O).
"""

from __future__ import annotations

from pcpp.common import *

# DEBUG.  Use this to conditionally execute some code
#   or include in some other expression.

__all__ = ('DEBUG break_match break_in_values break_count '
           'dbg_counter dbg_count brkg dbg_flag'.split())

DEBUG: bool = False
# Comment this out to disable tested code
DEBUG: bool = True

# Get or set global break state.  Can be set in one place and seen in an unrelated place as a breakpoint condition.
glob: bool = False
def brkg(state = None) -> bool:
    global glob
    if state is not None: glob = state
    return glob

'''
Debugger break conditions.  These can be used to set a breakpoint, or test with
an if statement, based on several parameters, which are defined below.  Various
object classes have a brk() method which tests some attributes of the object.
For example, PpTok.brk() checks a token's line number, lexer data position, and
source file name.

You can define several conditions.  One condition is defined for you from the global variables shown below.  Additional conditions can be created using the add() function, which takes keyword arguments for the various criteria.

A Condition is a function which tests various values (given by keywords) against corresponding Container objects.  The predefined containers are listed below.  A single value can be used in place of a container.  A string can also match any substring produced by str.split().  Hint: a range object is a container of ints.
'''

# Specific line numbers, if non-empty:
break_lines: Container[int] = (25, 26)

# Column number in tuple, if non-empty.
break_cols: Container[int] = ()

# File name in tuple, if non-empty.  Matches the base name of given filename.
break_files: tuple[str, ...] = ()
#break_files += ('test.c', )
#break_files += ('cond.h', )
#break_files += ('defs.h', )
#break_files += ('dummy.h', )
#break_files += ('hide.h', )
#break_files += ('macro_helpers.h', )
#break_files += ('macro_sequence_for.h', )
#break_files += ('outloc.h', )
#break_files += ('sep.h', )
#break_files += ('t.c', )
break_files += ('x1.h', )
#break_files += ('y.h', )

# lexer position in container or empty tuple
break_pos: Container[int] = ()

# Specific counter values, if non-empty:
break_counts: Container[int] = (44, 818)

# Value in tuple or empty tuple.
# Use this for some miscellaneous value other than the above.
break_others: Container[int] = ()

def add(**kwds) -> None:
    """ Add a Condition initialized by the keyword arguments. """
    conditions.append(Condition(**kwds))

conditions: list[Condition] = []

dbg_flags: dict[int, bool] = {}

def dbg_flag(n: int, val: bool | None) -> bool:
    if val is not None: dbg_flags[n] = val
    elif n not in dbg_flags: dbg_flags[n] = False
    return dbg_flags[n]

@dataclasses.dataclass
class Condition:
    """
    Define criteria for a break match.  The default values will always match
    their corresponding parameters.  All tuples match any value if they are
    empty, and they may be a single value rather than a tuple.  Any Container[int] value can be a range object.
    """
    lines: Container[int] = ()      # line numbers
    cols: Container[int] = ()       # column numbers
    files: Container[str] = ()      # file base names
    pos: Container[int] = ()        # lexer positions
    others: Container[int] = ()     # miscellaneous values

    def match(self, *, line: int = None, col: int = None, pos: int = None,
              file: str = None, **other) -> bool:
        """ True if this Condition matches all the given parameters. """
        return (break_in_values(self.lines, line)
            and break_in_values(self.cols, col)
            and break_in_values(self.pos, pos)
            and break_in_values(self.files, file and os.path.basename(file))
            and (not other
                 or all(break_other(name, value)
                        for name, value in other.items()))
            )

# The primary Condition, using the above global variables.
add(lines=break_lines, cols=break_cols, pos=break_pos, files=break_files,
    others=break_others)

#add(lines=95, files='defs.h')

T = typing.TypeVar('T')

# Helper functions to test something.
def break_in_values(values: Tuple[T, ...] | T | None,
                    value: T = None) -> bool:
    return (value is None
            or isinstance(values, typing.Container) and (not values or value in values)
            or value == values)

def break_match(**kwds) -> bool:
    """ General routine to evaluate a break condition based on various
    parameters.  Only parameters which are provided are tested.
    """
    return any(map(operator.methodcaller('match', **kwds), conditions))
    return (break_in_values(break_lines, line)
            and break_in_values(break_cols, col)
            and break_range(break_start, break_stop, line)
            and break_in_values(break_pos, pos)
            and break_in_values(break_files, file and os.path.basename(file))
            and break_in_values(break_others, other)
            )

def break_other(name: str, value: T) -> bool:
    """
    True if value matches the criterion named 'name'.  This is a Container[T]
    global variable named 'break_{name}s'.  For example, ('foo', 42) checks
    that 42 is in or == break_foos.  ('foo', None) is always true.
    """
    return break_in_values(globals()[f"break_{name}s"], value)

# Counter utilities...
# dbg_count() returns 0, 1, ..., each time it is called.
# dbg_counter() = -1 or latest value from dbg_count()
# break_count() is True if dbg_counter() is in break_counts container.
_counter = itertools.count()
_count: int = -1
def dbg_counter() -> int:
    return _count
def dbg_count() -> int:
    """ Increasing values each time it is called, starting with 0. """
    global _count
    _count = next(_counter)
    return _count
def break_count() -> bool:
    return break_in_values(break_counts, _count)

x=0
