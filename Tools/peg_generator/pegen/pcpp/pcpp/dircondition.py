""" Module dircondition.py.
Manages a directive's controlling condition in a source file.
"""

from __future__ import annotations

from enum import Enum

from pcpp.common import *
from pcpp.tokens import TokIter

class Section(abc.ABC):
    """
    Manages content of the file from an #if* directive through the matching
    #endif directive, or the entire file.

    This corresponds to the syntax element "if-section" in (C99 6.10).  It is
    composed of one or more groups:
        An if-group, introduced by the #if* which introduces the section.
            This is treated as #if 0 to introduce the Section, followed by
            #elif* corresponding to the #if*.
        Zero or more elif-groups, introduced by #elif*.

        Optional else-group, introduced by #else.  This is treated as #elif 1.
    The section ends with an #endif line.

    The entire source file is treated as a section with a single group,
        as though it were surrounded by #if 1 ... #endif.

    A group corresponds to the syntax element "group" in (C99 6.10).  It goes
    from an #if* or #el* up to, but not including,
        the next #el* or #endif, not counting nested sections.
    It has an GroupState based on the directive which introduces it.
        This can be Process, Skip, or Passthru.
    See the GroupState class below for details of processing lines in a group.

    This Section is a state machine, with transitions after each group.  The
    purpose is to find and process the first group (if any) with a Process
        condition and skip all following groups.  Groups before this first
        Process group are:
            skipped if the condition is Skip, or passed through if the
            condition is Passthru.

    For each group, as it is encountered, there are two steps:

    1. Open the group, providing the result of the controlling condition.
        This is usually Process or Skip, but it can also be Passthru, which
        indicates an undefined macro in the control expression, and the
        preprocessor logic fails to determine a presumed value.

        The group is given a GroupState of its own, which is the same as the
        directive condition, unless otherwise indicated.

        The first group is opened with the condition of the Section's #if*.

    2. Lines in the group are processed according to the group's GroupState.
        An #if* directive opens a new section, which is handled recursively.
            Processing of lines resumes after the nested section ends.

    The state machine determines the group's state based on the section's
    current state and the new group's condition.  The SectionState class
    tracks the history of the groups and provides the current group state.

    The state machine ignores all groups with a Skip condition.

    If the first group not in the Skip condition has a Process condition, then
    the group state will be Process, and all following groups will be in the
    Skip state (regardless of their directive's condition).

    If the first group not in the Skip condition has a Passthru condition, the
    group will have the Passthru state.  The state machine is then looking for
    a group with the Process condition, which means that all following groups
    will be skipped.  Until then, groups with the Passthru condition will be
    in the Passthru state.  The reasoning here is that since it is not known
    which would be the first processed group if all the macros were defined,
    they are all passed through to the output unchanged, to be handled by a
    downstream compiler.

    The state machine has special handling for Skip sections nested in Skip
    groups.  It maintains a nesting level of Skip sections, so that the lines
    can be entirely ignored, other than tracking the beginning and end of
    nested sections.  When the nesting level returns to zero, the lines are
    then processed just as with the nested case, except that new groups can be
    created.

    """

    state: SectionState         # Current state.  Changes with each new group.
    nested: int = 0             # How many Skip sections are nested.
                                #   Incremented for #(el)if*, starting with 1.
                                #   Decremented for #endif, ending with 0.
                                #   The outermost processes #el*.
    last: Directive = None      # Current group is from #else.
                                #   This prevents any further #el* directives.

    # Set if current group is Passthru, and is an #if or #elif which uses 
    # __PCPP_ALWAYS_TRUE__ or __PCPP_ALWAYS_FALSE__ in its condition.
    # The effect is to execute #define and #undef as well as pass through.
    bypass_ifpassthru: bool

    @abc.abstractmethod
    def __init__(self, state: SectionState):
        """ Start a new section.  It has no group yet. """
        self.state = state

    def nest(self) -> NestedSection:
        """ A new section which is nested in self. """
        return NestedSection(self)

    def group(self, cond: GroupState, directive: Directive) -> None:
        """ Start a new group, with given control condition. """
        if self.last and not self.nested:
            # Cannot follow #else.
            # Note, if there is a nested Skip section, it won't call this
            #   method, but rather will only bump self.nested.
            directive.prep.on_error_token(
                directive.line[0],
                f"Directive #{directive.name} may not follow #else."
                )
            return
        self.state = self.state.next(cond)
        if self.skip: self.nested = 1
        return self

    def else_group(self, directive: Directive) -> None:
        # Open the last group with #else.
        self.group(Process, directive)
        self.last = directive

    @property
    def skip(self) -> bool: return self.state.groupstate.skip

    @property
    def process(self) -> bool: return self.state.groupstate.process

    @property
    def passthru(self) -> bool: return self.state.groupstate.passthru

    @property
    def iftrigger(self) -> bool:
        """ Legacy property, meaning group is enabled or an earlier group is enabled. """
        return self.state in (SectionState.Active, SectionState.Closed)

    @TokIter.from_generator
    def parsegen(self, source: Source, toks: TokIter) -> Iterable[PpTok]:
        """ Generate all tokens in this Section, consuming the input tokens.
        Nested Sections consume their own input tokens and generate their own
        tokens.
        """
        # Loop over all the groups.
        # A new group starts after a cond directive at nest level 1,
        # unless it is an #endif, which moves to level 0.
        nest: int = 1
        while nest:
            if self.skip:
                # Get tokens up to the matching #endif or #el*.
                with source.lexer.seterrors(False):
                    for tok in toks:
                        if tok.type.dir:
                            dir = tok.dir
                            if dir.cond:
                                nest += dir.nest
                                if nest <= 1:
                                    # Get new group.
                                    next(dir(toks, self), None)
                                    break
            else:
                for tok in toks:
                    if not tok.type.dir:
                        yield tok
                        continue
                    dir = tok.dir
                    if not dir.cond:
                        res = dir(toks, self)
                        if res: yield from res
                        continue
                    # Conditional directive.

                    # If this is a nested section, we'll get its entire
                    # contents.
                    res = dir(toks, self)
                    if res: yield from res
                    if dir.nest <= 0:
                        nest += dir.handler.nest
                        break

    def __repr__(self) -> str:
        extra = '+' * (self.nested - 1)
        if self.last and not extra: extra = '*'
        return f"<{repr(self.state)} : {repr(self.state.groupstate)}{extra}>"


class GroupState(Enum):
    """ Designates how lines are handled in a group.
    Exactly one of three attributes is true: process, skip, passthru.
    The get() method returns the member corresponding to a directive condition,
        possibly inverted.
    """
    Process = dict(process=True)    # Entire group will be processed.
                                    # Nested sections will start in the Initial state.
    Skip = dict(skip=True)          # Entire group will be skipped.
                                    # Same for nested sections at any level.
    Passthru = dict(passthru=True)  # All lines not in nested section will be passed through.

    def __init__(self, attrs: dict):
        self.skip = self.process = self.passthru = False
        self.__dict__.update(attrs)

    def get(value: Any | None):
        if value is None:
            return Passthru
        else:
            return value and Process or Skip

    def __repr__(self) -> str: return self.name

for e in GroupState.__members__:
    globals()[e] = GroupState[e]
del e


class SectionState(Enum):
    """ Enum for current state of a NestedSection.
    It reflects the status (class GroupState) of the current group and
    has a class method to determine the starting state of the section,
    and a method to move to another group after a #else or #elif*.

    The state has the same three bool attributes as the GroupState class,
    which tell how lines within the current group are to be handled.

    process     Lines are processed normally.

    skip        Lines are ignored, except for limited handling of
                directives which define nested sections.

    passthru    Text lines and directives are normally written to
                the output file unchanged.  This occurs when the
                current group or an earlier group involves undefined
                macros and the preprocessor chooses not to supply
                a true or false value for the directive's condition.


    self.groupstate: GroupState = state of the current group.
    self.next(cond): SectionState = new state for a new group with given condition.
    """

    Top = Process, None

    # The normal progression of states is Open ... -> Active -> Closed ....

    # This and all preceding groups are in Skip state.
    Open = Skip, lambda cond: nextOpen[cond]

    # Current group is the only group in Process state.
    # All preceding groups are in Skip state
    # All remaining groups will be in Skip state.
    Active = Process, lambda cond: Closed

    # This and all remaining groups will be in Skip state.
    Closed = Skip, lambda cond: Closed

    # Current group is in Passthru state.
    # All remaining groups will be in either Skip or Passthru state.
    Undefined = Passthru, lambda cond: nextUndefined[cond]

    # Current group is in Skip state.
    # Some earlier group is in Passthru state.
    # All remaining groups will be in either Skip or Passthru state.
    UndefSkip = Skip, lambda cond: nextUndefined[cond]

    # Current group is in Skip state.  It had a True condition.
    # Some earlier group is in Passthru state.
    # All remaining groups will be in Skip state.
    UndefLast = Skip, lambda cond: Closed

    def __init__(self, groupstate, next):
        self.groupstate = groupstate
        self.next = next

    def __repr__(self) -> str: return self.name

for e in SectionState.__members__:
    globals()[e] = SectionState[e]
del e


''' The state machine for SectionState --

The top level FileSection is in Process state.  There are no transitions.
Nested NestedSection starts in a state depending on the state of the
    enclosing group, as follows:

    outer state     ->      inner state

    Process                 Open
    Skip                    Closed
    Passthru                Undefined

The state machine advances with the condition for each new group,
    including the first.

The following mapping shows the transition from each state to the next state,
    as a function of the condition.  If the transition is a SectionState,
    that will be the new state, regardless of the condition.
'''
SectStateMachine: Mapping[SectionState,
                          SectionState | Mapping[GroupState, SectionState]
                          ] = {
    Open:       {Process: Active, Skip: Open,      Passthru: Undefined, },    
    Active:     Closed,   
    Closed:     Closed,   
    Undefined:  {Process: Closed, Skip: UndefSkip, Passthru: Undefined, },
    UndefSkip:  {Process: Closed, Skip: UndefSkip, Passthru: Undefined, },
    UndefLast:  Closed,
    }

for sect, next in SectStateMachine.items():
    # Initialize sect.next callable.
    def init(next):
        if type(next) is SectionState:
            sect.next = lambda *args: next
        else:
            sect.next = lambda cond: next[cond]
    init(next)

del sect, next, SectStateMachine


class NestedSection(Section):
    """
    A Section which is contained in another Section (possibly the whole file).
    """
    top: ClassVar[bool] = False
    # True for #else group, to prevent more groups.
    last: bool = False

    def __init__(self, nested: Section, *,
                 _lookup: Mapping[GroupState, SectionState] = {
                        Process:      Open,
                        Skip:         Closed,
                        Passthru:     Undefined,
                        }
                 ):
        super().__init__(_lookup[nested.state.groupstate])


class FileSection(Section):
    """
    A Section representing the entire source file.  It will be, and will
    remain, in the Active state.  It has only one Group, and cannot contain
    any #el* lines.
    """
    top: ClassVar[bool] = True

    def __init__(self):
        super().__init__(SectionState.Top)


