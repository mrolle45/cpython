""" Module debug_log.
DebugLog class.
    Methods to write interesting information while preprocessing.
"""
from __future__ import annotations

import os
from operator import attrgetter

from pcpp.common import *
from pcpp.tokens import split_lines

__all__ = 'DebugLog',

break_lines = [246]            # Put line number(s) within the log file,
                            # to hit a debug breakpoint.

# ----------------------------------------------------------------------
# class DebugLog
#
# Methods to write information to the debug log file, if enabled.
# ----------------------------------------------------------------------

class DebugLog:
    def __init__(self, prep: Preprocessor):
        self.prep = prep
        self.nest = prep.nest
        self.enable: int = prep.debug
        self.loglines = []
        self.wrapper = textwrap.TextWrapper(width=80)
        #self.file = open(self.enable + '2', "wt")

    def arg(self, name: str, arg: str, ref: PpTok, **kwds) -> None:
        if not self.enable: return
        val = str(arg).strip()
        self.write(f"Arg {name} = {arg!r}", token=ref, **kwds)

    def concatenate(self, pasted: PpTok | None,
                    *opnds: PpTok, **kwds
                    ) -> None:
        if not self.enable: return
        expr = ' ## '.join(map(repr, map(attrgetter('value'), opnds)))
        if pasted:
            self.write(f"Concatenate {expr} -> {pasted.value!r}",
                       token=pasted, **kwds)
        else:
            self.write(f"Concatenate {expr} FAILED", **kwds)

    def directive(self, dir: Directive, **kwds) -> None:
        if not self.enable: return
        if not dir.name:
            self.prep.log.write("# <null directive>", token=dir.line[0],
                                **kwds)
        else:
            self.prep.log.write(f"# {dir.name} {str(dir.args).strip()}",
                                token=dir.line[0], **kwds)

    def expand(self, ref: PpTok,
               macro: Macro = None,
               expand: bool = True,
               expanding_from: list[str] = [],
               call: MacroCall = None,
               **kwds,
               ) -> None:
        """ Name of the macro appears in `ref` token.
        Macros expanding from are logged separately.
        Arguments are logged separately.
        Definition is logged separately.
        """
        if not self.enable: return
        name = ref.value
        is_func = macro and macro.is_func and "()" or ""
        if not expand:
            self.write(
                f"Macro {name}{is_func} not expanded: already expanding.",
                token=ref, **kwds
                )
        elif is_func and call is None:
            self.write(
                f"-- Macro {name}{is_func} not expanded: no argument list.",
                token=ref, **kwds
                )
        else:
            self.write(f"Expand macro {name}{is_func}", token=ref, **kwds)
            with self.prep.nest():
                if expanding_from:
                    self.write(f"While expanding {', '.join(expanding_from)}",
                               token=ref, **kwds)
                if is_func:
                    for i, name in enumerate(macro.arglist):
                        if i < macro.nargs:
                            self.arg(name, str(call.args[i]), ref)
                if macro:
                    self.write(f"Replacement = {str(macro.value)!r}",
                               token=macro.nametok, **kwds)

    def eval_ctrl(self, ref: PpTok) -> None:
        """
        An undefined macro, or beginning of a defined-macro expression to be
        evaluated.  In context of a control expression.
        """
        self.write(f"Replace {ref.value!r} in control expression.")

    def msg(self, msg: str, ref: PpTok, **kwds) -> None:
        if self.enable:
            self.write(msg, token=ref, **kwds)

    def stringize(self, name: str, string: str, ref: PpTok, **kwds) -> None:
        if not self.enable: return
        self.write(f"Stringize # {name} = {string.value}", token=ref, **kwds)

    def write(self, text: str, *, indent = 0, nest: int = 0,
              token = None,
              source: Source = None, lineno = None, colno = None,
              ):
        """ Common method for all logging.
        Logs a single message, which may contain multiple lines.
        Resulting lines are saved in self.loglines, to be written
        to the log file by self.writelog().
        """
        if not self.enable: return
        if token:
            source = token.source
            lineno = lineno or token.lineno
        else:
            lineno = lineno or 1
        if source:
            filename = source.filename
        else:
            filename = __file__
        prep = self.prep
        if prep.verbose < 2:
            filename = os.path.basename(filename)
        left: str
        try:
            ifsect = source.ifsect
            left = (prep.verbose
                      and f"{ifsect.process:d}:{ifsect.iftrigger:d}"
                          f":{ifsect.passthru:d} "
                      or "")
        except: left = ""
        if colno is None and token: colno = token.colno
        col = f":{colno}" if colno else ""
        if source: left = f"{left}{filename}:{lineno}{col}"
        more = ""
        # Leading text for right side.
        wrapper = self.wrapper
        wrapper.initial_indent = leader = (
            f"{prep.sources_active.indent(more=-1 + indent)}"
            f"{'| ' * (prep.nesting + nest)}"
            )
        wrapper.subsequent_indent = wrapper.initial_indent + '... '
        for line in text.splitlines():
            if not line.strip(): continue
            for line2 in wrapper.wrap(line):
                self.loglines.append((
                    left,
                    f"{line2}")
                    )
                #print(left, line2, file=self.file)
                left = ''
            wrapper.indent = wrapper.subsequent_indent
            more = "... "

    def writelog(self):
        """ This writes the entire output log file, if enabled.
        Called after everything has been processed.
        """
        if self.enable:
            with open(self.enable, "wt") as file:
                if self.loglines:
                    lefts, _ = zip(*self.loglines)
                    leftwidth = max(len(s) for s in lefts)
                    for left, right in self.loglines:

                        print("%-*s %s" % (leftwidth, left, right), file=file)

    def brk(self) -> bool:
        """ Put this in a breakpoint condition to break on any line
        in break_lines[].
        """
        return len(self.loglines) + 1 in break_lines
