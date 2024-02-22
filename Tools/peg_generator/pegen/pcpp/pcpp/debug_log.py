""" Module debug_log.
DebugLog class.
    Methods to write interesting information while preprocessing.
"""
from __future__ import annotations

import os

__all__ = 'DebugLog',

break_lines = []            # Put a line number in the log file, to hit a debug breakpoint.

# ----------------------------------------------------------------------
# class DebugLog
#
# Methods to write information to the debug log file, if enabled.
# ----------------------------------------------------------------------

class DebugLog:
    def __init__(self, prep: Preprocessor):
        self.prep = prep
        self.nest = prep.nest
        self.enable: int = prep.args.debug
        self.loglines = []

    def expand(self, macro: Macro, expand: bool,
                ref: LexToken,
                expanding_from: list[str],
                args: Tokens = None,
                **kwds,
                ) -> None:
        """ Name of the macro appears in `ref` token.
        Arguments are logged separtely
        """
        if not self.enable: return
        if isinstance(macro, str):
            name = macro
            is_func = ''
        else:
            name = macro.name
            is_func = macro.is_func and "()" or ""
        if not expand:
            self.write(f"Macro {name}{is_func} not expanded: already expanding.", token=ref, **kwds)
        elif is_func and args is None:
            self.write(f"Macro {name}{is_func} not expanded: no argument list.", token=ref, **kwds)
        else:
            self.write(f"Expand macro {name}{is_func}", token=ref, **kwds)
            if expanding_from:
                with self.prep.nest():
                    self.write(f"While expanding {', '.join(expanding_from)}", token=ref, **kwds)

    def result(self, tokens: Tokens, ref: LexToken) -> None:
        text: str = self.prep.showtoks(tokens)
        lines = text.splitlines()
        with self.prep.nest():
            if len(lines) > 1:
                self.write("Expands to:", token=ref)
                for line in lines:
                    self.write(f"... {line}", token=ref)
            else:
                self.write(f"Expands to: {text!r}", token=ref)

    def arg(self, name: str, arg: str, ref: PpTok, **kwds) -> None:
        if not self.enable: return
        val = self.prep.showtoks(arg).strip()
        self.write(f"Arg {name} = {val!r}", token=ref, **kwds)

    def stringize(self, name: str, string: str, ref: LexToken) -> None:
        if not self.enable: return
        self.write(f"Stringize # {name} = {string}", token=ref)

    def concatenate(self, left: LexToken, right: LexToken) -> None:
        if not self.enable: return
        self.write(f"Concatenate {left.value!r} ## {right.value!r} = {left.value + right.value!r}", token=left)

    def msg(self, mst: str, ref: LexToken, **kwds) -> None:
        if self.enable:
            self.write(mst, token=ref, **kwds)

    def write(self, text: str, indent = 0, nest: int = 0,
                   token = None,
                   source: Source = None, lineno = None, colno = None,
                   contlocs: bool = True,       # Show location for continued lines.
                   ):
        if not self.enable: return
        if token:
            source = token.source
            lineno = lineno or token.lineno
        if source:
            filename = source.filename
        else:
            filename = __file__
        if self.prep.verbose < 2:
            filename = os.path.basename(filename)
        try: ifstate = source.ifstate
        except: pass
        #ifstate = source.ifstate
        leader = (self.prep.verbose
                  and f"{ifstate.enable:d}:{ifstate.iftrigger:d}:{ifstate.ifpassthru:d} "
                  or "")
        for i, line in enumerate(text.splitlines(), lineno):
            if not line.strip(): continue
            if colno is None and token: colno = token.colno
            col = f":{colno}" if colno and i == lineno else ""
            left: str = f"{leader}{filename}:{i}{col}"
            if not contlocs and i > lineno:
                left = ''
            self.loglines.append((
                left,
                f"{'| ' * (self.prep.nesting + nest)}{' ' * indent}"
                f"{i > lineno and '... ' or ''}{line}")
                )

    def writelog(self):
        """ This writes the output log file, if enabled. """
        if self.enable:
            with open("pcpp_debug.log", "wt") as file:
                if self.loglines:
                    lefts, _ = zip(*self.loglines)
                    leftwidth = max(len(s) for s in lefts)
                    for left, right in self.loglines:
                        print("%-*s %s" % (leftwidth, left, right), file=file)

    def brk(self) -> bool:
        """ Put this in a breakpoint condition to break on any line in break_lines[]. """
        return len(self.loglines) + 1 in self.break_lines

