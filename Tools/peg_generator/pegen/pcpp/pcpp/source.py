""" Module source.py
Source class, which handles a single input file.
"""

from __future__ import annotations

import codecs
from dataclasses import field
from enum import Enum, auto
import time
import traceback

from pcpp.common import *
from pcpp.directive import (Directive, Action, OutputDirective)
from pcpp.dircondition import (FileSection)
from pcpp.tokens import Tokens, TokIter, reduce_ws
from pcpp.position import (PosSrcMgr, PresumeMover, PosRange,
                           PosTab, PosLineTab)
from pcpp.writer import (OutLoc, OutLoc, OutPosFlag, OutPosChange,
                         OutPosMove, OutPosEnter)

clock = time.perf_counter

class Source:
    """
    Performs Translation on a single source file.  Generates tokens,
    including from '#include'ed files.
    """

    parent: Source      # The Source #include'ing self, if any.
    file: SourceFile    # The file being translated.
    line: Tokens        # Input line being processed.  May change during
                        # directive processing.

    # Use this to determine if this file is included only once.  Set only on
    # the first Source for the same file, then removed after parsing.
    once: IncludeOnce = None

    # Deciding whether to include only once.
    once_pend: bool = False

    # Handles output locations for the Writer.                    
    outloc: OutLoc = None     

    # Shifts line and filename to reflect latest #line directive, if any.
    # This is an object for the entire Source range, or one for the most
    # recent #line directive.
    move: PresumeMover
    # Table of all PresumeMover objects with their ranges of global positions.
    move_pos_tab: PosTab[PresumeMover]

    # Global positions of data.
    pos_range: PosRage

    # Table of physical line global positions.
    line_pos_tab: SSrcPosTab = None

    in_macro: MacroCall = None  # Set while scanning macro argument list.
    exp_macro: MacroCall = None # Set while expanding the macro.
    nesting: int = 0    # Used for debug log indentation.

    def __init__(self, file: SourceFile) -> None:
        self.file = file
        self.prep = prep = file.prep
        self.parent = prep.currsource
        self.filename = file.filename
        self.move_pos_tab = PosTab()


        self.hdr_name = file.filename
        if file.once_pend:
            self.once_pend = True
            self.once = file.once

    def set_move(self, movetok: MoveTok = None):
        lexer = self.lexer
        move = lexer.move = PresumeMover(self, movetok)
        self.positions.add_move(move, lexer.lex.lexpos)

    def input(self, input: str) -> None:
        """ Sets the input data and initializes the lexer to tokenize it. """
        prep = self.prep
        lex = self.lexer

        if '\t' in input:
            tabstop = prep.tabstop
            if tabstop and not prep.clang:
                input = input.expandtabs(tabstop)

        self.lexer.input(input, self)
        self.positions = prep.positions.add_source(self)

    def tokens(self, input: str) -> TokIter:
        """ Tokenize the input. """
        return self.lexer.tokens()

    newline_re: Pattern = re.compile(r'\n')

    def iterlines(self, base: Position = 0) -> Iterator[Position]:
        """
        Generates datapos for each newline character or elided line splice in
        the data.  Optional base Position added to each datapos.
        """
        data: str = self.lexer.data
        data_len: int = len(data)
        m: Match
        yield base
        repls = filter(lambda r: r.spliced, self.lexer.repls)
        repl = next(repls, 0)
        repl_pos = repl and repl.repl_pos
        for m in self.newline_re.finditer(data):
            pos = m.start() + base
            # Look for an earlier splice replacement.
            while 0 < repl_pos < pos:
                yield repl.repl_pos + base
                repl = next(repls, 0)
                repl_pos = repl and repl.repl_pos
            yield pos

    @TokIter.from_generator
    def parsegen(self, *, dir: PpTok = None) -> TokIter:
        """
        Parse an input string from startup script or source file (main or
        included).  For a source file, given the #include directive's token.
        Result recursively includes files #include'd by this Source.
        """

        prep: PreProcessor = self.prep
        self.TokType = prep.TokType
        file = self.file
        input = file.data

        lex = self.lexer = prep.lexer.clone()

        if not input:
            return

        self.input(input)

        if self.parent:
            include_time = FileInclusionTime(
                self.parent.file.abspath, file.filename, file.abspath,
                prep.sources_active.depth)
            prep.include_times.append(include_time)
            include_time.begin()

        # Delegate this to the Macros.
        toks: TokIter = prep.macros.expand(
            self.parsegen_after_directives(prep, input),
            top=self)
        # Bundle this in an iter token so that it doesn't go through macro
        # expansion in upstream sources.

        yield lex.make_passthru(toks)
        if self.once_pend:
            del self.once_pend
            self.once.finish()

        # This occurs after all the above tokens have been consumed.
        if self.parent:
            include_time.end()

    @TokIter.from_generator
    def parsegen_after_directives(self, prep: PreProcessor, input: str,
                                  ) -> Iterator[PpTok]:
        """
        Parse an input string.  Process directives.  Generate remaining PpTok
        objects along with anything generated by directives.
        """

        # True if auto pragma once still a possibility for this #include
        #   (it may have been disabled by the subclass constructor).
        auto_pragma_once_possible = prep.auto_pragma_once_enabled
        # (MACRO, 0) means #ifndef MACRO or #if !defined(MACRO) seen,
        # (MACRO,1) means #define MACRO seen.
        prep.on_potential_include_guard(None)

        tokens = self.tokens(input)
        for tok in tokens:
            all_whitespace = True
            #self.lineno = lineno = tok.lineno

            if tok.type.dir:
                # Preprocessor directive      

                dir = tok.dir

                res = dir(tokens)
                if res:
                    yield from res
                continue

            if not all_whitespace:
                self.no_guard()

            # Normal text
            yield tok

        # End of tokens loop.

        # End of the source

    @contextlib.contextmanager
    def inmacro(self, call: MacroCall) -> None:
        """
        Declare that the consumer of the next tokens is in or preceding a
        function macro argument list, during the context.  Some lexing,
        notably certain directives, are handled differently.
        """
        old = self.in_macro
        self.in_macro = call
        try: yield
        finally:
            if not old:
                del self.in_macro
            else:
                self.in_macro = old

    @contextlib.contextmanager
    def expanding(self, call: MacroCall) -> None:
        """
        Declare that the given macro call is being expanded.  This will be
        visible while emitting all expansion tokens, except in a nested macro
        call.
        """
        old = self.exp_macro
        self.exp_macro = call
        try: yield
        finally:
            if not old:
                del self.exp_macro
            else:
                self.exp_macro = old

    def define_guard(self, tok: PpTok) -> None:
        """ Found an actual include guard. """
        assert self.once is not None
        self.once.define_guard()
        self.prep.log.write(
            f"Determined that this file is entirely "
            f"wrapped "
            f"in an include guard.\n"
            f"Auto-applying #pragma once.",
            token=tok)

    def no_guard(self) -> None:
        """ Turn off search for an include guard. """
        assert self.once is not None
        self.once.no_guard()

    def __repr__(self) -> str:
        return repr(os.path.basename(self.filename))


class _IndirectToMacroHook(object):
    def __init__(self, p):
        self.prep = p.prep
        self.partial_expansion = False
    def __contains__(self, key):
        #return key != 'foo'
        return True
    def __getitem__(self, key):
        if key.startswith('defined('):
            self.partial_expansion = True
            return 0
        repl = self.prep.on_unknown_macro_in_expr(key)
        if repl is None:
            self.partial_expansion = True
            return key
        return repl


class _IndirectToMacroFunctionHook(object):
    def __init__(self, p):
        self.prep = p.prep
        self.partial_expansion = False
    def __contains__(self, key):
        return True
    def __getitem__(self, key):
        repl = self.prep.on_unknown_macro_function_in_expr(key)
        if repl is None:
            self.partial_expansion = True
            return key
        return repl


class SourceFile:
    """
    A unique entry in the filesystem, keyed by absolute path.  The file can be
    translated several times, and this object persists even after any
    translation is completed.
    """

    prep: Preprocessor

    abspath: str                # Absolute path which uniquely identifies self.
                                # self.prep.files[self.abspath] is self.
    basename: str               # Base name (without directories) of the file.

    filename: str               # Name in #include directive
                                # or command line input.
    rewritten: str              # abspath without prefix in prep.rewrite_paths.
    hdrname: str                # Name shown when writing a #line directive,
                                # unless superceded by a #line directive in the
                                # source file.
    pathdir: str              # The search path directory where file is found.
                                # '' if not found.
    once: IncludeOnce = None    # Whether file can be included only once, or if
                                # the status is yet to be determined.
                                # None or false value means no restrictions.
    once_pend: bool = False     # Deciding whether to include only once.
    data: str                   # Entire contents of the file.

    def __init__(self, prep: Preprocessor, abspath: str,
                 pathdir: str | None, filename: str, *,
                 file: IOBase = None, dirname:str = None, **kwds):
        self.prep = prep
        self.abspath = abspath
        self.filename = filename
        self.pathdir = pathdir
        self.dirname = dirname or os.path.dirname(abspath)
        if file is None:
            # Name comes from an #include.
            if pathdir is not None:
                with open(abspath, encoding=prep.input_encoding or 'utf-8'
                          ) as f:
                    data = f.read()
            else:
                data = ''
        else:
            # File opened from the command line, with default 'utf-8'. Reopen
            # the file if we want a different encoding.
            enc = prep.input_encoding
            if enc and codecs.lookup(enc) is not codecs.lookup('utf-8'):
                file = open(file.name, 'r', encoding=enc)
            data = file.read()
            if isinstance(data, bytes):
                data = data.decode()

        rewritten = filename
        if abspath:
            rewritten_source = abspath
            for rewrite in prep.rewrite_paths:
                temp = re.sub(rewrite[0], rewrite[1], rewritten_source)
                if temp != abspath:
                    # rewrite[0] was found.
                    rewritten = temp
                    break
        self.rewritten = rewritten

        # With clang,
        #   The top file is only the basename.
        #   Other file is the filename with fix path sep.
        if prep.clang:
            if prep.sources_active.depth > 1:
                if pathdir:
                    self.filename = os.path.join(self.pathdir, self.filename)


        self.data = data
        self.once = IncludeOnce(self)
        self.once_pend = True

    @classmethod
    def open(cls, filename: str,
             qfile: SourceFile | None,  # Look here first, if given.
             search_path: list[str],    # Look in these directories.
             prep: Preprocessor, **kwds) -> Self:
        """
        Find existing SourceFile, or create a new one and store it in the prep.
        """

        abspath, pathdir = cls.search(filename, qfile, search_path)
        try:
            return prep.files[abspath]
        except: pass
        file = cls(prep, abspath, pathdir, filename, **kwds)
        prep.files[abspath] = file
        return file

    @classmethod
    def openfile(cls, file: IOBase, prep: Preprocessor, **kwds) -> Self:
        """
        Find existing SourceFile, or create a new one and store it in the prep.
        Given an existing open file.
        """
        name = file.name
        if name.startswith('<'):
            abspath = name
        else:
            abspath = os.path.abspath(name)
        pathdir = '.'
        try:
            return prep.files[abspath]
        except: pass
        file = cls(prep, abspath, pathdir, file.name, file=file, **kwds)
        prep.files[abspath] = file
        return file

    @classmethod
    def search(cls, filename: str, 
               qfile: SourceFile | None,    # Look here first, if given.
               search_path: list[str]       # Look in these directories.
               ) -> tuple[str, str]:
        """
        Find existing file and return its abspath and the search dir (if any)
        where it was found.  That is, (pathdir) / (filename) names the file.
        """
        def found(path = None) -> tuple[str, str]:
            iname = os.path.join(path or '', filename)
            if os.path.exists(iname):
               return os.path.abspath(iname), path
            return "", ""

        if os.path.isabs(filename):
            return found('')

        if qfile:
            # Search this file's directory, but use its pathdir if found.
            abspath, pathdir = found(qfile.dirname)
            if abspath:
                return abspath, os.path.dirname(qfile.filename)

        for pathdir in search_path:
            abspath, pathdir = found(pathdir)
            if abspath:
                return abspath, pathdir
        return os.path.abspath(filename), ""

    def __repr__(self) -> str:
        return f"<File \"{self.filename}\""


class IncludeOnce:
    """ Determines if a Source has an include guard, or a #pragma once.
    This applies to the full filename of the Source, hence it affects all
    #include's of that name.  When bool(self) is true, then further #include's
    are totally skipped.

    The first time the name is seen in a file, an IncludeOnce object is stored
    in file.once, with a false value.  As it is being processed by a
    Source, if it meets the once-only requirements, it is set to a true value.
    If this doesn't occur, file.once is deleted.

    On subsequent includes, file.once is checked; if it exists and is true, then the file
    is skipped, otherwise it is processed normally.
    """
    file: SourceFile            # The file being analyzed.
    on: bool = False            # Set True by finding either a #pragma once
                                # or an actual include guard
    pending: bool = False       # Include guard is possible
    guard: str = None           # Name of potential or actual include guard.

    ''' The life cycle:
                                on      pending     guard   file.once

    1. Initially,               False   True        None    self
    2. no_guard() if pending            False
    3. set_guard(g) if pending          False       g
    4. define_guard() if guard  True
    5. pragma()                 True    False       None
    6. finish() if not on                                   deleted
    '''

    def __init__(self, file: SourceFile):
        self.file = file
        self.pending = True

    def __bool__ (self) -> bool: return self.on

    def finish(self) -> None:
        """ Clean up after processing the file. """
        if not self.on:
            del self.file.once
            del self.file.once_pend
        elif self.pending:
            del self.pending

    def define_guard(self) -> None:
        """ Handle #define directive naming self.guard. """
        self.on = True

    def set_guard(self, guard: str) -> None:
        """ Found a potential include guard. """
        self.guard = guard

    def no_guard(self) -> None:
        """
        Include guard no longer possible.  Non-whitespace or a directive with
        effect seen in file.
        """

        try: del self.pending        # Set = class variable = False
        except: pass

    def pragma(self) -> None:
        """ "#pragma once" found in file. """
        self.on = True
        try: del self.guard          # Remove guard name
        except: pass


# ------------------------------------------------------------------
# File inclusion timings
#
# Useful for figuring out how long a sequence of preprocessor inclusions 
# actually is taking.
# ------------------------------------------------------------------

class FileInclusionTime(object):
    """The seconds taken to #include another file"""
    def __init__(self,including_path,included_path,included_abspath,depth):
        self.including_path = including_path
        self.included_path = included_path
        self.included_abspath = included_abspath
        self.depth = depth
        self.elapsed = 0.0

    def begin(self) -> None:
        self.start_time = clock()

    def end(self) -> None:
        self.end_time = clock()
        self.elapsed = self.end_time - self.start_time
