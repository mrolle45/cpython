#!/usr/bin/python
# Python C99 conforming preprocessor useful for generating single include files
# (C) 2017-2021 Niall Douglas http://www.nedproductions.biz/
# and (C) 2007-2017 David Beazley http://www.dabeaz.com/
# Started: Feb 2017
#
# This C preprocessor was originally written by David Beazley and the
# original can be found at https://github.com/dabeaz/ply/blob/master/ply/cpp.py
# This edition substantially improves on standards conforming output,
# getting quite close to what clang or GCC outputs.

from __future__ import annotations
from __future__ import generators, print_function, absolute_import, division

import sys, os, re, codecs, time, copy, traceback
import contextlib
from dataclasses import dataclass, field
import textwrap
from typing import Iterator
if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

from pcpp.common import *
from pcpp.parser import (PreprocessorHooks)
from pcpp.lexer import (default_lexer, PpLex, OutPosFlag)
from pcpp.tokens import PpTok, Tokens, TokIter, Hide, HideNames, HideDict
from pcpp.evaluator import Evaluator
from pcpp.macros import Macro, Macros
from pcpp.position import (PosMgr, PosRange, PosTab, PosLineTab)
from pcpp.source import Source, SourceFile
from pcpp.writer import OutPosFlag, OutPosEnter, OutPosLeave
from pcpp.debug_log import DebugLog
FILE_TYPES = io.IOBase

__all__ = ['Preprocessor', 'PreprocessorHooks', 'Evaluator']

''' Overall plan:

The Preprocessor class specifies a single object (often called `prep` in the
code) which reads one source file and any other files #include'd by them, and
writes an output file.  This output file can then be compiled to produce the
same final result as if the original source file were compiled normally.
Typically, it has the same name as the input file but with a ".i" extension,
and so is called the ".i file".

It is analogous to running the command "gcc <input> -E > <output>".  In fact,
PCPP can be tested by running with a "-gcc" command line switch and comparing
the .i file with the <output> produced by gcc.

PCPP follows the ISO C or C++ Standard for preprocessing, which consists of
translation phases 1 through 4.  The result is a sequence of
`preprocessing-token`s .  The PpTok class is used for all such tokens.

The standards are written with the idea of being the first part of a complete
compiler, which processes the output of phase 4.  A standalone preprocessor,
instead, takes the PpTok's of phase 4 and writes a file which, when later
compiled, produces the same PpTok's from phase 4.

PCCP employs a series of iterators, each one being an input to a generator
function for the next one.  The objects generated are of class PpTok.

Locations.

    A Position object specifies where some text in the source file comes from.

    * source = Source object.
    * colno = the column where the text appears in the source file.  The column
      number starts at 1 in the line, but resets back to 1 after a line splice.
    * lineno = line number of the logical line.
    * phys_lineno = line number of the text.  Will be greater if line splices
      are involved.
    * pos = a special OutPosFlag object used to write line directives in GCC
      emulation.  None if being used for a token.

    It is also used to specify a presumed line number and file name, when
    writing a line directive.  In this case, colno and phys_lineno are None.

Tokens.

    A token is a PpTok object representing a preprocessing-token.  It has this
    information:

    * token.value.  The text in the data.
    * Position information.  These are properties delegated to token.pos, a
      Position object (see above):
        * token.lineno.
        * token.colno.
        * token.phys_lineno.
        * token.source.
    * token.type.  A TokType object.  It has several interesting properties
        which can be tested.
    * token.spacing.  Indicator of space characters, other than newlines,
        preceding the text in the data.  A true value if there are such
        characters, and a false value otherwise.  Note that whitespace at the
        end of a line is never used, so there is no need to indicate following
        spacing.  This value can be altered later in preprocessing.
    * token.hide.  This a set of macro names which were expanded to produce
        the token.  This is empty when coming directly from the lexer, but can
        be set during macro expansion.  If this is non-empty, then the location
        information won't refer to the present source file, but rather to a
        macro definition.

    Other tokens are used to convey special information.

    * Position token.  This has a token.pos member which is a OutLoc object.
      It indicates a change in presumed line number and file name.

1. Lexing.
    For each main or included source file, a Source object reads that file.  It
    has a PpLex (a.k.a. lexer).  It provides an Iterator[PpTok] which generates
    tokens as they are found in the data.

    It implements translation phases 1, 2 and 3 and generates the tokens
    described by phase 3.

    Translation phase 1: trigraphs and non-basic characters.

        The nine trigraphs are replaced by their equivalent characters in the
        data.  Trigraphs were removed from C++14 and onward, but still
        implemented by the lexer in certain cases.

        Characters that are not in the basic character set are replaced by
        universal character names, which have the form of escape sequences.
        The characters "$", "@", and "`" are left as-is in certain cases.

    Translation phase 2: line splices.

        Any "\\"-"\n" pair is removed.  Note that a "\\" could result from a
        "??/" trigraph, so that "??/\n" is entirely removed from the data.

    The lexer keeps track of these changes, and it can reconstruct the original
    spelling and location of a token.  It then parses the resulting data to
    generate tokens.

    Whitespace.

        Whitespace is generally not significant to the preprocessor.  But it
        does matter in these cases:

        * Delimiting preprocessor directives.
        * Distinguishing object-like and function-like macro definitions.
        * Within a function macro argument which is stringified with the '#'
          operator.

        Whitespace consists of whitespace characters and comments, but not
        within quoted strings.  In GCC emulation, comments are not whitespace
        but rather tokens in their own right.  An unmatched quoting character
        before reaching a newline is treated as an error up that point.
        However, a raw string, beginning with an "R" prefix, may include
        multiple lines.

    Translation phase 3: tokenizing.

        The lexer takes the data, as modified above, and parses it into tokens
        and whitespace.  Each token is a PpTok object and corresponds to the
        entity "preprocessing-token" in the ISO Standard.  Any character that
        cannot be one of the other specified types of token is a token by
        itself.

Lines.

    In the data, a line, sometimes called a logical line, is the result of
    splitting the data by newline characters, including the terminating
    newline.  According to the Standards, a newline is required at the end of a
    non-empty source file, and the Source object supplies a missing newline,
    with a warning.

    It can come from multiple physical source lines when line splices are
    involved.

    The line number is the line number of the first physical source line.

2. Translation phase 4: preprocessing.

    This is performed by the Source object in two stages, each producing an
    Iterator[PpTok].

    1. Directives.

        The Source takes tokens from its lexer and uses groupby() to break it
        up into groups of tokens with a common token.lineno.

        If the first token in a group is a "#", then the entire group is
        processed as a directive.  Then the group is replaced by the iterator
        (if any) resulting from the directive.

        In either case, the contents of the group are passed on to the output
        iterator.

        A directive is processed according to the directive name (the next
        token after the "#").  If there is no name, then the directive is
        ignored.

        * #include.  The result of preprocessing the entire included source
          file (translation phases 1 through 4) is the result of the directive.
          If the file contains a "#pragma once" directive anywhere, or some
          obvious #include guard, then any subsequent #include of the same
          file, even recursively within other includes, will be completely
          ignored.

          The tokens are preceded and followed by location tokens.  The first
          denotes the Source for the included file and line number 1.  The
          second denotes the present Source and the line number following the
          #include directive.

        * #define and #undef.  These modify the prep.macros dictionary.  The
          change will be seen during the macro expansion pass, starting with
          the remaining tokens given to the output iterator.

        * #if and other conditional-inclusion directives.  They divide the
          source into sections (corresponding to the entity "if-section" in the
          Standard).  A section is contains one or more groups (corresponding
          to "group" in the Standard), each preceded by one of the conditional
          directives.

          A group contains text lines and directives other than
          conditional-inclusions.  Recursively, any group may contain nested
          sections.

          A group has a "skip" status.  When this is true, the entire group is
          ignored, other than to find the end of the group.  When this status
          is false, text and directive lines are processed normally.

        * Certain other directives, like an unrecognized #pragma, can produce
          lines, which will generally be part of the text and not macro
          replaced.  Certain "passthru" options to PCPP can generate these.

        * Certain directives will perform macro replacement using a new and
          separate replacement iterator from the one being used by the present
          Source.  It is given only the specified tokens.  In the case of a #if
          or #elseif, it also handles "defined" according to the Standard.
        
    2. Macro replacement.

        This an iterator which takes the output of the Directives pass, above,
        and creates an output iterator.

        It uses the prep.macros dictionary to identify any tokens which are
        identifiers with a value in the dictionary.  Any tokens which are not
        such are passed on to the output iterator without any more action.

        The macros dict changes if the Directives pass processes a #define or
        #undef directive.  The sequence of events is:

        1. Directive pass produces token `t1`.
        2. Replacement pass receives `t1` and uses the current dict.
        3. Directive pass changes the macros dict.
        4. Directive pass produces token `t2`.
        5. Replacement pass receives `t2` and uses the updated dict.

        The macro name (token.value), and its definition, are examined.

        In any of the following cases, if the macro is not replaced, then it is
        passed on to the output iterator without any more action.

        If the macro name is in token.hide, then the macro is not replaced.
        This is because the token was produced while replacing the same macro,
        even if this occurred in an included file.

        If the macro is function-like, then:
            The input tokens are scanned, looking for an argument list,
            starting with a "(" token.  If there is no "(", then the macro is
            not replaced.

            The Standards say it is undefined behavior if a directive is seen
            in the source file before the end of the argument list is reached.
            GCC will treat any further tokens as normal.  So the closing
            parenthesis could be contained within an included file or a
            conditional group.

            But if the end of the source file is reached, then this is an error
            argument list is incomplete.  The macro is not expanded in this
            case, and the incomplete argument list is ignored.

            The macro's replacement list undergoes argument substitution,
            stringizing, and concatenation, according to the Standards.  In
            most cases, an argument is macro replaced, using a new and separate
            macro replacement iterator.

        If the macro is object-like, then the replacement list undergoes
        concatenation.

        In either case, the macro token (and argument list, if any) are
        replaced by the macro's replacement list as modified above.  The new
        sequence of tokens, along with remaining tokens to the end of the
        source file, is rescanned for replacements.

        With the PCCP implementation, the Replacement pass now takes its input
        from the new replacement, followed by the remaining tokens provided by
        its input.  The itertools.chain() function is used for this purpose.
        

3. Output .i file writing.
    The Writer object, belonging to the prep, has the job of taking the tokens
    from the top level Source's token iterator and translating them to text
    output.  Keep in mind that this iterator contains tokens from #include'd
    files, too.

    The current input location, known by the variable `inloc`, has a line
    number and source.  It is set by any location token (while emitting a #line
    directive).  It is the presumed location.

    The current output location, known by `outloc`, has a line number and the
    same source.   Its meaning is whatever a compiler will take as the current
    line, based on whatever has been written so far, including line directives.
    It is set by the same location token that sets inloc.

    Input tokens have line numbers which are not affected by any change in
    presumed location.  When the Writer sees a location token, this token bears
    the current actual location that caused the location token to be generated,
    as well as the presumed location stored in token.pos.  The writer will
    compute the difference in line numbers as a presumed line offset.
    Thereafter, whenever a source line number is seen, it is changed to the
    presumed line number by adding this offset.

    When the Writer gets a token to be written out, it sets inloc.lineno to the
    presumed input line (the source input line plus the offset).  Before
    writing the value, it forces outloc to match inloc.  If it's an increase of
    up to 7 lines (as GCC does), it writes that many newline characters.
    Otherwise if the change is nonzero (even negative), it writes a line
    directive.

    When writing data that contains some newlines, outloc is increased by that
    number.

    If an input token is the first one after changing outloc.lineno, it first
    writes space characters to match tok.colno.  Otherwise, if token.spacing is
    true, it writes a single space character.  After that, it writes the
    token.value string.

    The Source object generates no tokens for a blank line (only whitespace).
    However, macro replacement can cause a non-empty line to become empty.  In
    this case, the macro replacement will insert a "null" token (one with "" as
    value), so that the Writer write something.  This only applies to GCC
    emulation.
'''

# ------------------------------------------------------------------
# Preprocessor object
#
# Object representing a preprocessor.  Contains macro definitions, include
# directories, and other information
# ------------------------------------------------------------------

class Preprocessor(PreprocessorHooks):
    """
    Generic preprocessor object, which accepts various arguments, and
    generates preprocessed tokens.

    It interacts with a subclass as follows:
        1. Subclass __init__() sets some attributes to non-default values,
           then calls self.__init__().  These attributes are listed below.
        2. Subclass calls __init__().  Produces self.startup, a top-level
           source file containing various standard #define directives.
        3. Subclass may modify various members, including self.startup.
        4. self.startup may be extended by adding more text.  Input files are
           specified by #include directives.
        5. Subclass calls self.parse(), which is a generator of PpTok objects
           to parse self.startup.  This executes directives such as #define,
           and processes input files named in #include directives.  For legacy
           subclasses, the generator is also stored in self.parser.
        6. Iterate on this generator.  During the parsing, various hooks are
           called, which can be overridden in the subclass, to customize the
           behavior.
        7. Subclass does something with the tokens, such as formatting the
           tokens and writing to an output file.  Another possibility is to
           feed the tokens to a downstream compiler.
    """

    # These class attributes can be overridden on self as instance attributes
    # by the subclass constructor.  This should be done BEFORE calling
    # self.__init__, unless otherwise indicated. ...

    # Don't generate tokens of these types.  Legacy parse() method can also
    # set this.
    ignore: Collection[TokType] = {}

    # Version (4-digit year) of either C or C++ Standard being followed.
    # Set one but not both of them.
    c_ver: int = 0          # 1999, 2011, 2017, or 2023
    cplus_ver: int = 0      # 2011, 2014, 2017, 2020, or 2023

    # Handle trigraph sequences.  If None, then it will depend on whether the
    # C or C++ standard specifies them.
    trigraphs: bool | None = None

    # Expand leading tabs to this many spaces.  This will align the output
    # line to match the input line for readability.
    tabstop: int = 0

    # Form of a generated #line directive.  What's generated is
    # f'{self.line_directive} {line} {file}`.  None means don't generate any.
    line_directive: str = "#"

    # Startup source.  Write directives or other text, as well as #include
    # directives for input file(s) to be parsed.  For convenience,
    # self.startup_define() and self.startup_include() can be used.
    startup: io.StringIO = io.StringIO()

    # Tool emulation (gcc or clang), if any.  Value is the MAJOR.MINOR.
    # version string.  Don't set both of them.
    gcc: str = None
    clang: str = None
    @property
    def emulate(self) -> bool: return self.gcc or self.clang

    # Use GNU extensions.  Doesn't require gcc or clang.
    gnu: bool = False

    # Pass comments to the generated output.  With emulation, these are
    # ordinary tokens.  A comment before a directive causes the leading '#' to
    # be a regular token, not a directive.
    comments: bool = False

    # Reduce size of generated output.  If >= 2, all blank lines are suppressed.
    compress: int = 0

    # Only #include a file once if an #include guard is found.
    auto_pragma_once_enabled: bool = True

    # Pass through to the generated output any #include whose file name
    # matches this compiled regular expression.  Directives within the file
    # are still executed, but other generated tokens are ignored.
    passthru_includes: re.Pattern = None

    # Pass through magic macros (__COUNTER__, __LINE__, __FILE__, __DATE__ and
    # __TIME__) to the generated output without interpretation.  Set this
    # BEFORE calling __init__().
    passthru_magic_macros: bool = False

    # Pass through #include directives to generated output when the named file
    # is not found.
    passthru_unfound_includes: bool = False

    # Pass through conditional directives to generated output when they
    # contain an undefined macro.  If this is false, macro is evaluated as 0,
    # except that the subclass hooks can specify a different result.
    passthru_unknown_exprs: bool = False

    # Execute and pass through to the generated output all #define and #undef
    # directives.
    passthru_defines: bool = False

    # Ignore #define and #undef for these macro names, but pass them through
    # to the generated output.
    nevers: frozenset[str] = frozenset()

    # Write a diagnostic log file.
    debug: bool = False

    # Encoding for input files.  If None, then use UTF-8, but subclass hook
    # can override this.
    input_encoding: str = 'utf-8'

    # Encoding for output files.
    output_encoding: str = 'utf-8'

    # Write error and warning messages to this file.  Constructor will change
    # None to sys.stderr.
    diag: TextIO = None

    # These attributes vary during preprocessing...

    # Name of a macro which might be an #include guard for a source file if it
    # ever gets defined.
    potential_include_guard: str = None

    # Files currently being translated or already translated.  Indexed by
    # abspath.
    files: Mapping[str, SourceFile]

    # Files currently being translated.  The top file comes from an #include
    # in the startup script.  Its directory is a search location for #include
    # "..." but not <...>.
    files_active: Stack[SourceFile]

    # Source objects being translated, or already translated.
    sources: Stack[Source]

    # Manages all global Position values
    positions: PosMgr

    # Mapping of global positions to the sources.
    source_pos_tab: PosTab[Source]

    # All unique Hide objects.
    hides: HideDict = {}

    # Source objects currently being lexed, each one #include-ing the next.
    sources_active: Stack[Source]

    # Next global position to be allocated to a new Source.
    next_source_pos: Pos = 0

    # Count of error and warning messages issued.
    return_code: int = 0
    warnings: int = 0

    def __init__(self, lexer=None):
        super().__init__()
        self.diag = self.diag or sys.stderr
        if self.trigraphs is None:
            if self.emulate:
                self.trigraphs = (
                    (   self.c_ver
                        or (self.cplus_ver and self.cplus_ver < 2017)
                    )
                    and not self.gnu
                    )
            else:
                self.trigraphs = (
                    (   (self.c_ver and self.c_ver < 2023)
                        or (self.cplus_ver and self.cplus_ver < 2014)
                    )
                )
        self.rewrite_paths = [
            (re.escape(os.path.abspath('') + os.sep) + '(.*)', '\\1')]
        self.source_pos_tab = PosTab()
        self.sources_active = Stack[Source](lead=': ')
        self.files = {}
        self.positions = PosMgr(self)

        self.log = DebugLog(self)
        if lexer is None:
            lexer = default_lexer(self)
        self.lexer = lexer
        self.macros = Macros(self)

        # Script with text to parse.  #define's and #include's.
        self.startup = io.StringIO()
        self.startup.name = '< top level >'
        startup_define = lambda macro: self.startup_define(macro)
        # Put these defines in front of what's already in the file.
        self.startup.seek(0)
        # Magic macros
        if not self.passthru_magic_macros:
            self.macros.define_dynamic('__COUNTER__')
            self.macros.define_dynamic('__FILE__')
            self.macros.define_dynamic('__LINE__')
            tm = time.localtime()
            startup_define(f"__DATE__ \"{time.strftime('%b %e %Y', tm)}\"")
            startup_define(f"__TIME__ \"{time.strftime('%H:%M:%S', tm)}\"")

        if self.gcc:
            startup_define(f"__GNUC__ {self.gcc}")
        if self.clang:
            parts = self.clang.split('.')
            if len(parts) != 2:
                raise Exception("--clang option requires a MAJOR.MINOR value.")
            startup_define(f"__clang__ 1")
            startup_define(f"__clang_major__ {parts[0]}")
            startup_define(f"__clang_minor__ {parts[1]}")
            startup_define(f"__GNUC__ {parts[0]}")
        self.gnu_ext: bool = bool(self.gnu)
        if not self.gnu_ext:
            startup_define('__STRICT_ANSI__ 1')

        self.gplus_mode: bool = bool(self.emulate and self.cplus_ver)
        if self.trigraphs:
            startup_define('__PCPP_TRIGRAPHS__ 1')


        self.include_times = []  # list of FileInclusionTime
        self.TokType = lexer.TokType
        lexer.current_line = []     # All tokens found since start of current line.
        self.evaluator = Evaluator(self)
        self.path = ['.']           # list of -I formal search paths for includes
        self.files_active = Stack[SourceFile]()
        self.passthru_includes = None
        # A file is skipped if its name is in this dict and the object has a
        # true value.
        self.debugout = None
        self.auto_pragma_once_enabled = True

        # Probe the lexer for selected tokens
        self.__lexprobe()

        startup_define("__PCPP__ 1")
        if not self.gnu: startup_define("__STDC__ 1")
        if self.cplus_ver:
            # Standard value of __cplusplus.  Note, GCC gets C++20 wrong!
            modes = {
                    2011 : '201103L',
                    2014 : '201402L',
                    2017 : '201703L',
                    2020 : self.emulate and '201709L' or '202002L',
                    2023 : '202302L',
                }
            startup_define(f"__cplusplus {modes[self.cplus_ver]}")
        if self.c_ver:
            # Standard value of __STDC_VERSION__.
            modes = {
                    1999 : '199901L',
                    2011 : '201112L',
                    2017 : '201710L',
                    2023 : self.emulate and '202000L' or '202311L',
                }
            startup_define(f"__STDC_VERSION__ {modes[self.c_ver]}")
        # Go to the end, so that subclass can add more stuff.
        self.startup.seek(0, os.SEEK_END)

        self.parser = None

    def startup_define(self, defn: str) -> None:
        """
        Add a #define to the startup script.  Given the macro name and
        possibly the definition after some whitespace.
        """
        print(f"#define {defn}", file=self.startup)

    def startup_include(self, filename: str) -> None:
        """ Add a #include to the startup script. """
        print(f"#include \"{filename}\", file=self.startup")

    def startup_line(self, line: str, **kwds) -> None:
        """ Add a line to the startup script, like print(). """
        print(line, file=self.startup, **kwds)

    # ----------------------------------------------------------------------
    # parse()
    #
    # Method to parse a startup script or a top level input file.
    #   Store TokIter for the tokens in self.parser.
    # ----------------------------------------------------------------------
    def parse(self, input: str | io.IOBase = None, source: str = None,
              ignore: set[str] = {}) -> None:
        """
        Parse startup script or input file or plain data.
        """
        if isinstance(input, io.IOBase):
            input = input.read()
            if isinstance(input, bytes):
                input = input.decode()
        elif input:
            input = self.string_file(input, '<string>')
        else:
            input = self.startup
            input.seek(0)

        # Create SourceFile for the input file.
        top : SourceFile = SourceFile.openfile(input, self)
        #top = SourceFile.openfile(input, self, dirname='.')
        self.files_active.append(top)
        self.parser = self.parsegen(top)

    @staticmethod
    def string_file(name: str, data: str = '') -> StringIO:
        str = io.StringIO(data)
        str.name = name
        return str

    # ----------------------------------------------------------------------
    # token()
    #
    # Legacy method to return individual tokens.  These are set up by the call
    # to self.parse().  Returns next token each time, and None at the end.
    #
    # skips any token whose type (that is, 'CPP_XXX') was given in ignore
    # argument to self.parse().
    # ----------------------------------------------------------------------
    def token(self) -> PpTok | None:
        """
        Legacy method to return individual tokens, except if the name of the
        type is in self.ignore.
        """
        for tok in self.parser:
            if self.ignore and tok.type.name in self.ignore:
                continue
            return tok
        return None

    def alloc_source_pos_block(self, source: Source, span: Pos
                               ) -> PosRange:
        """
        Allocate next block of global positions to given Source.  Add to
        sources and source_pos_tab.  Returns (start pos, stop pos)
        """
        pos: Pos = self.next_source_pos
        end: Pos = pos + span
        self.next_source_pos = end
        self.source_pos_tab.add(PosRange(pos, end), source)
        return PosRange(pos, end)

    def find_source(self, pos: Position) -> Source:
        """
        Get the Source whose block of global positions include given pos.
        """
        return self.source_pos_tab.find(pos)

    def hide(self, *names: str) -> Hide:
        """ The unique Hide for these names, possibly creating a new one. """
        h = frozenset(names)
        try: return self.hides[h]
        except IndexError:
            h = self.hides[h] = Hide(self, h)
            return h

    # ----------------------------------------------------------------------
    # parsegen()
    #
    # Parse an input string from a main or included SourceFile.  Returns
    # TokIter for resulting tokens.  Creates and nests a Source object in
    # self.sources_active while doing the iteration.  Generates start and end locator
    # tokens in the parent (i.e., before and after nesting the Source).
    # ----------------------------------------------------------------------

    @TokIter.from_generator
    def parsegen(
            self,
            #input: str, # Contents of the file
            #*,
            file: SourceFile,
            dir: PpTok = None,         # Start of the #include directive
            #filename: str,
            #source: str = None,
            #abssource: str = None,
            #hdr_name: str = None,
            #path: str = None,
            #once: IncludeOnce = None,
            #parentlineno: int = 1,
            ) -> TokIter:
        """ Parse an input string.  Generate PpTok objects. """
        self.file = file
        source = file.filename
        #rewritten_source = source
        abspath = file.abspath
        #if abspath:
        #    rewritten_source = abspath
        #    for rewrite in self.rewrite_paths:
        #        temp = re.sub(rewrite[0], rewrite[1], rewritten_source)
        #        if temp != abspath:
        #            # rewrite[0] was found.
        #            rewritten_source = temp
        #            break

        #if not source:
        #    source = ""
        #if not rewritten_source:
        #    rewritten_source = ""
        if self.verbose < 2:
            source = os.path.basename(source)
        src = Source(file)
        #src = Source(self, source, abspath, hdr_name, rewritten_source,
        #             once=file.once)

        if dir:
            yield dir.make_pos(OutPosEnter, source=src)
        with self.sources_active.nest(src):
            yield from src.parsegen(dir=dir)
        if dir and dir.source.parent:
            tok = dir.make_pos(OutPosLeave)
            yield tok


    @property
    def currsource(self) -> Source | None:
        """ Currently parsing Source, if any. """
        return self.sources_active.top()

    # ----------------------------------------------------------------------
    # __lexprobe()
    #
    # This method probes the preprocessor lexer object to discover
    # the token types of symbols that are important to the preprocessor.
    # If this works right, the preprocessor will simply "work"
    # with any suitable lexer regardless of how tokens have been named.
    # ----------------------------------------------------------------------

    def __lexprobe(self):
        def probe(data: str, message: str) -> TokType | None:
            toks = self.lexer.parse(data)
            tok = next(toks, None)
            if not tok or tok.value != data:
                raise TypeError(f"Couldn't determine type for {message}.")
                return None
            return tok.type

        probe("(", "left paren")
        probe("##", "paste")
        probe("#", "stringize")
        # Determine the token type for identifiers
        self.t_ID = probe("identifier", "identifier")

        # Determine the token type for integers
        self.t_INTEGER = probe("12345", "integer")
        self.t_INTEGER_TYPE = type("12345")

        # Determine the token type for character
        self.t_CHAR = probe("'a'", "character")
            
        # Determine the token type for strings enclosed in double quotes
        self.t_STRING = probe("\"filename\"", "string")

        # Determine the token type for whitespace--if any
        self.t_SPACE = probe("  ", "space")

        # Determine the token type for newlines
        #self.t_NEWLINE = probe("\n", "newline")

        # Determine the token type for token pasting.
        self.t_DPOUND = probe("##", "token pasting operator")

        # Determine the token type for ellipsis.
        self.t_ELLIPSIS = probe("...", "elipsis")

        # Determine the token types for ternary operator.
        self.t_TERNARY = probe("?", "ternary \? operator")
        self.t_COLON = probe(":", "ternary : operator")

        # Determine the token types for comments.
        self.t_COMMENT1 = probe("/* comment */", "block comment")
        self.t_COMMENT2 = probe("// comment", "line comment")
        self.t_COMMENT = (self.t_COMMENT1, self.t_COMMENT2)

        # Determine the token types for any whitespace.
        self.t_WS = (self.t_SPACE, ) + self.t_COMMENT
        #self.t_WS = (self.t_SPACE, self.t_NEWLINE) + self.t_COMMENT

        # Check for other characters used by the preprocessor
        chars = [ '<','>','#','##','(',')',',','.']
        for c in chars:
            probe(c, f"{c!r} required for preprocessor")

    # ----------------------------------------------------------------------
    # add_path()
    #
    # Adds a search path to the preprocessor.  
    # ----------------------------------------------------------------------

    def add_path(self,path):
        """Adds a search path to the preprocessor. """
        self.path.append(path)
        # If the search path being added is relative, or has a common ancestor to the
        # current working directory, add a rewrite to relativise includes from this
        # search path
        relpath = None
        try:
            relpath = os.path.relpath(path)
        except: pass
        if relpath is not None:
            self.rewrite_paths += [(re.escape(os.path.abspath(path) + os.sep) + '(.*)', os.path.join(relpath, '\\1'))]

    def fix_path_sep(self, path: str, sep: str = '/') -> str:
        """ Changes the path separators in given path. """
        if self.emulate:
            # GCC preserves the separators, but doubles single backslashes.
            if os.sep == '\\':
                path = re.sub(r'\\',r'\\\\', path)
                #path = re.sub(r'\\\\?',r'\\\\', path)
                #path = path.replace(os.sep, '\\\\')
        elif os.sep != sep:
            path = path.replace(os.sep, sep)
        return path

    # ----------------------------------------------------------------------
    # define()
    #
    # Define a new macro
    # Called with the tokens following '#define' and any whitespace.
    # ----------------------------------------------------------------------

    def define(self, tokens: TokIter, **kwds):
        """Define a new macro"""
        self.macros.define(tokens)

    # ----------------------------------------------------------------------
    # undef()
    #
    # Undefine a macro
    # ----------------------------------------------------------------------

    def undef(self,tokens):
        """Undefine a macro"""
        if type(tokens) is str:
            id = tokens
        else:
            id = tokens[0].value
        try:
            del self.macros[id]
        except LookupError:
            pass

    # ----------------------------------------------------------------------
    # include()
    #
    # Implementation of file-inclusion
    # ----------------------------------------------------------------------

    def include(self, tokens: Tokens, original_line: str) -> Iterator[PpTok]:
        """ Implementation of file-inclusion.
        Given Tokens is what follows the '#include', or perhaps a macro expansion
        of same.
        The lexer has parsed either a <name> or a "name" into a single token.
        See (C99 6.4.7).
        """
        # Try to extract the filename and then process an include file

        if tokens[0].type.hhdr:
            is_system_include = True
            # Include <...>
            filename = tokens[0].value[1:-1]
            # Search only formally specified paths
            qfile = None
            path = self.path
        elif tokens[0].type.qhdr:
            is_system_include = False
            filename = tokens[0].value[1:-1]
            # Search from last enclosing include file, as well as formally
            # specified paths
            qfile = self.files_active.top()
            #path = [self.files_active.top().dirname] + self.path
        else:
            # Malformed filename.  Treat this similar to a missing file.
            p = self.on_include_not_found(
                is_malformed=True, is_system_include=False,
                curdir=self.files_active.top(''),
                includepath=tokens[0].value)
            assert p is None
            return
        path = self.path or ['']

        # Create a SourceFile object.
        file = SourceFile.open(filename, qfile, path, self)

        # Decide whether to translate this file.
        if file.pathdir is None:
            # File was not found.
            filename = self.on_include_not_found(
                is_malformed=False, is_system_include=is_system_include,
                curdir=self.files_active.top().dirname,
                includepath=filename)
            # Did not raise OutputDirective, so returns a replacement filename.
            assert filename is not None
            file = SourceFile.open(filename, path, self)
        if file.once:
            self.log.write(f"File \"{file.filename}\" skipped as already seen.",
                           token=tokens[0])
            if (self.passthru_includes is not None
                and self.passthru_includes.match(
                    ''.join([x.value for x in tokens]))
                ):
                yield from original_line
            return
        # Go ahead and translate.
        with self.files_active.nest(file):

            if (self.passthru_includes is not None
                and self.passthru_includes.match(
                    ''.join([x.value for x in tokens]))
                ):
                yield from original_line
                for tok in self.parsegen(data, filename, fulliname):
                    pass
            else:
                dir = original_line[0]
                # Tell prep.write() to begin a new source file.

                yield from self.parsegen(file, dir)

        return

    @TokIter.from_generator
    def tokens(self) -> TokIter:
        """Method to return individual tokens"""
        try:
            while True:
                tok = next(self.parser)
                if (tok and tok.type not in self.ignore):
                    yield tok
        except StopIteration:
            self.parser = None

    @staticmethod
    def showtok(tok: PpTok) -> str:
        return tok.value

    def showtoks(self, toks: Iterable[PpTok]) -> str:
        return ''.join(map(self.showtok, toks))

    # Keeping track of levels of nesting, using current source nesting.
    # This is available to log.write() to indent the output.

    @property
    def nesting(self) -> int:
        return self.currsource.nesting

    @nesting.setter
    def nesting(self, n) -> None:
        self.currsource.nesting = n

    @contextlib.contextmanager
    def nest(self, n: int = 1):
        self.nesting += n
        try:
            yield
        finally:
            self.nesting -= n


    def _error_msg(self, source: Source, msg: str, line: int = 0, col: int = 0,
                    warn: bool = False):
        """ Prints error or warning message to diagnostic output
        and increments the return code or warning count.
        """
        if not source: return
        file = source.filename
        if self.verbose < 2:
            file = os.path.basename(file)
        if line:
            file = f"{file}:{line}"
            if col:
                file = f"{file}:{col}"

        type = warn and 'warning' or 'ERROR'
        self.log.write(f"{type}: {msg}", source=source, lineno=line, colno=col)
        msg = f"{file} {type}: {msg}"
        print(msg, file = self.diag)
        if warn:
            self.warnings += 1
        else:
            self.return_code += 1

    def on_error_token(self, token: PpTok, msg: str, warn: bool = False
                       ) -> None:
        """
        Called when the preprocessor has encountered an error or warning
        associated with a token.
        """
        # The token might not yet have turned into a PpTok.
        if isinstance(token, PpTok):
            obj = token
        else:
            obj = token.lexer.owner
        self._error_msg(obj.source, msg, obj.lineno, obj.colno, warn=warn)

    def on_warn_token(self, token: PpTok, msg: str):
        """
        Called when the preprocessor has encountered a warning associated with
        a Token.
        """
        self.on_error_token(token, msg, warn=True)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
