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
from typing import Iterator
if __name__ == '__main__' and __package__ is None:
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from pcpp.directive import (Directive, Action, OutputDirective)
from pcpp.parser import (STRING_TYPES, PreprocessorHooks)
from pcpp.lexer import (default_lexer, PpLex, TokType, Tokens, reduce_ws, )
#from pcpp.parser import STRING_TYPES, default_lexer, trigraph, Macro, Action, OutputDirective, PreprocessorHooks
from pcpp.evaluator import Evaluator
from pcpp.ply.lex import LexToken
from pcpp.macros import Macro, Macros
from pcpp.debug_log import DebugLog
import io
FILE_TYPES = io.IOBase
clock = time.process_time

__all__ = ['Preprocessor', 'PreprocessorHooks', 'OutputDirective', 'Action', 'Evaluator']

# ------------------------------------------------------------------
# File inclusion timings
#
# Useful for figuring out how long a sequence of preprocessor inclusions actually is
# ------------------------------------------------------------------

class FileInclusionTime(object):
    """The seconds taken to #include another file"""
    def __init__(self,including_path,included_path,included_abspath,depth):
        self.including_path = including_path
        self.included_path = included_path
        self.included_abspath = included_abspath
        self.depth = depth
        self.elapsed = 0.0

# ------------------------------------------------------------------
# Preprocessor object
#
# Object representing a preprocessor.  Contains macro definitions,
# include directories, and other information
# ------------------------------------------------------------------

class Preprocessor(PreprocessorHooks):
    def __init__(self, lexer=None):
        super(Preprocessor, self).__init__()
        self.include_times = []  # list of FileInclusionTime
        self.log = DebugLog(self)
        self.currsource = None
        self.topsource = Source(self, '< top level >')
        if lexer is None:
            lexer = PpLex(self, default_lexer())
        self.lexer = lexer
        lexer.current_line = []     # All tokens found since start of current line.
        self.evaluator = Evaluator(self.lexer)
        self.macros = Macros(self)
        self.path = []           # list of -I formal search paths for includes
        self.temp_path = []      # list of temporary search paths for includes
        self.rewrite_paths = [(re.escape(os.path.abspath('') + os.sep) + '(.*)', '\\1')]
        self.passthru_includes = None
        self.include_once = {}
        self.include_depth = 0
        self.return_code = 0
        self.debugout = None
        self.auto_pragma_once_enabled = True
        self.line_directive = '#line'
        self.compress = False
        self.assume_encoding = None

        # Probe the lexer for selected tokens
        self.__lexprobe()

        tm = time.localtime()
        self.define("__DATE__ \"%s\"" % time.strftime("%b %e %Y",tm))
        self.define("__TIME__ \"%s\"" % time.strftime("%H:%M:%S",tm))
        self.define("__PCPP__ 1")
        self.expand_linemacro = True
        self.expand_filemacro = True
        self.expand_countermacro = True
        self.countermacro = 0
        self.parser = None
        self.nesting = 0
        self.loglines = []      # List of (left, right), so that all rights get aligned.

    # ----------------------------------------------------------------------
    # parse()
    #
    # Parse input text.
    # ----------------------------------------------------------------------
    def parse(self, input: str, source=None, ignore={}):
        """Parse input text."""
        if isinstance(input, FILE_TYPES):
            if source is None:
                source = input.name
            input = input.read()
        self.ignore = ignore
        self.parser = self.parsegen(input,source,os.path.abspath(source) if source else None)
        if source is not None:
            dname = os.path.dirname(source)
            self.temp_path.insert(0,dname)
        
    # ----------------------------------------------------------------------
    # parsegen()
    #
    # Parse an input string from a top level source file.
    # ----------------------------------------------------------------------

    def parsegen(self, input: str, source: str=None, abssource: str=None) -> Iterator[LexToken]:
        """Parse an input string.  Generate LexToken objects. """
        rewritten_source = source
        if abssource:
            rewritten_source = abssource
            for rewrite in self.rewrite_paths:
                temp = re.sub(rewrite[0], rewrite[1], rewritten_source)
                if temp != abssource:
                    rewritten_source = temp
                    if os.sep != '/':
                        rewritten_source = rewritten_source.replace(os.sep, '/')
                    break

        if not source:
            source = ""
        if not rewritten_source:
            rewritten_source = ""
            
        self.include_depth += 1
        if self.verbose < 2:
            source = os.path.basename(source)
        src = Source(self, source)
        if self.expand_filemacro:
            self.undef("__FILE__")
            self.define("__FILE__ \"%s\"" % rewritten_source)

        yield from src.parsegen(input)

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
            lexer = self.lexer.clone()
            lexer.input(data)
            lexer.begin('INITIAL')

            tok = lexer.token()
            if not tok or tok.value != data:
                raise TypeError(f"Couldn't determine type for {message}.")
                return None
            return tok.type

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
        self.t_NEWLINE = probe("\n", "newline")

        # Determine the token type for token pasting.
        self.t_DPOUND = probe("##", "token pasting operator")

        # Determine the token type for elipsis.
        self.t_ELLIPSIS = probe("...", "elipsis")

        # Determine the token types for ternary operator.
        self.t_TERNARY = probe("?", "ternary \? operator")
        self.t_COLON = probe(":", "ternary : operator")

        # Determine the token types for comments.
        self.t_COMMENT1 = probe("/* comment */", "block comment")
        self.t_COMMENT2 = probe("// comment", "line comment")
        self.t_COMMENT = (self.t_COMMENT1, self.t_COMMENT2)

        # Determine the token types for any whitespace.
        self.t_WS = (self.t_SPACE, self.t_NEWLINE) + self.t_COMMENT

        # Check for other characters used by the preprocessor
        chars = [ '<','>','#','##','\\','(',')',',','.']
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

    # ----------------------------------------------------------------------
    # define()
    #
    # Define a new macro
    # Called with either the string following '#define '
    #   or the tokens following '#define' and any whitespace.
    # ----------------------------------------------------------------------

    def define(self,tokens):
        """Define a new macro"""
        if isinstance(tokens,STRING_TYPES):
            self.macros.define_from_text(tokens, self.lexer)
        else:
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

    def include(self,tokens,original_line):
        """Implementation of file-inclusion"""
        # TODO: Some things are undefined behavior if part of the file name:
        #   "'" "\\".  They do NOT introduce character constants or string.
        #   "//", or "/*".  They do NOT introduce comments.
        #   '"' in a <filename>, but can't occur in a "filename>".
        #   See C99 Section 6.4.7.
        # Try to extract the filename and then process an include file
        if not tokens:
            return
        with self.nest():
            if tokens[0].value != '<' and tokens[0].type != self.t_STRING:
                tokens = self.tokenstrip(prep.macros.expand(tokens))

            is_system_include = False
            if tokens[0].value == '<':
                is_system_include = True
                # Include <...>
                i = 1
                while i < len(tokens):
                    if tokens[i].value == '>':
                        break
                    i += 1
                else:
                    self.on_error(tokens[0].source,tokens[0].lineno,"Malformed #include <...>")
                    return
                filename = "".join([x.value for x in tokens[1:i]])
                # Search only formally specified paths
                path = self.path
            elif tokens[0].type == self.t_STRING:
                filename = tokens[0].value[1:-1]
                # Search from each nested include file, as well as formally specified paths
                path = self.temp_path + self.path
            else:
                p = self.on_include_not_found(True,False,self.temp_path[0] if self.temp_path else '',tokens[0].value)
                assert p is None
                return
            if not path:
                path = ['']
            while True:
                #print path
                for p in path:
                    iname = os.path.join(p,filename)
                    fulliname = os.path.abspath(iname)
                    if fulliname in self.include_once:
                        self.log.write("#include \"%s\" skipped as already seen" % (fulliname))
                        if self.passthru_includes is not None and self.passthru_includes.match(''.join([x.value for x in tokens])):
                            for tok in original_line:
                                yield tok
                        return
                    try:
                        ih = self.on_file_open(is_system_include,fulliname)
                        data = ih.read()
                        ih.close()
                        dname = os.path.dirname(fulliname)
                        if dname:
                            self.temp_path.insert(0,dname)
                        if self.passthru_includes is not None and self.passthru_includes.match(''.join([x.value for x in tokens])):
                            for tok in original_line:
                                yield tok
                            for tok in self.parsegen(data,filename,fulliname):
                                pass
                        else:
                            # Notify the output write() of changes to the source, both before and after.
                            #self.changed_source = True
                            #yield None
                            for tok in self.parsegen(data,filename,fulliname):
                                yield tok
                            #self.changed_source = True
                            #yield None
                        if dname:
                            del self.temp_path[0]
                        return
                    except IOError:
                        pass
                else:
                    p = self.on_include_not_found(False,is_system_include,self.temp_path[0] if self.temp_path else '',filename)
                    assert p is not None
                    path.append(p)

    # ----------------------------------------------------------------------
    # tokenstrip()
    # 
    # Remove leading/trailing whitespace tokens from a token list
    # ----------------------------------------------------------------------

    def tokenstrip(self, tokens: Tokens) -> Tokens:
        """Remove leading/trailing whitespace tokens from a token list"""
        i = 0
        while i < len(tokens) and tokens[i].type.ws:
            i += 1
        del tokens[:i]
        i = len(tokens)-1
        while i >= 0 and tokens[i].type.ws:
            i -= 1
        del tokens[i+1:]
        return tokens

    # ----------------------------------------------------------------------
    # token()
    #
    # Method to return individual tokens
    # ----------------------------------------------------------------------
    def token(self):
        """Method to return individual tokens"""
        try:
            while True:
                tok = next(self.parser)
                if (tok and tok.type not in self.ignore):
                    return tok
        except StopIteration:
            self.parser = None
            return None

    @staticmethod
    def showtok(tok: LexToken) -> str:
        return tok.value

    def showtoks(self, toks: Iterable[LexToken]) -> str:
        return ''.join(map(self.showtok, toks))

    # Keeping track of levels of nesting, using self.nesting.
    # This is available to log.write() to indent the output.

    @contextlib.contextmanager
    def nest(self, n: int = 1):
        self.nesting += n
        try:
            yield
        finally:
            self.nesting -= n

    #def writedebug(self, text: str, indent = 0, nest: int = 0,
    #               token = None,
    #               source: Source = None, lineno = None, colno = None,
    #               contlocs: bool = True,       # Show location for continued lines.
    #               ):
    #    if self.debugout is None: return
    #    if not self.verbose and not enable: return
    #    if token:
    #        source = token.source
    #        lineno = lineno or token.lineno
    #    if source:
    #        filename = source.filename
    #    else:
    #        filename = __file__
    #    if self.verbose < 2:
    #        filename = os.path.basename(filename)
    #    try: ifstate = source.ifstate
    #    except: pass
    #    #ifstate = source.ifstate
    #    leader = (self.verbose
    #              and f"{ifstate.enable:d}:{ifstate.iftrigger:d}:{ifstate.ifpassthru:d} "
    #              or "")
    #    for i, line in enumerate(text.splitlines(), lineno):
    #        if not line.strip(): continue
    #        if colno is None and token: colno = token.colno
    #        col = f":{colno}" if colno and i == lineno else ""
    #        left: str = f"{leader}{filename}:{i}{col}"
    #        if not contlocs and i > lineno:
    #            left = ''
    #        self.loglines.append((
    #            left,
    #            f"{'| ' * (self.nesting + nest)}{' ' * indent}"
    #            f"{i > lineno and '... ' or ''}{line}")
    #            )

    #def writedebuglog(self):
    #    if self.loglines:
    #        lefts, _ = zip(*self.loglines)
    #        leftwidth = max(len(s) for s in lefts)
    #        for left, right in self.loglines:
    #            print("%-*s %s" % (leftwidth, left, right), file = self.debugout)

    def write(self, oh=sys.stdout):
        """Calls token() repeatedly, expanding tokens to their text and writing to the file like stream oh"""

        # Overall strategy:
        #   1. Get a logical line.  Keep track of line numbers.
        #       The line may have a smaller starting line number than an ending one.
        #       It's the starting line number that determines where it lies in the output.
        #   2. See if the line is all whitespace.  Comments don't count.
        #       If so, it won't be written out.
        #   3. See if the line is actually a change of source file marker.
        #       This changes the source file name and forces a line directive.
        #   4. Compress the line by combining consecutive whitespace into a single space,
        #       except at the start of the line.
        #   5. Account for any jumps in the line number.  Either write some blank lines
        #       or write a line directive.
        #   6. Write the line itself, one token at a time.

        line: Tokens = Tokens()
        outlineno: int                  # Current line number for output
        inlineno: int                   # Current line number for input
        lineno: int                     # Line number for current logical line.
        source: Source                  # Current file being parsed.
        all_ws: bool = True             # Current line is all space characters.
        write = oh.write

        def getline() -> Iterator:
            """ Generates next logical line, ignoring blank lines.
            """
            nonlocal lineno, inlineno, outlineno, source, all_ws
            # Loop over lines
            while True:
                # Loop over tokens
                while True:
                    tok: PpTok = self.token()
                    if not tok: return
                    if not tok.value:
                        if self.changed_source:
                            self.changed_source = False
                            source = tok.source
                            inlineno = outlineno = tok.lineno
                            linedirective()
                        continue
                    line.append(tok)
                    if tok.type.nl:
                        if not all_ws:
                            # We have a result line.
                            lineno = inlineno
                            yield
                            all_ws = True
                        inlineno = tok.lineno + 1
                        break
                    if not tok.type.space:
                        all_ws = False
                line.clear()

        def linedirective() -> None:
            # Writes a #line directive using the source and lineno variables.
            if source:
                filename = f" {source.filename}"
            else:
                filename = ""
            write(f"{self.line_directive} {outlineno}\"{filename}\"\n")

        def movetoline(lineno: int) -> None:
            """ Writes whatever necessary to set the current output line position.
            Either write blank lines or write a #line directive.
            """
            nonlocal outlineno
            skip = lineno - outlineno
            outlineno = lineno
            if skip:
                if skip > 6 and self.line_directive is not None:
                    linedirective()
                else:
                    write('\n' * skip)

        for _ in getline():
            # `line` holds the next line, which is not blank.
            movetoline(lineno)
            outlineno = lineno + 1
            for tok in line:
                write(tok.value)
            ...

        return
        # Establish these names as local variables, to keep the editor happy.
        del source, lineno, inlineno, outlineno

        lastlineno = 0
        lastsource = None
        done = False
        blanklines = 0
        while not done:
            emitlinedirective = False
            toks = []
            all_ws = True
            # Accumulate a line
            while not done:
                tok = self.token()
                if not tok:
                    done = True
                    break
                toks.append(tok)
                if tok.value:
                    if tok.value[0] == '\n':
                        break
                elif self.changed_source:
                    emitlinedirective = True
                    self.changed_source = False
                    lastsource = tok.source
                    blanklines = 0
                    break
                if not tok.type.ws:
                    all_ws = False
            if toks:
                linetok = toks[-1]
                if all_ws:
                    # Remove preceding whitespace so it becomes just a LF
                    if len(toks) > 1:
                        tok = linetok
                        toks = [ linetok ]
                    blanklines += linetok.value.count('\n')
                    continue
            # The line in toks is not all whitespace
            emitlinedirective |= (blanklines > 6) and self.line_directive is not None
            if not emitlinedirective and hasattr(linetok, 'source'):
                if lastsource is not linetok.source:
                    emitlinedirective = True
                    lastsource = linetok.source
                    blanklines = 0
            # Replace consecutive whitespace in output with a single space except at any indent
            first_ws = None
            #print(toks)
            for n in range(len(toks)-1, -1, -1):
                tok = toks[n]
                if first_ws is None:
                    if tok.type.ws and tok.value:
                        first_ws = n
                else:
                    if not tok.type.ws and len(tok.value) > 0:
                        m = n + 1
                        while m != first_ws:
                            del toks[m]
                            first_ws -= 1
                        first_ws = None
                        if self.compress > 0:
                            # Collapse a token of many whitespace into single
                            if toks[m].value and toks[m].value[0] == ' ':
                                toks[m].value = ' '
            if not self.compress > 1 and not emitlinedirective:
                newlinesneeded = linetok.lineno - lastlineno - 1
                if newlinesneeded > 6 and self.line_directive is not None:
                    emitlinedirective = True
                else:
                    while newlinesneeded > 0:
                        write('\n')
                        newlinesneeded -= 1
            lastlineno = linetok.lineno
            if not self.compress and blanklines >= 2:
                lastlineno -= blanklines
            if emitlinedirective and self.line_directive is not None:
                write(self.line_directive + ' ' + str(lastlineno) + ('' if lastsource is None else (' "' + lastsource.filename + '"' )) + '\n')
            if not self.compress and blanklines >= 2:
                write('\n' * blanklines)
                lastlineno += blanklines
            for tok in toks:
                if tok.type == self.t_COMMENT1:
                    lastlineno += tok.value.count('\n')
            blanklines = 0
            for tok in toks:
                write(tok.value)

class Source:
    """ Performs preprocessing on a single source file.
    Generates tokens, including from #include'ed files.
    """

    def __init__(self, prep: PreProcessor, filename: str) -> None:
        self.prep = prep
        self.filename = filename
        self.parent = prep.currsource

    # ----------------------------------------------------------------------
    # group_lines()
    #
    # Given an input string, this function splits it into lines.  Trailing whitespace
    # is removed. This function forms the lowest level of the preprocessor---grouping
    # text into a line-by-line format.
    # ----------------------------------------------------------------------

    def group_lines(self, input: str) -> Iterator[Tokens]:
        r"""Given an input string, this function splits it into lines.  Trailing whitespace
        is removed. This function forms the lowest level of the preprocessor---grouping
        text into a line-by-line format.
        """
        prep = self.prep
        lex = self.lexer = prep.lexer.clone()
        lines = [x.rstrip() for x in input.splitlines()]

        input = "\n".join(lines) + "\n"
        lex.input(input, self)

        current_line = []
        while True:
            tok = lex.token()
            if not tok:
                break
            current_line.append(tok)
            if tok.type.nl:
                current_line = list(reduce_ws(current_line, prep))
                yield current_line
                current_line = []

        if current_line:
            nltok = copy.copy(current_line[-1])
            nltok.type = prep.t_NEWLINE
            nltok.value = '\n'
            current_line.append(nltok)
            current_line = reduce_ws(current_line)
            yield current_line

    # ----------------------------------------------------------------------    
    # evalexpr()
    # 
    # Evaluate an expression token sequence for the purposes of evaluating
    # integral expressions.
    # ----------------------------------------------------------------------

    def evalexpr(self, tokens: Tokens) -> Tuple[bool, Tokens | None]:
        """Evaluate an expression token sequence for the purposes of evaluating
        integral expressions.  Result is either true or false.
        If false, there could be something to be passed through.
        """
        prep = self.prep
        if not tokens:
            prep.on_error_token(self.directive[0], "Empty control expression in directive")
            return (0, None)
        # tokens = tokenize(line)
        # Search for defined macros
        partial_expansion = False

        def replace_defined(tokens, again: bool = False) -> bool:
            """ Replace any 'defined X' or 'defined (X)' with integer 0 or 1.
            The token list is altered in-place.  Return True if anything replaced.
            """
            i = 0
            replaced = False
            while i < len(tokens):
                if tokens[i].type == prep.t_ID and tokens[i].value == 'defined':
                    j = i + 1
                    needparen = False
                    result = "0L"
                    while j < len(tokens):
                        if tokens[j].type.ws:
                            j += 1
                            continue
                        elif tokens[j].type is TokType.CPP_ID:
                            if tokens[j].value in self:
                                result = "1L"
                            else:
                                repl = prep.on_unknown_macro_in_defined_expr(tokens[j])
                                if repl is None:
                                    partial_expansion = True
                                    result = 'defined('+tokens[j].value+')'
                                else:
                                    result = "1L" if repl else "0L"
                            if not needparen: break
                        elif tokens[j].value == '(' and not needparen:
                            needparen = True
                        elif tokens[j].value == ')' and needparen:
                            break
                        else:
                            prep.on_error(tokens[i].source,tokens[i].lineno,"Malformed defined()")
                        j += 1
                    if result.startswith('defined'):
                        tokens[i].type = TokType.CPP_ID
                        tokens[i].value = result
                    else:
                        tokens[i].type = prep.t_INTEGER
                        tokens[i].value = prep.t_INTEGER_TYPE(result)
                    replaced = True
                    del tokens[i+1:j+1]
                i += 1
            return replaced

        # Replace any defined(macro) before macro expansion
        #replace_defined(tokens)
        tokens = prep.macros.expand(tokens, evalexpr=True)
        # Replace any defined(macro) after macro expansion
        #if replace_defined(tokens):
        #    # This is undefined behavior (C99 Section 6.10.1 para 4).
        #    prep.on_warn_token(tokens[0], "Macro expansion of control expression contains 'defined'")
        if not tokens:
            return (0, None)

        class IndirectToMacroHook(object):
            def __init__(self, p):
                self.__preprocessor = p.prep
                self.partial_expansion = False
            def __contains__(self, key):
                return True
            def __getitem__(self, key):
                if key.startswith('defined('):
                    self.partial_expansion = True
                    return 0
                repl = self.__preprocessor.on_unknown_macro_in_expr(key)
                #print("*** IndirectToMacroHook[", key, "] returns", repl, file = sys.stderr)
                if repl is None:
                    self.partial_expansion = True
                    return key
                return repl
        evalvars = IndirectToMacroHook(self)

        class IndirectToMacroFunctionHook(object):
            def __init__(self, p):
                self.__preprocessor = p.prep
                self.partial_expansion = False
            def __contains__(self, key):
                return True
            def __getitem__(self, key):
                repl = self.__preprocessor.on_unknown_macro_function_in_expr(key)
                #print("*** IndirectToMacroFunctionHook[", key, "] returns", repl, file = sys.stderr)
                if repl is None:
                    self.partial_expansion = True
                    return key
                return repl
        evalfuncts = IndirectToMacroFunctionHook(self)

        try:
            result = prep.evaluator(tokens, functions = evalfuncts, identifiers = evalvars).value()
            partial_expansion = partial_expansion or evalvars.partial_expansion or evalfuncts.partial_expansion
        except OutputDirective:
            raise
        except Exception as e:
            partial_expansion = partial_expansion or evalvars.partial_expansion or evalfuncts.partial_expansion
            if not partial_expansion:
                self.prep.on_error(tokens[0].source,tokens[0].lineno,"Could not evaluate expression due to %s (passed to evaluator: '%s')" % (repr(e), ''.join([tok.value for tok in tokens])))
                #import traceback; traceback.print_exc()
            result = 0
        return (result, tokens) if partial_expansion else (result, None)

    @dataclass
    class IfState:
        """ State of a control group controlled by any conditional inclusion
        directive other than #endif.  See C99/C23 Standard 6.10.1.
        Also #elifdef and #elifndef, new to C23.
        The entire source file itself is not a control group, but has an IfState.
        """
        enable: bool                # Preprocessing is active.
                                    # If not, directives are still examined.
        iftrigger: bool             # This or an earlier controlling directive is true.
                                    # This means that all following groups are skipped.
        ifpassthru: bool            # Enables certain things to be passed through to output.
        startlinetoks: List[LexToken] = field(default_factory=list)
                                    # The controlling directive.  [] if entire file.
        rewritten: bool = False
        top: bool = False           # This is the entire source file.

        @property
        def may_enable(self) -> bool:
            """ True if the next group can be enabled by a true condition. """
            return not self.iftrigger

        def advance(self, cond: bool) -> None:
            if cond:
                if self.enable: self.enable = False
                else: self.enable = self.iftrigger = True
            else:
                self.enable = False

    class IfStack(list[IfState]):
        """ Nested control groups in the current source file, starting from the outermost.
            The entry for the entire file is NOT on the list.
        """
        # Source.ifstate starts out with a state for the entire file, which is not on the stack.
        # Opening a new group pushes the current state onto the stack and creates a new state,
        #   which, again, is not on the stack.
        # Closing the group pops the top of the stack onto the Soutce.

        def __init__(self, source: Source):
            self.source = source

        def push(self, enable: bool, iftrigger: bool, ifpassthru: bool,
                    startlinetoks: Tokens) -> None:
            self.append(self.source.ifstate)
            self.source.ifstate = Source.IfState(enable, iftrigger, ifpassthru, startlinetoks)

        def pop(self) -> None:
            state = self.source.ifstate = super().pop()
            return state

    # ----------------------------------------------------------------------
    # parsegen()
    #
    # Parse an input string from top level or included source file.
    # ----------------------------------------------------------------------

    def parsegen(self, input: str = '', source=None,abssource=None) -> Iterator[LexToken]:
    #def parsegen(self, lines: Iterable[list[LexToken]] = None, source=None,abssource=None) -> Iterator[LexToken]:
        """Parse an input string.  Generate LexToken objects. """

        prep: PreProcessor = self.prep
        prep.currsource = self

        lex = self.lexer = prep.lexer.clone()
        lex.input(input, source=self)
        lex.lineno = 1
        # Tell prep.write() to begin a new source file.
        prep.changed_source = True
        yield lex.null()

        self.ifstack = self.IfStack(self)
        self.ifstate = self.IfState(True, True, False, top=True)
        if not input:
            return
        my_include_time_begin = clock()
        prep.include_times.append(FileInclusionTime(prep.macros['__FILE__'] if '__FILE__' in prep.macros else None, source, abssource, prep.include_depth))
        my_include_times_idx = len(prep.include_times) - 1
            

        # True until any non-whitespace output or anything with effects happens.
        self.at_front_of_file = True
        # True if auto pragma once still a possibility for this #include
        # (it may have been disabled by the subclass constructor).
        auto_pragma_once_possible = self.prep.auto_pragma_once_enabled
        # =(MACRO, 0) means #ifndef MACRO or #if !defined(MACRO) seen, =(MACRO,1) means #define MACRO seen
        self.include_guard = None
        self.prep.on_potential_include_guard(None)

        chunk: Tokens = Tokens()          # List of token to be scanned for macro replacement.
        lines = self.group_lines(input)
        for x in lines:
            all_whitespace = True
            skip_auto_pragma_once_possible_check = False
            # Handle comments
            for i,tok in enumerate(x):
                if tok.type in prep.t_COMMENT:
                    if not prep.on_comment(tok):
                        tok.value = ' '
                        tok.type = TokType.CPP_WS
            # Skip over whitespace
            for i,tok in enumerate(x):
                if not tok.type.ws:
                    all_whitespace = False
                    break
            output_and_expand_line = True
            output_unexpanded_line = False
            if tok.type is TokType.CPP_DIRECTIVE:
                self.dir = Directive(self, x)
                self.directive = x[i:]
                precedingtoks = [ tok ]
                output_and_expand_line = False
                try:
                    # Preprocessor directive      

                    # Expand and yield what we have collected so far.
                    # Last token cannot be a macro call because any '(' does
                    # not follow any whitespace
                    chunk = prep.macros.expand(chunk)
                    yield from chunk
                    chunk = Tokens()

                    i += 1
                    while i < len(x) and x[i].type.ws:
                        precedingtoks.append(x[i])
                        i += 1
                    dirtokens = prep.tokenstrip(x[i:])
                    if dirtokens:
                        name = dirtokens[0].value
                        args = prep.tokenstrip(dirtokens[1:])
                    
                        #if name in ('elif', 'else', 'endif'): prep.nesting -= 1
                        # Get the directive arguments, with newlines replacing line continuations
                        argvalue = [tok.value for tok in args]
                        prep.log.write("# %s %s" % (dirtokens[0].value, "".join(argvalue)), token=dirtokens[0])
                        #if name in ('elif', 'else'): prep.nesting += 1

                        handling = prep.on_directive_handle(dirtokens[0],args, self.ifstate.ifpassthru, precedingtoks)
                        # Did not raise OutputDirective.
                        assert handling == True or handling == None

                    else:
                        # Null directive.
                        name = ""
                        args = []
                        raise OutputDirective(Action.IgnoreAndRemove)

                    res = self.dir(handling)
                    if res: yield from res
                    else: yield x[-1]
                    ...
                    """
                    ##if name == 'define':
                    ##    self.at_front_of_file = False
                    ##    if enable:
                    ##        if self.include_guard and self.include_guard[1] == 0:
                    ##            if self.include_guard[0] == args[0].value and len(args) == 1:
                    ##                self.include_guard = (args[0].value, 1)
                    ##                # If ifpassthru is only turned on due to this include guard, turn it off
                    ##                if prep.ifpassthru and not self.ifstack[-1].ifpassthru:
                    ##                    prep.ifpassthru = False
                    ##        prep.define(args)
                    ##        macro = prep.macros[args[0].value]
                    ##        #self.writedebug(self.verbose > 1 and repr(macro) or macro.show(), token=dirtokens[0], indent=5)
                    ##        if handling is None:
                    ##            yield from x

                    ##elif name == 'include':
                    ##    if enable:
                    ##        oldfile = prep.macros['__FILE__'] if '__FILE__' in prep.macros else None
                    ##        if args and args[0].value != '<' and args[0].type != prep.t_STRING:
                    ##            args = self.tokenstrip(prep.macros.expand(args))
                    ##        # print('***', ''.join([x.value for x in args]), file = sys.stderr)
                    ##        yield from prep.include(args, x)
                    ##        if oldfile is not None:
                    ##            prep.macros['__FILE__'] = oldfile
                    ##        self.source = abssource

                    ##elif name == 'undef':
                    ##    self.at_front_of_file = False
                    ##    if enable:
                    ##        prep.undef(args)
                    ##        if handling is None:
                    ##            yield from x

                    ##elif name == 'ifdef':
                    ##    prep.nesting += 1
                    ##    self.at_front_of_file = False
                    ##    self.ifstack.append(self.IfState(enable,iftrigger, prep.ifpassthru,x))
                    ##    if enable:
                    ##        prep.ifpassthru = False
                    ##        if not args[0].value in prep.macros:
                    ##            res = prep.on_unknown_macro_in_defined_expr(args[0])
                    ##            if res is None:
                    ##                prep.ifpassthru = True
                    ##                self.ifstack[-1].rewritten = True
                    ##                raise OutputDirective(Action.IgnoreAndPassThrough)
                    ##            elif res is True:
                    ##                iftrigger = True
                    ##            else:
                    ##                enable = False
                    ##                iftrigger = False
                    ##        else:
                    ##            iftrigger = True

                    ##elif name == 'ifndef':
                    ##    prep.nesting += 1
                    ##    if not self.ifstack and self.at_front_of_file:
                    ##        prep.on_potential_include_guard(args[0].value)
                    ##        self.include_guard = (args[0].value, 0)
                    ##    self.at_front_of_file = False
                    ##    self.ifstack.append(self.IfState(enable,iftrigger, prep.ifpassthru,x))
                    ##    if enable:
                    ##        prep.ifpassthru = False
                    ##        if args[0].value in prep.macros:
                    ##            enable = False
                    ##            iftrigger = False
                    ##        else:
                    ##            res = prep.on_unknown_macro_in_defined_expr(args[0])
                    ##            if res is None:
                    ##                prep.ifpassthru = True
                    ##                self.ifstack[-1].rewritten = True
                    ##                raise OutputDirective(Action.IgnoreAndPassThrough)
                    ##            elif res is True:
                    ##                enable = False
                    ##                iftrigger = False
                    ##            else:
                    ##                iftrigger = True

                    ##elif name == 'if':
                    ##    prep.nesting += 1
                    ##    if not self.ifstack and self.at_front_of_file:
                    ##        if args and args[0].value == '!' and args[1].value == 'defined':
                    ##            n = 2
                    ##            if args[n].value == '(': n += 1
                    ##            self.on_potential_include_guard(args[n].value)
                    ##            self.include_guard = (args[n].value, 0)
                    ##    self.at_front_of_file = False
                    ##    self.ifstack.append(self.IfState(enable,iftrigger, prep.ifpassthru,x))
                    ##    if enable:
                    ##        iftrigger = False
                    ##        prep.ifpassthru = False
                    ##        result, rewritten = self.evalexpr(args)
                    ##        if rewritten is not None:
                    ##            x = x[:i+2] + rewritten + [x[-1]]
                    ##            x[i+1] = copy.copy(x[i+1])
                    ##            x[i+1].type = prep.t_SPACE
                    ##            x[i+1].value = ' '
                    ##            prep.ifpassthru = True
                    ##            self.ifstack[-1].rewritten = True
                    ##            raise OutputDirective(Action.IgnoreAndPassThrough)
                    ##        if not result:
                    ##            enable = False
                    ##        else:
                    ##            iftrigger = True

                    ##elif name == 'elif':
                    ##    self.at_front_of_file = False
                    ##    if self.ifstack:
                    ##        if self.ifstack[-1].enable:     # We only pay attention if outer "if" allows this
                    ##            if enable and not prep.ifpassthru:         # If already true, we flip enable False
                    ##                enable = False
                    ##            elif not iftrigger:   # If False, but not triggered yet, we'll check expression
                    ##                result, rewritten = self.evalexpr(args)
                    ##                if rewritten is not None:
                    ##                    enable = True
                    ##                    if not prep.ifpassthru:
                    ##                        # This is a passthru #elif after a False #if, so convert to an #if
                    ##                        x[i].value = 'if'
                    ##                    x = x[:i+2] + rewritten + [x[-1]]
                    ##                    x[i+1] = copy.copy(x[i+1])
                    ##                    x[i+1].type = prep.t_SPACE
                    ##                    x[i+1].value = ' '
                    ##                    prep.ifpassthru = True
                    ##                    self.ifstack[-1].rewritten = True
                    ##                    raise OutputDirective(Action.IgnoreAndPassThrough)
                    ##                if prep.ifpassthru:
                    ##                    # If this elif can only ever be true, simulate that
                    ##                    if result:
                    ##                        newtok = copy.copy(x[i+3])
                    ##                        newtok.type = prep.t_INTEGER
                    ##                        newtok.value = prep.t_INTEGER_TYPE(result)
                    ##                        x = x[:i+2] + [newtok] + [x[-1]]
                    ##                        raise OutputDirective(Action.IgnoreAndPassThrough)
                    ##                    # Otherwise elide
                    ##                    enable = False
                    ##                elif result:
                    ##                    enable  = True
                    ##                    iftrigger = True
                    ##    else:
                    ##        self.on_error(dirtokens[0].source,dirtokens[0].lineno,"Misplaced #elif")
                            
                    ##elif name == 'else':
                    ##    self.at_front_of_file = False
                    ##    if self.ifstack:
                    ##        if self.ifstack[-1].enable:
                    ##            if prep.ifpassthru:
                    ##                enable = True
                    ##                raise OutputDirective(Action.IgnoreAndPassThrough)
                    ##            if enable:
                    ##                enable = False
                    ##            elif not iftrigger:
                    ##                enable = True
                    ##                iftrigger = True
                    ##    else:
                    ##        self.on_error(dirtokens[0].source,dirtokens[0].lineno,"Misplaced #else")

                    ##elif name == 'endif':
                    ##    self.at_front_of_file = False
                    ##    if self.ifstack:
                    ##        oldifstackentry = self.ifstack.pop()
                    ##        enable = oldifstackentry.enable
                    ##        iftrigger = oldifstackentry.iftrigger
                    ##        ifpassthru = oldifstackentry.ifpassthru
                    ##        #self.writedebug("(%s:%d %s)" % (oldifstackentry.startlinetoks[0].source, oldifstackentry.startlinetoks[0].lineno, "".join([n.value for n in oldifstackentry.startlinetoks])))
                    ##        skip_auto_pragma_once_possible_check = True
                    ##        if oldifstackentry.rewritten:
                    ##            raise OutputDirective(Action.IgnoreAndPassThrough)
                    ##    else:
                    ##        self.on_error(dirtokens[0].source,dirtokens[0].lineno,"Misplaced #endif")

                    ##elif name == 'pragma' and args[0].value == 'once':
                    ##    if enable:
                    ##        self.include_once[self.source] = None

                    ##elif name == 'line':

                    ##    ...

                    ##elif enable:
                    ##    # Unknown preprocessor directive
                    ##    output_unexpanded_line = (prep.on_directive_unknown(dirtokens[0], args, prep.ifpassthru, precedingtoks) is None)
                    """

                except OutputDirective as e:
                    if e.action == Action.IgnoreAndPassThrough:
                        output_unexpanded_line = True
                    elif e.action == Action.IgnoreAndRemove:
                        pass
                    else:
                        assert False

            # If there is ever any non-whitespace output outside an include guard, auto pragma once is not possible
            if not skip_auto_pragma_once_possible_check and auto_pragma_once_possible and not self.ifstack and not all_whitespace and prep.include_depth > 1:
                auto_pragma_once_possible = False
                prep.log.write(f"Determined that #include \"{self.filename}\" is not entirely wrapped in an include guard macro, disabling auto-applying #pragma once", token=x[-1])
                
            if output_and_expand_line or output_unexpanded_line:
                if not all_whitespace:
                    self.at_front_of_file = False

                # Normal text
                if self.ifstate.enable:
                    if output_and_expand_line:
                        chunk.extend(x)
                    elif output_unexpanded_line:
                        for tok in prep.macros.expand(chunk):
                            yield tok
                        chunk = []
                        for tok in x:
                            yield tok
                else:
                    # Need to extend with the same number of blank lines
                    i = 0
                    while i < len(x):
                        if x[i].type not in prep.t_WS:
                            del x[i]
                        else:
                            i += 1
                    chunk.extend(x)

            lastline = x

        chunk = prep.macros.expand(chunk)
        yield from chunk

        for i in self.ifstack[1:]:
            prep.on_error(i.startlinetoks[0].source, i.startlinetoks[0].lineno, "Unterminated " + "".join([n.value for n in i.startlinetoks]))
        if auto_pragma_once_possible and self.include_guard and self.include_guard[1] == 1:
            prep.log.write("Determined that #include \"%s\" is entirely wrapped in an include guard macro called %s, auto-applying #pragma once" % (self.source, self.include_guard[0]), token=lastline[-1])
            #self.include_once[self.source] = self.include_guard[0]
        elif prep.auto_pragma_once_enabled and self.filename not in prep.include_once:
            prep.log.write(f"Did not auto apply #pragma once to this file due to auto_pragma_once_possible={auto_pragma_once_possible}, self.include_guard={self.include_guard}", token=lastline[-1])
        my_include_time_end = clock()
        prep.include_times[my_include_times_idx].elapsed = my_include_time_end - my_include_time_begin
        prep.include_depth -= 1

        # Tell prep.write() to resume the old source file (if any).
        prep.currsource = self.parent
        if self.parent:
            prep.changed_source = True
            yield self.parent.lexer.null()

    def __repr__(self) -> str:
        return repr(self.filename)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

