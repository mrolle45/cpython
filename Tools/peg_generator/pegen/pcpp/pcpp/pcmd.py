#!/usr/bin/python
# Python C99 conforming preprocessor command line
# (C) 2017-2020 Niall Douglas http://www.nedproductions.biz/
# Started: March 2017

from __future__ import annotations

import sys, argparse, traceback, os, copy, io, re
import codecs

if __name__ == '__main__' and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

from pcpp.preprocessor import Preprocessor
from pcpp.source import OutputDirective, Action, SourceFile
from pcpp.writer import Writer, OutPosFlag, OutPosEnter, OutPosLeave

version='1.30'

__all__ = []

class FileAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest)[0] == sys.stdin:
            items = []
        else:
            items = copy.copy(getattr(namespace, self.dest))
        items += [argparse.FileType('rt')(value)
                  for value in values]
        setattr(namespace, self.dest, items)

class CmdPreprocessor(Preprocessor):
    """
    Specialized Preprocessor subclass for use as a command-line tool.  It is
    constructed from an argv list, such as found in sys.argv.

    All of the methods in parser.PreProcessorHooks.on_xxx(), except
    on_error(), are overridden, as they depend on various items in argv.
    """

    args: Namespace

    def __init__(self, argv):
        if len(argv) < 2:
            argv = [argv[0], '--help']

        unknowns: list[str]         # Unknown argument names

        # Make all files open with UTF-8 encodings by default.
        # This is a hack, because it relies on an undocumented member of
        #   a library extension module.
        # After Python 3.15, UTF-8 will always be the default, so this
        #   won't be necessary.  See PEP 686 https://peps.python.org/pep-0686/.
        if sys.version_info < (3, 15):
            import _locale
            _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])
        args, unknowns = self.make_args(argv)
        for arg in unknowns:
            print(f"NOTE: Argument '{arg}' not known, ignoring!",
                  file=sys.stderr)

        if args.gcc and args.clang:
            raise Exception("Cannot have both --gcc and --clang.")

        self.verbose = args.v
        if args.cplusplus:
            self.cplus_ver = args.cplusplus + 2000

        if args.c_ver:
            self.c_ver = {
                                99: 1999,
                                11: 2011,
                                17: 2017,
                                23: 2023,
                                }[args.c_ver]

        self.clang = args.clang
        self.gcc = args.gcc
        self.comments = args.passthru_comments
        self.debug = args.debug
        self.diag = args.diag
        if args.tabstop: self.tabstop = args.tabstop
        self.passthru_unfound_includes = args.passthru_unfound_includes
        self.passthru_unknown_exprs = args.passthru_unknown_exprs
        if args.nevers: self.nevers = frozenset(args.nevers)
        if args.line_directive: self.line_directive = args.line_directive
        if (self.line_directive is not None
            and self.line_directive.lower() in ('nothing', 'none', '')
            ):
            self.line_directive = None
        if args.compress: self.compress = 2
        if args.auto_pragma_once_disabled:
            self.auto_pragma_once_enabled = False

        if args.passthru_includes is not None:
            self.passthru_includes = re.compile(args.passthru_includes)
        self.passthru_defines = args.passthru_defines
        super().__init__()
        
        # Override Preprocessor instance variables
        self.startup_define("__PCPP_VERSION__ " + version)
        self.startup_define("__PCPP_ALWAYS_FALSE__ 0")
        self.startup_define("__PCPP_ALWAYS_TRUE__ 1")
        if args.assume_input_encoding is not None:
            enc: str = args.assume_input_encoding
            if enc == 'auto': enc = None
            self.input_encoding = enc
            #if len(args.inputs) == 1:
            #    # Reopen our input files with the appropriate encoding
            #    _ = self.on_file_open(False, args.inputs[0].name)
            #    args.inputs[0].close()
            #    args.inputs[0] = _
            if args.output_encoding is None:
                args.output_encoding = enc
        if args.output_encoding:
            enc = args.output_encoding
            if codecs.lookup(enc).name != codecs.lookup('utf8').name:
                00
            # Reopen our output file with the appropriate encoding
            _ = io.open(args.output.name, 'w',
                        encoding=args.output_encoding)
            args.output.close()
            args.output = _
            if args.write_bom:
                args.output.write('\ufeff')
        
        # My own instance variables
        self.bypass_ifpassthru = False
        self.potential_include_guard = None

        # `defines` includes -U names with a trailing '-' and no '='
        if args.defines:
            for d in args.defines:
                if '=' not in d:
                    # Could be an undefine.
                    if d.endswith('-'):
                        self.startup_line(f"#undef {d[:-1]}")
                        continue
                    d += '=1'
                d = d.replace('=', ' ', 1)
                self.startup_define(d)
        for d in args.includes:
            self.add_path(d)

        msg = (f"Preprocessing "
               f"{' + '.join(repr(f.name) for f in args.inputs)}")
        if self.cplus_ver:
            msg += f" as C++ {self.cplus_ver % 100}."
        else:
            msg += f" as C {self.c_ver % 100}."
        if self.emulate:
            msg += (f"  Emulate {'GCC, CLANG'.split()[bool(self.clang)]}"
                    f" version {self.emulate}"
                    f"{' with GNU extensions' * bool(args.gnu)}"
                    ".")
        print(msg)
        out = args.output
        if out is not sys.stdout:
            print(f"Output file {os.path.abspath(out.name)!r}")
        if self.log.enable:
            print(f"Debug log written to {self.log.enable!r}.")
        try:
            for i in args.inputs:
                self.startup_line(f'#include "{i.name}"')
                SourceFile.openfile(i, self)
            self.write(self.parse(), args.output)
        except:
            print(traceback.print_exc(10), file=sys.stderr)
            print("\nINTERNAL PREPROCESSOR ERROR AT AROUND "
                  f"{self.currsource}:{self.currsource.lexer.lineno}, "
                  "FATALLY EXITING NOW\n",
                  file=sys.stderr)
            sys.exit(-99)
        finally:
            for i in args.inputs:
                i.close()
            if args.output != sys.stdout:
                args.output.close()
            self.log.writelog()

        if args.time:
            print("\nTime report:")
            print("============")
            for n in range(0, len(self.include_times)):
                if n == 0:
                    print("top level: %f seconds"
                          % self.include_times[n].elapsed)
                elif self.include_times[n].depth == 3:
                    print("\n  %s: %f seconds (%.2f%%)"
                          % (self.include_times[n].included_path,
                             self.include_times[n].elapsed,
                             100 * self.include_times[n].elapsed
                                / self.include_times[0].elapsed)
                          )
                else:
                    print("%s%s: %f seconds"
                          % ('  ' * (self.include_times[n].depth - 2),
                             self.include_times[n].included_path,
                             self.include_times[n].elapsed)
                          )
            once_objs = [file.once for file in self.files.values()
                          if file.once]
            if once_objs:
                print("\nPragma once files (including heuristically applied):")
                print("====================================================")
                for once in sorted(once_objs,
                                key=lambda once: (os.path.basename(once.file.filename), once.file.filename)):
                    hint = once.guard or '#pragma'
                    print(f" {once.file.filename} ({hint})")

            print()
        if args.filetimes:
            print('"Total seconds","Self seconds","File size","File path"',
                  file=args.filetimes)
            filetimes = {}
            currentfiles = []
            for n in range(0, len(self.include_times)):
                while self.include_times[n].depth < len(currentfiles):
                    currentfiles.pop()
                if self.include_times[n].depth > len(currentfiles) - 1:
                    currentfiles.append(
                        self.include_times[n].included_abspath
                        )
                path = currentfiles[-1]
                if path in filetimes:
                    filetimes[path][0] += self.include_times[n].elapsed
                    filetimes[path][1] += self.include_times[n].elapsed
                else:
                    filetimes[path] = [self.include_times[n].elapsed,
                                       self.include_times[n].elapsed]
                if self.include_times[n].elapsed > 0 and len(currentfiles) > 1:
                    filetimes[currentfiles[-2]][1] -= (
                              self.include_times[n].elapsed)
            filetimes = [(v[0],v[1],k) for k,v in filetimes.items()]
            filetimes.sort(reverse=True)
            for t,s,p in filetimes:
                print(('%f,%f,%d,"%s"'
                      % (t, s, os.stat(p).st_size, p)),
                      file=args.filetimes)

    def write(self, toks: TokIter, oh=sys.stdout):
        writer = Writer(self)
        writer.write(toks, oh)
        return

    def make_args(self, argv) -> tuple[Namespace, list[str]]:
        """
        Read the command line arguments with the argparse module.  Return a
        Namespace object from the known args, and a list of unknown items in
        argv.
        """
        argp = argparse.ArgumentParser(prog='pcpp',
            description=
    '''A pure universal Python C (pre-)preprocessor implementation very useful
    for pre-preprocessing header only C++ libraries into single file includes
    and other such build or packaging stage malarky.''',
            epilog=
    '''Note that so pcpp can stand in for other preprocessor tooling, it
    ignores any arguments it does not understand.''')
        add = argp.add_argument
        add('inputs', metavar='input', default=[sys.stdin], nargs='*',
            type=argparse.FileType('rt', encoding='utf-8-sig'),
            help='Files to preprocess (use \'-\' for stdin)'
            )
        add('-o', dest='output', metavar='path', default="-", nargs='?',
            type=argparse.FileType('wt'),
            help='Output to a file instead of stdout',
            )
        add('-D', dest='defines', metavar='MACRO[=VAL]',
            action='append',
            help='''Predefine MACRO as a macro with value VAL.  
                      If just MACRO is given, VAL is taken to be 1.''',
            )
        add('-U', dest='defines', metavar='macro',
            action=UndefAction,
            help='''Pre-undefine name as a macro.  Overrides any earlier -D of
                 the name.''',
            )
        add('-N', dest='nevers', metavar='macro', action='append',
            help='''
                Never define or undef name as a macro, but pass through the
                directive.
                ''',
            )
        add('-I', dest='includes', metavar='path',
            action='append', help="Path to search for unfound #include's"
            )

        # Group for passthru options.
        addg = self.add_arg_group_func(
            argp,
            '''Options for partial preprocessing and passing the directives to
            the output.
            ''',
            )
        addg('--passthru', dest='passthru', action='store_true',
             help='''
                Pass through everything unexecuted except for #include and
                include guards (which need to be the first thing in an
                #include file
                ''',
            )
        addg('--passthru-defines', dest='passthru_defines',
             action='store_true',
             help='''
                Pass through but still execute #defines and #undefs if not
                always removed by preprocessor logic
                ''',
             )
        addg('--passthru-unfound-includes',
             dest='passthru_unfound_includes', action='store_true',
             help='Pass through #includes not found without execution',
             )
        addg('--passthru-unknown-exprs', dest='passthru_unknown_exprs',
             action='store_true',
             help='''
                Unknown macros in expressions cause preprocessor logic to be
                passed through instead of being executed by treating them as 0
                ''',
             )
        addg('--passthru-comments', dest='passthru_comments',
             action='store_true',
             help='''
                Pass through comments unmodified. But with GCC emulation,
                (1) they are NOT whitespace, and (2) C++-style comments in a macro
                replacement are converted to C-style comments.
                ''',
             )
        addg('--passthru-magic-macros', dest='passthru_magic_macros',
             action='store_true',
             help='Pass through double underscore magic macros unmodified.'
             )
        addg('--passthru-includes', dest='passthru_includes',
             metavar='<regex>', default=None,
             help='''
                Regular expression for which #include file names (if they are
                found) to pass through. The files are always executed, as far
                as directives are concerned, but the text lines are ignored.
                ''',
             )

        add('--disable-auto-pragma-once', dest='auto_pragma_once_disabled',
            action='store_true', default=False,
            help='''
                Disable the heuristics which auto apply #pragma once to
                #include files wholly wrapped in an obvious include guard 
                macro
                ''',
            )
        add('--line-directive', dest='line_directive', metavar='form',
            default='#line', nargs='?',
            help="""
                Form of line directive to use, defaults to '#line'.  Specify
                'none' or 'nothing' or leave blank, to disable output of line
                directives.
                """,
            )
        add('--debug', dest='debug', metavar='path', type=str,
            const="pcpp_debug.log", nargs='?',
            help='''
                Generate a log file for logging execution.
                Default = pcpp_debug.log.
                ''',
            )
        add('--diag', dest='diag', metavar='path', nargs='?',
            type=argparse.FileType('wt'),
            help='Write diagnostics to a file instead of stderr.',
            )
        add('--time', dest='time', action='store_true',
            help='Print the time it took to #include each file',
            )
        add('--filetimes', dest='filetimes', metavar='path', nargs='?',
            type=argparse.FileType('wt'), default=None,
            help='''
                Write CSV file with time spent inside each included file,
                both inclusive and exclusive.
                ''',
            )
        add('--compress', dest='compress', action='store_true',
            help='Make output as small as possible')

        # Exclusive group for C or C++ language options.
        addg = self.add_arg_group_func(
            argp, 'Language standard (choose one)', 'Default is C99.',
            exclusive=True,
            )
        addg('--c++', dest='cplusplus', nargs='?', const=14, type=int,
             choices=[11, 14, 17, 20, 23, ], metavar='VER', 
             help='''
                Use given ISO C++ standard. Default = 14. Defines __cplusplus.
                ''',
             )
        addg('--c', dest='c_ver', nargs='?', const=99, type=int,
             choices=[99, 11, 17, 23, ], metavar='VER', 
             help='Use given ISO C standard. Default = 99.',
             )

        # Exclusive group for GCC emulation options.
        addg = self.add_arg_group_func(
            argp, 'external tool emulation',
            '''
            Make output similar to "gcc -E" "clang -E".,
            ''',
            )
        addg('--gcc', dest='gcc', nargs='?', const=10, type=int,
             metavar='GCCVER',
             help='''
             Emulate "gcc -E" with gcc version GCCVER (default 10). Defines
             __GNUC__ = GCCVER.
             ''')
        addg('--clang', dest='clang', nargs='?', const='18.1',
             metavar='MAJOR.MINOR',
             help='''
             Emulate "clang -E" with clang version MAJOR.MINOR (default 18.1).
             Defines __clang__ = 1, __clang_major__, and __clang_minor__.'''
             )

        addg('--gnu', dest='gnu', action=BooleanOptionalAction,
             help='''
                GNU extensions.  Requires --gcc or --clang.  With --no-gnu (the
                default), using -std=c[VER] or -std=c++[VER].  Defines __STDC__
                = 1.  With --gnu, using -std=gnu[VER] or -std=gnu++[VER].')
                ''')
        add('--trigraphs', dest='trigraphs', action=BooleanOptionalAction,
            help='''
                Process trigraphs. Default if --gcc or if --c++ < 17.
                Defines __PCPP_TRIGRAPHS__ = 1.
                ''',
            )
        add('--tabstop', dest='tabstop', type=int, nargs='?', const=8,
            help='''
                Expand leading tabs with given tabstop width. Default = 8.
                0 to not expand.
                ''',
            )
        add('--assume-input-encoding', dest='assume_input_encoding',
            metavar='<encoding>', default=None,
            help='''
                The text encoding to assume inputs are using. Default = utf-8.
                'auto' supplies None, which uses utf-8 unless the PCPP comes
                with a hook method to detect the encoding from the file
                contents.
                ''',
            )
        add('--output-encoding', dest='output_encoding', metavar='<encoding>',
            default=None,
            help='''
                The text encoding to use when writing files.
                Default = above --assume-input-encoding or utf-8.
                ''',
            )
        add('--write-bom', dest='write_bom', action='store_true',
            help='''
                Prefix any output with a Unicode BOM, equivalent to an
                encoding of utf-8-sig.
                ''',
            )
        add("-v", action="count", default=0,
            help="Print more log details; repeat for still more.",
            )
        add('--version', action='version', version='pcpp ' + version)
        return argp.parse_known_args(argv[1:])

    def add_arg_group_func(
            self, argp: ArgumentParser, title = None, description = None, *,
            exclusive: bool = False
            ) -> Callable[str, ...]:
        """ Defines an argument group, which may be exclusive.
        Returns a function which calls the group.add_argument() method.
        """
        group = argp.add_argument_group(title, description)
        if exclusive:
            group = group.add_mutually_exclusive_group()
        return group.add_argument

    def on_include_not_found(self, is_malformed, is_system_include, curdir,
                             includepath):
        if self.passthru_unfound_includes:
            raise OutputDirective(Action.IgnoreAndPassThrough)
        return super().on_include_not_found(
            is_malformed, is_system_include, curdir, includepath)

    def on_unknown_macro_in_defined_expr(self, tok):
        if self.passthru_unknown_exprs:
            return None  # Pass through as expanded as possible
        return super().on_unknown_macro_in_defined_expr(tok)
        
    def on_unknown_macro_in_expr(self,ident):
        if self.passthru_unknown_exprs:
            return None  # Pass through as expanded as possible
        return super().on_unknown_macro_in_expr(ident)
        
    def on_unknown_macro_function_in_expr(self,ident):
        if self.passthru_unknown_exprs:
            return None  # Pass through as expanded as possible
        return super().on_unknown_macro_function_in_expr(ident)
        
    def on_directive_handle(self,
                            directive: PpTok,
                            toks: Tokens,
                            ifpassthru: bool,
                            precedingtoks:Tokens,
                            ):
        """
        Called when there is any directive with a name (not a bare #).
        
        Return True to execute and remove from the output,
        raise OutputDirective to pass through or remove without execution,
        or return None to execute AND pass through to the output
        (this only works for #define, #undef).
        
        The default returns True (execute and remove from the output).
        A subclass could override this.

        directive is the directive name,
        toks is the tokens after the directive,
        ifpassthru is whether we are in passthru mode,
        precedingtoks is the tokens preceding the directive from the # token
            until the directive name.
        """
        if ifpassthru:
            ifsect = directive.dir.ifsect
            if directive.value in 'if elif'.split():
                if any(tok.value == '__PCPP_ALWAYS_FALSE__'
                       or tok.value == '__PCPP_ALWAYS_TRUE__'
                       for tok in toks):
                    ifsect.bypass_ifpassthru = True
            if (not ifsect.bypass_ifpassthru
                and directive.value in 'define undef'.split()
                ):
                if toks[0].value != self.potential_include_guard:
                    raise OutputDirective(Action.IgnoreAndPassThrough)
        if (directive.value in 'define undef'.split()):
            if self.nevers:
                if toks[0].value in self.nevers:
                    raise OutputDirective(Action.IgnoreAndPassThrough)
            if self.passthru_defines:
                super().on_directive_handle(
                    directive,toks,ifpassthru,precedingtoks)
                return None  # Pass through where possible
        return super().on_directive_handle(
            directive,toks,ifpassthru,precedingtoks)

    def on_directive_unknown(self,directive,toks,ifpassthru,precedingtoks):
        if ifpassthru:
            return None  # Pass through
        return super().on_directive_unknown(
            directive,toks,ifpassthru,precedingtoks)

    def on_potential_include_guard(self, macro: str):
        self.potential_include_guard = macro
        return super().on_potential_include_guard(macro)

    def on_comment(self,tok):
        if self.comments:
            return True  # Pass through
        return super().on_comment(tok)

class UndefAction(argparse.Action):
    """
    An action which stores its argument followed by a "-" to distinguish it
    from a define. """
    def __init__(self, **kwds): super().__init__(**kwds)

    def __call__(self, parser, namespace, values, option_string) -> None:
        getattr(namespace, self.dest).append(values + "-")

try: from argparse import BooleanOptionalAction
except ImportError:
    # Before Python 3.9.  This is copied from the 3.9 library argparse.py.
    class BooleanOptionalAction(argparse.Action):
        def __init__(self,
                     option_strings,
                     dest,
                     default=None,
                     type=None,
                     choices=None,
                     required=False,
                     help=None,
                     metavar=None):

            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)

                if option_string.startswith('--'):
                    option_string = '--no-' + option_string[2:]
                    _option_strings.append(option_string)

            if help is not None and default is not None:
                help += " (default: %(default)s)"

            super().__init__(
                option_strings=_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar)

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string in self.option_strings:
                setattr(namespace, self.dest,
                        not option_string.startswith('--no-'))

        def format_usage(self):
            return ' | '.join(self.option_strings)


def main(exit: bool = True):
    try:
        p = CmdPreprocessor(sys.argv)
        p.finish()
        if exit:
            sys.exit(p.return_code)
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    p = CmdPreprocessor(sys.argv)
    sys.exit(p.return_code)
