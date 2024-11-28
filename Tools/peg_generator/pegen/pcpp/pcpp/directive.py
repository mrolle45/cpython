""" Module directive.py.
Manages a directive in a source file.
The Directive class holds and executes one directive.
"""

from __future__ import annotations

__all__ = 'Directive, Action, OutputDirectivea'.split()

from operator import attrgetter, methodcaller
import traceback

from pcpp.common import *
from pcpp.tokens import Tokens, TokIter, TokenSep, tokenstrip, filt_line
from pcpp.dircondition import FileSection
from pcpp.writer import OutPosMove

class Directive:
    """ A single directive in a source file.
    Will consume the tokens in the entire source line.
    Conditional inclusion directives will consume an entire if-section.
    Can execute the directive, generating any tokens to be passed through.
    """
    cond: bool = False              # True for conditional-inclusion directive.
    nest: int = 0                   # Change in conditional nesting level.

    def __init__(self,
                 dirtok: PpTok,             # The '#' token.
                 ):
        self.source = source = dirtok.source
        self.prep = prep = source.prep
        #self.line = line = Tokens(line)
        #self.moretoks = moretoks
        self.line = dirtok.line
        self.TokType = prep.TokType
        self.dirtok = dirtok
        # Parse the line into # (name args?)?
        #print(line)
        self.precedingtoks, self.nametok, self.args = self.parse()
        if not self.nametok:
            # Null directive.  Do nothing.
            self.handler = handlers.get('')
            return

        self.name = self.nametok.value
        self.handler = handler = handlers.get(self.name)
        if not handler:
            # Unknown name, which could be an integer for #line.
            if self.nametok.type.int:
                self.handler = handlers['line']

        if handler and handler.dirattrs:
            self.__dict__.update(handler.dirattrs)

    def __call__(self,
                 moretoks: TokIter,         # All tokens afterwards in source.
                 ifsect: Section = None,    # Current section, for conditional.
                 ) -> Iterable[PpTok]:
        self.moretoks = moretoks
        self.ifsect = ifsect or FileSection()
        # Catch OutputDirective exceptions
        try:
            prep = self.prep
            if not self.handler:
                # Unknown directive.
                if not self.nametok: return
                res = prep.on_directive_unknown(
                    self.nametok, self.args, self.ifsect.passthru,
                    self.precedingtoks)
                # May have raised OutputDirective, which will go to the caller.
                # Otherwise, result is either True or None.
                if not res:
                    raise OutputDirective(Action.IgnoreAndPassThrough)
            else:
                if self.nametok:
                    self.nametok.dir = self
                    self.handling = prep.on_directive_handle(
                        self.nametok, self.args, self.ifsect.passthru,
                        self.precedingtoks)
                    # Did not raise OutputDirective.
                    assert self.handling in (True, None)
                res = self.handler(self)
                if res: yield from res
                elif self.check_macro_args(in_args=False):
                    # In a macro function call, before the opening '('.
                    # This will keep the macro from finding a '('.
                    yield self.dirtok.make_null()
        except OutputDirective as e:
            if e.action == Action.IgnoreAndPassThrough:
                if prep.clang:
                    return
                tok = self.dirtok.copy(
                    value=str(self.line), type=prep.TokType.CPP_PASSTHRU)
                yield tok
        except BaseException as e:
            traceback.print_exc()
            print("Ignoring the exception.\a")
        return None

    def parse(self) -> Tuple[Tokens, PpTok | None, Tokens]:
        """ Decompose the entire line (after the #) into component parts.
        1. Precedingtoks: From the # to before the name.
        2. Name token: None if a null directive.
        3. Args: the rest of the line, stripped.
            Includes the name token if it is an integer.
        """
        #tokiter = TokIter(self.line)
        tok: PpTok = None
        #next(tokiter)               # Skip past the #.
        precedingtoks: Tokens = Tokens()
        line = TokIter(self.line[1:])
        for nametok in line:
            if nametok.type.ws:
                precedingtoks.append(nametok)
                continue
            break
        else:
            # Null directive.
            nametok = None

        # The rest of the directive is the args (if any)
        args = tokenstrip(Tokens(line))
        if nametok and nametok.type.int:
            args.insert(0, nametok)
            args[1] = args[1].with_spacing()
        return precedingtoks, nametok, args

    ### Methods for all the different directive names, in alphabetical order...
    # They all generate any tokens to be passed through (mostly just newline).
    # Or possibly raise an OutputDirective exception.

    def on_define(self, check_once: bool = False) -> Iterator[PpTok]:
        if check_once:
            if self.source.once.guard == self.args[0].value:
                self.source.define_guard(self.dirtok)

        self.prep.define(TokIter(self.args))
        if self.handling is None:
            return iter(self.line)
        ## Make this an empty iterator
        #return
        #yield

    def on_elif(self) -> Iterator[PpTok]:
        # Start the next group.
        result = self.evalexpr()
        self.condition(result)
        ## Make this an empty iterator
        #return
        #yield

    def on_elifdef(self) -> Iterator[PpTok]:
        # Start the next group.
        res = self.defined(self.args[0])
        self.condition(res)
        ## Make this an empty iterator
        #return
        #yield

    def on_elifndef(self) -> Iterator[PpTok]:
        # Start the next group.
        res = self.defined(self.args[0])
        self.condition(res, invert=True)
        # Make this an empty iterator
        return
        yield

    def on_else(self) -> Iterator[PpTok]:
        """ #else directive, equivalent to #elif True,
        must be the last group.
        """
        self.ifsect.else_group(self)
        ## Make this an empty iterator
        #return
        #yield

    def on_endif(self) -> Iterator[PpTok]:
        oldstate = self.ifsect
        self.prep.skip_auto_pragma_once_possible_check = True
        if oldstate.passthru:
            raise OutputDirective(Action.IgnoreAndPassThrough)
        ## Make this an empty iterator
        #return
        #yield

    def on_error(self) -> Iterator[PpTok]:
        self.prep.on_error_token(self.line[0], str(self.line).rstrip())
        self.prep.return_code += 1
        ## Make this an empty iterator
        #return
        #yield

    def on_if(self, check_once: bool = False) -> Iterator[PpTok]:
        # Check for potential include guard.
        if check_once and self.args[0].value == '!':
            expr: Tokens = Tokens(self.args[1:])
            def check() -> PpTok | None:
                if expr[0].value != 'defined': return None
                if (expr[1].value == '(' and expr[2].type is self.TokType.CPP_ID
                        and expr[3].value == ')' and len(expr) == 4):
                    return expr[2]
                if expr[1].type is self.TokType.CPP_ID and len(expr) == 2:
                    return expr[1]
                return None
            guard = check()
            if guard:
                self.prep.on_potential_include_guard(guard)
                self.source.once.set_guard(guard)

        # Evaluate the condition in the current section.
        res = self.evalexpr()
        # Start the new section and first group.
        return self.begin_section(res)

    def on_ifdef(self) -> Iterator[PpTok]:
        res = self.defined(self.args[0])
        # Start the new section and first group.
        return self.begin_section(res)

    def on_ifndef(self, check_once: bool = False) -> Iterator[PpTok]:
        # Check for an include guard.
        if check_once:
            self.prep.on_potential_include_guard(self.args[0].value)
            self.source.once.set_guard(self.args[0].value)
        res = self.defined(self.args[0])
        # Start the new section and first group.
        return self.begin_section(res, invert=True)

    def on_include(self) -> Iterator[PpTok]:
        if self.check_macro_args():
            # clang doesn't suport this in a macro call.
            return

        if not self.args[0].type.hdr:
            # Neither H nor Q format.  Make expansion and parse that.
            name = Tokens(self.expand_args())
            lex = self.prep.lexer.clone()
            lex.input(str(name))
            lex.begin('INCLUDE')
            self.args = Tokens(lex.tokens())
        yield from self.prep.include(self.args, self.line)

    def on_line(self) -> Iterator[PpTok]:
        """
        A #line directive.  Gives a line number and optional file name.
        """
        # The strategy is:
        #   1. Don't change anything in the Source's lexer.  It will continue
        #      to generate tokens using line numbers in the source file,
        #      regardless of any #line directives in the file.
        #   2. Generate a location marker token.  This will reference a OutLoc
        #      object with the presumed line number and (possibly) file name.
        #      This will be passed on, ultimately, to the Preprocessor's
        #      Writer.
        #   3. The Writer will see the marker before the next token is seen,
        #      adjust its idea of the current location, and (possibly) output
        #      a line directive.

        args = self.expand_args()
        args = Tokens(args)
        if not args:
            return self.prep.on_error_token(
                self.line[0], "#line directive requires a line number")
        line = args[0]
        try: lineno = int(line.value)
        except ValueError:
            return self.prep.on_error_token(
                line, f"#line directive line number {line.value!r} "
                        "must be a digit sequence.")
        if not 0 < lineno < (1 << 31):
            return self.prep.on_error_token(
                line, f"#line directive line number {lineno} out of range.")
        lexer = self.source.lexer
        if len(args) > 1:
            file = args[1]
            if file.type is not self.TokType.CPP_STRING:
                return self.prep.on_error_token(
                    file,
                    "#line directive requires a string literal file name"
                    )
            filename = file.value[1:-1]
            lexer.filename = filename
        else:
            filename = None
        tok: PpTok = self.nametok
        move: MoveTok = tok.make_pos(
            OutPosMove, lineno=lineno, filename=filename)
        lexer.move = move
        lexer.source.set_move(move)
        yield move

    def on_pragma(self) -> Iterator[PpTok]:
        prep: Preprocessor = self.prep
        if self.check_macro_args():
            # clang doesn't suport this in a macro call.
            return
        if self.args[0].value == 'once':
            # Note, GCC recognizes 'once' only without macro expansion.
            if not self.ifsect.skip:
                self.source.file.once.pragma()
            if not prep.emulate:
                return
            # GCC writes spaces up to the 'once' arg, minus one.
            #if prep.emulate:
            #    arg = self.args[0]
            #    yield arg.copy(
            #        type=self.TokType.CPP_PASSTHRU, value='',
            #        colno = arg.colno - 1)
        else:
            line = self.line
            loc = line[0].loc
            if loc.colno > 1:
                loc = dataclasses.replace(loc, colno=1)
                line[0] = line[0].copy(loc=loc)
            for i in range(2):
                line[i] = line[i].without_spacing()
            if not line[2].spacing:
                line[2] = line[2].with_spacing()
            if prep.clang:
                line = Tokens(prep.macros.expand(TokIter(line),
                                                 origin=self.dirtok))
            # GCC writes "#pragma " + the args (unindented and unexpanded).
            line[0] = line[0].with_sep(TokenSep.create(indent=line[0].loc))

            yield self.dirtok.make_passthru(line)
            #yield from line

    def on_undef(self) -> Iterator[PpTok]:
        self.prep.undef(self.args)
        if self.handling is None:
            return iter(self.line)
        ## Make this an empty iterator
        #return
        #yield

    def on_warning(self) -> Iterator[PpTok]:
        self.prep.on_warning_token(self.line[0], str(self.line).rstrip())
        ## Make this an empty iterator
        #return
        #yield


    ### Other methods...

    def args_reqd(self, what: str) -> bool:
        """ Verify that at least one token exists in the args.
        Calls error() otherwise.
        """
        if self.args: return True
        self.error(f"Directive #{self.name} requires {what}.")
        return False

    def begin_section(self, cond: bool | None, invert: bool = False
                      ) -> TokIter:
        """
        Open a new section and its first group.  Return token generator for the
        entire section.
        """
        #self.source.sectstack.push()
        self.ifsect = self.ifsect.nest()
        self.condition(cond, invert)
        return self.ifsect.parsegen(self.source, self.moretoks)

    def evalexpr(self) -> bool | None:
        """ Evaluate a control expression for #if or #elif.
        The expression is in self.args as a Tokens, NOT yet macro expanded.
        It should be a constant integer expression after expansion.
        If it contains any identifiers, these are names which are not defined
        as macros, and the preprocessor has to decide what to do with them:
            either supply a boolean result, or have it passed through.
        If the current if-section is being skipped, the expression is not
        examined and the result is False.
        """

        if not self.args:
            return (0, None)
        result, partial_expansion = self.prep.evaluator.eval(
            self.args, origin=self.dirtok)
        if partial_expansion is not None:
            # partial_expansion is the expanded expression as far as it could
            # be expanded, and no presumed result was provided.
            # The response is to pass through the directive, with the
            # expanded expression.

            # Replace the current input line, which will be passed through.
            tokiter = iter(self.line)
            def newdir() -> Iterator[PpTok]:
                # The original line up to the args.
                count = 0
                for tok in tokiter:
                    yield tok
                    if tok.type.ws or not tok.value:
                        continue
                    count += 1
                    if count == 2:
                        break
                # The partial expnsion
                yield from partial_expansion
                # Ending newline
                yield self.line[-1]
            self.source.line[:] = newdir()
            self.ifsect.rewritten = True
            return None
        return bool(result)

    def defined(self, name: PpTok) -> bool | None:
        """ Is the name a defined macro.
        Return None to bail out.
        """
        n = name.value
        if n in self.prep.macros:
            return True
        else:
            return self.prep.on_unknown_macro_in_defined_expr(n)

    def condition(self, cond: bool | None, invert: bool = False):
        """ Handle the condition of an #(el)if((n)def).
        The cond is the actual or assumed result.  The assumed
        result occurs if an undefined macro is involved, and may
        be None to cause the entire directive to be passed through.
        The invert argument is for #(el)ifndef.
        Returns the actual condition result, and adjusts the current state,
        or raises OutputDirective.
        """
        section = self.ifsect
        if invert and cond is not None: cond = not cond
        section.group(GroupState.get(cond), self)
        if cond is None:
            # Some macro was undefined.  The directive line may have been
            # changed to reflect those macros actually defined already.
            if self.handler.name1 and section.skip:
                # For a #elif(*) which is the first group to pass through.
                # Change the directive name.
                self.source.line[self.namepos].value = self.handler.name1
            raise OutputDirective(Action.IgnoreAndPassThrough)
        return cond

    def error(self, msg: str) -> None:
        """ Write an error message. """
        self.prep.on_error_token(self.line[0], msg)

    def expand_args(self) -> Iterator[PpTok]:
        """ Expands and returns self.args. """
        with self.prep.nest():
            return self.prep.macros.expand(
                            TokIter(self.args), origin=self.dirtok,
                          ).strip()

    def in_macro_call(self, in_args: bool = False) -> MacroCall:
        """
        The call object if currently scanning a function macro call.  Only for
        clang.
        """
        if self.prep.clang:
            return self.source.in_macro
        return None

    def check_macro_args(self, in_args: bool = True) -> bool:
        """
        True if currently scanning a function macro call after the opening
        '('.  Posts an error message if so.  Only for clang.
        """
        call: MacroCall = self.in_macro_call()
        if call:
            if in_args and call.args is not None:
                # clang doesn't suport this in a macro call.
                self.prep.on_error_token(
                    self.line[0],
                    f"#{self.name} directive embedded in macro call is not supported "
                    "by clang.")
                return True
            if not in_args and call.args is None:
                return True
        return False

    @property
    def lineno(self) -> int:
        return self.line[0].lineno

    def brk(self) -> bool:
        return break_match(line=self.lineno)

    def __repr__(self) -> str:
        return str(self.line)

class Handler:
    """ Defines method to handle a specific directive name.
    This is a single object for each name.
    It is called with the particular directive as an argument.
    """

    # Attributes that can be modified by constructor keywords.
    always: bool = False        # Execute even if an error
    args: str = None            # At least one arg needed after directive name
    effect: bool = True         # Has an effect.  This resets once.pending
    cond: bool = False          # This is a conditional-inclusion directive.
                                #   after execution.
    guard: int = 0              # Identifies (1) or defines (2) an include guard
    name1: str = None           # Alternate directive name for Passthru
                                #   if first group.  E.g. change #elif to #if
    nest: int = 0               # Change nesting level before execution.
    nestme: int = 0             # Add to nesting level to log directive.
    section_reqd: bool = False  # There must be a section from matching #if*.
    skipping: bool = False      # Execute in a skipped group.
    showstate: bool = False     # Show group state if not GroupState.Process.
    dirattrs: dict              # Attributes to set on the owning Directive.

    def __init__(self, name, base: str = None, **kwds):
        if base: self.__dict__.update(handlers[base].__dict__)
        self.name = name
        self.__dict__.update(**kwds)
        if self.guard:
            attr = attrgetter(f"on_{name}")
            self.method = lambda self, **kwds: attr(self)(**kwds)
        else:
            self.method = methodcaller(f"on_{name}")
        self.dirattrs = {}
        if self.cond: self.dirattrs.update(cond=True)
        if self.nest: self.dirattrs.update(nest=self.nest)

    def __call__(self, dir: Directive) -> Iterable[PpTok]:
        """ Invoke the directive processing method.
        Also given the current depth of nested Skip groups.  This filters
        out some directives and processes others.
        """

        # Some directives are ignored in a Skip group.  Handle only:
        #   In nested Skip group: #if*, #endif.
        #   In unnested Skip group: #if*, #el*, #endif.

        prep = dir.prep
        nest = self.nest
        source = dir.source
        if self.args and not dir.args_reqd(self.args):
            if not self.always: return
        prep.log.directive(dir, nest=self.nestme)
        #if self.section_reqd and section.top:
        #    dir.error(f"#{dir.name} without earlier matching #if*.")
        #    return
        if nest:
            prep.nesting += nest
        kwds = {}
        if self.guard:
            # Directive uses an include-guard.
            if source.once_pend:
                if self.guard(source.once):
                    kwds.update(check_once=True)
        try:
            res = self.method(dir, **kwds)
            return res
            return res and [dir.dirtok.make_passthru(res)]

        finally:
            if self.effect and source.once_pend:
                source.no_guard()
            if nest:
                prep.nesting -= nest
            if self.cond:
                if self.showstate:
                    groupstate = dir.ifsect.state.groupstate
                    prep.log.msg(f"State = {dir.ifsect}", dir.line[0])
                if dir.ifsect.passthru:
                    raise OutputDirective(Action.IgnoreAndPassThrough)
    def __repr__(self) -> str:
        return f"<Handler # {self.name}>"

class NullHandler(Handler):
    """ A Handler for a null (no name) directive """
    dirattrs = {}

    def __call__(self, dir: Directive) -> Iterable[PpTok]:
        # Make this an empty iterator
        return
        #yield

# Create and index a Handler for every directive name.
handlers: Mapping[str, Handler] = {}

for init in dict(
    _condbase=  dict(cond=True, section_reqd=True, showstate=True, always=True,
                     skipping=True, nestme=-1),   
    _define=    dict(args='macro name',
                     guard=attrgetter('guard')),
    _elif=      dict(args='expression', base='condbase', name1='if'),
    _elifdef=   dict(args='macro name', base='condbase', name1='ifdef'),
    _elifndef=  dict(args='macro name', base='condbase', name1='ifndef'),
    _else=      dict(base='condbase', ),
    _endif=     dict(base='condbase', nest=-1),
    _error=     dict(effect=False),
    _if=        dict(args='expression', base='condbase', nest=1,
                     guard=attrgetter('pending')),
    _ifdef=     dict(args='macro name', base='condbase', nest=1),
    _ifndef=    dict(args='macro name', base='condbase', nest=1,
                     guard=attrgetter('pending')),
    _include=   dict(args='file'),
    _line=      dict(args='line number', effect=False),
    _pragma=    dict(effect=False),
    _undef=     dict(args='macro name'),
    _warning=   dict(effect=False),
    ).items():
    name, kwds = init
    name = name[1:]
    handlers[name] = Handler(name, **kwds)

del init

handlers[''] = NullHandler('<null>')

class Action(object):
    """What kind of abort processing to do in OutputDirective"""
    # Abort processing (don't execute), but pass the directive through to output
    IgnoreAndPassThrough = 0
    # Abort processing (don't execute), and remove from output
    IgnoreAndRemove = 1


class OutputDirective(Exception):
    """
    Raise this exception to abort processing of a preprocessor directive and
    to instead possibly output it as is into the output.
    """

    # This exception is raised:
    # An unknown directive, if on_directive_unknown() returns True.
    # A conditional directive when the group is in a passthru state.
    # An undefined macro in a conditional control expression.
    # An include file name not found or malformed.
    # on_directive_handle raises (in a custom preprocessor override).

    def __init__(self, action: Action):
        self.action = action

from pcpp.dircondition import GroupState
