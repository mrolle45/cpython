""" Module directive.py.
Manages a directive in a source file.
The Directive class holds and executes one directive.
"""

from __future__ import annotations

from operator import methodcaller

from pcpp.lexer import TokType, no_ws

class Directive:
    """ A single directive in a source file.
    Will consume the tokens in the entire source line.
    Can execute the directive, generating any tokes to be passed through.
    """
    def __init__(self, source: Source, line: Tokens):
        self.prep = prep = source.prep
        self.source = source
        self.line = line
        line = prep.tokenstrip(line[:])
        line = prep.tokenstrip(line[1:])
        if not line:
            self.name = None
        else:
            self.name = line[0].value
        self.handler = handlers.get(self.name)
        self.args = prep.tokenstrip(line[1:])

    def __call__(self, handling: bool) -> Iterable[LexToken]:
        self.handling = handling
        try: return self.handler(self)
        except: return iter([])

    ### Methods for all the different directive names...
    # They all generate any tokens to be passed through (mostly none).
    # Or possibly raise an OutputDirective exception, which will cause self() to do likewise.

    def on_define(self) -> Iterator[LexToken]:
        if self.ifstate.enable:
            if self.source.include_guard and self.source.include_guard[1] == 0:
                if self.source.include_guard[0] == self.args[0].value and len(self.args) == 1:
                    self.source.include_guard = (self.args[0].value, 1)
                    # If ifpassthru is only turned on due to this include guard, turn it off
                    if self.source.ifpassthru and not self.ifstack[-1].ifpassthru:
                        self.source.ifpassthru = False
            self.prep.define(self.args)
            macro = self.prep.macros[self.args[0].value]
            if self.handling is None:
                return iter(self.line)

    def on_elif(self) -> Iterator[LexToken]:
        if self.ifstack[-1].enable:     # We only pay attention if outer "if" allows this
            if self.source.enable and not self.source.ifpassthru:         # If already true, we flip enable False
                self.source.enable = False
            elif not self.source.iftrigger:   # If False, but not triggered yet, we'll check expression
                result, rewritten = self.evalexpr(self.args)
                if rewritten is not None:
                    self.source.enable = True
                    if not self.source.ifpassthru:
                        # This is a passthru #elif after a False #if, so convert to an #if
                        x[i].value = 'if'
                    x = x[:i+2] + rewritten + [x[-1]]
                    x[i+1] = copy.copy(x[i+1])
                    x[i+1].type = self.prep.t_SPACE
                    x[i+1].value = ' '
                    self.source.ifpassthru = True
                    self.ifstack[-1].rewritten = True
                    raise OutputDirective(Action.IgnoreAndPassThrough)
                if self.source.ifpassthru:
                    # If this elif can only ever be true, simulate that
                    if result:
                        newtok = copy.copy(x[i+3])
                        newtok.type = self.prep.t_INTEGER
                        newtok.value = self.prep.t_INTEGER_TYPE(result)
                        x = x[:i+2] + [newtok] + [x[-1]]
                        raise OutputDirective(Action.IgnoreAndPassThrough)
                    # Otherwise elide
                    self.source.enable = False
                elif result:
                    self.source.enable  = True
                    self.source.iftrigger = True
                            
    def on_elifdef(self) -> Iterator[LexToken]:
        ...

    def on_elifndef(self) -> Iterator[LexToken]:
        ...

    def on_else(self) -> Iterator[LexToken]: 
        if self.ifstack[-1].enable:
            if self.source.ifpassthru:
                self.source.enable = True
                raise OutputDirective(Action.IgnoreAndPassThrough)
            if self.source.enable:
                self.source.enable = False
            elif not self.source.iftrigger:
                self.source.enable = True
                self.source.iftrigger = True

    def on_endif(self) -> Iterator[LexToken]:
        state = self.ifstack.pop()
        self.state.skip_auto_pragma_once_possible_check = True
        if state.rewritten:
            raise OutputDirective(Action.IgnoreAndPassThrough)

    def on_error(self) -> Iterator[LexToken]: 
        ...

    def on_if(self) -> Iterator[LexToken]:
        oldstate = self.push_if_state()
        if not self.args: return
        if oldstate.top and self.source.at_front_of_file and self.args[0].value == '!':
            # Check for potential include guard.
            expr: Tokens = Tokens(no_ws(self.args[1:]))
            def check() -> LexToken | None:
                if expr[0].value != 'defined': return None
                if expr[1].value == '(' and expr[2].type is TokType.CPP_ID and expr[3].value == ')' and len(expr) == 4:
                    return expr[2]
                if expr[1].type is TokType.CPP_ID and len(expr) == 2:
                    return expr[1]
                return None
            guard = check()
            if guard:
                self.prep.on_potential_include_guard(guard)
                self.source.include_guard = (guard, 0)

        # Start the new group.
        if self.ifstate.may_enable:
            result = self.evalexpr()
            self.condition(result)

    def on_ifdef(self) -> Iterator[LexToken]:
        self.push_if_state()
        if not self.args: return
        # Start the new group.
        if self.ifstate.may_enable:
            res = self.defined(self.args[0])
            self.condition(res)

    def on_ifndef(self) -> Iterator[LexToken]:
        oldstate = self.push_if_state()
        if not self.args: return
        # Check for an include guard.
        if not oldstate and self.source.at_front_of_file:
            self.prep.on_potential_include_guard(self.args[0].value)
            self.source.include_guard = (self.args[0].value, 0)
        # Start the new group.
        if self.ifstate.may_enable:
            res = self.defined(self.args[0])
            self.condition(res, invert=True)

    def on_include(self) -> Iterator[LexToken]:
        if self.ifstate.enable:
            oldfile = self.prep.macros['__FILE__'] if '__FILE__' in self.prep.macros else None
            if self.args and self.args[0].value != '<' and self.args[0].type != self.prep.t_STRING:
                self.args = self.tokenstrip(self.prep.macros.expand(self.args))
            yield from self.prep.include(self.args, self.line)
            if oldfile is not None:
                self.prep.macros['__FILE__'] = oldfile

    def on_line(self) -> Iterator[LexToken]:
        ...

    def on_pragma(self) -> Iterator[LexToken]:
        if self.args[0].value == 'once':
            if self.source.enable:
                self.include_once[self.source] = None

    def on_undef(self) -> Iterator[LexToken]:
        if self.ifstate.enable:
            self.prep.undef(self.args)
            if self.handling is None:
                return iter(self.line)

    def on_warning(self) -> Iterator[LexToken]:
        ...


    ### Other methods...

    @property
    def ifstate(self) -> IfState:
        return self.source.ifstate

    @property
    def ifstack(self) -> IfStack:
        return self.source.ifstack

    def args_reqd(self, what: str) -> bool:
        """ Verify that at least one token exists in the args.
        Calls on_error() otherwise.
        """
        if self.args: return True
        self.error(f"Directive #{self.name} requires {what}.")
        return False

    def push_if_state(self) -> IfState:
        """ Open a new group.  Return state of previous group. """
        oldstate = self.ifstate
        self.ifstack.push(False, not oldstate.enable, False, self.line)
        return oldstate

    def evalexpr(self) -> bool | None:
        result, rewritten = self.source.evalexpr(self.args)
        if rewritten is not None:
            line = self.line
            line = line[:i+2] + rewritten + [line[-1]]
            line[i+1] = copy.copy(line[i+1])
            line[i+1].type = self.prep.t_SPACE
            line[i+1].value = ' '
            self.source.ifpassthru = True
            self.ifstack[-1].rewritten = True
            return None
        return result

    def defined(self, name: LexToken) -> bool | None:
        """ Is the name a defined macro.
        Return None to bail out.
        """
        n = name.value
        if n in self.prep.macros:
            return True
        else:
            return self.prep.on_unknown_macro_in_defined_expr(n)

    def condition(self, cond: bool | None, invert: bool = False):
        """ Handle the condition of an #(el)if, #(el)if(n)def.
        The cond is the actual or assumed result.  The assumed
        result occurs if an undefined macro is involved, and may
        be None to cause the entire directive to be passed through.
        The invert argument is for #(el)ifndef.
        Returns the actual condition result, and adjusts the current state,
        or raises OutputDirective.
        """
        if cond is None:
            # Some macro was undefined.  The directive line may have been
            # changed to reflect those macros actually defined already.
            raise OutputDirective(Action.IgnoreAndPassThrough)
        if invert: cond = not cond
        self.ifstate.advance(cond)
        return cond

    def error(self, msg: str) -> None:
        """ Write an error message. """
        self.prep.on_error_token(self.line[0], msg)

    @property
    def lineno(self) -> int:
        return self.line[0].lineno

class Handler:
    """ Defines method to handle a specific directive name. """

    # Attributes that can be modified by constructor keywords.
    args_reqd: str = None       # At least one arg needed after directive name
    state_reqd: bool = False    # There must be a group from matching #if*.
    always: bool = False        # Execute even if an error
    nest: int = 0               # Change nesting level before execution
    renest: bool = False        # Restore nesting level after execution
    effect: bool = True         # Has an effect.  This resets at_front_of_file after execution.

    def __init__(self, name, **kwds):
        self.name = name
        self.method = methodcaller(f"on_{name}")
        self.__dict__.update(**kwds)

    def __call__(self, dir: Directive) -> Iterable[LexToken]:
        if self.args_reqd and not dir.args_reqd(self.args_reqd):
            if not self.always: return
        if self.state_reqd and not dir.ifstack:
            dir.error("Misplaced #{dir.name}")
            return
        dir.prep.nesting += self.nest
        try:
            return self.method(dir)
        finally:
            if self.renest: dir.prep.nesting -= self.nest
            if self.effect: dir.source.at_front_of_file = False

    def __repr__(self) -> str:
        return f"<Handler {self.name}>"
handlers: Mapping[str, Handler] = {}

for init in [
        ('define',      dict(args_reqd='macro name', ), ),
        ('elif',        dict(args_reqd='expression', state_reqd=True, always=True, nest=-1, renest=True)),
        ('elifdef',     dict(args_reqd='macro name', state_reqd=True, always=True, nest=-1, renest=True)),
        ('elifndef',    dict(args_reqd='macro name', state_reqd=True, always=True, nest=-1, renest=True)),
        ('else',        dict(state_reqd=True, nest=-1, renest=True)),
        ('endif',       dict(state_reqd=True, nest=-1)),
        ('error',       dict(effect=False)),
        ('if',          dict(args_reqd='expression', always=True, nest=1)),
        ('ifdef',       dict(args_reqd='macro name', always=True, nest=1)),
        ('ifndef',      dict(args_reqd='macro name', always=True, nest=1)),
        ('include',     dict(args_reqd='file', ), ),
        ('line',        dict(args_reqd='line number', effect=False)),
        ('pragma',      dict(effect=False)),
        ('undef',       dict(args_reqd='macro name', ), ),
        ('warning',     dict(effect=False)),
    ]:
    name, *kwds = init
    handlers[name] = Handler(name, **(kwds and kwds[0] or {}))


class Action(object):
    """What kind of abort processing to do in OutputDirective"""
    """Abort processing (don't execute), but pass the directive through to output"""
    IgnoreAndPassThrough = 0
    """Abort processing (don't execute), and remove from output"""
    IgnoreAndRemove = 1

class OutputDirective(Exception):
    """Raise this exception to abort processing of a preprocessor directive and
    to instead possibly output it as is into the output"""
    def __init__(self, action: Action):
        self.action = action

class BadDirective(Exception):
    """ Raise this exception to abort processing of a directive. """

