""" macros
Manages macro definitions and lookups for a C translation unit.
"""
from __future__ import annotations

import contextlib
import copy
from enum import Enum, auto
from dataclasses import dataclass, InitVar
from functools import reduce
from itertools import chain, zip_longest
from operator import attrgetter, and_
from typing import Iterator

from pcpp.common import *
from pcpp.replacements import ReplStage
from pcpp.tokens import (PpTok, Tokens, TokIter, Hide,
                         TokenSep, TokenSepSpace,
                         )

''' The following is modeled after David Prosser's algorithm, which can be found at
https://www.spinellis.gr/blog/20060626/cpp.algo.pdf
The expand() function is performed by the Macros.expand() method.
The subst() function is performed by the TokSubstMgr.__call__() method.
    This incorporates the hsadd() call in that method.
'''

# ------------------------------------------------------------------
# Macro object
#
# This object holds information about one preprocessor macro.
#
#    .nametok   - Macro name token in the #define
#    .name      - Macro name string in the #define
#    .value     - Macro value (a list of intoks, from the #define or other source)
#    .substs    - Objects which generate replacement intoks.
#    .is_func   - A function macro (class attribute).
#    .source    - Source object containing the #define
#    .lineno    - Line number containing the #define
#
# For function macros only:
#    .param_names   - List of argument names, including __VA_ARG__ and __VA_OPT__
#    .variadic  - Boolean indicating whether or not variadic macro
#
# When a macro is created, the macro replacement token sequence is
# pre-scanned and used to create TokSubst objects that are later used
# during macro expansion.
# ------------------------------------------------------------------

class Macro:
    """ The definition of a preprocessor macro.
    The definition is stored in a Tokens self.value.
    It stores replacement generators in self.substs.
    """
    is_func: ClassVar[bool] = False         # True for FuncMacro class
    is_dyn: ClassVar[bool] = False          # True for DynMacro class
    has_paste: bool = False                 # Contains any π tokens
    substs: tuple[TokSubst, ...]            # The expansion token generators.
    error: str = None                       # Error message if invalid.

    def __init__(self, prep: Preprocessor, nametok: PpTok, value: TokIter):
        self.nametok =  nametok
        def iter_defn() -> Iterator[PpTok]:
            """ Generate the replacement tokens. """
            for tok in value:
                if tok.type.paste:
                    self.has_paste = True
                tok.in_macro = True
                yield tok
        self.value = Tokens(iter_defn())

        self.macros = prep.macros
        
        self.substs = tuple(self.make_substs())
        if self.has_paste:
            if self.substs[0].is_paste:
                self.set_error(
                    self.substs[0].tok,
                    "'##' cannot be at the start of macro replacement." )
            elif self.substs[-1].is_paste:
                self.set_error(
                    self.substs[-1].tok,
                    "'##' cannot be at the end of macro replacement." )
            # Fix up param substs adjacent to a paste subst.
            prev: TokSubst = self.substs[0]
            for subst in self.substs[1:]:
                if prev.is_paste and subst.is_param:
                    subst.expand = False
                elif subst.is_paste and prev.is_param:
                    prev.expand = False
                prev = subst

    @property
    def name(self) -> str:
        return self.nametok.value

    @property
    def lineno(self) -> int:
        return self.nametok.lineno

    @property
    def source(self) -> Source:
        return self.nametok.source

    @property
    def prep(self) -> Preprocessor:
        return self.macros.prep

    def subst_padded(self, mgr: TokenSubstMgr, nametok: PpTok = None
                     ) -> Iterator[PpTok]:
        """
        Generate the substitutions, preceded and followed by padding (only if
        not a nested __VA_OPT).  The first token, if any, takes its separation
        from the macro name token.
        """
        top: bool = self is mgr.m
        name: PpTok = nametok or mgr.call.nametok
        toks = TokIter(self.subst_tokens(mgr))
        if not top:
            yield from toks
            return
        # First substitute token, or placemarker.
        for tok in toks:
            break
        else:
            # No tokens from substitution.  Make a placemarker.
            tok = name.make_marker()
        tok = tok.add_sep(name.sep)
        # Remaining subst tokens.
        prev = tok
        for tok in toks:
            yield prev
            prev = tok
        yield prev

    @property
    def log(self) -> _DebugLog:
        return self.macros.log

    def set_error(self, tok: PpTok, msg: str) -> None:
        self.prep.on_error_token(tok, msg)
        self.error = msg

    def sameas(self, other: Self) -> bool:
        """ True if the two definitions are the same, per (C99 6.10.3p2). """
        if type(self) is not type(other): return False
        # Compare the replacement lists, treating all whitespace separation
        # the same.
        for x, y in zip_longest(self.value, other.value):
            if x is None or y is None: return False
            if x.value != y.value: return False
            if x.spacing != y.spacing: return False
        return True


class ObjMacro(Macro):
    """
    Object-like macro.  May contain paste operators but nothing involving
    parameter names, __VA_OPT__, or stringizing, which are only relevant to
    function-like macros.
    """

    def make_substs(self) -> Iterator[TokSubst]:
        """
        Build substitution items in self.substs.  These are immutable and
        reused every time the macro is invoked.

        A substitution item can be:
          - TokSubstPaste  -- a π token.
          - TokSubstLiteral  -- a single token, otherwise.
        """
        repltoks: Iterator[PpTok] = iter(self.value.data)

        for repltok in repltoks:
            if repltok.type.paste:
                # '##'.  Alters adjacent substs, which must exist.
                yield TokSubstPaste(repltok)
            else:
                yield TokSubstLiteral(repltok)

    def subst_tokens(self, mgr: TokenSubstMgr) -> Iterator[PpTok]:
        """
        Do all substitution operations except for pasting.  Object macros have
        no parameters to substitute, so this method is a pass-through of the
        replacement tokens themselves.
        """
        yield from self.value

    def __repr__(self):
        return f"{self.name}={self.value!r}"


class FuncMacro(Macro):
    is_func: ClassVar[bool] = True
    param_names: list[str]          # Names of params, plus __VA_ARGS__.
    nparams: int                    # len of param_names.

    def __init__(self, prep: Preprocessor, nametok: PpTok, value: Tokens,
                 param_names, variadic: bool):
        self.param_names = param_names
        self.nparams = len(param_names)
        self.variadic = variadic

        super().__init__(prep, nametok, value)

    def make_substs(self) -> Iterator[TokSubst]:
        """
        Generate substitution items.  These are immutable and reused every
        time the macro is invoked.

        A substitution item can be:
          - TokSubstString -- Σ + (a parameter name or __VA_OPT__).
          - TokSubstPaste  -- a π token.
          - TokSubstVaOpt  -- a "VA_OPT ( ... )" expression.
          - TokSubstParam  -- a parameter name.  Might be adjacent to a π.
          - TokSubstLiteral  -- a single token, not any of the above.
        Note, if a Param or VaOpt is adjacent to a Paste, then it is
        constructed to avoid expansion.
        """
        repltoks: Iterator[PpTok] = iter(self.value[:].data)

        param_names: list[str] = self.param_names

        def skip_ws() -> PpTok | None:
            for tok in repltoks:
                if tok.type.ws:
                    pass
                else:
                    return tok
            return None

        def next_subst(*, after_paste: bool = False) -> TokSubst | None:
            """
            A substitution object from the given replacement token, if any.
            May consume more replacement tokens.  Preceding whitespace is
            either added to the output or ignored.
            """
            tok = next(repltoks, None)
            if not tok:
                return None
            val = tok.value
            keep_spacing = True
            typ = tok.type
            if typ.paste:
                # '##'.  Alters adjacent substs, which must exist.
                return TokSubstPaste(tok)
            elif typ.stringize:
                # Stringize.  Get following param name or __VA_OPT__.

                try:
                    param = next(repltoks)
                    argnum = param_names.index(param.value)
                    param = TokSubstParam(param, argnum)
                    param.expand = False
                    return TokSubstString(tok, param, after_paste)
                except ValueError:
                    # Did not find following parameter name.  It could be
                    # __VA_OPT__.
                    if self.variadic and param.value == '__VA_OPT__':
                        opt = make_va_opt(param)
                        return TokSubstString(tok, opt, after_paste)
                        
                    self.set_error(
                        tok,
                        "'#' must be followed by parameter name "
                        "or __VA_OPT__." )
                    return None

            elif val == '__VA_OPT__':
                return make_va_opt(tok, after_paste=after_paste)
            elif val in param_names:
                # Parameter name.  Check if following a paste.
                argnum = param_names.index(val)
                #subst = TokSubstParam(tok, argnum)
                subst = TokSubstParam(tok, argnum, after_paste=after_paste)
                if after_paste:
                    subst.expand = False
                return subst
            else:
                # Anything else.
                return TokSubstLiteral(tok)

        def make_va_opt(tok: PpTok, *, after_paste: bool = False) -> TokSubstVaOpt:
            """ Create __VA_OPT__ ( ... ) substitution. """
            m : FuncMacro = self.parse_va_opt(repltoks, tok)
            return TokSubstVaOpt(tok, m, after_paste=after_paste)

        after_paste: bool = False

        # Loop over intoks to turn into substs.
        while True:
            subst = next_subst(after_paste=after_paste)
            if not subst:
                break
            yield subst
            after_paste = subst.is_paste
        if after_paste:
            self.set_error(
                tok,
                "'##' cannot be at the end of macro replacement." )

    def subst_tokens(self, mgr: TokenSubstMgr) -> Iterator[PpTok]:
        """
        Do all substitution operations except for pasting.  Iterate resulting
        tokens, which can include paste operators.
        """

        toks = itertools.chain(
            *map(operator.methodcaller('pad_toks', mgr), self.substs))
        #tokens = toks.copy_tokens()
        for tok in toks:
            yield tok

    @staticmethod
    def stringize(toks: Iterable[PpTok]) -> str:
        """
        Implement the '#' operator in a function macro (C99 6.3.10.2).  Takes
        the tokens for the argument of the operator and returns a string.
        """

        def parts() -> Iterator[str]:
            """ Generate the pieces of the result from the tokens. """

            has_ws = False
            begin: bool = True
            prev: PpTok = None

            for tok in toks:
                if tok.sep.spacing:
                    # Start or continue a run of ws, but not at the beginning.
                    if not begin:
                        has_ws = True
                if not tok.value:
                    continue
                if begin:
                    begin = False
                elif has_ws:
                    yield ' '
                    has_ws = False
                typ: TokType = tok.type
                if tok.repls:
                    # Revert to original spelling of UCNs.
                    tok = tok.revert(ReplStage.ESCAPE, force=True)
                val: str = tok.value
                if typ.str:
                    # Escape every " and \.
                    val = val.replace("\\","\\\\").replace('"', '\\"')
                elif typ.chr:
                    # Escape every \.`
                    val = val.replace("\\","\\\\")
                elif tok.value.startswith('//'):
                    val = f"/*{val[2:]}*/"
                yield val

        sp = list(parts())
        string = ''.join(sp)
     
        # Clang won't end with an odd number of \ chars.
        if string.endswith('\\'):
            m = re.search(r'\\+$', string)
            if len(m.group()) & 1:
                string = string[:-1]

        return string

        
    def parse_va_opt(self, toks: Iterator[PpTok],
                        name: PpTok
                        ) -> FuncMacro | None:
        """ Finds the balanced '(' ... ')' following __VA_OPT__.
        Consumes intoks in the iterator up to the closing ')'
        Returns a Macro for the replacement list or None if ill-formed.
        """
        prep = self.prep
        if not self.variadic:
            self.set_error(name,
                            "'__VA_OPT__' only allowed in a variadic macro.")
            return None
        for t in toks:
            if not t.type.ws:
                break
        # t is first token, after any whitespace.
        if not t or t.value != '(':
            self.set_error(name, "'__VA_OPT__' requires a replacement list.")
            return None
        repl = Tokens()
        def to_matching_paren() -> None:
            """ Copy intoks to repl Tokens after '(' up to matching ')'.
            Raise exception if ran out of input intoks first.
            """
            for t in toks:
                repl.append(t)
                v = t.value
                if v == '(':
                    to_matching_paren()
                elif v == ')':
                    return
            raise SyntaxError("Malformed '__VA_OPT__'.")

        try: to_matching_paren()
        except Exception as e:
            self.set_error(name, str(e))
            return None

        return FuncMacro(
            prep, name, repl[:-1], self.param_names, variadic=True
            )

    def sameas(self, other: FuncMacro) -> bool:
        """ True if the two definitions are the same (C99 6.10.3p2). """
        if not super().sameas(other): return False
        return self.param_names == other.param_names

    def __repr__(self):
        args = self.param_names
        if self.variadic:
            args = args[:-2] + ['...']
        argstr = ', '.join(args)
        return f"{self.name}({argstr})={self.value!r}"


class DynMacro(Macro):
    """ Special macro, such as __LINE__. """
    is_dyn: ClassVar[bool] = True

    def __init__(self, prep: Preprocessor, name: PpTok):
        super().__init__(prep, name, Tokens())

    def make_substs(self) -> Iterator[TokSubst]:
        """
        Generate substitution items.  These are immutable and reused every
        time the macro is invoked.
        """
        yield dynamic_substs_tab[self.name](self.nametok)

    def subst_tokens(self, mgr: TokenSubstMgr) -> Iterator[PpTok]:
        """
        Do all substitution operations except for pasting.  Dynamic macros
        have no parameters to substitute.  However, they do generate a single
        result token based dynamically on the context in which the macro was
        called.
        """
        yield from self.substs[0].toks(mgr)

    def __repr__(self) -> str:
        return self.name


class MacroExp:
    """ Handles a single call to Macros.expand() using its __call__(). """

    # The preprocessor.
    prep: Preprocessor

    # Current definitions of macro names.
    macros: Macros

    # Replacement tokens to consume during the expansion.  Includes macro
    # expansion tokens awaiting rescan.  Empty when expansion is complete.
    repltoks: TokIter

    # Token in source file which is being expanded.  Used to evaluate __LINE__
    # and __FILE__.  At the top level, it is the macro name token currently
    # being considered for expansion.  In a directive, it is the directive '#'
    # token.  For a function argument, it is the origin in the MacroExp which
    # is expanding the function.
    origin: PpTok

    # The source where intoks are located, if they are the file's contents.
    # This is called only once for any Source file, and intoks = all tokens
    # generated by that Source by lexing its contents, including from included
    # files (which have already been expanded and are protected from expansion
    # at this level).  None if expanding something else.  
    top: Source = None

    # When expanding a token that didn't come from an expansion, this is the
    # call to that token.  If any of its expansion tokens is expanded on
    # rescan, this stays the same.
    orig_call: MacroCall = None

    def __init__(self, macros: Macros, repltoks: TokIter, *,
                 top: Source = None,
                ):
        self.macros = macros
        self.prep = macros.prep
        self.repltoks = repltoks
        self.top = top
        #print("MacroExp", macros.depth, top)

    @TokIter.from_generator
    def __call__(self) -> Iterable[PpTok]:
        """
        Expands iterable of replacement tokens, expanding macro names.  After
        each expansion, rescan that expansion plus all following tokens.
        Generates result iterable of tokens.

        Note: this is recursive.  Replacing a function macro will expand any
        arguments that are used in the replacement, using a nested MacroExp
        call.  self.macros.depth is the current recursion level.

        This is equivalent of expand() in Prosser's algorithm.
            See https://www.spinellis.gr/blog/20060626/cpp.algo.pdf.
        """
        prep = self.prep

        top = self.top
        if top:
            if top.parent:
                prep.log.write(f"Preprocessing source file", source=top)
            else:
                prep.log.write(f"Preprocessing startup")
        with self.macros.nest(self):

            repltoks: TokIter = self.repltoks
            orig = None
            # Flag to change C++ comment to C comment in some cases.
            # 1. Tok was produced by a macro replacement.  Will have tok.hide.
            # 2. Follows an unexpanded macro and possibly some newlines.
            # This is only for GCC mode.
            fix_comment: bool = False

            # Current token, None after empty expansion.
            repltok: PpTok

            while True:
                for repltok in repltoks:
                    #if repltok.brk():
                    #    toks = Tokens.join(repltok, *repltoks.copy_tokens())
                    #    print(f"-- {str(toks)!r}")
                    #    res = Prosser(prep.macros, toks)
                    #    print(f"---- {str(res)!r}")
                    if not orig or repltok.indent: orig = repltok
                    break
                else:
                    return
                # replace_and_rescan replaces repltok possibly, and generates
                # any passthru repltoks from the input, followed by first
                # unexpanded token (if any).
                for repltok in self.replace_and_rescan(repltok):
                    if not repltok.type.norm:
                        yield repltok
                        continue
                    # repltok = the next token
                    break
                else:
                    # Go back and do the next token, keeping same orig.
                    #orig = None
                    continue
                # repltok = the next token

                if repltok is not orig:
                    self.macros.log.msg(
                        f"End replace.  New token {repltok!r}", orig
                        )
                    # orig got replaced with something else.  Source token got
                    # replaced with another token.  Pick up original spacing
                    # and indent.
                    if top and orig.indent and not repltok.indent:
                        if repltok.hide:
                            indent = orig.indent
                            # gcc and clang add a space at the start of the
                            # line if orig is at the start and repltok has
                            # spacing.
                            if (prep.lang.emulate
                                and indent.colno == 1
                                and (repltok.spacing
                                    #or (indent.lineno == repltok.lineno
                                    #    and indent.source is repltok.source)
                                        )
                                ):
                                indent = indent.copy(colno=2)
                        else:
                            # Use orig line for repltok.indent, keep
                            # repltok.colno.
                            indent = repltok.loc.copy(lineno=orig.lineno,
                                                phys_offset=0)
                        repltok = repltok.with_sep(
                            TokenSep.create(indent=indent))
                    elif orig.spacing:
                        repltok = repltok.with_sep(
                            TokenSep.create(spacing=True))

                orig = None
                yield repltok

    @TokIter.from_generator
    def replace_and_rescan(self, repltok: PpTok) -> Iterator[PpTok]:
        """
        Do macro replacement with rescan for given input token `repltok`.
        Generates the next token (if any) to be processed, possibly the same
        token.  May modify repltoks if a macro replacement occurs.

        If any passthru tokens are seen while consuming a function argument
        list, these are generated first.

        This method _could_ just keep working through the entire input.
        However, the caller will want to add some output positioning
        information and lines that became blank as a result of expansion.

        When repltok is a macro subject to replacement:

            Any function argument list is consumed from repltoks.

            Any passthru repltoks found while consuming the argument list are
            generated.

            Its replacement repltoks are put back in the front of repltoks.

            If there were any replacement tokens, the first of these is
            consumed from repltoks and becomes the new repltok.  Then the
            entire process is repeated.

            If there were no replacement tokens, the method ends, without
            generating a result token.

        Otherwise, repltok is generated.

        A macro is NOT subject to replacement if:

            It is already being expanded at an outer level.

            A function macro has no argument list, or there's a problem with
            it.

            It is part of a defined-macro expression, which is looking for a
            'defined name' or 'defined ( name )' sequence of tokens.  It is
            part of a #if or #elif directive.

        """
        m: Macro | None     # Macro for name in repltok.value.

        log = self.macros.log
        repltoks = self.repltoks
 
        while repltok:
            # Examine repltok, the current token just removed from repltoks.  

            # Log this pass.
            log.msg(f"Next token {repltok!r}", repltok)
            # End the loop if repltok is not a macro which is to be expanded.
            if not repltok.type.id:
                #if repltok.type.null:
                #    repltok = None
                # Not an identifier.  Don't expand.
                break
            name = repltok.value
            m = self.macros.get(name)
            if not m:
                # Not a defined macro.  Don't expand.
                break
            # Token is a macro.
            hide = repltok.hide
            if hide and name in hide:
                # Hidden.  Don't expand.
                log.expand(repltok, m, False, hide)
                break
            if repltoks.in_defined_expr:
                # In 'defined...', don't expand.
                break

            call = MacroCall(m, repltok, repltoks, hide, self)
            if not repltok.hide:
                self.orig_call = call
            if call.args and call.args.extras:
                yield from call.args.passthrus()

            if call.expanding:
                log.expand(repltok, m, True, hide, call=call)
                new = call.subst()
                if new is None:
                    break
                if __debug__ and 0000:
                    # Useful for debugging.
                    newtokens: Tokens = new.copy_tokens()

                # If there are no replacement repltoks, then
                # quit, with repltok = the first token in the
                # remaining input (if any) or None.  This
                # allows repltok (possibly expanded) to be placed on
                # a new output line.
                repltok = next(new, None)
                if not repltok:
                    if self.top:
                        # No more replacements, at top level
                        # expansion.  Don't expand.
                        break
                    # Get next remaining input token
                    repltok = next(repltoks, None)
                    if not repltok:
                        # Nothing remaining at all.
                        break
                else:
                    # Effectively put the replacement repltoks
                    # in front of the remaining input repltoks.
                    # repltok is already the first replacement token
                    # and any remaining replacement repltoks are
                    # put in front of the remaining input
                    # repltoks.

                    repltoks.prepend(new)

            elif call.error:
                # Something wrong with the function call syntax.
                # Don't expand.
                self.prep.on_error_token(repltok, call.error)
                repltok = None
                break
            else:
                # No argument list.  Don't expand.
                log.expand(repltok, m, True, hide)
                fix_comment = True
                break
            log.msg(f"Rescanning.", repltok)

        yield repltok

    def __repr__(self) -> str:
        return f'< Expander {self.top} >'

class Expanders(Stack[MacroExp]):
    def __init__(self, macros: Macros):
        self.prep = macros.prep
        self.log = self.prep.log
        super().__init__()

    def append(self, exp: MacroExp) -> None:
        super().append(exp)
        self.log.write(f'Push expander {exp}, depth {self.depth}')

    def pop(self) -> MacroExp:
        exp: MacroExp = super().pop()
        self.log.write(f'Pop expander {exp}, depth {self.depth + 1}')
        return exp

 
class Macros(dict[PpTok, 'Macro']):
    """
    All the macro definitions in a translation unit -- top level source file,
    all included headers, and predefined stuff.

    Definitions are in the form of a token list.  Expansion generates a token
    sequence from an input token list.
    """

    # self[token] = Macro defined for token.name

    # Currently active expand() calls in order of most recent.
    expanders: ClassVar[Stack[MacroExp]] = Stack()

    # Next value of __COUNTER__ macro.
    countermacro: int = 0

    # The owning preprocessor, given to the constructor.
    prep: Preprocessor

    # The preprocessor's lexer.
    lexer: Lexer

    # The preprocessor's token types enumeration.
    TokType: TokType

    # The preprocessor's debug logger.
    log: DebugLog

    def __init__(self, prep: Preprocessor):
        self.prep = prep
        self.lexer = prep.lexer
        self.log = prep.log
        self.TokType = prep.TokType
        self.expanders = Expanders(self)

    def define(self, defn: TokIter,
               _bad_names = set('defined __VA_ARGS__ __VA_OPT__'.split()),
               **kwds) -> Macro:
        """
        Define a new macro from tokens following #define in a directive.
        """
        TokIter.check_type(defn, "Macros.define()")
        prep = self.prep
        try:
            # Name is first token, skipping whitespace.
            defn = defn.strip()
            name = next(defn)
            # Validate the macro name.
            if not name.type.id:
                self.prep.on_error_token(
                    name,
                    f"Macro definition {name.value!r} requires an identifier")
                return None
            if name.value in _bad_names:
                self.prep.on_error_token(
                    name, f"{name.value!r} is not a valid macro name")
                return None

            defn = defn.strip()
            if name.type is self.TokType.CPP_OBJ_MACRO:
                if name.value in dynamic_substs_tab:
                    m = DynMacro(prep, name)
                else:
                    m = ObjMacro(prep, name, defn, **kwds)
            elif name.type is self.TokType.CPP_FUNC_MACRO:
                # A macro with arguments
                variadic = False
                # Get the argument names.  Gets intoks through the closing
                # ')'.
                def argtokens() -> Iterator[PpTok]:
                    for tok in defn:
                        if tok.value == ')':
                            return
                        yield tok

                def iter_argnames() -> Iterator[str]:
                    """
                    Generate names of arguments.  For variadic, the ...
                    generates __VA_ARGS__.
                    """
                    nonlocal variadic
                    seen_comma: PpTok = None
                    intoks: Iterator[PpTok] = argtokens()
                    next(intoks)                # Opening '('.
                    for tok in intoks:
                        # The arg name -- ID or ELLIPSIS.
                        seen_comma = None
                        if tok.type is self.TokType.CPP_ID:
                            if variadic:
                                prep.on_error_token(
                                    tok,
                                    "No more arguments may follow "
                                    "a variadic argument")
                                return
                            yield tok.value
                        elif tok.type is self.TokType.CPP_ELLIPSIS:
                            yield '__VA_ARGS__'
                            variadic = True
                        else:
                            prep.on_error_token(
                                tok, f"Invalid macro argument {tok.value!r}"
                                )
                            break
                        # Expect a comma or end of list.
                        for tok in intoks:
                            if tok.type is self.TokType.CPP_COMMA:
                                seen_comma = tok
                            else:
                                prep.on_error_token(tok, "Expected ',' or ')'.")
                                return
                            break
                    if seen_comma:
                        prep.on_error_token(seen_comma, "Missing name after comma.")

                argnames = list(iter_argnames())
                m = FuncMacro(prep, name, defn.strip(), argnames, variadic)

            else:
                prep.on_error_token(name,"Bad macro definition")
                return None
            # OK.  Either an object or a function macro.
            # Check for redefinition -> error message but use new definition.
            name = m.name
            if name in self:
                older = self[name]
                if not m.sameas(older):
                    self.prep.on_error_token(
                        m.nametok,
                        f"Macro {name} redefined with different meaning."
                        )
            self[name] = m
            return m

        except:
            traceback.print_exc()
            print('\a')
            raise

    def defined(self, name: str) -> bool:
        return name in self

    def undef(self, name: str) -> None:
        if name in self:
            del self[name]

    def expand(self, intoks: TokIter,  **kwds) -> TokIter:
        """
        Completely macro expands a token iterator, and generates expanded
        tokens.  See the MacroExp() constructor for explanation of arguments.

        This is recursive.  A recursive call comes from expanding an argument
        to a function macro.
        """
        expander = MacroExp(self, intoks, **kwds)
        with self.expanders.nest(expander):
            return expander()

    @property
    def expander(self) -> MacroExp | None:
        """ Current expander, if any. """
        return self.expanders.top()

    @property
    def depth(self) -> int:
        """ Depth of nested expand() calls. """
        return self.expanders.depth

    @property
    def top(self) -> Source | None:
        """ Current expander (if any) top source. """
        if self.expanders:
            return self.expander.top
        else:
            return None

    # For debugging or debug logging.
    @contextlib.contextmanager
    def nest(self, exp: MacroExp) -> ContextManager[None]:
        """
        Run the context with given expander and prep.nesting.  self.depth
        incremented.
        """
        with self.prep.nest(), self.expanders.nest(exp):
            try:
                yield
            finally:
                pass

    @property
    def nested(self) -> bool:
        """ If in expand() called within another expand(). """
        return self.depth > 1

    # Break only if self.depth in this, or if is empty container.
    break_depth: Container[int] = ()
    # Break only if self.top
    break_top: bool = True

    # Method to use as a breakpoint condition.
    def brk(self, obj: Any = None) -> bool:
        """ True if debugger should break.
        Test given object, if any, and self.depth.
        """
        return ((obj is None or obj.brk())
                and break_in_values(self.break_depth, self.depth)
                and (not self.break_top or self.top)
                )

    def __repr__(self) -> str:
        return f"{'*' * (self.depth - 1)} len={len(self)}"


class MacroCall:
    """
    An invocation of a macro.  Consumes all intoks in the argument list with
    the constructor, which also generates intoks it doesn't use.  Can
    produce the replacement intoks with the subst() method.

    Can deliver expanded or stringized individual arguments.

    Also used for any __VA_OPT__ macros within the macro so that each argument
    is expanded or stringized only once.
    """

    # The macro being called
    m: Macro

    # Argument tokens for each argnum, if there is an argument list.
    args: list[Tokens] = None

    # The last token in the call.  Either the name token or the closing ')' of
    # the argument list
    endtok: PpTok

    nametok: PpTok                  # The macro name in the invocation.
    error: str = ''                 # Message if there was an error.

    # The expansion manager that created this MacroCall.
    exp: MacroExp

    # True if the macro will be expanded.  False for a function macro without
    # (), or if already expanding the macro, or any error.
    expanding: bool = True

    # Hidden names to be applied to all tokens in resulting expansion.
    hide: Hide                    

    # The argument list, for a function macro, if present.  None otherwise.
    args: MacroArgs = None

    def __init__(self, m: macro, nametok: PpTok, intoks: TokIter,
                 hide: Hide, exp: MacroExp,
                 ):
        self.m = m
        self.nametok = nametok
        self.hide = hide
        self.exp = exp
        if m.is_func:
            # A function macro.  Go collect the arguments.
            self.args = MacroArgs(self, intoks)
            self.endtok = self.args.endtok
            self.expanding = self.args.expanding
            if self.expanding:
                hide = hide and hide & self.args.endtok.hide
        else:
            self.endtok = self.nametok
        if hide:
            self.hide = hide.add(m.name)
            if m.name in hide:
                self.expanding = False
        else:
            self.hide = Hide(m.prep, m.name)

    def set_error(self, msg: str) -> None:
        if not self.error:
            self.error = msg
            self.expanding = False

    @functools.cached_property
    def first_nonempty_argnum(self) -> int | None:
        for i, arg in enumerate(self.args):
            if arg: return i
        return None

    def subst(self, opt: Macro = None) -> Iterator[PpTok]:
        """
        Result of substituting the macro replacement list using call
        arguments.  Use the macro in self, unless a __VA_OPT__(...) macro is
        provided instead.  All result tokens are new copies.
        """
        s = TokSubstMgr(self)
        return s(opt or self.m)

    def dump(self, leader:str = '') -> None:
        """ Prints details. """
        print(f"{leader}MacroCall {self!r}")
        leader += "  "
        print(f"{leader}name = {self.nametok!r}")
        print(f"{leader}hide = {self.hide!r}")
        if self.m.is_func:
            if self.nametok.value in self.nametok.hide:
                print(f"{leader}Already expanding")
            elif self.args is not None:
                for param, arg in zip(self.m.param_names, self.args):
                    print(f"{leader}{param} =")
                    for tok in arg:
                        print(f"{leader}  {tok!r}")
            else:
                print(f"{leader}No arg list")
        if self.m.is_func: print(f"{leader}{self.rparen!r}")

    def __repr__(self) -> str:
        rep = self.m.name
        if self.args is None:
            if self.m.is_func:
                rep += " <no arg list>"
        else:
            rep = f"{rep} ({', '.join(map(str, self.args))})"
        rep = f"{rep} {self.nametok.loc.showpos}"
        return rep


class MacroArgs:
    """
    Holds the actual arguments used to call a FuncMacro.  Also produces macro
    expanded and stringized arguments.

    If the macro replacements include a __VA_OPT__(...) expression, this same
    MacroArgs is used in substituting the (...) tokens.

    Expanded and stringized arguments are cached and calculated only once,
    including both the macro and all of its __VA_OPT__(...) expressions.
    """

    '''
    Possible outcomes from the constructor.  These do not change later:
                            args            endtok      expanding   error
    1. Valid arg list.      list of args    closing ')' True        None
    2. Invalid arg list.    list of args    closing ')' False       message
    3. No closing ')'.      list of args    None        False       message
    4. No argument list.    None.           call name   False       None

    '''
    call: MacroCall

    # Argument tokens for each argnum, if there is an argument list.  None if
    # arg list invalid, or during constructor before opening '(' seen.
    args: list[Tokens] = None

    # Message if there was an error in the arg list (other than being absent).
    error: str = ""

    # ')' token at the end of the arg list, if there is one, else None.
    _rparen: PpTok = None

    # Caches for expanded and stringized arguments.  The attribute is missing
    # unless it is needed for at least one argument, and then only the
    # necessary arg numbers are keys.

    expanded: Mapping[int, Tokens]              # Expansion of argument

    # Stringize of argument (just the string, not a token).
    strings: Mapping[int, str]
    
    # Any tokens in the argument list which are not used, but rather to be
    # passed through to higher levels.  Set by constructor, deleted by
    # self.passthrus().
    extras: Tokens

    def __init__(self, call: MacroCall, argtoks: TokIter):
        self.call = call
        self.extras = Tokens(self._getargs(argtoks))

    def __getitem__(self, argnum: int) -> Tokens:
        """ Get the given argument number.  None if no argument list. """
        if self.args is None:
            return None
        else:
            return self.args[argnum]

    @TokIter.from_generator
    def passthrus(self) -> Iterator[PpTok]:
        """ Generate unused tokens from constructor, then delete them. """
        yield from self.extras
        del self.extras

    @property
    def expanding(self) -> bool:
        """ The macro should be expanded. """
        return self.args is not None and not self.error

    @property
    def endtok(self) -> PpTok:
        """
        The token which ends the call, which could be just the bare macro name
        with no argument list.
        """
        return self._rparen or self.call.nametok

    def expansion(self, argnum: int, with_spacing: bool = True) -> TokIter:
        """
        The expansion for the given argument #.  The expansion of the argument
        is created only once.  Generates copies of the expansion tokens.
        However, if not `with_spacing`, OR argnum is the first nonempty arg,
        then spacing is removed from the first token of the return value,
        although the stored arg expansion still has it.
        """
        # Resulting expansion.
        exp: Tokens
        prep = self.call.m.macros.prep

        def getexp() -> Tokens:
            """ Get expanded tokens for argnum, calculated the first time. """

            try:
                # self.expanded exists, and [argnum] does also.
                return self.expanded[argnum]
            except AttributeError:
                # self.expanded does not exist.  Create it now.
                self.expanded = dict()
            except KeyError:
                # self.expanded exists, but self.expanded[argnum] does not.
                pass
            # First time for this argnum.
            # Create the expansion and add it to self.expanded.
            toks = TokIter(self.args[argnum])
            toks = self.call.m.macros.expand(toks)
            with prep.nest():
                prep.log.write(
                    f"Expanding arg {self.call.m.param_names[argnum]}",
                    token=self.call.nametok
                    )
            exp: Tokens
            exp = self.expanded[argnum] = toks.get_tokens()
            return exp
            # end of getexp().

        exp = getexp()
        toks = (tok.copy() for tok in exp)
        if (not with_spacing
            or self.call.first_nonempty_argnum == argnum
            ):
            # Remove spacing for the first token (if any).
            for tok in toks:
                # This is first token.
                yield tok.without_spacing()
                break
        # Get the remaining tokens verbatim.
        yield from toks
        return

    def string(self, argnum: int) -> str:
        """
        The stringization (calculated only once) for the given argument
        tokens.
        """
        string: str

        try:
            return self.strings[argnum]
        except AttributeError:
            # self.strings does not exist.  Create it now.
            self.strings = dict()
        except KeyError:
            # self.strings exists, but self.strings[argnum] does not.
            pass
        # First time for this argnum.
        # Create the string value and add it to self.strings.
        string = self.strings[argnum] = FuncMacro.stringize(self.args[argnum])
        return string

    @TokIter.from_generator
    def _getargs(self, intoks: TokIter, *,
                _prednorm = operator.attrgetter('type.norm'),
                ) -> Iterator[PpTok]:
        """
        Parse argument list from input intoks.  Generate any passthru tokens.
        """
        m: Macro = self.call.m

        args: list[Tokens] = []
        nparams = m.nparams
        varnum = m.variadic and nparams - 1
        nesting = 0

        def argtoken(tok: PpTok, arg: Tokens) -> None:
            """
            Handle token that is part of an arg.  Spacing and indent may
            be modified.
            """
            if not arg:
                # First token.  Remove spacing
                tok = tok.without_spacing()
            arg.append(tok)

        def gen_arglist() -> Iterator[PpTok]:
            """
            Generate all tokens after opening '(' up to and including matching
            ')'.  Closure variable `nesting` tracks inner '('s not yet
            matched.
            """
            nonlocal nesting
            nesting = 0
            for intok in intoks:
                yield intok
                if intok.value == ')':
                    if not nesting: return
                    nesting -= 1
                elif intok.value == '(':
                    nesting += 1

        def gen_arg() -> Iterator[PpTok]:
            """
            Generates argtoks for next argument.  The arg terminates with
              - unnested ')', or
              - unnested ',' other than in __VA_ARGS__.
            Consumes all tokens up to the terminator and generates the tokens
            before the terminator.

            Set self._rparen if terminator is ')'.  Set self.error if ran out
            of argtoks.
            """

            for tok in argtoks:
                # Looking for terminator at top nesting level
                if not nesting:
                    if tok.value == ',':
                        # Comma is not part of the arg if it is within the
                        # __VA_ARGS__.
                        if not (m.variadic
                                and (len(self.args) == nparams)
                                ):
                            # The comma is not part of the arg.
                            return
                    elif tok.value == ')':
                        self._rparen = tok
                        # The ')' is not part of the arg.
                        return
                # Part of the arg.
                yield tok

            # End of tokens reached before end of the arg list
            self.set_error(f"Macro {m.name!r}"
                            " missing ')' in argument list."
                            )

        argtoks = gen_arglist()

        argtok: PpTok

        # Looking for argument list.  Directives are handled specially, but
        # differently before and after the opening '('.  Tokens after the '('
        # won't have any indents.

        with self.call.nametok.lexer.inmacro(self):
            tok: PpTok = intoks.peek()
            if tok and tok.value == '(':

                next(intoks)                # Consume the '('

                # Now within the arg list, so do special lexing differently.
                self.args = args = []
                nargs = 0

                # Loop for each argument.
                while not self._rparen and not self.error:
                    arg = Tokens()
                    args.append(arg)
                    nargs += 1
                    for argtok in gen_arg():
                        if not argtok.type.norm:
                            yield argtok
                            continue
                        argtoken(argtok, arg)

                # Close of argument list.

                # Validate arg count.
                if nargs != nparams:
                    if m.variadic:
                        # Variadic args allowed to be one short.
                        #   Supply trailing arg.
                        if nargs == nparams - 1:
                            args.append(Tokens())
                            nargs += 1
                        else:
                            self.set_error(
                                f"Macro {m.name!r} requires at least "
                                f"{nparams} argument(s) "
                                f"but was passed {nargs}.")
                    elif nargs == 1 and not args[0] and nparams == 0:
                        # Empty only arg is OK if no params.
                        del args[:]
                    else:
                        self.set_error(f"Macro {m.name!r} requires exactly "
                                f"{nparams} argument(s) "
                                f"but was passed {nargs}.")
                # Done.  We have all the args.
            return


    def set_error(self, msg: str) -> None:
        if not self.error:
            self.error = msg
            self.call.set_error(msg)

    def __repr__(self) -> str:
        rep = self.call.m.name
        if self.args is None:
            rep += " <no arg list>"
        else:
            rep = f"{rep} ({', '.join(map(str, self.args))})"
        return rep


class TokSubst(abc.ABC):
    """ Performs substitution for a single token, or sequence of tokens,
    in the replacement list of a Macro.  When called with the details of a
    macro invocation, it iterates over the tokens resulting.  These are newly
    created tokens.

    """
    # A token in the macro's replacement list, or the first of a sequence.
    tok: PpToken

    is_paste: ClassVar[bool] = False    # True for TokSubstPaste.
    is_param: ClassVar[bool] = False    # True for TokSubstParam.
    do_pad: ClassVar[bool] = True       # Generate padding before and after.

    def __init__(self, tok: PpTok, after_paste: bool = False):
        if after_paste:
            tok = tok.without_spacing()
        self.tok = tok

    @abc.abstractmethod
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates substituted tokens, not including any padding. """
        ...

    def pad_toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates substituted tokens, including any padding. """
        # Base class has no padding.
        return self.toks(mgr)

    @staticmethod
    @TokIter.from_generator
    def frame_pads(ref: PpTok, toks: Iterable[PpTok]) -> Iterator[PpTok]:
        """
        Generate the given tokens, preceded and followed by pads.  A φ is
        supplied for an empty iterator.
        """
        tok: PpTok
        try: tok = next(toks)
        except StopIteration: tok = ref.make_marker()
        # First token, or new marker.  Gets sep before itself.
        tok = tok.add_sep(ref.sep)
        prev = tok
        for tok in toks:
            yield prev
            prev = tok
        # Last token.  Gets sep after itself.
        yield tok

    def __repr__(self) -> str:
        return repr(self.tok)


class TokSubstPadded(TokSubst):
    """ Base class for subst which generates surrounding padding tokens. """

    after_paste: bool = False

    def pad_toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """
        Generates tokens, surrounded by padding tokens.  See
        TokIter.frame_pads().
        """

        toks = self.toks(mgr)
        toks = self.frame_pads(self.tok, toks)
        yield from toks


class TokSubstLiteral(TokSubst):
    """ An ordinary token, which substitutes to itself. """

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token(s) for this subst. """
        yield self.tok


class TokSubstParam(TokSubstPadded):
    """ Substitutes for a parameter name token.
    Generates the expansion of the corresponding argument, or possibly just
    the argument unexpanded.  If there are no intoks, generate a φ token,
    which is significant if later being pasted or stringized.  With preceding
    whitespace from either the expansion or the name token.
    """
    is_param: ClassVar[bool] = True     # True for TokSubstParam.

    # Should be expanded.  False if adjacent to a paste or stringize.
    expand: bool = True

    argnum: int         # The index of the parameter in macro parameter list.

    # This follows a π subst.  If the argnum is the first non-empty argument
    #   (which may be different each time this macro is called), the spacing
    #   is dropped.  This mimics GCC's behavior.  
    # If not, then the spacing is always dropped.  
    # 
    # Note, 'spacing' refers only to the first token.
    after_paste: bool

    def __init__(self, tok: PpToken, argnum: int, after_paste: bool = False):
        super().__init__(tok, after_paste)
        self.argnum = argnum
        self.after_paste = after_paste

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr, **kwds,
             ) -> Iterator[PpTok]:
        """
        Generates new token(s) for this subst.  A non-expanding empty
        parameter generates a single placemarker.
        """
        res: TokIter
        argnum = self.argnum
        arg: Tokens = mgr.call.args[argnum]
        if not arg:
            mgr.prep.log.write(
                f"Expanding arg {mgr.call.m.param_names[argnum]}",
                token=mgr.call.nametok
                )
            yield self.tok.make_marker()
            return
        if self.expand:
            res = TokIter(mgr.expansion(
                          argnum, arg, with_spacing=self.after_paste))
        else:
            # Preceded and/or followed by π.
            res = TokIter(arg)

        # The first token's spacing may need adjustment.  This is handled by
        # the mgr.
        #  
        # If not after_paste: Get spacing from the parameter name token.  
        # 
        if self.tok.sep and not self.after_paste:
            res = mgr.add_sep_first(self.tok.sep, res)
        yield from res


class TokSubstPaste(TokSubst):
    """ Substitutes a π token, which in the second pass of the TokenSubstMgr
    pastes the tokens on either side.
    """
    is_paste: ClassVar[bool] = True     # True for TokSubstPaste.

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token(s) for this subst.
        Simply generates a π token, which the manager will interpret.
        """
        yield self.tok


class TokSubstString(TokSubst):
#class TokSubstString(TokSubstPadded):
    """
    Substitutes for a 'σ' token plus a parameter name or __VA_OPT__ (...)
    expression.
    """
    arg: TokSubstParam | TokSubstVaOpt
    argnum: int | None              # None is for a VA OPT.

    def __init__(self, tok: PpToken, arg: TokSubst,
                 after_paste: bool = False):
        super().__init__(tok, after_paste)
        self.arg = arg
        self.argnum = arg.argnum

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """
        Generates new tokens for this subst.  A single string token with
        framing pads.
        """
        argnum = self.argnum
        vaopt: bool = argnum is None
        if vaopt:
            # self.arg is a VaOpt.
            toks = self.arg.toks(mgr)
            value = FuncMacro.stringize(toks)
        else:
            # self.arg is a Param.
            value = mgr.call.args.string(argnum)
        mgr.prep.log.stringize(self.arg.tok.value, value, self.arg.tok)
        tok: PpTok = self.tok.make_string(value)
        if vaopt:
            # clang carries the separation of the σ over to whatever is next.
            # If value is empty, clang also carries the separation of the VO
            # over to whatever is next.
            argsep: TokenSep = self.tok.sep
            if not value:
                argsep += self.arg.tok.sep
                if argsep: tok = tok.spacing_after()
        else:
            # clang ignores separation of the Param
            pass
        yield from self.frame_pads(tok, iter([tok]))


class TokSubstVaOpt(TokSubstPadded):
    """
    Substitutes for entire "__VA_OPT__ ( repl list )".  Whitespace around the
    repl list is ignored.  Expands the replacement list from __VA_OPT__, if
    __VA_ARGS__ is non-empty, otherwise a single φ token.
    """
    argnum: ClassVar[int] = None    # None indicates a __VA_OPT__ "parameter".
    repl: FuncMacro                 # Macro created from the repl list.                
    after_paste: bool = False       # True if follows a paste

    def __init__(self, tok: PpToken, repl: FuncMacro,
                 after_paste: bool = False):
        super().__init__(tok, after_paste)
        self.repl = repl
        self.after_paste = after_paste

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token(s) for this subst. """
        res = TokIter(mgr.expand_va_opt(self.repl))
        yield from res


# The following classes are for dynamic macros, such as __LINE__, 
#   whose replacement varies with where and when they are called ...

class TokSubstCounter(TokSubstPadded):
    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """
        Generates new token for __COUNTER__ macro.  Increments each time it is
        expanded.
        """
        prep = mgr.prep
        t = mgr.call.nametok.copy(
            type = prep.t_INTEGER,
            value = prep.t_INTEGER_TYPE(prep.macros.countermacro),
            patt = re.compile(prep.lexer.REs.int),
            )
        prep.macros.countermacro += 1
        yield t


class TokSubstFile(TokSubstPadded):
    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """
        Generates new token for __FILE__ macro.  This changes with the current
        source file or a #line directive.
        """
        t = mgr.call.nametok
        prep = mgr.prep 
        filename = mgr.orig_call().nametok.loc.output_filename
        t = t.copy(
            type=mgr.prep.t_STRING,
            value=f'"{prep.fix_path_sep(filename)}"',
            )
        yield t


class TokSubstLine(TokSubstPadded):
    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token for __LINE__ macro. """
        t = mgr.call.nametok
        prep = mgr.prep
        line = mgr.orig_call().endtok.loc.output_lineno
        t = t.copy(
            type=prep.t_INTEGER,
            value=prep.t_INTEGER_TYPE(line),
            patt=re.compile(prep.lexer.REs.int),
            )
        yield t

# Maps the class of the token substitutor for each dynamic macro name.
dynamic_substs_tab: Mapping[str, TokSubst] = dict(
    __COUNTER__=TokSubstCounter,
    __FILE__=TokSubstFile,
    __LINE__=TokSubstLine,
    )


class TokSubstMgr:
    """
    Performs the equivalent of Prosser's subst() function.

    It takes a MacroCall and the call operator generates PpTok's.

    For function macros, it substitutes for parameter names and
    __VA_OPT__(...) expressions.  Expands and stringizes these on demand, but
    does each parameter only once.  Result is iteration of intoks.

    For object macros, the intoks in the replacement list itself are used.

    Then in either case, it performs paste operations, for both object and
    function macros.
    """

    # Set by the constructor...
    prep: Preprocessor
    m: Macro
    call: MacroCall
    if __debug__:
        _serial = itertools.count(1)
        ser: int

    def __init__(self, call: MacroCall):

        self.m = m = call.m
        self.prep = m.prep
        self.call = call
        if __debug__: self.ser = next(self._serial)

    @TokIter.from_generator
    def __call__(self, m: Macro, nametok: PpTok = None) -> Iterator[PpTok]:
        """ Equivalent of subst() in Prosser's algorithm.
        Generates the replacement intoks for the macro call.

        If substituting the __VA_OPT__(replacement list) for this macro, then
        the `m` argument is made from that replacement list and nametok is the
        __VA_OPT__ token.

        Also adds to the hide sets of all generated tokens.
        """
        """
        Three-pass substitution.

        First pass generates tokens for all the m.substs objects.  However, a
        paste in the original replacement list becomes simply a π token.  An
        object macro has no subst objects, and the original replacement tokens
        are used.  Parameter names are expanded or stringized, when required.

        This pass varies with the type of macro being called.  The result is a
        sequence of tokens with some π tokens.

        Second pass merges adjacent [] tokens.

        Third pass (if there are π tokens) performs the pasting by finding π
        tokens and replacing them and their adjacent tokens (and any padding in between) with the
        concatenation tokens.
        """

        # First pass.
        intoks = TokIter(m.subst_padded(self, nametok))

        prev: PpTok

        toks: TokIter
        if not m.has_paste:
            toks = intoks
        else:
            # If there are any pastes:
            toks = intoks
            #toks = self.attach_pads(intoks)
            # Second pass (clang only).  Change some pastes as clang does.

            if self.prep.lang.clang:
                toks = self.filt_pastes_clang(toks)

            # Third pass.  Perform pastes.
            toks = self.do_pastes(toks)

        # Finally, augment the hide sets of all results.
        hide = self.call.hide
        for tok in toks:
            tok = tok.add_hide(hide)
            yield tok

    @TokIter.from_generator
    def filt_pastes_clang(self, toks: TokIter) -> Iterator[PpTok]:
        """
        Changes the token stream as clang does before doing the actual pastes.
        Removes any π φ or φ π (along with preceding seps), then changes every
        second π to ##.
        """
        lhs: PpTok
        rhs: PpTok
        prep: PPreprocessor = self.m.prep
        npastes: int = 0        # How many π just generated.

        # Pending lhs, waiting to see if it is π followed by φ.
        pend_lhs: PpTok = None

        for lhs in toks:
            ltype: TokType = lhs.type
            if ltype.paste:
                rhs = next(toks)
                rtype: TokType = rhs.type
                # π rhs
                if rtype.marker:
                    # π φ.
                    # Remove π φ.
                    lhs = None
                else:
                    # rhs will be next lhs.
                    toks.putback(rhs)
            elif ltype.marker:
                rhs = next(toks, None)
                if rhs:
                    rtype: TokType = rhs.type
                    # φ rhs
                    if rtype.paste:
                        # φ π.
                        # Move any • spacing from φ to following token.
                        if lhs.spacing:
                            toks = toks.apply_first(
                                lambda tok: tok.add_spacing())
                        # Remove φ π.
                        lhs = None
                        # Remove pending π, if any.
                        if pend_lhs and pend_lhs.type.paste:
                            pend_lhs = None
                    else:
                        toks.putback(rhs)
                else: assert rhs is None
            if lhs:
                if lhs.type.paste:
                    if npastes & 1:
                        lhs = lhs.copy(type=prep.TokType.CPP_DPOUND)
                    npastes += 1
                else:
                    npastes = 0
                if pend_lhs:
                    yield pend_lhs
                pend_lhs = lhs
        if pend_lhs:
            yield pend_lhs

    @TokIter.from_generator
    def do_pastes(self, toks: TokIter) -> Iterator[PpTok]:
        """
        Perform any π pastes contained within the tokens and generate new
        tokens, otherwise pass on the tokens unchanged.  Any sequence of (lhs,
        π, rhs) pastes lhs with rhs (neither of which is π) and the result
        will be next lhs.  If the paste fails, generate lhs and use rhs as the
        next lhs.

        Each token is either an ordinary token, π, or [].  π will not be
        either the first or the last token.  Any padding which preceded the
        token is not put in token.pads.  Padding tokens will be placed around
        each paste operation, even if failed.
        """
        prep: PPreprocessor = self.m.prep

        '''
        State of iteration is denoted by O • (tokens being processed) • I.

        O = tokens already generated.  I = input tokens not yet received.
        Either of these can be {} if known to be empty.

        Initial state = {} • • I.  Final state = O • • {}.
        '''

        '''
        Transitions implemented:

        L and R are ordinary tokens.  Δ represents the preceding padding plus
        the token sep.

        Empty.  {} • • {} -> ends the function with no tokens generated.

        Initial.
          - {} • • Δ L I           -> Δ L • I.

        Middle.
          - O • Δ L •      Δʺ R I -> O Δ L • Δʺ R • I.
          - O • Δ L • Δʹ π Δʺ R I -> O     • Δ LR • I   if paste LR is valid.
          - O • Δ L • Δʹ π Δʺ R I -> O Δ L • Δʺ R • I   if paste LR not valid.

        Final.
          - O • Δ L • {}           -> O Δ L • • {}, ends the function.

        '''
        # Initial state, {} • • O.

        # Get first token, if any.

        lhs = next(toks, None)
        if lhs is None:
            # {} • • {}.  No tokens at all.
            return
        #self.log(lhs, "pastes lhs")

        # First lhs involved in a paste.
        pasting: PpTok = None

        # Handle remaining tokens one at a time.  lhs gets updated sometimes.

        # O • lhs • I

        for rhs in toks:

            # O • lhs rhs • I

            rtype: TokType = rhs.type

            if rtype.paste:
                # O • lhs π • rhs I
                op = rhs
                rhs = next(toks, None)
                assert rhs, "Paste operator at the end of macro."
                if not pasting:
                    # First time, emit a padding for lhs, including pads (if
                    # any) attached to lhs already.
                    pasting = lhs

                # O • lhs π rhs • I

                # Try pasting lhs and rhs.  If OK, update lhs and generate
                # nothing.  Otherwise generate lhs and use rhs as next lhs.
                t: PpTok | None
                # clang removes leading space from lhs if it came from failed
                # paste the last time.
                if lhs.type.id and rhs.type.id:
                    # Fast track for a common use case.
                    t = lhs.copy(value=lhs.value + rhs.value,
                                type=lhs.type,
                                hide=lhs.hide and rhs.hide
                                and lhs.hide & rhs.hide)
                elif lhs.type.marker:
                    lhs = rhs
                    continue
                elif rhs.type.marker:
                    continue
                else:
                    t = lhs.lexer.try_paste(lhs, rhs)
                prep.log.concatenate(t, lhs, rhs, nest=1)
                if t:
                    # The paste succeeded.  t = copy of lhs with new value and
                    # type.
                    lhs = t
                    continue
                # Paste failed.
                if lhs.sep_after:
                    del lhs.sep_after
                pasting = None

            # Either lhs rhs or failed lhs π rhs.
            yield lhs
            lhs = rhs

        # End of rhs in toks loop.
        # The last token.
        yield lhs


    # Helper methods used by the TokSubst's ...

    def expansion(
            self, argnum: int, arg: list[PpTok],
            with_spacing: bool = True
            ) -> Tokens:
        """
        The expansion (calculated only once) for the given argument #.  Caller
        must not modify the return result, but can make a copy.  If not
        `with_spacing`, OR argnum is the first nonempty arg, then spacing is
        removed from the first token of the return value, although the stored
        arg value still has it.
        """
        # Resulting expansion.  May have initial whitespace removed.
        exp: Tokens
        prep = self.prep
        exp = self.call.args.expansion(argnum, with_spacing)
        toks = TokIter(exp)
        if (not with_spacing
            or self.call.first_nonempty_argnum == argnum
            ):
            # Remove spacing for the first token (if any).
            for tok in toks:
                # This is first token.
                yield tok.without_spacing()
                break
        # Get the remaining intoks verbatim.
        yield from toks

    def expand_va_opt(self, repl: FuncMacro
                      ) -> Iterator[PpTok]:
        """ Generates intoks from a __VA_OPT__ (toks) expression,
        Expands the toks as if it were the replacement for calling the same
        macro, except if the expansion of __VA_ARGS__ is empty, a single φ
        token is generated.
        """
        varargnum: int = self.m.nparams - 1
        varargs = self.call.args[varargnum]
        for tok in self.expansion(varargnum, varargs):
            if tok.type.marker:
                break
            # __VA_ARGS__ expands to something
            toks: TokIter = self(repl, repl.nametok)
            #tokens = toks.copy_tokens()
            for tok in toks:
                yield tok
            return
        # __VA_ARGS__ expands to nothing or a marker.
        yield repl.nametok.make_marker()

    def add_sep_first(self, sep: TokenSep, toks: TokIter) -> TokIter:
        """
        Changes the given token iterator by replacing its first token with one
        with given separator added to it.  Returns new iterator.
        """
        if sep:
            return toks.apply_first(operator.methodcaller('add_sep', sep))
        return toks

    def orig_call(self) -> MacroCall:
        """ The latest macro call which didn't come from an expansion. """
        return self.prep.macros.expander.orig_call

    def log(self, tok: PpTok, msg: str) -> None:
        """ Log message at token location, with token details. """
        log = self.prep.log
        #try: pads = ''.join(str(pad) for pad in tok.pads)
        #except AttributeError: pads = ''
        log.msg(f"{msg} {tok.show_pads()}", tok)
        #log.msg(f"  id = {id(tok):X}", tok)
        #try:
        #    pads = tok.pads
        #    log.msg(f"  pads = {id(pads):X}", tok)
        #    for pad in pads:
        #        log.msg(f"    {pad!r}", tok)



    if __debug__:
        def brk(self) -> bool: return self.call.nametok.brk()

    def __repr__(self) -> str:
        if __debug__: ser = f"{self.ser} "
        else: ser = ""
        return f"<{ser}{self.call}>"
        return f"<{self.call}>"

#from pcpp.prosser import Prosser

