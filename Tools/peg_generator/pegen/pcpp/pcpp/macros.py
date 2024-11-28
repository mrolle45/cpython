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
from pcpp.tokens import (PpTok, Tokens, TokIter, TokenSep, Hide)

''' The following is modeled after David Prosser's algorithm, which can be found at
https://www.spinellis.gr/blog/20060626/cpp.algo.pdf
The expand() function is performed by the Macros.expand() method.
The subst() function is performed by the TokSubstMgr.__call__() method.
    This incorporates the hsadd() call in that method.
'''

dumpme: bool = False

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
#    .arglist   - List of argument names, including __VA_ARG__ and __VA_OPT__
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
    has_paste: ClassVar[bool] = False       # Contains any π tokens
    error: ClassVar[str] = None             # Error message if invalid.

    def __init__(self, prep: Preprocessor, nametok: PpTok, value: TokIter):
        self.nametok =  nametok
        self.value = Tokens(value)
        if any(map(operator.attrgetter('type.paste'), self.value)):
            self.has_paste = True
        for t in self.value:
            t.exp_from = self
        self.macros = prep.macros

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

    @property
    def log(self) -> _DebugLog:
        return self.macros.log

    def set_error(self, tok: PpTok, msg: str) -> None:
        self.prep.on_error_token(tok, msg)
        self.error = msg

    def dynamic(self) -> None:
        """ Initialize a dynamic Macro. """

        self.substs.append(dynamic_substs_tab[self.name](self.nametok))

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

    def subst_tokens(self, mgr: TokenSubstMgr) -> Iterator[PpTok]:
        """
        Do all substitution operations except for pasting.  Object macros have
        no parameters to substitute, so this method is a pass-through of the
        replacement intoks themselves.  Each token has its call_from set.
        """
        for tok in self.value:
            yield tok.copy(call_from=mgr.call)

    @TokIter.from_generator
    def fix_paste_substs(self) -> Iterator[PpTok]:
        for tok in self.value:
            if tok.type.dhash:
                yield tok.copy(type=self.prep.TokType.CPP_PASTE)
            else:
                yield tok

    def __repr__(self):
        return f"{self.name}={self.value!r}"


class FuncMacro(Macro):
    is_func: ClassVar[bool] = True
    substs: tuple[TokSubst, ...]            # The replacement list generators.
    arglist: list[str]              # Names of params, plus __VA_OPT__.
    nargs: int                      # len or arglist, without __VA_OPT__.

    def __init__(self, prep: Preprocessor, nametok: PpTok, value: Tokens,
                 arglist, variadic: bool):
        self.arglist = arglist
        self.nargs = len(arglist) - variadic
        self.variadic = variadic

        super().__init__(prep, nametok, value)
        self.make_substs()

    def make_substs(self) -> None:
        """
        Build substitution items in self.substs.  These are immutable and
        reused every time the macro is invoked.

        A substitution item can be:
          - StringTokSubst -- Σ + (a parameter name or __VA_OPT__).
          - PasteTokSubst -- a π token.
          - VaOptTokSubst -- a "VA_OPT ( ... )" expression.
          - ParamTokSubst -- a parameter name.  Might be adjacent to a π.
          - PlainTokSubst -- a single token, not any of the above.
        Note, if a Param or VaOpt is adjacent to a Paste, then it is set to
        avoid expansion.
        """
        repltoks: Iterator[PpTok] = iter(self.value.data)

        self.substs = []
        param_names: list[str] = self.arglist

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
            either added to the output or ignored.  `no_ws` means that a Param
            subst will strip leading whitespace.
            """
            tok = next(repltoks, None)
            if not tok:
                return None
            val = tok.value
            keep_spacing = True
            typ = tok.type
            if typ.paste:
                # '##'.  Alters previous subst, which must exist.
                if not prev:
                    self.set_error(
                        tok,
                        "'##' cannot be at the start of macro replacement." )
                    return None
                prev.expand = False
                return PasteTokSubst(tok)
            elif typ.stringize:
                # Stringize.  Get following param name or __VA_OPT__.
                try:
                    param = next(repltoks)
                    argnum = param_names.index(param.value)
                    if self.variadic and argnum == self.nargs:
                        opt = make_va_opt(tok)
                        return StringTokSubst(tok, opt)
                        
                    param = ParamTokSubst(param, argnum)
                    param.expand = False
                    return StringTokSubst(tok, param, argnum)
                except:
                    # Did not find following parameter name.
                    self.set_error(
                        tok,
                        "'#' must be followed by parameter name "
                        "or __VA_OPT__." )
                    return None

            elif val == '__VA_OPT__':
                return make_va_opt(tok)
            elif val in param_names:
                # Parameter name.  Check if following a paste.
                argnum = param_names.index(val)
                subst = ParamTokSubst(tok, argnum, after_paste=after_paste)
                if prev and prev.is_paste:
                    subst.expand = False
                return subst
            else:
                # Anything else.
                return PlainTokSubst(tok)

        def make_va_opt(tok: PpTok) -> VaOptTokSubst:
            """ Create __VA_OPT__ ( ... ) substitution. """
            m : FuncMacro = self.parse_va_opt(repltoks, tok)
            return VaOptTokSubst(tok, m)

        prev: TokSubst = None

        # Loop over intoks to turn into substs.
        while True:
            subst = next_subst()
            if not subst:
                return
            self.substs.append(subst)
            prev = subst
        if prev and prev.is_paste:
            self.set_error(
                tok,
                "'##' cannot be at the end of macro replacement." )

    def subst_tokens(self, mgr: TokenSubstMgr) -> Iterator[PpTok]:
        """
        Do all substitution operations except for pasting.  Iterate resulting
        intoks, which can include paste operators.
        """

        return itertools.chain(
            *map(operator.methodcaller('toks', mgr), self.substs))

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
        # t is first token.
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
            prep, name, repl[:-1], self.arglist[:-1], variadic=True
            )

    @staticmethod
    def stringize(toks: Tokens) -> str:
        """ Implement the '#' operator in a function macro (C99 6.3.10.2). """
        parts: list[str] = []
        has_ws = False
        prev: PpTok = None
        for tok in toks:
            if not tok.value:
                continue
            typ: TokType = tok.type
            if typ.ws:
                if not parts: continue      # Skip initial whitespace.
                has_ws = True               # Start or continue a run of ws.
            else:
                if prev and not tok.hide and tok.lineno != prev.lineno:
                    has_ws = True
                prev = tok
                if has_ws or (parts and tok.spacing):
                    parts.append(' ')
                    has_ws = False
                if tok.repls:
                    # Revert to original spelling of UCNs.
                    orig = tok.reverted_from
                    if orig:
                        tok = orig.revert(unicode=True, orig_spelling=True)
                    else:
                        tok = tok.revert(unicode=True)
                if typ.str:
                    # Escape every " and \.
                    tok.value = tok.value.replace("\\","\\\\").replace('"', '\\"')
                elif typ.chr:
                    # Escape every \.`
                    tok.value = tok.value.replace("\\","\\\\")
                elif tok.value.startswith('//'):
                    tok.value = f"/*{tok.value[2:]}*/"
                parts.append(tok.value)

        # Don't end with a backslash.
        if parts and parts[-1] == '\\':
            parts.append ('\\')
        string = ''.join(parts)
        return f'"{string}"'

    def sameas(self, other: FuncMacro) -> bool:
        """ True if the two definitions are the same (C99 6.10.3p2). """
        if not super().sameas(other): return False
        return self.arglist == other.arglist

    def __repr__(self):
        args = self.arglist
        if self.variadic:
            args = args[:-2] + ['...']
        return f"{self.name}({args})={self.value!r}"


class DynMacro(Macro):
    """ Special macro, such as __LINE__. """
    is_dyn: ClassVar[bool] = True

    def __init__(self, prep: Preprocessor, name: str):
        nametok = next(prep.lexer.parse_tokens(name))
        super().__init__(prep, nametok, Tokens())
        self.subst = dynamic_substs_tab[self.name](self.nametok)

    def subst_tokens(self, mgr: TokenSubstMgr) -> Iterator[PpTok]:
        """
        Do all substitution operations except for pasting.  Dynamic macros
        have no parameters to substitute.  However, they do generate a single
        result token based dynamically on the context in which the macro was
        called.
        """
        yield from self.subst.toks(mgr)

    def __repr__(self) -> str:
        return self.name


class MacroExp:
    """ Handles a single call to Macros.expand() using its __call__(). """

    # The preprocessor.
    prep: Preprocessor

    # Current definitions of macro names.
    macros: Macros

    # Input tokens to consume during the expansion.  Includes macro
    # replacement tokens awaiting rescan.  Empty when expansion is complete.
    intoks: TokIter

    # Token in source file which is being expanded.  Used to evaluate __LINE__
    # and __FILE__.  At the top level, it is the macro name token currently
    # being considered for expansion.  In a directive, it is the directive '#'
    # token.  For a function argument, it is the origin in the MacroExp which
    # is expanding the function.
    origin: PpTok

    # Enclosing MacroCall, if any.
    call_from: MacroCall = None

    # The source where intoks are located, if they are the file's contents.
    # This is called only once for any Source file, and intoks = all tokens
    # generated by that Source by lexing its contents, including from included
    # files (which have already been expanded and are protected from expansion
    # at this level).  None if expanding something else.  
    top: Source = None

    # This is a control expression.  Handle 'defined' expressions and
    # undefined identifiers.
    ctrlexpr: bool = False

    def __init__(self, macros: Macros, intoks: TokIter, *,
        top: Source = None, ctrlexpr: bool = False, origin: TokLoc = None,
        call_from: MacroCall = None,
        ):
        self.macros = macros
        self.prep = macros.prep
        self.intoks = intoks
        self.top = top
        self.origin = origin
        self.call_from = call_from
        self.ctrlexpr = ctrlexpr

    @TokIter.from_generator
    def __call__(self) -> TokIter:
        """
        Expands iterable of intoks, expanding macro names.  After each
        expansion, rescan that expansion plus all following intoks.  Generates
        result iterable of tokens.

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
        with self.macros.nest(top):

            intoks: TokIter = self.intoks
            orig = None
            # Flag to change C++ comment to C comment in some cases.
            # 1. Tok was produced by a macro replacement.  Will have tok.hide.
            # 2. Follows an unexpanded macro and possibly some newlines.
            # This is only for GCC mode.
            fix_comment: bool = False

            intok: PpTok            # Current token, None after empty expansion.

            #for intok in intoks:

            #    typ: TokType = intok.type
            #    if not typ.norm:
            #        yield intok
            #        continue
            #    if not orig or intok.indent: orig = intok

            #    if not intok.hide:
            #        loc = intok.loc
            #    #if intok.indent:
            #    #    loc = intok.indent

            while True:
                for intok in intoks:
                    #if intok.brk():
                    #    toks = Tokens.join(intok, *intoks.copy_tokens())
                    #    print(f"-- {str(toks)!r}")
                    #    res = Prosser(prep.macros, toks)
                    #    print(f"---- {str(res)!r}")
                    if not orig or intok.indent: orig = intok
                    break
                else:
                    return
                # replace_and_rescan replaces intok possibly, and generates
                # any passthru intoks from the input, followed first
                # unexpanded token (if any).
                for intok in self.replace_and_rescan(intok):
                    global dumpme
                    #if intok.brk(): dumpme = True
                    if not intok.type.norm:
                        yield intok
                        continue
                    # intok = the next token
                    break
                else:
                    # Go back and do the next token, keeping same orig.
                    #orig = None
                    continue
                # intok = the next token

                if intok is not orig:
                    self.macros.log.msg(f"End replace.  New token {intok!r}", orig)
                    # orig got replaced with something else.  Source token got
                    # replaced with another token.  Pick up original spacing
                    # and indent.
                    if top and orig.indent and not intok.indent:
                        if intok.hide:
                            indent = orig.indent
                            # gcc and clang add a space at the start of the
                            # line if orig is at the start and intok has spacing.
                            if (prep.emulate
                                and indent.colno == 1
                                and (intok.spacing
                                        #or (indent.lineno == intok.lineno
                                        #    and indent.source is intok.source)
                                        )
                                ):
                                indent = indent.copy(colno=2)
                        else:
                            # Use orig line for intok.indent, keep
                            # intok.colno.
                            indent = intok.loc.copy(lineno=orig.lineno,
                                                phys_offset=0)
                        intok = intok.with_sep(TokenSep.create(indent=indent))
                    elif orig.spacing:
                        intok = intok.with_sep(TokenSep.create(spacing=True))

                elif self.ctrlexpr and intok.type.id:
                    # In control expression, all unreplaced identifiers become
                    # 0 and defined-macro expressions are evaluated.
                    intok = self.repl_undefined(intok)
                #repr(intok.sep)
                orig = None
                yield intok

    def repl_undefined(self, intok: PpTok) -> PpTok:
        """
        Replace an undefined identifier with
        '0', with special treatment for 'defined' expressions.
        """
        prep = self.prep
        prep.log.eval_ctrl(intok)
        if intok.value == 'defined':

            # Undefined behavior (C99 6.10.1).
            if prep.clang:
                # clang evaluates the expression in the usual way.
                # Replace 'defined' in intok + rest of intoks, then get
                # the first result token, which should be 0 or 1.
                intok = self._replace_defined_expr(intok)
                return intok
            else:
                # pcpp will issue a warning and treat it as any
                # other undefined identifier.
                prep.on_warn_token(
                    intok,
                    "Macro expansion of control expression "
                    "contains 'defined'.  "
                    "This is being interpreted as '0'.",
                    )

        return intok.copy(value='0', type=prep.t_INTEGER)
        # end of repl_undefined()


    @TokIter.from_generator
    def replace_and_rescan(self, intok: PpTok) -> Iterator[PpTok]:
        """
        Do macro replacement with rescan for given input token `intok`.
        Generates the next token (if any) to be processed, possibly the same
        token.  May modify intoks if a macro replacement occurs.

        If any passthru intoks are seen while consuming a function argument
        list, these are generated in turn.

        This method _could_ just keep working through the entire input.
        However, the caller will want to add some output positioning
        information and lines that became blank as a result of expansion.

        When intok is a macro subject to replacement:

            Any function argument list is consumed from intoks.

            Any passthru intoks found while consuming the argument list are
            generated in turn.

            Its replacement intoks are put back in the front of intoks.

            If there were any replacement tokens, the first of these is
            consumed from intoks and becomes the new intok.  Then the entire
            process is repeated.

            If there were no replacement tokens, the method ends, without
            generating a result token.

        Otherwise, intok is generated.

        A macro is NOT subject to replacement if:

            It is already being expanded at an outer level.

            A function macro has no argument list, or there's a problem with
            it.

            It is part of a defined-macro expression, which is looking for a
            'defined name' or 'defined ( name )' sequence of tokens.

        """
        m: Macro | None     # Macro for name in intok.value.

        log = self.macros.log
        intoks = self.intoks
 
        while intok:
            # Examine intok, the current token just removed from intoks.  

            # Log this pass.
            log.msg(f"Next token {intok!r}", intok)
            # End the loop if intok is not a macro which is to be expanded.
            if not intok.type.id:
                if intok.type.null:
                    intok = None
                # Not an identifier.  Don't expand.
                break
            name = intok.value
            m = self.macros.get(name)
            if not m:
                # Not a defined macro.  Don't expand.
                break
            # Token is a macro.
            hide = intok.hide
            if hide and name in hide:
                # Hidden.  Don't expand.
                log.expand(intok, m, False, hide)
                break
            if intoks.in_defined_expr:
                # In 'defined...', don't expand.
                break

            if self.top and not intok.hide:
                self.origin = intok
            call = MacroCall(m, intok, intoks, hide, self, self.call_from)
            with self.expanding(call):
                yield from Tokens(call.getargs())

                if call.expanding:
                    try:
                        log.expand(intok, m, True, hide, call=call)
                        if dumpme:
                            call.dump("  " * self.macros.depth)
                        new = call.subst()
                        if new is None:
                            break
                        if __debug__:
                            # Useful for debugging.
                            newtokens: Tokens = new.copy_tokens()
                            if dumpme:
                                leader = "  " * self.macros.depth
                                print(f"{leader}Expansion =")
                                for tok in newtokens:
                                    print(f"{leader}  {tok!r}")

                            newtokens
                        # If there are no replacement intoks, then
                        # quit, with intok = the first token in the
                        # remaining input (if any) or None.  This
                        # allows intok (possibly expanded) to be placed on
                        # a new output line.
                        intok = next(new, None)
                        if not intok:
                            if self.top:
                                # No more replacements, at top level
                                # expansion.  Don't expand.
                                break
                            # Get next remaining input token
                            intok = next(intoks, None)
                            if not intok:
                                # Nothing remaining at all.
                                break
                        else:
                            # Effectively put the replacement intoks
                            # in front of the remaining input intoks.
                            # intok is already the first replacement token
                            # and any remaining replacement intoks are
                            # put in front of the remaining input
                            # intoks.

                            intoks.prepend(new)

                    except:
                        traceback.print_exc()
                        print("Ignoring the exception.\a")
                        break
                elif call.error:
                    # Something wrong with the function call syntax.
                    # Don't expand.
                    self.prep.on_error_token(intok, call.error)
                    intok = None
                    break
                else:
                    # No argument list.  Don't expand.
                    log.expand(intok, m, True, hide)
                    fix_comment = True
                    break
                log.msg(f"Rescanning.", intok)

        yield intok

    def expanding(self, call: MacroCall) -> contextlib.contextmgr:
        if self.top:
            return self.top.expanding(call)
        else:
            return contextlib.nullcontext()

    def _replace_defined_expr(self, deftok: PpTok) -> PpTok:
        """
        Replace 'defined X' or 'defined (X)' with integer 0 or 1, or a string
        to be passed through to the output file.  The 'defined' token is
        given, and the rest of the expression is in self.intoks.

        The token `defined` can come from the original expression, or from a
        macro replcement.  In the latter case, it is undefined behavior and is
        implemented according to prep attributes.
        """

        prep = self.prep

        # Already have "defined".  Look for X or ( X ).

        # In a macro expansion, check prep options.  May emit a warning.
        # May treat as just 0, or may evaluate the expression.
        if prep.emulate and deftok.hide:
            prep.on_warn_token(
                deftok,
                "Macro expansion of control expression "
                "contains 'defined'.  "
                "This is being interpreted as '0'.",
                )

        intoks = self.intoks

        # Trap anything that wants another token from an empty iterator.
        try:
            name = None
            tok = next(intoks)
            # Looking for "name" or "(name)".
            if tok.type.id:
                # Found "name".
                name = tok.value
            elif tok.value == "(":
                tok = next(intoks)
                if tok.type.id:
                    tok2 = next(intoks)
                    if tok2.value == ")":
                        # Found "(name)".
                        name = tok.value
        except StopIteration:
            # The "defined" ... was incomplete.
            pass

        if name:
            # Found the name.
            if name in self.macros:
                result = 1
            else:
                repl = prep.on_unknown_macro_in_defined_expr(tok)
                if repl is None:
                    partial_expansion = True
                    result = f'defined({tok.value}'
                else:
                    result = int(repl)
            if isinstance(result, int):
                tok = tok.copy(value=str(result), type=prep.t_INTEGER)
            else:
                tok = tok.copy(value=result, type=prep.t_ID)
            return tok
        else:
            # Malformed.
            prep.on_error_token(
                deftok, "Malformed 'defined' in control expression")
            return deftok.copy(type=prep.TokType.error)


class Macros(dict[PpTok, 'Macro']):
    """ All the macro definitions in a translation unit --
    top level source file, all included headers, and predefined stuff.

    Definitions are in the form of a token list.
    Expansion generates a token sequence from an input token list.
    """

    def __init__(self, prep: Preprocessor):
        self.prep = prep
        self.lexer = prep.lexer
        self.log = prep.log
        self.TokType = prep.TokType
        self.depth = 0
        self.countermacro = 0
        self.top = None

    def define(self, defn: TokIter,
                **kwds) -> Macro:
        """Define a new macro from intoks following #define
        in a directive."""
        TokIter.check_type(defn, "Macros.define()")
        prep = self.prep
        try:
            # Name is first token, skipping whitespace.
            #x = next(defn)
            defn = defn.strip()
            name = next(defn)
            # Validate the macro name.
            if not name.type.id:
                self.prep.on_error_token(
                    name,
                    f"Macro definition {name.value!r} requires an identifier")
                return None
            if name.value == 'defined':
                self.prep.on_error_token(
                    name, "'defined' is not a valid macro name")
                return None

            defn = defn.strip()
            if name.type is self.TokType.CPP_OBJ_MACRO:
                m = ObjMacro(prep, name, defn, **kwds)
            elif name.type is self.TokType.CPP_FUNC_MACRO:
                # A macro with arguments
                variadic = False
                # Get the argument names.  Gets intoks through the closing ')'.
                def argtokens() -> Iterator[PpTok]:
                    for tok in defn:
                        if tok.value == ')':
                            return
                        yield tok

                def iter_argnames() -> Iterator[str]:
                    """
                    Generate names of arguments.  For variadic, the ...
                    generates both __VA_ARGS__ and __VA_OPT__.
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
                            yield '__VA_OPT__'
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
                    self.prep.on_error_token(m.nametok, f"Macro {name} redefined with different meaning.")
            self[name] = m
            return m

        except:
            traceback.print_exc()
            print('\a')
            raise

    def define_dynamic(self, name: str) -> Macro:
        """ Define a dynamic macro, such as __LINE__. """
        m = DynMacro(self.prep, name)
        self[name] = m
        return m

    def defined(self, name: str) -> bool:
        return name in self

    def undef(self, name: str) -> None:
        if name in self:
            del self[name]

    def expand(self, intoks: TokIter,  **kwds) -> TokIter:
        """
        Completely macro expands a token iterator, and generates expanded
        tokens.  See the MacroExp() constructor for explanation of arguments.
        """
        return MacroExp(self, intoks, **kwds)()

    # For debugging or debug logging.
    @contextlib.contextmanager
    def nest(self, top: Source):
        """
        Run the context with higher self.depth and prep.nesting.  self.depth =
        1 at the top level.
        """
        self.depth += 1
        oldtop = self.top
        self.top = top
        with self.prep.nest():
            try:
                yield
            finally:
                self.depth -= 1
                self.top = oldtop
    @property
    def nested(self) -> bool:
        """ If in expand() called within another expand(). """
        return self.depth > 1

    # Break only if self.depth == or in this, or if is empty container.
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
    the getargs() method, which also generates intoks it doesn't use.  Can
    produce the replacement intoks with the subst() method.
    """
    # The macro being called
    m: Macro

    # Argument intoks for each argnum, if there is an argument list.
    args: list[Tokens] = None

    # ')' token which ends an argument list.  Used by Prosser's algorithm in
    # computing hide sets.
    rparen: ClassVar[PpTok] = None

    nametok: PpTok                  # The macro name in the invocation.
    error: str = ''                 # Message if there was an error.

    # The expansion manager that created this MacroCall.
    exp: MacroExp

    # Enclosing MacroCall, if any.
    call_from: MacroCall = None

    # True if the macro will be expanded.  False for a function macro without
    # (), or if already expanding the macro, or any error.
    expanding: ClassVar[bool] = True

    # Hidden names to be applied to all tokens in resulting expansion.
    hide: Hide = None                    

    def __init__(self, m: macro, nametok: PpTok, intoks: TokIter,
                 hide: Hide, exp: MacroExp, call_from: MacroCall):
        self.m = m
        self.intoks = intoks
        self.nametok = nametok
        if hide: self.hide = hide
        self.exp = exp
        if call_from: self.call_from = call_from

    @TokIter.from_generator
    def getargs(self) -> Iterator[PpTok]:
        """
        Parse argument list from input intoks.  Generate any passthru intoks.
        Does nothing if an object macro.
        """
        try:
            m: Macro = self.m
            hide = self.hide
            if not m.is_func:
                return                  # An object macro
            # A function macro.  Go collect the arguments.

            args: list[Tokens] = []
            nparams = len(m.arglist) - m.variadic
            varnum = m.variadic and nparams - 1

            def argtoken(tok: PpTok, arg: Tokens) -> None:
                """
                Handle token that is part of an arg.  Spacing and indent may
                be modified.
                """
                if tok.indent:
                    # Remove indent and add spacing.
                    tok = tok.with_sep(TokenSep.create(
                        spacing=tok.colno > 1))
                if not arg:
                    # First token.  Remove spacing
                    tok = tok.without_spacing()
                arg.append(tok)

            tok: PpTok

            # Looking for argument list.
            # Directives are handled specially, but differently after the
            # opening '('.
            with self.nametok.source.inmacro(self):

                for tok in self.gen_to_normal():
                    if not tok.type.norm:
                        yield tok
                        continue

                    # tok is next normal token, if any.
                    if tok.value != '(':
                        # Func macro without arg list doesn't expand.
                        self.intoks.putback(tok)
                        self.expanding = False
                        return
                    # Got opening '('.
                    break
                else:
                    # No normal tokens
                    self.expanding = False
                    return

                # Now within the arg list, so do special lexing
                # differently.
                self.args = args = []
                nargs = 0

                # Loop for each argument.
                while not self.rparen and not self.error:
                    arg = Tokens()
                    args.append(arg)
                    nargs += 1
                    for tok in self.gen_arg():
                        if not tok.type.norm:
                            yield tok
                            continue
                        argtoken(tok, arg)

                # Close of argument list.

                # Special adjustment for hide set in function macro.
                if self.rparen:
                    hide = hide and hide & self.rparen.hide
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
                                f"{nparams - 1} argument(s) "
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

        finally:
            if hide:
                self.hide = hide.add(m.name)
                if m.name in hide:
                    self.expanding = False
            else:
                self.hide = Hide(m.prep, m.name)

    def gen_arglist(self) -> Iterator[PpTok]:
        """
        Generate all tokens after opening '(' up to and including matching
        ')'.  self.nesting tracks '(' not yet matched.
        """
        self.nesting = 0
        for intok in self.intoks:
            yield intok
            if intok.value == ')':
                if not self.nesting: return
                self.nesting -= 1
            elif intok.value == '(':
                self.nesting += 1

    def gen_arg(self) -> Iterator[PpTok]:
        """
        Generates intoks for next argument.  Consumes, without generating, the
        following comma or matching rparen.  Includes all balanced parens.  Set
        self.rparen if ending with a matching rparen.  Set self.error if ran
        out of intoks.
        """
        # Parse as far as matching ')', maybe stop at comma.
        self.nesting = 0

        for tok in self.gen_arglist():
            if not self.nesting:
                # Looking for separating ',' at top nesting level.
                if tok.value == ',':
                    # Comma is not part of the arg if it is within the
                    # __VA_ARGS__.
                    if not (self.m.variadic
                            and len(self.args) == len(self.m.arglist) - 1
                            ):
                        # The comma is not part of the arg
                        return
                elif tok.value == ')':
                    self.rparen = tok
                    # The rparen is not part of the arg
                    return
            # Part of the arg.
            yield tok
        # End of tokens before end of the arg list
        self.set_error(
            f"Macro {self.m.name!r} missing ')' in argument list.")

    def gen_to_normal(self) -> Iterator[PpTok]:
        """
        Generates input intoks up to the first normal token (if any).
        """
        for tok in self.intoks:
            yield tok
            typ = tok.type
            if not typ.norm:
                continue
            break

    @property
    def end_tok(self) -> PpTok:
        """ Last token in the macro call. """
        return self.rparen or self.nametok

    def orig_loc(self) -> TokLoc:
        """
        Where this call ultimately came from.  If directly in a source data, it's just
        the location in the source.  If it is a macro expansion, it's the
        location of the enclosing macro call.
        """
        c: MacroCall = self.nametok.call_from
        if c:
            return c.orig_loc()
        else:
            return self.nametok.loc

    def orig_call(self) -> MacroCall:
        """
        Where this call ultimately came from.  If directly in a source data, it's just
        this call.  If it is a macro expansion, it's the
        call of the enclosing macro call.
        """
        c: MacroCall = self.nametok.call_from
        if c:
            return c.orig_call()
        else:
            return self

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
        arguments.  Use the macro in self.call, unless a __VA_OPT__(...) macro
        is provided instead.  All result tokens are new copies.
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
                for param, arg in zip(self.m.arglist, self.args):
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
        return rep


class TokSubst(abc.ABC):
    """ Performs substitution for a single token, or sequence of intoks,
    in the replacement list of a Macro.  When called with the details of a
    macro invocation, it iterates over the intoks resulting.  These are newly
    created intoks.

    This base class is used for a single token, including a paste operator,
    and generates just that token.

    """
    # A token in the macro's replacement list, or the first of a sequence.
    tok: PpToken

    is_paste: ClassVar[bool] = False    # True for PasteTokSubst.
    is_param: ClassVar[bool] = False    # True for ParamTokSubst.

    def __init__(self, tok: PpTok):
        self.tok = tok

    @abc.abstractmethod
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        ...

    def __repr__(self) -> str:
        return repr(self.tok)


class PlainTokSubst(TokSubst):
    """ An ordinary token.  Generated token will set its call_from. """

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token(s) for this subst. """
        yield self.tok.copy(call_from=mgr.call)



class ParamTokSubst(TokSubst):
    """ Substitutes for a parameter name token.
    Generates the expansion of the corresponding argument, or possibly just
    the argument unexpanded.  If there are no intoks, generate a φ token,
    which is significant if later being pasted or stringized.  With preceding
    whitespace from either the expansion or the name token.
    """
    is_param: ClassVar[bool] = True     # True for ParamTokSubst.
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

    def __init__(self, tok: PpToken, argnum: int, after_paste: bool = True):
        super().__init__(tok)
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
        if self.expand:
            res = mgr.expansion(
                argnum, arg, with_spacing=self.after_paste)
        else:
            # Preceded and/or followed by π.  May supply a placemarker.
            if not arg:
                arg = Tokens.join(self.tok.make_null())
            res = TokIter(arg)

        # The first token's spacing may need adjustment.  This is handled by
        # the mgr.
        #  
        # If after_paste: Get spacing from the parameter value, but none if
        # the arg num is the first nonempty argument.  
        # 
        # Otherwise: no spacing.

        for tok in res:
            # First token.
            if not self.after_paste:
                tok = tok.with_spacing(self.tok.spacing)
            yield tok
            break
        # Everything else verbatim.
        yield from res


class StringTokSubst(TokSubst):
    """
    Substitutes for a '#' token plus a parameter name or __VA_OPT__ (...)
    expression.
    """
    arg: ParamTokSubst | VaOptTokSubst
    argnum: int | None              # None is for a VA OPT.

    def __init__(self, tok: PpToken, arg: TokSubst, argnum: int = None):
        super().__init__(tok)
        self.arg = arg
        self.argnum = argnum

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token(s) for this subst. """
        m = mgr.call.m
        argnum = self.argnum
        toks = self.arg.toks(mgr)
        value = mgr.string(toks, argnum)
        prep = mgr.prep

        string = self.tok.copy(value = value, type = prep.TokType.CPP_STRING)
        name = m.arglist[argnum or -1]
        prep.log.stringize(name, string, self.tok, nest=1)
        yield string


class PasteTokSubst(TokSubst):
    """ Substitutes for one or more π intoks plus the intoks on either side
    and in between.  Whitespace next to each π is ignored.
    """
    is_paste: ClassVar[bool] = True     # True for PasteTokSubst.
    # All the π and other substs (called S), in the order found in the macro
    # replacement list.  Each S expands to 1 or more normal intoks, or a φ for
    # an empty parameter.  Consecutive π's are possible, but consecutive S's
    # are not.  First and last elements are always S's.

    items: Sequence[TokSubst]           # Two or more elements being pasted.
    # One or more '##' intoks, between the items.
    ops: Sequence[PpTok]

    def __init__(self, tok: PpToken):
        super().__init__(tok)

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token(s) for this subst.
        Pastes two or more S substitutions together.  These each represent a
        token sequence or φ, and these intoks are used.  Last token of one S
        is pasted to the first token of the next S.  Either or both intoks can
        be φ placemarkers.  If the rhs has only one token, then the paste
        result becomes the lhs token to paste to the next item.  If a middle S
        has only one token, then multiple intoks are pasted.  Special cases if
        any item is empty.
        """

        yield self.tok.copy(call_from=mgr.call)
        return

        ## State of processing:  o* t? I0 I*
        ## o* = operands found so far to be pasted together.
        ## t? = last token, if any, iterated from I0.
        ## I0 = current item, maybe partially iterated.
        ## I* = any subsequent items, not iterated.

        ##items: list[TokSubst] = self.items[:]
        #items: list[TokIter] = [item.toks(mgr)
        #                        for item in self.items]

        #opnds: list[PpTok] = []         # o*
        #t: PpTok                        # current token
        #i: int                          # current item index
        #i = 0
        #item: TokSubst                  # I0
        #item = None
        #n = len(items)
        ## The '##' intoks between the items.
        #ops: list[PpTok] = [*self.ops]
        ## The '##' token before current item, if any.
        #op: PpTok = None
        #lex: PpLex = self.tok.lexer
        #first_result: bool
        #first_result = True

        #def next_tok() -> PpTok | None:
        #    """ Move to next non-empty item and return its first token.
        #    Set item = the item, with first token iterated.
        #    Return the token, or None if all items are empty.
        #    """
        #    nonlocal item, i
        #    while items:
        #        item = items[0]
        #        t = next(item, None)
        #        if t: return t
        #        # Item is empty.
        #        del items[0]
        #        i += 1
        #    # Every item is empty.
        #    return None

        #def paste() -> Iterator[PpTok]:
        #    """ Paste together the operands, then clear the operand list.
        #    Paste first two, then paste the result to the next, and so on.
        #    """
        #    if not opnds:
        #        return
        #    lhs: PpTok = opnds.pop(0)
        #    rhs: PpTok
        #    while opnds:
        #        rhs = opnds.pop(0)
        #        both = lhs, rhs
        #        pasted = ''.join(map(attrgetter('value'), both))
        #        # Prosser's algorithm says to take the intersection of the
        #        #   hide sets of the pasted intoks.
        #        hide = reduce(and_, map(attrgetter('hide'), both))
        #        t: PpTok | None = lhs.copy(
        #            value=pasted, type=None, hide=hide,
        #            )
        #        t = lex.fix_paste(t)
        #        prep.log.concatenate(t, *both, nest=1)
        #        if t:
        #            # The paste succeeded.
        #            lhs = t
        #        else:
        #            # Paste failed.  Generate the lhs operand and keep the rhs.
        #            # Also may need to generate a space.
        #            yield lhs
        #            if 0: pass
        #            #elif rhs.type in lhs.type.sep_from_paste:
        #            #    rhs = rhs.with_spacing()
        #            # G++ drops the space between quoted string and ID.
        #            #   But not GCC.
        #            elif prep.gplus_mode and lhs.type.quoted and rhs.type.id:
        #                rhs = rhs.without_spacing()
        #            #elif prep.clang:
        #            #    rhs = rhs.without_spacing()
        #            lhs = rhs

        #    # Only one item remaining

        #    yield lhs
        #    return
        #    self                # Make visible for breakpoint

        ## o* is empty.  State = I*
        ## Find first non-empty item (if any) and its first token
        #t = next_tok()
        #if not t: return
        #ws = self.tok.spacing
        ##ws = self.items[0].tok.spacing
        #if ws: t = t.with_spacing()

        ## Loop for each paste operation performed.
        #while True:
        #    # No operands yet, at least one token in current item.
        #    # Produce intoks up to last token in current item,
        #    # then paste last token with some intoks from remaining items.
        #    # (1) State = t I0 I*
        #    # Treat I0 as t*, state as t* o I*.  Emit the t*.
        #    for t2 in item:
        #        # State = t t2 I0 I*.  Emit t.
        #        yield t
        #        t = t2
        #    # State = t I*.

        #    opnds.append(t)

        #    # At least one operand, possibly some non-empty items.
        #    # State = o+ I*
        #    while True:
        #        # At least one operand plus some items.
        #        # Get next non-empty item and its first token.
        #        # (2) State = o+ I*
        #        # Check if I* all empty
        #        t = next_tok()
        #        if not t:
        #            # I* all empty.  State = o*
        #            yield from paste()
        #            return

        #        # State = o+ t I0 I*.  Move t to o+.
        #        opnds.append(t)

        #        # State = o+ I0 I*.
        #        t = next(item, None)
        #        if not t:
        #            # State = o+ I*
        #            continue

        #        # State = o+ t I0 I*
        #        break
        #    yield from paste()
        #    # State = t I0 I*.  Go to state (1).
        #    pass


class VaOptTokSubst(TokSubst):
    """
    Substitutes for entire "__VA_OPT__ ( repl list )".  Whitespace around the
    repl list is ignored.  Expands the replacement list from __VA_OPT__, if
    __VA_ARGS__ is non-empty, otherwise a single φ token.
    """
    repl: FuncMacro                 # Macro created from the repl list.                

    def __init__(self, tok: PpToken, repl: FuncMacro):
        super().__init__(tok)
        self.repl = repl

    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token(s) for this subst. """
        return mgr.expand_va_opt(self.repl)

# The following classes are for dynamic macros, like __LINE__, 
#   whose replacement varies with where and when they are called ...

class CounterTokSubst(TokSubst):
    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """
        Generates new token(s) for __COUNTER__ macro.  Increments each time it
        is expanded.
        """
        prep = mgr.prep
        t = mgr.call.nametok.copy(
            type = prep.t_INTEGER,
            value = prep.t_INTEGER_TYPE(prep.macros.countermacro),
            patt = re.compile(prep.lexer.REs.int),
            )
        prep.macros.countermacro += 1
        yield t


class FileTokSubst(TokSubst):
    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """
        Generates new token(s) for __FILE__ macro.  This changes with the
        current source file or a #line directive.
        """
        t = mgr.call.nametok
        prep = mgr.prep 
        filename = mgr.call.orig_call().nametok.loc.output_filename
        t = t.copy(
            type=mgr.prep.t_STRING,
            value=f'"{prep.fix_path_sep(filename)}"',
            )
        yield t


class LineTokSubst(TokSubst):
    @TokIter.from_generator
    def toks(self, mgr: TokSubstMgr) -> Iterator[PpTok]:
        """ Generates new token(s) for __LINE__ macro. """
        t = mgr.call.nametok
        prep = mgr.prep
        #line = t.linemacro()
        line = mgr.call.orig_call().end_tok.loc.output_lineno
        t = t.copy(
            type=prep.t_INTEGER,
            value=prep.t_INTEGER_TYPE(line),
            patt=re.compile(prep.lexer.REs.int),
            )
        yield t

# Maps the class of the token substitutor for each dynamic macro name.
dynamic_substs_tab: Mapping[str, TokSubst] = dict(
    __COUNTER__=CounterTokSubst,
    __FILE__=FileTokSubst,
    __LINE__=LineTokSubst,
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

    # Set during call to self()...
    expanded: Mapping[int, Tokens]      # The expansion for a parameter
                                        #   number, if needed.
    strings: Mapping[int, PpTok]        # The stringization for a parameter
                                        #   number, if needed.

    def __init__(self, call: MacroCall):

        self.m = m = call.m
        self.prep = m.prep
        self.call = call

    @TokIter.from_generator
    def __call__(self, m: Macro) -> Iterator[PpTok]:
        """ Equivalent of subst() in Prosser's algorithm.
        Generates the replacement intoks for the macro call.
        If substituting the __VA_OPT__(replacement list) for this macro,
        then the `m` argument is made from that replacement list.
        """
        """
        Two-pass substitution.

        First pass generates tokens for all the m.substs objects.  However, a
        paste in the original replacement list becomes simply a π token.  An
        object macro has no subst objects, and the original replacement tokens
        are used.  Parameter names are expanded or stringized, when required.

        This pass varies with the type of macro being called.  The result is a
        sequence of tokens with some paste operators.

        Second pass performs the pasting by finding π tokens and replacing
        them and their adjacent tokens with the concatenation tokens.  Also
        adds to the hide sets of all generated tokens.
        """

        # First pass.
        intoks = TokIter(m.subst_tokens(self))

        hide = self.call.hide
        # Second pass

        prev: PpTok

        toks: TokIter
        if not self.m.has_paste:
            toks = intoks
        else:
            if self.prep.clang:
                toks = self.filt_pastes_clang(intoks)
            if self.call.nametok.brk():
                print(*toks.copy_tokens())
            toks = self.do_pastes(toks)
        for tok in toks:
            # Filter out φ token and any following π token.
            #try:
            #    while tok.type.null or tok.value == 'φ':
            #        tok = next(toks)
            #        #if tok.type.paste:
            #        #    tok = next(toks)
            #except StopIteration:
            #    break
            tok = tok.copy()
            tok.add_hide(hide)
            if self.call.nametok.brk():
                print(f"-> {str(tok)!r}")
            yield tok

    @TokIter.from_generator
    def filt_pastes_clang(self, toks: TokIter) -> Iterator[PpTok]:
        """
        Changes the token stream as clang does before doing the actual pastes.
        Removes any π φ or φ π, then changes (π π)+ π? to (π ##)+ π?.
        """
        lhs: PpTok
        rhs: PpTok
        prep: PPreprocessor = self.m.prep
        npastes: int = 0        # How many π just generated.
        for lhs in toks:
            # - π 
            # - φ
            # - lhs -> lhs.  npastes = 0.
            #  
            if lhs.type.paste:
                for rhs in toks:
                    # π rhs
                    if rhs.type.null:
                        # π φ.
                        # Remove π φ.
                        lhs = None
                    else:
                        toks.putback(rhs)
                    break
            elif lhs.type.null:
                for rhs in toks:
                    # φ rhs
                    if rhs.type.paste:
                        # φ π.
                        # Remove φ π.
                        lhs = None
                    else:
                        toks.putback(rhs)
                    break
            if lhs:
                if lhs.type.paste:
                    if npastes & 1:
                        lhs = lhs.copy(type=prep.TokType.CPP_DPOUND)
                    npastes += 1
                else:
                    npastes = 0
                yield lhs

    @TokIter.from_generator
    def do_pastes(self, toks: TokIter) -> Iterator[PpTok]:
        """
        Perform any π pastes contained within the tokens and generate new
        tokens, otherwise pass on the tokens unchanged.  Any sequence of (lhs,
        π, rhs) pastes lhs with rhs (neither of which is π) and the result
        will be next lhs.  If the paste fails, generate lhs and use rhs as the
        next lhs.

        Each token is either an ordinary token, π, or φ.  π will not be either
        the first or the last token.  φ will be preceded and/or followed by π.
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

        L and R can be φ.  Pasting equivalent to ordinary token with '' value.
        Not generated.

        Empty.  {} • • {} -> ends the function with no tokens generated.

        Initial.
          - {} • • L I -> {} • L • I.  {} • • φ π I -> {} • • I.

        Middle.
          - O • L • R   I -> O L • R • I.
          - O • L • π R I -> O • LR • I if paste LR is valid.
          - O • L • π R I -> O L • R • I if paste LR is not valid.

        Final.
          - O • L • {} -> O L • • {}, ends the function.

        '''
        # Initial state, {} • • O.

        # Get first token, if any.
        lhs = next(toks, None)
        if lhs is None:
            # {} • • {}.  No tokens at all.
            return

        # Handle remaining tokens one at a time.  lhs gets updated sometimes.

        # O • lhs • I

        for rhs in toks:
            # O • lhs rhs • I

            if rhs.type.paste:
                # O • lhs π • rhs I
                op = rhs
                rhs = next(toks)
                # O • lhs π rhs • I

                # Try pasting lhs and rhs.  If OK, update lhs and generate
                # nothing.
                if lhs.type.id and rhs.type.id:
                    # Fast track for a common use case.
                    #lhs = lhs.copy(value=lhs.value + rhs.value,
                    #               hide=lhs.hide and rhs.hide
                    #               and lhs.hide & rhs.hide)
                    lhs = op.copy(value=lhs.value + rhs.value,
                                  type=lhs.type,
                                  hide=lhs.hide and rhs.hide
                                  and lhs.hide & rhs.hide)
                    continue

                t: PpTok | None = lhs.lexer.try_paste(lhs, rhs)
                prep.log.concatenate(t, lhs, rhs, nest=1)
                if t:
                    # The paste succeeded.
                    lhs = t
                    continue
                # Paste failed.
            # Either lhs rhs or failed lhs π rhs.
            if not lhs.type.null:
                yield lhs
            lhs = rhs

        # End of rhs in toks loop.

        # O • lhs • {}

        if not lhs.type.null:
            # O lhs • • {}
            yield lhs

        return

        def getrhs() -> PpTok | None:
            """
            Find the rhs token, which is usually the next one.  Current state
            is O • lhs • I.
            """

            intoks = Tokens([lhs])

            rhs = next(toks, None)
            if rhs is None:
                # O • lhs • {}
                return None

            intoks.append(rhs)
            # O • lhs rhs • I
            if rhs.type.null:
                # O • lhs φ • π I
                # φ is not preceded by π, so it must be
                # followed by π.  We skip the φ and move to the following
                # π.
                #
                # clang skips the π as well and moves to the next token
                # (if any) after that, and then processes (lhs, rhs).  
                # 
                # Get the following π.
                rhs = next(toks)
                intoks.append(rhs)
                # O • lhs φ π • I
                if prep.clang:
                    rhs = next(toks, None)
                    if rhs is None:
                        # O • lhs φ π • {}
                        # Ran out of tokens, so end the loop.  We still
                        # have lhs.
                        return None
                    intoks.append(rhs)
                    # O • lhs φ π rhs • I
            if self.call.nametok.brk():
                print(*intoks)
            return rhs
        # End of getrhs().

        while True:
            # lhs can be φ, which is followed, but not preceded, by π.  lhs is
            # never a π; it could be a π replaced with '##', an ordinary
            # token, however.
            rhs = getrhs()
            if rhs is None:
                yield lhs
                return

            if lhs.type.null:
                # O • φ • π I.
                next(toks)
            if not rhs.type.paste:
                # O • lhs rhs • I.  rhs is not .
                yield lhs
                lhs = rhs
                continue
            # O • lhs π • rhs I.  π is always followed by π.
            rhs = next(toks)
            # O • lhs π rhs • I.
            if rhs.type.paste:
                if prep.clang:
                    # clang treats π as ordinary token.  Change to '##'.
                    rhs = rhs.copy(type=prep.TokType.CPP_DPOUND)

            # Try pasting lhs and rhs.  If OK, update lhs and generate
            # nothing.
            if lhs.type.id and rhs.type.id:
                # Fast track for a common use case.
                lhs = lhs.copy(value=lhs.value + rhs.value)
                continue

            assert not lhs.type.null
            if rhs.type.null:
                # O • lhs π φ • I.
                continue
            t: PpTok | None = lhs.lexer.try_paste(lhs, rhs)
            prep.log.concatenate(t, lhs, rhs, nest=1)
            if t:
                # The paste succeeded.
                lhs = t
            else:
                # Paste failed.
                yield lhs
                rhs = lhs

        # O • lhs • {}
        yield lhs
        return


    # Helper methods used by the TokSubst's ...

    def expansion(
            self, argnum: int, arg: list[PpTok],
            with_spacing: bool = True
            ) -> Tokens:
        """ The expansion (calculated only once) for the given argument #.
        Caller must not modify the return result, but can make a copy.
        If not `with_spacing`, OR argnum is the first nonempty arg,
            then spacing is removed from the first token of the return value,
            although the stored arg value still has it.
        """
        # Resulting expansion.  May have initial whitespace removed.
        exp: Tokens
        prep = self.prep
        try:
            try:
                exp = self.expanded[argnum]
                # self.expanded exists, and [argnum] does also.
                return
            except AttributeError:
                # self.expanded does not exist.  Create one now.
                self.expanded = dict()
            except KeyError:
                # self.expanded exists, but [argnum] does not.
                pass
            # First time for this argnum.
            # Create the expansion and add it to self.expanded.
            toks = TokIter(arg)
            toks = self.call.m.macros.expand(toks,
                                             origin=self.call.exp.origin)
            with prep.nest():
                prep.log.write(f"Expanding arg {self.call.m.arglist[argnum]}",
                               token=self.call.nametok)
                exp = self.expanded[argnum] = toks.get_tokens()
        finally:
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

    def string(self, arg: TokIter, argnum: int) -> PpTok:
        """
        The stringization (calculated only once if argnum is provided) for the
        given argument tokens.  Caller must not modify the return result, but
        can make a copy.
        """
        try:
            return self.strings[argnum]
        except AttributeError:
            # self.strings does not exist.  Create it now.
            self.strings = dict()
        except KeyError:
            # self.strings exists, but [argnum] does not.
            pass
        # First time for this argnum.
        # Create the string token and maybe add it to self.strings.
        string: PpTok = FuncMacro.stringize(arg)
        if argnum is not None:
            self.strings[argnum] = string
        return string

    def expand_va_opt(self, repl: FuncMacro
                      ) -> Iterator[PpTok]:
        """ Generates intoks from a __VA_OPT__ (toks) expression,
        Expands the toks as if it were the replacement for calling the same
        macro, except if the __VA_ARGS__ parameter is empty, a single φ token
        is generated.
        """
        varargs = self.call.args[-1]
        # Check for a placemarker
        if varargs:
            yield from self.call.subst(repl)
        else:
            yield repl.nametok.make_null()

    if __debug__:
        def brk(self) -> bool: return self.call.nametok.brk()

def Prosser(macros: Macros, TS: Tokens, debug: bool = False) -> Tokens:
    """
    Python versions of Prosser's algorithm.  Implements the outermost
    expand(TS) call and returns the resulting OS list.  Returned tokens are
    copies of tokens taken from either the input TS or the replacement list of
    some macro.

    This is here just for demonstration and debugging purposes.  You can call
    this function with a Tokens to expand, to see if it is the same result as
    macros.expand() produces.  debug=True will print some information.

    Comments with '##' are taken verbatim from the Prosser document.

    Added code to handle __VA_OPT__, which did not exist at the time of the
    document.  The C++23 Standard says to take the body of the (...) following
    __VA_OPT__ as an alternate replacement list for the same macro called with
    the same arguments, or a placemarker if __VA_ARGS__ has no tokens.
    """

    # Here are various functions called during the body of Prosser()...

    ## expand(TS) /* recur, substitute, pushback, rescan */
    ## {
    def expand(TS: Tokens, top: bool = False) -> Tokens:
        TS_: Tokens
        TS__: Tokens
        T: PpTok
        HS: Hide
        HS_: Hide
        OS: Tokens          # From call to subst()
        m: Macro = None

        ## if TS is {} then
        if not TS:
        ## return {};
            return Tokens()
        try:
            ## else if TS is T ↑ HS • TS’ and T is in HS then
            T = TS[0]
            HS = T.HS
            TS1 = TS[1:]
            if TS1:
                T1 = TS[1]
                TS2 = TS[2:]
            msg(f"expand TS = {T.value!r} {HS} • {str(TS1)!r}")
            if T.value in HS:
                ## return T ↑ HS • expand(TS’);
                with indent(): print(f"not expanded ")
                exp = Tokens.join(T, *expand(TS1))
                return exp

            ## else if TS is T ↑ HS • TS’ and T is a "()-less macro" then
            if T.value in macros:
                m = macros.get(T.value)
                repl: Tokens = Tokens(tok.copy(HS=Hide()) for tok in m.value)
                with indent(): msg(f"replacement = {str(repl)!r}")
            if m and not m.is_func:
                # TS is macro • TS1
                ## return expand(subst(ts(T),{},{},HS∪{T},{}) • TS’);
                with indent():
                    OS = subst(repl, [], [], HS.add(T.value), Tokens(),
                               top=True)
                    exp = Tokens.join(*expand(OS), *TS1)
                    with indent(): msg(f"expansion = {exp}")
                return exp

            ## else if TS is T ↑ HS • ( • TS’ and T is a "()’d macro" then
            ## check TS’ is actuals • ) ↑ HS’ • TS’’ and actuals are "correct for T"
            if m and m.is_func and TS1 and T1.value == '(':
                # TS is macro • ( • actuals • ) ↑ HS2 • TS2
                ## return expand(subst(ts(T),fp(T),actuals,(HS∩HS’)∪{T},{}) • TS’’);
                TI = TokIter(TS1)
                call = MacroCall(m, T, TI, Hide(), None)
                list(call.getargs())
                TS2 = Tokens(TI)
                HS2 = call.rparen.HS
                with indent(): OS = subst(
                    repl, m.arglist, call.args,
                    (HS & HS2).add(T.value), Tokens(), top=True
                    )
                exp = expand(Tokens.join(*OS, *TS2))
                with indent(): msg(f"expansion = {exp}")
                return exp
        
            ## note TS must be T ↑ HS • TS’
            ## return T ↑ HS • expand(TS’);
            exp = Tokens.join(T, *expand(TS1))
            return exp
            ## }
        finally:
            if top:
                msg(f"expansion = {str(exp)!r}")

    ## subst(IS,FP,AP,HS,OS) /* substitute args, handle stringize and paste */
    ## {
    def subst(
        IS: Tokens,
        FP: list[str],
        AP: list[Tokens],
        HS: Hide,
        OS: Tokens,
        top: bool = False,
        ) -> Tokens:

        T: PpTok
        IS1: Tokens
        IS2: Tokens

        msg(f"subst IS = {str(IS)!r}")
        if top:
            with indent():
                msg(f"FP = {FP}")
                msg(f"AP = {AP}")
                msg(f"HS = {HS}")

        ## if IS is {} then
        if not IS:
            ## return hsadd(HS,OS);
            result = hsadd(HS, OS)
            msg(f"result = {str(result)!r}")
            return result

        T = IS[0]
        IS1 = IS[1:]
        if IS1:
            T1 = IS[1]
            IS2 = IS[2:]
            if IS2:
                T2 = IS[2]
                IS3 = IS[3:]

        ## else if IS is # • T • IS’ and T is FP[i] then
        if T.value == '#' and IS2 and T1.value in FP:
            ## return subst(IS’,FP,AP,HS,OS • stringize(select(i,AP)));
            # IS = # • param1 • IS2
            actual: Tokens = getactual(T1)
            return subst(IS2, FP, AP, HS,
                         Tokens.join(*OS, stringize(actual)))

        ## else if IS is ## • T • IS’ and T is FP[i] then
        if T.value == '##' and IS1 and T1.value in FP:
        ## {
            actual: Tokens = getactual(T1)
            ## if select(i,AP) is {} then /* only if actuals can be empty */
            # IS = ## • param1 • IS2
            if not actual:
                ## return subst(IS’,FP,AP,HS,OS);
                return subst(IS2, FP, AP, HS, OS)

            ## else
            else:
                ## return subst(IS’,FP,AP,HS,glue(OS,select(i,AP)));
                return subst(IS2, FP, AP, HS, glue(OS, actual))
        ## }

        ## else if IS is ## • T ↑ HS’ • IS’ then
        if T.value == '##' and IS1:
            # IS = ## • T1 • IS2
            ## return subst(IS’,FP,AP,HS,glue(OS,T ↑ HS’ ));
            return subst(IS2, FP, AP, HS, glue(OS, [T1]))

        ## else if IS is T • ## ↑ HS’ • IS’ and T is FP[i] then
        ## {
        if IS1 and T1.value == '##' and T.value in FP:
            # IS = param • ## • IS2
            actual: Tokens = getactual(T)
            ## if select(i,AP) is {} then /* only if actuals can be empty */
            if not actual:
            ## {
                if IS2 and T2 in FP:
                ## if IS’ is T’ • IS’’ and T’ is FP[j] then
                    # IS = empty param • ## • param2 • IS3
                    ## return subst(IS’’,FP,AP,HS,OS • select(j,AP));
                    return subst(IS3, FP, AP, HS,
                                 Tokens.join(*OS, getactual(T2))
                                 )
                ## else
                else:
                    ## return subst(IS’,FP,AP,HS,OS);
                    # IS = empty param • ## • IS2
                    return subst(IS2, FP, AP, HS, OS)
            ## }
            ## else
            else:
                # IS = nonempty param • ## • IS2
                ## return subst(## ↑ HS’ • IS’,FP,AP,HS,OS • select(i,AP));
                return subst(Tokens.join(T1, *IS2), FP, AP, HS,
                             Tokens.join(*OS, *actual)
                             )
        ## }

        ## else if IS is T • IS’ and T is FP[i] then
        if T.value in FP:
            # IS = param • IS1
            ## return subst(IS’,FP,AP,HS,OS • expand(select(i,AP)));
            actual: Tokens = getactual(T)
            with indent(): exp = expand(actual, top=True)
            return subst(IS1, FP, AP, HS, Tokens.join(*OS, *exp)
                         )
        ## note IS must be T ↑ HS’ • IS’
        #IS = T • IS1
        ## return subst(IS’,FP,AP,HS,OS • T ↑ HS’ )
        return subst(IS1, FP, AP, HS, Tokens.join(*OS, T))
        ## }

    # paste last of left side with first of right side
    def glue(LS: Tokens, RS: Tokens) -> Tokens:
        ## if LS is L ↑ HS and RS is ↑ R HS’ • RS’ then
            ## return L&R ↑ HS∩HS’ • RS’; /* undefined if L&R is invalid */
        if not LS1 and RS:
            RS1: Tokens = RS[1:]            # RS’ in document
            L: PpTok = LS[0]
            R: PpTok = RS[0]
            # Make a PpTok from concatenated value.
            cat: str = L.value + R.value
            tok: PpTok | None = L.copy(
                value=cat, type=None,
                )
            tok = L.lexer.fix_paste(tok)
            if tok:
                tok.HS = L.HS & R.HS
                return Tokens.join(tok, *RS1)
            else:
                # Paste failed.  We'll just leave L and R as they are.
                return Tokens.join(L, *RS)
        else:
            LS1: Tokens = LS[1:]                # LS’ in document
            ## note LS must be L ↑ HS • LS’
            ## return L ↑ HS • glue(LS’,RS);
            return Tokens.join(L, self.glue(LS1, RS))

    # add to token sequence’s hide sets
    def hsadd(HS: Hide, TS: Tokens) -> Tokens:
        if not TS:
            return Tokens()
        T: PpTok = TS[0]
        TS1: Tokens = TS[1:]            # TS' in document
        return Tokens.join(
            T.copy(HS=HS | T.HS), *hsadd(HS, TS1))

    def getactual(T: PpTok, IS1: Tokens) -> Tokens | None:
        """
        The actual argument for the parameter name in given token.  Can be
        '__VA_OPT__', which will expand the (...) expression contained in
        IS1.  Returns None if it is not a parameter name.

        This is equivalent of select(i, TS) in the document.
        """
        try:
            return AP[FP.index(T.value)]
        except ValueError:
            # Not a parameter
            return None
        except IndexError:
            # Parameter is __VA_OPT__.  It is followed by ( replacement list )
            # for a function macro TODO: get and expand the opt expression.
            m: FuncMacro = T.exp_from
            toks = TokIter(IS1)
            mopt : FuncMacro = m.parse_va_opt(toks, m.nametok)
            IS1 = Tokens(toks)
            if not AP[-1]:
                # No __VA_OPT__ tokens results in placemarker.
                return Tokens()
            with indent():
                # Do same substitution with mopt as we have been doing
                # with m.
                OS = subst(mopt.value, FP, AP, HS, Tokens())

            return OS

    def stringize(TS: Tokens) -> PpTok:
        """ Make a single string token from values of given tokens. """
        s: PpTok = FuncMacro.stringize(TokIter(TS))
        return s

    @contextlib.contextmanager
    def indent() -> Iterator:
        nonlocal nesting
        nesting += 1
        try: yield
        finally: nesting -= 1

    def msg(text: str) -> None:
        if debug: print(f"{'  ' * nesting}{text}")

    # This is the code for Prosser().
    nesting: int
    nesting = 0

    TS_copy = Tokens(tok.copy(HS=Hide()) for tok in TS)
    return expand(TS_copy, top=True)

