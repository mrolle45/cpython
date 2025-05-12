""" prosser.py.
Implements macro expansion in the manner of D. M. Prosser's paper.
With added code to handle __VA_OPT__ expressions.
Makes use of the macros package.
"""

'''
David Prosser's algorithm can be found at
https://www.spinellis.gr/blog/20060626/cpp.algo.pdf
'''

from pcpp.macros import *

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

    m: Macro = None

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
        nonlocal m

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
                    repl, m.param_names, call.args,
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
                # Prosser doesn't cover this case.
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
        '__VA_OPT__', which will expand the (...) expression contained in IS1.
        Returns None if it is not a parameter name.

        This is equivalent of select(i, TS) in the document, with the addition
        of __VA_OPT__ which is not covered in the document.
        """
        try:
            return AP[FP.index(T.value)]
        except ValueError:
            # Not a parameter
            return None
        except IndexError:
            # Parameter is __VA_OPT__.  It is followed by ( replacement list )
            # for a function macro.
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
    def indent() -> ContextManager[None]:
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

