""" macros
Manages macro definitions and lookups for a C translation unit.
"""
from __future__ import annotations

import copy
from enum import Enum, auto
from dataclasses import dataclass, InitVar
from itertools import chain, zip_longest
from operator import attrgetter
import traceback

from ply.lex import LexToken
from pcpp.lexer import TokType, no_ws, merge_ws, Tokens, Hide


# ------------------------------------------------------------------
# Macro object
#
# This object holds information about preprocessor macros
#
#    .name      - Macro name (string)
#    .value     - Macro value (a list of tokens, from the #define)
#    .is_func   - A function macro
#    .source    - Source object containing the #define
#    .lineno    - Line number containing the #define
#
# For function macros only:
#    .arglist   - List of argument names, including .vararg (if present)
#    .variadic  - Boolean indicating whether or not variadic macro
#    .vararg    - Name of the variadic parameter
#
# When a macro is created, the macro replacement token sequence is
# pre-scanned and used to create patch lists that are later used
# during macro expansion
# ------------------------------------------------------------------

class Macro:
    subs: tuple[MacSubst,...]                   # The replacement list.

    def __init__(self, macros: Macros, nametok: PpTok, value: Tokens):
        self.nametok =  nametok
        self.value = value
        self.macros = macros
        self.prescan()
        self.makesubst()

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

    class TokInfo:
        m: Macro
        tok: LexToken
        pos: int                # Where it appears in the macro replacement list.
        arg: bool = False       # This is an argument name.  Function macro only.
        str_op: bool = False    # This is a '#' token.  Function macro only.
        cat_op: bool = False    # This is a '##' token.
        opt_op: bool = False    # This is a __VA_OPT__ identifier.
        other: bool = False     # Anything else.
        ws: bool = False        # Whitespace

        # for arg tokens...
        argnum: int = None      # Index of the argument in macro argument list.
        expand: bool = False    # Will be fully macro expanded.
        string: bool = False    # Will be stringized.
        concat: bool = False    # Will be concatenated with previous and/or next token.
        pending: bool = False   # Waiting to determine expand/string/concat.

        def __init__(self, m: Macro, tok: LexToken = None, pos: int = None):
            """ Set the flags appropriate to the token. """
            ...
            self.m = m
            self.tok = tok
            self.pos = pos
            if not tok: return
            is_func = m.is_func
            if is_func and tok.type is TokType.CPP_ID and tok.value in m.arglist:
                self.arg = True
                self.argnum = m.arglist.index(tok.value)
                self.pending = True
            elif is_func and tok.type is TokType.CPP_POUND:
                self.str_op = True
            elif tok.type is TokType.CPP_DPOUND:
                self.cat_op = True
                m.num_concat += 1
            elif is_func and tok.type is TokType.CPP_ID and tok.value == '__VA_OPT__':
                self.opt_op = True
            elif tok.type.ws:
                self.ws = True
            else:
                self.other = True

        def __bool__(self): return bool(self.tok)

        def error(self, msg: str) -> None:
            self.macro.prep.on_error(self.tok.value, self.tok.lineno, msg)

        def __repr__(self) -> str:
            return repr(self.tok)

    # ----------------------------------------------------------------------
    # prescan()
    #
    # Examine the macro value (token sequence) and identify patch points
    # This is used to speed up macro expansion later on---we'll know
    # right away where to apply patches to the value to form the expansion
    #
    # Also called for object macros, though only the ## operators are involved.
    # ----------------------------------------------------------------------
    
    def prescan(self):
        """Examine the macro value (token sequence) and identify patch points
        This is used to speed up macro expansion later on---we'll know
        right away where to apply patches to the value to form the expansion"""
        self.var_comma_patch = []       # Variadic macro comma patch
        prep = self.macros.prep
        self.num_concat = 0
        if self.is_func:
            self.expand_args = set()        # Arg numbers which get expanded
            self.string_args = set()        # Arg numbers which get stringized
            self.concat_args = set()        # Arg numbers which get concatenated
            self.expand_points = []         # Positions for args that are expanded
            self.concat_points = []         # Positions for args that are concatenated
            self.opt_points = []            # Positions for __VA_OPT__ expressions
            self.replace_points = []        # Either expand or concat or opt, in descending order.
            self.string_points: list[TokInfo] = []  # Infos for args that are stringized
            self.opts = {}                  # info for an opt arg at given position
            # Information about tokens in replacement list, indexed by their position.
            self.argnums: Mapping[int, int] = {}    # [position] -> arg number.

        def pass1() -> Iterable[TokInfo]:
            """ Make infos from all the tokens.
            Remove whitespace after # and around ##.
            Make single info for "__VA_OPT__ ( ... )".
            """
            hold_ws: list[TokInfo] = []            # Whitespace that might be generated.
            toks = enumerate(self.value)
            last_seen: TokInfo = self.TokInfo(self)    # Last non-ws token seen.
            for i, tok in toks:
                info = self.TokInfo(self, tok, i)
                if info.ws:
                    hold_ws.append(info)
                    continue
                if hold_ws:
                    if last_seen.cat_op or last_seen.str_op or info.cat_op:
                        pass
                    else:
                        yield from hold_ws
                    hold_ws = []
                if info.opt_op:
                    # Gobble up the following replacement list.
                    m : Macro = self.parse_opt(toks, info.tok)
                    if not m or self.name == '__VA_OPT__':
                        prep.on_error_token(tok, "Ill-formed __VA_OPT__ expression.")
                        continue
                    info.repl = m
                if info.str_op:
                    # Gobble up whitespace plus arg.
                    for i, tok in toks:
                        if tok.type.ws: continue
                        break
                        if tok.arg:
                            info.argnum = tok.argnum
                            break
                    else:
                        # No more tokens
                        prep.on_error_token(tok, "# operator must be followed by parameter name.")
                        continue
                    arg = self.TokInfo(self, tok, i)
                    if tok.value not in self.arglist:
                        prep.on_error_token(tok, "# operator must be followed by parameter name.")
                        continue
                    info.argnum = arg.argnum
                yield info
                last_seen = info

        infos: list[Info] = list(pass1())

        # Find all the argument name tokens, and classify them.
        tok: LexToken
        last_seen: TokInfo = self.TokInfo(self, None)    # Last token seen, not counting whitespace.

        # Enumerator for the infos, with None at the end.

        iterator = enumerate(infos + [self.TokInfo(self, None)])
        for i, info in iterator:
            # Tokens can be any of: whitespace, arg name, #, ##, opt, and other.
            # Arg name, #, and opt aren't part of an object macro.
            # Ignoring all whitespace.
            # '#' arg is legal.
            # anything '##' anything is legal, including a # arg or another ##.
            info.pos = i
            if info.ws: continue
            if info.arg:
                # An arg name.  Might follow # or ##.
                argnum = info.argnum
                self.argnums[i] = argnum
                # See what context we are in.
                if last_seen.cat_op:
                    # '##' + arg -> concatenate arg.
                    info.concat = True
                    info.pending = False
                    self.concat_args.add(argnum)
                    self.concat_points.append(i)
            elif info.str_op:
                info.tok.argnum = info.argnum
                info.string = True
                info.pending = False
                self.string_args.add(info.argnum)
                self.string_points.append(info)
            elif info.cat_op:
                if not last_seen:
                    # nothing + '##' is error.
                    tok.error("'##' operator cannot be at the start.")
                if last_seen.pending:
                    # not (# or ##) + arg + '##' -> concatenate arg.
                    last_seen.concat = True
                    last_seen.pending = False
                    self.concat_args.add(last_seen.argnum)
                    self.concat_points.append(last_seen.pos)
            elif last_seen.arg:
                if last_seen.pending:
                    # not (# or ##) + arg + other -> expand arg.
                    last_seen.expand = True
                    last_seen.pending = False
                    self.expand_args.add(last_seen.argnum)
                    self.expand_points.append(last_seen.pos)
            elif not info and last_seen.cat_op:
                tok.error("'##' operator cannot be at the end.")
            if info.opt_op:
                self.opt_points.append(i)
                self.opts[i] = info
            last_seen = info

        self.repl = Tokens(info.tok for info in infos)

        if self.is_func:
            self.replace_points = self.expand_points + self.concat_points + self.opt_points
            self.replace_points.sort(reverse=True)

    def makesubst(self) -> None:
        """ Build substitution items in self.subs. """
        tokiter: Iterator[tuple[int, PpTok]] = enumerate(self.value)

        self.subs = []
        is_func: bool = self.is_func
        param_names: list[str]
        if is_func:
            param_names = self .arglist

        hold_ws: list[PpTok] = []

        def skip_ws() -> PpTok | None:
            hold_ws.clear()
            for i, tok in tokiter:
                if tok.type.ws:
                    hold_ws.append(tok)
                else:
                    return tok
            return None

        def emit_ws() -> None:
            self.subs += map(TokSubst, hold_ws)

        def nextsub(tok: PpTok = None) -> MacSubst | None:
            """ A substitution object from the input stream, if any.
            Preceding whitespace is either added to the output or ignored.
            A '##' returns just a Paste token, without finding the rest of the expression.
            """
            if not tok:
                tok = skip_ws()
                if not tok:
                    return None
            val = tok.value
            if val == '##':
                hold_ws.clear()
                return PasteTokSubst(tok)
            # No other cases eat preceding whitespace.
            #emit_ws()
            # All other special tokens apply only to function macros.
            if is_func:
                if val == '#':
                    # Stringize.  Eat following whitespace and get following param name.
                    param = skip_ws()
                    hold_ws.clear()
                    argnum = param_names.index(param.value)
                    return StringTokSubst(tok, argnum)
                elif val == '__VA_OPT__':
                    m : FuncMacro = self.parse_opt(tokiter, tok)
                    return OptTokSubst(tok, m)
                elif val in param_names:
                    argnum = param_names.index(val)
                    return ParamTokSubst(tok, argnum)
            # Anything else.
            return TokSubst(tok)

        def do_paste(p: PasteTokSubst) -> TokSubst | None:
            """ Process entire paste expression.  Given sub is the first ##.
            First operand is at the end of self.subs, and is removed from there.
            Returns sub for first token (if any) after the paste expression.
            """
            items: list[TokSubst] = []
            ops: list[PpTok] = [p.tok]

            try: items.append(self.subs.pop())
            except IndexError:
                # '##' at the start is illegal.
                ...

            while True:
                item = nextsub()
                hold_ws.clear()
                if not item:
                    # '##' at the end is illegal.
                    ...
                if item.is_paste:
                    # Consecutive ##, make same as just one.
                    continue
                items.append(item)
                sub = nextsub()
                if not (sub and sub.is_paste):
                    break
                ops.append(sub.tok)
            p.items = items
            p.ops = ops
            self.subs.append(p)
            return sub
            
        tok = skip_ws()
        if not tok: return                  # Entire value is whitespace.  Empty result.


        sub: TokSubst = nextsub(tok)

        # Loop over tokens to turn into subs.
        while sub:
            if sub.is_paste:
                # Handle the entire paste expression and get following sub (if any).
                sub = do_paste(sub)
                continue

            emit_ws()
            self.subs.append(sub)
            sub = nextsub()
            ## Classify this token.
            #val = tok.value
            #if val == '##':
            #    # Move past entire paste expression and get what follows it.
            #    tok = do_paste()
            #    continue
            ## Whitespace is significant in all other cases.
            #emit_ws()

            #if is_func and val == '#':
            #    param = skip_ws()
            #    hold_ws.clear()
            #    self.subs.append(StringTokSubst(tok, param, self))
            #elif is_func and val == '__VA_OPT__':
            #    m : Macro = self.parse_opt(tokiter, tok)
            #    self.subs.append(OptTokSubst(tok, m))
            #elif is_func and val in self.arglist:
            #    self.subs.append(ParamTokSubst(tok, self))
            #else:
            #    self.subs.append(TokSubst(tok))
            #tok = skip_ws()

    def glue(self, op: PpTok, left: Tokens, right: Tokens) -> Tokens:
        """ Concatenate last of left with first of right (skipping whitespace) into a PASTED token.
        Return rest of left + pasted + rest of right.
        """
        i = len(left) - 1
        while i and left[i].type.ws:
            i -= 1
        if i:
            return Tokens([left[0]] + self.glue(op, left[1:], right))
        # left is now a single token plus any whitespace.
        # Skip right whitespace.
        j = 0
        while j < len(right) and right[j].type.ws:
            j += 1
        l = left[0]
        r = right[j]
        op.type = TokType.CPP_PASTED
        op.value = l.value + r.value
        op.hide = l.hide & r.hide
        return Tokens([op] + right[j + 1:])

        ...

    def concat(self, toks: Tokens) -> None:
        """ Do a token concatenation pass, stitching any tokens separated by ## into a single token
        Surrounding whitespace has already been removed.
        Tokens are altered in-place.
        """
        if self.num_concat:
            i = 1
            while i < len(toks) - 1:
                left, right = i - 1, i + 1
                if toks[i].type is TokType.CPP_DPOUND:
                    # If consecutive '##'s, insert a placemarker
                    if toks[right].type is TokType.CPP_DPOUND:
                        pm = copy.copy(toks[right])
                        pm.type = TokType.CPP_PLACEMARKER
                        pm.value = ''
                        toks.insert(right, pm)

                    with self.prep.nest():
                        self.log.concatenate(toks[left], toks[right])
                    if toks[left].type is TokType.CPP_PLACEMARKER:
                        del toks[left : right]
                    elif toks[right].type is TokType.CPP_PLACEMARKER:
                        del toks[i : right+1]
                    else:
                        toks[left] = copy.copy(toks[left])
                        toks[left].type = TokType.CPP_PASTED
                        toks[left].value += toks[right].value
                        del toks[i : right+1]
                    toks.changed = True
                else:
                    i += 1

            self.fix_pastes(toks)

    def fix_pastes(self, toks: Tokens) -> None:
        """ Handle CPP_PASTED tokens.
        If value is for a valid token, set its type.
        Otherwise, replace with two or more tokens parsed from the value.
        The token list is modified in-place.
        """
        i = 0
        lex = self.macros.lexer.clone()
        while i < len(toks):
            tok = toks[i]
            if tok.type is TokType.CPP_PASTED:
                lex.input(tok.value)
                toks2 = list(lex)

                if len(toks2) != 1:
                    # Insert spaces between the tokens.
                    for j in reversed(range(1, len(toks2))):
                        lex.input(' ')
                        toks2[j : j] = list(lex)
                    del toks[i]
                    for t in reversed(toks2):
                        newtok = copy.copy(tok)
                        newtok.value = t.value
                        newtok.type = t.type
                        toks.insert(i, newtok)
                    self.prep.on_error_token(tok,
                        f"Pasting result {tok.value!r} is not a valid token.")
                    continue
                else:
                    tok.type = toks2[0].type
            i += 1

    def sameas(self, other: Self) -> bool:
        """ True if the two definitions are the same, as per Standard 6.10.3 paragraph 2. """
        if type(self) is not  type(other): return False
        # Compare the replacement lists, treating all whitespace separation the same.
        for x, y in zip_longest(self.value, other.value):
            if x is None or y is None: return False
            if x.value != y.value: return False
        return True

    def show(self):
        if self.arglist is None:
            args = ""
        else:
            args = "(%s)" % ', '.join(self.arglist)
        return "%s%s = %s" % (self.name, args, ''.join(x.value for x in self.value))

class ObjMacro(Macro):
    is_func: bool = False

    def subst(self, call: MacroCall) -> Tokens:
        """ Substitutes actual arguments (expanded) for parameter names.
        New token list is returned.
        Same as FuncMacro.subst(), but without args or # or __VA_OPT__.
        """
        out: Tokens = Tokens()
        # Use self.repl rather than self.value because it has whitespace removed
        #   after #.
        toks: Tokens = Tokens(self.repl)

        while toks:
            t = toks[0]
            val: str = t.value
            if val == '##':
                op = copy.copy(t)
                t2 = (toks[1])
                if t2.value == '##':
                    # ## + ##.  becomes single ##.
                    pass
                else:
                    # ## + other
                    out = self.glue(op, out, [t2])
                    del toks[1]
            else:
                # Some other token.
                out.append(t)
            del toks[0]

        hsadd(call.hide, out)
        return out

    def __repr__(self):
        return f"{self.name}={self.repl}"

class FuncMacro(Macro):
    is_func: bool = True
    def __init__(self, macros: Macros, nametok: LexToken, value: Tokens, arglist, variadic: bool):
        self.arglist = arglist
        self.variadic = variadic
        if variadic:
            self.vararg = arglist[-1]

        super().__init__(macros, nametok, value)

    def parse_opt(self, toks: Iterator[Tuple[int, LexToken]], tok: LexToken) -> Tuple[int, int]:
        """ Finds the balanced '(' ... ')' following __VA_OPT__.
        Consumes tokens in the iterator up to the closing ')'
        Returns the indices of the '(' and ')' tokens, or (0, 0) if ill-formed.
        """
        for i, tok in toks:
            if not tok: return 0
            if not tok.type.ws:
                break
        # tok is first non-whitespace token.
        start = i
        if tok.value != '(': return 0
        nesting = 1
        for i, tok in toks:
            v = tok.value
            if v == '(': nesting += 1
            elif v == ')':
                nesting -= 1
                if not nesting:
                    # Succeeded in grabbing all the replacement tokens.
                    name = copy.copy(tok)
                    name.value = '__VA_OPT__'
                    return FuncMacro(self.macros, name, self.value[start+1: i], self.arglist, variadic=True)
        return None

    def subst(self, call: MacroCall) -> Tokens:
        """ Substitutes actual arguments (expanded) for parameter names.
        For now, ignoring # and ## tokens.
        New token list is returned.
        Not called for object macros.  Rather, hsadd() is called directly
        """
        out: Tokens = Tokens()
        args = call.args
        # Use self.repl rather than self.value because it has whitespace removed
        #   before and after ## and after #.
        toks: Tokens = Tokens(self.repl)
        params: list[str] = self.arglist

        expanded: Mapping[int, Tokens] = { }        # argnum -> expansion of that arg
        str_expansion: Mapping[int, Token] = { }    # argnum -> stringize of that arg
        with self.prep.nest():
            for argnum, arg in enumerate(args):
                self.log.arg(self.arglist[argnum], arg, call.tokens[call.positions[argnum]], nest=1)
                if argnum in self.expand_args:
                    exp = Tokens(args[argnum])
                    exp = self.macros.expand(exp, quiet=True)
                    expanded[argnum] = exp
                if argnum in self.string_args:
                    # Stringize the arg
                    string = self.stringize(args[argnum])
                    str_expansion[argnum] = string
                    self.log.stringize(self.arglist[argnum], str_expansion[argnum], call.tokens[call.positions[argnum]])

        # Do stringizing first, as this is more binding than other operators
        #   and doesn't change the positions of anything else.
        for info in self.string_points:
            t = toks[info.pos]
            t.value = str_expansion[t.argnum]
            t.type = TokType.CPP_STRING

        # Other replacements, in right-to-left order.
        for i in self.replace_points:
            # Either expansion or concatenation or __VA_OPT__.
            if i in self.concat_points:
                argnum = self.argnums[i]
                if args[argnum]:
                    toks[i:i+1] = args[argnum]
                else:
                    toks[i].type = TokType.CPP_PLACEMARKER
                    toks[i].value = ''

            elif i in self.opt_points:
                # Opt expansion.  The entire "__VA_OPT__ ( toks )" is replaced by
                # either an expansion of the replacement list, if __VA_ARGS__ is nonempty
                # or by a placemarker.
                var = args[-1]
                info = self.opts[i]
                if var:
                    exp = info.repl.subst(call)
                    info.repl.concat(exp)
                    toks[i : i+1] = exp
                else:
                    toks[i].type = TokType.CPP_PLACEMARKER
                    toks[i].value = ''

            # Normal expansion.  Argument is macro expanded first
            else:
                argnum = self.argnums[i]
                toks[i:i+1] = expanded[argnum]

        while toks:
            t = toks[0]
            val: str = t.value
            if val == '##':
                op = copy.copy(t)
                t2 = (toks[1])
                # Remove whitespace
                while t2.type.ws:
                    del toks[1]
                    t2 = toks[1]
                if t2.value in params:
                    # ## + param
                    arg = args[params.index(t2.value)]
                    if arg:
                        # ## + non-empty param
                        out = self.glue(op, out, arg)
                        del toks[1]
                    else:
                        # ## + empty param
                        pass
                elif t2.value == '##':
                    # ## + ##.  becomes single ##.
                    pass
                else:
                    # ## + other
                    out = self.glue(op, out, [t2])
                    del toks[1]
            elif val in params:
                # Look for ## next.
                if len(toks) > 1 and toks[1].value == '##':
                    # param + ##.
                    arg = args[params.index(t.value)]
                    if not arg:
                        # empty param + ##
                        del toks[1]
                    else:
                        # non-empty param + ##
                        out += arg
                else:
                    # plain param, not followed by ##.
                    out += self.macros.expand(args[params.index(val)])
            else:
                # Some other token.
                out.append(t)
            del toks[0]

        if self.num_concat:
            self.fix_pastes(out)
        hsadd(call.hide, out)
        return out

    # ----------------------------------------------------------------------
    # expandargs()
    #
    # Given a function Macro and list of arguments (each a token list), this method
    # returns an expanded version of a macro.  The return value is a token sequence
    # representing the replacement macro tokens.
    #
    # It performs argument substitution according to C99 Section 6.20.3.1.
    # It also perfoms the # operator according to C99 Section 6.20.3.1.
    # It does NOT perform the ## operator, because this applies to object macros as well.
    #   Any arguments involved in a ##, however, are not macro expanded before substitution.
    # ----------------------------------------------------------------------

    def expandargs(self,
            args: Tokens,
            tokens: Tokens,
            positions: list[int],
            caller: LexToken,
            #expanding_from: list[str],
            ):
        """Given a Macro and list of arguments (each a token list), this method
        returns an expanded version of a macro.  The return value is a token sequence
        representing the replacement macro tokens"""
        prep = self.prep

        # Make a copy of the macro token sequence
        rep = Tokens(self.repl)

        # There are three ways each occurrence of a parameter name in the replacement list
        # can be treated:
        #   1. Stringized, if preceded by # (ignoring whitespace).
        #   2. Left as-is, if preceded or followed by ## (ignoring whitespace).
        #   3. Fully macro expanded, otherwise.
        # Also, a __VA_OPT__(...) will be replaced.

        # Map argument number to expansion or stringizing of that argument.
        #   Only if the arg is used at least once.
        #   This avoids repeated computation for an arg that is used more than once.
        expanded: Mapping[int, Tokens] = { }
        str_expansion: Mapping[int, PpTok] = { }
        for argnum, arg in enumerate(args):
            self.log.arg(self.arglist[argnum], arg, tokens[positions[argnum]], nest=1)
            if argnum in self.expand_args:
               with prep.nest():
                  exp = expanded[argnum] = Tokens(args[argnum])
                  self.macros.expand(exp, quiet=True)
                  #self.expand(exp, expanding_from, quiet=True)
            if argnum in self.string_args:
                # Stringize the arg
                string = self.stringize(args[argnum])
                str_expansion[argnum] = string
                with prep.nest():
                    self.log.stringize(self.arglist[argnum], str_expansion[argnum], tokens[positions[argnum]])


        # Make string expansion patches.  These do not alter the length of the replacement sequence.
        # The Info reflects the location of the '#' token, and references the following parameter token.
        # The param token and any whitespace in between have been removed from the token list.
        for i in self.string_points:
            rep[i] = copy.copy(rep[i])
            rep[i].value = str_expansion[self.argnums[i]]
            rep[i].type = TokType.CPP_STRING
            while rep[i-1].type.ws:
                rep[i-1].lexer.erase(rep[i-1])
                i -= 1
            rep[i-1].lexer.erase(rep[i-1])

        # Make the variadic macro comma patch.  If the variadic macro argument is empty, we get rid
        comma_patch = False
        if self.variadic and not args[-1]:
            for i in self.var_comma_patch:
                rep[i] = None
                comma_patch = True

        # Make all other patches.   The order of these matters.  It is assumed that the patch list
        # has been sorted in reverse order of patch location since replacements will cause the
        # size of the replacement sequence to expand from the patch point.
        
        for i in self.replace_points:
            # Either expansion or concatenation or __VA_OPT__.
            if i in self.concat_points:
                argnum = self.argnums[i]
                if args[argnum]:
                    rep[i:i+1] = args[argnum]
                else:
                    rep[i].type = TokType.CPP_PLACEMARKER
                    rep[i].value = ''

            elif i in self.opt_points:
                # Opt expansion.  The entire "__VA_OPT__ ( toks )" is replaced by
                # either an expansion of the replacement list, if __VA_ARGS__ is nonempty
                # or by a placemarker.
                var = args[-1]
                info = self.opts[i]
                if var:
                    exp = info.repl.expandargs(args, tokens, positions, caller)
                    info.repl.concat(exp)
                    rep[i : i+1] = exp
                else:
                    rep[i].type = TokType.CPP_PLACEMARKER
                    rep[i].value = ''

            # Normal expansion.  Argument is macro expanded first
            else:
                argnum = self.argnums[i]
                rep[i:i+1] = expanded[argnum]

        # Get rid of removed comma if necessary
        ### TODO: Check this out.
        if comma_patch:
            rep = [_i for _i in rep if _i]

        return rep

    @staticmethod
    def stringize(toks: Tokens) -> str:
        strtokens: Tokens = Tokens()
        has_ws = None
        for tok in toks:
            if tok.type.ws:
                if not strtokens: continue     # Skip initial whitespace.
                has_ws = tok                # Start or continue a string
            else:
                if has_ws:
                    has_ws = copy.copy(has_ws)
                    has_ws.value = ' '
                    strtokens.append(has_ws)
                    has_ws = None
                strtokens.append(copy.copy(tok))

        string = str(Tokens(strtokens))
        string = string.replace("\\","\\\\").replace('"', '\\"')
        string = f'"{string}"'
        return string

    def sameas(self, other: FuncMacro) -> bool:
        """ True if the two definitions are the same, as per Standard 6.10.3 paragraph 2. """
        if not super().sameas(other): return False
        return self.arglist == other.arglist

    def __repr__(self):
        return f"{self.name}({self.arglist})={self.value}"

class Macros(dict[LexToken, 'Macro']):
    """ All the macro definitions in a transltion unit --
    top level source file, all included headers, and predefined stuff.

    Definitions are in the form of a token iterator (which will be consumed).
    Expansion generates a token sequence from an input token iterator
        (which will be consumed).
    """

    def __init__(self, prep: Preprocessor):
        self.prep = prep
        self.lexer = prep.lexer.clone()
        self.log = prep.log

    def define (self, defn: Iterable[LexToken], src: str = '') -> None:
        """Define a new macro from tokens following #define in a directive."""

        linetok: Iterator[LexToken] = iter(defn)
        try:
            # Name is first token, skipping whitespace.
            name = next(no_ws(linetok))
            if name.type is TokType.CPP_OBJ_MACRO:
                # A normal macro
                m = ObjMacro(self, name, self.prep.tokenstrip(list(linetok)))
            elif name.type is TokType.CPP_FUNC_MACRO:
                # A macro with arguments
                prep = self.prep
                arglist = no_ws(linetok)
                variadic = False
                # Get the argument names.  Gets tokens through the closing ')'.
                def argtokens() -> Iterator[LexToken]:
                    for tok in arglist:
                        if tok.type is TokType.CPP_RPAREN:
                            return
                        yield tok

                def iter_argnames() -> Iterator[str]:
                    nonlocal variadic
                    seen_comma: LexToken = None
                    tokens: Iterator[LexToken] = argtokens()
                    next(tokens)                # Opening '('.
                    for tok in tokens:
                        # The arg name -- ID or ELLIPSIS.
                        seen_comma = None
                        if tok.type is TokType.CPP_ID:
                            if variadic:
                                prep.on_error_token(
                                    tok, "No more arguments may follow a variadic argument")
                                return
                            yield tok.value
                        elif tok.type is TokType.CPP_ELLIPSIS:
                            yield '__VA_ARGS__'
                            variadic = True
                        else:
                            prep.on_error_token(tok, f"Invalid macro argument {tok.value!r}")
                            break
                        # Expect a comma or end of list.
                        for tok in tokens:
                            if tok.type is TokType.CPP_COMMA:
                                seen_comma = tok
                            else:
                                prep.on_error_token(tok, "Expected ',' or ')'.")
                                return
                            break
                    if seen_comma:
                        prep.on_error_token(seen_comma, "Missing name after comma.")
                        prep.on_error()

                argnames = list(iter_argnames())
                m = FuncMacro(self, name, self.prep.tokenstrip(Tokens(linetok)), argnames, variadic)

            else:
                prep.on_error_token(name,"Bad macro definition")
                return
            # OK.  Either an object or a function macro.
            # Check for redefinition
            name = m.name
            if name in self:
                older = self[name]
                if not m.sameas(older):
                    self.prep.on_error_token(m.nametok, f"Macro {name} redefined with different meaning.")
            self[name] = m

        except:
            raise

    def define_from_text(self, text: str, lexer: Lexer):
        lexer.begin('DEFINE')
        lexer.input(text.strip())
        self.define(iter(lexer))

    def defined(self, name: str) -> bool:
        return name in self

    def undef(self, name: str) -> None:
        if name in self:
            del self[name]

    def expand(self, toks: Tokens,
               evalexpr: bool = False,              # This is a control expression.  Handle 'defined'.
               **kwds) -> Tokens:
        """ Expands sequence of tokens, replacing macro names with expansions thereof.
        Returns result sequence of tokens.
        """
        # Make a copy, so as not to alter the input.
        toks = Tokens(toks)

        i: int = 0                      # Current position in toks to examine
        j: int                          # Position after tokens to replace

        if evalexpr:
            self._replace_defined(toks)

        while i < len(toks):
            t = toks[i]
            # Possible actions:
            #   1.  Move to next position.
            #   2.  Replace several tokens with new tokens.
            name = t.value
            m = self.get(name)
            new = None
            if m:
                hide = t.hide
                call = MacroCall(m, toks, i, hide)
                if name in hide:
                    # Hidden.  Don't expand.
                    self.log.expand(m, False, t, hide, args=call.args)
                else:
                    if call.expanding:
                        try:
                            new = Tokens(call.subst())
                            #subster = MacSubst(call)
                            #new = list(subster())
                            print(Tokens(new))
                            j = call.endpos
                        except:
                            pass
                    #self.log.expand(m, True, t, hide, args=call.args)
                    #if call.m.is_func:
                    #    # Special adjustment for hide set.
                    #    hide = hide & toks[call.endpos - 1].hide
                    #if call.expanding:
                    #    hide = hide & toks[call.endpos - 1].hide | {name}
                    #    with self.prep.nest():
                    #        new = m.subst(call)
                    #        print('--', new)
                        j = call.endpos
            elif self.prep.expand_linemacro and name == '__LINE__':
                self.log.expand(t.value, True, t, [])
                t.type = self.prep.t_INTEGER
                t.value = self.prep.t_INTEGER_TYPE(t.lineno)
                new = [t]
                j = i + 1
                
            elif self.prep.expand_countermacro and name == '__COUNTER__':
                self.log.expand(t.value, True, t, [])
                t.type = self.prep.t_INTEGER
                t.value = self.prep.t_INTEGER_TYPE(self.prep.countermacro)
                self.prep.countermacro += 1
                new = [t]
                j = i + 1

            if new is None:
                # Not replacing the token.
                i += 1
            else:
                # Replacing initial token(s)
                toks[i : j] = new
                if evalexpr:
                    if self._replace_defined(toks):
                        self.prep.on_warn_token(
                            toks[0],
                            "Macro expansion of control expression contains 'defined'"
                            )
                self.log.write(f"-> {new}", token=t, nest=1)

        return toks

    def _replace_defined(self, tokens: Tokens) -> bool:
        """ Replace any 'defined X' or 'defined (X)' with integer 0 or 1.
        The token list is altered in-place.  Return True if anything replaced.
        """
        i = 0
        replaced = False
        while i < len(tokens):
            if tokens[i].type is TokType.CPP_ID and tokens[i].value == 'defined':
                needparen = False
                result = "0L"
                name = None
                # Look for either name or (name), skipping whitespace.
                j = i + 1
                while j < len(tokens):
                    if tokens[j].type.ws:
                        j += 1
                        continue
                    elif tokens[j].type is TokType.CPP_ID and not name:
                        name = tokens[j].value
                        if name in self:
                            result = "1L"
                        else:
                            repl = self.prep.on_unknown_macro_in_defined_expr(tokens[j])
                            if repl is None:
                                partial_expansion = True
                                result = 'defined(' + tokens[j].value + ')'
                            else:
                                result = "1L" if repl else "0L"
                        if not needparen: break
                    elif tokens[j].value == '(' and not needparen:
                        needparen = True
                    elif tokens[j].value == ')' and needparen:
                        break
                    else:
                        self.prep.on_error_token(tokens[i], "Malformed 'defined' in control expression")
                    j += 1
                if result.startswith('defined'):
                    tokens[i].type = TokType.CPP_ID
                    tokens[i].value = result
                else:
                    tokens[i].type = self.prep.t_INTEGER
                    tokens[i].value = self.prep.t_INTEGER_TYPE(result)
                replaced = True
                del tokens[i+1:j+1]
            i += 1
        return replaced

class MacroCall:
    """ An invocation of a macro.
    Can produce the replacement tokens with the subst() method.
    """
    m: Macro
    args: list[Tokens] = None
    tokens: Tokens                  # Where the invocation is located.
    pos: int                        # Index for the macro name.
    endpos: int = 0                 # Index after the invocation.
    positions: list[int]            # Index of start of each argument.
    error: str = ''                 # Message if there was an error.
    expanding: bool = True          # The Macro will be expanded.
    hide: Hide                      # Hidden names to be applied to all result tokens.

    def __init__(self, m: macro, tokens: Tokens, pos: int, hide: Hide):
        try:
            self.m = m
            positions = self.positions = []
            self.tokens = tokens
            self.pos = pos
            endpos = pos + 1            # Will become larger with an argument list.
            self.nametok: PpTok = tokens[pos]
            if not m.is_func:
                return                  # An object macro
            args = []
            nesting = 0
            argpos = 0
            nparams = len(m.arglist)
            varnum = m.variadic and nparams - 1
            nargs = 0

            def argtoken() -> bool:
                """ Handle token that is part of an arg.
                Return False if there is no arg list.
                """
                nonlocal argpos, argend
                if not argpos:
                    if not nesting:
                        # Missing argument list.
                        self.expanding = False
                        return False
                    argpos = i
                argend = i + 1
                return True

            def endarg() -> int:
                nonlocal argpos, argend
                if not argpos:
                    argpos = argend = i
                args.append(Tokens(tokens[argpos : argend]))
                positions.append(argpos)
                argpos = 0
                return nargs + 1

            # Looking for argument list.
            for i, tok in enumerate(tokens[pos+1:], pos + 1):
                if tok.type.ws:
                    continue
                val = tok.value
                if val == '(':
                    nesting += 1
                    if nesting == 2:
                        # Start of first arg.
                        argtoken()
                elif not nesting:
                    # No argument list.
                    self.expanding = False
                    return
                elif val == ')':
                    nesting -= 1
                    if not nesting:
                        # Ends the last arg
                        nargs = endarg()
                        endpos = i + 1
                        # Validate arg count.
                        if nargs != nparams:
                            if m.variadic:
                                # Variadic args allowed to be one short.  Supply trailing arg.
                                if nargs == nparams - 1:
                                    endarg()
                                else:
                                     self.error = (
                                         f"Macro {m.name} requires at least {nparams - 1}"
                                         f" argument(s) but was passed {nargs}")
                            elif nargs == 1 and not args[0] and nparams == 0:
                                # Empty only arg is OK if no params.
                                del args[:]
                                del positions[:]
                            else:
                                self.error = (
                                    f"Macro {m.name} requires exactly {nparams}"
                                    f" argument(s) but was passed {nargs}")
                        self.args = args
                        return
                    else:
                        argtoken()
                elif val == ',':
                    if nesting > 1:
                        argtoken()
                    elif m.variadic and len(args) == varnum:
                        if not argtoken():
                            return
                    else:
                        # Ends an arg and starts a new one.
                        nargs = endarg()
                        argpos = 0
                elif not argtoken():
                    return                  # No arg list found.

            # Missing argument list or end of one.
            self.expanding = False
            if nesting:
                self.error = f"Macro {m.name} missing ')' in argument list."
            return
            argend = 0                      # Make argend a local variable.
        finally:
            self.endpos = endpos
            self.m.macros.log.write(
                f"{Tokens(tokens[pos:endpos])}", token=tokens[pos])
            if m.is_func:
                # Special adjustment for hide set in function macto.
                hide = hide & tokens[endpos - 1].hide
            self.hide = hide | {m.name}

    def subst(self, m: Macro = None) -> Iterator[PpTok]:
        """ Result of substituting the macro replacement list using call arguments.
        All tokens are new copies.
        """
        s = MacSubst(self)
        yield from s(m)
        #result = expand(self.tokens[self.pos : self.endpos], self.m.macros)

        #if not self.expanding or self.error: return None
        #hide = self.nametok.hide | {self.nametok.value}
        #if self.m.is_func:
        #    rep = self.m.expandargs(self.args, self.tokens, self.positions, self.m)
        #    hide &= self.tokens[self.endpos - 1].hide
        #else:
        #    rep = Tokens(self.m.repl)

        #hide |= {self.nametok.value}

        #for tok in rep:
        #    if tok.type is TokType.CPP_ID:
        #        tok.hide |= hide

        #return rep

''' The following is modeled after David Prosser's algorithm, which can be found at
https://www.spinellis.gr/blog/20060626/cpp.algo.pdf
'''

class TokSubst:
    """ Performs substitution for a single token, or sequence of tokens,
    in the replacement list of a Macro.
    These objects form a syntax tree.
    When called with the details of a macro invocation, it iterates over the tokens
    resulting.  These are newly created tokens.
    """
    tok: PpToken                        # A token in the macro's replacement list.
                                        # It will be copied to produce the result.
    is_paste: ClassVar[bool] = False    # True for PasteTokSubst.

    def __init__(self, tok: PpTok):
        self.tok = tok

    def toks(self, sub: MacSubst) -> Iterator[PpTok]:
        """ Generates new token(s) for the given sub. """
        yield copy.copy(self.tok)

    def __repr__(self) -> str:
        return repr(self.tok)

class ParamTokSubst(TokSubst):
    """ Substitutes for a parameter name token.
    Generates the expansion of the corresponding argument.
    Can also stringify the argument (unexpanded).
    """
    argnum: int                         # The index of the parameter in macro parameter list.

    def __init__(self, tok: PpToken, argnum: int):
        super().__init__(tok)
        self.argnum = argnum

    def toks(self, sub: MacSubst) -> Iterator[PpTok]:
        yield from sub.expansion(self.argnum)

    def string(self, sub: MacSubst) -> Iterator[PpTok]:
        return sub.string(self.argnum)

class StringTokSubst(TokSubst):
    """ Substitutes for a '#' token plus any whitespace plus a parameter name. """
    argnum: int

    def __init__(self, tok: PpToken, argnum: int):
        super().__init__(tok)
        self.argnum = argnum

    def toks(self, sub: MacSubst) -> Iterator[PpTok]:
        """ Generates new token(s) for the given sub. """
        tok = copy.copy(self.tok)
        tok.value = sub.string(self.argnum)
        tok.type = TokType.CPP_STRING
        yield tok

class PasteTokSubst(TokSubst):
    """ Substitutes for one or more '##' tokens plus the tokens on either side
    and in between.  Whitespace next to each '##' is ignored.
    """
    is_paste: ClassVar[bool] = True             # True for PasteTokSubst.
    items: Sequence[TokSubst]                   # Two or more elements being pasted.
    ops: Sequence[PpTok]                        # One or more '##' tokens, between the items.

    def __init__(self, tok: PpToken, *items: TokSubst):
        super().__init__(tok)
        self.items = items

    def toks(self, sub: MacSubst) -> Iterator[PpTok]:
        """ Generates new token(s) for the given sub. """
        yield from sub.glue(self)

class OptTokSubst(TokSubst):
    """ Substitutes for entire "__VA_OPT__ ( repl list )".  Whitespace is ignored.
    """
    # Expands the replacement list from __VA_OPT__, if __VA_ARGS__ is non-empty.
    repl: FuncMacro                     

    def __init__(self, tok: PpToken, repl: FuncMacro):
        super().__init__(tok)
        self.repl = repl

    def toks(self, sub: MacSubst) -> Iterator[PpTok]:
        """ Generates new token(s) for the given sub. """
        return sub.expand_opt(self.repl)

class MacSubst:
    """ Performs the equivalent of Prosser's subst() function.
    It handles a specific invocation of a Macro in a list of tokens being expanded.
    When called, it generates newly copied PpTok's.
    Expands and stringizes parameter names, but does each only once.
    """
    call: MacroCall                     # The macro and its invocation.
    expanded: Mapping[int, Tokens]      # The expansion for a parameter number, if needed.
    strings: Mapping[int, PpTok]        # The stringization for a parameter number, if needed.

    def __init__(self, call: MacroCall):
        self.call = call

    def __call__(self, m: Macro = None) -> Iterator[PpTok]:
        if not m: m = self.call.m
        subs = m.subs
        toks: Iterable[PpTok] = chain(*(sub.toks(self) for sub in subs))
        hide = self.call.hide
        for tok in toks:
            tok.hide |= hide
            yield tok

    # Helper methods used by the TokSubst's ...

    def expansion(self, argnum: int) -> Tokens:
        """ The expansion (calculated only once) for the given argument.
        Caller cannot modify the return result.
        """
        try:
            expanded = self.expanded
            return expanded[argnum]
        except AttributeError:
            expanded = self.expanded = dict()
        except KeyError:
            pass
        exp: Tokens = Tokens(self.call.args[argnum])
        exp = self.call.m.macros.expand(exp, quiet=True)
        expanded[argnum] = exp
        return exp

    def string(self, argnum: int) -> PpTok:
        """ The stringization (calculated only once) for the given argument.
        Caller cannot modify the return result.
        """
        try:
            strings = self.strings
            return strings[argnum]
        except AttributeError:
            strings = self.strings = dict()
        except KeyError:
            pass
        string: PpTok = FuncMacro.stringize(self.call.args[argnum])
        strings[argnum] = string
        return string

    def glue(self, paster: PasteTokSubst) -> Iterator[PpTok]:
        """ Pastes two or more token generators together.
        Last token of one item is pasted to the first token of the next item.
        If a middle item has only one token, then multiple tokens are pasted.
        Special cases if any item is empty.
        """

        # State of processing:  o* t? I0 I*
        # o* = operands found so far to be pasted together.
        # t? = token iterated from I0.
        # I0 = current item, maybe partially iterated.
        # I* = any subsequent items, not iterated.

        items: list[TokSubst] = [item.toks(self) for item in paster.items]
        opnds: list[PpTok] = []         # o*
        t: PpTok | None = None          # t?
        i: int                          # current item index
        i = 0
        item: TokSubst = items[i]       # I0
        n = len(items)
        ops: list[PpTok] = paster.ops   # The '##' tokens between the items.
        op: PpTok
        lex: PpLex = paster.tok.lexer.clone()

        def nexttok() -> PpTok | None:
            """ Get next token from item, or None if item is empty. """
            for t in item:
                return t
            return None

        def nextitem() -> TokSubst | None:
            nonlocal i, op
            if i < n:
                item = items[i]
                if i: op = ops[i - 1]

                i += 1
                return item
            return None

        def paste() -> Iterator[PpTok]:
            """ Paste together the operand, then clear the operand list. """
            t: PpTok = copy.copy(op)
            t.value = ''.join(map(attrgetter('value'), opnds))
            t.type = TokType.CPP_PASTED
            opnds.clear()
            yield from lex.fix_paste(t)

        item = nextitem()
        if not item: return

        # o* is empty.  State = I0 I*
        while True:
            # Check if I0 is empty.
            t = nexttok()
            if t:
                break

            item = nextitem()
            if not item:
                # No more items.
                return

        while True:
            # (1) State = t I0 I*
            # Treat I0 as t*, state as t* o I*.  Emit the t*.
            for t2 in item:
                # State = t t2 I0 I*.  Emit t.
                yield t
                t = t2
            # State = t I*.

            opnds.append(t)

            # State = o+ I*
            item = nextitem()
            if not item:
                # State = o*
                yield from paste()
                return

            while True:
                # (2) State = o+ I0 I*
                # Check if I0 is empty
                while True:
                    t = nexttok()
                    if not t:
                        # State o+ I*
                        item = nextitem()
                        if not item:
                            # State = o+
                            yield from paste()
                            return
                        # State = o+ I0 I*.
                        continue
                    break

                # State = o+ t I0 I*.  Move t to o+.
                opnds.append(t)

                # State = o+ I0 I*.
                t = nexttok()
                if not t:
                    # State = o+ I*
                    item = nextitem()
                    if item:
                        # State = o+ I0 I*.  Go to (2).
                        continue
                    else:
                        # State = o+
                        yield from paste()
                        return

                # State = o+ t I0 I*
                yield from paste()
                break
            # State = t I0 I*.  Go to state (1).
            continue


    def expand_opt(self, repl: FuncMacro) -> Iterator[PpTok]:
        """ Generates tokens from a __VA_OPT__ (toks) expression,
        Expands the toks as if it were the replacement for calling the
        same macro, except if __VA_ARGS__ is empty, no tokens are generated.
        """
        var: Tokens = self.call.args[-1]
        if not var:
            return                      # Generate nothing.
        exp: Tokens = self.call.subst(m=repl)
        #exp: Tokens = repl.subst(self.call)
        yield from exp


#def expand(toks: Tokens, macros: Macros) -> Tokens:
#    """ Expands sequence of tokens, replacing macro names with expansions thereof.
#    Returns result sequence of tokens.
#    """
#    # Make a copy, so as not to alter the input.
#    toks = Tokens(toks)

#    i: int = 0                      # Current position in toks to examine
#    while i < len(toks):
#        t = toks[i]
#        # Possible actions:
#        #   1.  Move to next position.
#        #   2.  Replace several tokens with new tokens.
#        name = t.value
#        m = macros.get(name)
#        new = None
#        if m:
#            call = MacroCall(m, toks, i)
#            hide = t.hide
#            if name in hide:
#                # Hidden.  Don't expand.
#                pass
#            elif call.m.is_func:
#                # A function macro to expand (if there are arguments)
#                if call.args is None:
#                    # No arg list.  Don't expand.
#                    pass
#                else:
#                    # With argument list (may be empty).
#                    hide = hide & toks[call.endpos - 1].hide | {name}
#                    new = subst(m.value, m.arglist, call.args, hide)
#            else:
#                # An object macro to expand.
#                new = subst(call.m.value, {}, {}, hide | {name})
#        if new:
#            # Replacing initial token(s)
#            toks[i : call.endpos] = new
#        else:
#            # Not replacing the token.
#            i += 1

#    return toks

#def subst(toks: Tokens, params: list[str], args: list[Tokens], hide: Hide, ) -> Tokens:
#    """ Substitutes actual arguments (expanded) for parameter names.
#    For now, ignoring # and ## tokens.
#    This also works for object macros; there the params and args are empty.
#    New token list is returned.
#    """
#    out: Tokens = Tokens()
#    toks: Tokens = Tokens(toks)

#    while toks:
#        t = toks[0]
#        if t.value in params:
#            out += args[params.index(t.value)]
#        else:
#            out.append(t)
#        del toks[0]

#    hsadd(hide, out)
#    return out

def hsadd(hide: Hide, toks: Tokens) -> None:
    """ Adds given hide names to every one of given tokens.
    The tokens are modified in place.
    """
    for tok in toks:
        tok.hide |= hide

