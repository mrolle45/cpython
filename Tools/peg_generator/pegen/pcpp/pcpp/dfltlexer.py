""" dfltlexer.py

Builds the default PpLex, using a LexerFactory class instance.  It is
customized to the attributes of a Preprocessor object.  The lexer holds
another lexer which is the same except for being used to lex pasted values.

It uses the ply.lex module.  See "Alternative specification of lexers" section
in https://github.com/dabeaz/ply/blob/master/doc/ply.md for example of
creating a lexer using a class to hold the specs and create the lex.Lexer.
"""

from __future__ import annotations

from typing import Callable, NewType, TypeVar, Union

from pcpp.common import *
from pcpp.lexer import PpLex, RawLexer
from pcpp.ply import lex
from pcpp.ply.lex import LexToken
from pcpp.regexes import *
from pcpp.tokens import (PpTok, RawTok, Tokens, TokIter, TokLoc)
from pcpp.tokentype import *

__all__ = 'default_lexer'.split()

""" Lex Rules ...  A lex rule is given to lex.lex() by means of creating a
    Rule variable in the LexerFactory.  The name of the variable is the name
    of the rule, and has the form "t_(state_)*token".  The Rule is associated
    with:

    - a rulename, which is the name of the variable.
    - a regex, which the lexer tries to match with the current input data.
    - a type, which is the string 'token' taken from the rulename.
    - states, which is the set of all the (state_) components of the name.

    The lexer tries to match the current data with the regex, provided that
    the current lexer state is valid for the Rule state set.  If successful,
    it creates a LexToken object with the matched vlue and the Rule type.

    Two types of Rule are used:
      - StrRule.  Just a str.  The string is the regex.  The lexer returns the
        above LexToken.
      - FuncRule.  A callable rule(RawLexer, LexToken) -> LexToken.  The
        attribute rule.regex is the associated regex.  The lexer calls
        rule(lexer, the LexToken) and returns the result.

    The Lexer tries to match Rules in a particular order.  In all cases of
    ambiguous regexes, their corresponding Rules must appear in the desired
    order in this list.

      1. FuncRules created by proxy, in no particular order.
      2. Other FuncRules, in the order that the functions were defined in the
         LexerFactory code.
      3. StrRules, in descending order of the length of the regexes.  This is
         okay for ambiguous regexes if one regex is an initial substring of
         the other.

"""
StrRule = NewType('StrRule', str)
FuncRule = Callable[[RawLexer, LexToken], LexToken]
Rule = Union[StrRule, FuncRule]

def default_lexer(prep: Preprocessor) -> PpLex:
    lexer: PpLex = LexerFactory(prep).create()
    pasting: PpLex = LexerFactory(prep, pasting=True).create(pasting=True)
    lexer.pasting = pasting
    return lexer

class LexerFactory:
    """
    An instance of this class will create a lex.Lexer object with the build()
    method.  The class attributes are used by lex.lex(module=self).  The
    lexing rules are methods.
    """

    # All possible token names.  Some of these might not be used in a
    # particular instance.
    tokens: list[str]

    # All possible state names
    states: list[str]

    # Mapping of a punctuator value to the name(s) of its token.
    punct_values: Mapping[str, list[str]]

    # Regular expressions used.
    REs: RegExes

    # An enumeration type for tokens.
    TokType: type

    # The name of the lextab module created while creating the lexer.
    lextab: str = None

    def __init__(self, prep: Preprocessor, pasting: bool = False):
        """
        Construct to match the language of the Preprocessor, with option to
        customize it for lexing pasted values.
        """
        self.prep = prep
        self.lang = lang = prep.lang
        self.tokens = type_names
        self.states = [
            ('DIRECTIVE', 'inclusive'),
            ('INCLUDE', 'inclusive'),
            ('DEFINE', 'inclusive'),        # In #define directive
            ('OBJREPL', 'inclusive'),       # In #define after object macro name
            ('FUNCREPL', 'inclusive'),      # In #define after function macro name
            ('CONTROL', 'inclusive'),
        ]
        self.punct_values = collections.defaultdict(list)
        if pasting:
            self.REs = REs = prep.lang.REs_pasting
        else:
            self.REs = REs = prep.lang.REs

        # Set up the necessary class attributes.
        self.rules()

        # create or reuse the TokType enumeration.
        if pasting:
            TokType = prep.TokType
        else:
            TokType = make_TokType(prep, REs, self.punct_values)

        prep.TokType = self.TokType = TokType

        # We need to have the lextab module name be specific to the same
        # parameters that govern the content of the lexer, i.e.,
        # 
        # lang.cplus_ver selects C or C++ as the language, standard version
        # doesn't matter.  C++ enables the extra punctuators
        #
        # prep.emulate includes ` @ and $ as literals..
        #
        # prep.gnu enables R-strings for all languages.
        # pasting provides alternate regular expressions for pasted values.
        lextab = f"""\
            lextab\
            {'_c _cplusplus'.split()[bool(lang.cplus_ver)]}\
            {'_clang' * bool(lang.clang)}\
            {'_gcc' * bool(lang.gcc)}\
            {'_gnu' * bool(lang.gnu)}\
            {'_pasting' * bool(pasting)}\
            """
        self.lextab = lextab.replace(' ', '')


    # Rules and helper methods to create them.  Every Rule is stored in
    # self.t_(state_)*token.  Types of Rules, in the order tested by the
    # lexer:
    # - FuncRule.  Created with decorator @funcrule(regex, rulename).
    # - StrRule.  Created by assignment self.rulename = regex, or by
    #   plainrule().

    class RuleName:
        """
        The name of a lexer Rule.  Construct from the token name (beginning
        optionally with 't_') and optional state names (which can be joined by
        '_'s.
        """
        def __init__(self, token: str, *states: str):
            if token.startswith('t_'):
                token = token[2:]
            self.token = token
            self.states = [s for state in states for s in state.split('_')]

        def __repr__(self) -> str:
            return '_'.join(('t', *self.states, self.token))


        @classmethod
        def expand(cls, fact: LexerFactory, name: str) -> RuleName:
            """
            Create a RuleName from a complete name string (with 't_' prefix
            optional)
            """
            if name.startswith('t_'):
                name = name[2:]
            parts: list[str] = name.split('_')
            for i, part in enumerate(parts):
                if part not in [state[0] for state in fact.states]:
                    break
            return cls('_'.join(parts[i:]), *parts[:i])

    def add_rule(self, name: RuleName, rule: Rule) -> None:
        setattr(self, name, rule)

    def funcrule(self, regex: str, name: str = None
                 ) -> Callable[[FuncRule], FuncRule]:
        """
        Decorator for a function f(t: LexToken) -> Lextoken.  This creates ALL
        function rules.  They all have the same code line number, so that the
        Lexer will apply the regexes in alphabetical order of rule name.
        """
        def func(f: FuncRule) -> FuncRule:
            def proxy(t: LexToken) -> LexToken:
            #def proxy(self, t: LexToken) -> LexToken:
                return f(t)

            proxy.regex = regex
            if name:
                fname = f.__name__ = name
            else:
                fname = f.__name__
            proxy.__name__ = fname
            self.add_rule(fname, proxy)
        return func

    def makefunc(self, regex: str, token: str) -> None:
        """
        Creates a function t_{token} with regex and returning its argument.
        This is stored in cls.{name}.
        """
        @self.funcrule(regex, f't_{token}')
        def f(t: LexToken):
            return t

    def puncs(self) -> None:
        """
        Creates the subset of lexing rules which are for punctuator tokens.  A
        punctuator is any token which has a single specific value.
        """
        lang = self.lang

        # Helpers which create the rules for various ways of specifying them.

        # All token values that are restricted to C++.
        cplusplus_values = set('''
            .* ->* <=> :: 
            and and_eq bitand bitor compl not not_eq or or_eq xor xor_eq'''
            .split())

        def cplusplus_filt(value: str) -> bool:
            """ Is the value valid based on prep c++ flag? """
            if value in cplusplus_values:
                return self.lang.cplus_ver
            else:
                return True

        def makeproxy(regex: str, name: str, proxy: FuncRule) -> None:
            """ Create a rule self.{name} which calls proxy(). """
            @self.funcrule(regex, name)
            def f(t: LexToken) -> LexToken:
                return proxy(t)

        def puncrule(*values: str) -> Callable[[FuncRule], FuncRule]:
            """
            Decorator for a function f(t: LexToken) -> LexToken.

            A proxy for f is added to rules[name of f].  Name is modified for
            extra values.  punc_values[value] = name.
            """
            def func(f: FuncRule) -> FuncRule:
                fname = f.__name__
                punc(fname[2:], *values, proxy=f)
            return func

        def plainrule(regex: str, name: str) -> None:
            """ Creates a string-valued rule in self.t_{name}. """
            self.add_rule(f'{name}', regex)

        def punc(token: str, *values: str, func: bool = False,
                 proxy: FuncRule = None, state: str = '',
                 ) -> None:
            """
            Create a lex rule for the token, which matches the value exactly.
            C++ values are included only if --c++ on command line.  Any values
            after the first value become separate rules with different names.

            self.punct_values will map the value to the token name(s).
            self.t_{token} or self.t_{state}_{token} will be the actual rule.
            """
            if state: state += '_'
            for i, value in enumerate(filter(cplusplus_filt, values)):
                if i:
                    # We need to alter the type name here.
                    if value in cplusplus_values and value[0].islower():
                        token = f"CXX_{value.upper()}"
                    else:
                        token = f"CPP_ALT_{token[4:]}"
                assert token in self.tokens, f"Token name {token} unknown."
                # The name of the rule in the lexer.
                name = f"t_{state}{token}"
                regex = re.escape(value)
                if proxy:
                    makeproxy(regex, name, proxy)
                elif func:
                    self.makefunc(regex, name[2:])
                else:
                    plainrule(regex, name)
                self.punct_values[value].append(token)

        # Arithmetic operators
        punc('CPP_PLUS',            '+')
        punc('CPP_PLUSPLUS',        '++')
        punc('CPP_MINUS',           '-')
        punc('CPP_MINUSMINUS',      '--')
        punc('CPP_STAR',            '*')
        punc('CPP_FSLASH',          '/')
        punc('CPP_PERCENT',         '%')
        punc('CPP_LSHIFT',          '<<')
        punc('CPP_RSHIFT',          '>>')

        # Logical operators
        punc('CPP_LOGICALAND',      '&&',   'and')
        punc('CPP_LOGICALOR',       '||',   'or')
        punc('CPP_EXCLAMATION',     '!',    'not')

        # bitwise operators

        punc('CPP_AMPERSAND',       '&',   'bitand')
        punc('CPP_BAR',             '|',   'bitor')
        punc('CPP_HAT',             '^',   'xor')
        punc('CPP_TILDE',           '~',   'compl')

        # Comparison operators
        punc('CPP_EQUALITY',        '==')
        punc('CPP_INEQUALITY',      '!=',   'not_eq')
        punc('CPP_GREATEREQUAL',    '>=')
        punc('CPP_GREATER',         '>')
        punc('CPP_LESS',            '<')
        punc('CPP_LESSEQUAL',       '<=')
        punc('CXX_SPACESHIP',       '<=>')

        # Conditional expression operators
        punc('CPP_QUESTION',        '?')
        punc('CPP_COLON',           ':')

        # Member access operators
        punc('CPP_DOT',             '.')
        punc('CPP_DEREFERENCE',     '->')
        punc('CXX_DOTPTR',          '.*')
        punc('CXX_DEREFPTR',        '->*')
        punc('CXX_DCOLON',          '::')

        # Assignment operators
        punc('CPP_EQUAL',           '=')
        punc('CPP_XOREQUAL',        '^=',   'xor_eq')
        punc('CPP_MULTIPLYEQUAL',   '*=')
        punc('CPP_DIVIDEEQUAL',     '/=')
        punc('CPP_PLUSEQUAL',       '+=')
        punc('CPP_MINUSEQUAL',      '-=')
        punc('CPP_OREQUAL',         '|=',   'or_eq')
        punc('CPP_ANDEQUAL',        '&=',   'and_eq')
        punc('CPP_PERCENTEQUAL',    '%=')
        punc('CPP_LSHIFTEQUAL',     '<<=')
        punc('CPP_RSHIFTEQUAL',     '>>=')

        # Grouping and separators
        punc('CPP_LPAREN',          '(')
        punc('CPP_RPAREN',          ')')
        punc('CPP_LBRACKET',        '[',    '<:')
        punc('CPP_RBRACKET',        ']',    ':>')
        punc('CPP_LCURLY',          '{',    '<%')
        punc('CPP_RCURLY',          '}',    '%>')

        # Rules for '#', '##' and alternates.  '#' must be a func rule.  So to
        # make the rule for '##' appear earlier than '#' in the master regex,
        # both rules have to be functions using `def t_rule(t): return t`.
        # Likewise for '%:' and '%:%:'.

        # Put this before '#'.
        @puncrule('##', '%:%:')
        def t_CPP_DPOUND(t: PpTok) -> PpTok:
            return t

        @puncrule('#', '%:')
        def t_CPP_POUND(t: PpTok) -> PpTok:
            # A PpLex indicates if at the start of a line, RawLexer does not.
            try:
                if t.lexer.owner.only_ws_this_line:
                    t.lexer.begin('DIRECTIVE')
                    t.type = 'CPP_DIRECTIVE'
            except AttributeError: pass
            return t

        self.punct_values['#'].append('CPP_DIRECTIVE')
        # '##' is special in any macro definition.  '#' is special only in
        # function macro.
        punc('CPP_PASTE',           '##', '%:%:',   state='FUNCREPL_OBJREPL')
        punc('CPP_MKSTR',           '#',  '%:',     state='FUNCREPL')

        punc('CPP_COMMA',           ',')
        punc('CPP_SEMICOLON',       ';')
        punc('CPP_ELLIPSIS',        '...')

        # Single-characters not in the source character set (valid in GCC).
        if lang.emulate:
            punc('CPP_DOLLAR',      '$')
            punc('CPP_AT',          '@')
            punc('CPP_GRAVE',       '`')

        punc('CPP_DELTA', r'\u03b4')

    def rules(self) -> None:
        """ Create all the lexer rules as class attributes. """
        lang = self.lang
        REs = self.REs

        # Whitespace, one or more consecutive whitespace character(s) 
        # other than newline.  ASCII only, no unicode.
        self.t_ANY_CPP_WS = r'((?a)(?!\n)\s)+'

        # Special newline in a directive.  Returns to INITIAL state.

        # Place before the newline rule below!
        # A newline in any state other than INITIAL returns to INITIAL.
        @self.funcrule(REs.newline)
        def t_DIRECTIVE_INCLUDE_DEFINE_OBJREPL_FUNCREPL_CONTROL_CPP_NEWLINE(t):
            t.lexer.begin('INITIAL')
            return t

        # Newline, other than in a directive.
        self.makefunc(REs.newline, 'INITIAL_CPP_NEWLINE')

        # Certain directive names.
        @self.funcrule(r'[A-Za-z_][\w_]*')
        def t_DIRECTIVE_CPP_ID(t):
            
            if t.value == 'include':
                t.lexer.begin('INCLUDE')
            elif t.value == 'define':
                t.lexer.begin('DEFINE')
            elif t.value.endswith('if'):
                t.lexer.begin('CONTROL')
            else:
                t.lexer.begin('INITIAL')
            return t

        # Floating literal.  Put these before integer.
        self.makefunc(REs.float, 'CPP_FLOAT', )
        self.makefunc(REs.dotfloat, 'CPP_DOT_FLOAT', )

        # Integer constant 
        self.makefunc(REs.int, 'CPP_INTEGER', )

        # General pp-number, other than integer or float constant.  (C99
        # 6.4.8, C++14 5.9).  Put this after integer and float.
        @self.funcrule(REs.ppnum)
        def t_CPP_NUMBER(t):
            message = f'Illegal preprocessing number: {t.value}'
            return self.error(t, message, warn=True, keep_type=True)

        # Char and String tokens with a prefix could also lex as an ident.
        # They need to be functions so as to come before identifiers.

        # String literal.  
        # Terminating " required on same logical line.
        self.makefunc(REs.string, 'CPP_STRING')

        # Raw string literal.  
        # Terminating matching delimiter required, possibly on later logical
        # line.  Only tokenized if C++ or (GCC with GNU extensions) or clang.

        if lang.cplus_ver or lang.emulate:
            self.makefunc(REs.rstring, 'CPP_RSTRING')

        # h-type and q-type header names.  Only used in INCLUDE state.  
        # Note, some things in these names are undefined behavior (C99 6.4.7),
        # and this is checked in the preprocessor.include() method.

        self.t_INCLUDE_CPP_H_HDR_NAME = REs.hhdrname
        self.t_INCLUDE_CPP_Q_HDR_NAME = REs.qhdrname

        ## Character constant (L|U|u|u8)?'cchar*', within a CONTROL expression.  
        ## Terminating ' required.  yacc evaluates this differently.
        self.makefunc(REs.char, 'CONTROL_CPP_EXPRCHAR')

        # Character constant (L|U|u|u8)?'cchar*'ud_sfx?.  
        # Terminating ' required.
        self.makefunc(REs.char, 'CPP_CHAR')

        # Identifier.  Place this after string and char literals.
        # The RE only matches up to the first codepoint, which may or may not
        # be valid.
        @self.funcrule(REs.identifier)
        def t_CPP_ID(t: Token) -> Token:
            m = t.lexer.lexmatch
            groups = m.group('unistart', 'cont')
            if any(groups):
                # Contains a codepoint.
                import pcpp.unicode as uni
                uni.ident(self.lang, *groups, t, m)
            return t

        # Object and function macro identifiers.  Place after char and string
        # literals and identifiers.  
        # An identifier may be modified if it contains a unicode codepoint.  A
        # token function needs to find the modified identifier string.  And
        # then it needs to look for a '(' immediately following, then set the
        # token type accordingly.
        #  
        # CPP_FUNC_MACRO is the macro name, if followed immediately by '('.
        # CPP_OBJ_MACRO is the macro name, otherwise.

        @self.funcrule(REs.identifier)
        def t_DEFINE_CPP_MACRO(t):
            t = self.t_CPP_ID(t)
            m = t.lexer.lexmatch
            nextpos = t.lexer.lexpos
            nextchar = m.string[nextpos : nextpos + 1]
            if t.type == 'error':
                pass
            elif nextchar == '(':
                t.type = 'CPP_FUNC_MACRO'
                t.lexer.begin('FUNCREPL')
            else:
                t.type = 'CPP_OBJ_MACRO'
                t.lexer.begin('OBJREPL')
            return t

         # Block comment (C), possibly spanning multiple lines.  
        self.t_CPP_COMMENT1 = r'(/\*(.|\n)*?\*/)'

        # Line comment (C++).  PCCP accepts them in C files also.  
        self.t_CPP_COMMENT2 = r'(//[^\n]*)'
    
        self.puncs()

        def error(self, t: PpTok, msg: str, keep_type: bool = False) -> PpTok:
            if not keep_type:
                t.type = TokType.error
            if t.lexer.owner.errors:
                t.lexer.prep.on_error_token(t, msg)
            return t

        @self.funcrule(None)
        def t_ANY_error(t):
            # Check for unmatched quote character.  
            if t.value[0] in '\'\"':
                endline = t.value.find('\n')
                t.value = t.value[:endline]
                message = f"Unmatched quote character {t.value}"
            else:
                t.value = t.value[0]
                message = f"Illegal character {t.value!r}"
            t.lexer.owner.skip(len(t.value))
            return self.error(t, message)

    @classmethod
    def error(cls, t: PpTok, msg: str, keep_type: bool = False,
              warn: bool = False) -> PpTok:
        owner: RawLexer = t.lexer.owner
        if not keep_type:
            t.type = owner.TokType.error.name
        if owner.errors:
            owner.prep.on_error_token(t, msg, warn=warn)
        return t

    def create(self, pasting: bool = False) -> PpLex:
        """
        Create the PpLex from attributes of self.  Optionally create a pasting
        version.
        """
        # TODO: Build with optimize=True if the lextab is up to date.  The
        # lexer might use anything in the pcpp package.
        lexer = self.build(self.__dict__, lextab=self.lextab)
        lexer.prep = self.prep
        lexer.TokType = self.TokType
        lexer.REs = self.REs
        return PpLex(from_lexer=lexer)

    @staticmethod
    def build(items: dict, **kwds) -> Lexer:
        """
        Makes the lex.Lexer for given variables, preserving order of
        rules.
        """
        for key, value in items.items():
            exec(f"{key} = value")

        lexer = lex.lex(**kwds)
        return lexer

