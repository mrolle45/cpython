""" dfltlexer.py

Builds the default PpLex, using a LexerFactory class instance.  It is
customized to the attributes of a Preprocessor object.

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
    return LexerFactory.create(prep)

class LexerFactory:
    """
    An instance of this class will create a lex.Lexer object with the build()
    method.  The class attributes are used by lex.lex(module=self).  The
    lexing rules are methods.
    """

    # Some attributes are independent of the prep, so they can be specified here directly.

    tokens: list[str] = type_names

    states = [
        ('DIRECTIVE', 'inclusive'),
        ('INCLUDE', 'inclusive'),
        ('DEFINE', 'inclusive'),        # In #define directive
        ('OBJREPL', 'inclusive'),       # In #define after object macro name
        ('FUNCREPL', 'inclusive'),      # In #define after function macro name
        ('CONTROL', 'inclusive'),
    ]

    # Mapping of a punctuator value to the name of its token.
    punct_values: Mapping[str, str] = {}

    t_CPP_ID = r'\+'

    # Rules and helper methods to create them.  Every Rule is stored in
    # cls.t_(state_)*token.  Types of Rules, in the order tested by the lexer:
    # - Proxy FuncRule.  Created by makeproxy() for a rule with just a regex
    #   and no function, to place it first in the order.
    # - Other FuncRule.  Created with decorator @funcrule(regex, rulename).
    # - Plain StrRule.  Created by assignment cls.rulename = regex.

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

    @classmethod
    def add_rule(cls, name: RuleName, rule: Rule) -> None:
        setattr(cls, name, rule)

    @classmethod
    def funcrule(cls, regex: str, name: str = None
                 ) -> Callable[[FuncRule], FuncRule]:
        """
        Decorator for a function f(t: LexToken) -> Lextoken.  This creates ALL
        function rules.  They all have the same code line number, so that the
        Lexer will apply the regexes in alphabetical order of rule name.
        """
        def func(f: FuncRule) -> FuncRule:
            def proxy(self, t: LexToken) -> LexToken:
                return f(t)

            proxy.regex = regex
            if name:
                fname = f.__name__ = name
            else:
                fname = f.__name__
            proxy.__name__ = fname
            cls.add_rule(fname, proxy)
        return func

    @classmethod
    def makefunc(cls, regex: str, token: str) -> None:
        """
        Creates a function t_{token} with regex and returning its argument.
        This is stored in cls.{name}.
        """
        @cls.funcrule(regex, f't_{token}')
        def f(t: LexToken):
            return t

    @classmethod
    def puncs(cls) -> None:
        """
        Creates the subset of lexing rules which are for punctuator tokens.  A
        punctuator is any token which has a single specific value.
        """
        prep = cls.prep

        # Helpers which create the rules for various ways of specifying them.

        # All token values that are restricted to C++.
        cplusplus_values = set('''
            .* -.* <=> :: 
            and and_eq bitand bitor compl not not_eq or or_eq xor xor_eq'''
            .split())

        def cplusplus_filt(value: str) -> bool:
            """ Is the value valid based on prep c++ flag? """
            if value in cplusplus_values:
                return cls.prep.cplus_ver
            else:
                return True

        def makeproxy(regex: str, name: str, proxy: FuncRule) -> None:
            """ Create a rule cls.{name} which calls proxy(). """
            @cls.funcrule(regex, name)
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
            """ Creates a string-valued rule in cls.t_{name}. """
            cls.add_rule(f'{name}', regex)

        def punc(token: str, *values: str, func: bool = False,
                 proxy: FuncRule = None, state: str = '',
                 ) -> None:
            """
            Create a lex rule for the token, which matches the value exactly.
            C++ values are included only if --c++ on command line.  Any values
            after the first value become separate rules with different names.

            cls.punct_values will map the value to the token name.
            cls.t_{token} or cls.t_{state}_{token} will be the actual rule.
            """
            if state: state += '_'
            for i, value in enumerate(filter(cplusplus_filt, values)):
                if i:
                    # We need to alter the type name here.
                    if value in cplusplus_values and value[0].islower():
                        token = f"CXX_{value.upper()}"
                    else:
                        token = f"CPP_ALT_{token[4:]}"
                assert token in cls.tokens, f"Token name {token} unknown."
                # The name of the rule in the lexer.
                name = f"t_{state}{token}"
                regex = re.escape(value)
                if proxy:
                    makeproxy(regex, name, proxy)
                elif func:
                    cls.makefunc(regex, name[2:])
                else:
                    plainrule(regex, name)
                cls.punct_values[value] = token

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
        #   Place && and || before & and |
        punc('CPP_LOGICALAND',      '&&',   'and',    func=True)
        punc('CPP_LOGICALOR',       '||',   'or',     func=True)
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

        # '##' is special in any macro definition.  '#' is special only in
        # function macro.
        punc('CPP_PASTE',           '##', '%:%:',   state='FUNCREPL_OBJREPL')
        punc('CPP_MKSTR',           '#',  '%:',     state='FUNCREPL')

        punc('CPP_COMMA',           ',')
        punc('CPP_SEMICOLON',       ';')
        punc('CPP_ELLIPSIS',        '...')

        # Single-characters not in the source character set (valid in GCC).
        if prep.emulate:
            punc('CPP_DOLLAR',      '$')
            punc('CPP_AT',          '@')
            punc('CPP_GRAVE',       '`')

    @classmethod
    def rules(cls) -> None:
        """ Create all the lexer rules as class attributes. """
        prep = cls.prep
        REs = cls.REs

        # Whitespace, one or more consecutive whitespace character(s) 
        # other than newline.
        cls.t_ANY_CPP_WS = r'((?!\n)\s)+'

        # Special newline in a directive.  Returns to INITIAL state.

        # Place before the newline rule below!
        # A newline in any state other than INITIAL returns to INITIAL.
        @cls.funcrule(REs.newline)
        def t_DIRECTIVE_INCLUDE_DEFINE_OBJREPL_FUNCREPL_CONTROL_CPP_NEWLINE(t):
            t.lexer.begin('INITIAL')
            return t

        # Newline, other than in a directive.
        cls.makefunc(REs.newline, 'INITIAL_CPP_NEWLINE')

        # Certain directive names.
        @cls.funcrule(r'[A-Za-z_][\w_]*')
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

        # Identifier 
        cls.t_CPP_ID = REs.ident

        # Object and function macro identifiers.  
        # CPP_FUNC_MACRO is the macro name, if followed immediately by '('.
        # CPP_OBJ_MACRO is the macro name, otherwise.
        @cls.funcrule(rf'{REs.ident}(?=\()')
        def t_DEFINE_CPP_FUNC_MACRO(t):
            t.type = 'CPP_FUNC_MACRO'
            t.lexer.begin('FUNCREPL')
            return t
        # Place this AFTER FUNC_MACRO!
        @cls.funcrule(REs.ident)
        def t_DEFINE_CPP_OBJ_MACRO(t):      
            t.type = 'CPP_OBJ_MACRO'
            t.lexer.begin('OBJREPL')
            return t

        ## Paste operator ('##') in a macro definition only.  Place before
        ## punctuator rules.
        #cls.makefunc(re.escape('##'), 'MACREPL_CPP_PASTE', )
        #cls.makefunc('%:%:', 'MACREPL_CPP_ALT_PASTE', )
        ##t_MACREPL_CPP_ALT_PASTE = '%:%:'

        ## Stringize operator ('#') in a macro definition only.  Place before
        ## punctuator rules and after ##.
        #cls.makefunc(re.escape('#'), 'MACREPL_CPP_MKSTR', )
        #cls.makefunc('%:', 'MACREPL_CPP_ALT_MKSTR', )
        ##t_MACREPL_CPP_ALT_MKSTR = '%:'

        # Floating literal.  Put these before integer.
        cls.makefunc(REs.float, 'CPP_FLOAT', )
        cls.makefunc(REs.dotfloat, 'CPP_DOT_FLOAT', )

        # Integer constant 
        cls.makefunc(REs.int, 'CPP_INTEGER', )

        # General pp-number, other than integer or float constant.  (C99
        # 6.4.8, C++14 5.9).  Put this after integer and float.
        @cls.funcrule(REs.ppnum)
        def t_CPP_NUMBER(t):
            message = f'Illegal preprocessing number: {t.value}'
            return cls.error(t, message, warn=True, keep_type=True)

        # String literal.  # Terminating " required on same logical line.
        cls.t_CPP_STRING = REs.string

        # Raw string literal.  
        # Terminating matching delimiter required, possibly on later logical line.
        # Only tokenized if C++ or (C with GNU extensions).

        if prep.cplus_ver or prep.emulate:
            cls.t_CPP_RSTRING = REs.rstring

        # h-type and q-type header names.  Only used in INCLUDE state.  
        # Note, some things in these names are undefined behavior (C99 6.4.7),
        # and this is checked in the preprocessor.include() method.

        cls.t_INCLUDE_CPP_H_HDR_NAME = REs.hhdrname
        cls.t_INCLUDE_CPP_Q_HDR_NAME = REs.qhdrname

        # Character constant (L|U|u|u8)?'cchar*'.  # Terminating ' required.
        cls.t_CPP_CHAR = REs.char

        # Same, within a CONTROL expression.  yacc evaluates this differently.
        cls.t_CONTROL_CPP_EXPRCHAR = REs.char

        # Block comment (C), possibly spanning multiple lines.  
        cls.t_CPP_COMMENT1 = r'(/\*(.|\n)*?\*/)'

        # Line comment (C++).  PCCP accepts them in C files also.  
        cls.t_CPP_COMMENT2 = r'(//[^\n]*)'
    
        cls.puncs()

        def error(self, t: PpTok, msg: str, keep_type: bool = False) -> PpTok:
            if not keep_type:
                t.type = TokType.error
            if t.lexer.owner.errors:
                t.lexer.prep.on_error_token(t, msg)
            return t

        @cls.funcrule(None)
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
            return cls.error(t, message)

    @classmethod
    def error(cls, t: PpTok, msg: str, keep_type: bool = False,
              warn: bool = False) -> PpTok:
        owner: RawLexer = t.lexer.owner
        if not keep_type:
            t.type = owner.TokType.error
        if owner.errors:
            owner.prep.on_error_token(t, msg, warn=warn)
        return t

    @classmethod
    def create(cls, prep: Preprocessor) -> PpLex:
        cls.prep = prep
        cls.REs = REs = RegExes(prep)
        # Set up the necessary class attributes.

        cls.rules()

        # create the TokType enumeration.
        TokType = make_tok_type(prep, REs, cls.punct_values)

        prep.TokType = TokType

        # We need to have the lextab module name be specific to the same
        # parameters that govern the content of the lexer, i.e.,
        # 
        # prep.cplus_ver selects C or C++ as the language, standard version
        # doesn't matter.  C++ enables the extra punctuators
        #
        # prep.emulate includes ` @ and $ as literals..
        #
        # prep.gnu enables R-strings for all languages.
        lextab = f"""\
            lextab\
            {'-c -cplusplus'.split()[bool(prep.cplus_ver)]}\
            {'-gcc' * bool(prep.emulate)}\
            {'-gnu' * bool(prep.gnu)}\
            """
        lextab = lextab.replace(' ', '')

        fact = cls()
        lexer = lex.lex(module=fact, lextab=lextab)
        #lexer = PpLex(prep, lex.lex(lextab=lextab))
        lexer.prep = prep
        lexer.TokType = TokType
        
        lexer.REs = REs
        return PpLex(from_lexer=lexer)


