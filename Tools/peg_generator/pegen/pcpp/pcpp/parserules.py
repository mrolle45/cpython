""" parserules.py.
Module containing various yacc parser definitions.
Each is contained in, and built by, a class called *Parser.
"""
from __future__ import annotations

from pcpp.parser import in_production
from pcpp.lexer import default_lexer

from ply import lex, yacc

class EvalParser:
    def __init__(self, prep: Preprocessor):
        self.prep = prep
        self.parser = yacc.yacc(module=self,
            optimize=in_production, tabmodule='parsetabs.ctrlexpr',
            debug=not in_production, write_tables=not in_production)

    # PLY yacc specification
    # Valid C preprocessor expression items:
    #   - Integer constants
    #   - Character constants
    #   - Addition, subtraction, multiplication, division, bitwise and-or-xor, shifts,
    #     comparisons, logical and-or-not
    #   - defined()
    #
    # The C preprocessor does not support:
    #   - assignment
    #   - increment and decrement
    #   - array indexing, indirection
    #   - casting
    #   - sizeof, alignof

    # The subset of tokens from Preprocessor used in preprocessor expressions
    tokens = (
       'CPP_ID', 'CPP_INTEGER', 'CPP_CHAR', 'CPP_STRING',
       'CPP_PLUS', 'CPP_MINUS', 'CPP_STAR', 'CPP_FSLASH', 'CPP_PERCENT', 'CPP_BAR',
       'CPP_AMPERSAND', 'CPP_TILDE', 'CPP_HAT', 'CPP_LESS', 'CPP_GREATER', 'CPP_EXCLAMATION',
       'CPP_QUESTION', 'CPP_LPAREN', 'CPP_RPAREN',
       'CPP_COMMA', 'CPP_COLON',

       'CPP_LSHIFT', 'CPP_LESSEQUAL', 'CPP_RSHIFT',
       'CPP_GREATEREQUAL', 'CPP_LOGICALOR', 'CPP_LOGICALAND', 'CPP_EQUALITY',
       'CPP_INEQUALITY'
       )
    # 'CPP_WS', 'CPP_EQUAL',  'CPP_BSLASH', 'CPP_SQUOTE',


    precedence = (
        ('left', 'CPP_COMMA'),                                                     # 15
                                                                                   # 14 (assignments, unused)
        ('left', 'CPP_QUESTION', 'CPP_COLON'),                                     # 13
        ('left', 'CPP_LOGICALOR'),                                                 # 12
        ('left', 'CPP_LOGICALAND'),                                                # 11
        ('left', 'CPP_BAR'),                                                       # 10
        ('left', 'CPP_HAT'),                                                       # 9
        ('left', 'CPP_AMPERSAND'),                                                 # 8
        ('left', 'CPP_EQUALITY', 'CPP_INEQUALITY'),                                # 7
        ('left', 'CPP_LESS', 'CPP_LESSEQUAL', 'CPP_GREATER', 'CPP_GREATEREQUAL'),  # 6
        ('left', 'CPP_LSHIFT', 'CPP_RSHIFT'),                                      # 5
        ('left', 'CPP_PLUS', 'CPP_MINUS'),                                         # 4
        ('left', 'CPP_STAR', 'CPP_FSLASH', 'CPP_PERCENT'),                         # 3
        ('right', 'UPLUS', 'UMINUS', 'CPP_EXCLAMATION', 'CPP_TILDE'),              # 2
                                                                                   # 1 (unused in the C preprocessor)
    )

    def p_error(self, p: YaccProduction):
        if p:
            raise SyntaxError("around token '%s' type %s" % (p.value, p.type))
        else:
            raise SyntaxError("at EOF")

    def p_expression_number(self, p: YaccProduction):
        'expression : CPP_INTEGER'
        p[0] = EvalValue(p[1])

    def p_expression_character(self, p: YaccProduction):
        'expression : CPP_CHAR'
        p[0] = EvalValue(p[1])

    def p_expression_string(self, p: YaccProduction):
        """
        expression : CPP_STRING
                  | CPP_LESS expression CPP_GREATER
        """
        p[0] = p[1]

    def p_expression_group(self, p: YaccProduction):
        'expression : CPP_LPAREN expression CPP_RPAREN'
        p[0] = p[2]

    def p_expression_uplus(self, p: YaccProduction):
        'expression : CPP_PLUS expression %prec UPLUS'
        p[0] = +EvalValue(p[2])

    def p_expression_uminus(self, p: YaccProduction):
        'expression : CPP_MINUS expression %prec UMINUS'
        p[0] = -EvalValue(p[2])

    def p_expression_unop(self, p: YaccProduction):
        """
        expression : CPP_EXCLAMATION expression
                  | CPP_TILDE expression
        """
        try:
            if p[1] == '!':
                p[0] = EvalValue(0) if (EvalValue(p[2]).value()!=0) else EvalValue(1)
            elif p[1] == '~':
                p[0] = ~EvalValue(p[2])
        except Exception as e:
            p[0] = EvalValue(0, exception = e)

    def p_expression_binop(self, p: YaccProduction):
        """
        expression : expression CPP_STAR expression
                  | expression CPP_FSLASH expression
                  | expression CPP_PERCENT expression
                  | expression CPP_PLUS expression
                  | expression CPP_MINUS expression
                  | expression CPP_LSHIFT expression
                  | expression CPP_RSHIFT expression
                  | expression CPP_LESS expression
                  | expression CPP_LESSEQUAL expression
                  | expression CPP_GREATER expression
                  | expression CPP_GREATEREQUAL expression
                  | expression CPP_EQUALITY expression
                  | expression CPP_INEQUALITY expression
                  | expression CPP_AMPERSAND expression
                  | expression CPP_HAT expression
                  | expression CPP_BAR expression
                  | expression CPP_LOGICALAND expression
                  | expression CPP_LOGICALOR expression
                  | expression CPP_COMMA expression
        """
        # print [repr(p[i]) for i in range(0,4)]
        try:
            if p[2] == '*':
                p[0] = EvalValue(p[1]) * EvalValue(p[3])
            elif p[2] == '/':
                p[0] = EvalValue(p[1]) / EvalValue(p[3])
            elif p[2] == '%':
                p[0] = EvalValue(p[1]) % EvalValue(p[3])
            elif p[2] == '+':
                p[0] = EvalValue(p[1]) + EvalValue(p[3])
            elif p[2] == '-':
                p[0] = EvalValue(p[1]) - EvalValue(p[3])
            elif p[2] == '<<':
                p[0] = EvalValue(p[1]) << EvalValue(p[3])
            elif p[2] == '>>':
                p[0] = EvalValue(p[1]) >> EvalValue(p[3])
            elif p[2] == '<':
                p[0] = EvalValue(p[1]) < EvalValue(p[3])
            elif p[2] == '<=':
                p[0] = EvalValue(p[1]) <= EvalValue(p[3])
            elif p[2] == '>':
                p[0] = EvalValue(p[1]) > EvalValue(p[3])
            elif p[2] == '>=':
                p[0] = EvalValue(p[1]) >= EvalValue(p[3])
            elif p[2] == '==':
                p[0] = EvalValue(p[1]) == EvalValue(p[3])
            elif p[2] == '!=':
                p[0] = EvalValue(p[1]) != EvalValue(p[3])
            elif p[2] == '&':
                p[0] = EvalValue(p[1]) & EvalValue(p[3])
            elif p[2] == '^':
                p[0] = EvalValue(p[1]) ^ EvalValue(p[3])
            elif p[2] == '|':
                p[0] = EvalValue(p[1]) | EvalValue(p[3])
            elif p[2] == '&&':
                p[0] = EvalValue(1) if (EvalValue(p[1]).value()!=0 and EvalValue(p[3]).value()!=0) else EvalValue(0)
            elif p[2] == '||':
                p[0] = EvalValue(1) if (EvalValue(p[1]).value()!=0 or EvalValue(p[3]).value()!=0) else EvalValue(0)
            elif p[2] == ',':
                p[0] = EvalValue(p[3])
        except Exception as e:
            p[0] = EvalValue(0, exception = e)

    def p_expression_conditional(self, p: YaccProduction):
        'expression : expression CPP_QUESTION expression CPP_COLON expression'
        try:
            # Output type must cast up to unsigned if either input is unsigned
            p[0] = EvalValue(p[3]) if (EvalValue(p[1]).value()!=0) else EvalValue(p[5])
            try:
                p[0] = EvalValue(p[0].value(), unsigned = EvalValue(p[3]).unsigned or EvalValue(p[5]).unsigned)
            except:
                pass
        except Exception as e:
            p[0] = EvalValue(0, exception = e)

    def p_expression_function_call(self, p: YaccProduction):
        "expression : CPP_ID CPP_LPAREN expression CPP_RPAREN"
        try:
            p.lexer.on_function_call(p)
        except Exception as e:
            p[0] = EvalValue(0, exception = e)

    def p_expression_identifier(self, p: YaccProduction):
        "expression : CPP_ID"
        try:
            p.lexer.on_identifier(p)
        except Exception as e:
            p[0] = EvalValue(0, exception = e)

from evaluator import Value as EvalValue

class MacroArgsParser:
    """ Parses sequence of tokens representing the arguments in a macro call. """

    def __init__(self, prep: Preprocessor = None):
        self.prep = prep
        #self.lexer = lex.lex(module=self,
        #    optimize=in_production,
        #    debug=not in_production)
        self.lexer = self.ProxyLexer(default_lexer())

        self.parser = yacc.yacc(module=self,
            optimize=in_production, tabmodule='parsetabs.macroargs',
            debug=not in_production, write_tables=not in_production)

    tokens = ('CPP_LPAREN', 'CPP_RPAREN', 'CPP_COMMA', 'OTHER', )

    # args - top level balanced parentheses -> list of the args.
    def p_top(self, p: YaccProduction):
        """ args : CPP_LPAREN args CPP_RPAREN """
        p[0] = p[2]

    # args = Zero or more args separated by commas -> list of the args.
    def p_args(self, p: YaccProduction):
        """ args :
                | arg
                | arg CPP_COMMA args
        """
        p[0] = [*p[1:2], *(p[2:] and p[3])]

    # arg = zero or more primaries -> list of their tokens
    def p_arg(self, p: YaccProduction):
        """ arg :
                | primary
                | primary arg
        """
        prim = p[1:] and p[1]
        arg = p[2:] and p[2]
        p[0] = [*prim, *arg]

    # group - balanced parentheses, including commas.
    def p_group(self, p: YaccProduction):
        """ group : CPP_LPAREN groupcontents CPP_RPAREN """
        p[0] = [p.slice[1], *p[2], p.slice[3]]

    # group contents - zero or more group items
    def p_groupcontents(self, p: YaccProduction):
        """ groupcontents :
                | groupitem
                | groupitem groupcontents
        """
        groupitem = p[1:] and p[1]
        groupcontents = p[2:] and p[2]
        p[0] = [*groupitem, *groupcontents]

    # group item - primary.
    def p_groupitem(self, p: YaccProduction):
        """ groupitem : primary """
        p[0] = p[1]

    # group item - comma.
    def p_groupitemcomma(self, p: YaccProduction):
        """ groupitem :  CPP_COMMA """
        p[0] = [p.slice[1]]

    # Primary element - group
    def p_prim(self, p: YaccProduction):
        """ primary : group """
        p[0] = p[1]

    # Primary element - other token
    def p_other(self, p: YaccProduction):
        """ primary : OTHER """
        tok = p.slice[1]
        tok.type = tok.orig_type
        del tok.orig_type
        p[0] = [tok]

    class ProxyLexer:
        """ Filters another Lexer by altering certain tokens.
        Passes through '(', ',', and ')', changes the rest to type OTHER.
        """

        def __init__(self, lexer):
            self.lexer = lexer

        def input(self, data: str):
            self.lexer.input(data)

        def token(self) -> LexToken | none:
            tok = self.lexer.token()
            value = tok and tok.value
            if (value and value[0] not in '(,)'):
                tok.orig_type = tok.type
                tok.type = 'OTHER'
            return tok


#p = MacroArgsParser()

#def parse(s: str):
#    """ Tester to perform a parse with the MacroArgs parser. """
#    p.lexer.input(s)
#    result = p.parser.parse(s, p.lexer, debug=True)
#    for arg in result:
#        print (f"Arg =")
#        for tok in arg:
#            print (f"\t{tok!r}")

##parse('(ab , 42)')
#parse('(ab [42] (39, 40), def)')


