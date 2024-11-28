#!/usr/bin/python
# Python C99 conforming preprocessor expression evaluator
# (C) 2019-2020 Niall Douglas http://www.nedproductions.biz/
# Started: Apr 2019
# Updated: Jun 2024 by Michael Rolle

from __future__ import annotations
from __future__ import generators, print_function, absolute_import, division

import sys, os, re, codecs, copy
import operator

if __name__ == '__main__' and __package__ is None:
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from pcpp.common import *
from pcpp.directive import OutputDirective
from pcpp.lexer import default_lexer
from pcpp.parser import yacc
from pcpp.tokens import Tokens, TokIter

# The width of signed integer which this evaluator will use
INTMAXBITS = 64

# Some Python 3 compatibility shims
INTBASETYPE = int


# Precompile the regular expression for correctly expanding unicode escape
# sequences.
_expand_escape_sequences_pat = re.compile(r'''
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
)''', re.UNICODE | re.VERBOSE)

class Value(int):
    """A signed or unsigned integer within a preprocessor expression, bounded
    to within INT_MIN and INT_MAX, or 0 and UINT_MAX. Signed overflow is
    handled like a two's complement CPU, despite being UB, as that's what GCC
    and clang do.
    
    >>> Value(5)
    Value(5)
    >>> Value('5L')
    Value(5)
    >>> Value('5U')
    Value(5U)
    >>> Value('0')
    Value(0)
    >>> Value('0U')
    Value(0U)
    >>> Value('-1U')
    Value(18446744073709551615U)
    >>> Value(5) * Value(2)
    Value(10)
    >>> Value(5) + Value('2u')
    Value(7U)
    >>> Value(5) * 2
    Value(10)
    >>> Value(5) / 2   # Must return integer
    Value(2)
    >>> Value(50) % 8
    Value(2)
    >>> -Value(5)
    Value(-5)
    >>> +Value(-5)
    Value(-5)
    >>> ~Value(5)
    Value(-6)
    >>> Value(6) & 2
    Value(2)
    >>> Value(4) | 2
    Value(6)
    >>> Value(6) ^ 2
    Value(4)
    >>> Value(2) << 2
    Value(8)
    >>> Value(8) >> 2
    Value(2)
    >>> Value(9223372036854775808)
    Value(-9223372036854775808)
    >>> Value(-9223372036854775809)
    Value(9223372036854775807)
    >>> Value(18446744073709551615)
    Value(-1)
    >>> Value(False)
    Value(0)
    >>> Value(True)
    Value(1)
    >>> Value(5) == Value(6)
    Value(0)
    >>> Value(5) == Value(5)
    Value(1)
    >>> not Value(2)
    Traceback (most recent call last):
    ...
    AssertionError
    >>> Value(4) and Value(2)
    Traceback (most recent call last):
    ...
    AssertionError
    >>> Value(5) and not Value(6)
    Traceback (most recent call last):
    ...
    AssertionError
    >>> Value('0x3f')
    Value(63)
    >>> Value('077')
    Value(63)
    >>> Value("'N'")
    Value(78)
    >>> Value("L'N'")
    Value(78)
    >>> Value("'\\n'")
    Value(10)
    >>> Value("'\\\\n'")
    Value(10)
    >>> Value("'\\\\'")
    Value(92)
    >>> Value("'\\'")
    Traceback (most recent call last):
    ...
    SyntaxError: Empty character escape sequence
    """
    INT_MIN = -(1 << (INTMAXBITS - 1))
    INT_MAX = (1 << (INTMAXBITS - 1)) - 1
    INT_MASK = (1 << INTMAXBITS) - 1
    UINT_MIN = 0
    UINT_MAX = (1 << INTMAXBITS) - 1
    @staticmethod
    def _clamp(value: int, unsigned: bool,
                MIN = INT_MIN, MASK = INT_MASK, UMASK = UINT_MAX
                ) -> int:
        """ Limit value to maximum signed/unsigned. """
        return value & UMASK if unsigned else ((value - MIN) & MASK) + MIN

    @classmethod
    def __sclamp(cls, value) -> int:
        value = int(value)
        return ((value - cls.INT_MIN) & cls.INT_MASK) + cls.INT_MIN
    @classmethod
    def __uclamp(cls, value) -> int:
        value = int(value)
        return value & cls.UINT_MAX
    def __new__(cls, value, unsigned = False, exception = None):
        if isinstance(value, Value):
            unsigned = value.unsigned
            exception = value.exception
        elif isinstance(value, int):
            value = cls._clamp(value, unsigned)
        elif isinstance(value, str):
            # This is a character constant: [L|u|U|u8]'...'.
            startidx = (2 if value.startswith(("L'", "u'", "U'"))
                       else 3 if value.startswith("u8'")
                       else 1)
            value = value[startidx:-1]
            if len(value) == 0:
                raise SyntaxError('Empty character escape sequence')
            value = _expand_escape_sequences_pat.sub(
                lambda x: codecs.decode(x.group(0), 'unicode-escape'), value)
            x = int(ord(value))
            value = cls.__uclamp(x) if unsigned else cls.__sclamp(x)
        else:
            print('Unknown value type: %s' % repr(type(value)),
                  file = sys.stderr)
            assert False  # Input is an unrecognised type
        inst = super(Value, cls).__new__(cls, value)
        inst.unsigned = unsigned
        inst.exception = exception
        return inst
    def value(self):
        if self.exception is not None:
            raise self.exception
        return int(self)

    @classmethod
    def make_binops(cls, *ops: str, y_unsigned: bool = True) -> None:
        """ Create binary operation methods. """
        def make_method(name: str, func: function) -> None:
            def binop(x: Value, y: Value) -> int:
                if x.exception: return x
                if y.exception: return y
                return Value(func(int(x), int(y)),
                             x.unsigned or (y_unsigned and y.unsigned))
            setattr(cls, name, binop)

        for op in ops:
            make_method(op, getattr(operator, op))

    def __neg__(self):
        if self.exception is not None:
            return self
        return Value(- int(self), self.unsigned)
    def __invert__(self):
        if self.exception is not None:
            return self
        return Value(~ int(self), self.unsigned)
    def __lshift__(self, other):
        return (self.either_exception(other)
                or Value(int(self) << other, self.unsigned or other.unsigned)
                )
    def __rshift__(self, other):
        return (self.either_exception(other)
                or Value(int(self) >> other, self.unsigned or other.unsigned)
                )
    def __repr__(self):
        if self.exception is not None:
            return "Exception(%s)" % repr(self.exception)
        elif self.unsigned:
            return "Value(%dU)" % int(self)
        else:
            return "Value(%d)" % int(self)

        if self.exception: return self

Value.make_binops(
    *'__add__ __sub__ __mul__ __floordiv__ __mod__'.split(),
    *'__and__ __or__ __xor__'.split(),
    *'__lt__ __le__ __eq__ __ne__ __ge__ __gt__'.split(),
    )
Value.make_binops(
    *'__lshift__ __rshift__'.split(), y_unsigned=False,
    )

class Evaluator:
    """Evaluator of #if C preprocessor expressions.
    
    >>> e = Evaluator()
    >>> e('5')
    Value(5)
    >>> e('5+6')
    Value(11)
    >>> e('5+6*2')
    Value(17)
    >>> e('5/2+6*2')
    Value(14)
    >>> e('5 < 6 <= 7')
    Value(1)
    >>> e('5 < 6 && 8 > 7')
    Value(1)
    >>> e('18446744073709551615 == -1')
    Value(1)
    >>> e('-9223372036854775809 == 9223372036854775807')
    Value(1)
    >>> e('-1 < 0U')
    Value(0U)
    >>> e('(( 0L && 0) || (!0L && !0 ))')
    Value(1)
    >>> e('(1)?2:3')
    Value(2)
    >>> e('(1 ? -1 : 0) <= 0')
    Value(1)
    >>> e('(1 ? -1 : 0U)')       # Output type of ? must be common between both choices
    Value(18446744073709551615U)
    >>> e('(1 ? -1 : 0U) <= 0')
    Value(0U)
    >>> e('1 && 10 / 0')         # doctest: +ELLIPSIS
    Exception(ZeroDivisionError(...
    >>> e('0 && 10 / 0')         # && must shortcut
    Value(0)
    >>> e('1 ? 10 / 0 : 0')      # doctest: +ELLIPSIS
    Exception(ZeroDivisionError(...
    >>> e('0 ? 10 / 0 : 0')      # ? must shortcut
    Value(0)
    >>> e('(3 ^ 5) != 6 || (3 | 5) != 7 || (3 & 5) != 1')
    Value(0)
    >>> e('1 << 2 != 4 || 8 >> 1 != 4')
    Value(0)
    >>> e('(2 || 3) != 1 || (2 && 3) != 1 || (0 || 4) != 1 || (0 && 5) != 0')
    Value(0)
    >>> e('-1 << 3U > 0')
    Value(0)
    >>> e("'N' == 78")
    Value(1)
    >>> e('0x3f == 63')
    Value(1)
    >>> e("'\\\\n'")
    Value(10)
    >>> e("'\\\\\\\\'")
    Value(92)
    >>> e("'\\\\n' == 0xA")
    Value(1)
    >>> e("'\\\\\\\\' == 0x5c")
    Value(1)
    >>> e("L'\\\\0' == 0")
    Value(1)
    >>> e('12 == 12')
    Value(1)
    >>> e('12L == 12')
    Value(1)
    >>> e('-1 >= 0U')
    Value(1U)
    >>> e('(1<<2) == 4')
    Value(1)
    >>> e('(-!+!9) == -1')
    Value(1)
    >>> e('(2 || 3) == 1')
    Value(1)
    >>> e('1L * 3 != 3')
    Value(0)
    >>> e('(!1L != 0) || (-1L != -1)')
    Value(0)
    >>> e('0177777 == 65535')
    Value(1)
    >>> e('0Xffff != 65535 || 0XFfFf == 65535')
    Value(1)
    >>> e('0L != 0 || 0l != 0')
    Value(0)
    >>> e('1U != 1 || 1u == 1')
    Value(1)
    >>> e('0 <= -1')
    Value(0)
    >>> e('1 << 2 != 4 || 8 >> 1 == 4')
    Value(1)
    >>> e('(3 ^ 5) == 6')
    Value(1)
    >>> e('(3 | 5) == 7')
    Value(1)
    >>> e('(3 & 5) == 1')
    Value(1)
    >>> e('(3 ^ 5) != 6 || (3 | 5) != 7 || (3 & 5) != 1')
    Value(0)
    >>> e('(0 ? 1 : 2) != 2')
    Value(0)
    >>> e('-1 << 3U > 0')
    Value(0)
    >>> e('0 && 10 / 0')
    Value(0)
    >>> e('not_defined && 10 / not_defined')  # doctest: +ELLIPSIS
    Exception(SyntaxError('Unknown identifier not_defined'...
    >>> e('0 && 10 / 0 > 1')
    Value(0)
    >>> e('(0) ? 10 / 0 : 0')
    Value(0)
    >>> e('0 == 0 || 10 / 0 > 1')
    Value(1)
    >>> e('(15 >> 2 >> 1 != 1) || (3 << 2 << 1 != 24)')
    Value(0)
    >>> e('(1 | 2) == 3 && 4 != 5 || 0')
    Value(1)
    >>> e('1  >  0')
    Value(1)
    >>> e("'\123' != 83")
    Value(0)
    >>> e("'\x1b' != '\033'")
    Value(0)
    >>> e('0 + (1 - (2 + (3 - (4 + (5 - (6 + (7 - (8 + (9 - (10 + (11 - (12 +
            (13 - (14 + (15 - (16 + (17 - (18 + (19 - (20 + (21 - (22 + (23 -
            (24 + (25 - (26 + (27 - (28 + (29 - (30 + (31 - (32 + 0)
            ))))))))))))))))))))))))))))))) == 0')
    Value(1)
    >>> e('test_function(X)', functions={'test_function':lambda x: 55})
    Value(55)
    >>> e('test_identifier', identifiers={'test_identifier':11})
    Value(11)
    >>> e('defined(X)', functions={'defined':lambda x: 55})
    Value(55)
    >>> e('defined(X)')  # doctest: +ELLIPSIS
    Exception(SyntaxError('Unknown function defined'...
    >>> e('__has_include("variant")')  # doctest: +ELLIPSIS
    Exception(SyntaxError('Unknown function __has_include'...
    >>> e('__has_include(<variant>)')  # doctest: +ELLIPSIS
    Exception(SyntaxError('Unknown function __has_include'...
    >>> e('5  // comment')
    Value(5)
    >>> e('5  /* comment */')
    Value(5)
    >>> e('5  /* comment // more */')
    Value(5)
    >>> e('5  // /* comment */')
    Value(5)
    >>> e('defined X', functions={'defined':lambda x: 55})
    Value(55)
    """

    def __init__(self, prep: Preprocessor):
        self.lexer = prep.lexer if prep.lexer is not None else default_lexer()
        self.parser = yacc.yacc(module=EvalParser(prep),
                                optimize=in_production,
                                debug=not in_production,
                                write_tables=not in_production)
        self.eval = EvalExpr(prep, self)

    class __lexer(object):
        """ Lexer to provide tokens to yacc.parse. """

        def __init__(self, functions, identifiers):
            self.__toks = []
            self.__functions = functions
            self.__identifiers = identifiers

        def input(self, toks: Iterable[lexToken]):
            """ Supply iterable (which is consumed) of tokens for the
            expression.
            """
            self.toks = iter(toks)

        def token(self):
            return next(self.toks, None)

        def on_function_call(self, p):
            if p[1] not in self.__functions:
                raise SyntaxError('Unknown function %s' % p[1])
            p[0] = Value(self.__functions[p[1]](p[3]))

        def on_identifier(self, p):
            if p[1] not in self.__identifiers:
                raise SyntaxError('Unknown identifier %s' % p[1])
            p[0] = Value(self.__identifiers[p[1]])
            
    def __call__(self, input: Iterable[PpTok],
                 functions = {}, identifiers = {}):
        """
        Execute a fully macro expanded set of tokens representing an
        expression, returning the result of the evaluation.
        """
        if not isinstance(input, (list, collections.abc.MutableSequence)):
            self.lexer.input(input)
            input = []
            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                input.append(tok)

        input = Tokens(filter(operator.attrgetter('value'), input))
        return self.parser.parse(
            input, lexer = self.__lexer(functions, identifiers))


class EvalParser:
    """ This is the module argument for creating the yacc parser and the
    expression evaluator.  self.parser is the yacc parser.  self.eval is the
    evaluator (see EvalExpr).
    """

    def __init__(self, prep: Preprocessor):
        self.prep = prep
        self.yacc = yacc.yacc(module=self,
            optimize=in_production, tabmodule='ctrlexpr',
            #optimize=in_production, tabmodule='ctrlexpr',
            debug=not in_production, write_tables=not in_production)


    # PLY yacc specification
    # Valid C preprocessor expression items:
    #   - Integer constants
    #   - Character constants
    #   - Addition, subtraction, multiplication, division, bitwise and-or-xor,
    #     shifts, comparisons, logical and-or-not
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
       'CPP_ID', 'CPP_INTEGER', 'CPP_EXPRCHAR', 'CPP_CHAR', 'CPP_STRING',
       'CPP_PLUS', 'CPP_MINUS', 'CPP_STAR', 'CPP_FSLASH', 'CPP_PERCENT',
       'CPP_BAR', 'CPP_AMPERSAND', 'CPP_TILDE', 'CPP_HAT', 'CPP_LESS',
       'CPP_GREATER', 'CPP_EXCLAMATION', 'CPP_QUESTION',
       'CPP_LPAREN', 'CPP_RPAREN',
       'CPP_COMMA', 'CPP_COLON', 'CPP_LSHIFT', 'CPP_LESSEQUAL', 'CPP_RSHIFT',
       'CPP_GREATEREQUAL', 'CPP_LOGICALOR', 'CPP_LOGICALAND', 'CPP_EQUALITY',
       'CPP_INEQUALITY'
       )

    precedence = (
        ('left', 'CPP_COMMA'),                                          # 15
                                                                        # 14
        ('left', 'CPP_QUESTION', 'CPP_COLON'),                          # 13
        ('left', 'CPP_LOGICALOR'),                                      # 12
        ('left', 'CPP_LOGICALAND'),                                     # 11
        ('left', 'CPP_BAR'),                                            # 10
        ('left', 'CPP_HAT'),                                            # 9
        ('left', 'CPP_AMPERSAND'),                                      # 8
        ('left', 'CPP_EQUALITY', 'CPP_INEQUALITY'),                     # 7
        ('left', 'CPP_LESS', 'CPP_LESSEQUAL', 'CPP_GREATER',
         'CPP_GREATEREQUAL'),                                           # 6
        ('left', 'CPP_LSHIFT', 'CPP_RSHIFT'),                           # 5
        ('left', 'CPP_PLUS', 'CPP_MINUS'),                              # 4
        ('left', 'CPP_STAR', 'CPP_FSLASH', 'CPP_PERCENT'),              # 3
        ('right', 'UPLUS', 'UMINUS', 'CPP_EXCLAMATION', 'CPP_TILDE'),   # 2
                                                                        # 1
    )

    def p_error(self, p: YaccProduction):
        if p:
            raise SyntaxError("around token '%s' type %s" % (p.value, p.type))
        else:
            raise SyntaxError("at EOF")

    def p_expression_number(self, p: YaccProduction):
        'expression : CPP_INTEGER'
        # Evaluate the token's numeric value and unsignedness.
        m: re.Match = p.slice[1].type.patt.match(p[1])
        num = m['num']  # the spelling of the token without the suffix
        # Octal constant needs '0o' in front of it.
        if m['oct']: num = '0o' + num
        if "'" in num:
            # Remove the digit separators.
            num = num.replace("'", "")
        numval = int(num, base=0)
        # Check for unsigned.
        sfx = m['sfx']
        unsigned = sfx and 'u' in sfx.lower()
        #if m['hex']
        p[0] = Value(numval, unsigned=unsigned)

    def p_expression_character(
            self, p: YaccProduction, *,
            _max = dict(u=0xffff, u8=0xff, U=0xffffffff, L=1<<64-1, dflt=0xff),
            ):
        '''expression : CPP_EXPRCHAR
                        | CPP_CHAR
                        '''
        # Process the token's string value and char size.
        tok: PpTok = p.slice[1]
        m: re.Match = tok.match
        val = m.group('val')
        pfx = m.group('pfx')
        # GCC with c++14 doesn't recognize the u8 prefix.

        maxnum = pfx and _max[pfx] or 0xffffffff
        text = codecs.decode(val, 'unicode-escape')

        if not text:
            self.prep.on_error_token(
                tok,
                f'Empty character constant in control expression: {text!r}.'
                )
            if self.prep.clang:
                p[0] = Value(0, exception=SyntaxError())
                return
            num = 0
        elif len(text) != 1 and not pfx:
            msg = (f'Multi-character char constant in control expression: '
                   f'{text!r}.')
            self.prep.on_warn_token(tok, msg)
            num = 0
            # Plain '...' assembles up to 4 trailing bytes from the value.
            for c in text[-4:]:
                num = (num << 8) + ord(c)
        elif len(text) != 1 and pfx:
            msg = (f'Multi-character char constant in control expression: '
                   f'{pfx}{text!r}.')
            self.prep.on_error_token(tok, msg)
            if self.prep.clang:
                p[0] = Value(0, exception=SyntaxError(msg))
                return
            # Sized '...' uses only the last character.
            num = ord(text[-1])
        else:
            num = int(ord(text))
        # Possible truncation.
        if num > maxnum:
            num &= maxnum
            if len(text) == 1:
                self.prep.on_error_token(
                    tok,
                    f'Character constant too long for its type '
                    f'n control expression: {text!r}.'
                    )

        p[0] = Value(num)

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
        p[0] = +Value(p[2])

    def p_expression_uminus(self, p: YaccProduction):
        'expression : CPP_MINUS expression %prec UMINUS'
        p[0] = -Value(p[2])

    def p_expression_unop(self, p: YaccProduction):
        """
        expression : CPP_EXCLAMATION expression
                  | CPP_TILDE expression
        """
        try:
            if p[1] == '!':
                p[0] = Value(0) if (Value(p[2]).value()!=0) else Value(1)
            elif p[1] == '~':
                p[0] = ~Value(p[2])
        except Exception as e:
            p[0] = Value(0, exception = e)

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
                p[0] = Value(p[1]) * Value(p[3])
            elif p[2] == '/':
                p[0] = Value(p[1]) / Value(p[3])
            elif p[2] == '%':
                p[0] = Value(p[1]) % Value(p[3])
            elif p[2] == '+':
                p[0] = Value(p[1]) + Value(p[3])
            elif p[2] == '-':
                p[0] = Value(p[1]) - Value(p[3])
            elif p[2] == '<<':
                p[0] = Value(p[1]) << Value(p[3])
            elif p[2] == '>>':
                p[0] = Value(p[1]) >> Value(p[3])
            elif p[2] == '<':
                p[0] = Value(p[1]) < Value(p[3])
            elif p[2] == '<=':
                p[0] = Value(p[1]) <= Value(p[3])
            elif p[2] == '>':
                p[0] = Value(p[1]) > Value(p[3])
            elif p[2] == '>=':
                p[0] = Value(p[1]) >= Value(p[3])
            elif p[2] == '==':
                p[0] = Value(p[1]) == Value(p[3])
            elif p[2] == '!=':
                p[0] = Value(p[1]) != Value(p[3])
            elif p[2] == '&':
                p[0] = Value(p[1]) & Value(p[3])
            elif p[2] == '^':
                p[0] = Value(p[1]) ^ Value(p[3])
            elif p[2] == '|':
                p[0] = Value(p[1]) | Value(p[3])
            elif p[2] == '&&':
                p[0] = (Value(1)
                        if (Value(p[1]).value()!=0 and Value(p[3]).value()!=0)
                        else Value(0))
            elif p[2] == '||':
                p[0] = (Value(1)
                        if (Value(p[1]).value()!=0 or Value(p[3]).value()!=0)
                        else Value(0))
            elif p[2] == ',':
                p[0] = Value(p[3])
        except Exception as e:
            p[0] = Value(0, exception = e)

    def p_expression_conditional(self, p: YaccProduction):
        'expression : expression CPP_QUESTION expression CPP_COLON expression'
        try:
            # Output type must cast up to unsigned if either input is unsigned.
            p[0] = Value(p[3]) if (Value(p[1]).value()!=0) else Value(p[5])
            try:
                p[0] = Value(p[0].value(), unsigned = Value(p[3]).unsigned or Value(p[5]).unsigned)
            except:
                pass
        except Exception as e:
            p[0] = Value(0, exception = e)

    def p_expression_function_call(self, p: YaccProduction):
        "expression : CPP_ID CPP_LPAREN expression CPP_RPAREN"
        try:
            p.lexer.on_function_call(p)
        except Exception as e:
            p[0] = Value(0, exception = e)

    def p_expression_identifier(self, p: YaccProduction):
        "expression : CPP_ID"
        try:
            p.lexer.on_identifier(p)
        except Exception as e:
            p[0] = Value(0, exception = e)

class EvalExpr:
    """ This is a callable object which will evaluate a control expression.
    """
    def __init__(self, prep: Preprocessor, evaluator: Evaluator):
        self.prep = prep
        self.evaluator = evaluator
        self.evalvars = self.IndirectToMacroHook(self)
        self.evalfuncts = self.IndirectToMacroFunctionHook(self)

    def __call__(self, intoks: Tokens, origin: PpTok
                 ) -> Tuple[bool, Tokens | None]:
        """ Evaluate expression in given tokens, taken from the directive.
        Result is either True or False.  If False, there may be some tokens to
        be passed through.  Can also raise OutputDirective.
        """

        self.origin = origin
        self.partial_expansion = False

        self.initer = TokIter(intoks)
        exptoks: TokIter = self.prep.macros.expand(self.initer, origin=origin)
        try:
            repltoks = Tokens(self.replacements(exptoks))
            # Call the yacc parser.
            result = self.evaluator(repltoks,
                                    functions = self.evalfuncts,
                                    identifiers = self.evalvars
                                    ).value()
        except OutputDirective:
            raise
        except Exception as e:
            if not self.partial_expansion:
                self.prep.on_error_token(
                    intoks[0],
                    f"Could not evaluate expression due to {e!r} "
                    f"(passed to evaluator: {str(intoks)!r})"
                    )
            result = False
        return (result, exptoks) if self.partial_expansion else (result, None)

    def replacements(self, exptoks: TokIter) -> Iterator[PpTok]:
        """
        Filter tokens after the normal macro expansion, handling undefined
        identifiers and defined-macro expressions.
        """
        for tok in exptoks:
            if not tok.type.id:
                yield tok
                continue
            # Token is an identifier, which was not replaced.  Either it is
            # not a macro, or expansion of the macro was disabled.
            result = 0
            if tok.value == 'defined':
                if tok.hide:
                    self.prep.on_warn_token(
                        self.origin,
                        "Macro expansion of control expression "
                        "contains 'defined'.  "
                        )
                name = self.defined_expr_name(exptoks)
                if not name:
                    # Malformed expression
                    raise self.DefinedMacroError(
                        "Malformed 'defined' in control expression"   
                        )
                elif name.value in self.prep.macros:
                    # Name defined, value = 1
                    result = 1

            yield tok.copy(value=self.prep.t_INTEGER_TYPE(result),
                           type=self.prep.t_INTEGER)


    def defined_expr_name(self, exptoks: TokIter) -> PpTok | None:
        """
        Get the identifier from defined-macro expression from tokens following
        the 'defined' token.  May raise DefinedMacroError.
        """
        try:
            self.initer.in_defined_expr = True
            tok : PpTok = next(exptoks, None)
            if not tok:
                return None
            if tok.type.id:
                # Simple identifier.
                return tok
            if not tok or tok.value != '(':
                return None
            tok : PpTok = next(exptoks, None)
            if not tok or not tok.type.id:
                return None
            tok2 = next(exptoks, None)
            if not tok2 or tok2.value != ')':
                return None
            return tok
        finally:
            self.initer.in_defined_expr = False

    class DefinedMacroError(Exception):
        pass

    class IndirectToMacroHook(object):
        def __init__(self, eval: EvalExpr):
            self.eval = eval
        def __contains__(self, key):
            return True
        def __getitem__(self, key):
            if key.startswith('defined('):
                self.partial_expansion = True
                return 0
            repl = self.prep.on_unknown_macro_in_expr(key)
            if repl is None:
                self.eval.partial_expansion = True
                return key
            return repl


    class IndirectToMacroFunctionHook(object):
        def __init__(self, eval: EvalExpr):
            self.eval = eval
        def __contains__(self, key):
            return True
        def __getitem__(self, key):
            repl = self.prep.on_unknown_macro_function_in_expr(key)
            if repl is None:
                self.eval.partial_expansion = True
                return key
            return repl


if __name__ == "__main__":
    import doctest
    doctest.testmod()

