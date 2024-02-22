#!/usr/bin/python
# Python C99 conforming preprocessor expression evaluator
# (C) 2019-2020 Niall Douglas http://www.nedproductions.biz/
# Started: Apr 2019

from __future__ import annotations
from __future__ import generators, print_function, absolute_import, division

import sys, os, re, codecs, copy
if __name__ == '__main__' and __package__ is None:
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
from pcpp.parser import STRING_TYPES, yacc, in_production
from pcpp.lexer import TokType, default_lexer
from pcpp.parserules import EvalParser

# The width of signed integer which this evaluator will use
INTMAXBITS = 64

# Some Python 3 compatibility shims
# Some Python 3 compatibility shims
if sys.version_info.major < 3:
    INTBASETYPE = long
else:
    INTBASETYPE = int

# Precompile the regular expression for correctly expanding unicode escape
# sequences in Python 2 and 3. See https://stackoverflow.com/questions/4020539/process-escape-sequences-in-a-string-in-python
# for more information.
_expand_escape_sequences_pat = re.compile(r'''
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
)''', re.UNICODE | re.VERBOSE)

class Value(INTBASETYPE):
    """A signed or unsigned integer within a preprocessor expression, bounded
    to within INT_MIN and INT_MAX, or 0 and UINT_MAX. Signed overflow is handled
    like a two's complement CPU, despite being UB, as that's what GCC and clang do.
    
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
    @classmethod
    def __sclamp(cls, value):
        value = INTBASETYPE(value)
        return ((value - cls.INT_MIN) & cls.INT_MASK) + cls.INT_MIN
    @classmethod
    def __uclamp(cls, value):
        value = INTBASETYPE(value)
        return value & cls.UINT_MAX
    def __new__(cls, value, unsigned = False, exception = None):
        if isinstance(value, Value):
            unsigned = value.unsigned
            exception = value.exception
        elif isinstance(value, INTBASETYPE) or isinstance(value, int) or isinstance(value, float):
            value = cls.__uclamp(value) if unsigned else cls.__sclamp(value)
        elif isinstance(value, STRING_TYPES):
            if (value.startswith("L'") or value[0] == "'") and value[-1] == "'":
                startidx = 2 if value.startswith("L'") else 1
                #print("1. ***", value, file = sys.stderr)
                value = value[startidx:-1]
                if len(value) == 0:
                    raise SyntaxError('Empty character escape sequence')
                #print("2. ***", value, file = sys.stderr)
                value = _expand_escape_sequences_pat.sub(lambda x: codecs.decode(x.group(0), 'unicode-escape'), value)
                #print("3. ***", value, file = sys.stderr)
                x = INTBASETYPE(ord(value))
                #print("4. ***", x, file = sys.stderr)
            elif value.startswith('0x') or value.startswith('0X'):
                # Strip any terminators
                while not ((value[-1] >= '0' and value[-1] <= '9') or (value[-1] >= 'a' and value[-1] <= 'f') or (value[-1] >= 'A' and value[-1] <= 'F')):
                    if value[-1] == 'u' or value[-1] == 'U':
                        unsigned = True
                    value = value[:-1]
                x = INTBASETYPE(value, base = 16)
            elif value.startswith('0'):
                # Strip any terminators
                while not (value[-1] >= '0' and value[-1] <= '7'):
                    if value[-1] == 'u' or value[-1] == 'U':
                        unsigned = True
                    value = value[:-1]
                x = INTBASETYPE(value, base = 8)
            else:
                # Strip any terminators
                while not (value[-1] >= '0' and value[-1] <= '9'):
                    if value[-1] == 'u' or value[-1] == 'U':
                        unsigned = True
                    value = value[:-1]
                x = INTBASETYPE(value)
            value = cls.__uclamp(x) if unsigned else cls.__sclamp(x)
            #assert x == value
        else:
            print('Unknown value type: %s' % repr(type(value)), file = sys.stderr)
            assert False  # Input is an unrecognised type
        inst = super(Value, cls).__new__(cls, value)
        inst.unsigned = unsigned
        inst.exception = exception
        return inst
    def value(self):
        if self.exception is not None:
            raise self.exception
        return INTBASETYPE(self)
    def __add__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) + self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__add__(other))
    def __sub__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) - self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__sub__(other))
    def __mul__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) * self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__mul__(other))
    def __div__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) / self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__div__(other))
    def __truediv__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) / self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__truediv__(other))
    def __mod__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) % self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__mod__(other))
    def __neg__(self):
        if self.exception is not None:
            return self
        return Value(super(Value, self).__neg__(), self.unsigned)
    def __invert__(self):
        if self.exception is not None:
            return self
        return Value(super(Value, self).__invert__(), self.unsigned)
    def __and__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) & self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__and__(other))
    def __or__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) | self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__or__(other))
    def __pos__(self):
        if self.exception is not None:
            return self
        return Value(super(Value, self).__pos__())
    def __pow__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) ** self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__pow__(other))
    def __lshift__(self, other):
        if self.exception is not None:
            return self
        # Ignore other signedness
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) << self.__uclamp(other), True) if (self.unsigned) else Value(super(Value, self).__lshift__(other))
    def __rshift__(self, other):
        if self.exception is not None:
            return self
        # Ignore other signedness
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) >> self.__uclamp(other), True) if (self.unsigned) else Value(super(Value, self).__rshift__(other))
    def __xor__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) ^ self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(super(Value, self).__xor__(other))
    def __repr__(self):
        if self.exception is not None:
            return "Exception(%s)" % repr(self.exception)
        elif self.unsigned:
            return "Value(%dU)" % INTBASETYPE(self)
        else:
            return "Value(%d)" % INTBASETYPE(self)
    def __bool__(self):
        assert False  # Do not use Python logical operations
    def __nonzero__(self):
        assert False  # Do not use Python logical operations
    def __cmp__(self, other):
        assert False
    def __lt__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) < self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(self.__sclamp(self) < self.__sclamp(other), False)
    def __le__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) <= self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(self.__sclamp(self) <= self.__sclamp(other), False)
    def __eq__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) == self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(self.__sclamp(self) == self.__sclamp(other), False)
    def __ne__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) != self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(self.__sclamp(self) != self.__sclamp(other), False)
    def __ge__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) >= self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(self.__sclamp(self) >= self.__sclamp(other), False)
    def __gt__(self, other):
        if self.exception is not None:
            return self
        other = Value(other)
        if other.exception is not None:
            return other
        return Value(self.__uclamp(self) > self.__uclamp(other), True) if (self.unsigned or other.unsigned) else Value(self.__sclamp(self) > self.__sclamp(other), False)

        
class Evaluator(object):
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
    >>> e('0 + (1 - (2 + (3 - (4 + (5 - (6 + (7 - (8 + (9 - (10 + (11 - (12 +          (13 - (14 + (15 - (16 + (17 - (18 + (19 - (20 + (21 - (22 + (23 -           (24 + (25 - (26 + (27 - (28 + (29 - (30 + (31 - (32 + 0))))))))))           )))))))))))))))))))))) == 0')
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
    """
#    >>> e('defined X', functions={'defined':lambda x: 55})
#    Value(55)

    def __init__(self, lexer = None):
        self.lexer = lexer if lexer is not None else default_lexer()
        #self.parser = yacc.yacc(optimize=in_production, tabmodule='parsetabs.ctrlexpr', debug=not in_production,write_tables=not in_production)
        self.parser = EvalParser(self).parser

    class __lexer(object):
        """ Lexer to provide tokens to yacc.parse. """

        def __init__(self, functions, identifiers):
            self.__toks = []
            self.__functions = functions
            self.__identifiers = identifiers

        def input(self, toks: Iterable[lexToken]):
            """ Supply iterable (which is consumed) of tokens for the expression. """
            self.toks = iter(toks)

        def token(self):
            while True:
                try: tok = next(self.toks)
                except StopIteration: return None
                if tok.type.ws:
                    continue
                return tok

        def on_function_call(self, p):
            if p[1] not in self.__functions:
                raise SyntaxError('Unknown function %s' % p[1])
            p[0] = Value(self.__functions[p[1]](p[3]))

        def on_identifier(self, p):
            if p[1] not in self.__identifiers:
                raise SyntaxError('Unknown identifier %s' % p[1])
            p[0] = Value(self.__identifiers[p[1]])
            
    def __call__(self, input, functions = {}, identifiers = {}):
        """Execute a fully macro expanded set of tokens representing an expression,
        returning the result of the evaluation.
        """
        if not isinstance(input,list):
            self.lexer.input(input)
            input = []
            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                input.append(tok)

        return self.parser.parse(input, lexer = self.__lexer(functions, identifiers))


if __name__ == "__main__":
    import doctest
    doctest.testmod()

