#!/usr/bin/env python3.8
# @generated by pegen from test_rec.gram

from __future__ import annotations

import ast
import sys
import tokenize

from typing import Any, Optional, Callable, cast

from pegen.parser import (
    memoize, memoize_left_rec, logger, Parser, ParseResult
    )
# Keywords and soft keywords are listed at the end of the parser definition.
class GeneratedParser(Parser):

    # start(): target '='
    def start(self) -> ParseResult[Any]:
        def _rhs() -> ParseResult[Any]:

            # target '='
            def _alt() -> ParseResult[Any]:

                # target
                def _item__target() -> ParseResult[Any]:
                    return self.target()
                _target: Any; _result__target: ParseResult[Any]
                _result__target = _item__target()
                if not _result__target: return None
                _target, = _result__target

                # '='
                def _item__literal() -> ParseResult[Token]:
                    return self._expect_type(22)   # token = "="
                _literal: Token; _result__literal: ParseResult[Token]
                _result__literal = _item__literal()
                if not _result__literal: return None
                _literal, = _result__literal

                # parse succeeded
                return [_target, _literal],

            _alts = [
                _alt,
            ]
            return self._alts(_alts)
        return self._rule(_rhs)

    # target(): maybe '+' | NAME
    # Left-recursive
    @logger
    def target(self) -> ParseResult[Any]:
        def _rhs() -> ParseResult[Any]:

            # maybe '+'
            def _alt1() -> ParseResult[Any]:

                # maybe
                def _item__maybe() -> ParseResult[Any]:
                    return self.maybe()
                _maybe: Any; _result__maybe: ParseResult[Any]
                _result__maybe = _item__maybe()
                if not _result__maybe: return None
                _maybe, = _result__maybe

                # '+'
                def _item__literal() -> ParseResult[Token]:
                    return self._expect_type(14)   # token = "+"
                _literal: Token; _result__literal: ParseResult[Token]
                _result__literal = _item__literal()
                if not _result__literal: return None
                _literal, = _result__literal

                # parse succeeded
                return [_maybe, _literal],

            # NAME
            def _alt2() -> ParseResult[Any]:

                # NAME
                def _item__NAME() -> ParseResult[Token]:
                    return self._name()
                _NAME: Token; _result__NAME: ParseResult[Token]
                _result__NAME = _item__NAME()
                if not _result__NAME: return None
                _NAME, = _result__NAME

                # parse succeeded
                return _NAME,

            _alts = [
                _alt1,
                _alt2,
            ]
            return self._alts(_alts)
        return self._rule(_rhs)

    # maybe(): maybe '-' | target
    # Left-recursive leader
    @memoize_left_rec
    def maybe(self) -> ParseResult[Any]:
        def _rhs() -> ParseResult[Any]:

            # maybe '-'
            def _alt1() -> ParseResult[Any]:

                # maybe
                def _item__maybe() -> ParseResult[Any]:
                    return self.maybe()
                _maybe: Any; _result__maybe: ParseResult[Any]
                _result__maybe = _item__maybe()
                if not _result__maybe: return None
                _maybe, = _result__maybe

                # '-'
                def _item__literal() -> ParseResult[Token]:
                    return self._expect_type(15)   # token = "-"
                _literal: Token; _result__literal: ParseResult[Token]
                _result__literal = _item__literal()
                if not _result__literal: return None
                _literal, = _result__literal

                # parse succeeded
                return [_maybe, _literal],

            # target
            def _alt2() -> ParseResult[Any]:

                # target
                def _item__target() -> ParseResult[Any]:
                    return self.target()
                _target: Any; _result__target: ParseResult[Any]
                _result__target = _item__target()
                if not _result__target: return None
                _target, = _result__target

                # parse succeeded
                return _target,

            _alts = [
                _alt1,
                _alt2,
            ]
            return self._alts(_alts)
        return self._rule(_rhs)

    # prim(): (STRING) | (NAME) | (NUMBER)
    def prim(self) -> ParseResult[Any]:
        def _rhs() -> ParseResult[Any]:

            # (STRING)
            def _alt1() -> ParseResult[Any]:

                # (STRING)
                def _item__group() -> ParseResult[Any]:
                    def _rhs() -> ParseResult[Any]:

                        # STRING
                        def _alt() -> ParseResult[Any]:

                            # STRING
                            def _item__STRING() -> ParseResult[Token]:
                                return self._string()
                            _STRING: Token; _result__STRING: ParseResult[Token]
                            _result__STRING = _item__STRING()
                            if not _result__STRING: return None
                            _STRING, = _result__STRING

                            # parse succeeded
                            return "string",

                        _alts = [
                            _alt,
                        ]
                        return self._alts(_alts)
                    return _rhs()
                _group: Any; _result__group: ParseResult[Any]
                _result__group = _item__group()
                if not _result__group: return None
                _group, = _result__group

                # parse succeeded
                return _group,

            # (NAME)
            def _alt2() -> ParseResult[Any]:

                # (NAME)
                def _item__group() -> ParseResult[Any]:
                    def _rhs() -> ParseResult[Any]:

                        # NAME
                        def _alt() -> ParseResult[Any]:

                            # NAME
                            def _item__NAME() -> ParseResult[Token]:
                                return self._name()
                            _NAME: Token; _result__NAME: ParseResult[Token]
                            _result__NAME = _item__NAME()
                            if not _result__NAME: return None
                            _NAME, = _result__NAME

                            # parse succeeded
                            return "name",

                        _alts = [
                            _alt,
                        ]
                        return self._alts(_alts)
                    return _rhs()
                _group: Any; _result__group: ParseResult[Any]
                _result__group = _item__group()
                if not _result__group: return None
                _group, = _result__group

                # parse succeeded
                return _group,

            # (NUMBER)
            def _alt3() -> ParseResult[Any]:

                # (NUMBER)
                def _item__group() -> ParseResult[Any]:
                    def _rhs() -> ParseResult[Any]:

                        # NUMBER
                        def _alt() -> ParseResult[Any]:

                            # NUMBER
                            def _item__NUMBER() -> ParseResult[Token]:
                                return self._number()
                            _NUMBER: Token; _result__NUMBER: ParseResult[Token]
                            _result__NUMBER = _item__NUMBER()
                            if not _result__NUMBER: return None
                            _NUMBER, = _result__NUMBER

                            # parse succeeded
                            return _NUMBER,

                        _alts = [
                            _alt,
                        ]
                        return self._alts(_alts)
                    return _rhs()
                _group: Any; _result__group: ParseResult[Any]
                _result__group = _item__group()
                if not _result__group: return None
                _group, = _result__group

                # parse succeeded
                return _group,

            _alts = [
                _alt1,
                _alt2,
                _alt3,
            ]
            return self._alts(_alts)
        return self._rule(_rhs)

    KEYWORDS = ()
    SOFT_KEYWORDS = ()


if __name__ == '__main__':
    from pegen.parser import simple_parser_main
    simple_parser_main(GeneratedParser)