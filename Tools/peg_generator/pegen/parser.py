from __future__ import annotations

import argparse
import sys
import dataclasses
import time
import token
import tokenize
import traceback
from abc import abstractmethod
from typing import (
    Any, Callable, ClassVar, Dict, Iterator, Literal, Optional, List, Tuple,
    Type, TypeVar, Union, cast
    )

from pegen.tokenizer import Mark, Tokenizer, exact_token_types, TokenRange
from pegen.grammar import Cut, Alt, OptVal

T = TypeVar("T")
P = TypeVar("P", bound="Parser")
F = TypeVar("F", bound=Callable[..., Any])

Token = tokenize.TokenInfo

ParseStatus = bool
ParseResult = Union[Tuple[T], None]
ParseFunc = Callable[[], ParseResult]


def logger(method: F) -> F:
    """For non-memoized functions that we want to be logged.
    In practice, this is only generated when the parser._verbose is true,
    and not for memoized or left-recursive leader functions (which do their own logging).
    """
    method_name = method.__name__

    def logger_wrapper(self: P, *args: object) -> T:
        if not self._verbose:
            return method(self, *args)
        argsr = ",".join(repr(arg) for arg in args)
        fill = "  " * self._level
        self._log(f"{method_name}({argsr}) .... (looking at {self._showpeek()})")
        self._level += 1
        tree = method(self, *args)
        self._level -= 1
        self._log(f"... {method_name}({argsr}) --> {tree!s:.200}")
        return tree

    logger_wrapper.__wrapped__ = method  # type: ignore
    logger_wrapper._nolog = True
    return cast(F, logger_wrapper)


def memoize(method: F) -> F:
    """Memoize a symbol method."""
    method_name = method.__name__

    def memoize_wrapper(self: P, *args: object) -> T:
        mark = self._mark()
        key = mark, method_name, args
        # Fast path: cache hit, and not verbose.
        if key in self._cache and not self._verbose:
            tree, endmark = self._cache[key]
            self._reset(endmark)
            return tree
        # Slow path: no cache hit, or verbose.
        verbose = self._verbose
        argsr = ",".join(repr(arg) for arg in args)
        fill = "  " * self._level
        if key not in self._cache:
            if verbose:
                self._log(f"{method_name}({argsr}) ... (looking at {self._showpeek()})")
            self._level += 1
            tree = method(self, *args)
            self._level -= 1
            if verbose:
                self._log(f"... {method_name}({argsr}) -> {tree!s:.200}")
            endmark = self._mark()
            self._cache[key] = tree, endmark
        else:
            tree, endmark = self._cache[key]
            if verbose:
                self._log(f"{method_name}({argsr}) -> {tree!s:.200}")
            self._reset(endmark)
        return tree

    memoize_wrapper.__wrapped__ = method  # type: ignore
    memoize_wrapper._nolog = True
    return cast(F, memoize_wrapper)


def memoize_left_rec(method: Callable[[P], Optional[T]]) -> Callable[[P], Optional[T]]:
    """Memoize a left-recursive symbol method."""
    method_name = method.__name__

    def memoize_left_rec_wrapper(self: P) -> Optional[T]:
        mark = self._mark()
        key = mark, method_name, ()
        # Fast path: cache hit, and not verbose.
        if key in self._cache and not self._verbose:
            tree, endmark = self._cache[key]
            self._reset(endmark)
            return tree
        # Slow path: no cache hit, or verbose.
        verbose = self._verbose
        fill = "  " * self._level
        if key not in self._cache:
            if verbose:
                self._log(f"{method_name} ... (looking at {self._showpeek()})")
            self._level += 1

            # For left-recursive rules we manipulate the cache and
            # loop until the rule shows no progress, then pick the
            # previous result.  For an explanation why this works, see
            # https://github.com/PhilippeSigaud/Pegged/wiki/Left-Recursion
            # (But we use the memoization cache instead of a static
            # variable; perhaps this is similar to a paper by Warth et al.
            # (http://web.cs.ucla.edu/~todd/research/pub.php?id=pepm08).

            # Prime the cache with a failure.
            self._cache[key] = None, mark
            lastresult, lastmark = None, mark
            depth = 0
            if verbose:
                self._log(f"Recursive {method_name} at {mark} depth {depth}")

            while True:
                self._reset(mark)
                self.in_recursive_rule += 1
                try:
                    result = method(self)
                finally:
                    self.in_recursive_rule -= 1
                endmark = self._mark()
                depth += 1
                if verbose:
                    print(
                        f"{fill}Recursive {method_name} at {mark} depth {depth}: {result!s:.200} to {endmark}"
                    )
                if not result:
                    if verbose:
                        self._log(f"Fail with {lastresult!s:.200} to {lastmark}")
                    break
                if endmark <= lastmark:
                    if verbose:
                        self._log(f"Bailing with {lastresult!s:.200} to {lastmark}")
                    break
                self._cache[key] = lastresult, lastmark = result, endmark

            self._reset(lastmark)
            tree = lastresult

            self._level -= 1
            if verbose:
                self._log(f"{method_name}() -> {tree!s:.200} [cached]")
            if tree:
                endmark = self._mark()
            else:
                endmark = mark
                self._reset(endmark)
            self._cache[key] = tree, endmark
        else:
            tree, endmark = self._cache[key]
            if verbose:
                self._log(f"{method_name}() -> {tree!s:.200} [fresh]")
            if tree:
                self._reset(endmark)
        return tree

    memoize_left_rec_wrapper.__wrapped__ = method  # type: ignore
    memoize_left_rec_wrapper._nolog = True
    return memoize_left_rec_wrapper


class Parser:
    """Parsing base class for Python parser."""

    KEYWORDS: ClassVar[Tuple[str, ...]]

    SOFT_KEYWORDS: ClassVar[Tuple[str, ...]]

    _cut_occurred: bool = False             # Set by parsing a Cut, cleared at start of every Alt.

    def __init__(self, tokenizer: Tokenizer, *, verbose: bool = False):
        self._tokenizer = tokenizer
        self._verbose = verbose
        self._level = 0
        self.start_mark: int = 0                # Tokenizer position at start of parsing the current Alt.
                                                # Saved and restored by self._alt().
        self._cache: Dict[Tuple[Mark, str, Tuple[Any, ...]], Tuple[Any, Mark]] = {}
        # Integer tracking whether we are in a left recursive rule or not. Can be useful
        # for error reporting.
        self.in_recursive_rule = 0
        if verbose:
            # Apply logger() to some methods to make instance variables with same name.
            for name in dir(self):
                if name.startswith('__'): continue
                method = getattr(self, name)
                if not callable(method): continue
                if name.startswith('_'):
                    # Log sunder methods 
                    continue
                else:
                    # Log regular methods (which are rules) unless they have method._nolog = True.
                    if getattr(method, '_nolog', False): continue
                setattr(self, name,
                    self._call_verbose(method),
                    )
                x = 0
        # Pass through common tokenizer methods.
        self._mark = self._tokenizer.mark
        self._reset = self._tokenizer.reset


    def _call_verbose(self, method) -> Callable:
        def wrapper(*args):
            method_name = method.__name__
            if not self._verbose:
                return method(self, *args)
            argsr = ",".join(repr(arg) for arg in args)
            self._log(f"{method_name}({argsr}) .... (looking at {self._showpeek()})")
            self._level += 1
            tree = method(*args)
            self._level -= 1
            self._log(f"... {method_name}({argsr}) --> {tree!s:.200}")
            return tree
        return wrapper

    @abstractmethod
    def start(self) -> Any:
        pass

    @property
    def _locations(self) -> TokenRange:
        """ The tokenizer start and end positions for the node being parsed.
        Called by the node's constructor, to be stored in the node.
        """
        return TokenRange(self._tokenizer._tokens[self.start_mark],
                   self._tokenizer.get_last_non_whitespace_token())

    @property
    def _lineno(self) -> int:
        """ The line number in the grammar file where the next token is found. """
        return self._tokenizer.peek().start[0]

    def _showpeek(self) -> str:
        tok = self._tokenizer.peek()
        return f"{tok.start[0]}.{tok.start[1]}: {token.tok_name[tok.type]}:{tok.string!r}"

    @memoize
    def _name(self) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.type == token.NAME and tok.string not in self.KEYWORDS:
            return self._tokenizer.getnext(),
        return None

    @memoize
    def _number(self) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.type == token.NUMBER:
            return self._tokenizer.getnext(),
        return None

    @memoize
    def _string(self) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.type == token.STRING:
            return self._tokenizer.getnext(),
        return None

    @memoize
    def _op(self) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.type == token.OP:
            return self._tokenizer.getnext(),
        return None

    @memoize
    def _type_comment(self) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.type == token.TYPE_COMMENT:
            return self._tokenizer.getnext()
        return None

    @memoize
    def _soft_keyword(self) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.type == token.NAME and tok.string in self.SOFT_KEYWORDS:
            return self._tokenizer.getnext()
        return None

    def _expect_name(self, name: str) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.type == token.NAME and tok.string == name:
            return self._tokenizer.getnext(),

    def _expect_type(self, type: int) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.exact_type == type:
            return self._tokenizer.getnext(),
        return None

    def _expect_char(self, char: str) -> ParseResult[Token]:
        tok = self._tokenizer.peek()
        if tok.type == token.CHAR and tok.string == char:
            return self._tokenizer.getnext(),
        return None

    def _expect_forced(self, res: Any, expectation: str) -> Any:
        if res is None:
            raise self._make_syntax_error(f"expected {expectation}")
        return res

    def _lookahead(self, positive: bool, func: Callable[..., T], *args: object) -> ParseStatus:
        mark = self._mark()
        item = bool(func(*args))
        self._reset(mark)
        return item == positive

    def _rule(self, rule: ParseFunc) -> ParseResult:
        """ Calls the rule's parse function and returns the result.
        This function is nested within the rule's main body.
        """
        # TODO: Can add some diagnostic info in self._verbose mode, similar to the memoize* wrappers.

        return rule()

    def _alts(self, alts: List[ParseFunc]) -> ParseResult:
        """ Parse several Alts.  Return first success result, or failure if an alt parses as Cut. """
        # TODO: Can add some diagnostic info in self._verbose mode, similar to pegen.c for a C parser.

        for alt in alts:
            alt = self._alt(alt)
            if alt: return alt
            if self._cut_occurred:
                # This Alt failed with a cut, quit.
                return None
            # This Alt failed, try again with the next one.
        # All Alts failed.
        return None

    def _alt(self, alt: ParseFunc) -> ParseResult:
        """ Parse an Alt.  Restore mark on failure. """
        save = self.start_mark
        mark = self.start_mark = self._mark()
        self._cut_occurred = False
        result = alt()
        self.start_mark = save
        if result: return result
        self._reset(mark)
        return result

    def _opt(self, item_func: Callable[..., ParseResult]) -> ParseResult[OptVal]:
        """ A list of (the item if parsed) or (empty if not parsed).
        This is wrapped in an OptVal object.
        """
        obj = item_func()
        return OptVal(obj and [obj[0]] or []),

    def _loop(self,
            item_func: Callable[[], ParseResult],
            sep_func: Callable[[], ParseResult],
            min: int,
            max: int = 0,
            ) -> ParseResult[List[GrammarNode]]:
        """ Parse some item some number of times, possibly with a separator between items.
        Specify min and max number of items.  Parse fails is min number is not reached.
        If the maximum number is reached, no more items are attempted.  max <= 0 means no limit.
        Result is a list of all the parsed items.
        """

        child: ParseResult = item_func()
        if not child:
            if min:
                # Only way to fail is a Repeat1 or Gather1, with no elements parsed.
                return None
            else:
                # Success is an empty list.
                return [],
        result: List[GrammarNode] = [child[0]]
        while max <= 0 or len(result) < max:
            mark = self._mark()
            if sep_func:
                sep = sep_func()
                if not sep:
                    self._reset(mark)
                    break
            child = item_func()
            if not child:
                self._reset(mark)
                break
            result.append(child[0])
        if len(result) < min:
            return None
        return result,

    def _repeat(self,
                item_func: Callable[[], ParseResult],
                repeat1: int,
                ) -> ParseResult[list]:
        return self._loop(item_func, None, repeat1)

    def _gather(self,
                item_func: Callable[[], ParseResult],
                sep_func: Callable[[], ParseResult],
                repeat1: int,
                ) -> ParseResult[list]:
        return self._loop(item_func, sep_func, repeat1)

    def _cut(self) -> None:
        self._cut_occurred = True

    def _get_val(self, item: ParseResult[GrammarNode]) -> Optional[GrammarNode]:
        if item is None: return None
        return item[0]

    def _make_syntax_error(self, message: str, filename: str = "<unknown>") -> SyntaxError:
        tok = self._tokenizer.diagnose()
        return SyntaxError(message, (filename, tok.start[0], 1 + tok.start[1], tok.line))

    def _log(self, msg: str) -> None:
        """ Print the message, with indentation and text wrapping. """
        fill = "  " * self._level
        import textwrap
        lines = textwrap.wrap(
            msg, width=80,
            initial_indent=fill,
            subsequent_indent=fill + '    '
            )
        for line in lines:
            print(line)

    def __repr__(self) -> str:
        return f"<Parser [{self._mark()}] at {self._tokenizer.peek().start}>"

def simple_parser_main(parser_class: Type[Parser]) -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print timing stats; repeat for more debug output",
    )
    argparser.add_argument(
        "-q", "--quiet", action="store_true", help="Don't print the parsed program"
    )
    argparser.add_argument("filename", help="Input file ('-' to use stdin)")

    args = argparser.parse_args()
    verbose = args.verbose
    verbose_tokenizer = verbose >= 3
    verbose_parser = verbose == 2 or verbose >= 4

    t0 = time.time()

    filename = args.filename
    if filename == "" or filename == "-":
        filename = "<stdin>"
        file = sys.stdin
    else:
        file = open(args.filename)
    try:
        tokengen = tokenize.generate_tokens(file.readline)
        tokenizer = Tokenizer(tokengen, verbose=verbose_tokenizer)
        parser = parser_class(tokenizer, verbose=verbose_parser)
        tree = parser.start()
        try:
            if file.isatty():
                endpos = 0
            else:
                endpos = file.tell()
        except IOError:
            endpos = 0
    finally:
        if file is not sys.stdin:
            file.close()

    t1 = time.time()

    if not tree:
        err = parser._make_syntax_error(filename)
        traceback.print_exception(err.__class__, err, None)
        sys.exit(1)

    if not args.quiet:
        print(tree)

    if verbose:
        dt = t1 - t0
        diag = tokenizer.diagnose()
        nlines = diag.end[0]
        if diag.type == token.ENDMARKER:
            nlines -= 1
        print(f"Total time: {dt:.3f} sec; {nlines} lines", end="")
        if endpos:
            print(f" ({endpos} bytes)", end="")
        if dt:
            print(f"; {nlines / dt:.0f} lines/sec")
        else:
            print()
        print("Caches sizes:")
        print(f"  token array : {len(tokenizer._tokens):10}")
        print(f"        cache : {len(parser._cache):10}")
