from __future__ import annotations

import token
import tokenize
from typing import Dict, Iterator, List
import dataclasses
import re

Mark = int  # NewType('Mark', int)

exact_token_types = token.EXACT_TOKEN_TYPES


def shorttok(tok: tokenize.TokenInfo) -> str:
    return "%-25.25s" % f"{tok.start[0]:2d}.{tok.start[1]:2d}: {token.tok_name[tok.type]}:{tok.string!r}"


class Tokenizer:
    """Caching wrapper for the tokenize module.

    This is pretty tied to Python's syntax.
    """

    class Token(tokenize.TokenInfo):
        """ The tokenize token class, with a better repr(). """
        def __repr__(self) -> str:
            return f"<Token {self.string!r}: {token.tok_name[self.type]}>"

    _tokens: List[Token]

    def __init__(
        self, tokengen: Iterator[Token], *, path: str = "", verbose: bool = False
    ):
        self._tokengen = tokengen
        self._tokens = []
        self._index = 0
        self._verbose = verbose
        self._lines: Dict[int, str] = {}
        self._path = path
        if verbose:
            self.report(False, False)

    def getnext(self) -> Token:
        """Return the next token and update the index.  Does NOT advance past ENDMARKER."""
        cached = not self._index == len(self._tokens)
        tok = self.peek()
        if tok.type != tokenize.ENDMARKER:
            self._index += 1
        if self._verbose:
            self.report(cached, False)
        return tok

    def peek(
        self,
        merge: bool = True          # Combine a '&' with a following '&'
        ) -> Token:
        """Return the next token *without* updating the index."""
        while self._index == len(self._tokens):
            tok = next(self._tokengen)
            tok = self.Token(*tok)
            if tok.type in (tokenize.NL, tokenize.COMMENT):
                continue
            if tok.type == token.ERRORTOKEN:
                if tok.string.isspace(): continue
                if tok.string[0] in '\'"':
                    raise SyntaxError(f'Unmatched quote, at line {tok.start[0]}:{tok.start[1]}')
            if (
                tok.type == token.NEWLINE
                and self._tokens
                and self._tokens[-1].type == token.NEWLINE
            ):
                continue
            self._tokens.append(tok)
            if not self._path:
                self._lines[tok.start[0]] = tok.line

            # Check for two adjacent '& tokens.
            if tok.type == token.OP and tok.string == '&' and merge:
                # Look ahead one more time
                tok2 = self.peek(merge=False)
                if (tok2.type == token.OP and tok2.string == '&'
                    and tok.end == tok2.start
                    ):
                    # Merge the two, and remove the second one
                    tok.end = tok2.end
                    tok.string = '&&'
                    del self._tokens[-1]

        return self._tokens[self._index]

    def fill(self) -> list[Token]:
        """ Reads all remaining Tokens into the cache and returns the lot. """
        while True:
            try:
                if self.getnext().type == tokenize.ENDMARKER: break
            except StopIteration: break
        return self._tokens

    def diagnose(self) -> Token:
        if not self._tokens:
            self.getnext()
        return self._tokens[-1]

    def get_last_non_whitespace_token(self) -> Token:
        if not self._index:
            return self._tokens[0]
        for tok in reversed(self._tokens[: self._index]):
            if not token.ISWHITESPACE(tok.type):
                break
        return tok

    def get_lines(self, line_numbers: List[int]) -> List[str]:
        """Retrieve source lines corresponding to line numbers."""
        if self._lines:
            lines = self._lines
        else:
            n = len(line_numbers)
            lines = {}
            count = 0
            seen = 0
            with open(self._path) as f:
                for l in f:
                    count += 1
                    if count in line_numbers:
                        seen += 1
                        lines[count] = l
                        if seen == n:
                            break

        return [lines[n] for n in line_numbers]

    def mark(self) -> Mark:
        return self._index

    def reset(self, index: Mark) -> None:
        if index == self._index:
            return
        assert 0 <= index <= len(self._tokens), (index, len(self._tokens))
        old_index = self._index
        self._index = index
        if self._verbose:
            self.report(True, index < old_index)

    def report(self, cached: bool, back: bool) -> None:
        if back:
            fill = "-" * self._index + "-"
        elif cached:
            fill = "-" * self._index + ">"
        else:
            fill = "-" * self._index + "*"
        if self._index == 0:
            print(f"{fill} (Bof)")
        else:
            tok = self._tokens[self._index - 1]
            print(f"{fill} {shorttok(tok)}")

    def showtokens(self) -> List[Token]:
        return [shorttok(tok) for tok in self._tokens]

    def dump(self, ctx: int = 0) -> None:
        """ Prints all the tokens, without consuming any input or altering the index.
        Optionally, prints only several items before and after the current size.
        """
        list(map(print, (self.showtokens())))
        index = self._index
        size = len(self._tokens)
        # Fill the array.
        self.fill()
        for i, tok in enumerate(self._tokens):
            if ctx:
                if i < size - ctx: continue
                if i == size: print('---')
                if i > size + ctx: break
            print(f'{i!s:>3} {shorttok(tok)}')
        self._index = index
        # Restore the array to original size.
        del self._tokens[size:]

@dataclasses.dataclass
class TokenLocations:
    """ First and last input positions associated with some parsed result.
    Column numbers are 0-based, as with the Token class, but display as 1-based.
    """
    srow: int
    scol: int
    erow: int
    ecol: int

    def __contains__(self, other: TokenLocations) -> bool:
        """ is the other locations entirely contained in self locations? """
        if other.srow < self.srow: return False
        if other.erow > self.erow: return False
        if other.srow == self.srow and other.scol < self.scol: return False
        if other.erow == self.erow and self.ecol and other.ecol > self.ecol: return False
        return True

    def showstart(self) -> str:
        return f"{self.srow}:{self.scol + 1}"

    def showrange(self) -> str:
        if self.erow == self.srow:
            return f"{self.srow}:{self.scol + 1}-{self.ecol + 1}"
        else:
            return f"{self.srow}:{self.scol + 1} - {self.erow}:{self.ecol + 1}"

    def __str__(self) -> str:
        return f"{self.srow},{self.scol + 1:3d} -{f' {self.erow}, ' if self.erow != self.srow else ''}{self.ecol + 1:3d}"

    def __repr__(self) -> str:
        return f"TokenLocations({self})"

class TokenLocationsMatchExact(TokenLocations):
    """ Test whether given locations are exactly equal to this range. """
    def __call__(self, other: TokenLocations) -> bool:
        return (
            (self.srow, self.scol, self.erow, self.ecol)
            == (other.srow, other.scol, other.erow, other.ecol)
            )

class TokenLocationsMatchContains(TokenLocations):
    """ Test whether given locations are exactly equal to this range. """

    def __call__(self, other: TokenLocations) -> bool:
        if other.srow < self.srow: return False
        if other.erow > self.erow: return False
        if other.srow == self.srow and other.scol < self.scol: return False
        if other.erow == self.erow and self.ecol and other.ecol > self.ecol: return False
        return True

class TokenLocationsMatchOverlap(TokenLocations):
    """ Test whether given locations partially intersect with this range. """

    def __call__(self, locs: TokenLocations) -> bool:
        if other.srow > self.erow: return False
        if other.erow < self.srow: return False
        if other.srow == self.erow and self.ecol and other.scol >= self.ecol: return False
        if other.erow == self.srow and other.ecol <= self.scol: return False
        return True

class TokenRange(TokenLocations):
    """ First and last tokens associated with some parsed result. """
    def __init__(self, first: Token, last: Token):
        super().__init__(*first.start, *last.end)
        self.first, self.last = first, last

class TokenMatch(list[TokenLocations]):
    """ An object which will test a TokenRange for overlap with any of given ranges.
    The initializer is a multi-line string of lines that each represent a TokenRange.
    """
    def __init__(self, inits: str):
        for init in inits.split('\n'):
            init = init.split('#')[0]
            split = init.split(None, 1)
            if not split: continue 
            match_mode = split[0]
            if match_mode == '=':
                cls = TokenLocationsMatchExact
                init = split[1]
            elif match_mode == '<':
                cls = TokenLocationsMatchExact
                init = split[1]
            elif match_mode == '&':
                cls = TokenLocationsMatchExact
                init = split[1]
            else:
                cls = TokenLocationsMatchExact

            items = re.findall(r"[\w]+|[^\s\w]", init)
            if not items or items[0][0] == '#': continue
            # Format is: srow [ [ ',' scol ]  '-'   (   erow ',' ecol
            #                                       |   ecol
            #                 ]
            # scol defaults to 0.
            # erow defaults to srow.
            # ecol defaults to 0, meaning the end of the row.

            srow = int(items.pop(0))
            scol, erow, ecol = 0, srow, 0

            if items:
                s = items.pop(0)
                if s == ',':
                    scol = int(items.pop(0))
                if items:
                    assert items.pop(0) == '-'
                    s = int(items.pop(0))        # erow or ecol, depending on what follows
                    if items:
                        erow = s
                        assert items.pop(0) == ','
                        ecol = int(items.pop(0))
                    else:
                        ecol = int(s)

            self.append(cls(srow, scol - 1, erow, ecol - 1))

    def match(self, locations: TokenLocations) -> bool:
        """ True if any of stored locations list overlaps given locations. """
        for locs in self:
            if locs(locations): return True
        return False
