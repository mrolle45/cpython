""" Manages target language strings which are (usually) parsed from a grammar file. """

from __future__ import annotations

from itertools import chain
from pegen.grammar import GrammarTree, GrammarVisitor, GrammarError, ObjName
from pegen.tokenizer import Tokenizer
import token
import tokenize

Token = Tokenizer.Token

class Code(GrammarTree):
    """ Target-specific code which is for emitting into the generated parser.
    It is constructed from an optional sequence of Tokens or a string.
    For convenience, constructing from None returns None, and not a Code object.
    Constructing from a string tokenizes the string.
    The str() representation concatenates the tokens with appropriate spacing.
    If constructed with expand = True (the default), then macro replacement happens.
    """
    def __new__(cls, tokens: list[Token] | str = None, expand: bool = True, **kwds):
        if tokens is None:
            return None
        return super().__new__(cls)

    def __init__(self, tokens: list[Token] | str, expand: bool = True, **kwds):
        self.expand = expand
        if isinstance(tokens, str):
            # Tokenize the string
            tokens = _tokenize(tokens)
        if isinstance(tokens, Code):
            # Use tokens from Code
            tokens = tokens.tokens
        self.tokens = tokens
        super().__init__(**kwds)
        assert all(type(token) is Token for token in tokens)

    def __bool__(self) -> bool: return bool(self.tokens)

    def __contains__(self, name: str) -> bool:
        """ Does self contain any Token with given name? """
        for token in self.tokens:
            if token.string == name: return True
        return False

    def pre_init(self):
        super().pre_init()
        # Expand tokens with macros (if desired).
        if self.expand:
            self.tokens = list(self.do_expand(*self.tokens))

    def do_expand(self, *tokens) -> Iterator[Token]:
        macros = self.gen.macros
        if not macros:
            yield from tokens
            return
        expanding: set[str] = set()
        def recursive(*tokens: Token) -> Iterator[Token]:
            for token in tokens:
                if token.string in macros and token.string not in expanding:
                    expanding.add(token)
                    expanded = macros[token.string].tokens
                    yield from recursive(*expanded)
                    expanding.remove(token)
                else:
                    yield token

        yield from recursive(*tokens)

    def untokenize(self) -> str:
        """ Convert to a string, with only necessary spaces added. """
        def gen() -> Iterator[str]:
            last = None
            for token in self.tokens:
                yield self.add_space(last, token)
                yield token.string
                last = token

        return ''.join(gen())

    def add_space(self, left: Token | None, right: Token) -> str:
        if not left: return ''
        if all(token.type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING) for token in (left, right)):
            return ' '
        if left.string == ',' and right.string[:1] not in ')]}':
            return ' '
        return ''

    def __str__(self) ->str:
        return self.untokenize()

    def __repr__(self) -> str:
        return f"<Code {self}>"

    def iter_names(self, filt: set(str) = None) -> Iterator[str]:
        """ All names which are found in any Name token, possibly restricted to given set. """
        for tok in self.tokens:
            if tok.type == token.NAME:
                name = tok.string
                if not filt or name in filt:
                    yield name

    @property
    def names(self, filt: set(str) = None) -> set[str]:
        """ All names which are found in any Name token, possibly restricted to given set. """
        return set(self.iter_names())

    def iter_vars(self, env: LocEnv, filt: set(str) = None) -> set(str):
        """ Names found in any Name token, but only if the name is the same variable
        in the code as it is in given local env.  Possibly restricted to given set.
        """
        my_env = self.local_env
        for tok in self.tokens:
            if tok.type == token.NAME:
                name = tok.string
                if not filt or name in filt:
                    val = env.info(ObjName(tok))
                    if val:
                        my_val = my_env.info(ObjName(tok))
                        if not my_val or val is my_val:
                            yield ObjName(name)


class NoCode(Code):
    """ Specialized Code with no tokens, and tests false. """
    def __new__(cls):
        return super().__new__(cls, [])

    def __init__(self):
        super().__init__([])

    def __bool__(self) -> bool: return False

    def __repr__(self) -> str: return "<no code>"


class ValueCode(Code):
    """ A Code which is found in an annotation, denoting a value type. """
    has_value: bool = True


class NoValueCode(ValueCode):
    """ A Code which is found in an annotation, denoting NO value type.
    Tests as true.
    """
    has_value: bool = False

    def __new__(cls):
        return super().__new__(cls, [])

    def __init__(self):
        super().__init__([])

    def __bool__(self) -> bool: return True

    def pre_init(self):
        super().pre_init()
        super().__init__(self.gen.no_value_type())

    def __str__(self) -> str:
        return super().__str__() or "no value"

    def __repr__(self) -> str:
        res = super().__str__()
        return res and f"<ValueCode {res}>" or "<no value>"

class TargetCodeVisitor(GrammarVisitor):
    """ Visitor that collects all the target code items in a GrammarTree.
    Object Codes (from arguments and actions) and type Codes (from annotations) are separate.
    Has methods to analyze names found in these Codes.
    """

    def __init__(self, root: GrammarTree):
        """Builds sets of Code objects found recursively in given root tree.
        The constructor does the work.
        self.objs and self.types are the Code's, separated by where they came from.
        self.objs contains Codes which are in an Alt.action or Arg.name attribute.
        self.types contains Codes which are in a TypedName.type or Alt.type attribute.
        An action might contain a type name, but since the action is not analyzed syntactically,
            this will go undetected and the type name will be classified as an object name.
        """
        self.root = root
        self.objs = set()
        self.types = set()
        self.visit(root)

    # Name analysis methods.  A name is any NAME token.

    def check_type_names(self, excluded: Iterable[str]) -> None:
        """ Checks that NONE of the given names is used in any type Code.
        Raises a GrammarError otherwise.
        """
        names = self.type_names()
        if names & set(excluded):
            raise GrammarError(f"Variable names {names & set(excluded)} cannot be used as types.")

        x = 0

    def type_names(self, filt: Iterable[str] = None) -> set[Token]:
        return set(chain(*(code.iter_names() for code in self.types)))

    def vars(self, env: LocEnv, filt: set(ObjName) = None) -> set(ObjName):
        """ Names found in any Name token in any obj Code,
        but only if the name is the same variable in the code as it is in given local env.
        Possibly restricted to given set.
        """
        return set(chain(*(code.iter_vars(env) for code in self.objs)))

    def visit_Alt(self, tree: Alt) -> None:
        self.add_type(tree.val_type)
        self.add_obj(tree.action)
        self.generic_visit(tree)

    def visit_Arg(self, tree: Arg) -> None:
        self.add_obj(tree.code)
        self.generic_visit(tree)

    def visit_TypedName(self, tree: TypedName) -> None:
        self.add_type(tree.val_type)
        self.generic_visit(tree)

    def add_obj(self, code: Code) -> None:
        if code: self.objs.add(code)

    def add_type(self, typ: Code) -> None:
        if typ: self.types.add(typ)

def _tokenize(text: str) -> list[Token]:
    """ Separate given string into Tokens and return list of them. """
    import tokenize, pegen.tokenizer
    # list with one line including line terminator.
    lines: Iterator[str] = iter([f"{text}\n"])
    def readline():
        return next(lines)
    tokens = pegen.tokenizer.Tokenizer(tokenize.generate_tokens(readline))
    tokens = tokens.fill()[:-2]      # Skip NEWLINE and ENDMARKER
    return tokens


