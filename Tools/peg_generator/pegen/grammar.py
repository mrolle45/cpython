from __future__ import annotations

from abc import abstractmethod
import itertools

from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TypeVar,
)

if TYPE_CHECKING:
    from pegen.parser_generator import ParserGenerator


class GrammarError(Exception):
    pass


class GrammarVisitor:
    def visit(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, *args, **kwargs)

    def generic_visit(self, node: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        """Called if no explicit visitor function exists for a node."""
        for value in node:
            if type(value) is list:
                for item in value:
                    self.visit(item, *args, **kwargs)
            else:
                self.visit(value, *args, **kwargs)

    def visit_Attr(self, node: Attr) -> None:
        pass


class GrammarNode:
    """ Base class for all nodes found in a Grammar tree. """
    showme: bool = True

    def dump(self, all: bool = False) -> None:
        from pegen.parser_generator import DumpVisitor
        DumpVisitor(all).visit(self)

    def show(self, leader: str = ''):
        """ Same as str(self), but extra lines are indented with the leader.
        Override in subclass to implement this.
        """
        return str(self)

    def itershow(self) -> Iterable[GrammarNode]:
        return iter(self)

class Grammar(GrammarNode):
    def __init__(self, rules: Iterable[Rule], metas: Iterable[Tuple[str, Optional[str]]]):
        self.rules = {rule.name: rule for rule in rules}
        self.metas = dict(metas)

    def __str__(self) -> str:
        res = ", ".join(((name) for name, rule in self.rules.items()),)
        return res
        return "\n".join(str(rule) for name, rule in self.rules.items())

    def __repr__(self) -> str:
        lines = ["Grammar("]
        lines.append("  [")
        for rule in self.rules.values():
            lines.append(f"    {repr(rule)},")
        lines.append("  ],")
        lines.append("  {repr(list(self.metas.items()))}")
        lines.append(")")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[Rule]:
        yield from self.rules.values()


# Global flag whether we want actions in __str__() -- default off.
SIMPLE_STR = True


T = TypeVar('T')

class TrueHashableList(List[T], GrammarNode):
    """ Same as List[T] except that it is hashable and always true. """
    def __hash__(self) -> int: return id(self)

    def __bool__(self) -> bool: return True

    def __str__(self) -> str:
        return f'[{", ".join([str(x) for x in self])}]'

class TypedName(GrammarNode):
    def __init__(self, name: Optional[str], params: Optional[Params] = None, type: Optional[str] = None):
        self.name = name
        self.params = params
        self.type = type

    def __iter__(self) -> Iterator[GrammarNode]:
        yield Attr(self, 'name')

        if self.params: yield self.params

    def __str__(self) -> str:
        if not SIMPLE_STR and (self.params or self.type):
            res = self.name
            if self.params: res += f'({self.params})'
            if self.type: res += f'[{self.type}]'
            return res
        else:
            return str(self.name)

    def __repr__(self) -> str:
        return f"TypedName({self.name!r}, {self.type!r})"


class Meta:
    def __init__(self, name: str, val: str = None):
        self.name = name
        self.val = val

    def __iter__(self) -> Iterable:
        yield self.name
        yield self.val


class Rule(TypedName):
    def __init__(self, rulename: TypedName, rhs: Rhs, memo: Optional[object] = None):
        params = rulename.params or Params(())
        super().__init__(rulename.name, params, rulename.type)
        self.rhs = rhs
        self.memo = bool(memo)
        self.left_recursive = False
        self.leader = False

    def is_loop(self) -> bool:
        return self.name.startswith("_loop")

    def is_loop1(self) -> bool:
        return self.name.startswith("_loop1")

    def is_gather(self) -> bool:
        return self.name.startswith("_gather")

    def __str__(self) -> str:
        return super().__str__() + f': {self.rhs}'

    def show(self, leader: str = ''):
        res = str(self)
        if len(res) < 88:
            return res
        lines = [res.split(":")[0] + ":"]
        lines += [f"{leader}| {alt}" for alt in self.rhs]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Rule({self.name!r}, {self.type!r}, {self.rhs!r})"

    def __iter__(self) -> Iterator[Rhs]:
        yield TypedName(self.name, self.params, self.type)
        yield self.rhs

    def itershow(self) -> Iterator[Rhs]:
        yield TypedName(self.name, self.params, self.type)
        yield from self.rhs

    def flatten(self) -> Rhs:
        # If it's a single parenthesized group, flatten it.
        rhs = self.rhs
        if (
            not self.is_loop()
            and len(rhs) == 1
            and len(rhs[0].items) == 1
            and isinstance(rhs[0].items[0].item, Group)
        ):
            rhs = rhs[0].items[0].item.rhs
        return rhs

    @property
    def param_names(self) -> List[str]:
        return [param.name for param in self.params]

    @classmethod
    def simple(cls, name: str, params: Params, *args, **kwds) -> Rule:
        # Make Rule from just the name and any parameters.
        return cls(TypedName(name, params), *args, **kwds)


class Params(TrueHashableList[TypedName]):
    """ Parameters for a Rule or other callable. """

    def __init__(self, params: List[TypedName]):
        super().__init__(params)
        if params:
            # Verify that the names are unique.
            unique_names = {param.name for param in params}
            if len(unique_names) < len(params):
                raise GrammarError(f'Parameter names must be different.')

    def __str__(self) -> str:
        """ String used in a call expression, including the parent. """
        return f'({", ".join([param.name for param in self])})'

    def get(self, name: str) -> Optional[TypedName]:
        for param in self:
            if param.name == name: return param
        return None


class Leaf(GrammarNode):
    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __iter__(self) -> Iterable[str]:
        if False:
            yield


class Args(TrueHashableList[str]):
    """ List of rule arguments, and optional trailing comma. """
    empty: bool = False
    def __init__(self, args: List[str] = [], *, comma: str = ''):
        super().__init__(args)
        self.comma = comma or ''

    def show(self) -> str:
        return f'({", ".join(self)})'

    def __str__(self) -> str:
        res = ", ".join(self)
        res1 = '(' + res + ')'
        return self.show()
        return res1

    def __repr__(self) -> str:
        return self.show


class NoArgs(Args):
    empty: bool = True
    def __repr__(self) -> str:
        return '<no args>'
    def __str__(self) -> str:
        return ''


class NameLeaf(Leaf):
    """The value is the name, may also carry arguments."""
    args: Args = NoArgs()

    def __init__(self, name: str, args: Optional[Args] = None):
        super().__init__(name)
        if args: self.args = args

    def __str__(self) -> str:
        if self.value == "ENDMARKER":
            return "$"
        return super().__str__() + str(self.args)

    def __repr__(self) -> str:
        return f"NameLeaf({self.value!r})"


class StringLeaf(Leaf):
    """The value is a string literal, including quotes."""

    def __repr__(self) -> str:
        return f"StringLeaf({self.value!r})"


class Alt(GrammarNode):
    has_cut: bool = False               # True on instance if any item is a Cut object.

    def __init__(self, items: List[NamedItem], *, icut: int = -1, action: Optional[str] = None):
        self.items = items
        self.icut = icut
        self.action = action
        if any(isinstance(item.item, Cut) for item in items):
            self.has_cut = True

    def __str__(self) -> str:
        core = " ".join(str(item) for item in self.items) or '<always>'
        if not SIMPLE_STR and self.action:
            return f"{core} {{ {self.action} }}"
        else:
            return core

    def __repr__(self) -> str:
        args = [repr(self.items)]
        if self.icut >= 0:
            args.append(f"icut={self.icut}")
        if self.action:
            args.append(f"action={self.action!r}")
        return f"Alt({', '.join(args)})"

    def __iter__(self) -> Iterator[List[NamedItem]]:
        yield self.items


class Rhs(TrueHashableList[Alt]):

    def __str__(self) -> str:
        return " | ".join(str(alt) for alt in self)

    def __repr__(self) -> str:
        return f"Rhs({list(self)!r})"

    @property
    def can_be_inlined(self) -> bool:
        if len(self) != 1 or len(self[0].items) != 1:
            return False
        # If the alternative has an action we cannot inline
        if getattr(self[0], "action", None) is not None:
            return False
        return True


class NamedItem(TypedName):
    def __init__(self, name: Optional[TypedName], item: Item):
        if not name: name = TypedName(None)
        super().__init__(name.name, name.params, name.type)
        self.item = item

    def __str__(self) -> str:
        if not SIMPLE_STR and self.name:
            return f"{self.name}={self.item}"
        else:
            res = str(self.item)
            return res

    def __repr__(self) -> str:
        return f"NamedItem({self.name!r}, {self.item!r})"

    def __iter__(self) -> Iterator[Item]:
        if self.name: yield Attr(self, 'name')
        yield self.item

    @property
    def showme(self) -> bool:
        return bool(self.name)

class NamedItems(TrueHashableList[NamedItem]):
    pass


class Forced(GrammarNode):
    def __init__(self, node: Plain):
        self.node = node

    def __str__(self) -> str:
        return f"&&{self.node}"

    def __iter__(self) -> Iterator[Plain]:
        yield self.node


class Lookahead(GrammarNode):
    def __init__(self, node: Plain, sign: str):
        self.node = node
        self.sign = sign

    def __str__(self) -> str:
        return f"{self.sign}{self.node}"

    def __iter__(self) -> Iterator[Plain]:
        yield self.node


class PositiveLookahead(Lookahead):
    def __init__(self, node: Plain):
        super().__init__(node, "&")

    def __repr__(self) -> str:
        return f"PositiveLookahead({self.node!r})"


class NegativeLookahead(Lookahead):
    def __init__(self, node: Plain):
        super().__init__(node, "!")

    def __repr__(self) -> str:
        return f"NegativeLookahead({self.node!r})"


class Opt(GrammarNode):
    def __init__(self, node: Item):
        self.node = node

    def __str__(self) -> str:
        s = str(self.node)
        # TODO: Decide whether to use [X] or X? based on type of X
        if type(self.node) is Rhs:
            return s
        else:
            return f"{s}?"

    def __repr__(self) -> str:
        return f"Opt({self.node!r})"

    def __iter__(self) -> Iterator[Item]:
        yield self.node

    def itershow(self) -> Iterator[Item]:
        if type(self.node) is Rhs:
            yield from self.node
        else:
            yield self.node


class Repeat(GrammarNode):
    """Shared base class for x* and x+."""

    def __init__(self, node: Plain):
        self.node = node
        self.memo: Optional[Tuple[Optional[str], str]] = None

    def __iter__(self) -> Iterator[Plain]:
        yield self.node


class Repeat0(Repeat):
    def __str__(self) -> str:
        s = str(self.node)
        # TODO: Decide whether to use (X)* or X* based on type of X
        if " " in s:
            return f"({s})*"
        else:
            return f"{s}*"

    def __repr__(self) -> str:
        return f"Repeat0({self.node!r})"


class Repeat1(Repeat):
    def __str__(self) -> str:
        s = str(self.node)
        # TODO: Decide whether to use (X)+ or X+ based on type of X
        if " " in s:
            return f"({s})+"
        else:
            return f"{s}+"

    def __repr__(self) -> str:
        return f"Repeat1({self.node!r})"


class Gather(Repeat):
    def __init__(self, separator: Plain, node: Plain):
        self.separator = separator
        self.node = node

    def __str__(self) -> str:
        return f"{self.separator!s}.{self.node!s}+"

    def __repr__(self) -> str:
        return f"Gather({self.separator!r}, {self.node!r})"


class Group(GrammarNode):
    def __init__(self, rhs: Rhs):
        self.rhs = rhs

    def __str__(self) -> str:
        return f"({self.rhs})"

    def __repr__(self) -> str:
        return f"Group({self.rhs!r})"

    def __iter__(self) -> Iterator[Rhs]:
        yield self.rhs

    def itershow(self) -> Iterator[Rhs]:
        yield from list(itertools.chain(*itertools.chain(self.rhs)))


class Cut(GrammarNode):
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"Cut()"

    def __str__(self) -> str:
        return f"~"

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        if False:
            yield

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cut):
            return NotImplemented
        return True

    def initial_names(self) -> AbstractSet[str]:
        return set()


class Attr(GrammarNode):
    """ Describes an attribute of another node, for display purposes. """
    def __init__(self, node: GrammarNode, attr: str):
        self.node = node
        self.attr = attr

    def __str__(self) -> str:
        return f'{self.attr} = {getattr(self.node, self.attr)!r}'

Plain = Union[Leaf, Group]
Item = Union[Plain, Opt, Repeat, Forced, Lookahead, Rhs, Cut]
