from __future__ import annotations

from abc import abstractmethod
import collections
import itertools
from enum import Enum, auto
import functools
import operator
import re
import ast
import threading
import dataclasses
import typing
from abc import ABC

from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Union,
    TypeVar,
)


from pegen.tokenizer import Tokenizer
Token = Tokenizer.Token

if TYPE_CHECKING:
    from pegen.parser_generator import ParserGenerator

ValueType = TypeVar('ValueType')

class GrammarError(Exception):
    pass


class GrammarVisitor:
    """ Walk a GrammarTree, top-down, and perform custom methods on certain classes of trees.
    If a method wants to descend further, it has to call generic_visit() explicitly.
    Extra args and keywords are passed on to all subtrees.
    """

    def visit(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, *args, **kwargs)

    def generic_visit(self, node: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        """Called if no explicit visitor function exists for a node."""
        for value in node.children():
            if type(value) is list:
                for item in value:
                    self.visit(item, *args, **kwargs)
            else:
                self.visit(value, *args, **kwargs)

    def visit_Attr(self, node: Attr) -> None:
        pass

    def visit_ObjName(self, node: ObjName) -> None:
        pass


@dataclasses.dataclass
class ThreadVars(threading.local):
    parser: Parser
    gen: ParserGenerator = None


class GrammarTree:
    """ Base class for everything here.
    Maintains a tree structure, using the .parent attribute, with Grammar at the top.
    The Grammar tree is built in five stages:
    1. The parser is stored in a thread-local object in the GrammarTree class.  This allows
    tree constructors to know what portion of the input file they correspond to.
    2. The parser's start() method is called.  This reads the input file and constructs the
    tree objects, from the bottom up, each constructor taking whatever child trees are needed.
    The constructors are actions in the Python or C parser script.
    The constructor copies the parser's current start and end locations into the tree.
    Now, each subtree knows its childrengrgrrrrr but does not know its parent.
    3. Create a ParserGenerator (C or Python), and store it in the GrammarTree's thread-local object.
    4. The Grammar object's initialize() method is called.  This completes initialization of all
    subtrees in the Grammar tree, by calling initialize() on them, going top-down.
    Class-specific operations are performed by pre_init() and post_init() methods.
    See the initialize() method for more details.
    5. Parse recipes, where supported, are created for all trees, going bottom_up.
    """
    name: str = None
    parent: GrammarTree = None      # The Node this Node is a part of.
    assigned_name: str = None       # Set by parent if an Alt or VarItem.
    gen: ParserGenerator
    showme: ClassVar[bool] = True
    _thread_vars: ClassVar[ThreadVars]
    _init_stage: InitStage

    def __init__(self, name: str = None, parent: GrammarTree = None, **kwds):
        super().__init__(**kwds)
        self._init_stage = self.InitStage.Constructor
        if name: self.name = name
        self.parser = self._thread_vars.parser
        self.locations = self.parser._locations
        if parent:
            # Do pre_ and post_init now, as the parent won't initialize.
            self.initialize(parent)

    def children(self) -> Iterator[GrammarTree]:
        try: yield from self
        except TypeError:
            # self has no __iter__ method
            pass
        if hasattr(self, 'parse_recipe'): yield self.parse_recipe

    @classmethod
    def set_parser(cls, parser: Parser) -> None:
        """ Call this after creating the Tokenizer used by the parser
        which builds the Grammar tree.
        All GrammarTree constructors can use this to get the current state
        of the input grammar file.
        The thread-local gen will be set later in the gen's constructor
        """
        cls._thread_vars = ThreadVars(parser)
        pass

    @property
    def grammar(self) -> Grammar:
        return self.parent and self.parent.grammar

    @property
    def rule(self) -> Rule:
        return self.parent and self.parent.rule

    @property
    def alt(self) -> Alt:
        return self.parent and self.parent.alt

    def initialize(self, parent: GrammarTree = None) -> None:
        """ Complete part 4 of the grammar tree initialization, for this tree and all descendants.
        1. Link tree to its parent.
        2. Initialize things which depend on the parent, and are common to all tree classes.
        3. Call self.pre_init().  Class-dependent code.  It can depend on steps 1 to 3 above
            being performed already for all the ancestors.
        4. Call child.intiialize(self) for all the child subtrees.
        5. Call self.post_init().  This can depend on initialize() being performed
            already for all the descendants.
        6. Create the parse recipe for the tree, if its class supports it.
        """
        self.parent = parent
        if parent:
            self.local_env = parent.local_env
        #self.message(f'{type(self).__name__} = {self.show()}')
        #if self.local_env.owner is not self:
        #    self.local_env.dump(self)
        self.pre_init()
        for child in self.children():
            try: child.initialize(self)
            except GrammarError as e:
                if child.rule is not child: raise
                child.delete()

        self.post_init()

    def pre_init(self) -> None:
        """ Class-specific initialization, after all ancestors have been pre_initialized. """
        self.gen = self._thread_vars.gen
        self._init_stage = self.InitStage.PreInit

    def post_init(self) -> None:
        """ Class-specific initialization, after all descendants have been initialized. """
        self._init_stage = self.InitStage.PostInit
        pass

    def make_parse_recipes(self) -> None:
        """ Create a parse recipe, where supported by its class, for self and all descendants.
        Proceeds bottom_up.
        """
        for child in self.children():
            if child.rule and child.rule._deleted:
                continue
            try: child.make_parse_recipes()
            except GrammarError as e:
                if child.rule is not child: raise
                child.delete()

        make_parse_recipe = getattr(self, 'make_parse_recipe', None)
        if make_parse_recipe:
            recipe = self.parse_recipe = make_parse_recipe()
            if recipe:
                recipe.initialize(self)

    def value_type(self) -> str:
        """ The generated expression for the type of the parsed result. """
        try: return self.parse_recipe.value_type()
        except AttributeError: return self.type

    def add_local_name(self, value: GrammarTree, name: str = None) -> None:
        """ Adds entry for the name in the local names map.
        """
        self.local_env.add(self, name, value)

    def remove_local_names(self, *names: str) -> None:
        """ Removes entries for the names (if any) in the local names map.
        """
        self.local_env.remove(self, *names)

    def uniq_name(self) -> str:
        """ A name which is unique among all tree objects. """
        anc = self.parent
        if isinstance(anc, VarItem):
            return anc.uniq_name()
        return f"{anc.uniq_name()}_{self.assigned_name or self.name}"

    def walk(self, parent: GrammarTree = None) -> Iterator[Tuple[GrammarTree, GrammarTree]]:
        """ Iterates top-down yielding (parent, child), starting with (given parent, self). """
        yield parent, self
        for child in self:
            if child:
                yield from child.walk(self)

    @staticmethod
    def chk_type(obj, *typ: type, opt: bool = False) -> None:
        """ Assert that object is instance of given type(s), or optionally None. """
        if opt and obj is None: return
        assert isinstance(obj, typ), (
            f"Expected {' or '.join(t.__name__ for t in typ)}, not {obj!r}.")

    def chk_types(self, objs: Iterator[object], *typ: Type) -> None:
        for obj in objs:
            self.chk_type(obj, *typ)

    def validate(self, visited: set = set(), level: int = 0) -> None:
        """ Raise an exception if anything is incorrect.  Recursively.
        Also validates all recipes.
        Subclass should call super().validate().
        """
        if self in visited: return
        visited.add(self)
        #print('  ' * level, repr(self))
        if hasattr(self, 'parse_recipe'):
            self.parse_recipe.validate(visited, level + 1)
        for sub in self.children():
            sub.validate(visited, level + 1)

    def full_name(self) -> str:
        res = self.name.string
        if self.type: res += f' [{self.return_type}]'
        if self.params: res += f'{self.params.in_func(use_default=False)}'
        return res

    def func_params(self) -> str:
        """ The string for the parameters of a function, if this is callable.
        Otherwise empty string.
        """
        if not self.params: return ""
        return f"({self.params.in_func()})"

    def grammar_error(self, msg: str) -> NoReturn:
        e = GrammarError(
            f"Grammar error: rule {self.rule.name.string!r}, {msg}  At line {self.locations.showrange()}.")
        self.grammar._errors[self.rule.name.string] = e
        raise e

    def show(self, leader: str = ''):
        """ Same as str(self), but extra lines are indented with the leader.
        Override in subclass to implement this.
        """
        return str(self)

    def itershow(self) -> Iterator[GrammarTree]:
        return iter(self)

    def attrs(self) -> Iterator[Attr]:
        """ Enumerate attributes of this node.
        Subclass with more attributes calls the super().attrs() first.
        """
        if False:
            yield None

    def message(self, msg: str, indent: str = '... ') -> None:
        """ Print message with leader for the nesting depth. """
        lines = msg.splitlines()
        print(f"{'    ' * self.depth()}{lines[0]}")
        for line in lines[1:]:
            print(f"{'    ' * self.depth()}{indent}{line}")

    def dump(self, all: bool = False, env: bool = False) -> None:
        """ Prints hierarchical dump of the tree starting with self. """

        from pegen.parser_generator import DumpVisitor
        DumpVisitor(all, env=env).visit(self)

    def depth(self) -> int:
        return self.parent and self.parent.depth() + 1 or 1

    def brk(self, delta: int = 0) -> bool:
        """ For setting a breakpoint in a debugger. """
        if self.parser:
            if self.parser.brk(delta):
                return True
            if self.parser.brk_token(self):
                return True
        return False

    class InitStage(Enum):
        Constructor = auto()
        PreInit = auto()
        PostInit = auto()


class OptVal(GrammarTree, Generic[ValueType]):
    """ Wrapper around a parse result for an Opt expression.
    The parse results in either [] or [result].  OptVal.val behaves the same way.
    OptVal.opt attribute is either None or result.
    """
    def __init__(self, val: Sequence[ValueType]):
        assert len(val) <= 1
        self.val = val

    def __bool__(self) -> bool:
        return bool(self.val)

    def __iter__(self) -> Iterator[ValueType]:
        if self.val: yield self.opt

    @property
    def opt(self) -> ValueType | None:
        return self.val and self.val[0] or None

    def __repr__(self) -> str:
        return f"<OptVal {self.opt}>"


class ParseExpr(GrammarTree):
    """ Represents some expression within the grammar file, which parses the input file
    and produces a success/failure status and (if success) a value.
    Some expressions are always successful, and some don't produce a value.
    In the generated parser, it is a function which is called from a parent function,
    corresponding to a parent ParseExpr.
    self.parse_recipe tells how to generate the code for this function.
    """

    src: ParseSource = None

    def __init__(self,
        name: Optional[str] = None,
        typ: Optional[str] = None,         # target language default type if not given to constructor.
        params: Optional[Params] = None,
        func_type: ParseFuncType = None,
        **kwds,
        ):

        if name: self.name = name
        elif self.name: self.name = ObjName(self.name)
        self.type = typ or NoType()
        super().__init__(**kwds)
        self.func_type = func_type or ParseFunc
        self.params = params

    def pre_init(self) -> None:
        super().pre_init()
        if not self.type:
            self.type = NoType()

    def children(self) -> Iterator[GrammarTree]:
        yield from super().children()
        if self.src: yield self.src

    def has_value(self) -> bool:
        return self.func_type.has_result and self.val_type.has_value

    @property
    def return_type(self) -> Type:
        return self.type.return_type

    @return_type.setter
    def return_type(self, typ: Type) -> Type:
        self.type.return_type = typ

    @property
    def val_type(self) -> Code:
        return self.type.val_type

    def vars_used(self) -> Iterator[ObjName]:
        """ All variable names used directly in this expression.
        They require being defined in the generated code.
        """
        if False: yield

    def decl_var(self, name: str = None) -> str:
        """ The declaration of a name (without the value), as a variable.
        If callable, the variable is a function pointer,including function parameters.
        """
        if name is None: name = self.name
        params = self.params
        typ = self.type
        if not params: return f"{type} {name}"
        # It's a function.
        return f"{typ} (*{name}) ({params.in_func()})"

    def  make_proxy_type(self, *targets: ParseExpr, params: Params = None) -> ProxyType:
        return ProxyType(targets, params, node=self)

@dataclasses.dataclass()
class LocEnv:
    """ Information about visible names, other than rules, in some owner tree.
    LocEnv is inherited from the owner's parent, unless that owner
    defines additional local names.  In this case, the owner gets a new LocEnv
    when each name is defined, reflecting only those new names defined up to then.

    When the owner has a child, the child inherits the owner's LocEnv.  However,
    if the owner defines a local name for the child, the new local name is NOT
    inherited by the child; the owner's LocEnv is not updated until after the
    child is initialized.

    Each Rule maintains information about all names defined in the rule and its
    descendants.  The rule tracks the maximum number of names visible in any descendant
    and makes an array of result pointers to that size.

    An owner may define a new name for a child which is a duplicate of one in its parent.
    The name will NOT be visible in that child, and any earlier child of the owner.


    """
    class ValueInfo(typing.NamedTuple):
        value: GrammarTree
        index: int                  # Position in list of names and Parser local_vars[].

    owner: GrammarTree                       # The tree that self was created by.
    map: Mapping[ObjName, ValueInfo] = (
        dataclasses.field(default_factory=dict)) # The innermost variable for each name.
    count: int = 0

    def __contains__(self, name: ObjName) -> bool:
        return name in self.map

    def lookup(self, name: ObjName) -> GrammarTree | None:
        if name in self.map:
            return self.map[name].value
        return None

    def index(self, name: ObjName) -> int:
        return self.map[name].index

    def info(self, name: ObjName) -> ValueInfo | None:
        return self.map.get(name, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self.map)

    def values(self) -> Iterator[GrammarTree]:
        return (info.value for info in self.map.values())

    def items(self) -> Iterator[Tuple[ObjName, GrammarTree]]:
        return ((name, info.value) for name, info in self.map.items())

    def add(self, child: GrammarTree, name: ObjName, value: GrammarTree) -> None:
        """ Add a local variable to the child tree.
        The child gets a copy of the current local vars if its local vars
        is the same object as the current.
        The rule's maximum local vars is updated.
        """
        #child.message(f"ADD {name}...")
        assert(isinstance(name, ObjName))
        self = child.local_env = dataclasses.replace(self, owner=child, map=self.map.copy())
        self.map.pop(name, None)            # To make map[name] iterate last.
        self.map[name] = self.ValueInfo(value, self.count)
        self.count += 1
        child.rule.need_local_vars(self.count)
        #self.dump(child)

    def remove(self, child: GrammarTree, *names: str) -> None:
        """ Remove some local variables from the child tree.  These are variables which
        will eventually be added.  It does not affect the current count, only the map.
        The child gets a copy of the current local vars.
        """
        self = child.local_env = dataclasses.replace(self, owner=child, map=self.map.copy())
        for name in names:
            # Remove the name if it is currently defined.
            self.map.pop(name, None)

    def __str__(self) -> str:
        return ', '.join(
            f"{name}={val.index}" for name, val in self.map.items())

    def __repr__(self) -> str:
        return f"<LocEnv {self}>"

    def dump(self, owner: GrammarTree) -> None:
        #if owner is not self.owner:
        #    return
        for name, (value, index) in self.map.items():
            owner.message(f"{index} / {self.count} [{name}] = {value!r}")


class ObjName(GrammarTree):
    """ Wrapper around a string which represents the name of some object value.
    It comes from a NAME Token, or possibly just a plain str.
    """
    def __new__(cls, tok: Token | str = None):
        if tok is None:
            return None
        return super().__new__(cls)

    def __init__(self, string: Token | str):
        super().__init__()
        if isinstance(string, Token):
            string = string.string
        else: assert isinstance(string, str)
        self.string = string

    def __eq__(self, other: ObjName) -> bool:
        return self.string == other.string

    def __hash__(self) -> int:
        return hash(self.string)

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return f"<ObjName {self}>"


class NoName(ObjName):
    def __init__(self):
        super().__init__()

    def __bool__ (self) -> bool: return False


class TypedName(ParseExpr):

    def __init__(self,
        name: str | Token | ObjName = None,
        typ: str | Type = None,         # target language default type if not given to constructor.
        **kwds,
        ):
        if isinstance(typ, (str, ValueCode)):
            typ = ObjType(typ)
        super().__init__(typ=typ, **kwds)
        if isinstance(name, (str, Token)):
            name = ObjName(name)
        self.name = name
        if self.type.callable:
            self.params = self.type.params
        self.chk_type(name, ObjName, opt=True)
        self.chk_type(typ, Type, opt=True)

    def set_name(self, name: TypedName) -> None:
        """ Initialize from another name.  An alternative to supplying arguments to constructor. """
        self.name = name.name
        self.params = name.params
        self.type = name.type

    def __iter__(self) -> Iterator[GrammarTree]:
        if self.name: yield self.name
        if self.type: yield self.type

    @property
    def callable(self) -> bool:
        return self.type.callable

    def validate(self, visited: set = set(), level: int = 0) -> None:
        super().validate(visited, level)
        ##assert self.type

    def pre_init(self) -> None:
        super().pre_init()
        if not self.type:
            self.type = Type(self.gen.default_type())

    def typed_name(self, name: str = '') -> str:
        """ What is generated for the type of this TypedName, including optional name.
        This TypedName may have its own parameters, which are generated recursively.
        """
        parts: list[str] = []
        base_type = str(self.val_type)
        if base_type: parts.append(base_type)
        if self.callable:
            # This node is a callable type.
            for params in self.type.param_lists():
                subtypes = [subparam.typed_name() for subparam in params]
                parts.append(f'(*) ({", ".join(subtypes)})')
        else:
            if name: parts.append(str(name))
        p = ' '.join(parts)
        return p

    def full_type(self) -> str:
        """ Complete type name (without the name), including function parameters. """
        params = self.params
        type = self.type
        if not params: return type
        # It's a function.
        return f"{type} (*) ({params.in_func()})"

    def decl_var(self, name: str = None) -> str:
        """ The declaration of a name (without the value), as a variable.
        If callable, the variable is a function pointer,including function parameters.
        """
        if name is None: name = self.name
        params = self.params
        type = self.type
        if not params: return f"{type} {name}"
        # It's a function.
        return f"{type} (*{name}) ({params.in_func()})"

    def decl_func(self, fwd: bool = False, node: ParseExpr = None) -> str:
        """ The declaration of a callable name (without the value), as a function.
        The variable is a function (not a function pointer), including function parameters.
        """
        assert self.type.params and self.name
        type = str(self.val_type)
        name = self.name
        if isinstance(node, Rule):
            return self.gen.format_rule(node)

        if fwd:
            nparams = len(self.params)
            if type == 'ParseStatus' and nparams == 2:
                return f"ParseFunc {name}"
            if fwd and type == 'ParseStatus' and nparams == 1:
                return f"ParseTest {name}"
            if fwd and type == 'void' and nparams == 2:
                return f"ParseTrue {name}"
        return f"{type} {name}{self.params.in_func()}"

    def attrs(self) -> Iterator[Attr]:
        yield from super().attrs()
        yield Attr(self, 'name')
        yield Attr(self, 'type')

    def in_grammar(self) -> str:
        """ Text of the name, as it appears in the grammar file. """
        parts: list[str] = []
        if self.name:
            parts.append(str(self.name))
        parts.append(self.type.in_grammar())
        return ' '.join(parts)

    def __str__(self) -> str:
        if not SIMPLE_STR and (self.params or self.type):
            res = self.name
            if self.params: res += f'({self.params})'
            if self.type: res += f'[{self.type}]'
            return res
        else:
            return self.name.string

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}, {self.type}>"


class Grammar(GrammarTree):
    rule: Rule = None
    alt: Alt = None
    _errors: list[GrammarError] = {}

    def __init__(self, rules: Iterable[Rule], metas: Iterable[Tuple[str, Optional[str]]]):
        super().__init__()
        self.rules = {rule.name: rule for rule in rules}
        self.metas = dict(metas)

    @property
    def grammar(self) -> Grammar:
        return self

    def initialize(self) -> None:
        self.local_env = LocEnv(self)
        super().initialize()

    #def pre_init(self) -> None:

    def post_init(self) -> None:
        super().post_init()
        self.make_parse_recipes()
        self.validate()

    def validate(self, visited: set = set(), level: int = 0) -> None:
        for rule in dict(self.rules).values():
            try: rule.validate(visited, level + 1)
            except GrammarError as e:
                rule.delete()
        if self._errors:
            for name in sorted(self._errors):
                print(self._errors[name])
        else: print('Validate() passed.')

    def __str__(self) -> str:
        res = ", ".join((str(name) for name, rule in self.rules.items()),)
        return res
        return "\n".join(str(rule) for name, rule in self.rules.items())

    def __repr__(self) -> str:
        lines = ["Grammar("]
        lines.append("  [")
        for rule in self.rules.values():
            lines.append(f"    {repr(rule)},")
        lines.append("  ],")
        lines.append(f"  {repr(list(self.metas.items()))}")
        lines.append(")")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[Rule]:
        yield from dict(self.rules).values()


# Global flag whether we want actions in __str__() -- default off.
SIMPLE_STR = True


T = TypeVar('T')

class TrueHashableList(Generic[T]):
    """ Acts like List[T] except that it is hashable and always true. """
    def __init__(self, items: Sequence[T] = [], **kwds):
        self._items = list(items)
        super().__init__(**kwds)

    def __hash__(self) -> int: return id(self)

    def __bool__(self) -> bool: return True
    def __len__(self): return len(self._items)
    def __iter__(self): return iter(self._items)
    def __getitem__(self, i): return self._items[i]
    def __add__(self, other: Self) -> List[T]:
        return self._items + other._items

    def __str__(self) -> str:
        return f'[{", ".join([str(x) for x in self._items])}]'


class Meta:
    def __init__(self, name: str, val: str = None):
        super().__init__()
        self.name = name
        self.val = val

    def __iter__(self) -> Iterator:
        yield self.name
        yield self.val


class Rule(TypedName):
    max_local_vars: int = 0         # Greatest number of local names for any node in the tree.
    _deleted: bool = False          # True after a GrammarError has occurred.
    left_recursive: bool = False
    leader: bool = False

    def __init__(self, rulename: TypedName, params: OptVal[Params], rhs: Rhs, memo: Optional[object] = None):
        params = params.opt or NoParams()
        super().__init__(rulename.name, self.make_proxy_type(rhs, params=params))
        #super().__init__(rulename.name, rulename.type, params)
        self.name, self.params = (rulename.name, params)
        self.rhs = rhs
        self.memo = bool(memo)
        if rulename.return_type:
            self.return_type = rulename.return_type

    def pre_init(self) -> None:
        super().pre_init()
        #if not self.type: self.type = self.gen.default_type()
        if len(self.params):
            for param in self.params:
                self.add_local_name(param, param.name)

    def post_init(self) -> None:
        super().post_init()
        #self.type.dump()
        pass

    def need_local_vars(self, n: int) -> None:
        """ At least this many local variables may be needed. """
        if n > self.max_local_vars:
            self.max_local_vars = n

    @property
    def rule(self) -> Rule:
        return self

    def delete(self) -> None:
        """ Mark for later deletion.  Names can still resolve to self, and\
        a dummy parse function will be generated.
        """
        self._deleted = True

    def uniq_name(self) -> str:
        """ A name which is unique among all tree objects. """
        return f"_{self.name}"

    def show(self, leader: str = ''):
        res = str(self)
        if len(res) < 88:
            return res
        lines = [res.split(":")[0] + ":"]
        lines += [f"{leader}| {alt}" for alt in self.rhs]
        return "\n".join(lines)

    def __iter__(self) -> Iterator[Rhs]:
        if self.name: yield self.name
        yield self.params
        yield self.rhs

    def flatten(self) -> Rhs:
        # If it's a single parenthesized group, flatten it.
        rhs = self.rhs
        if (
            len(rhs) == 1
            and len(rhs[0]) == 1
            and isinstance(rhs[0][0], Group)
        ):
            rhs = rhs[0][0].rhs
        return rhs

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        memo = False
        if self.left_recursive and self.leader:
                parser = self.gen.parse_rule_recursive
                memo = True
        elif self.memo:
            parser = self.gen.parse_rule_memo
            memo = True
        else:
            parser = self.gen.parse_rule

        if len(self.params) and memo:
            self.grammar_error(
                f"Rule with parameters may not be memoized or left-recursive leader.")

        return self.gen.rule_recipe(self, parser)

    @property
    def param_names(self) -> List[str]:
        return [param.name for param in self.params]

    def __str__(self) -> str:
        self.gen.format_typed_name(self)
        return f"{self.full_name()}: {self.rhs}"

    def __repr__(self) -> str:
        return f"<Rule {self.name}, {self.type}, {self.rhs!r}>"


class Param(TypedName, GrammarTree):
    """ A single parameter description for a callable object.
    It may itself be a callable, with its own parameters (recursively).
    """
    def __init__(self, name: TypedName):
        super().__init__(None)
        self.set_name(name)


#class NoParams(Params):
#    empty: bool = True

#    def __init__(self):
#        super().__init__()

#    def __bool__(self) -> bool:
#        return False

#    def in_func(self) -> str:
#        return ''

#    def __str__(self) -> str:
#        return ''

#    def __repr__(self) -> str:
#        return '<No params>'


class Arg(GrammarTree):
    inline: GrammarTree = None

    def __init__(self, code: Code = None, inline: GrammarTree = None, gen: ParserGenerator = None):
        super().__init__(code)
        self.code = code
        if inline: self.inline = inline

    def __iter__(self) -> Iterator[GrammarTree]:
        yield self.code

    #def show_typed(self, type: TypedName) -> str:
    #    """ Name of this Arg, possibly coerced to given type. """

    #    result = str(self)
    #    # Check for a type cast needed.
    #    if self.inline:
    #        my_type: TypedName = self.inline.parse_recipe.func_name
    #        my_type_name = my_type.full_type()
    #        type_name = str(type.type)
    #        if my_type_name != type_name:
    #            if type_name == 'ParseFunc*' and my_type_name == 'ParseStatus (*) (Parser* _p, ParseResultPtr* _ppRes)':
    #                pass
    #            elif type_name == 'ParseTest*' and my_type_name == 'ParseStatus (*) (Parser* _p)':
    #                pass
    #            elif type_name == 'ParseTrue*' and my_type_name == 'void (*) (Parser* _p, ParseResultPtr* _ppRes)':
    #                pass
    #            else:
    #                result = type.gen.cast(result, type)
    #    return result

    def __str__(self) -> str:
        return self.code and str(self.code) or self.inline.parse_recipe.func_name.name.string

    def __repr__(self) -> str:
        if self.inline and self.name:
            return f"{self.name} = {self.inline!r}"
        elif self.name:
            return repr(self.name)
        else:
            return repr(self.inline)


class Args(TrueHashableList[Arg], GrammarTree):
    """ List of rule or parameter or variable arguments.
    An argument is either a string in the grammar file, or a TypedName for an inline item.
    For a parameter or variable call without arguments, use NoArgs subclass.
    """
    empty: bool = False
    def __init__(self, args: List[Arg | str | Tuple[str, GrammarTree]] = []):
        def arg(arg: Arg | str | Tuple[str, GrammarTree]):
            if isinstance(arg, str): return Arg(arg)
            elif isinstance(arg, Arg): return arg
            else: return Arg(*arg)

        super().__init__(items=[arg(a) for a in args])

    def show(self, *after_args: str) -> str:
        """ For display of an item following the name, including brackets.
        """
        args = ', '.join([*after_args, *(str(arg) for arg in self[:])])
        return f'<{args}>'

    def __str__(self) -> str:
        res = ", ".join(str(arg) for arg in self)
        res1 = '(' + res + ')'
        return res1

    def __repr__(self) -> str:
        return f"Args({self.show()!r})"


class NoArgs(Args):
    empty: bool = True

    def show(self, *after_args: str) -> str:
        """ For display of an item following the name, including brackets.
        """
        return ''

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return '<no args>'

    def __str__(self) -> str:
        return ''


class ArgsList(GrammarTree):
    """ Multiple layers of call arguments.  Each set of args is applied to some callable,
    and the return value is called with the next set of args.
    The list can be empty, meaning that the target is an object type.
    """
    def __init__(self, arg_lists: Sequence[Args]):
        self.arg_lists = arg_lists

    def children(self) -> Iterator[GrammarTree]:
        yield from self.arg_lists

    def __str__(self) -> str:
        x

class AltItem(ParseExpr):
    """ Anything which can be directly a part of an Alt.
    If the item produces a value, then:
        It may have a variable name, taken from the grammar file.  It becomes a local variable, and its value
        is available by name anywhere in the Alt following this item.
        Otherwise the Alt determines a name, which can be used in a default action for the Alt.
    """


class Alt(TrueHashableList[AltItem], ParseExpr):

    def __init__(self, items: List[AltItem], *, action: Optional[Code] = None):
        # The name will be chosen later when it is adopted by the Rhs.
        # The return type won't be set yet.  
        #   It may be set by the parent's constructor to a specific type.
        #   Otherwise, it will be set in pre_init() as a proxy.
        super().__init__(items, typ=FuncType(ValueCode([])))
        self.action = action or NoCode()
        self.assign_names()
        assert (isinstance(item, VarItem) for item in items)

    def __iter__(self) -> Iterator[GrammarTree]:
        yield from self.items()
        if self.action: yield self.action
        if isinstance(self.type, Type): yield self.type
        
    def pre_init(self) -> None:
        super().pre_init()
        # Check unique item variable names.
        item_names = self.var_names()
        unique_names = set(item_names)
        if len(unique_names) < len(item_names):
            self.grammar_error(f'Variable names {item_names} must be different.')
        # Make ProxyType for the item(s), unless it is set to something already.
        if not self.return_type:
            self.type = ProxyType(self.items(), node=self)
        t = self.val_type
        if self.var_names():
            self.remove_local_names(*self.var_names())

    def post_init(self) -> None:
        super().post_init()
        if not self.action:
            ret_action = self.gen.gen_action(self)
            self.action = Code(ret_action.expr, parent=self)
            self.action_type = ret_action.type
            if not self.val_type.has_value: self.type = ret_action.type
        else:
            self.action_type = NoType()
            if not self.val_type.has_value:
                self.grammar_error(f"Alternative <{self}> with an action must return a value")
        names = list(self.local_env)        # All variable names, old and new.
        self.codes = TargetCodeVisitor(self)
        if not self.val_type: self.type = Type(self.gen.default_type())

        # Analyze names used by each item in turn.
        # A name which is defined in this or a later item are a GrammarError.
        names_avail = set(self.parent.local_env) - set(self.var_names())
        env = self.local_env
        for item in self.items():
            vis = TargetCodeVisitor(item)
            item_names = vis.vars(env, names)
            bad_names = item_names - names_avail
            if bad_names:
                item.grammar_error(f"Variable(s) {', '.join(map(str, bad_names))!r} used before being parsed.")
            if item.var_name:
                names_avail.add(item.var_name.name)

        if self.codes.types:
            self.codes.check_type_names(names)
        if self.codes.objs:
            vars = self.codes.vars(self.local_env)
            x = 0

    @property
    def alt(self) -> Alt:
        return self

    def items(self) -> list[VarItem]:
        return self._items

    def default_action_vars(self) -> list[VarItem]:
        """ Which items participate in a default action. """
        vars = self._items
        if len(vars) <= 1:
            return vars
        # Multiple items.
        # Try restricting to names variables.
        named_vars = [var for var in vars if var.var_name]
        if named_vars: return named_vars
        # No named variables, so use all variables.
        return vars

    def vars_used(self) -> Iterator[ObjName]:
        """ All variable names used directly in this expression.
        They require being defined in the generated code.
        """
        x=0
        for item in self.items():
            if hasattr(item, 'parse_recipe'):
                if item.parse_recipe.mode in (item.parse_recipe.Rule, item.parse_recipe.Loc):
                    yield from item.vars_used()
        if self.action:
            yield from self.action.iter_vars(self.local_env)

    def assign_names(self) -> None:
        """ Set the assigned name for each VarItem.  Duplicate names will add a suffix. """
        counts = collections.Counter()
        for i in self.items():
            if i.var_name:
                name = i.var_name.name
            else:
                name = i.name
                n = counts[name]
                counts[name] += 1
                if n:
                    name = ObjName(f"{name}_{n}")
            i.assigned_name = name

    @functools.cache
    def var_names(self):
        return tuple(item.var_name.name for item in self.items() if item.var_name)

    @functools.cache
    def all_vars(self):
        """ All assigned variables, including for anonymous items, which have a parse result. """
        return tuple(item for item in self.items() if item.func_type.has_result)

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeInline(self,
            None, self.gen.parse_alt,
            func_type=ParseFunc,
            extra=lambda: self.gen.gen_alt(self),
            inlines=[item
                     for item in self.items()
                     ],
            inline_locals=False,
            value_type=self.val_type,
            comment=str(self),
            **kwds,
            )

    def show(self) -> str:
        return " ".join(item.show(show_var=False) for item in self.items()) or '<always>'

    def __str__(self) -> str:
        core = " ".join(item.show(show_var=False) for item in self.items()) or '<always>'
        if not SIMPLE_STR and self.action:
            return f"{core} {{ {self.action} }}"
        else:
            return core

    def __repr__(self) -> str:
        repr = str(self)
        if self.action:
            repr += (f" {{{self.action}}}")
        return f"<Alt {repr}>"


class Rhs(TrueHashableList[Alt], AltItem):

    name = '_rhs'
    nested: bool = False                    # Set True on instance if part of a Group.

    def __init__(self, alts: Sequence[Alt]):
        super().__init__(items=alts, typ=self.make_proxy_type(*alts))
        self.alts = alts
        for i, alt in enumerate(alts, 1):
            if len(alts) == 1:
                name = '_alt'
            else:
                name = f"_alt{i}"
            alt.name = name

    def post_init(self) -> None:
        r = self.return_type
        super().post_init()

    def make_parse_recipe(self, name: str = None, dflt_name: str = '_rhs', **kwds) -> ParseRecipe:

        # Args for the recipe are the Alts with generated names.
        alt_names: List[str]
        if len(self) == 1:
            alt_names = ['_alt']
            parser_name = self.gen.parse_alt
        else:
            alt_names = [f"_alt{i}" for i, _ in enumerate(self, 1)]
            parser_name = self.gen.parse_alts

        descr_name, extra = self.gen.gen_rhs_descriptors(self, alt_names)

        return ParseRecipeExternal(
            self, name or dflt_name, parser_name, descr_name, extra=extra,
            inlines=list(self),
            func_type=ParseFunc,
            value_type=self.val_type,
            **kwds,
            )

    def show(self, leader: str = ''):
        res = " | ".join(str(alt.show()) for alt in self[:])
        if len(res) < 88:
            return res
        lines = [f"{leader}| {alt}" for alt in self]
        return "\n".join(lines)

    def __str__(self) -> str:
        return " | ".join(str(alt) for alt in self[:])

    def __repr__(self) -> str:
        return f"<Rhs {list(self)!r}>"

    @property
    def can_be_inlined(self) -> bool:
        if len(self) != 1 or len(self[0].items) != 1:
            return False
        # If the alternative has an action we cannot inline
        if getattr(self[0], "action", None) is not None:
            return False
        return True


class Item(AltItem):
    """ Any expression which can be part of an Alt, directly or nested. """
    pass


class VarItem(AltItem):
    """ An Item which is found directly in an Alt, possibly associated with a named variable.
    Item with a name must have a parse result.
    The variable can be used as a nonterminal anywhere in the parent Alt *after* this Item.
    """
    assigned_name: ObjName

    def __init__(self, name: TypedName | None, item: Item):
        if name and name.type:
            typ = name.type
        else:
            typ = ProxyType([item], node=self)
        super().__init__(typ=typ, func_type=item.func_type)
        assert item is not None
        self.var_name = name
        self.name = name and name.name or item.name
        self.item = item

    def pre_init(self) -> None:
        super().pre_init()
        parent = self.parent
        assert type(parent) is Alt
        self.item.assigned_name = self.assigned_name
        if self.var_name:
            parent.add_local_name(self.item, self.var_name.name)
            #if not self.var_name.type:
            #    self.type = self.var_name.type = Type(self.gen.default_type())

    def validate(self, visited: set = set(), level: int = 0) -> None:
        if self.var_name:
            if not self.has_value():
                self.item.grammar_error(
                    f"Item {self.name} with variable name must have a parse result.")

    def vars_used(self) -> Iterator[ObjName]:
        """ All variable names used directly in this expression.
        They require being defined in the generated code.
        """
        return self.item.vars_used()

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return self.item.make_parse_recipe(
            assigned_name=self.assigned_name, owner=self, **kwds)

    def attrs(self) -> Iterator[Attr]:
        yield from super().attrs()
        yield Attr(self, 'assigned_name')

    def show(self, show_var: bool = True) -> str:
        if self.var_name and show_var:
            return f"{self.var_name.name}={self.item.show()}"
        else:
            return self.item.show()

    def __str__(self) -> str:
        return self.show()

    def __repr__(self) -> str:
        return f"<VarItem {self}>"

    def __iter__(self) -> Iterator[Item]:
        if self.var_name: yield self.var_name
        yield self.item


class AltItems(TrueHashableList[AltItem], GrammarTree):
    def __init__(self, items: List[AltItem] = []):
        super().__init__(items=items)


class Forced(Item):
    name = '_forced'

    def __init__(self, node: Item):
        super().__init__()
        self.node = node
        self.func_type = ParseTrue

    def post_init(self) -> None:
        super().post_init()
        src = self.src = self.gen.parse_forced(self.node)
        self.type = src.type

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeExternal(
            self,
            '_forced',
            self.gen.parse_forced(self.node),
            (None, self.node),
            f'"{self.gen.str_value(str(self.node))}"',
            **kwds
            )

    def __str__(self) -> str:
        return f"&& {self.node}"

    def __repr__(self) -> str:
        return f"Forced({self.node})"

    def __iter__(self) -> Iterator[Plain]:
        yield self.node


class Lookahead(Item):
    name = '_lookahead'

    def __init__(self, node: Atom, positive: bool):
        super().__init__(func_type=ParseTest)
        self.node = node
        node.name = '_atom'
        self.positive = positive

    def __iter__(self) -> Iterator[Plain]:
        yield self.node

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeExternal(
            self, '_lookahead', self.gen.parse_lookahead,
            self.gen.bool_value(self.positive),
            Arg(inline=self.node),
            func_type=ParseTest,
            **kwds)

    def __str__(self) -> str:
        return f"{'!&'[self.positive]}{self.node}"


class PositiveLookahead(Lookahead):
    def __init__(self, node: Atom):
        super().__init__(node, True)

    def __repr__(self) -> str:
        return f"PositiveLookahead({self.node!r})"


class NegativeLookahead(Lookahead):
    def __init__(self, node: Atom):
        super().__init__(node, False)

    def __repr__(self) -> str:
        return f"NegativeLookahead({self.node!r})"


class Cut(Item):
    name = '_cut'

    def __init__(self):
        super().__init__(typ=Type(NoValueCode()), func_type=ParseVoid)
        #super().__init__()

    def validate(self, visited: set = set(), level: int = 0) -> None:
        pass

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeExternal(
            self, "_cut", self.gen.parse_cut,
            func_type=ParseNone,
            **kwds, )

    def __repr__(self) -> str:
        return f"Cut()"

    def __str__(self) -> str:
        return f"~"


class Atom(Item):
    """ Common base class for a Primary or some combinations of other Atoms.
    """
    pass


class Call(Atom):
    """ An Atom which calls a given element, with given arguments. """
    name: str = '_call'

    def __init__(self, elem: Atom, args: Args, **kwds):
        super().__init__(**kwds, typ=self.make_proxy_type(elem))
        self.elem = elem
        self.args = args

    def children(self) -> Iterator[GrammarTree]:
        yield from super().children()
        yield self.elem
        yield self.args

    def pre_init(self) -> None:
        super().pre_init()
        self.src = self.gen.parse_call

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        # FOR NOW: do similar to Lookahead.
        return ParseRecipeExternal(
            self, '_call', self.src,
            Arg(inline=self.elem),
            func_type=ParseTest,
            **kwds)

    def __str__(self) -> str:
        return f"{self.elem} {self.args}"

    def __repr__(self) -> str:
        return f"<Call {self}>"

class Seq(Atom):
    """ An Atom whose parse result is a sequence of parse results of given element. """

    def __init__(self, elem: Atom, **kwds):
        super().__init__(**kwds)
        self.elem = elem

    def post_init(self) -> None:
        super().post_init()
        src = self.src = self.recipe_src()(self.elem)
        self.type = src.type

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeExternal(
            self, self.name, self.src,
            *self.gen.sequence_recipe_args(self, *self.parse_args()),
            func_type=self.func_type,
            **kwds)


class Opt(Seq):
    name: str = '_opt'

    def __init__(self, elem: Atom):
        super().__init__(elem, func_type=ParseTrue)

    def recipe_src(self) -> ParseSource: return self.gen.parse_opt

    def parse_args(self) -> Tuple[str, str]:
        return ()

    def __str__(self) -> str:
        return f"{self.elem}?"

    def __repr__(self) -> str:
        return f"<Opt {self}>"

    def __iter__(self) -> Iterator[Item]:
        yield self.elem

    def itershow(self) -> Iterator[Item]:
        if type(self.elem) is Rhs:
            yield from self.elem
        else:
            yield self.elem


class Repeat(Seq):
    """Shared base class for x* and x+."""

    name = '_repeat'

    def __iter__(self) -> Iterator[Plain]:
        yield self.elem

    def recipe_src(self) -> ParseSource: return self.gen.parse_repeat

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.elem!r})>"


class Repeat0(Repeat):

    def __init__(self, elem: Atom):
        super().__init__(elem, func_type=ParseTrue)

    def parse_args(self) -> Tuple[str, str]:
        return (self.gen.bool_value(False), )

    def __str__(self) -> str:
        return f"{self.elem}*"


class Repeat1(Repeat):

    def parse_args(self) -> Tuple[str, str]:
        return (self.gen.bool_value(True), )

    def __str__(self) -> str:
        return f"{self.elem}+"


class Gather(Seq):

    name = '_gather'

    def __init__(self, separator: Atom, elem: Plain, **kwds):
        super().__init__(elem, **kwds)
        self.separator = separator
        separator.name = '_sep'

    def __iter__(self) -> Iterator[Plain]:
        yield self.elem
        yield self.separator

    def recipe_src(self) -> ParseSource: return self.gen.parse_gather

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.separator!r}, {self.elem!r})>"


class Gather0(Gather):

    def __init__(self, separator: Atom, elem: Atom):
        super().__init__(separator, elem, func_type=ParseTrue)

    def parse_args(self) -> Tuple[str, str]:
        return (Arg(inline=self.separator), self.gen.bool_value(False))

    def __str__(self) -> str:
        return f"{self.separator!s}.{self.elem!s}*"


class Gather1(Gather):

    def parse_args(self) -> Tuple[str, str]:
        return (Arg(inline=self.separator), self.gen.bool_value(True))

    def __str__(self) -> str:
        return f"{self.separator!s}.{self.elem!s}+"


## Lowest level expressions...

class Primary(Atom):
    """ Common base class for an expression that is the lowest level of operator binding.
    These are Leaf expressions and general expressions in (...) or [...].
    """


class Group(Primary):
    """ A lowest level parsed expression which wraps an arbitrary ParseExpr in parentheses. """

    name = '_group'

    def __init__(self, rhs: Rhs, typ: Type = None):
        super().__init__(typ=self.make_proxy_type(rhs))
        if typ:
            self.return_type = typ.return_type
        self.rhs = rhs
        rhs.nested = True

    def __iter__(self) -> Iterator:
        yield self.rhs

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        func_name = self.rhs.parse_recipe.func_name
        src = ParseSource(
            self.gen,
            func_name.name,
            self.rhs.parse_recipe.value_type(),
            #self.rhs.parse_recipe.params,
            #Params(),
            #func_type=ParseFunc.inline,
            )
        return ParseRecipeInline(self, self.name, src,
            inlines=[self.rhs],
            func_type=ParseFunc,
            **kwds)

    def __str__(self) -> str:
        return f"({self.rhs.__str__()})"

    def __repr__(self) -> str:
        return f"Group({self.rhs.__repr__()})"


class OptGroup(Opt):
    """ A lowest level parsed expression which wraps an arbitrary ParseExpr in square brackets.
    Same as an ordinary Group, except that the result is optional, and returns a sequence,
    either containing the parsed result or an empty sequence.
    Parsing an OptGroup always succeeds.
    """

    def __init__(self, rhs: Rhs, typ: Type = None):
        super().__init__(Group(rhs, typ))
        #super().__init__(typ=self.make_proxy_type(rhs))
        #if typ: self.return_type = typ.return_type
        #self.rhs = rhs

    def __str__(self) -> str:
        return f"[{self.elem.rhs}]"

    def __repr__(self) -> str:
        return f"<OptGroup {self.elem.rhs}>"


class Leaf(Primary):
    def __init__(self, value: Any, **kwds):
        super().__init__(**kwds)
        self.value = value

    def __str__(self) -> str:
        return str(self.value)


class NameLeaf(Leaf):
    """ A parse expression given in the grammar by an identifer, possibly followed
    by arguments (expressions in the target language) in parentheses.
    An all UPPERCASE identifier corresponds to a type of Token, such as NEWLINE, and
    it matches any token of that type.  Arguments are not allowed here.
    A name which is not all uppercase denotes a named value,
    which may be an object type or a function type.
    A name beginning with '_' is a grammar error.
    For an object type, the plain name appears, and arguments are not allowed.
    For a function type, argument list (possibly empty) is required.  The number of
    arguments must equal the number of parameters of the function.
    An exception to the above is that if the named value is a rule, which is a function,
    a missing argument list is allowed, and is considered as an empty list, i.e., 'rule()'.

    Each name has a scope, which means the parse expressions in the grammar in which it
    may appear.
    The same name can be defined in multiple scopes with one being a subset of the other.
    In this case, the definition in the innermost scope is the one which is used.

    In addition to a NameLeaf expression, the name may appear in the target language
    expression used as an argument in a NameLeaf, and as the action for a rule alternative.

    The kinds of names that are defined in the grammar file, and their scopes, are:
    *   A rule.  The scope is the entire grammar file.
    *   A parameter of a rule.  The scope is the entire body of the rule (i.e., all
        rule alternatives).
    *   A named item in a rule alternative.  The scope is the entire alternative.
        However, if it appears within or before the said named item, it is a grammar error.
    """

    args: Args

    def __init__(self, name: Token, args: list[Args] = []):
        super().__init__(ObjName(name))
        self.arg_lists = ArgsList(args)
        self.name = ObjName(f"_{self.value}")

    def children(self) -> Iterator[GrammarTree]:
        yield from super().children()
        yield self.value
        yield self.arg_lists

    def pre_init(self) -> None:
        super().pre_init()
        self.src = self.get_src()
        if str(self.value).startswith('_'):
            self.grammar_error(f"identifier {self.value} cannot start with an underscore.")
        if self.isexternal():
            self.type = self.src.type
            self.func_type = self.src.func_type
        else:
            self.type = ProxyType([self.resolved], node=self)

    def isexternal(self) -> bool:
        return str(self.value).isupper()

    def get_src(self) -> ParseSource:
        name = self.value
        if self.isexternal():
            # The name is a type of token.

            # Try a basic token name.
            func_name: str = self.basic_parser_name()
            if func_name:
                return getattr(self.gen, f"parse{func_name}")

            # Try some other token name.
            tok_name: str = self.token_parser_name()
            if tok_name:
                return self.gen.parse_token

            self.grammar_error(f"Token type {name} not recognized.")
        else:
            # The name is a rule, parameter, or variable.
            resolved = self.resolved = self.resolve()
            isrule = isinstance(resolved, Rule)
            return ParseSource(
                self.gen,
                isrule and self.gen.rule_func_name(resolved) or self.value,
                resolved.return_type,
                #resolved.params,
                func_type=(
                    isrule and ParseFunc
                    or ParseFunc
                    ),
                parent=self,
                )

    def external_name(self) -> Optional[ObjName]:
        return self.basic_parser_name() or self.token_parser_name()

    def basic_parser_name(self) -> Optional[ObjName]:
        """ If this is the name of a basic parser method, return the method name. """
        if str(self.value) in ("NAME", "NUMBER", "STRING", "OP", "TYPE_COMMENT", "SOFT_KEYWORD"):
            return ObjName(f"_{self.value}")

    def token_parser_name(self) -> Optional[ObjName]:
        """ If this is the name of a particular token, return the name for parser function. """
        if self.basic_parser_name(): return
        if str(self.value) in self.gen.tokens:
            return ObjName(f"_{self.value}")

    def vars_used(self) -> Iterator[ObjName]:
        """ All variable names used directly in this expression.
        They require being defined in the generated code.
        """
        if self.external_name():
            return
        try: resolved = self.resolve()
        except GrammarError: return
        # The name of a variable or parameter is used.
        if not isinstance(resolved, Rule):
            yield self.value
        if self.arg_lists.arg_lists:
            vis = TargetCodeVisitor(self.arg_lists.arg_lists[0])
            item_names = vis.vars(self.local_env)
            yield from item_names

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        name = self.value
        try: src = self.src
        except AttributeError: return None
        if self.isexternal():
            # The name is a type of token.

            # Try a basic token name.
            if self.basic_parser_name():
                return ParseRecipeExternal(self, None, src, **kwds)

            # Try some other token name.
            else:
                return ParseRecipeExternal(
                    self, None, self.gen.parse_token,
                    f"{self.gen.token_types[name.string]}",
                    **kwds,
                    )

        # The name is a rule, parameter, or variable.
        resolved = self.resolve()
        arg_lists = self.arg_lists
        isrule = isinstance(resolved, Rule)
        islocal = self.local_env.lookup(name) is resolved
        assert bool(isrule) != bool(islocal)
        # Validate the resolved parameters vs. the arguments.
        # NOW: there are no arguments.  These are now in a Call parent.
        args = NoArgs()
        ####args = arg_lists.arg_lists and arg_lists.arg_lists[0] or NoArgs()
        ####if isrule:
        ####    params = resolved.type.params or Params()
        ####    if not args: args = Args()
        ####else:
        ####    params = resolved.return_type.params

        ####if not args:
        ####    # No arguments, so there must be no parameters.
        ####    if not params.empty:
        ####        self.grammar_error(f"Reference to name {resolved.name.string!r} requires argument list.")
        ####elif params is None:
        ####    self.grammar_error(f"Reference to name {resolved.name.string!r} cannot have argument list.")
        ####elif len(args) != len(params):
        ####    self.grammar_error(f"Reference to name {resolved.name.string!r} requires exactly {len(args)} argument(s).")
        if islocal:
            return ParseRecipeLocal(self, name, src, *args, func_type=ParseFunc, **kwds)
        elif isrule:
            return ParseRecipeRule(self, name, src, args, func_type=ParseFunc, **kwds)
        else:
            return ParseRecipeInline(self, name, src, args, func_type=ParseFunc, **kwds)

    def resolve(self) -> GrammarTree:
        name = self.value
        # Try a local name which is in scope.
        # Could be a parameter in the current rule or an item in some containing Alt.
        item = self.local_env.lookup(name)
        if item:
            return item

        # It could be a later variable in the current Alt.  This is a GrammarError
        if name in self.alt.var_names():
            self.grammar_error(f"Reference to name {name.string!r}, which is not yet parsed.")

        # Try a rule name
        rule = self.grammar.rules.get(name)
        if rule:
            return rule

        self.grammar_error(f"Undefined reference to name {name.string!r}.")

    def show(self) -> str:
        if self.value.string == "ENDMARKER":
            return "$"
        return ''.join((super().__str__(), * (args.show() for args in self.arg_lists.arg_lists)))

    def __str__(self) -> str:
        if self.value.string == "ENDMARKER":
            return "$"
        return ''.join((super().__str__(), * map(str, self.arg_lists.arg_lists)))

    def __repr__(self) -> str:
        return f"<NameLeaf {self.value}>"


class StringLeaf(Leaf):
    """The value is a string literal, including quotes."""
    keyword_re: ClassVar = re.compile(r"[a-zA-Z_]\w*\Z")

    def __init__(self, value: Token):
        super().__init__(value.string)
        if self.is_keyword:
            self.name = ObjName('_keyword')
        else:
            self.name = ObjName('_literal')

    def pre_init(self) -> None:
        super().pre_init()
        src = self.src = (self.keyword_src()
                          or self.soft_keyword_src()
                          or self.literal_src()
                          )
        self.type = src.type

    @functools.cached_property
    def val(self) -> str:
        return ast.literal_eval(self.value)

    @functools.cached_property
    def is_keyword(self) -> bool:
        return bool(self.keyword_re.match(self.val))

    def keyword_src(self) -> Optional[ParseSource]:
        """ If this is a regular keyword, return name of parser function. """
        if self.is_keyword and self.value.endswith("'"):
            return self.gen.parse_keyword

    def soft_keyword_src(self) -> Optional[ParseSource]:
        """ If this is a soft keyword, return name of parser function. """
        if self.is_keyword and not self.value.endswith("'"):
            return self.gen.parse_soft_keyword

    def literal_src(self) -> Optional[ParseSource]:
        """ If this is not a keyword, return name of parser function. """
        if not self.is_keyword:
            if self.val in self.gen.exact_tokens:
                return self.gen.parse_token
            assert len(self.val) == 1, f"{self.value} is not a known literal"
            return self.gen.parse_char

    def make_parse_recipe(self, **kwds) -> ParseRecipe:

        val = self.val
        if self.keyword_src():
            name = "_keyword"
            arg = f"{self.gen.keywords[val]}"
            comment = f"keyword = '{val}'"
        elif self.soft_keyword_src():
            name = "_keyword"
            arg = f"{self.value}"
            comment = f"keyword = '{val}'"
        elif self.literal_src():
            name = "_literal"
            type = self.gen.exact_tokens.get(val, val)
            arg = f"{type!r}"
            comment = f"keyword = '{val}'"

        return ParseRecipeExternal(self, name, self.src, arg,
            comment=comment,
            func_type=ParseFunc,
            **kwds,
            )

    def __repr__(self) -> str:
        return f"StringLeaf({self.value!r})"


## Other GrammarTree classes that aren't parsed from the grammar...

class Attr(GrammarTree):
    """ Describes an attribute of another node, for display purposes.
    A str as the node is considered the attribute value.
    Otherwise, a node without the given attribute is considered False.
    """
    def __init__(self, node: GrammarTree, attr: str):
        super().__init__()
        self.node = node
        self.attr = attr

    def validate(self, visited: set = set(), level: int = 0) -> None:
        pass

    def __iter__(self) -> Iterator:
        if False: yield

    def __bool__(self) -> bool:
        return (
            type(self.node) is str
            or hasattr(self.node, self.attr)
            )

    def __repr__(self) -> str:
        if type(self.node) is str:
            value = self.node
        else:
            try: value = getattr(self.node, self.attr)
            except AttributeError: value = '<no value>'
        return f'{self.attr} = {value!r}'


from pegen.parse_recipe import (
    ParseRecipe,
    ParseRecipeExternal,
    ParseRecipeLocal,
    ParseRecipeRule,
    ParseRecipeInline,
    ParseSource,
    ParseFuncType,
    ParseFunc,
    ParseTest,
    ParseTrue,
    ParseData,
    ParseNone,
    ParseVoid,
    )

from pegen.target_code import Code, NoCode, NoValueCode, ValueCode, TargetCodeVisitor

from pegen.expr_type import Type, ObjType, ProxyType, FuncType, NoType, Params, NoParams

