""" grammar module. """

from __future__ import annotations

from abc import abstractmethod
import collections
import itertools
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

    2. All GrammarTree objects are constructed, from bottom up, with the Grammar at the top.
        This is done by the The parser's start() method.
        Each tree constructor is given whatever subtrees it needs, which have been constructed already.
        The constructors are actions in the Python or C parser script.
        The constructor copies the parser's current start and end locations into the tree.locations member.
        Now, each subtree knows its children but does not know its parent.

    3. Create a ParserGenerator (C or Python), and store it in the GrammarTree's thread-local object.

    4. The Grammar object's initialize() method is called.  This completes initialization of all
        subtrees in the Grammar tree, by calling initialize() on them, going top-down.
        Class-specific operations are performed by pre_init() and post_init() methods.
        See the initialize() method for more details.

    5. Parse recipes, where supported, are created for all trees, going bottom_up.
    """
    name: str = None
    parent: GrammarTree = None      # The Node this Node is a part of.
    gen: ParserGenerator
    showme: ClassVar[bool] = True
    # The type resulting from parsing, as a target code string.
    # Sources, in priority order:
    #   1. Tree constructor for some classes like Rule.
    #   2. Type of parent, for some classes such as Rhs, Alt.  In pre_init().
    #   3. Value type of parse_recipe.
    #   4. Type of Alt item(s), if they are all the same.
    #   5. Target language default.

    _thread_vars: ClassVar[ThreadVars]

    def __init__(self, name: str = None, **kwds):
        super().__init__()
        if name: self.name = name
        try: self.locations = self._thread_vars.parser._locations
        except AttributeError: self.locations = None

    def children(self) -> Iterable[GrammarTree]:
        if hasattr(self, 'parse_recipe'): yield self.parse_recipe

    def __iter__(self) -> Iterable: return ()

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

    #@property
    #def gen(self) -> ParserGenerator:
    #    return self._thread_vars.gen

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
        """ Complete part 3 of the grammar tree initialization, for this tree and all descendants,
            given the tree's parent (the top Grammar tree has no parenta)
        1. Link tree to its parent.  This makes certain properties available,
            such as self.gen, self.rule, and self.alt.
        2. Initialize things which depend on the parent, and are common to all tree classes.
        3. Call self.pre_init().  Class-dependent code.  It can depend on steps 1 to 3 above
            being performed already for all the ancestors.
        4. Call child.intialize(self) for all the child subtrees.  Note, some nodes
            from child.children() are not actually subtrees and are ignored.
        5. Create the parse recipe for the tree, if its class supports it.
        6. Call self.post_init().  Class-dependent code.
            This can depend on initialize() being performed already for all the descendants.
        """
        self.parent = parent
        if parent:
            self.local_env = parent.local_env
        #self.message(f'{type(self).__name__} = {self.show()}')
        #if self.local_env.owner is not self:
        #    self.local_env.dump(self)
        self.pre_init()
        # Initialize all the descendnts.
        for child in self.children():
            #if child.parent is not self: continue
            try: child.initialize(self)
            except GrammarError as e:
                if child.rule is not child:
                    raise e
                child.error = True

        self.post_init()

    def pre_init(self) -> None:
        """ Class-specific initialization, after all ancestors have been pre_initialized. """
        self.gen = self._thread_vars.gen

    def post_init(self) -> None:
        """ Class-specific initialization, after all descendants have been initialized. """
        pass

    def _gen(self) -> ParserGenerator:
        """ Get the current generator, for cases where pre_init() has not been called. """
        try: return self.gen
        except AttributeError:
            return self._thread_vars.gen

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
        return self.gen.uniq_name(self, self.assigned_name)
        #anc = self.parent
        #return f"{anc.uniq_name()}_{assigned_name or self.name}"

    WalkResult = typing.NewType('WalkResult', 'Tuple[GrammarTree, GrammarTree, int]')

    def walk(self, parent: GrammarTree = None,
             level: int = 0, *,
             reversed: bool = False,
             ) -> Iterable[WalkResult]:
        """ Iterates top-down yielding (parent, child, level), starting with (given parent, self). """
        def recursive(child: GrammarTree, parent: GrammarTree, level: int
                      )-> Iterable[WalkResult]:
            if not reversed:
                yield parent, child, level
            for child2 in child.children():
                if child2:
                    yield from recursive(child2, child, level + 1)
            if reversed:
                yield parent, child, level
        yield from recursive(self, parent, level)

    def walk_fun(self, pre: Callable[[GrammarTree, GrammarTree, int], None] = None,
                 *, post: Callable[[GrammarTree, GrammarTree, int], None] = None,
                 parent: GrammarTree = None,
             ) -> None:
        """ Iterates top-down making callbacks (parent, child), starting with (self.parent, self).
        pre() is called before descending into children, and post() is called after.
        """
        def recursive(child: GrammarTree, parent: GrammarTree, level: int = 0
                      )-> None:
            if pre:
                pre(parent, child, level)
            for child2 in child.children():
                if child2:
                    recursive(child2, child, level + 1)
            if post:
                post(parent, child, level)
        recursive(self, parent or self.parent)

    def validate(self) -> None:
        """ Raise an exception if anything is incorrect.
        Also validates all recipes.
        Subclass should call super().validate().
        """
        if hasattr(self, 'parse_recipe'):
            self.parse_recipe.validate()

    def chk_type(self, obj: Any, typ: type, opt: bool = False) -> None:
        if opt and obj is None: return
        assert isinstance(obj, typ), f"Expected {typ.__name__}, not {obj!r}."

    def full_name(self) -> str:
        res = self.name.string
        self.type.dump()
        if self.type.val_type: res += f' [{self.type.val_type}]'
        if self.type.params: res += f' {self.type.params}'
        return res

    def func_params(self) -> str:
        """ The string for the parameters of a function, if this is callable.
        Otherwise empty string.
        """
        if not self.type.params: return ""
        return f"({self.type.params.in_func()})"

    def grammar_error(self, msg: str) -> NoReturn:
        e = GrammarError(
            f"Grammar error: rule {self.rule.name.string!r}, {msg}  At line {self.locations.showrange()}.")
        self.grammar._errors[self.rule.name.string].append(e)
        raise e

    def show(self, leader: str = ''):
        """ Same as str(self), but extra lines are indented with the leader.
        Override in subclass to implement this.
        """
        return str(self)

    def itershow(self) -> Iterable[GrammarTree]:
        return iter(self)

    def attrs(self) -> Iterator[Attr]:
        """ Enumerate attributes of this node.
        Subclass with more attributes calls the super().attrs() first.
        """
        if False:
            yield None

    def message(self, msg: str, indent: str = '... ', level: int = None) -> None:
        """ Print message with leader for the nesting depth. """
        lines = msg.splitlines()
        if level is None: level = self.depth()
        print(f"{'    ' * level}{lines[0]}")
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
        gen = getattr(self, 'gen', None)
        if gen:
            if gen.brk(delta):
                return True
            if gen.brk_token(self):
                return True
        return False


class OptVal(GrammarTree, Generic[ValueType]):
    """ Wrapper around a parse result for an Opt expression.
    The parse results in either () or (result,).  OptVal behaves the same way.
    OptVal.opt attribute is either None or result.
    """
    opt: ValueType | None
    true: bool

    def __init__(self, opt: Sequence[ValueType] = ()):
        assert len(opt) <= 1
        self.true = true = bool(opt)
        if true:
            self.opt = opt[0]
        else:
            self.opt = None

    def __bool__(self) -> bool:
        return self.true

    def __iter__(self) -> Iterable[ValueType]:
        if self.true: yield self.opt

    def __str__(self) -> str:
        return str(self.opt)

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
    always_true: bool = False
    local_src: bool = False
    type: Type                              # Type of the parse function.
    func_type: ParseFuncType = None
    assigned_name: ObjName = None           # Set by a VarItem parent.
    has_result: bool = True

    def __init__(self,
        name: Optional[str] = None,
        typ: Optional[Type] = None,
        func_type: ParseFuncType = None,
        **kwds,
        ):

        self.chk_type(typ, Type,  opt=True)
        super().__init__(**kwds)
        if name: self.name = name
        elif self.name: self.name = ObjName(self.name)
        self.type = typ or FuncType(NoCode())
        #self.type = typ or NoType().make_func()
        self.type.node = self
        self.func_type = func_type or ParseFunc

    def children(self) -> Iterator[Rhs]:
        if self.name: yield self.name
        if self.type is not None: yield self.type

    def pre_init(self) -> None:
        super().pre_init()

    @property
    def val_type(self) -> Type:
        return self.type.val_type

    @property
    def return_type(self) -> Type:
        return self.type.return_type

    @return_type.setter
    def return_type(self, typ: Type) -> None:
        self.chk_type(typ, Type)
        self.type.return_type = typ

    #@property
    #def func_type(self) -> ParseFuncType:
    #    return self.parse_recipe.func_type

    def vars_used(self) -> Iterator[ObjName]:
        """ All variable names used directly in this expression.
        They require being defined in the generated code.
        """
        if False: yield

    def decl_var(self, name: str = None) -> str:
        """ The declaration of a name (without the value), as a variable.
        If callable, the variable is a function pointer, including function parameters.
        """
        if name is None: name = self.name
        typ = self.type
        if not typ.callable: return f"{typ} {name}"
        # It's a function.
        return f"{typ} (*{name}) ({typ.params.in_func()})"

    def attrs(self) -> Iterator[Attr]:
        yield Attr(self, 'assigned_name')

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
    """ Denotes a missing name, only applies to a Param. """
    def __init__(self):
        super().__init__('')

    def __bool__(self) -> bool:
       return False


class TypedName(GrammarTree):

    assigned_name: ObjName = None           # Set by a VarItem parent.

    def __init__(self,
        name: Optional[str | Token | ObjName],
        typ: OptVal[Type] | Type = OptVal(),         # target language default type if not given to constructor.
        ):
        super().__init__()
        if isinstance(name, (str, Token)):
            name = ObjName(name)
        self.chk_type(name, ObjName)
        self.name = name
        if isinstance(typ, OptVal):
            typ = typ.opt
        if typ is None:
            typ = NoType()
        self.chk_type(typ, Type)
        self.type = typ
        #self.val_type = typ.val_type
        #self.params = typ.params
        #self.chk_type(self.val_type, Code)

    def set_name(self, name: TypedName) -> None:
        """ Initialize from another name.  An alternative to supplying arguments to constructor. """
        self.name = name.name
        self.type = name.type

    def children(self) -> Iterator[GrammarTree]:
        yield from super().children()
        if self.name: yield self.name
        if self.type: yield self.type

    @property
    def return_type(self) -> Type:
        return self.type.return_type

    @return_type.setter
    def return_type(self, typ: Type) -> Type:
        self.chk_type(typ, Type)
        self.type.return_type = typ

    @property
    def callable(self) -> bool:
        return self.type.callable

    @property
    def val_type(self) -> bool:
        return self.type.val_type

    def pre_init(self) -> None:
        super().pre_init()
        x = 0
        # Resolve
        #if not self.type:
        #    self.type = self.gen.default_type()

    # TODO: This is only valid for C syntax.
    def typed_name(self, name: str = '') -> str:
        """ What is generated for the type of this TypedName, including optional name.
        This TypedName may have its own parameters, which are generated recursively.
        """
        parts: list[str] = []
        base_type = str(self.type.val_type)
        if base_type: parts.append(base_type)
        if self.callable:
            # This node is a callable type.
            subtypes = [self.gen.typed_name(param) for param in self.type.params]
            parts.append(f'(*{name}) ({", ".join(subtypes)})')
        else:
            if name: parts.append(str(name))
        p = ' '.join(parts)
        return p

    def full_type(self) -> str:
        """ Complete type name (without the name), including function parameters. """
        type = self.type
        if not type.callable: return type
        # It's a function.
        return f"{type} (*) ({type.params.in_func()})"

    def decl_var(self, name: str = None) -> str:
        """ The declaration of a name (without the value), as a variable.
        If callable, the variable is a function pointer,including function parameters.
        """
        if name is None: name = self.name
        typ = self.type
        if not typ.callable: return f"{typ} {name}"
        # It's a function.
        return f"{typ} (*{name}) ({typ.params.in_func()})"

    # TODO: Separate version for Python syntax.
    def decl_func(self, fwd: bool = False) -> str:
        """ The declaration of a callable name (without the value), as a function.
        The variable is a function (not a function pointer), including function parameters.
        """
        params = self.type.params
        assert params and self.name
        typ = str(self.type)
        name = self.name
        if fwd:
            nparams = len(params)
            if typ == 'ParseStatus' and nparams == 2:
                return f"ParseFunc {name}"
            if fwd and typ == 'ParseStatus' and nparams == 1:
                return f"ParseTest {name}"
            if fwd and typ == 'void' and nparams == 2:
                return f"ParseTrue {name}"
        return f"{typ} {name}({params.in_func()})"

    def attrs(self) -> Iterator[Attr]:
        yield from super().attrs()
        yield Attr(self, 'name')
        yield Attr(self, 'type')

    def __str__(self) -> str:
        if not SIMPLE_STR and (self.type.params or self.type):
            res = self.name
            if self.type.params: res += f'({self.type.params})'
            if self.type: res += f'[{self.type}]'
            return res
        else:
            return self.name.string

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}, {self.type}>"


class Grammar(GrammarTree):
    rule: Rule = None
    alt: Alt = None
    _errors: Mapping[[str], GrammarError]

    def __init__(self, rules: Iterable[Rule], metas: Iterable[Tuple[str, Optional[str]]]):
        super().__init__()
        self.rules = {rule.name: rule for rule in rules}
        self.metas = dict(metas)
        self._errors = collections.defaultdict(list)
        self._type_deps = collections.defaultdict(set)

    @property
    def grammar(self) -> Grammar:
        return self

    def initialize(self) -> None:
        self.local_env = LocEnv(self)
        super().initialize()

    #def pre_init(self) -> None:

    def post_init(self) -> None:
        ### Find any cycles in Rule return type dependencies.
        ### This will avoid infinite recursion in Rule.return_type() method.
        ### Must do this after initialize().
        ##deps: Mapping[Rule, Set[Rule]]
        ##deps = collections.defaultdict(set)
        ##def dep_cb(parent: GrammarTree, child: GrammarTree, level: int) -> None:
        ##    rulename = isinstance(child, Rule) and f', {child.name}' or ''
        ##    print(f"{'  ' * level}{type(parent).__name__}, -> {type(child).__name__}{rulename}")
        ##    if isinstance(child, Rule):
        ##        if parent is not self:
        ##            deps[child.rule].add(child)
        ##self.walk_fun(dep_cb)

        ### Display all, or some, nodes in the tree.
        ### Must do this after initialize().
        ##def msg(parent: GrammarTree, child: GrammarTree, level: int) -> None:
        ##    if isinstance(child, Grammar): return
        ##    #if isinstance(child, (Code, ObjType, FuncType, ObjName, ParseRecipe, TypedName, Arg, Args, Params)): continue
        ##    #if child.val_type: continue
        ##    child.message(f"{child!r} -- {getattr(child, 'type', None)}", level=level)
        ##self.walk_fun(msg)

        # Use this only when running the C parser
        #r1 = self.rules[ObjName('r1')]
        #t = r1.return_type
        #r2 = self.rules[ObjName('r2')]
        #t = r2.return_type
        # Make the parse recipes for the tree, bottom-up.
        def make_recipe(parent: GrammarTree, child: GrammarTree, level: int) -> None:
            make_parse_recipe = getattr(child, 'make_parse_recipe', None)
            if make_parse_recipe:
                recipe = child.parse_recipe = make_parse_recipe()
                if recipe.node is child:
                    recipe.initialize(child)

        self.walk_fun(post=make_recipe)

        visited: set[GrammarTree] = set()
        def val(_: GrammarTree, child: GrammarTree, level: int) -> None:
            if child in visited: return
            visited.add(child)
            child.validate()

        self.walk_fun(val)

        if self._errors:
            for name in sorted(self._errors):
                print(self._errors[name])
        else: print('Validate() passed.')

    def update_type_defs(self, dep: NameLeaf) -> None:
        # Record a NameLeaf which resolves to a Rule.
        self._type_deps[dep.rule].add(dep)

    def validate(self, visited: set = set(), level: int = 0) -> None:
        for rule in dict(self.rules).values():
            try: rule.validate(visited, level + 1)
            except GrammarError as e:
                rule._error = True

    def walk_fun(self, pre: Callable[[GrammarTree, GrammarTree, int], None] = None,
                 *, post: Callable[[GrammarTree, GrammarTree, int], None] = None,
             ) -> None:
        """ Iterates top-down calling pre and/or post (parent, child, level), starting with
        (given parent, self).
        Any grammar error raised during a rule will delete that rule and then continue.
        """
        for rule in dict(self.rules).values():
            print(rule)
            try: rule.walk_fun(pre, post=post, parent=self)
            except GrammarError as e:
                rule._error = True

    def children(self) -> Iterator[Rule]:
        yield from super().children()
        yield from dict(self.rules).values()

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


# Global flag whether we want actions in __str__() -- default off.
SIMPLE_STR = True


T = TypeVar('T')

class TrueHashableList(Generic[T]):
    """ Acts like List[T] except that it is hashable and always true. """
    def __init__(self, items: Sequence[T] = [], **kwds):
        super().__init__(**kwds)
        self._items = list(items)

    def __hash__(self) -> int: return id(self)

    def __bool__(self) -> bool: return True
    def __len__(self): return len(self._items)
    def __iter__(self):
        yield from super().__iter__()
        yield from iter(self._items)
    def __getitem__(self, i): return self._items[i]
    def __add__(self, other: Self) -> List[T]:
        return self._items + other._items

    def children(self):
        yield from super().children()
        yield from self._items

    def __str__(self) -> str:
        return f'[{", ".join([str(x) for x in self._items])}]'


class Meta:
    def __init__(self, name: str, val: str = None):
        super().__init__()
        self.name = name
        self.val = val

    def __iter__(self) -> Iterable:
        yield self.name
        yield self.val


class Rule(ParseExpr):
    max_local_vars: int = 0         # Greatest number of local names for any node in the tree.
    left_recursive: bool = False
    leader: bool = False            # Implies left_recursive.
    _error: bool = False

    def __init__(self, rulename: TypedName, rhs: Rhs, memo: Optional[str] = None):
        typ = rulename.type
        if not typ.callable:
            typ = FuncType(typ.val_type, Params())
        typ = ProxyType((rhs,), typ.params)
        super().__init__(rulename.name, typ)
        #super().__init__(rulename.name, rulename.type, params)
        self.rhs = rhs
        self.return_type = rulename.return_type
        self.memo = bool(memo)

    def pre_init(self) -> None:
        super().pre_init()
        if len(self.type.params):
            for param in self.type.params:
                self.add_local_name(param, param.name)

    def post_init(self) -> None:
        # Trickle up the val type
        if not self.return_type:
            self.return_type = self.rhs.return_type
        pass
        #self.message(f"Rule {self.name} # vars = {self.max_local_vars}")

    def need_local_vars(self, n: int) -> None:
        """ At least this many local variables may be needed. """
        if n > self.max_local_vars:
            self.max_local_vars = n

    @property
    def rule(self) -> Rule:
        return self

    def uniq_name(self, assigned_name: str = None) -> str:
        """ A name which is unique among all tree objects. """
        return f"_{self.name}"

    def __str__(self) -> str:
        return f"{self.full_name()}: {self.rhs}"

    def show(self, leader: str = ''):
        res = str(self)
        if len(res) < 88:
            return res
        lines = [res.split(":")[0] + ":"]
        lines += [f"{leader}| {alt}" for alt in self.rhs]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Rule {self.name}, {self.return_type}, {self.rhs}>"

    def children(self) -> Iterator:
        yield from super().children()
        yield self.rhs

    def flatten(self) -> Rhs:
        # If it's a single parenthesized group, flatten it.
        rhs = self.rhs
        if (
            len(rhs.alts) == 1
            and len(rhs.alts[0]) == 1
            and isinstance(rhs.alts[0][0], Group)
        ):
            rhs = rhs.alts[0][0].rhs
        return rhs

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        if self.left_recursive and self.leader:
            parser = self.gen.parse_rule_recursive
            memo = True
        elif self.memo:
            parser = self.gen.parse_rule_memo
            memo = True
        else:
            parser = self.gen.parse_rule
            memo = False

        if len(self.type.params) and memo:
            # There's no code (yet) to memoize a rule with parameters.
            # Someday, when there is such code, this test will be removed.
            try: self.grammar_error(
                f"Rule with parameters may not be memoized or left-recursive leader.")
            except: pass
            self._error = True

        return self.gen.rule_recipe(self, parser)

    @property
    def param_names(self) -> List[str]:
        return [param.name for param in self.params]

    @classmethod
    def simple(cls, name: str, params: Params, *args, **kwds) -> Rule:
        # Make Rule from just the name and any parameters.
        return cls(TypedName(name, params=params), *args, **kwds)


class Rhs(ParseExpr):

    name = '_rhs'
    nested: bool = False                    # Set True on instance if part of a Group.

    def __init__(self, alts: Sequence[Alt]):
        super().__init__(typ=ProxyType(alts))
        self.alts = alts
        for i, alt in enumerate(alts, 1):
            if len(alts) == 1:
                name = '_alt'
            else:
                name = f"_alt{i}"
            alt.name = ObjName(name)

    def __iter__(self):
        yield from self.alts

    def children(self):
        yield from super().children()
        yield from self.alts

    def pre_init(self) -> None:
        super().pre_init()
        for alt in self:
            alt.return_type = self.return_type

    def post_init(self) -> None:
        if not self.return_type:
            # Use common type of the alts if they are all the same.
            msg: list[str] = [
                f"{self.uniq_name()} {self} -- making return type from alts..."]
            return_type = self.return_type
            if return_type:
                msg.append(f"{return_type}")
            else:
                msg.append('mixed or missing types.')
                for alt in self:
                    msg.append(f"{alt} -> {alt.return_type or 'missing'}")
                    if not alt.return_type and alt.return_type.node:
                        msg.append(f"from node {alt.return_type.node}")
            self.message('\n'.join(msg))

    def make_parse_recipe(self, name: str = None, dflt_name: str = '_rhs', **kwds) -> ParseRecipe:

        # Args for the recipe are the Alts with generated names.
        alt_names: List[str]
        if len(self.alts) == 1:
            alt_names = ['_alt']
            parser_name = self.gen.parse_alt
        else:
            alt_names = [f"_alt{i}" for i, _ in enumerate(self.alts, 1)]
            parser_name = self.gen.parse_alts

        descr_name, extra = self.gen.gen_rhs_descriptors(self, alt_names)

        return ParseRecipeInline(
            self, name or dflt_name, parser_name, descr_name,
            params=Params(),
            extra=extra,
            inlines=list(self),
            func_type=ParseFunc,
            typ=self.type,
            **kwds,
            )

    def show(self, leader: str = ''):
        res = str(self)
        if len(res) < 88:
            return res
        lines = [f"{leader}| {alt}" for alt in self]
        return "\n".join(lines)

    def __str__(self) -> str:
        return " | ".join(str(alt) for alt in self.alts)

    def __repr__(self) -> str:
        return f"<Rhs {' | '.join((map(str, self.alts)))}>"

    @property
    def can_be_inlined(self) -> bool:
        if len(self) != 1 or len(self[0].items) != 1:
            return False
        # If the alternative has an action we cannot inline
        if getattr(self[0], "action", None) is not None:
            return False
        return True


class Arg(GrammarTree):
    inline: GrammarTree = None

    def __init__(self, code: Code = None, inline: GrammarTree = None, gen: ParserGenerator = None):
        super().__init__(code)
        self.code = code
        if inline: self.inline = inline

    def children(self) -> Iterable[GrammarTree]:
        yield self.code

    def show_typed(self, typ: TypedName) -> str:
        """ Name of this Arg, possibly coerced to given type. """

        result = str(self)
        # Check for a type cast needed.
        if self.inline:
            my_type: TypedName = self.inline.parse_recipe.func_name
            my_type_name = my_type.full_type()
            type_name = str(typ.type)
            if my_type_name != type_name:
                if type_name == 'ParseFunc*' and my_type_name == 'ParseStatus (*) (Parser* _p, ParseResultPtr* _ppRes)':
                    pass
                elif type_name == 'ParseTest*' and my_type_name == 'ParseStatus (*) (Parser* _p)':
                    pass
                elif type_name == 'ParseTrue*' and my_type_name == 'void (*) (Parser* _p, ParseResultPtr* _ppRes)':
                    pass
                else:
                    result = typ.gen.cast(result, typ)
        return result

    def __str__(self) -> str:
        return self.code and str(self.code) or self.inline.parse_recipe.func_name.name.string

    def __repr__(self) -> str:
        if self.inline and self.name:
            return f"<Arg {self.name} = {self.inline!r}>"
        elif self.name:
            return f"<Arg {self.name}>"
        else:
            return f"<Arg {self.inline}>"


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
        self.show()

    def show(self, *after_args: str) -> str:
        """ For display of an item following the name, including parens.
        Includes trailing comma for a single argument.
        """
        args = ', '.join([*after_args, *(repr(arg) for arg in self[:])])
        if len(args) == 1: args += ','              # Trailing comma for a single arg.
        return f'({args})'

    def show_typed(self, params: Params) -> str:
        """ String with the arg names,
        but some of them may be coerced to the corresponding param type.
        """
        if params.empty: return ''
        args = list(
            arg.show_typed(param)
            for arg, param in zip(self[:], params[:])
            )
        return f"({', '.join(args)})"

    def __str__(self) -> str:
        res = ", ".join(str(arg) for arg in self)
        res1 = '<' + res + '>'
        return res1

    def __repr__(self) -> str:
        return f"<Args {self}>"


class NoArgs(Args):
    empty: bool = True

    def show(self, *after_args: str) -> str:
        """ For display of an item following the name, including parens.
        Includes trailing comma for a single argument.
        """

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return '<no args>'

    def __str__(self) -> str:
        return ''


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
        super().__init__(items)
        self.action = action
        item_names = []
        for item in items:
            self.chk_type(item, VarItem)
            if item.var_name:
                item_names.append(item.var_name)
            item.assign_name(*items)

        # Check unique item variable names.
        unique_names = set(item_names)
        if len(unique_names) < len(item_names):
            self.grammar_error(f'Variable names {item_names} must be different.')

    def children(self) -> Iterable[GrammarTree]:
        yield from super().children()
        if self.action: yield self.action

    @property
    def alt(self) -> Alt:
        return self

    def items(self) -> list[ParseExpr]:
        return self._items

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

    def pre_init(self) -> None:
        super().pre_init()
        if self.var_names():
            self.remove_local_names(*self.var_names())
        for item in self.items():
            if item.var_name:
                self.add_local_name(item.item, item.var_name.name)
        ret_action: self.Action
        if self.gen.skip_actions:
            ret_action = self.gen.dummy_action()
        elif self.action:
            ret_action = self.gen.action(self)
        else:
            ret_action = self.gen.default_action(self)
        self.action_expr = ret_action.expr
        if not self.return_type:
            self.type = ret_action.type
            self.chk_type(self.type, Type)

    def post_init(self) -> None:
        super().post_init()
        names = list(self.local_env)        # All variable names, old and new.
        self.codes = TargetCodeVisitor(self)

        # Analyze names used by each item in turn.
        # A name which is defined in this or a later item is a GrammarError.
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

    @functools.cache
    def var_names(self):
        return tuple(item.var_name.name for item in self.items() if item.var_name)

    @functools.cache
    def all_vars(self):
        """ All assigned variables, including for anonymous items, which have a parse result. """
        return tuple(item for item in self if item.item.has_result)

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeInline(self,
            None, self.gen.parse_alt,
            typ=self.type,
            params=Params(),
            func_type=ParseFunc,
            extra=lambda: self.gen.gen_alt(self),
            inlines=[item.item for item in self.items()],
            comment=str(self),
            **kwds,
            )

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
        super().__init__()
        assert item is not None
        self.var_name = name
        self.name = name and name.name or item.name
        self.item = item
        if name and name.type:
            target = name
        else:
            target = self.item
        self.type = ProxyType((target,))

    def pre_init(self) -> None:
        super().pre_init()

    def assign_name(self, *sibs: VarItem) -> None:
        # Set assigned name.
        name: ObjName = self.var_name and self.var_name.name
        if not name:
            # Add a suffix to given name if it appears earlier in parent Alt.
            origname = name = self.name
            counter = itertools.count(1)
            for sib in sibs:
                if sib is self: break
                if sib.assigned_name == name:
                    # Earlier duplicate.  Add a suffix and keep looking for more.
                    name = self.name = ObjName(f"{origname}_{next(counter)}")
        self.assigned_name = self.item.assigned_name = name

    #def post_init(self) -> None:
    #    self.type = (
    #        self.var_name and self.var_name.type
    #        or ProxyType((self.item,))
    #        #or self.item.type
    #        )

    #    self.chk_type(self.type, Type)

    def validate(self) -> None:
        super().validate()
        if self.var_name:
            if not self.item.has_result:
                self.item.grammar_error(
                    f"Item {self.name} with variable name must have a parse result.")

    def vars_used(self) -> Iterator[ObjName]:
        """ All variable names used directly in this expression.
        They require being defined in the generated code.
        """
        return self.item.vars_used()

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return self.item.parse_recipe
        #return self.item.make_parse_recipe(**kwds)

    def attrs(self) -> Iterator[Attr]:
        yield from super().attrs()
        yield Attr(self, 'assigned_name')

    def uniq_name(self, assigned_name: str = None) -> str:
        return self.parent.uniq_name()

    def show(self, show_var: bool = True) -> str:
        if self.var_name and show_var:
            return f"{self.var_name.name}={self.item}"
        else:
            return str(self.item)

    def __str__(self) -> str:
        return self.show()

    def __repr__(self) -> str:
        return f"<VarItem {self}>"

    def children(self) -> Iterator[Item]:
        yield from super().children()
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

    def post_init(self) -> None:
        self.type = self.node.return_type.make_func()

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeInline(
            self,
            '_forced',
            self.gen.parse_forced,
            (None, self.node),
            f'"{self.gen.str_value(str(self.node))}"',
            typ=self.type,
            **kwds
            )

    def __str__(self) -> str:
        return f"&& {self.node}"

    def __repr__(self) -> str:
        return f"<Forced {self.node}>"

    def children(self) -> Iterator[Plain]:
        yield from super().children()
        yield self.node


class Lookahead(Item):
    name = '_lookahead'
    has_result: bool = False

    def __init__(self, node: Item, positive: bool):
        super().__init__()
        self.node = node
        node.name = ObjName('_atom')
        self.positive = positive

    def post_init(self) -> None:
        self.type = self.node.type

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeInline(
            self, '_lookahead', self.gen.parse_lookahead,
            self.gen.bool_value(self.positive),
            Arg(inline=self.node),
            params=Params(),
            typ=self.type,
            func_type=ParseTest,
            **kwds)

    def __str__(self) -> str:
        return f"{'!&'[self.positive]}{self.node}"

    def children(self) -> Iterator[Plain]:
        yield from super().children()
        yield self.node


class PositiveLookahead(Lookahead):
    def __init__(self, node: Plain):
        super().__init__(node, True)

    def __repr__(self) -> str:
        return f"<Pos Lkahead {self.node}>"


class NegativeLookahead(Lookahead):
    def __init__(self, node: Plain):
        super().__init__(node, False)

    def __repr__(self) -> str:
        return f"<Neg Lkahead {self.node}>"


class Cut(Item):
    has_result: bool = False

    def __init__(self):
        super().__init__()
        self.name = ObjName('_cut')

    def pre_init(self) -> None:
        super().pre_init()
        self.type = ObjType(self.gen.null_type())

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeExternal(
            self, "_cut", self.gen.parse_cut,
            **kwds, )

    def __str__(self) -> str:
        return "~"

    def __repr__(self) -> str:
        return "<Cut>"


class Atom(Item):
    """ Common base class for a Primary or some combinations of other Atoms.
    """
    pass


class Seq(Atom):
    """ An Atom whose par1se result is a sequence of parse results of given element. """

    no_local_recipe: bool = False
    no_local_recipe: bool = False

    def __init__(self, elem: Atom, **kwds):
        super().__init__(**kwds)
        self.elem = elem

    def pre_init(self) -> None:
        super().pre_init()

    def post_init(self) -> None:
        self.src = self.recipe_src()(self.elem)
        self.type = self.src.return_type.make_func()
        pass

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return ParseRecipeInline(
            self, self.name, self.src,
            *self.gen.sequence_recipe_args(self, *self.parse_args()),
            typ=self.type,
            params=Params(),
            func_type=self.always_true and ParseTrue or ParseFunc,
            **kwds)


class Opt(Seq):
    always_true: bool = True
    name: str = '_opt'

    def recipe_src(self) -> ParseSource: return self.gen.parse_opt

    def parse_args(self) -> Tuple[str, str]:
        return ()

    def __str__(self) -> str:
        return f"{self.elem}?"

    def __repr__(self) -> str:
        return f"<Opt {self}>"

    def children(self) -> Iterator[Item]:
        yield from super().children()
        yield self.elem

    def itershow(self) -> Iterator[Item]:
        if type(self.elem) is Rhs:
            yield from self.elem
        else:
            yield self.elem


class Repeat(Seq):
    """Shared base class for x* and x+."""

    name = '_repeat'

    def children(self) -> Iterator[Plain]:
        yield from super().children()
        yield self.elem

    def recipe_src(self) -> ParseSource: return self.gen.parse_repeat

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.elem!r})>"


class Repeat0(Repeat):
    always_true: bool = True

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

    def __init__(self, separator: Plain, elem: Plain):
        super().__init__(elem)
        self.separator = separator
        separator.name = ObjName('_sep')

    def children(self) -> Iterator[Plain]:
        yield from super().children()
        yield self.elem
        yield self.separator

    def recipe_src(self) -> ParseSource: return self.gen.parse_gather

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.separator!r}, {self.elem!r})>"


class Gather0(Gather):
    always_true: bool = True

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

    def __init__(self, rhs: Rhs, typ: OptVal[Type] = OptVal()):
        super().__init__(typ=ProxyType((rhs,)))
        self.rhs = rhs
        self.chk_type(typ, OptVal)
        rhs.nested = True
        typ = typ and typ.opt.make_func() or NoType()
        self.return_type = typ.return_type

    def children(self) -> Iterator:
        yield from super().children()
        yield self.rhs

    def post_init(self) -> None:
        self.type = self.rhs.type

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return self.rhs.parse_recipe
        #func_name = self.rhs.parse_recipe.func_name
        #src = ParseSource(
        #    func_name.name,
        #    self.rhs.type,
        #    #self.rhs.parse_recipe.params,
        #    func_type=ParseFunc.inline,
        #    )
        #return ParseRecipeInline(self, self.name, src,
        #    inlines=[self.rhs],
        #    params=Params(),
        #    func_type=ParseFunc,
        #    **kwds)

    def __str__(self) -> str:
        return f"({self.rhs.__str__()})"

    def __repr__(self) -> str:
        return f"<Group {self.rhs})"


class OptGroup(Primary, Opt):
    """ A lowest level parsed expression which wraps an arbitrary ParseExpr in square brackets.
    Same as an ordinary Group, except that the result is optional, and returns a sequence,
    either containing the parsed result or an empty sequence.
    Parsing an OptGroup always succeeds.
    """

    always_true: bool = True
    name = '_opt'

    def __init__(self, rhs: Rhs, typ: OptVal[Type] = OptVal()):
        super().__init__(rhs, typ=ProxyType((rhs,)))
        self.rhs = rhs
        self.chk_type(typ, OptVal)
        rhs.nested = True
        typ = typ and typ.opt.make_func() or NoType()
        self.return_type = typ.return_type

    def children(self) -> Iterator:
        yield from super().children()

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        return super().make_parse_recipe(**kwds)

    def __str__(self) -> str:
        return f"[{self.rhs.__str__()}]"

    def __repr__(self) -> str:
        return f"<OptGroup {self.rhs}>"


class Leaf(Primary):
    def __init__(self, value: Any):
        super().__init__()
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
    src: ParseSource                # Used to mske the parse recipe.
    resolved: ParseExpr             # Expr denoted by the name, if not a token.
    
    def __init__(self, name: Token, args: Optional[Args] = None):
        super().__init__(ObjName(name))
        self.args = args or NoArgs()
        self.name = ObjName(f"_{self.value}")

    def children(self) -> Iterator[GrammarTree]:
        yield from super().children()
        if self.isexternal():
            yield self.value
        #else:
        #    try: yield self.resolved
        #    except: pass
        yield self.args

    def pre_init(self) -> None:
        super().pre_init()
        if self.value.string.startswith('_'):
            self.grammar_error(f"identifier {self.value} cannot start with an underscore.")
        # Get the source object for the parse recipe.
        name = self.value
        def src() -> ParseSource:


            if self.isexternal():
                # This is a Token parsed from the input file.
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
                # It's some object in local scope.
                resolved = self.resolved = self.resolve()
                if self.assigned_name or 0x0000:
                    self.local_src = True
                src_type = resolved.type
                if isinstance(resolved, Rule):
                    name = self.gen.rule_func_name(resolved)
                    func_type =  ParseFunc
                    self.grammar.update_type_defs(self)
                else:
                    if self.local_src:
                        src_type = resolved.return_type
                    else:
                        src_type = src_type.make_func()
                        
                    name = self.value
                    # src_type is the value type of the parameter/variable.
                    # Turn it into a function, unless the resolved object
                    # is declared as a VarItem in the same Alt as self.
                    #if not self.assigned_name or self.alt is not resolved.alt:
                    #if not self.assigned_name:
                    #    src_type = src_type.make_func()
                    func_type =  ParseNone

                isrule = isinstance(resolved, Rule)
                islocal = self.local_env.lookup(name) is resolved
                assert isrule != islocal
                return ParseSource(name, src_type, func_type=func_type)

        self.src = src()
        if self.isexternal():
            #self.type = self.src.type
            self.type = self.src.return_type.make_func()
        else:
            self.type = self.src.type
            #self.type = self.src.return_type.make_func()
            #self.type = self.resolved.type or ObjType(self.gen.default_type())
        self.chk_type(self.type, Type)

    def isexternal(self) -> bool:                                                                                                   
        return str(self.value).isupper()

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
        resolved = self.resolve()
        # The name of a variable or parameter is used.
        if not isinstance(resolved, Rule):
            yield self.value
        if self.args:
            vis = TargetCodeVisitor(self.args)
            item_names = vis.vars(self.local_env)
            yield from item_names

    def make_parse_recipe(self, **kwds) -> ParseRecipe:
        name = self.value
        if self.isexternal():
            # The name is a type of token.

            # Try a basic token name.
            func_name: str = self.basic_parser_name()
            if func_name:
                return ParseRecipeInline(
                    self, None, self.src,
                    params=Params(),
                    **kwds)

            # Try some other token name.
            tok_name: str = self.token_parser_name()
            if tok_name:
                return ParseRecipeInline(
                    self, None, self.src,
                    f"{self.gen.token_types[name.string]}",
                    params=Params(),
                    typ=self.type,
                    **kwds)
            self.grammar_error(f"Token type {name} not recognized.")

        # The name is a rule, parameter, or variable.
        resolved = self.resolved
        args = self.args
        isrule = isinstance(resolved, Rule)
        islocal = self.local_env.lookup(name) is resolved
        assert bool(isrule) != bool(islocal)
        if isrule and args.empty:
            args = Args()
        # Validate the resolved parameters vs. the arguments.
        resolved_type = resolved.assigned_name and resolved.return_type or resolved.type
        #resolved_type = isrule and resolved.type or resolved.type
        #resolved_type = isrule and resolved.type or resolved.return_type
        if args.empty:
            # No arguments, so there must be no parameters.
            if resolved_type.callable:
                self.grammar_error(f"Reference to name {name.string!r} requires argument list.")
        elif not resolved_type.callable:
            if not args.empty:
                self.grammar_error(f"Reference to name {resolved.name.string!r} cannot have argument list.")
        elif len(args) != len(resolved_type.params):
            self.grammar_error(f"Reference to name {resolved.name.string!r} requires exactly {len(resolved_type.params)} arguments.")
        if islocal:
            return ParseRecipeLocal(
                self, name, self.src, *self.args,
                params=NoParams(),
                typ=self.type,
                func_type=ParseFunc,
                use_inline=bool(self.assigned_name),
                **kwds)
        elif isrule:
            if not self.return_type:
                self.return_type = ObjType(self.gen.default_type())
            return ParseRecipeRule(
                self, name, self.src, args,
                params=Params(),
                typ=self.type,
                func_type=ParseFunc, **kwds)
        else:
            return ParseRecipeInline(
                self, name, self.src, args,
                params=params,
                func_type=ParseFunc, **kwds)

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

    def __str__(self) -> str:
        if self.value.string == "ENDMARKER":
            return "$"
        return super().__str__() + str(self.args)

    def __repr__(self) -> str:
        return f"<NameLeaf {self}>"


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
        def src() -> ParseSource:
            func_name = self.keyword_parser_name()
            if func_name:
                return func_name
            func_name = self.soft_keyword_parser_name()
            if func_name:
                return func_name
            func_name = self.literal_parser_name()
            if func_name:
                return func_name
        self.src = src()
        self.type = self.src.return_type.make_func()

    @functools.cached_property
    def val(self) -> str:
        return ast.literal_eval(self.value)

    @functools.cached_property
    def is_keyword(self) -> bool:
        return bool(self.keyword_re.match(self.val))

    def keyword_parser_name(self) -> Optional[ParseSource]:
        """ If this is a regular keyword, return name of parser function. """
        if self.is_keyword and self.value.endswith("'"):
            return self.gen.parse_token

    def soft_keyword_parser_name(self) -> Optional[ParseSource]:
        """ If this is a soft keyword, return name of parser function. """
        if self.is_keyword and not self.value.endswith("'"):
            return self.gen.parse_soft_keyword

    def literal_parser_name(self) -> Optional[ParseSource]:
        """ If this is not a keyword, return name of parser function. """
        if not self.is_keyword:
            if self.val in self.gen.exact_tokens:
                return self.gen.parse_token
            assert len(self.val) == 1, f"{self.value} is not a known literal"
            return self.gen.parse_char

    def make_parse_recipe(self, **kwds) -> ParseRecipe:

        val = self.val
        func_name = self.keyword_parser_name()
        if func_name:
            return ParseRecipeExternal(self, "_keyword", func_name, f"{self.gen.keywords[val]}",
                typ=self.type,
                comment=f"keyword = '{val}'",
                params=Params(),
                func_type=ParseFunc,
                **kwds,
                )
        func_name = self.soft_keyword_parser_name()
        if func_name:
            return ParseRecipeExternal(self, "_keyword", func_name, f"{self.value}",
                typ=self.type,
                comment=f"keyword = '{val}'",
                params=Params(),
                func_type=ParseFunc,
                **kwds,
                )
        func_name = self.literal_parser_name()
        if func_name:
            typ = self.gen.exact_tokens.get(val, val)
            return ParseRecipeExternal(self, "_literal", func_name, f"{typ!r}",
                typ=self.type,
                comment=f"token = \"{val}\"",
                params=Params(),
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

    def children(self) -> Iterator:
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
    )

from pegen.expr_type import (
    ProxyType,
    FuncType,
    NoParams,
    NoType,
    ObjType,
    Param,
    Params,
    Type,
    )

from pegen.target_code import Code, NoCode, TargetCodeVisitor

