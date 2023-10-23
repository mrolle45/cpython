"""
expr_type module.
Manages the types and function parmeters of the
results of parse expressions.

"""
from __future__ import annotations

import dataclasses
import contextlib

from pegen.grammar import *
from pegen.target_code import Code, NoCode

class ExprParams(GrammarTree):
    """ A set, possibly empty, of parameters for a FuncType. """
    params: tuple[ExprParam]
    empty: bool = False

    def __init__(self, params: Iterable[ExprParam]):
        self.params = tuple(params)

    def __len__(self) -> int:
        return len(self.params)

    def __str__(self) -> str:
        return f"(', '.join([str(param) for param in self.params])"

    def in_func(self) -> str:
        """ Text of the parameters, as included in a function def or decl. """
        if not len(self):
            return self.gen.empty_params()
        return ', '.join(
            [f"{self.gen.typed_name(param)}" for param in self.params])



class ExprNoParams(ExprParams):
    """ A placeholder for the missing parameter list of an ObjType. """
    empty: bool = True
    def __new__(cls) -> Self:
        return object.__new__(cls)

    def __init__(cls) -> Self:
        super().__init__(())

    def __bool__(self) -> bool:
        return False

    def in_func(self) -> str:
        return ''

    def __str__(self) -> str:
        return ''

    def __repr__(self) -> str:
        return f"<No Params>"


class Param(TypedName, GrammarTree):
    """ A single parameter description for a callable object.
    It may itself be a callable, with its own parameters (recursively).
    It is just like TypedName, except that the name is optional.
        A NoName object is supplied as a default.
    """
    def __init__(self, name: TypedName):
        super().__init__(name.name, name.type)
        #super().__init__(name.name, name.type.make_func())

    def pre_init(self) -> None:
        super().pre_init()
        if not self.type:
            self.type = ObjType(self.gen.default_type())
        elif not self.return_type:
            self.return_type = ObjType(self.gen.default_type())

    def __str__(self) -> str:
        return self._gen().typed_name(self)
        #return self.gen.typed_name(self)


class Params(TrueHashableList[Param], GrammarTree):
    """ Parameters for a callable Type.
    Subclass NoParams is used for a non-callable Type.
    """

    empty: bool = False

    def __init__(self, params: List[Param] = []):
        super().__init__(items=params)
        if params:
            # Create dummy names.
            for n, param in enumerate(params, 1):
                if not param.name:
                    param.name = ObjName(f"__p{n}")

    def pre_init(self) -> None:
        super().pre_init()
        if len(self):
            params = list(self)
            # Verify that the names are unique.
            unique_names = set()
            for param in params:
                unique_names.add(param.name)
            if len(unique_names) != len(params):
                names = tuple(param.name for param in params)
                self.grammar_error(f'Parameter names {names} must be different.')

    def __str__(self) -> str:
        """ String used in a call expression, including the parens. """
        return f'({", ".join([str(param) for param in self])})'

    def __repr__(self) -> str:
        return f"<Params {self}>"

    def get(self, name: str) -> Optional[Param]:
        for param in self:
            if param.name == name: return param
        return None

    def in_func(self) -> str:
        """ Text of the parameters, as included in a function def or decl. """
        gen = self._gen()
        if not len(self):
            return gen.empty_params()
        return ', '.join(
            [f"{gen.typed_name(param)}"
             for param in self])


class NoParams(Params):
    empty: bool = True

    def __init__(self):
        super().__init__()

    def __bool__(self) -> bool:
        return False

    def in_func(self) -> str:
        return ''

    def __str__(self) -> str:
        return ''

    def __repr__(self) -> str:
        return '<No params>'


def get_type(typ: Type | OptVal[Type] | None, dflt: Type = None) -> Type:
    if isinstance(typ, OptVal):
        typ = typ.opt
    if not typ:
        typ = dflt or NoType()
    assert isinstance(typ, Type), f"Expected Type, not {typ!r}"
    return typ



"""
Any ParseExpr, when parsed from the input, results in a
value of some particular type. givem by a Type object.
This Type may be taken from the grammar of the expression,
or it will be derived from the types the expression's elements.

A Type may be either an object type, or a function type.
An ObjType represents a fundamental type in the target language,
It comes from an annotation in the grammar, resulting in a Code.

A FuncType represents a function which is called with arguments.
The call returns a result which is given by another Type.
The return type might be another function, in which case, that
function is called with another argument list.
This can be repeated any number of times.  The ultimate result is an ObjectType.

The syntax in the grammar file is:
    '[' obj_type ']' ( '(' params ')' ) *

"""

class Type(GrammarTree):
    """ The type of either an object or a function.
    It is analogous to a "C" type expression, including function pointers,
    function parameters, and functions returning functions.
    General form is V type, or V (*type) (params) ... (params) [1 or more (params)].
    V is a basic type, represented by a Code.  (params) is a Params object.
    Type() constructor (not a subclass) produces either an ObjType or a FuncType or a NoType.
    """

    def __new__(
        cls: Type[Self],
        val_type: Code | str = None,
        params: Params = None,
        *more_param_lists: Params
        ) -> Type:
        if cls is Type:
            cls = (
                params and FuncType
                or val_type and ObjType
                or NoType
                )
        return super().__new__(cls)

class TypeCommon(Type):
    """ Common base for ObjType and FuncType. """
    node: ParseExpr = None

    def children(self) -> Iterable[ValueType]:
        yield self.val_type

    def make_func(self, *param: Param) -> FuncType:
        return FuncType(self, Params(param))


class ObjType (TypeCommon):
    """ A simple object type, without parameters. """
    callable: bool = False

    def __init__(self, val_type: Code | str | None):
        super().__init__()
        self.val_type = val_type and Code(val_type)

    def pre_init(self) -> None:
        super().pre_init()
        if self.val_type is None:
            self.val_type = self.gen.default_type()

    @property
    def return_type(self) -> Type:
        return self

    def param_lists(self) -> Iterable[Params]:
        return ()

    def __str__(self) -> str:
        return str(self.val_type)

    def __repr__(self) -> str:
        return f"<ObjType {self}>"


class FuncType(TypeCommon):
    """ A function with a return type and one parameter list.
    The return type can be a function with its own parameters,
        and so on, recursively.
    Can be created by calling:
        * Type(val_type: Code, params [, params]...).
        * FuncType(return_type: Type, params)
    """
    return_type: Type
    callable: bool = True

    def __init__(self, typ: Code | str | Type,
                 params: Params = None,
                 *more_params: Params
                 ):
        if isinstance(typ, Type):
            super().__init__(typ.val_type)
            assert not more_params, "FuncType() has at most one parameter list"
            self.return_type = typ
        else:
            super().__init__(typ)
            self.return_type = Type(typ, *more_params)
        self.params = params or Params()
        for p in [params, *more_params]:
            self.chk_type(p, Params, opt=True)

    @property
    def val_type(self) -> Code:
        return self.return_type.val_type

    def children(self) -> Iterable[ValueType]:
        yield from super().children()
        yield self.params
        yield self.return_type

    def param_lists(self) -> Iterable[Params]:
        yield self.params
        yield from self.return_type.param_lists()

    def __str__(self) -> str:
        params = ''.join(str(p) for p in self.param_lists())
        return f"{self.val_type}(*){params}"

    def __repr__(self) -> str:
        return f"<FuncType {self}>"

class NoType(ObjType):

    def __init__(self, val_type: Code = None):
        self.chk_type(val_type, Code, opt=True)
        super().__init__(NoCode())

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return '<No Type>'


class ProxyType(Type):
    """ A Type which gives the common return type of several
    target ParseExpr objects, provided that all the return types are equal.
    If there are different true Types, result is the language default.
    If there are one true type and one false type, the result is the false type.
    If no targets, return the language default type.
    """

    get_return_type: Callable[[Self], Type]
    pend_return: bool = False                   # True during a recursive return_type() call.
    targets: Tuple[ParseExpr, ...]

    def __init__(self, targets: Iterable[ParseExpr], params: Params = None):
        self.targets = tuple(targets)
        self.params = params or Params()
        if len(self.targets) > 1:
            self.get_return_type = self.return_type_multi
        elif self.targets: self.get_return_type = self.return_type_single
        else: self.get_return_type = self.return_type_none
        assert all(isinstance(t, GrammarTree) for t in self.targets)

    def children(self) -> Iterable[ValueType]:
        yield from super().children()
        yield self.params
        yield self.return_type

    @property
    def callable(self) -> bool:
        return bool(self.params)

    @property
    def val_type(self) -> Code:
        return self.return_type.val_type

    @property
    def return_type(self) -> Type:
        return self.get_return_type()

    @return_type.setter
    def return_type(self, typ: Type) -> None:
        for target in self.targets:
            target.return_type = typ

    # Specialized return_type methods, depending on the number of targets...

    def return_type_multi(self) -> Type:
        """ Return type with multiple targets. """
        with self.pending_return_type() as circular:
            if circular:
                return ObjType(self.gen.circular_type())
        results = [target.return_type for target in self.targets]
        r = set()
        for result in results:
            r.add(str(result))
        results = set(str(result) for result in results)
        if len(results) == 1:
            return self.targets[0].return_type
        else:
            false_type = NoType()
            if false_type in results:
                results.remove(false_type)
            if len(results) > 1:
                return Type(self.gen.default_type())
            else:
                return false_type

            return NoType()

    def return_type_single(self) -> Type:
        """ Return type with one target. """
        with self.pending_return_type() as circular:
            if circular:
                return ObjType(self.gen.circular_type())
            return self.targets[0].return_type

    def return_type_none(self) -> Type:
        """ Return type with no targets. """
        return NoType()

    @contextlib.contextmanager
    def pending_return_type(self) -> Iterator[bool]:
        """ Evaluate return_type() during the context, unless already in another call.
        The context is True if it is a circular situation.
        """
        if self.pend_return:
            yield True
            return
        self.pend_return = True
        try:
            yield False
        finally:
            del self.pend_return

    def make_func(self) -> Type:
        return self

    def param_lists(self) -> Iterable[Params]:
        if self.params: yield self.params
        yield from self.return_type.param_lists()

    def missing(self) -> list(Type):
        """ List of targets whose return types are false (i.e. NoType). """
        return [target
                for target in self.targets
                if not target.return_type]

    def dump(self) -> None:
        visited: Set = set()
        def recursive(typ, level=0) -> None:
            def pr(msg: str) -> None:
                print(f"{'    ' * level}{msg}")
            lead = f"{type(self.node).__name__} {self.node.uniq_name()}"
            if typ in visited:
                return pr(f"{lead} **")
            visited.add(typ)
            if not isinstance(typ, ProxyType):
                return pr(f"{lead} {typ}")
            pr(f"{lead} [{len(typ.targets)}]")
            for target in typ.targets:
                recursive(target,level + 1)
        recursive(self)

    def __str__(self) -> str:
        params = ''.join(str(p) for p in self.param_lists())
        return f"{self.val_type}(*){params}"

    def __repr__(self) -> str:
        return f"<ProxyType {self}>"

