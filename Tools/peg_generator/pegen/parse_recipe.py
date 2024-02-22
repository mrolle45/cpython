""" parse_recipe module.
ParseRecipe class and other related things to manage the generation of
code to parse a particular Grammar parse expression object.
"""

from __future__ import annotations

from enum import Enum, auto
import dataclasses
import functools

from pegen.grammar import(
    Args,
    GrammarTree,
    NoArgs,
    ParseExpr,
    Rule,
    TypedName,
    VarItem,
    )

from pegen.expr_type import(
    FuncType,
    Params,
    )

class RecipeMode(Enum):
    Ext = auto()
    Loc = auto()
    Rule = auto()
    Inl = auto()


@dataclasses.dataclass
class ParseFuncType:
    """ Base class for indicating how an expression is parsed.
    #The name of the subclass can be used to generate a type name for the generated function.
    """
    always_true: bool = False
    returns_status: bool = True
    has_result: bool = True
    use_parser: bool = True
    use_res_ptr: bool = True

    def __repr__(self) -> str:
        return f"<{self._name}>"

def make_func_types(
    **kwds: ParseFuncType
    ) -> None:
    """ Set given func types as global variables, and set their ._name attributes.
    Also make inline variants, with .inline = True.
    """
    for name, value in kwds.items():
        globals()[name] = value
        value._name = name

make_func_types(
    # Function returns both status and result object.
    ParseFunc = ParseFuncType(
        ),

    # Function returns result object only, no status.  Always succeeds.
    ParseTrue = ParseFuncType(
        always_true = True,
        ),

    # Function returns status only, no result object.
    ParseTest = ParseFuncType(
        use_res_ptr = False,
        has_result = False,
        ),

    # Function returns neither status nor result object, still calls the parser.
    ParseVoid = ParseFuncType(
        returns_status = False,
        use_res_ptr = False,
        has_result = False,
        always_true = True,
        ),

    # Function returns result directly, status is truth value of result.
    ParseData = ParseFuncType(
        returns_status = False,
        use_res_ptr = False,
        ),

    #Function with neither parser nor status nor result pointer.
    #For local names.  Always true.
    ParseNone = ParseFuncType(
        always_true = True,
        returns_status = False,
        has_result = False,
        use_parser = False,
        use_res_ptr = False,
        ),

    )

class ParseCall(TypedName):
    """ How to generate the code to call either the inner or the outer function of a Recipe.
    """

    '''
    Properties of the call:
     
    1. Return type.
        This is the Type of the result of the parse.
        The call might not have a result value, in which case
        the Type is NoType, or perhaps some other false object.
        The outer call has the same Type as the inner call,
        except that the outer call could be NoType, to ignore the
        inner call result.
    2. Destination.
        This is where the result will be stored.
        It could be:
        * ppRes, which is the address and size of the result.
        * A variable.  self.assigned_name is the name of this variable.
        * Nothing.  There is no result, or if there is a result, it is ignored.
        If the outer goes to a variable, and the inner goes to a ppRes,
        then the inner is called with &_ptr_{var name}.
    3. Status Return?
        If True, then it returns a value, indicating if the parse succeeded.
        If False, then the parse always succeeds, and returns a status True.
    4. Name.
        The name of the function (or possibly a variable) to do the parse.
    5. Parameters.
        A Params object with parameters peculiar to the called function.
        This does not include any parser or result pointer parameters.
        If a NoParams, then the called name is an object, not a function.
    6. Arguments.
        Actual argument objects corresponding to the parameters (above).
    7. Use parser?
        True if the Parser is used by the calling function.
    8. Use result pointer?

    '''

    args: Args
    assigned_name: ObjName = None           # Used to augment args in fixup.
                                            # Is the arg corresponding to use_result_ptr (if any).

    def __init__(self,
            name: TypedName,
            args: Args,
            *,
            assigned_name: ObjName = None,
            value_type: Type = None,
            ) -> None:
        super().__init__(name.name,
                         value_type and FuncType(value_type) or name.type,
                         #name.params,
                         )
        self.args = args
        if assigned_name: self.assigned_name = assigned_name

    def pre_init(self) -> None:
        """ Add args and params as required by func_type. """
        super().pre_init()
        gen: ParserGenerator = self.gen
        recipe: ParseRecipe = self.parent

        self.locations = self.parent.locations
        call_params = gen.parse_func_params(recipe.src.func_type)[:]
        #if self.func_type.use_parser:
        #    call_params.append(gen.parser_param())
        #if self.func_type.use_res_ptr:
        #    call_params.append(gen.return_param())
        if len(self.params):
            call_params += self.params[:]

        self.call_params = Params(call_params)

        args = gen.parse_func_args(recipe)[:]
        if args:
            self.args = Args([*args, *(self.args[:])])

    def value_expr(self) -> str:
        """ The expression which evaluates to the value of this recipe. """
        return self.gen.format_parse_call(self)
            #self.parent, self.name, self.args, self.call_params, func_type=self.func_type,
            #assigned_name=self.assigned_name
            #)


class ParseCallInner(ParseCall):
    """ ParseCall class used for recipe.inner_call. """


class ParseCallOuter(ParseCall):
    """ ParseCall class used for recipe.outer_call. """


class ParseSource(TypedName):
    """ Specifies how to generate the code to produce the parsed value.
    Contains information only common to several Recipes.
    The return type may be None, and has to be supplied to each Recipe() constructor.
    """

    def __init__(self,
        gen: ParserGenerator,
        *name_args,
        args: Args = None,
        assigned_name: ObjName = None,
        **kwds,
        ):
        super().__init__(*name_args, **kwds)
        self.gen = gen
        self.assigned_name = assigned_name

    def pre_init(self):
        super().pre_init()

class ParseRecipe(GrammarTree):
    """ Recipe for generating code to obtain a parsed result.
    The code is generated by calling self(gen: ParserGenerator, **kwds).
    The source of the result may be:
        A rule, called with optional parameters.
        A variable or parameter name, which will be used directly or
            called with optional parameters.
        A helper function, with parameters.
    Constructed from:
        The node which is being parsed.  It will be the .item of some VarItem.
        A default name for the parsed result.
        A TypedName for the source, which includes type and maybe parameters.
        Optional arguments, to supply arguments for a call to the source.
            The argument can be a text string, copied from the grammar file,
            or a tuple (name, node) denoting an inline generated function,
            or an Args object (possibly NoArgs).
        Optional named inlined nodes, which are not given as args.
        Optional function to generate more code in the body,
            between the inlines and the return expr.
        Optional comment (without the // or #).
    Note: this class must be mixed with GrammarTree to inherit its behavior.
    """
    expr_name: str
    args: Args
    inner_call: ParseCallInner
    outer_call: ParseCallOuter

    def __init__(self,
            node: GrammarTree,
            name: str,
            src: ParseSource,
            *args: str | Tuple[str, GrammarTree] | Args,
            owner: ParseExpr = None,
            params: Params = None,
            value_type: Type = None,            # Replaces src.type when src.type is None.
            inlines: list[GrammarTree] = [],
            assigned_name: str = None,
            # Callback to generate more code at the end of the function.
            # If it returns a str, this is the return value.
            extra: Callable[[], str | None] = None,
            comment: str = None,
            **kwds,
        ):
        any(self.chk_type(inl, ParseExpr) for inl in inlines)
        self.chk_type(src, ParseSource)
        self.node = node
        self.owner = owner or node
        self.name = assigned_name or name or node.name
        self.src = src
        if not value_type: value_type = src.type
        self.params = params or Params()
        if args and isinstance(args[0], Args):
            args = args[0]
        elif src and src.params is None:
            args = NoArgs()
        else:
            args = Args(args)

        self.inner_call = ParseCallInner(src, args, value_type=value_type
            )
        self.outer_call = ParseCallOuter(
            TypedName(node.uniq_name(), FuncType(value_type)),
            NoArgs(),
            assigned_name=assigned_name)
        self.inlines = [arg.inline for arg in args[:] if arg.inline]
        if inlines: self.inlines += inlines
        self.extra = extra
        self.comment = comment

    def __iter__(self) -> Iterator[GrammarTree]:
        if self.src: yield self.src
        yield self.inner_call
        yield self.outer_call

    def __call__(self, gen: ParserGenerator, **kwds) -> None:
        gen.gen_parse(self, **kwds)

    def validate(self, visited: set = set(), level: int = 0) -> None:
        if self in visited: return
        visited.add(self)

        #print('  ' * level, repr(self))

        #assert isinstance(self.src, TypedName)
        assert isinstance(self.params, Params)
        self.src.validate(visited, level + 1)
        pass

    @property
    def func_type(self) -> ParseFuncType:
        return self.outer_call.func_type

    def value_expr(self) -> str:
        """ The expression which evaluates to the value of this recipe. """
        return self.inner_call.value_expr()

    def value_type(self) -> Code:
        return self.outer_call.type

    #@functools.cache
    #def uniq_name(self) -> str:
    #    """ A name which is unique among all recipes. """
    #    # Find recipe of next node in the parent chain which has one.
    #    anc = self.parent.parent
    #    if hasattr(anc, 'parse_recipe'):
    #        return f"{anc.parse_recipe.uniq_name()}_{self.name}"
    #    elif isinstance(self.node, Rule):
    #        return f"_{self.name}"
    #    else:
    #        return self.name

    def inline_recipes(self) -> Iterator[ParseRecipe]:
        for inline in self.inlines:
            recipe = inline.parse_recipe
            #if recipe.mode in (recipe.Rule,):
            #    continue
            if recipe.mode is recipe.Loc and recipe.outer_call.assigned_name:
                continue
            yield recipe

    def pre_init(self) -> Self:
        """ Apply tweaks appropriate to the target language and the src object
        Tweaks may include:
            Modifying the source name, including making the name into a method of the parser.
            Modifying the arguments and parameters to include the parser.
            Modifying the default name.
        """
        super().pre_init()
        self.gen.fix_parse_recipe(self)
        return self

    @property
    def locations(self) -> TokenLocations:
        return self.node.locations

    def dump(self, level: int = 0) -> None:
        """ Print this recipe and all inline recipes (recursively). """
        lead = '    ' * level
        print(f"{lead}Recipe {self.mode.name} {self!r}")
        print(f"{lead}  uniq_name   = {self.uniq_name()!r}")
        print(f"{lead}  node        = {self.node!r}")
        print(f"{lead}  value       = {self.value_expr()!r}")

        for inline in self.inlines:
            inline.parse_recipe.dump(level + 1)

    def brk(self) -> bool:
        return self.node.brk()

    def __repr__(self) -> str:
        return f"<{self.mode.name} {self.name} = {self.src and self.src.name or self.name}>"

    for mode in RecipeMode:
        exec(f"{mode.name} = {mode}")
    mode = None


class ParseRecipeExternal(ParseRecipe):
    mode = RecipeMode.Ext


class ParseRecipeLocal(ParseRecipe):
    mode = RecipeMode.Loc


class ParseRecipeRule(ParseRecipe):
    mode = RecipeMode.Rule


class ParseRecipeInline(ParseRecipe):
    mode = RecipeMode.Inl


