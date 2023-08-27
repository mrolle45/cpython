""" parse_recipe module.
ParseRecipe class and other related things to manage the generation of
code to parse a particular Grammar parse expression object.
"""

from __future__ import annotations

from enum import Enum, auto
import dataclasses
import functools

from pegen import grammar

class RecipeMode(Enum):
    Ext = auto()
    Loc = auto()
    Rule = auto()
    Inl = auto()

class ParseFuncType(Enum):
    def __init__(self):
        pass

@dataclasses.dataclass
class ParseFuncType:
    """ Base class for indicating how an expression is parsed.
    #The name of the subclass can be used to generate a type name for the generated function.
    """
    always_true: bool = False
    has_result: bool = True
    returns_status: bool = True
    use_parser: bool = True
    use_res_ptr: bool = True
    local_result: bool = False
    local: ParseFuncType = None             # An equivalent func type, with local_result = True.

    def __repr__(self) -> str:
        return f"<{self._name}>"

def make_func_types(
    **kwds: ParseFuncType
    ) -> None:
    """ Set given func types as global variables, and set their ._name attributes.
    Also make Local variants, with .local_result = True.
    """
    for name, value in kwds.items():
        globals()[name] = value
        value._name = name
        local = dataclasses.replace(value, local_result=True)
        value.local = local
        local_name = f"{name}Local"
        globals()[local_name] = local
        local._name = local_name

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

    # Function returns neither status nor result object.
    ParseVoid = ParseFuncType(
        use_res_ptr = False,
        always_true = True,
        ),

    # Function returns result directly, status is truth value of result.
    ParseData = ParseFuncType(
        returns_status = False,
        use_res_ptr = False,
        ),

    ## Function with neither parser nor status nor result pointer.
    ## For function or variable defined locally in the calling function.
    #ParseLocal = ParseFuncType(
    #    use_parser = False,
    #    #use_res_ptr = False,
    #    always_true = True,
    #    ),

    #ParseTestLocal = ParseFuncType(
    #    use_parser = False,
    #    use_res_ptr = False,
    #    has_result = False,
    #    ),

    #ParseTrueLocal = ParseFuncType(
    #    use_parser = False,
    #    use_res_ptr = False,
    #    always_true = True,
    #    ),

    #Function with neither parser nor status nor result pointer.
    #For local names.  Always true.
    ParseNone = ParseFuncType(
        always_true = True,
        use_parser = False,
        use_res_ptr = False,
        ),

    )

class ParseCall(grammar.TypedName):
    """ How to generate the code to call either the inner or the outer function of a Recipe.
    """
    args: Args
    func_type: ParseFuncType                # Used to augment args and params in fixup.
    assigned_name: ObjName = None           # Used to augment args in fixup.
                                            # Is the arg corresponding to use_result_ptr (if any).

    def __init__(self,
            name: TypedName,
            args: Args,
            func_type: ParseFuncType,
            assigned_name: ObjName = None,
            value_type: Code = None,
            ) -> None:
        super().__init__(name.name, value_type or name.type, name.params)
        self.args = args
        self.func_type = func_type
        if assigned_name: self.assigned_name = assigned_name

    def pre_init(self) -> None:
        """ Add args and params as required by func_type. """
        super().pre_init()
        gen: ParserGenerator = self.gen
        recipe: ParseRecipe = self.parent


        call_params = []
        #if self.func_type.use_parser:
        #    call_params.append(gen.parser_param())
        #if self.func_type.use_res_ptr:
        #    call_params.append(gen.return_param())
        if self.params:
            call_params += self.params[:]

        self.call_params = grammar.Params(call_params)

        args = []
        #if recipe.mode is not recipe.Loc:
        #    if self.func_type.use_parser:
        #        # The src is a function which needs a parser argument.
        #        args.append(gen.parser_param().name.string)
        #    if recipe.src.func_type.use_res_ptr:
        #        # The src is a function which needs a return pointer.
        #        args.append(gen.parse_result_ptr_param(self.assigned_name).name.string)
        if args:
            self.args = grammar.Args([*args, *(self.args[:])])

    def value_expr(self) -> str:
        """ The expression which evaluates to the value of this recipe. """
        return self.gen.parse_value_expr(
            self.name, self.args, self.call_params, func_type=self.func_type,
            assigned_name=self.assigned_name
            )


class ParseCallInner(ParseCall):
    """ ParseCall class used for recipe.inner_call. """


class ParseCallOuter(ParseCall):
    """ ParseCall class used for recipe.outer_call. """


class ParseSource(grammar.TypedName):
    """ Specifies how to generate the code to produce the parsed value.
    Contains information only common to several Recipes.
    The return type may be None, and has to be supplied to each Recipe() constructor.
    """

    def __init__(self,
        *name_args,
        func_type: FuncBase = None,
        args: Args = None,
        assigned_name: ObjName = None,
        ):
        super().__init__(*name_args)
        self.func_type = func_type or self.gen.default_func_type()
        self.assigned_name = assigned_name

class ParseRecipe(grammar.GrammarTree):
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
    Note: this class must be mixed with grammar.GrammarTree to inherit its behavior.
    """
    expr_name: str
    args: Args
    inner_call: ParseCallInner
    outer_call: ParseCallOuter
    inline_locals: bool = True              # inline recipes that are Loc get inlined.

    def __init__(self,
        node: GrammarTree,
        name: str,
        src: ParseSource,
        *args: str | Tuple[str, GrammarTree] | Args,
        params: Params = None,
        value_type: Code = None,            # Replaces src.type when src.type is None.
        func_type: ParseFuncType = None,    # Overrides src.func_type in outer_call.
        inlines: list[GrammarTree] = [],
        inline_locals: bool = True,
        assigned_name: str = None,
        # Callback to generate more code at the end of the function.
        # If it returns a str, this is the return value.
        extra: Callable[[], str | None] = None,
        comment: str = None,
        **kwds,
        ):
        assert all(isinstance(inl, grammar.ParseExpr) for inl in inlines)
        assert type(src) is ParseSource
        self.node = node
        if not inline_locals: self.inline_locals = False
        parent = node.parent
        if isinstance(parent, grammar.VarItem):
            if parent.name:
                name = parent.name
            else:
                name = parent.dedupe(name)
        self.name = self.dflt_name = node.name or name
        self.src = src
        if not value_type: value_type = src.type
        self.params = params or grammar.Params()
        if args and isinstance(args[0], grammar.Args):
            args = args[0]
        elif src and src.params is None:
            args = grammar.NoArgs()
        else:
            args = grammar.Args(args)

        self.inner_call = ParseCallInner(src, args, src.func_type, value_type=value_type,
            assigned_name=assigned_name)
        self.outer_call = ParseCallOuter(
            grammar.TypedName(node.uniq_name(), value_type, params),
            grammar.NoArgs(),
            func_type or src.func_type,
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
        assert isinstance(self.params, grammar.Params)
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

    @functools.cache
    def uniq_name(self) -> str:
        """ A name which is unique among all recipes. """
        # Find recipe of next node in the parent chain which has one.
        anc = self.parent.parent
        if hasattr(anc, 'parse_recipe'):
            return f"{anc.parse_recipe.uniq_name()}_{self.name}"
        elif isinstance(self.node, grammar.Rule):
            return f"_{self.name}"
        else:
            return self.name

    def inline_recipes(self) -> Iterable[ParseRecipe]:
        for inline in self.inlines:
            recipe = inline.parse_recipe
            #if recipe.mode in (recipe.Inl,):
            if recipe.mode is recipe.Loc and not self.inline_locals:
                continue
            yield recipe

    def pre_init(self) -> Self:
        """ Apply tweaks appropriate to the target language and the src object
        Tweaks may include:
            Modifying the source name, including making the name into a method of the parser.
            Modifying the arguments and parameters to include the parser.
            Modifying the default name.
        """
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


#import grammar
