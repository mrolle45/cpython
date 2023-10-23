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
    NoParams,
    ObjType,
    Params,
    Type,
    )

from pegen.target_code import(
    Code,
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
    has_result: bool = True
    returns_status: bool = True
    use_parser: bool = True
    use_res_ptr: bool = True
    nested: bool = False                     # True for an inline.
    inline: ParseFuncType = None             # An equivalent func type, with nested = True.
    isrule: bool = False

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
        inline = dataclasses.replace(value, nested=True)
        value.inline = inline
        inline_name = f"{name}Inl"
        globals()[inline_name] = inline
        inline._name = inline_name

make_func_types(
    # Function returns both status and result object.
    ParseFunc = ParseFuncType(
        ),

    # Function is for a Rule.
    ParseRule = ParseFuncType(
        isrule = True,
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
        has_result = False,
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

class ParseCall(TypedName):
    """ How to generate the code to call either the inner or the outer function of a Recipe.
    """
    args: Args
    func_type: ParseFuncType                # Used to augment args and params in fixup.

    def __init__(self,
            name: TypedName,
            args: Args,
            func_type: ParseFuncType,
            value_type: Code = None,
            ) -> None:
        super().__init__(
            name.name,
            name.type,
            )
        self.args = args
        self.func_type = func_type

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
        if self.type and self.type.callable:
            self.call_params = self.type.params
        else:
            self.call_params = NoParams()

        args = []
        if args:
            self.args = Args([*args, *(self.args[:])])

    def value_expr(self, **kwds) -> str:
        """ The expression which evaluates to the value of this recipe. """
        return self.gen.parse_value_expr(
            self.name, self.args, self.call_params,
            func_type=self.func_type,
            **kwds,
            )


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
        *name_args,
        func_type: FuncBase = ParseFunc,
        args: Args = None,
        ):
        super().__init__(*name_args)
        self.func_type = func_type


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
    Note: this class must be mixed with grammar.GrammarTree to inherit its behavior.
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
        params: Params = None,
        typ: Type = None,                   # Overrides src.type in outer_call.
        func_type: ParseFuncType = None,    # Overrides src.func_type in outer_call.
        inlines: list[GrammarTree] = [],
        use_inline: bool = True,            # This is an inline recipe
        # Callback to generate more code at the end of the function.
        # If it returns a str, this is the return value.
        extra: Callable[[], str | None] = None,
        comment: str = None,
        **kwds,
        ):
        assert all(isinstance(inl, ParseExpr) for inl in inlines)
        assert type(src) is ParseSource
        self.node = node
        self.name = self.dflt_name = node.uniq_name() or name
        self.src = src
        if not typ:
            typ = src.type or ObjType(node.gen.default_type())
        self.type = typ
        if params is None: params = Params()
        if args and isinstance(args[0], Args):
            args = args[0]
        elif not params:
            args = NoArgs()
        else:
            args = Args(args)

        self.inner_call = ParseCallInner(
            src, args, src.func_type,
            )
        outer_func_type = func_type or src.func_type
        if use_inline:
            outer_func_type = outer_func_type.inline
        self.outer_call = ParseCallOuter(
            TypedName(
                self.name,
                typ,
                ),
            args,
            #NoArgs(),
            outer_func_type,
            )
        self.inlines = [arg.inline for arg in args[:] if arg.inline]
        if inlines: self.inlines += [inline for inline in inlines if not inline.local_src]
        #self.inlines = [inline for inline in self.inlines if not inline.local_src]
        self.extra = extra
        self.comment = comment

    def children(self) -> Iterator[GrammarTree]:
        if self.src: yield self.src
        yield self.inner_call
        yield self.outer_call

    def gen_parse(self: ParseRecipe,
        **kwds
        ) -> None:
        gen = self.gen
        err = getattr(self.node, '_error', False)
        if not err:
            gen.forward_declare_inlines(self)

        with gen.enter_function(
            self.func_name,
            self.func_type,
            ):
            if err:
                gen.print("return false;")
                return
            for inline in self.inline_recipes():
                # Expand the inline items.
                gen.gen_node(inline.node)
            gen.gen_copy_local_vars(self.node)
            #if self.node is not self.alt:
            #    gen.print(comment=str(self.node))
            call: str | None = None
            if self.extra:
                call = self.extra()
            if call is None:
                # Was not supplied by extra()
                call = self.value_expr()
                if call:
                    call = gen.recipe_result(call, self)

            if call: gen.print(call)

    def __call__(self, **kwds) -> None:
        try: self.gen_parse(**kwds)
        except: print(f"Exception in recipe {self!r}"); raise

    def validate(self) -> None:
        self.src.validate()
        pass

    @property
    def func_type(self) -> ParseFuncType:
        return self.outer_call.func_type

    def value_expr(self) -> str:
        """ The expression which evaluates to the value of this recipe. """
        return self.inner_call.value_expr()

    def value_type(self) -> Type:
        return self.outer_call.type.return_type

    def inline_recipes(self) -> Iterable[ParseRecipe]:
        for inline in self.inlines:
            recipe = inline.parse_recipe
            #if recipe.mode in (recipe.Inl,):
            #if recipe.mode is recipe.Loc:
            #    continue
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


