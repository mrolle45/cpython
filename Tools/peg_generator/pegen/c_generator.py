from __future__ import annotations

import ast
import os.path
import contextlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import IO, Any, Dict, List, Optional, Set, Text, Tuple
from token import OP

from pegen import grammar
from pegen import parse_recipe
from pegen.grammar import (
    Alt,
    AltItem,
    AltItems,
    Arg,
    Cut,
    Forced,
    Gather0,
    Gather1,
    #GrammarNode,
    GrammarVisitor,
    Group,
    Item,
    Leaf,
    Lookahead,
    NameLeaf,
    NegativeLookahead,
    Opt,
    PositiveLookahead,
    Repeat0,
    Repeat1,
    Rhs,
    Rule,
    StringLeaf,
    TypedName,
    VarItem,
)
from pegen.target_code import Code
from pegen.parser_generator import *
import cparser
#from pegen import cparser

EXTENSION_PREFIX = """\
#include "pegen.h"

#if defined(Py_DEBUG) && defined(Py_BUILD_CORE)
#  define D(x) if (Py_DebugFlag) x;
#else
#  define D(x)
#endif

#ifdef __wasi__
#  define MAXSTACK 4000
#else
#  define MAXSTACK 6000
#endif

"""


EXTENSION_SUFFIX = """

void *
_PyPegen_parse(Parser *_p)
{
    // Initialize keywords
    _p->keywords = reserved_keywords;
    _p->n_keyword_lists = n_keyword_lists;
    _p->soft_keywords = soft_keywords;

    _PyPegen_DECL_LOC_VAR(res, void *);

    if (_start_rule(_p, &_ptr_res))
        return res;
    // Parse failed
    return NULL;
}
"""



class _Traits(TargetLanguageTraits):
    from cparser import CParser
    from cparser.visitor import ASTVisitor
    #from pegen.cparser import CParser
    #from pegen.cparser.visitor import ASTVisitor

    c_parser: CParser = None

    language: ClassVar[str] = 'c'

    def default_header(self) -> str:
        return ''

    def default_trailer(self) -> str:
        return ''

    def comment_leader(self) -> str:
        return '// '

    def default_type(self, type: Code = None) -> Code:
        return type or Code('void *')

    def null_type(self) -> Code:
        return Code('void')

    def default_value(self, value: str = None) -> str:
        return value or 'NULL'

    def empty_params(self) -> str:
        return 'void'

    def typed_name(self, name: TypedName) -> str:
        parts: list[str] = []
        base_type = str(name.val_type or '') or str(self.default_type())
        if base_type: parts.append(base_type)
        if name.callable:
            # This node is a callable type.
            subtypes = [param.typed_name() for param in name.type.params]
            parts.append(f'(*{name.name}) ({", ".join(subtypes)})')
        else:
            if name.name: parts.append(str(name.name))
        p = ' '.join(parts)
        return p

    def parser_param(self) -> Param:
        return Param(TypedName('_p', Code('Parser *')))

    def parse_result_ptr_arg(self, assigned_name: ObjName = None) -> Arg:
        name = assigned_name and f"&_ptr_{assigned_name}" or "_ppRes"
        return Arg(name)

    def return_param(self, type: str = None) -> Param:
        return Param(TypedName('_ppRes', Code(f'{self.default_type(type)} *')))

    def parse_func_params(self, params: Params, func_type: ParseFuncType = None) -> Params:
        if not func_type and name:
            func_type = name.func_type
        new_params = []
        if not func_type or func_type.use_parser:
            new_params.append(TypedName('_p', ObjType(Code('Parser *'))))
        if not func_type or func_type.use_res_ptr:
        #if not func_type or func_type.use_res_ptr and not name.assigned_name:
            new_params.append(TypedName('_ppRes', ObjType(Code('ParseResultPtr *'))))
        new_params += params
        return Params(new_params)

    def recipe_result(self, stmt: str, recipe: ParseRecipe) -> str:
        """ The last line of the recipe generated code. """
        if recipe.outer_call.func_type.always_true:
            pass
        else:
            stmt = f"return {stmt}"
        stmt += ';'
        return stmt

    def rule_func_name(self, rule: Rule) -> ObjName:
        return ObjName(f"_{rule.name}_rule")

    def cast(self, value: str, to_type: TypedName) -> str:
        return f"({to_type.typed_name()}) ({value})"

    def cut_value(self, value: str = None) -> str:
        return '_PyPegen_cut_sentinel'

    def bool_value(self, value: bool) -> str:
        return f"{str(bool(value)).lower()}"

    def bool_type(self) -> str: return Code('ParseStatus')

    def circular_type(self) -> Code: return Code('circ_ty')

    def str_value(self, value: bool) -> str:
        return value.replace('"', r'\"')

    def uniq_name(self, node: GrammarTree, assigned_name: str = None) -> str:
        """ A name which is unique among all objects in scope.
        The scope for a C file is file scope, thus is all functions.
        """
        anc = node.parent
        return f"{anc.uniq_name()}_{assigned_name or node.name}"

    def rule_recipe(self, rule: Rule, src: ParseRecipeSource) -> ParseRecipe:
        """ Create a recipe for a Rule. """
        return ParseRecipeExternal(
            rule, rule.name, src,
            # Add an argument for the rule descriptor.
            '&_rule_descriptor',
            rule.local_env.count and "_local_values" or self.default_value(),
            params=rule.type.params,
            func_type=ParseRule,
            extra=lambda: self.gen_rule(rule),
            inlines = [rule.rhs],
            typ=rule.type,
            )

    def sequence_recipe_args(self, seq: Seq, *seq_args: Args) -> Args:
        """ Actual arguments for a Seq parse function.
        seq_args varies with the type of sequence.  It's an initial subset of:
            * repeat1
            * sep
        """
        return Args([
            Arg(inline=seq.elem),
            Arg(f"sizeof ({seq.elem.return_type})"),
            *seq_args,
            ])

    def parse_value_expr(
            self, name: ObjName, args: Args, params: Params, *,
            func_type: Type[FuncBase] = None,
            assigned_name: ObjName = None,
        ) -> str:
        """ An expression which produces the desired parse result. """
        if not args: return f"{name}"
        new_args = []
        new_params = []
        if func_type and func_type.use_parser:
            new_args.append('_p')
            new_params.append(TypedName('parser', ObjType(Code('Parser *'))))
        if func_type and func_type.use_res_ptr:
            new_args.append(self.parse_result_ptr_arg(assigned_name))
            new_params.append(TypedName('ppres', ObjType(Code('ParseResultPtr *'))))
        if new_args:
            args = Args([*new_args, *args])
            params = Params([*new_params, *params])
        return f"{name}{args.show_typed(params)}"

    def forward_declare_inlines(self, recipe: ParseRecipe) -> None:
        x = 0
        for node_recipe in recipe.inline_recipes():
            if node_recipe.func_name.callable:
                self.print(
                    f"static {self.decl_func(node_recipe.func_name, func_type=node_recipe.func_type, fwd=True)};")

    def gen_copy_local_vars(self, node: GrammarTree) -> None:
        """ Code to define inherited names and set their values stored in the parser. """
        items = list(node.local_env.items())
        vars_used = set(node.vars_used())
        for name, var in items:
            # Check if the name is actually used.
            if name in vars_used:
                self.print(self.setup_parent_name(name, var, node))

    def enter_function(
        self, name: TypedName,
        func_type: ParseFuncType = ParseFunc,
        comment: str = '',
        ) -> Iterator:
        return self.enter_scope(
            self.decl_func(name, func_type=func_type),
            '{}',
            comment=comment
            )

    def decl_func(
        self, func: TypedName,
        func_type: ParseFuncType = ParseFunc,
        fwd: bool = False
        ) -> str:
        """ The declaration of a callable name (without the value), as a function.
        The variable is a function (not a function pointer), including function parameters.
        """
        assert func.callable, f"{func} must be callable."
        name = func.name
        params = func.type.params
        if func_type.isrule:
            # Special code for a rule function, using PyPegen macro.
            params_str = len(params) and f", ({params.in_func()})" or ""

            return f"_PyPegen__RULE({name}{params_str})"
        params = self.parse_func_params(params, func_type)
        #assert params and func.name
        type = func_type.returns_status and self.bool_type() or str(func.type.val_type)
        if fwd:
            nparams = len(params)
            if str(type) == 'ParseStatus' and nparams == 2:
                return f"ParseFunc {name}"
            if str(type) == 'ParseStatus' and nparams == 1:
                return f"ParseTest {name}"
            if str(type) == 'void' and nparams == 2:
                return f"ParseTrue {name}"
        return f"{type} {name}({params.in_func()})"

    def gen_start(self, alt: Alt) -> None:
        """ Generate code to set starting line and col variables if required by action.
        Call before starting to parse alt if EXTRA is, or might be, contained in the action.
        """
        self.print()
        self.print("_PyPegen_EXTRA_START(_p);")

    def alt_action(self, alt: Alt) -> _Traits.Action:
        """ Generate code for the action in the Alt. """
        if self.skip_actions:
            return self.dummy_action()
        elif alt.action:
            return self.action(alt)
        else:
            return self.default_action(alt)

    def action(self, alt: Alt) -> _Traits.Action:
        if self.debug:
            self.print(
                f'D(fprintf(stderr, "Hit with action [%d-%d]: %s\\n", _mark, _p->mark, "{alt}"));'
            )
        # TODO: try to deduce the type of the expression in simple cases.
        return self.Action(alt.action, Type(self.default_type(), Params()))

    def default_action(self, alt: Alt) -> Self.Action:
        expr: str
        type: str
        vars = alt.all_vars()
        if len(vars) > 1:
            if self.debug:
                self.print(
                    f'D(fprintf(stderr, "Hit without action [%d:%d]: %s\\n", _mark, _p->mark, "{alt}"));'
                )
            expr = f"_PyPegen_dummy_name(_p, {', '.join(var.name.string for var in vars)})"
            type = Type("dummy_ty", Params())
        else:
            if self.debug:
                self.print(
                    f'D(fprintf(stderr, "Hit with default action [%d:%d]: %s\\n", _mark, _p->mark, "{node}"));'
                )
            if vars:
                cast = ""
                expr = f"{cast}{vars[0].name}"
                #type = vars[0].return_type
                type = ProxyType(vars)
            else:
                expr = f"{self.default_value()}"
                type = ObjType(self.default_type())
        return self.Action(Code(expr), type)

    def dummy_action(self) -> _Traits.Action:
        return self.Action("_PyPegen_dummy_name(_p)", ObjType(Code("expr_ty")))

    def fix_parse_recipe(self, recipe: ParseRecipe) -> None:

        recipe.expr_name = recipe.src.name or recipe.name
        #if isinstance(recipe.src, Rule):
        #    recipe.expr_name = f'_{recipe.expr_name}_rule'

        #if recipe.mode is recipe.Rule:
        #    recipe.params = Params([*self.parse_func_params(recipe.src), *recipe.params])
        #elif isinstance(recipe.node, Rule):
        #    recipe.params = Params([*self.parse_func_params(recipe.src), *recipe.params])
        #else:
        #    recipe.params = Params([*self.parse_func_params(recipe.src, recipe.func_type), *recipe.params])

        func_name = recipe.name
        #if isinstance(recipe.node, Rule):
        #    func_name = self.rule_func_name(recipe.node)
        #else:
        #    func_name = recipe.node.uniq_name(recipe.outer_call.assigned_name)
        #base = (
        #    Code('void') if recipe.node.func_type.always_true
        #        else self.bool_type() if recipe.src.func_type.returns_status
        #        else recipe.type.val_type)
        
        recipe.func_name = TypedName(
            func_name,
            recipe.outer_call.type,
            )

    def parse_recipe(self, recipe: ParseRecipe) -> None:
        # Save the recipe, to generate the code at file scope later.
        
        self.pending_recipes.append(
            recipe
        )


class CParserGenerator(ParserGenerator, _Traits):
    def __init__(
        self,
        grammar: grammar.Grammar,
        tokens: Dict[int, str],
        exact_tokens: Dict[str, int],
        non_exact_tokens: Set[str],
        file: Optional[IO[Text]],
        debug: bool = False,
        skip_actions: bool = False,
        ):
        self.skip_actions = skip_actions
        self.debug = debug
        super().__init__(grammar, tokens, exact_tokens, non_exact_tokens, file)
        self.token_types = {name: type for type, name in tokens.items()}
        self._varname_counter = 0
        self.cleanup_statements: List[str] = []
        self.pending_recipes: List[Callable[[], None]] = []

    def generate(self, filename: str) -> None:

        super().generate(filename)

        basename = os.path.basename(filename)
        self.print(self.comment(f"@generated by pegen from {basename}"))
        header = self.grammar.metas.get("header", EXTENSION_PREFIX)
        if header:
            self.print(header.rstrip("\n"))
        trailer = self.grammar.metas.get("trailer", None)
        start_name = ObjName("start")
        if not trailer and start_name in self.grammar.rules:
            trailer = EXTENSION_SUFFIX
            self.print('#include "pegen.h"')
        subheader = self.grammar.metas.get("subheader", "") or self.grammar.metas.get("subheader_c", "")
        if subheader:
            self.print(subheader)
        self._setup_keywords()
        self._setup_soft_keywords()
        for i, (rulename, rule) in enumerate(self.rules.items(), 1000):
            rule.memo_key = i
            comment = "Left-recursive" if rule.left_recursive else ""
            if rule.leader: comment += " leader"
            if comment: comment = f"  {self.comment(comment)}"
            self.print(f"#define {rulename}_type {i}{comment}")
        self.print()
        for rulename, rule in self.rules.items():
            if rule.type:
                type = rule.type
            else:
                type = None
                #type = self.default_type()
            params = rule.type.params
            params_str = len(params) and f", ({params.in_func()})" or ""
            self.print(
                f"_PyPegen_DECL_RULE({rulename}{params_str})")
            #self.fwd_decl_func(f"_{rulename}_rule", "ParseStatus",
            #                   self.parse_func_params(rule.type.params, ParseFunc))
        self.print()
        for rulename, rule in list(self.rules.items()):
            self.print()
            if rule.left_recursive:
                if rule.leader:
                    self.print(comment="Left-recursive leader")
                else:
                    self.print(comment="Left-recursive")
            self.print(comment=str(rule))
            if rule._error:
                self.print(comment="ERROR OCCURRED")
                #with self.enter_scope(f"_PyPegen__RULE(_{rule.name})", "{}"):
                #    self.print("return false;")

                #continue
            self.gen_node(rule)
            self.gen_pending_recipes()
        if self.skip_actions:
            mode = 0
        else:
            mode = int(self.rules[start_name].type == "mod_ty") if start_name in self.rules else 1
            if mode == 1 and self.grammar.metas.get("bytecode"):
                mode += 1
        modulename = self.grammar.metas.get("modulename", "parse")
        if trailer:
            self.print(trailer.rstrip("\n") % dict(mode=mode, modulename=modulename))

    def add_return(self, ret_val: str) -> None:
        for stmt in self.cleanup_statements:
            self.print(stmt)
        self.print(f"return {ret_val};")

    def fwd_decl_func(self, name: str, type: str, params: Params | None) -> None:
        params_str = params and params.in_func() or self.empty_params()
        self.print(f"static {type} {name}({params_str});")

    def _group_keywords_by_length(self) -> Dict[int, List[Tuple[str, int]]]:
        groups: Dict[int, List[Tuple[str, int]]] = {}
        for keyword_str, keyword_type in self.keywords.items():
            length = len(keyword_str)
            if length in groups:
                groups[length].append((keyword_str, keyword_type))
            else:
                groups[length] = [(keyword_str, keyword_type)]
        return groups

    def _setup_keywords(self) -> None:
        n_keyword_lists = (
            len(max(self.keywords.keys(), key=len)) + 1 if len(self.keywords) > 0 else 0
        )
        self.print(f"static const int n_keyword_lists = {n_keyword_lists};")
        groups = self._group_keywords_by_length()
        self.print("static KeywordToken *reserved_keywords[] = {")
        with self.indent():
            num_groups = max(groups) + 1 if groups else 1
            for keywords_length in range(num_groups):
                if keywords_length not in groups.keys():
                    self.print("(KeywordToken[]) {{NULL, -1}},")
                else:
                    self.print("(KeywordToken[]) {")
                    with self.indent():
                        for keyword_str, keyword_type in groups[keywords_length]:
                            self.print(f'{{"{keyword_str}", {keyword_type}}},')
                        self.print("{NULL, -1},")
                    self.print("},")
        self.print("};")

    def _setup_soft_keywords(self) -> None:
        soft_keywords = sorted(self.soft_keywords)
        self.print("static char *soft_keywords[] = {")
        with self.indent():
            for keyword in soft_keywords:
                self.print(f'"{keyword}",')
            self.print("NULL,")
        self.print("};")

    def _should_memoize(self, node: Rule) -> bool:
        return node.memo and not node.left_recursive

    def gen_rule(self, rule: Rule, **kwds) -> None:
        """ Writes the full definition of the given Rule. """

        rhs = rule.flatten()

        #result_type = self.default_type(rule.type)

        if rule.name.string.endswith("without_invalid"):
            with self.indent():
                self.print("int _prev_call_invalid = _p->call_invalid_rules;")
                self.print("_p->call_invalid_rules = 0;")
                self.cleanup_statements.append("_p->call_invalid_rules = _prev_call_invalid;")

        # Create the rule return as local variable.
        #self.print("ParseStatus _res;")

        # Capture pointers to the rule return and rule parameters (if any) in local array.
        if rule.max_local_vars:
            dim = str(rule.max_local_vars) if rule.max_local_vars > len(rule.type.params) else ""
            with self.enter_scope(f"void * _local_values [{dim}] =", '{};'):
                #self.print("& _res,")
                for param in rule.type.params:
                    self.print(f"& {param.name},")

        # Create the rule descriptor.
        child_name = rule.rhs.parse_recipe.func_name
        rhs_str = self.str_value(str(rule.rhs))
        type = rule.type
        if not type.return_type:
            type.return_type = ObjType(self.default_type())
        self.print(f"_PyPegen_DECL_RULE_DESCR(_rule_descriptor, {rule.name}, {type.return_type}, \"{rhs_str}\");")

        if rule.name.string.endswith("without_invalid"):
            self.cleanup_statements.pop()

    def gen_pending_recipes(self) -> None:
        """ Recursively execute the pending recipes.
        Any recipe can add more pending recipes.
        """
        recipes = self.pending_recipes[:]
        self.pending_recipes[:] = []
        for recipe in recipes:
            recipe()
            self.gen_pending_recipes()

    def gen_rhs_descriptors(self, rhs: Rhs, alt_names: List[str]) -> Tuple[str, Callable[[], Nonr]]:
        """ Generate code to parse the individual alts and create descriptor(s) for them.
        Return the name of the descriptor(s) variable and the function to generate the code.
        """
        def gen() -> None:

            def do_alt(alt: Alt, term: str = '') -> None:
                func_name = alt.uniq_name()
                alt_name = alt.parse_recipe.name
                alt_str = self.str_value(str(alt))
                self.print(f'{{{func_name}, "{alt_name}", "{alt_str}"}}{term}')

            if len(rhs.alts) == 1:
                name = alt_names[0]
                with self.enter_scope(f"RuleAltDescr _alt_descriptor ="):
                    do_alt(rhs.alts[0], ';')
            else:
                names = []
                # Make descriptor table.
                with self.enter_scope(f"RuleAltDescr _alt_descriptors[] =", '{};'):
                    for alt in rhs.alts:
                        do_alt(alt, ',')

        if len(rhs.alts) == 1:
            name = '&_alt_descriptor'
        else:
            name = '_alt_descriptors'
        return name, gen

    def setup_parent_name(self, name: str, var: TypedName, node: GrammarNode) -> str | None:
        """ Returns declaration and initialization of local variable of node's parent,
        but only if the name is not also a local variable of the node.
        """

        if node.parent.local_env.lookup(name) is var:
            ptr_type = var.decl_var('*')
            index = node.parent.local_env.index(name)
            return self.setup_var('_PyPegen_GET_GLOB', name, var, index)
        # Node has own local variable.
        return None

    def setup_local_name(self, item: VarItem) -> str | None:
        """ Returns declaration and initialization of local variable.
        with value of a parameter of given rule.
        If the var is new to the given Node, its address is stored in the Parser.
        A ParseResultPtr is also defined.
        """

        if not item.item.has_result:
            return None
        var: TypedName = TypedName(
            item.assigned_name,
            item.type,
            )
        if not item.var_name:
            # An anonymous VarItem.
            #   Just declare the variable.
            return self.setup_var('_PyPegen_DECL_LOC', var.name, var)
        else:
            index = item.parent.local_env.index(var.name)
            return self.setup_var('_PyPegen_ADD_GLOB', var.name, var, index)

    def setup_var(self, func: str, name: str, var: TypedName, index: int = None) -> str | None:
        return_type = var.return_type or ObjType(self.default_type())
        index_str = index is not None and f"{index}, " or ''
        params_str = return_type.callable and f", {var.func_params()}" or ''
        func = return_type.callable and f"{func}_FUN" or f"{func}_VAR"
        return f"{func}({index_str}{name}, {return_type}{params_str});"

    @staticmethod
    def gen_cast(dst_type: str, src_type: str = 'void * *', *, sep: str = ' ') -> str:
        if not dst_type or dst_type == src_type: return ''
        return f"({dst_type}){sep}"

    @staticmethod
    def gen_var_ptr(item: TypedName) -> str:
        """ Where the pointer to the value is stored in the parser. """
        return f"_p->local_values[{item.local_env.index(item.var_name.name)}]"

    def gen_copy_local_vars(self, node: GrammarTree) -> None:
        """ Code to define inherited names and set their values stored in the parser. """
        items = list(node.local_env.items())
        vars_used = set(node.vars_used())
        for name, var in items:
            # Check if the name is actually used.
            if name in vars_used:
                self.print(self.setup_parent_name(name, var, node))
        x = 0

    def gen_return_var(self, name: TypedName) -> None:
        """ Code to define local variable for the return value.
        No pointer is stored in the parser. """
        self.print(f"{name.decl_var()};")

    def gen_alt(self, alt: Alt) -> str:
        """ Extra code to generate body of Alt parser.
        Returns the return value.
        """

        def gen() -> str:
            parent_vars = set(alt.parent.local_env)
            alt_vars = set(alt.var_names())

            for item in alt.items():
                self.print(self.setup_local_name(item))
            extra = alt.action and "EXTRA" in str(alt.action)
            if extra:
                self.gen_start(alt)

            # Generate a function for each item, using the corresponding variable name
            for item in alt.items():
                self.print(comment=f"{item}")

                self.gen_alt_item(item)

            self.print(comment="parse succeeded.")

            return_type = alt.return_type or ObjType(self.default_type())
            # Prepare to emit the alt action and do so
            if extra:
                self.print("_PyPegen_EXTRA_END(_p);")
            # If the alt returns a different type from the action, use a cast.
            if return_type.callable:
                self.print(f"_PyPegen_RETURN_FUN(_ppRes, {self.gen_cast(return_type.val_type, alt.return_type)}({alt.action_expr}), {return_type.val_type}, {return_type.params});")
            else:
                self.print(f"_PyPegen_RETURN_VAR(_ppRes, {self.gen_cast(return_type.val_type, alt.return_type)}({alt.action_expr}), {return_type.val_type});")

        invalid = len(alt) == 1 and str(alt[0]).startswith("invalid_")
        if invalid:
            self.print(f"if (!_p->call_invalid_rules) return {self.bool_value(True)};")
        gen()
        return ""


    def gen_alt_item(self, item: VarItem) -> None:
        """ Code which parses a single item in an alt.
        Exits the alt if the parse fails.
            It is up to the caller of the alt to reset the mark.
        Otherwise assigns the result to a variable.
        """

        assert isinstance (item, VarItem)

        # The item node has a recipe to get its value.
        
        recipe = item.parse_recipe
        name: ObjName = item.assigned_name

        item_type = recipe.value_type()
        rawname = f"_result_{name}"
        src_type = recipe.src.func_type
        if recipe.mode is recipe.Inl:
            expr = recipe.outer_call.value_expr(assigned_name=item.assigned_name)
            if not recipe.outer_call.func_type.returns_status:
                expr = f"{item.name} = {expr}"
        elif recipe.mode is recipe.Loc:
            expr = f"{item.name} = {recipe.inner_call.value_expr(assigned_name=item.assigned_name)}"
        elif recipe.node.local_src:
            expr = f"{recipe.inner_call.value_expr(assigned_name=item.assigned_name)}"
        else:
            expr = recipe.outer_call.value_expr(assigned_name=item.assigned_name)
            if not src_type.returns_status:
                expr = f"{item.name} = {expr}"
        fail_value = self.bool_value(False)
        if src_type.always_true:
            if src_type.use_parser:
                self.print(f"if (({expr}), _p->error_indicator) return {fail_value};")
            else:
                self.print(f"{expr};")
        elif name in item.local_env:
            self.print(f"if (!({expr})) return {fail_value};")
        elif recipe.node.func_type.always_true:
            self.print(f"{expr};")
        else:
            self.print(f"if (!({expr})) return {fail_value};")

    # Descriptions of helper parsing functions...

    @functools.cached_property
    def parse_rule(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_rule', None,
            ('rule', 'RuleDescr *'),
            ('vars', 'void **'),
        )
    @functools.cached_property
    def parse_rule_memo(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_memo_rule', None,
            ('rule', 'RuleDescr *'),
            ('vars', 'void **'),
        )
    @functools.cached_property
    def parse_rule_recursive(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_recursive_rule', None,
            ('rule', 'RuleDescr *'),
            ('vars', 'void **'),
        )
    @functools.cached_property
    def parse_alt(self) -> ParseSource:
        return self.make_parser_name(
        '_PyPegen_parse_alt', None,
        ('alt', 'const RuleAltDescr *'),
        )
    @functools.cached_property
    def parse_alts(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_PARSE_ALT_ARRAY', None,
            ('alts', 'const RuleAltDescr []'),
        )
    @functools.cached_property
    def parse_NAME(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_name', 'expr_ty',
        )
    @functools.cached_property
    def parse_NUMBER(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_number_token', 'expr_ty',
            func_type=parse_recipe.ParseData,
        )
    @functools.cached_property
    def parse_STRING(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_string', 'Token *',
        )
    @functools.cached_property
    def parse_OP(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_op', 'Token *',
        )
    @functools.cached_property
    def parse_TYPE_COMMENT(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_type_comment_token', 'Token *',
            func_type=parse_recipe.ParseData,
        )
    @functools.cached_property
    def parse_SOFT_KEYWORD(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_any_soft_keyword', 'Token *',
        )
    @functools.cached_property
    def parse_token(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_token', 'Token *',
            ('type', 'int'),
            )
    @functools.cached_property
    def parse_char(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_char', 'Token *',
            ('c', 'char'),
        )
    @functools.cached_property
    def parse_forced(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_forced', None,
            ('item', 'ParseFunc *'),
            ('expected', 'const char *'),
            func_type=parse_recipe.ParseTrue,
        )
    @functools.cached_property
    def parse_soft_keyword(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_soft_keyword', 'Token *',
            ('keyword', 'const char *'),
        )
    def parse_repeat(self, elem: ParseExpr) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_repeat', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            ('item_size', 'size_t'),
            ('repeat1', 'int'),
        )
    def parse_gather(self, elem: ParseExpr) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_gather', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            ('item_size', 'size_t'),
            ('sep', 'ParseFunc *'),
            ('repeat1', 'int'),
        )
    def parse_opt(self, elem: ParseExpr) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_opt', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            ('item_size', 'size_t'),
        )
    @functools.cached_property
    def parse_lookahead(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_lookahead', None,
            ('positive', 'ParseStatus'),
            ('atom', 'ParseTest *'),
            func_type=parse_recipe.ParseTest,
        )
    @functools.cached_property
    def parse_cut(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_cut', None,
            func_type=parse_recipe.ParseVoid,
        )

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
