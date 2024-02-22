from __future__ import annotations

import ast
import os.path
import contextlib
import re
from dataclasses import dataclass, field
from enum import Enum
from operator import attrgetter
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

    _PyPegen_DECL_LOCAL(res, void* _type_res);

    if (_start_rule(_p, &_ptr_res))
        return res;
    // Parse failed
    return NULL;
}
"""


class NodeTypes(Enum):
    NAME_TOKEN = 0
    NUMBER_TOKEN = 1
    STRING_TOKEN = 2
    GENERIC_TOKEN = 3
    KEYWORD = 4
    SOFT_KEYWORD = 5
    CUT_OPERATOR = 6


BASE_NODETYPES = {
    "NAME": NodeTypes.NAME_TOKEN,
    "NUMBER": NodeTypes.NUMBER_TOKEN,
    "STRING": NodeTypes.STRING_TOKEN,
    "SOFT_KEYWORD": NodeTypes.SOFT_KEYWORD,
}


class _Traits(TargetLanguageTraits):
    from cparser import CParser
    from cparser.visitor import ASTVisitor
    #from pegen.cparser import CParser
    #from pegen.cparser.visitor import ASTVisitor

    c_parser: CParser = None

    def comment_leader(self) -> str:
        return '// '

    def default_header(self) -> str:
        return ''

    def default_trailer(self) -> str:
        return ''

    def default_type(self, type: Code = None) -> Code:
        return type or ValueCode('void *')

    def circular_type(self) -> ValueCode:
        return ValueCode('circ_ty')

    def default_value(self, value: str = None) -> str:
        return value or 'NULL'

    def default_params(self) -> str:
        return 'void'

    def no_value_type(self) -> str:
        return 'void'

    def parser_param(self) -> Param:
        return Param(TypedName('_p', 'Parser *'))

    def parse_result_ptr_param(self, has_value: bool, assigned_name: str = None) -> Param:
        name = has_value and (assigned_name and f"&_ptr_{assigned_name}" or "_ppRes") or "NULL"
        return Param(TypedName(name, ObjType('ParseResultPtr *')))

    def return_param(self, type: str = None) -> Param:
        return Param(TypedName('_ppRes', f'{self.default_type(type)} *'))

    def parse_func_params(self, func_type: ParseFuncType = None) -> Params:
        params = []
        if not func_type or func_type.use_parser:
            params.append(TypedName('_p', 'Parser *'))
        if not func_type or func_type.use_res_ptr:
            params.append(TypedName('_ppRes', 'ParseResultPtr *'))
        return Params(params)

    def parse_func_args(self, recipe: ParseRecipe) -> Args:
        args = []
        if recipe.mode is not recipe.Loc:
            src = recipe.src
            if src.func_type.use_parser:
                args.append('_p')
            if src.func_type.use_res_ptr:
                args.append(self.parse_result_ptr_param(
                    recipe.outer_call.val_type.has_value,
                    recipe.outer_call.assigned_name
                    ).name.string)
        return Args(args)

    def rule_func_name(self, rule: Rule) -> ObjName:
        return ObjName(f"_{rule.name}_rule")

    def rule_recipe(self, rule: Rule, src: ParseRecipeSource) -> ParseRecipe:
        """ Create a recipe for a Rule. """
        return ParseRecipeExternal(
            rule, rule.name, src, '&_rule_descriptor',
            rule.local_env.count and "_local_values" or self.default_value(),
            params=rule.params,
            extra=lambda: self.gen_rule(rule),
            inlines = [rule.rhs],
            value_type=rule.type,
            )
        return ParseRecipeExternal(
            rule, rule.name, src,
            '_rhs',
            func_type=ParseFunc,
            extra=lambda: self.gen_rule(rule),
            inlines = [rule.rhs],
            value_type=rule.val_type,
            )

    def sequence_recipe_args(self, seq: Seq, *seq_args: Args) -> Args:
        """ Actual arguments for a Seq parse function.
        seq_args varies with the type of sequence.  It's an initial subset of:
            * repeat1
            * sep
        """
        return Args([
            Arg(inline=seq.elem),
            *seq_args,
            ])

    def format_parse_call(self, call: ParseCall) -> str:
        return f"{call.name}{self.format_arg_list(call.args)}"

    def format_args(self, args: ArgsList) -> str:
        return ''.join(map(self.format_arg_list, args))

    def format_arg_list(self, args: Args) -> str:
        return f"({', '.join(map(str, args))})"

    def parse_value_expr(
            self, recipe: ParseRecipe, name: ObjName, args: Args, params: Params, *,
            func_type: ParseFuncType = None,
            assigned_name: ObjName = None,
        ) -> str:
        """ An expression which produces the desired parse result. """
        return f"{name}{args}"

    def cast(self, value: str, to_type: TypedName) -> str:
        return f"({to_type.typed_name()}) ({value})"

    def cut_value(self, value: str = None) -> str:
        return '_PyPegen_cut_sentinel'

    def bool_value(self, value: bool) -> str:
        return f"{str(bool(value)).lower()}"

    def bool_type(self) -> Code: return ValueCode('ParseStatus')

    def gen_subscr_type(self, val_type: str, *subs: Type) -> str:
        """ How the value type, with subscripts, appears in target languge. """
        # The subscripts are ignored.  Just show the basic type.
        return val_type

    def str_value(self, value: bool) -> str:
        return value.replace('"', r'\"')

    def target_names(self, code: str) -> set[str]:
        """ All the identifiers found in given target language code.
        This might be a call argument, an object type, or an alt action.
        """
        class IDVisitor(self.ASTVisitor):
            """Visitor that collects all identifiers in the code."""

            def __init__(self, ast):
                self.names = set()
                self.depth = 0
                self.visit(ast)

            def visit_id(self, node, debug=False):
                self.names.add(node.value)
                print(f"{'   ' * self.depth}name = {node.value}")

            def visit(self, ast, debug=False):
                print(f"{'   ' * self.depth}{ast.symbol.name}")
                self.depth += 1
                super().visit(ast)
                self.depth -= 1

        if not self.c_parser:
            self.c_parser = self.CParser()
            #self.c_parser = self.CParser(start_symbol='assignment_exp')
        parser = self.c_parser
        ast = parser.parse(code, debug=True)
        visitor = IDVisitor(ast)
        return visitor.names

    def forward_declare_inlines(self, recipe: ParseRecipe) -> None:
        x = 0
        for node_recipe in recipe.inline_recipes():
            self.print(f"static {self.format_decl_func_fwd(node_recipe.outer_call)};")
            #self.print(f"static {node_recipe.func_name.decl_func(fwd=True)};")

    def enter_function(
        self, func: ParseCall, comment: str = '', **kwds
        ) -> Iterator:
        return self.enter_scope(
            self.format_decl_func(func, **kwds),
            '{}',
            comment=comment
            )

    def format_decl_func(self, func: ParseCall) -> str:
        if isinstance(func.parent.node, Rule):
            return self.format_rule(func.parent.node)
        func_type = func.parent.node.func_type
        val_type = (
            ObjType('void') if func_type.always_true
            else self.bool_type() if func_type.returns_status
            else func.val_type
            )
        params = [*self.parse_func_params(func_type), *func.params]
        return f"{val_type} {func.name}{self.format_params(params)}"

    def format_decl_func_fwd(self, func: ParseCall) -> str:
        return f"{func.parent.node.func_type._name} {func.name}"

    def format_params(
        self, params: Params,
        brackets: str = '()',
        hide_names: bool = False,
        hide_param_names: bool = False,
        ) -> str:
        """ A parameter list, with optional () or other brackets. """
        parts: list[str] = []
        for param in params:
            parts.append(self.format_typed_name
                (param,
                    hide_names=hide_names,
                    hide_param_names=hide_param_names,
                    )
                )
        return f"{brackets[:1]}{', '.join(parts)}{brackets[1:]}"

    def format_typed_name(self, name: TypedName, **kwds) -> str:
        """ A C declaration for a variable, typedef or function parameter. """
        return self.format_type_and_name(name.type, name.name, **kwds)

    def format_type_and_name(
            self,
            typ: Type,
            name: ObjName = None,
            ptr: bool = False,                  # Declare as pointer to type.  Functions are always pointers.
            hide_names: bool = False,           # Names hidden at all depths.
            hide_param_names: bool = False,     # Names of function params hidden at all depths.`
        ) -> str:
        """ A C declaration for a variable, typedef or function parameter.
        The name is optional.
        """

        decl: str = ''
        if typ.callable:
            # Build the declarator expression, possibly at multiple levels.
            # The first parameter list is the innermost.
            # All parameters are formatted recursively.
            if name and not hide_names:
                decl = str(name)
            for params in typ.param_lists():
                # Wrap each declarator level in (* ... )(params)
                params_str = self.format_params(
                    params, hide_names=hide_names or hide_param_names)
                decl = f"(*{decl}){params_str}"
        else:
            if ptr:
                decl = '*'
            if name and not hide_names:
                decl += str(name)

        return f"{typ.val_type} {decl}"

    def format_type(
        self, typ: Type, **kwds,
        ) -> str:
        """ A C declaration for a type.
        Equivalent to declaring a variable of that type but omitting the name.
        """
        return self.format_type_and_name(typ, **kwds)

    def format_rule(self, rule: Rule) -> str:
        if len(rule.params):
            params = f", {rule.gen.format_params(rule.params, hide_param_names=True)}"
        else:
            params = ''
        return f"_PyPegen_RULE({rule.name}{params})"

    def test(self):
        test = Type('void *',
                    Params((
                        TypedName('p11', ObjType('int')),
                        TypedName('p12', ObjType('char')),
                    )),
                    Params((
                        TypedName('p21', ObjType('int *')),
                        TypedName('p22', ObjType('char *')),
                    )),
                   )
        tn = TypedName('foo', test)
        result = self.format_typed_name(tn)
        result = self.format_typed_name(tn, hide_names=True)
        result = self.format_typed_name(tn, hide_param_names=True)
        result = self.format_typed_name(tn, hide_names=True, hide_param_names=True)
        result = self.format_type(tn.type)
        result = self.format_type(tn.type, hide_param_names=True)
        tn = TypedName('foo', ObjType('void'))
        result = self.format_typed_name(tn)
        result = self.format_typed_name(tn, ptr=True)
        result = self.format_type(tn.type)
        result = self.format_type(tn.type, ptr=True)
        x = 0

    def gen_start(self, alt: Alt) -> None:
        """ Generate code to set starting line and col variables if required by action.
        Call before starting to parse alt if EXTRA is, or might be, contained in the action.
        """
        self.print()
        if alt.action and "EXTRA" in alt.action:
            self.print("_PyPegen_EXTRA_START(_p);")

    def gen_action(self, alt: Alt) -> _Traits.Action:
        """ Generate code for the action in the Alt. """
        if alt.action and "EXTRA" in alt.action:
            self.print("_PyPegen_EXTRA_END(_p);")
        if self.skip_actions:
            return self.dummy_action()
        elif alt.action:
            return self.action(alt)
        else:
            return self.default_action(alt)

    def fix_parse_recipe(self, recipe: ParseRecipe) -> None:

        recipe.expr_name = recipe.src.name or recipe.name
        #if isinstance(recipe.src, Rule):
        #    recipe.expr_name = f'_{recipe.expr_name}_rule'

        if recipe.mode is recipe.Rule:
            recipe.params = Params([*self.parse_func_params(recipe.src.func_type), *recipe.params])
        elif isinstance(recipe.node, Rule):
            recipe.params = Params([*self.parse_func_params(recipe.src.func_type), *recipe.params])
        #elif recipe.mode is recipe.Loc:
        #    pass
        else:
            recipe.params = Params([*self.parse_func_params(recipe.src.func_type), *recipe.params])
            #recipe.params = self.inline_params(recipe.src)

        func_name = recipe.name
        if isinstance(recipe.node, Rule):
            func_name = f'_{func_name}_rule'
        #elif recipe.mode is recipe.Loc:
        #    pass
        else:
            func_name = recipe.node.uniq_name()
        val_type = (
            ObjType('void') if recipe.node.func_type.always_true
            else self.bool_type() if recipe.src.func_type.returns_status
            else recipe.src.type
            )
        recipe.func_name = TypedName(
            func_name,
            Type(val_type, recipe.params, parent=recipe)
            )
        recipe.func_name.initialize(recipe)

    def parse_recipe(self, recipe: ParseRecipe, **kwds) -> None:
        # Save the recipe, to generate the code at file scope later,
        # but only if it's a not a local variable.
        
        if True:
        #if recipe.mode in (recipe.Inl, recipe.Ext, recipe.Rule):
            self.pending_recipes.append(
                lambda: recipe(self, **kwds)
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
        self.test()

    def add_return(self, ret_val: str) -> None:
        for stmt in self.cleanup_statements:
            self.print(stmt)
        self.print(f"return {ret_val};")

    def fwd_decl_rule(self, name: str, params: Params | None) -> None:
    #def fwd_decl_rule(self, name: str, *params: Params | None) -> None:
        params_str = len(params) and f", {params.in_func()}" or ','
        self.print(f"_PyPegen_DECL_RULE({name}{params_str});")

    def generate(self, filename: str) -> None:

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
                type = self.default_type()
            self.fwd_decl_rule(f"{rulename}", rule.type.params)
            #self.fwd_decl_rule(f"{rulename}", *rule.type.param_lists())
        self.print()
        for rulename, rule in list(self.rules.items()):
            self.print()
            if rule.left_recursive:
                if rule.leader:
                    self.print(comment="Left-recursive leader")
                else:
                    self.print(comment="Left-recursive")
            self.print(
                comment=f"{rule.in_grammar()}: {rule.rhs.show()}")
            if rule._deleted:
                # Generate some dummy code for deleted recipe.
                with self.enter_scope(self.format_rule(rule), '{};'):
                    self.print(comment="RULE DELETED.")
            else:
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

        result_type = self.default_type(rule.type)

        if rule.name.string.endswith("without_invalid"):
            with self.indent():
                self.print("int _prev_call_invalid = _p->call_invalid_rules;")
                self.print("_p->call_invalid_rules = 0;")
                self.cleanup_statements.append("_p->call_invalid_rules = _prev_call_invalid;")

        # Create the rule return as local variable.
        #self.print("ParseStatus _res;")

        # Capture pointers to the rule return and rule parameters (if any) in local array.
        if rule.max_local_vars:
            dim = str(rule.max_local_vars) if rule.max_local_vars > len(rule.params) else ""
            with self.enter_scope(f"void * _local_values [{dim}] =", '{};'):
                #self.print("& _res,")
                for param in rule.params:
                    self.print(f"& {param.name},")

        # Create the rule descriptor.
        child_name = rule.rhs.parse_recipe.func_name
        rhs_str = self.str_value(str(rule.rhs))

        self.print(f"_PyPegen_RULE_DESCR(_rule_descriptor, {rule.name}, {rule.val_type}, \"{rhs_str}\");")
        #with self.enter_scope(f"RuleDescr _rule_descriptor ="):
            
        #    rhs_str = self.str_value(str(rule.rhs))
        #    self.print(f'{{{child_name}, "{rule.name}", "{rhs_str}"'
        #               f'}};'
        #               )

        t = rule.val_type
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

    def gen_rhs_descriptors(
            self,
            rhs: Rhs,
            alt_names: List[str]
            ) -> Tuple[str, Callable[[], None]]:
        """ Generate code to parse the individual alts and create descriptor(s) for them.
        Return the name of the descriptor(s) variable and the function to generate the code.
        """
        def gen() -> str:
            """ Generate the code to parse the Rhs.
            Returns empty string to indicate that the return value was generated.
            """

            def do_alt(alt: Alt, term: str = '') -> None:
                func_name = alt.uniq_name()
                alt_name = alt.parse_recipe.name
                alt_str = self.str_value(str(alt))
                self.print(f'_PyPegen_ALT_DESCR({func_name}, "{alt_str}"){term}')
                #self.print(f'{{{func_name}, "{func_name}", "{alt_str}"}}{term}')

            if len(rhs) == 1:
                name = alt_names[0]
                with self.enter_scope(f"_PyPegen_RETURN_PARSE_ALTS", '();'):
                #with self.enter_scope(f"RuleAltDescr _alt_descriptor ="):
                    do_alt(rhs[0], '')
            else:
                names = []
                # Make descriptor table.
                with self.enter_scope(f"_PyPegen_RETURN_PARSE_ALTS", '();'):
                #with self.enter_scope(f"RuleAltDescr _alt_descriptors[] =", '{};'):
                    for i, alt in enumerate(rhs):
                        do_alt(alt, '' if i == len(rhs) - 1 else ',')
            return ''

        if len(rhs) == 1:
            name = '&_alt_descriptor'
        else:
            name = '_alt_descriptors'
        return name, gen

    def gen_setup_parent_name(self, name: str, var: TypedName, node: GrammarNode) -> None:
        """ Generates declaration and initialization of local variable of node's parent,
        but only if the name is not also a local variable of the node.
        """

        if node.parent.local_env.lookup(name) is var:
            call: str = self.format_typed_call(
                '_PyPegen_GET_GLOBAL',
                var.assigned_name or var.name,
                var.return_type,
                node.local_env.index(name),
                )
            self.print(call)

    def gen_setup_local_name(self, item: VarItem) -> None:
        """ Generates declaration and initialization of local variable.
        with value of a parameter of given rule.
        If the var is new to the given Node, its address is stored in the Parser.
        A ParseResultPtr is also defined;
        """

        if not item.func_type.has_result or not item.val_type.has_value:
            return 
        if not item.var_name or item not in item.parent.codes.objs:
            # An anonymous VarItem.
            #   Just declare the variable.
            self.print(self.format_typed_call(
                '_PyPegen_DECL_LOCAL', item.assigned_name, item.return_type))
        else:
            index = item.parent.local_env.index(item.assigned_name)
            self.print(self.format_typed_call(
                '_PyPegen_ADD_GLOBAL', item.assigned_name, item.return_type, index))

    @staticmethod
    def format_cast(
            dst_type: str, src_type: str = 'void * *', *,
            sep: str = ' ',
            expr: str = '',
            ) -> str:
        if not dst_type or str(dst_type) == str(src_type): return expr
        return f"({dst_type}){sep}{expr and f'({expr})'}"

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
                self.gen_setup_parent_name(name, var, node)
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
                self.gen_setup_local_name(item)
            if alt.action and "EXTRA" in alt.action:
                self.gen_start(alt)

            # Generate a function for each item, using the corresponding variable name
            for item in alt.items():
                self.print(comment=f"{item}")
                # Create local variables for the item values.

                self.gen_alt_item(item)

            self.print(comment="parse succeeded.")

            # If the alt returns a different type from the action, use a cast.
            #self.print(f"_PyPegen_RETURN_RESULT(_ppRes, {alt.type}, {self.format_cast(alt.type, ret_action.type)}{ret_action.expr});")
            self.print(self.format_return_result(
                '_PyPegen_RETURN', alt.action, alt.action_type,
                alt.return_type,
                hide_param_names=True)
            )
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
        name = item.assigned_name

        item_type = recipe.value_type()
        #var_type = self.default_type(info.vartype or item_type)
        rawname = f"_result_{name}"
        #rawtype = f"{var_type}"
        #if recipe.mode is recipe.Local:
        #    expr = recipe.func_name
        #else:
        #    expr = recipe.func_name
            #expr = f"{recipe.func_name}(_p)"
        cast = ''
        #cast = f"({var_type}) "
        #self.print(f"{item_type} {name};")
        #if var_type != rawtype:
        #    expr = f"({var_type}) {expr}"
        func_type = recipe.node.func_type
        if recipe.mode is recipe.Inl:
            expr = recipe.outer_call.value_expr()
            if not func_type.returns_status:
                expr = f"{item.name} = {expr}"
        elif recipe.mode is recipe.Loc:
            expr = f"{item.name} = {recipe.inner_call.value_expr()}"
        elif recipe.mode is recipe.Rule:
            expr = f"{recipe.inner_call.value_expr()}"
        else:
            expr = recipe.outer_call.value_expr()
            if not func_type.returns_status and func_type.has_result:
                expr = f"{item.name} = {expr}"
        fail_value = self.bool_value(False)
        if func_type.always_true:
        #if func_type.always_true and not func_type.has_result:
            self.print(f"if (({cast}{expr}), _p->error_indicator)")
        elif name in item.local_env:
            self.print(f"if (!{cast}{expr})")

        else:
            self.print(f"if (!({expr}))")
        with self.indent():
            self.print(f"return {fail_value};")

    #def generic_visit(self, node, **kwds) -> None:
    #    """ This is for a visit function not yet directly implemented. """
    #    assert 0, f"PythonParserGenerator has no visitor for {node!r}"

    def decl_inline(self, node: GrammarNode) -> None:
        """ Generate forward declaration of parser for the node, at file scope. """
        info = node.parse_recipe.func_info(self)
        self.print(f"static {info.type}{info.name}(NodeDescr *);")

    def gen_parse(self, recipe: ParseRecipe,
        **kwds
        ) -> None:
        self.forward_declare_inlines(recipe)

        return_type = recipe.src.type
        if isinstance(recipe.node, Rule):
            return_type = recipe.node.type
        if recipe.mode is not recipe.Loc or 0x00001:
            with self.enter_function(recipe.outer_call):
                #recipe.func_name,
                #node=recipe.node,
                #):
                for inline in recipe.inline_recipes():
                    # Expand the inline items.
                    self.gen_node(inline.node)
                self.gen_copy_local_vars(recipe.node)
                ####if not isinstance(recipe.node, (Rule, Alt)):
                ####    self.print(comment=str(recipe.node))
                call: str | None = None
                if recipe.extra:
                    call = recipe.extra()
                if call is None:
                    # Was not supplied by extra()
                    call = recipe.value_expr()

                    if not recipe.node.func_type.always_true:
                        if recipe.mode is recipe.Loc or 0x0000:
                            #call = f"_PyPegen_RETURN_RESULT(_ppRes, {recipe.node.type}, {call})"
                            call = self.format_return_result('_PyPegen_RETURN', call, recipe.src.type, recipe.node.return_type)
                        else:
                            call = f"return {call}"
                            #call = f"return {self.format_cast(recipe.node.type, return_type)}{call}"
                    call += ';'
                #if recipe.comment:
                #    call = f"{call}   {self.comment(recipe.comment)}"
                if call: self.print(call)

    def format_typed_call(
            self,
            func_pfx: str,
            name: ObjName | None,
            typ: Type,
            *args: Any,
            **kwds
        ) -> str:
        """ Generate a function call which varies with the given type.
        The given args are followed by name and type information.
        This type info is a declarator for a typedef named _type_{name].
        If name is None, it doesn't appear in the result,
        but '_type__return' is declared in the type info.
        """
        arg_strs: list[str] = list(map(str, args))
        if name: arg_strs.append(str(name))
        else: name = '_return'
        #arg_strs.append(self.format_typed_name(
        #    TypedName(f"_type_{name}", typ)))
        arg_strs.append(self.format_type_and_name(typ, f"_type_{name}", **kwds))
        return f"{func_pfx}({', '.join(arg_strs)});"

    def format_return_result(self, prefix: str, src: str, src_type: Type, dst_type: Type, **kwds) -> str:
        """ Generate a statement which returns the given value expression. """
        # _PyPegen_RETURN_VAR(_ppRes, (void*) (_r2), void*);
        #src = self.format_cast(dst_type, src_type, expr=src)
        return self.format_typed_call(prefix, None, dst_type, '_ppRes', src, **kwds)

    def action(self, alt: Alt) -> str:
        if self.debug:
            self.print(
                f'D(fprintf(stderr, "Hit with action [%d-%d]: %s\\n", _mark, _p->mark, "{alt}"));'
            )
        return self.Action(alt.action, ObjType("void *"))

    def default_action(self, alt: Alt) -> _Traits.Action:
        expr: str
        typ: str
        if self.debug:
            self.print(
                f'D(fprintf(stderr, "Hit with default action [%d:%d]: %s\\n", _mark, _p->mark, "{alt}"));'
            )
        vars = alt.all_vars()
        named_vars = tuple(filter(attrgetter('var_name'), vars))
        if 1 <= len(named_vars) < len(vars):
            # There are both named and anonymous items.  Use only the named ones.
            vars = named_vars
        if len(vars) > 1:

            expr = f"_PyPegen_dummy_name(_p, {', '.join(var.assigned_name.string for var in vars)})"
            typ = ObjType("expr_ty")
        else:
            if vars:
                cast = ""
                expr = f"{cast}{vars[0].name}"
                typ = vars[0].type
            else:
                expr = f"{self.default_value()}"
                typ = ObjType(self.default_type())
        return self.Action(expr, typ)

    def dummy_action(self) -> _Traits.Action:
        return Action("_PyPegen_dummy_name(_p)", "expr_ty")

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
        '_PyPegen_parse_alt', 'ParseStatus',
        ('alt', 'const RuleAltDescr *'),
        )
    @functools.cached_property
    def parse_alts(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_PARSE_ALT_ARRAY', 'Any',
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
            '_PyPegen_parse_number', 'expr_ty',
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
    def parse_CHAR(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_char', 'Token *',
        )
    @functools.cached_property
    def parse_TYPE_COMMENT(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_type_comment', 'Token *',
            #func_type=parse_recipe.ParseData,
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
            '_PyPegen_parse_specific_char', 'Token *',
            ('c', 'char'),
        )
    @functools.cached_property
    def parse_keyword(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_token', 'Token *',
            ('type', 'int'),
            )
    @functools.cached_property
    def parse_soft_keyword(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_soft_keyword', 'Token *',
            ('keyword', 'const char *'),
        )
    def parse_forced(self, node: ParseExpr) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_forced', node.return_type,
            ('item', 'ParseFunc *'),
            ('expected', 'const char *'),
            func_type=parse_recipe.ParseTrue,
        )
    @functools.cached_property
    def parse_call(self) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_CALL', 'void *',
            ('elem', 'ParseFunc *'),
        )
    def parse_repeat(self, elem: ParseExpr) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_repeat', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            ('repeat1', 'int'),
        )
    def parse_gather(self, elem: ParseExpr) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_gather', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            ('sep', 'ParseFunc *'),
            ('repeat1', 'int'),
        )
    def parse_opt(self, elem: ParseExpr) -> ParseSource:
        return self.make_parser_name(
            '_PyPegen_parse_opt', elem.return_type,
            #'_PyPegen_parse_opt', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            func_type=ParseTrue,
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
    ParseRecipeExternal,
    ParseFunc,
    ParseTrue,
    )

