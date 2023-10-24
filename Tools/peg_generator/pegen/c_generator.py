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

    _PyPegen_DECL_LOCAL_PARSE_RESULT(void *, res, );

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

    def default_type(self, type: Code = None) -> Code:
        return type or Code('void *')

    def default_value(self, value: str = None) -> str:
        return value or 'NULL'

    def default_params(self) -> str:
        return 'void'

    def parser_param(self) -> Param:
        return Param(TypedName('_p', Code('Parser *')))

    def parse_result_ptr_param(self, assigned_name: str = None) -> Param:
        name = assigned_name and f"&_ptr_{assigned_name}" or "_ppRes"
        return Param(TypedName(name, Code('ParseResultPtr *')))

    def return_param(self, type: str = None) -> Param:
        return Param(TypedName('_ppRes', Code(f'{self.default_type(type)} *')))

    def parse_func_params(self, name: TypedName = None) -> Params:
        params = []
        if not name or name.func_type.use_parser:
            params.append(TypedName('_p', Code('Parser *')))
        if not name or name.func_type.use_res_ptr:
            params.append(TypedName('_ppRes', Code('ParseResultPtr *')))
        return Params(params)

    def rule_func_name(self, rule: Rule) -> ObjName:
        return ObjName(f"_{rule.name}_rule")

    def cast(self, value: str, to_type: TypedName) -> str:
        return f"({to_type.typed_name()}) ({value})"

    def cut_value(self, value: str = None) -> str:
        return '_PyPegen_cut_sentinel'

    def bool_value(self, value: bool) -> str:
        return f"{str(bool(value)).lower()}"

    def bool_type(self) -> str: return Code('ParseStatus')

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
            self.print(f"static {node_recipe.func_name.decl_func(fwd=True)};")

    def enter_function(
        self, name: TypedName, comment: str = ''
        ) -> Iterator:
        return self.enter_scope(
            name.decl_func(),
            '{}',
            comment=comment
            )

    def gen_start(self, alt: Alt) -> None:
        """ Generate code to set starting line and col variables if required by action.
        Call before starting to parse alt if EXTRA is, or might be, contained in the action.
        """
        self.print()
        if alt.action and "EXTRA" in alt.action:
        #if alt.action and "EXTRA" in self.target_names(alt.action):
            self.print("int _start_lineno, _start_col_offset, _end_lineno, _end_col_offset;")
            self.print("_PyPegen_location_start(_p, &_start_lineno, &_start_col_offset);")

    def gen_action(self, alt: Alt) -> _Traits.Action:
        """ Generate code for the action in the Alt. """
        if alt.action and "EXTRA" in alt.action:
            self.print("_PyPegen_location_end(_p, &_end_lineno, &_end_col_offset);")
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
            recipe.params = Params([*self.parse_func_params(recipe.src), *recipe.params])
        elif isinstance(recipe.node, Rule):
            recipe.params = Params([*self.parse_func_params(recipe.src), *recipe.params])
        #elif recipe.mode is recipe.Loc:
        #    pass
        else:
            recipe.params = Params([*self.parse_func_params(recipe.src), *recipe.params])
            #recipe.params = self.inline_params(recipe.src)

        func_name = recipe.name
        if isinstance(recipe.node, Rule):
            func_name = f'_{func_name}_rule'
        #elif recipe.mode is recipe.Loc:
        #    pass
        else:
            func_name = recipe.node.uniq_name()
        recipe.func_name = TypedName(
            func_name,
            Code('void') if recipe.node.func_type.always_true
                else self.bool_type() if recipe.src.func_type.returns_status
                else recipe.src.type,
            recipe.params)

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
        super().__init__(grammar, tokens, exact_tokens, non_exact_tokens, file)
        self.token_types = {name: type for type, name in tokens.items()}
        self._varname_counter = 0
        self.debug = debug
        self.skip_actions = skip_actions
        self.cleanup_statements: List[str] = []
        self.pending_recipes: List[Callable[[], None]] = []

    def add_return(self, ret_val: str) -> None:
        for stmt in self.cleanup_statements:
            self.print(stmt)
        self.print(f"return {ret_val};")

    def fwd_decl_func(self, name: str, type: str, params: Params | None) -> None:
        params_str = params and params.in_func() or self.default_params()
        self.print(f"static {type} {name}({params_str});")

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
            self.fwd_decl_func(f"_{rulename}_rule", "ParseStatus", rule.parse_recipe.params)
        self.print()
        for rulename, rule in list(self.rules.items()):
            self.print()
            if rule.left_recursive:
                if rule.leader:
                    self.print(comment="Left-recursive leader")
                else:
                    self.print(comment="Left-recursive")
            self.print(comment=str(rule))
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

        self.print(f"DECL_RULE_DESCR(_rule_descriptor, {rule.name}, {rule.type}, \"{rhs_str}\");")
        #with self.enter_scope(f"RuleDescr _rule_descriptor ="):
            
        #    rhs_str = self.str_value(str(rule.rhs))
        #    self.print(f'{{{child_name}, "{rule.name}", "{rhs_str}"'
        #               f'}};'
        #               )

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

            if len(rhs) == 1:
                name = alt_names[0]
                with self.enter_scope(f"RuleAltDescr _alt_descriptor ="):
                    do_alt(rhs[0], ';')
            else:
                names = []
                # Make descriptor table.
                with self.enter_scope(f"RuleAltDescr _alt_descriptors[] =", '{};'):
                    for alt in rhs:
                        do_alt(alt, ',')

        if len(rhs) == 1:
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
            return (f"_PyPegen_GET_GLOBAL_PARSE_RESULT("
                    f"{var.type}, {name}, {var.func_params()}, {node.local_env.index(name)});"
                    )
            #return f"{var.decl_var()} = * {self.gen_cast(ptr_type)}&{self.gen_var_ptr(var)};"
        # Node has own local variable.
        return None

    def setup_local_name(self, item: VarItem) -> str | None:
        """ Returns declaration and initialization of local variable.
        with value of a parameter of given rule.
        If the var is new to the given Node, its address is stored in the Parser.
        A ParseResultPtr is also defined;
        """

        if not item.func_type.has_result:
            return None
        var: TypedName = TypedName(item.assigned_name.string, item.value_type(), item.params)
        if not item.var_name:
            # An anonymous VarItem.
            #   Just declare the variable.
            return f"_PyPegen_DECL_LOCAL_PARSE_RESULT({var.type}, {var.name}, {var.func_params()});"
        else:
            index = item.parent.local_env.index(var.name)
            return f"_PyPegen_ADD_GLOBAL_PARSE_RESULT({var.type}, {var.name}, {var.func_params()}, {index});"

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
            if alt.action and "EXTRA" in alt.action:
                self.gen_start(alt)

            # Generate a function for each item, using the corresponding variable name
            for item in alt.items():
                self.print(comment=f"{item}")
                # Create local variables for the item values.

                self.gen_alt_item(item)

            self.print(comment="parse succeeded.")

            # Prepare to emit the rule action and do so
            ret_action: self.Action = self.gen_action(alt)
            # If the alt returns a different type from the action, use a cast.
            self.print(f"_PyPegen_RETURN_RESULT(_ppRes, {alt.type}, {self.gen_cast(alt.type, ret_action.type)}{ret_action.expr});")

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
        if recipe.mode is recipe.Inl:
            expr = recipe.outer_call.value_expr()
            if not recipe.outer_call.func_type.returns_status:
                expr = f"{item.name} = {expr}"
        elif recipe.mode is recipe.Loc:
            expr = f"{item.name} = {recipe.inner_call.value_expr()}"
        else:
            expr = recipe.outer_call.value_expr()
            if not recipe.src.func_type.returns_status:
                expr = f"{item.name} = {expr}"
        fail_value = self.bool_value(False)
        if recipe.src.func_type.always_true:
            self.print(f"if (({cast}{expr}), _p->error_indicator) return {fail_value};")
        elif name in item.local_env:
            self.print(f"if (!{cast}{expr}) return {fail_value};")

        else:
            self.print(f"if (!({expr})) return {fail_value};")

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
        self.print()
        self.forward_declare_inlines(recipe)
        #lead = recipe.node.depth() * '    '
        #print(f"{lead}{recipe.node!r}")
        #print(f"{lead}{recipe.mode.name} {recipe.func_name}")

        return_type = recipe.src.type
        if isinstance(recipe.node, Rule):
            return_type = recipe.node.type
        if recipe.mode is not recipe.Loc or 0x00001:
            with self.enter_function(
                recipe.func_name,
                ):
                for inline in recipe.inline_recipes():
                    # Expand the inline items.
                    self.gen_node(inline.node)
                self.gen_copy_local_vars(recipe.node)
                if recipe.node is not recipe.alt:
                    self.print(comment=str(recipe.node))
                #if recipe.mode is recipe.Rule:
                #    self.gen_return_var(TypedName('_result'))
                call: str | None = None
                if recipe.extra:
                    call = recipe.extra()
                if call is None:
                    # Was not supplied by extra()
                    call = recipe.value_expr()

                    if not recipe.inner_call.func_type.always_true:
                        if recipe.mode is recipe.Loc or 0x0000:
                            call = f"_PyPegen_RETURN_RESULT(_ppRes, {recipe.node.type}, {call})"
                        else:
                            call = f"return {call}"
                            #call = f"return {self.gen_cast(recipe.node.type, return_type)}{call}"
                    call += ';'
                #if recipe.comment:
                #    call = f"{call}   {self.comment(recipe.comment)}"
                if call: self.print(call)

    def action(self, alt: Alt) -> str:
        if self.debug:
            self.print(
                f'D(fprintf(stderr, "Hit with action [%d-%d]: %s\\n", _mark, _p->mark, "{alt}"));'
            )
        return self.Action(alt.action, "void *")

    def default_action(self, alt: Alt) -> _Traits.Action:
        expr: str
        type: str
        vars = alt.all_vars()
        if len(vars) > 1:
            if self.debug:
                self.print(
                    f'D(fprintf(stderr, "Hit without action [%d:%d]: %s\\n", _mark, _p->mark, "{alt}"));'
                )
            expr = f"_PyPegen_dummy_name(_p, {', '.join(var.name.string for var in vars)})"
            type = "expr_ty"
        else:
            if self.debug:
                self.print(
                    f'D(fprintf(stderr, "Hit with default action [%d:%d]: %s\\n", _mark, _p->mark, "{node}"));'
                )
            if len(vars):
                cast = ""
                expr = f"{cast}{vars[0].name}"
                type = vars[0].type
            else:
                expr = f"{self.default_value()}"
                type = f"{self.default_type()}"
        return self.Action(expr, type)

    def dummy_action(self) -> _Traits.Action:
        return Action("_PyPegen_dummy_name(_p)", "expr_ty")

    # Descriptions of helper parsing functions...

    @functools.cached_property
    def parse_rule(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_rule', None,
            ('rule', 'RuleDescr *'),
            ('vars', 'void **'),
        )
    @functools.cached_property
    def parse_rule_memo(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_memo_rule', None,
            ('rule', 'RuleDescr *'),
            ('vars', 'void **'),
        )
    @functools.cached_property
    def parse_rule_recursive(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_recursive_rule', None,
            ('rule', 'RuleDescr *'),
            ('vars', 'void **'),
        )
    @functools.cached_property
    def parse_alt(self) -> TypedName:
        return self.make_parser_name(
        '_PyPegen_parse_alt', None,
        ('alt', 'const RuleAltDescr *'),
        )
    @functools.cached_property
    def parse_alts(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_PARSE_ALT_ARRAY', None,
            ('alts', 'const RuleAltDescr []'),
        )
    @functools.cached_property
    def parse_NAME(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_name', 'expr_ty',
        )
    @functools.cached_property
    def parse_NUMBER(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_number_token', 'expr_ty',
            func_type=parse_recipe.ParseData,
        )
    @functools.cached_property
    def parse_STRING(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_string', 'Token *',
        )
    @functools.cached_property
    def parse_OP(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_op', 'Token *',
        )
    @functools.cached_property
    def parse_TYPE_COMMENT(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_type_comment_token', 'Token *',
            func_type=parse_recipe.ParseData,
        )
    @functools.cached_property
    def parse_SOFT_KEYWORD(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_any_soft_keyword', 'Token *',
        )
    @functools.cached_property
    def parse_token(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_token', 'Token *',
            ('type', 'int'),
            )
    @functools.cached_property
    def parse_char(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_char', 'Token *',
            ('c', 'char'),
        )
    @functools.cached_property
    def parse_forced(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_forced', None,
            ('item', 'ParseFunc *'),
            ('expected', 'const char *'),
            func_type=parse_recipe.ParseTrue,
        )
    @functools.cached_property
    def parse_soft_keyword(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_soft_keyword', 'Token *',
            ('keyword', 'const char *'),
        )
    @functools.cached_property
    def parse_repeat(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_repeat', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            ('item_size', 'size_t'),
            ('repeat1', 'int'),
        )
    @functools.cached_property
    def parse_gather(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_gather', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            ('item_size', 'size_t'),
            ('sep', 'ParseFunc *'),
            ('repeat1', 'int'),
        )
    @functools.cached_property
    def parse_opt(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_opt', 'asdl_seq *',
            ('item', 'ParseFunc *'),
            ('item_size', 'size_t'),
        )
    @functools.cached_property
    def parse_lookahead(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_lookahead', None,
            ('positive', 'ParseStatus'),
            ('atom', 'ParseTest *'),
            func_type=parse_recipe.ParseTest,
        )
    @functools.cached_property
    def parse_cut(self) -> TypedName:
        return self.make_parser_name(
            '_PyPegen_parse_cut', None,
            func_type=parse_recipe.ParseVoid,
        )
