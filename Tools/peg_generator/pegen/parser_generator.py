from __future__ import annotations

import ast
import contextlib
import functools
import itertools
import os
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass, field, replace
from enum import Enum, auto

from typing import (
    IO,
    AbstractSet,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Text,
    Tuple,
    Union,
)

from pegen import sccutils
from pegen.grammar import (
    Alt,
    Args,
    Attr,
    Cut,
    Forced,
    Gather0,
    Gather1,
    Grammar,
    GrammarTree,
    GrammarError,
    GrammarVisitor,
    Group,
    Lookahead,
    VarItem,
    NameLeaf,
    NoArgs,
    Opt,
    ObjName,
    Param,
    Params,
    ParseExpr,
    Repeat0,
    Repeat1,
    Rhs,
    Rule,
    StringLeaf,
    TypedName,
)

from pegen.target_code import Code
from pegen.parse_recipe import ParseSource
from pegen.tokenizer import TokenMatch, TokenLocations

class KeywordCollectorVisitor(GrammarVisitor):
    """Visitor that collects all the keywords and soft keywords in the Grammar.
    Constructor provides collection objects for the results.
    """

    def __init__(self, keywords: Dict[str, int], soft_keywords: Set[str]):
        self.keywords = keywords
        self.soft_keywords = soft_keywords

    def visit_StringLeaf(self, node: StringLeaf) -> None:
        val = ast.literal_eval(node.value)
        if re.match(r"[a-zA-Z_]\w*\Z", val):  # This is a keyword
            if node.value.endswith("'"):
                if val not in self.keywords:
                    self.keywords[val] = node.gen.keyword_type()
            else:
                return self.soft_keywords.add(node.value.replace('"', ""))


class DumpVisitor(GrammarVisitor):
    def __init__(self, all: bool = False, env: bool = False):
        self.level = 0
        self.all = all
        self.env = env

    def visit(self, node) -> None:
        if node.showme or self.all:
            node.message(f'{type(node).__name__} = {node.show()}')
            #if self.env:
            #    if hasattr(node, 'local_env'):
            #        node.local_env.dump(node)
            self.level += 1
            self.generic_visit(node)
            self.level -= 1
        else:
            self.generic_visit(node)

    def generic_visit(self, node: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        """Called if no explicit visitor function exists for a node."""
        for attr in node.attrs():
            if attr:
                node.message(str(attr))
        for value in (node if self.all else node.itershow()):
            if type(value) is list:
                for item in value:
                    self.visit(item, *args, **kwargs)
            else:
                self.visit(value, *args, **kwargs)


class RuleCheckingVisitor(GrammarVisitor):
    def __init__(self, rules: Dict[str, Rule], tokens: Set[str], gen: ParserGenerator):
        self.rules = rules
        self.tokens = tokens
        self.gen = gen

    def visit_Rule(self, node: Rule) -> None:
        self.generic_visit(node)

    def visit_Alt(self, node: Alt) -> None:
        self.generic_visit(node)

    def visit_NameLeaf(self, node: NameLeaf) -> None:
        self.validate_rule_args(node)

    def visit_VarItem(self, node: VarItem) -> None:
        if node.name and node.name.string.startswith("_"):
            raise GrammarError(f"Variable names cannot start with underscore: '{node.name}'")
        self.generic_visit(node)

    def validate_rule_args(self, node: NameLeaf) -> None:
        """ Verify that the name corresponds to a known Rule, and the arguments match.
        This is called after the entire grammar is parsed.
        """
        if node.value.string in self.tokens:
            return
        name = node.resolve()
        params = name.params
        # If a rule name, then an empty args is same as not empty.

        args = node.args
        if params is None:
            if args.empty: return
            if name.name in self.rules and not len(args): return
            raise GrammarError(f"Calling {node}, no arguments allowed.")
        if args.empty and not isinstance(name, Rule):
            raise GrammarError(f"Calling {node}, arguments required.")
        nparams = len(params)
        if len (args) != nparams:
            raise GrammarError(f"Calling {node}, requires exactly {nparams} arguments.")


# TODO: Remove these classes when no longer used by Py Generator.
class FuncCtx:
    """ Specifies how the generation of the parse function was requested.
    Determines some properties of the generated function.
    """
    @abstractmethod
    def name(self, dflt_name: str) -> str: ...

    @abstractmethod
    def varname(self, dflt_name: str) -> str: ...

    @abstractmethod
    def type(self) -> str: ...

    @abstractmethod
    def vartype(self) -> str: ...


class AltFuncCtx(FuncCtx):
    """ A parse function for a VarItem in an Alt. """
    def __init__(self, item: VarItem, gen: ParserGenerator):
        self.item = item
        self.gen = gen
        self._varname = None

    def varname(self, dflt_name: str) -> str:
        if not self._varname:
            self._varname = self.item.dedupe(self.item.name or (dflt_name + '_var'))
        return self._varname

    def name(self, dflt_name: str) -> str:
        return f"_item_{self.varname(dflt_name)}"

    def type(self) -> str:
        return ""

    def vartype(self) -> str:
        return self.item.type


class InlFuncCtx(FuncCtx):
    """ A parse function for an inline node in another parse function. """
    def __init__(self, name: str):
        self._name = name

    def varname(self, dflt_name: str) -> str:
        return self._name or dflt_name

    def name(self, dflt_name: str) -> str:
        return self._name or dflt_name

    def type(self) -> str:
        return ""

    def vartype(self) -> str:
        return ""


class TargetLanguageTraits(ABC):
    """ Methods of a ParserGenerator which vary with the target language. """
    language: ClassVar[str]

    @abstractmethod
    def default_header(self) -> str: ...

    @abstractmethod
    def default_trailer(self) -> str: ...

    @abstractmethod
    def comment_leader(self) -> str: ...

    @abstractmethod
    def default_type(self, type: str = None) -> str: ...

    @abstractmethod
    def default_value(self, value: str = None) -> str: ...

    @abstractmethod
    def cut_value(self, value: str = None) -> str: ...

    @abstractmethod
    def bool_value(self, value: bool) -> str: ...

    @abstractmethod
    def bool_type(self) -> str: ...

    @abstractmethod
    def str_value(self, value: bool) -> str: ...

    @abstractmethod
    def default_params(self) -> str: ...

    @abstractmethod
    def parser_param(self) -> Param: ...

    @abstractmethod
    def parse_result_ptr_param(self, assigned_name: str = None) -> Param: ...

    @abstractmethod
    def return_param(self) -> Param: ...

    @abstractmethod
    def parse_func_params(self, name: TypedName = None) -> Params: ...

    @abstractmethod
    def rule_func_name(self, rule: Rule) -> ObjName: ...

    @abstractmethod
    def cast(self, value: str, to_type: TypedName) -> str:
        """ String for given value coerced to given type. """
        ...

    @abstractmethod
    def forward_declare_inlines(self, recipe: ParseRecipe) -> None:
        """ When the current function has inlines which are generated at file scope,
        this declares those inline function so that the current function can call them.
        """

    @abstractmethod
    def gen_copy_local_vars(self, node: GrammarTree) -> None:
        """ Code to define inherited names and set their values stored in the parser. """
        ...

    @abstractmethod
    def enter_function(
        self, name: str, type: str, params: Params, comment: str = ''
        ) -> Iterator: ...

    @abstractmethod
    def gen_start(self, alt: Alt) -> None:
        """ Generate code to set starting line and col variables if required by action """

    @dataclass
    class Action:
        expr: str           # Expression for the value of the action
        type: str           # Expression for the type of the value.

    @abstractmethod
    def gen_action(self, node: Alt) -> Action:
        """ Generate code to return the action in the Alt. """

    @abstractmethod
    def fix_parse_recipe(self, recipe: ParseRecipe) -> None:
        """ Alter a given recipe appropriately for the target language. """

    @abstractmethod
    def parse_recipe(self, recipe: ParseRecipe, **kwds) -> None:
        """ Generate code to parse the recipe, either inline now, or save for later. """
        ...

    @contextlib.contextmanager
    def enter_scope(self, stmt: str, brackets: str = '', comment: str = '', enable: bool = True) -> Iterator:
        if not enable:
            yield
            return
        if brackets:
            stmt = (stmt and f"{stmt} " or "") + brackets[0]
        self.print(stmt, comment=comment)
        with self.indent():
            yield
        if brackets:
            self.print(brackets[1:])


class ParserGenerator(ABC):

    # Macro definitions for target language Code objects.
    # Any occurrence of a macro name is replaced with the macro value.
    macros: Mapping[str, str] = {}

    def __init__(
        self,
        grammar: Grammar,
        tokens: Dict[int, str],
        exact_tokens: Dict[str, int],
        non_exact_tokens: Set[str],
        file: Optional[IO[Text]],
        *,
        verbose: bool = False,
        ):
        self.grammar = grammar
        grammar._thread_vars.gen = self
        self.tokens = set(tokens.values())
        self.keywords: Dict[str, int] = {}
        self.soft_keywords: Set[str] = set()
        self.token_types = {name: type for type, name in tokens.items()}
        self.rules = grammar.rules
        self.collect_keywords(self.rules)
        self.exact_tokens = exact_tokens
        self.non_exact_tokens = non_exact_tokens

        self.file = file
        self.first_graph, self.first_sccs = compute_left_recursives(self.rules)
        defs = self.grammar.metas.get("defs")
        if defs:
            self.macros = {}
            active: bool = False
            for line in defs.splitlines():
                line = line.strip()
                if line.startswith('#'):
                    continue
                if line == f"[{self.language}]":
                    active = True
                    continue
                if not active:
                    continue
                if line.startswith('['):
                    active = False
                    continue
                # Here's a definition.
                if '=' not in line:
                    raise GrammarError(f"Invalid definition {line!r} in @defs meta.")
                name, value = line.split('=', 1)
                self.macros[name.strip()] = Code(value.strip())
        grammar.initialize()
        self.validate_rule_names()
        checker = RuleCheckingVisitor(self.rules, self.tokens, self)
        for rule in self.rules.values():
            try: checker.visit(rule)
            except GrammarError: pass

        self.level = 0
        self.counter = 0  # For name_rule()/name_loop()
        self.verbose = verbose

    def validate_rule_names(self) -> None:
        for rule in self.rules:
            if rule.string.startswith("_"):
                raise GrammarError(f"Rule names cannot start with underscore: '{rule}'")

    @abstractmethod
    def generate(self, filename: str) -> None:
        """ Generate the result parser file.
        This is what is common to all target languages.
        Subclass will add more text.
        """
        pass

    @contextlib.contextmanager
    def gen_header_and_trailer(self, filename: str) -> Iterator[None]:
        header = self.grammar.metas.get("header", self.default_header())
        if header is not None:
            basename = os.path.basename(filename)
            self.print(header.rstrip("\n").format(filename=basename))
        subheader = self.grammar.metas.get("subheader", "")
        if subheader:
            self.print(subheader)
        subheader = self.grammar.metas.get(f"subheader_{self.language}", "")
        if subheader:
            self.print(subheader)

        yield

        trailer = self.grammar.metas.get("trailer", self.default_trailer())
        if trailer is not None:
            self.print(trailer.rstrip("\n"))

    @contextlib.contextmanager
    def indent(self, levels: int = 1) -> Iterator[None]:
        self.level += levels
        try:
            yield
        finally:
            self.level -= levels

    break_lines: List[int] = [224]
    break_tokens: TokenMatch = TokenMatch('''
        #52, 25 - 37
        ''')

    def print(self, *args: object, comment: str = None) -> None:
        if not (args or comment):
            print(file=self.file)
        else:
            if args == (None,): return
            leader = "    " * self.level
            lines = [f"{leader}{' '.join(args)}"]
            if comment:
                comment = f"{self.comment_leader()}{comment}"
                if args:
                    lines[0] += "  "
                # wrap the comment over multiple lines.
                import textwrap
                lines = textwrap.wrap(
                    comment, width=80,
                    initial_indent=lines[0],
                    subsequent_indent=leader + self.comment_leader() + '    '
                    )

            for line in lines:
                print(line, file=self.file)

    def printblock(self, lines: str) -> None:
        for line in lines.splitlines():
            self.print(line)

    def showout(self, last: int = 0) -> None:
        lines = self.file.getvalue().splitlines()
        for i, line in enumerate(lines, 1):
            if last and i <= len(lines) - last: continue
            print(f"{i}\t{line}")

    def comment(self, text: str) -> str:
        return self.comment_leader() + text

    def typed_param(self, param: Param, name: str = None) -> str:
        """ What is generated for the type of a parameter, including the name.
        The param may have its own parameters, which are generated recursively.
        """
        base_type = param.type
        p = f"{base_type} {name}" if name else base_type
        if param.params:
            # This node is a callable type.
            subtypes = [self.typed_param(subparam) for subparam in param.params]
            return f'{p}({", ".join(subtypes)})'
        else:
            return p

    def gen_node(self, node: GrammarTree, **kwds) -> None:
        """ Generate the code (possibly at a later time) to parse given node. """
        if hasattr(node, 'parse_recipe'):
            self.parse_recipe(node.parse_recipe, **kwds)

    def lineno(self) -> int:
        try: return len(self.file.getvalue().splitlines()) + 1
        except: pass

    def collect_keywords(self, rules: Dict[str, Rule]) -> None:
        self.keyword_counter = 499          # For keyword_type()
        keyword_collector = KeywordCollectorVisitor(self.keywords, self.soft_keywords)
        for rule in rules.values():
            keyword_collector.visit(rule)

    def keyword_type(self) -> int:
        self.keyword_counter += 1
        return self.keyword_counter

    def make_parser_name(
        self,
        name: str, typ: str, *params: Tuple[str, str],
        **kwds,
        ) -> ParseSource:
        return ParseSource(
            name, Code(typ), Params(tuple(Param(TypedName(name, Code(type)))
                                    for name, type in params)),
            **kwds,
            )

    # Older version, still used by Py generator
    @staticmethod
    def _make_parser_name(
        name: str, typ: str, *params: Tuple[str, str], use_parser: bool = True,
        ) -> TypedName:
        return TypedName(
            name, typ, Params(tuple(TypedName(name, type) for name, type in params)),
            use_parser=use_parser
            )

    def brk(self, delta: int = 0) -> bool:
        """ True if current output line + delta is in break_lines container.
        Used as a breakpoint condition with a debugger.
        """
        return self.lineno() + delta in self.break_lines

    def brk_token(self, node: GrammarTree) -> bool:
        """ True if input range of given node matches with break_tokens matcher.
        Used as a breakpoint condition with a debugger.
        """
        return self.break_tokens.match(node.locations)


class NullableVisitor(GrammarVisitor):
    def __init__(self, rules: Dict[str, Rule]) -> None:
        self.rules = rules
        self.visited: Set[Any] = set()
        self.nullables: Set[Union[Rule, VarItem]] = set()

    def visit_Rule(self, rule: Rule) -> bool:
        if rule in self.visited:
            return False
        self.visited.add(rule)
        if self.visit(rule.rhs):
            self.nullables.add(rule)
        return rule in self.nullables

    def visit_Rhs(self, rhs: Rhs) -> bool:
        for alt in rhs:
            if self.visit(alt):
                return True
        return False

    def visit_Alt(self, alt: Alt) -> bool:
        for item in alt:
            if not self.visit(item):
                return False
        return True

    def visit_Forced(self, force: Forced) -> bool:
        return True

    def visit_LookAhead(self, lookahead: Lookahead) -> bool:
        return True

    def visit_Opt(self, opt: Opt) -> bool:
        return True

    def visit_Repeat0(self, repeat: Repeat0) -> bool:
        return True

    def visit_Repeat1(self, repeat: Repeat1) -> bool:
        return False

    def visit_Gather(self, gather: Gather) -> bool:
        return False

    def visit_Cut(self, cut: Cut) -> bool:
        return False

    def visit_Group(self, group: Group) -> bool:
        return self.visit_Rhs(group)

    def visit_VarItem(self, item: VarItem) -> bool:
        if self.visit(item.item):
            self.nullables.add(item)
        return item in self.nullables

    def visit_NameLeaf(self, node: NameLeaf) -> bool:
        if node.value in self.rules:
            return self.visit(self.rules[node.value])
        # Token or unknown; never empty.
        return False

    def visit_StringLeaf(self, node: StringLeaf) -> bool:
        # The string token '' is considered empty.
        return not node.value


def compute_nullables(rules: Dict[str, Rule]) -> Set[Any]:
    """Compute which rules in a grammar are nullable.

    Thanks to TatSu (tatsu/leftrec.py) for inspiration.
    """
    nullable_visitor = NullableVisitor(rules)
    for rule in rules.values():
        nullable_visitor.visit(rule)
    return nullable_visitor.nullables


class InitialNamesVisitor(GrammarVisitor):
    def __init__(self, rules: Dict[str, Rule]) -> None:
        self.rules = rules
        self.nullables = compute_nullables(rules)

    def generic_visit(self, node: Iterable[Any], *args: Any, **kwargs: Any) -> Set[Any]:
        names: Set[str] = set()
        for value in node:
            if isinstance(value, list):
                for item in value:
                    names |= self.visit(item, *args, **kwargs)
            else:
                names |= self.visit(value, *args, **kwargs)
        return names

    def visit_Alt(self, alt: Alt) -> Set[Any]:
        names: Set[str] = set()
        for item in alt:
            names |= self.visit(item)
            if item not in self.nullables:
                break
        return names

    def visit_Forced(self, force: Forced) -> Set[Any]:
        return set()

    def visit_LookAhead(self, lookahead: Lookahead) -> Set[Any]:
        return set()

    def visit_Cut(self, cut: Cut) -> Set[Any]:
        return set()

    def visit_NameLeaf(self, node: NameLeaf) -> Set[Any]:
        return {node.value}

    def visit_StringLeaf(self, node: StringLeaf) -> Set[Any]:
        return set()

    def visit_Attr(self, node: Attr) -> Set[Any]:
        return set()

    def visit_ObjName(self, node: ObjName) -> Set[Any]:
        return set()

def compute_left_recursives(
    rules: Dict[str, Rule]
) -> Tuple[Dict[str, AbstractSet[str]], List[AbstractSet[str]]]:
    graph = make_first_graph(rules)
    sccs = list(sccutils.strongly_connected_components(graph.keys(), graph))
    for scc in sccs:
        if len(scc) > 1:
            for name in scc:
                rules[name].left_recursive = True
            # Try to find a leader such that all cycles go through it.
            leaders = set(scc)
            for start in scc:
                for cycle in sccutils.find_cycles_in_scc(graph, scc, start):
                    # print("Cycle:", " -> ".join(cycle))
                    leaders -= scc - set(cycle)
                    if not leaders:
                        raise ValueError(
                            f"SCC {scc} has no leadership candidate (no element is included in all cycles)"
                        )
            # print("Leaders:", leaders)
            leader = min(leaders, key=str)  # Pick an arbitrary leader from the candidates.
            rules[leader].leader = True
        else:
            name = min(scc)  # The only element.
            if name in graph[name]:
                rules[name].left_recursive = True
                rules[name].leader = True
    return graph, sccs


def make_first_graph(rules: Dict[str, Rule]) -> Dict[str, AbstractSet[str]]:
    """Compute the graph of left-invocations.

    There's an edge from A to B if A may invoke B at its initial
    position.

    Note that this requires the nullable flags to have been computed.
    """
    initial_name_visitor = InitialNamesVisitor(rules)
    graph = {}
    vertices: Set[str] = set()
    for rulename, rhs in rules.items():
        graph[rulename] = names = initial_name_visitor.visit(rhs)
        vertices |= names
    for vertex in vertices:
        graph.setdefault(vertex, set())
    return graph
