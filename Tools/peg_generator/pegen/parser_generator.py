import ast
import contextlib
import re
from abc import abstractmethod
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
    Gather,
    Grammar,
    GrammarError,
    GrammarVisitor,
    Group,
    Lookahead,
    NamedItem,
    NameLeaf,
    Opt,
    Params,
    Plain,
    Repeat0,
    Repeat1,
    Rhs,
    Rule,
    StringLeaf,
    TypedName,
)

# TODO: replace self.all_rules with self.rules when there are no more artificial rules.

class RuleCollectorVisitor(GrammarVisitor):
    """Visitor that invokes a provided callmaker visitor with just the NamedItem nodes"""

    def __init__(self, gen: "ParserGenerator", callmakervisitor: GrammarVisitor) -> None:
        self.gen = gen
        self.rules: Dict[str, Rule] = gen.rules
        self.callmaker = callmakervisitor

    def visit_Rule(self, rule: Rule) -> None:
        self.visit(rule.flatten())

    def visit_NamedItem(self, item: NamedItem) -> None:
        self.callmaker.visit(item)


class KeywordCollectorVisitor(GrammarVisitor):
    """Visitor that collects all the keywods and soft keywords in the Grammar"""

    def __init__(self, gen: "ParserGenerator", keywords: Dict[str, int], soft_keywords: Set[str]):
        self.generator = gen
        self.keywords = keywords
        self.soft_keywords = soft_keywords

    def visit_StringLeaf(self, node: StringLeaf) -> None:
        val = ast.literal_eval(node.value)
        if re.match(r"[a-zA-Z_]\w*\Z", val):  # This is a keyword
            if node.value.endswith("'") and node.value not in self.keywords:
                self.keywords[val] = self.generator.keyword_type()
            else:
                return self.soft_keywords.add(node.value.replace('"', ""))

class DumpVisitor(GrammarVisitor):
    def __init__(self, all: bool = False):
        self.level = 0
        self.all = all

    def visit(self, node) -> None:
        leader = "    " * self.level
        if isinstance(node, Attr):
            print(f'{leader}{node}')
            return
        if node.showme or self.all:
            print(f'{leader}{type(node).__name__} = {node.show(leader)}')
            self.level += 1
            self.generic_visit(node)
            self.level -= 1
        else:
            self.generic_visit(node)

    def generic_visit(self, node: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        """Called if no explicit visitor function exists for a node."""
        for value in (node if self.all else node.itershow()):
            if type(value) is list:
                for item in value:
                    self.visit(item, *args, **kwargs)
            else:
                self.visit(value, *args, **kwargs)


class ParserGeneratorBase: pass

class RuleCheckingVisitor(GrammarVisitor, ParserGeneratorBase):
    def __init__(self, rules: Dict[str, Rule], tokens: Set[str]):
        self.rules = rules
        self.tokens = tokens
        self.current_item = None

    def visit_Rule(self, node: Rule) -> None:
        self.current_rule = node
        self.current_alt = None
        self.current_item = None
        self.generic_visit(node)
        del self.current_rule

    def visit_Alt(self, node: Alt) -> None:
        save = self.current_alt, self.current_item
        self.current_alt = node
        self.current_item = None
        self.generic_visit(node)
        self.current_alt, self.current_item = save

    def visit_NameLeaf(self, node: NameLeaf) -> None:
        self.validate_rule_args(node)

    def visit_NamedItem(self, node: NamedItem) -> None:
        if node.name and node.name.startswith("_"):
            raise GrammarError(f"Variable names cannot start with underscore: '{node.name}'")
        self.current_item = node
        self.generic_visit(node)

    def validate_rule_args(self, node: NameLeaf) -> None:
        """ Verify that the name corresponds to a known Rule, and the arguments match.
        This is called after the entire grammar is parsed.
        """
        name = node.value
        if name in self.tokens:
            return
        def check(params: Optional[Params], is_rule: bool = False) -> None:
            # If is_rule, then an empty args is same as not empty.

            args = node.args
            if params is None:
                if args.empty: return
                raise GrammarError(f"Calling {node}, no arguments allowed.")
            if args.empty and not is_rule:
                raise GrammarError(f"Calling {node}, arguments required.")
            nparams = len(params)
            if len (args) != nparams:
                raise GrammarError(f"Calling {node}, requires exactly {nparams} arguments.")
            return
        # Try a rule name
        rule = self.rules.get(name)
        if rule:
            return check(rule.params, is_rule=True)
        # Try a parameter name in the current rule
        param = self.current_rule.params.get(name)
        if param:
            return check(param.params)
        # Try an earlier variable name in the current alt
        # Check the named items in the alt until the current item is reached.
        for item in self.current_alt.items:
            if item is self.current_item: break     # Search failed.
            if item.name == name:
                return check(item.params)

        raise GrammarError(f"Dangling reference to name {node.value!r}")


class ParserGenerator(ParserGeneratorBase):

    callmakervisitor: GrammarVisitor

    def __init__(self, grammar: Grammar, tokens: Set[str], file: Optional[IO[Text]], *, verbose: bool = False):
        self.grammar = grammar
        self.tokens = tokens
        self.keywords: Dict[str, int] = {}
        self.soft_keywords: Set[str] = set()
        self.rules = grammar.rules
        self.validate_rule_names()
        if "trailer" not in grammar.metas and "start" not in self.rules:
            raise GrammarError("Grammar without a trailer must have a 'start' rule")
        checker = RuleCheckingVisitor(self.rules, self.tokens)
        for rule in self.rules.values():
            checker.visit(rule)
        self.file = file
        self.level = 0
        self.first_graph, self.first_sccs = compute_left_recursives(self.rules)
        self.counter = 0  # For name_rule()/name_loop()
        self.keyword_counter = 499  # For keyword_type()
        self._local_variable_stack: List[List[str]] = []
        self.verbose = verbose

    def validate_rule_names(self) -> None:
        for rule in self.rules:
            if rule.startswith("_"):
                raise GrammarError(f"Rule names cannot start with underscore: '{rule}'")

    @contextlib.contextmanager
    def local_variable_context(self) -> Iterator[None]:
        self._local_variable_stack.append([])
        yield
        self._local_variable_stack.pop()

    @property
    def local_variable_names(self) -> List[str]:
        return self._local_variable_stack[-1]

    @abstractmethod
    def generate(self, filename: str) -> None:
        raise NotImplementedError

    @contextlib.contextmanager
    def indent(self, levels: int = 1) -> Iterator[None]:
        self.level += levels
        try:
            yield
        finally:
            self.level -= levels

    def print(self, *args: object) -> None:
        if not args:
            print(file=self.file)
        else:
            print("    " * self.level, end="", file=self.file)
            print(*args, file=self.file)

    def printblock(self, lines: str) -> None:
        for line in lines.splitlines():
            self.print(line)

    def lineno(self) -> int:
        return len(self.file.getvalue().splitlines()) + 1

    def collect_rules(self) -> None:
        self.all_rules = dict(self.rules)
        self.collect_keywords(self.all_rules)
        rule_collector = RuleCollectorVisitor(self, self.callmakervisitor)
        done: Set[str] = set()
        while True:
            computed_rules = list(self.all_rules)
            todo = [i for i in computed_rules if i not in done]
            if not todo:
                break
            done = set(self.all_rules)
            for rulename in todo:
                self.current_rule = self.all_rules[rulename]
                rule_collector.visit(self.current_rule)
            self.current_rule = None

    def collect_keywords(self, rules: Dict[str, Rule]) -> None:
        keyword_collector = KeywordCollectorVisitor(self, self.keywords, self.soft_keywords)
        for rule in rules.values():
            keyword_collector.visit(rule)

    def keyword_type(self) -> int:
        self.keyword_counter += 1
        return self.keyword_counter

    def artificial_rule_from_rhs(self, rhs: Rhs) -> str:
        self.counter += 1
        name = f"_group_{self.counter}"
        self.all_rules[name] = Rule.simple(name, self.current_rule.params, rhs)
        return name

    # TODO: Remove these artificial rules when no longer called in c_generator.
    # They are not called in python_generator.

    def artificial_rule_from_repeat(self, node: Plain, is_repeat1: bool) -> str:
        self.counter += 1
        if is_repeat1:
            prefix = "_loop1_"
        else:
            prefix = "_loop0_"
        name = f"{prefix}{self.counter}"
        self.all_rules[name] = Rule.simple(
            name,
            self.current_rule.params,
            Rhs([Alt([NamedItem(None, node)])])
        )
        return name

    def artificial_rule_from_gather(self, node: Gather) -> str:
        self.counter += 1
        name = f"_gather_{self.counter}"

        elem = NamedItem(TypedName("elem"), node.node)
        sep = NamedItem(TypedName("sep"), node.separator)
        group = Group(Rhs([Alt([sep, elem])]))
        rep = Repeat0(group)
        alt = Alt([NamedItem(TypedName("elem"), elem), NamedItem(TypedName("seq"), rep)])
        self.all_rules[name] = Rule.simple(
            name,
            self.current_rule.params,
            Rhs([alt]),
        )

        return name

    def dedupe(self, name: str) -> str:
        """ Add a suffix to given name if it appears earlier.
        Any parameter names are searched before current local variable names.
        """
        origname = name
        counter = 0
        param_names = self.current_rule.param_names
        while name in param_names + self.local_variable_names:
            counter += 1
            name = f"{origname}_{counter}"
        self.local_variable_names.append(name)
        return name


class NullableVisitor(GrammarVisitor):
    def __init__(self, rules: Dict[str, Rule]) -> None:
        self.rules = rules
        self.visited: Set[Any] = set()
        self.nullables: Set[Union[Rule, NamedItem]] = set()

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
        for item in alt.items:
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
        return self.visit(group.rhs)

    def visit_NamedItem(self, item: NamedItem) -> bool:
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
        for item in alt.items:
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
            leader = min(leaders)  # Pick an arbitrary leader from the candidates.
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
