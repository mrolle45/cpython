import os.path
import token
from typing import IO, Any, Dict, Optional, Sequence, Set, Text, Tuple, Iterable
from dataclasses import dataclass, field, replace

from pegen import grammar
from pegen.grammar import (
    Alt,
    Cut,
    Forced,
    Gather,
    GrammarVisitor,
    Group,
    Lookahead,
    NamedItem,
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
)
from pegen.parser_generator import ParserGenerator

MODULE_PREFIX = """\
#!/usr/bin/env python3.8
# @generated by pegen from {filename}

import ast
import sys
import tokenize

from typing import Any, Optional

from pegen.parser import memoize, memoize_left_rec, logger, Parser, cut_sentinel

"""
MODULE_SUFFIX = """

if __name__ == '__main__':
    from pegen.parser import simple_parser_main
    simple_parser_main({class_name})
"""


class InvalidNodeVisitor(GrammarVisitor):
    def visit_NameLeaf(self, node: NameLeaf) -> bool:
        name = node.value
        return name.startswith("invalid")

    def visit_StringLeaf(self, node: StringLeaf) -> bool:
        return False

    def visit_NamedItem(self, node: NamedItem) -> bool:
        return self.visit(node.item)

    def visit_Rhs(self, node: Rhs) -> bool:
        return any(self.visit(alt) for alt in node)

    def visit_Alt(self, node: Alt) -> bool:
        return any(self.visit(item) for item in node.items)

    def lookahead_call_helper(self, node: Lookahead) -> bool:
        return self.visit(node.node)

    def visit_PositiveLookahead(self, node: PositiveLookahead) -> bool:
        return self.lookahead_call_helper(node)

    def visit_NegativeLookahead(self, node: NegativeLookahead) -> bool:
        return self.lookahead_call_helper(node)

    def visit_Opt(self, node: Opt) -> bool:
        return self.visit(node.node)

    def visit_Repeat(self, node: Repeat0) -> Tuple[str, str]:
        return self.visit(node.node)

    def visit_Gather(self, node: Gather) -> Tuple[str, str]:
        return self.visit(node.node)

    def visit_Group(self, node: Group) -> bool:
        return self.visit(node.rhs)

    def visit_Cut(self, node: Cut) -> bool:
        return False

    def visit_Forced(self, node: Forced) -> bool:
        return self.visit(node.node)


@dataclass
class FunctionCall:
    var: Optional[str]
    call: str
    var_type: Optional[str] = None      # The type of a designated variable for a NamedItem.
    return_type: Optional[str] = None

    def __iter__(self) -> Iterable:
        yield self.var
        yield self.call
        yield self.return_type
        #yield self.var_type


class PythonCallMakerVisitor(GrammarVisitor):
    def __init__(self, parser_generator: ParserGenerator):
        self.gen = parser_generator
        self.cache: Dict[Any, Any] = {}

    def visit_NameLeaf(self, node: NameLeaf) -> Tuple[Optional[str], str]:
        name = node.value
        if name == "SOFT_KEYWORD":
            return FunctionCall("_soft_keyword", "self._soft_keyword()")
        if name in ("NAME", "NUMBER", "STRING", "OP", "TYPE_COMMENT"):
            name = name.lower()
            return FunctionCall(f"_{name}", f"self._{name}()")
        if name in ("NEWLINE", "DEDENT", "INDENT", "ENDMARKER", "ASYNC", "AWAIT"):
            # Avoid using names that can be Python keywords
            return FunctionCall("_" + name.lower(), f"self._expect({name!r})")
        return FunctionCall(name, f"self.{name}{node.args.show()}")

    def visit_StringLeaf(self, node: StringLeaf) -> Tuple[str, str]:
        return FunctionCall("_literal", f"self._expect({node.value})")

    def visit_Rhs(self, node: Rhs) -> Tuple[Optional[str], str]:
        if node in self.cache:
            return self.cache[node]
        if len(node) == 1 and len(node[0].items) == 1:
            self.cache[node] = self.visit(node[0].items[0])
        else:
            name = self.gen.artificial_rule_from_rhs((node))
            self.cache[node] = FunctionCall(name, f"self.{name}{self.args_from_params()}")
        return self.cache[node]

    def visit_NamedItem(self, node: NamedItem) -> Tuple[Optional[str], str]:
        func = self.visit(node.item)
        if node.name:
            func = replace(func, var=node.name)
        return func

    def lookahead_call_helper(self, node: Lookahead, method: str) -> Tuple[str, str]:
        func = self.visit(node.node)
        call = func.call
        head, tail = call.split("(", 1)
        assert tail[-1] == ")"
        tail = tail[:-1]
        return replace(func, var="_lookahead", call=f"self.{method}({head}, {tail})")

    def visit_PositiveLookahead(self, node: PositiveLookahead) -> Tuple[None, str]:
        return self.lookahead_call_helper(node, '_positive_lookahead')

    def visit_NegativeLookahead(self, node: NegativeLookahead) -> Tuple[None, str]:
        return self.lookahead_call_helper(node, '_negative_lookahead')

    def visit_Opt(self, node: Opt) -> Tuple[str, str]:
        func = self.visit(node.node)
        return replace(func, var="_opt")

    def visit_Repeat0(self, node: Repeat0) -> Tuple[str, str]:
        if node in self.cache:
            return self.cache[node]
        self.cache[node] = FunctionCall("_loop", "")
        self.visit(node.node)
        return self.cache[node]

    def visit_Repeat1(self, node: Repeat1) -> Tuple[str, str]:
        if node in self.cache:
            return self.cache[node]
        self.cache[node] = FunctionCall("_loop", "")
        self.visit(node.node)
        return self.cache[node]

    def visit_Gather(self, node: Gather) -> Tuple[str, str]:
        if node in self.cache:
            return self.cache[node]
        name = self.gen.artificial_rule_from_gather(node)
        self.cache[node] = FunctionCall(name, f"self.{name}{self.args_from_params()}")  # No trailing comma here either!
        return self.cache[node]

    def visit_Group(self, node: Group) -> Tuple[Optional[str], str]:
        return self.visit(node.rhs)

    def visit_Cut(self, node: Cut) -> Tuple[str, str]:
        return FunctionCall("cut", "cut_sentinel,")

    def visit_Forced(self, node: Forced) -> Tuple[str, str]:
        if isinstance(node.node, Group):
            _, val = self.visit(node.node.rhs)
            return FunctionCall("forced", f"self._expect_forced({val}, '''({node.node.rhs!s})''')")
        else:
            return FunctionCall(
                "forced",
                f"self._expect_forced(self._expect({node.node.value}), {node.node.value!r})",
            )

    def return_type(self, node: Any) -> str:
        """ The type resulting from calling the node. """
        return self.visit(node).type

    def args_from_params(self) -> str:
        """ Argument list to call an artificial rule using names of main Rule parameters. """
        params = self.gen.current_rule.params
        if not params: return "()"
        return f'({", ".join([param.name for param in params])})'

class PythonParserGenerator(ParserGenerator, GrammarVisitor):
    def __init__(
        self,
        grammar: grammar.Grammar,
        file: Optional[IO[Text]],
        tokens: Set[str] = set(token.tok_name.values()),
        location_formatting: Optional[str] = None,
        unreachable_formatting: Optional[str] = None,
        verbose: bool = False,
    ):
        tokens.add("SOFT_KEYWORD")
        super().__init__(grammar, tokens, file, verbose=verbose)
        self.callmakervisitor: PythonCallMakerVisitor = PythonCallMakerVisitor(self)
        self.invalidvisitor: InvalidNodeVisitor = InvalidNodeVisitor()
        self.unreachable_formatting = unreachable_formatting or "None  # pragma: no cover"
        self.location_formatting = (
            location_formatting
            or "lineno=start_lineno, col_offset=start_col_offset, "
            "end_lineno=end_lineno, end_col_offset=end_col_offset"
        )

    def generate(self, filename: str) -> None:
        self.collect_rules()
        header = self.grammar.metas.get("header", MODULE_PREFIX)
        if header is not None:
            basename = os.path.basename(filename)
            self.print(header.rstrip("\n").format(filename=basename))
        subheader = self.grammar.metas.get("subheader", "")
        if subheader:
            self.print(subheader)
        cls_name = self.grammar.metas.get("class", "GeneratedParser")
        self.print("# Keywords and soft keywords are listed at the end of the parser definition.")
        self.print(f"class {cls_name}(Parser):")
        for rule in dict(self.all_rules).values():
            self.print()
            with self.indent():
                self.visit(rule)

        self.print()
        with self.indent():
            self.print(f"KEYWORDS = {tuple(self.keywords)}")
            self.print(f"SOFT_KEYWORDS = {tuple(self.soft_keywords)}")

        trailer = self.grammar.metas.get("trailer", MODULE_SUFFIX.format(class_name=cls_name))
        if trailer is not None:
            self.print(trailer.rstrip("\n"))

    def alts_uses_locations(self, alts: Sequence[Alt]) -> bool:
        for alt in alts:
            if alt.action and "LOCATIONS" in alt.action:
                return True
            for n in alt.items:
                if isinstance(n.item, Group):
                    if self.alts_uses_locations(n.item.rhs): return True
        return False

    def rule_params(self, rule: Rule) -> str:
        """ The text for parameters to declare a rule. """
        params = ''.join([f', {param.name}: {self.param_type(param)}' for param in (rule.params)])
        return f"(self{params})"

    def param_type(self, param: TypedName) -> str:
        """ What is generated for the type of a parameter, following '{param.name}:'
        The name may have its own parameters, which are generated recursively.
        """
        base_type = param.type or "Any"
        if param.params and len(param.params):
            # This node is a callable type.
            subtypes = [self.param_type(subparam) for subparam in param.params]
            return f'Callable[[{", ".join(subtypes)}], {base_type}]'
        else:
            return base_type

    def visit_Rule(self, node: Rule) -> None:
        self.current_rule = node
        is_loop = node.is_loop()
        is_loop1 = node.is_loop1()
        is_gather = node.is_gather()
        rhs = node.flatten()
        if node.left_recursive:
            if node.leader:
                self.print("@memoize_left_rec")
            else:
                # Non-leader rules in a cycle are not memoized,
                # but they must still be logged.
                self.print("@logger")
        elif node.memo or self.verbose:
            self.print("@memoize")
        node_type = node.type or "Any"
        self.print(f"def {node.name}{self.rule_params(node)} -> Optional[{node_type}]:")
        alt_names = [f'_alt{i}_{node.name}' for i, alt in enumerate(node.rhs, 1)]

        def gen_alt(alt: Alt) -> None:
            self.print("def _alt():")
            with self.indent():
                # The body of the alternative, returns (result,) or None.
                self.visit(alt, is_loop=is_loop, is_gather=is_gather)

        with self.indent():
            alts = node.rhs
            self.print(f"# {node.name}: {rhs}")
            if len(alts) == 1:
                gen_alt(alts[0])
                self.print("return self._alt(_alt)")
            else:
                self.print("def _alts():")
                with self.indent():
                    for alt in node.rhs:
                        gen_alt(alt)
                        self.print("yield _alt()")
                self.print("return self._alts(_alts())")

        self.current_rule = None

    def visit_NamedItem(self, node: NamedItem) -> None:
        name, call, type = self.callmakervisitor.visit(node.item)
        if node.name:
            name = node.name
        if not name:
            self.print(call)
        else:
            if name != "cut":
                name = self.dedupe(name)
            if call.endswith(','):
                call = f"({call})"
            expr = f"({name} := self._get_val({call}))"
            if isinstance(node.item, (Opt, Cut)) or call.startswith('self._loop0'):
                expr = f"({expr} or True)"
            self.print(expr)


    def visit_Alt(self, node: Alt, is_loop: bool = False, is_loop1: bool = False, is_gather: bool = False) -> None:
        self.failed_value = "None"
        with self.local_variable_context():
            if is_loop:
                self.print("_children = []")
                self.print("while True:")
            with self.indent(is_loop):
                for item in node.items:
                    # Test each item, and break from the while loop if failed
                    self.gen_item(item, is_loop=is_loop)

                # If the function has reached this point, then the child is successful
                self.print ("# successful parse")
                action = self.action(node, is_gather=is_gather)
                    
                if is_loop:
                    self.print(f"_children.append({action})")
            if is_loop:
                if is_loop1:
                    self.print("if not _children: return None")
                self.print("return _children,")
            else:
                self.print(f"return ({action}),")

    def gen_item(self, item: NamedItem, is_loop: bool = False) -> None:
        """ Code for a single item in an alt or loop function. """

        assert isinstance (item, NamedItem)
        self.print(f"# {item}")
        if isinstance(item.item, Cut):
            # Return cut indicator if anything fails from here forward.
            self.failed_value = "cut_sentinel"
            return

        name, call, type = self.callmakervisitor.visit(item)
        if item.name:
            name = item.name

        name = self.dedupe(name)
        if type is None: type = 'Any'
        rawname = f"_item_{name}"
        rawtype = f"Optional[Tuple[{type}]]"
        if isinstance(item.item, Opt):
            type = f"Optional[{type}]"
        self.print(f"{rawname}: {rawtype}; {name}: {type}")
        if call:
            self.print(f"{rawname} = {call}")
        else:
            # Expand the item inline
            if isinstance(item.item, Repeat0):
                is_loop1 = False
            if isinstance(item.item, Repeat1):
                is_loop1 = True
            self.print("def _loop():")
            with self.indent():
                self.visit(Alt([NamedItem(None, item.item.node)]), is_loop=True, is_loop1=is_loop1)
            self.print(f"{rawname} = _loop()")
        if isinstance(item.item, Opt):
            # The item may be failed, but we keep going anyway.
            self.print(f"if {rawname} is None: {name} = None")
            self.print(f"else: {name}, = {rawname}")
        else:
            # Test for failure.  A Repeat0 never fails.
            if not isinstance(item.item, Repeat0):
                if is_loop:
                    self.print(f"if {rawname} is None: break")
                else:
                    self.print(f"if {rawname} is None: return {self.failed_value}")
            self.print(f"{name}, = {rawname}")

    def action(self, node: Alt, is_gather: bool = False) -> str:
        """ The action for the alt, if it succeeds. """
        action = node.action
        if not action:
            if is_gather:
                assert len(self.local_variable_names) == 2
                action = (
                    f"[{self.local_variable_names[0]}] + {self.local_variable_names[1]}"
                )
            else:
                if not node.items:
                    action = "True"
                elif self.invalidvisitor.visit(node):
                    action = "UNREACHABLE"
                elif len(self.local_variable_names) == 1:
                    action = f"{self.local_variable_names[0]}"
                else:
                    action = f"[{', '.join(self.local_variable_names)}]"
        elif "LOCATIONS" in action:
            self.print("tok = self._tokenizer.get_last_non_whitespace_token()")
            self.print("end_lineno, end_col_offset = tok.end")
            action = action.replace("LOCATIONS", self.location_formatting)

        if "UNREACHABLE" in action:
            action = action.replace("UNREACHABLE", self.unreachable_formatting)

        return action
