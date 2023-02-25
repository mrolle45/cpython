import os, sys
cur = os.path.abspath('Tools/peg_generator')
print(cur)
sys.path.append(cur)
os.chdir(cur)

import pegen.build
from pegen.validator import validate_grammar

grammar, parser, tokenizer, gen = pegen.build.build_python_parser_and_generator('pegen/test.gram', 'result_python.txt', verify=False)
validate_grammar(grammar)
pegen.build.build_c_parser_and_generator('pegen/test.gram', '../../Grammar/Tokens', 'result_c.txt')
validate_grammar(grammar)

x = 0
