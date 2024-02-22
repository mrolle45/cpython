""" pegen/test.py
Run as standalone script to exercise various parser generator functionality.

"""
import os, sys
#import inspect
#print(inspect.stack())
print("Current dir =", os.path.abspath('.'))
cur = os.path.abspath('./Tools/peg_generator')
print(cur)
sys.path.insert(0, os.path.abspath('Lib/MyLib'))
sys.modules.pop('token', None)
sys.modules.pop('tokenize', None)
sys.path.append(cur)
sys.path.append(os.path.abspath('Tools/scripts'))
os.chdir(cur)
print("Current dir =", os.path.abspath('.'))

#import pegen.parser_generator

import token, tokenize
#del sys.path[0]


import pegen.build
from pegen.validator import validate_grammar

# GENERATE Grammar/Tokens -> Lib/token.py

#import generate_token
#generate_token.make_py('../../Grammar/Tokens', '../../Lib/MyLib/token.py')

##### GENERATE PYTHON (NO VERIFY) pegen/metagrammar.gram -> pegen/result_meta.py

if 0x0:
    grammar, parser, tokenizer, gen = pegen.build.build_python_parser_and_generator(
        'pegen/metagrammar.gram', 'pegen/result_meta.py', verify=False,
        #verbose_parser=True,
        )
#validate_grammar(grammar)

# GENERATE PYTHON from the result_meta.py parser.  metagrammar.gram -> result_meta_meta.py.
if 0x0:
    import result_meta
    grammar, parser, tokenizer, gen = pegen.build.build_python_parser_and_generator(
        'pegen/metagrammar.gram', 'pegen/result_meta_meta.py', verify=False, parser_class=result_meta.GeneratedParser)

# GENERATE PYTHON (NO VERIFY) pegen/test.gram -> pegen/pegen/result_python.py

#grammar, parser, tokenizer, gen = pegen.build.build_python_parser_and_generator(
#    'pegen/test.gram', 'pegen/result_python.py', verify=False)
#validate_grammar(grammar)
#grammar.dump()

# Test recursive rules, both C and Python.
#grammar, parser, tokenizer, gen = pegen.build.build_python_parser_and_generator('pegen/test_rec.gram', 'pegen/result_rec.py', verify=False)
#import result_rec
#from pegen.parser import simple_parser_main
#sys.argv.append('pegen/rec.txt')
#sys.argv.append('-vv')
#simple_parser_main(result_rec.GeneratedParser)

#grammar, parser, tokenizer, gen = pegen.build.build_c_parser_and_generator('pegen/test_rec.gram', '../../Grammar/Tokens', 'pegen/result_rec.c')

##### GENERATE C pegen/test.gram, Grammar/Tokens -> pegen/result_c.c

if 0x01:
    grammar, parser, tokenizer, gen = pegen.build.build_c_parser_and_generator(
        'pegen/test.gram', '../../Grammar/Tokens', 'pegen/result_c.c',
        #verbose_parser=True,
        )
    #grammar.dump()
    #validate_grammar(grammar)

# GENERATE Grammar/python.gram, Grammar/Tokens -> pegen/parser.c

if 0x0:
    grammar, parser, tokenizer, gen = pegen.build.build_c_parser_and_generator(
        '../../Grammar/python.gram', '../../Grammar/Tokens', 'pegen/parser.c')
    #validate_grammar(grammar)

# GENERATE Grammar/python.gram, Grammar/Tokens -> pegen/parser.c in 3.11 Repo.

if 0x0:
    grammar, parser, tokenizer, gen = pegen.build.build_c_parser_and_generator(
        '../../Grammar/python.311.gram', '../../Grammar/Tokens.311',
        'pegen/parser.311.c')
    #validate_grammar(grammar)

x = 0
