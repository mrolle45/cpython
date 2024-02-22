NEWTAB = 0x01

import sys, os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.argv += '''
Tools/peg_generator/pegen/test.c
-o Tools/peg_generator/pegen/test.pcpp.i
-IParser
'''.split()
sys.argv.append("--debug")
#sys.argv.append("--help")
sys.argv.append("-v")
#sys.argv.append("-v")
#sys.argv.append("--passthru-comments")
sys.argv.append("--line-directive=#")
import pcmd
pcmd.main()
x
