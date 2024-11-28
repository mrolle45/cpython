# This constants configures the command line options...
# The syntax is: <input file basename (.c extension by default)>
#                <language = C | C++>
#                <languge standard version in (99, 11, 17, 23) for C,
#                                          or (11, 14, 17, 20, 23) for C++.
#                optional <emulation type = gcc | gnu>
#                       gnu means gcc with GNU extensions.

CONFIG = "test C 11 gnu "
CONFIG = "test C 11 gcc "

INPUT, LANG, VER = CONFIG.split()[:3]
GNU = CONFIG.split()[3:]
GNU = GNU and GNU[0] or ""

# Set this to run all combinations of LANG, VER, and GNU ...
RUN_ALL = 0x01

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir('Tools/peg_generator/pegen')
argv0 = sys.argv[0]

def run() -> None:
    """ Run with current options. """
    gccsfx = GNU and f".{GNU}" or ""
    EXT = (LANG == 'C++' and "cpp" or LANG).lower()
    OUT = f'out/pcpp/{INPUT}.i{gccsfx}.{VER}.{EXT}'
    DIAG = f'out/pcpp/{INPUT}.diag{gccsfx}.{VER}.{LANG}.txt' * bool(RUN_ALL)
    LOG = f'out/pcpp/{INPUT}.log{gccsfx}.{VER}.{LANG}.txt' * bool(RUN_ALL)
    sys.argv = [argv0]
    sys.argv += f'''
    {INPUT}.c
    --{LANG} {VER}
    {(f'--{GNU}' * bool(GNU))}
    -o {OUT}
    -I../../../Parser
    --tabstop 4
    '''.split()
    if DIAG:
        sys.argv.extend(f"--diag {DIAG}".split())

    # Additional command line args which can be commented out...
    #sys.argv.append(
    #   '-ID:/Program Files (x86)/Microsoft Visual Studio 12.0/VC/include')
    sys.argv.extend(f"--debug {LOG}".split())
    if not DIAG: sys.argv.append("--diag")
    #sys.argv.append("--help")
    sys.argv.append("-v")
    #sys.argv.append("--trigraphs")
    #sys.argv.append("-v")
    #sys.argv.append("--passthru-comments")
    #sys.argv.append("--passthru-unknown-exprs")
    sys.argv.extend("--line-directive #".split())
    sys.argv.append("--tabstop")


    import pcmd
    pcmd.main(not RUN_ALL)

if RUN_ALL:


    def runlang():
        global GNU
        #for GNU in ("gcc", "gnu", ):
        for GNU in ("", "gcc", "gnu", ):
            run()

    for VER in (99, 11, 17, 23):
        LANG = "c"
        runlang()
    for VER in (11, 14, 17, 20, 23):
        LANG = "c++"
        runlang()
else:
    run()
x = 0
