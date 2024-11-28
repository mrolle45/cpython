""" regexes.py
Collection of regular expressions used by a lexer.
"""

from __future__ import annotations

from pcpp.common import *

class RegExes:
    """ Collection of various REs used by the lexer.
    Most are class variables, but some are customized by the preprocessor.
    """
    def __init__(self, prep: Preprocessor):
        """ Set public REs, which vary with preprocessor options. """

        def joinalts(*alts: str) -> str:
            """ Make a regex from (non-empty) given alternatives. """
            return ' | '.join(filter(None, alts))

        def verb(regex: str) -> str:
            """ Add a VERBOSE flag to regex. """
            return '(?x)' + regex

        # Newline.
        self.newline = r'(\n)'
        # Whitespace, other than newline, i.e. ' ', \f, \t, \v, \r.
        ws = r'(((?!\n)\s)+)'

        # Hex char.
        hex = r'[0-9a-fA-F]'
        # Hex nondigit
        hexletter = r'[a-fA-F]'
        # Digit
        digit = r'[0-9]'
        self.digit = digit

        # Universal character name.  \uxxxx or \Uxxxxxxxx.
        #   The backslash might be a trigraph.
        #   Used in an identifier or escape sequence in a quoted string.
        ucnbody = fr'(u{hex}{{4}} | U{hex}{{8}})'
        ucn2 = fr'\\{ucnbody}'
        escape = prep.trigraphs and r'(\\|\?\?/)' or r'\\'
        ucn = fr'{escape}(u{hex}{{4}}|U{hex}{{8}})'
        # Unicode character escape by its name (C++23 5.3p3).
        nuc = (
            (prep.cplus_ver >= 2023 or prep.clang)
            and verb(rf'''
            {escape}N{{(
                    [^\n}}]         # n-char = any but }} or '\n'.
                    +               # n-char-sequence:
                    )
                }}
            '''
            ))
        namedchar = (
            (prep.cplus_ver >= 2023 or prep.clang)
            and verb(rf'''
            {escape}N{{(?P<nuc>
                    [^\n}}]         # n-char = any but }} or '\n'.
                    +               # n-char-sequence:
                    )
                }}
            '''
            ))
        # Escape sequence (C99 6.4.4.4).  Can appear in cchar or schar.
        escape = rf'''
            \\[\'"?\\abfnrtv]
            |\\[0-7]{{1,3}}
            |\\x{hex}+
            |\\{ucnbody}
            |\\{nuc}
            |\\.
            '''
        # Identifier (C99 6.4.2.1).
        ident_nondigit = rf'([A-Za-z_]|{ucn}|{nuc})'
        self.ident_body = rf'{digit}|{ident_nondigit}'
        self.ident = verb(rf'{ident_nondigit}({self.ident_body})*')

        # Possible Unicode codepoint.  Outside source charset.
        unicode = '[\u0100-\U0010FFFF]'

        # Unicode characters cannot be in the actual lexer regexes, because 
        # they get written to the lextab, which cannot be changed to utf-8.
        # Solution: replace all the unicode characters in the lexdata
        # then just lex their UCN spellings.

        # Source character set member (C99 5.2.1p3).
        if prep.emulate:
            # Include all ASCII graphics chars.
            self.srcchar = r'[!-~\s]'
            self.ascii_not_source = ""
        else:
            # All from 0x20 - 0x7E, plus \n \r \f \v \t minus $ @ and `.
            self.srcchar = r'[!-#%-?A-_a-~\s]'
            self.ascii_not_source = "$@`"

        # Prefix to a char constant or string literal.
        #   Available prefixes vary with the language and version.
        #   u8 sometimes recognized only for strings.
        #
        #                   L   u   U   u8"" u8''
        #                   ---------------------
        #       c99         ✓
        #       C11, C17    ✓  ✓  ✓  ✓   ✓  
        #       C23         ✓  ✓  ✓  ✓   ✓
        #       C++11, 14   ✓  ✓  ✓  ✓
        #       C++17 - 23  ✓  ✓  ✓  ✓   ✓

        # gcc and clang don't follow these standards in a few cases.

        # C++...
        if prep.cplus_ver:
            if prep.cplus_ver >= 2017:
                strprefix = chrprefix = 'L u U u8'
            elif prep.cplus_ver >= 2011:
                strprefix =  'L u U u8'
                chrprefix =  'L u U'
        # C...
        elif prep.c_ver >= 2023:
            strprefix = chrprefix = 'L u U u8'
        elif prep.c_ver >= 2011:
            strprefix = chrprefix =  'L u U u8'
        else:
            # C99
            strprefix = chrprefix = 'L'

        strprefix = f"(?P<pfx>({'|'.join(strprefix.split())})?)"
        chrprefix = f"(?P<pfx>({'|'.join(chrprefix.split())})?)"

        # User-defined suffix for number, char, or string literal (C++ only,
        # also clang).  Any identifier that begins with _.
        if prep.cplus_ver or prep.clang:
            opt_ud_sfx = f'(?P<ud_sfx>(?=_){self.ident})?'
        else:
            opt_ud_sfx = '(?P<ud_sfx>)'

        # s-char.  Part of a string literal (C99 6.4.5).  
        #   Any source char other than newline, \ or ", 
        #   or escape sequence, or unicode codepoint,
        schar = fr'''
            ((?!["\\\n]){self.srcchar}      # exclude ", \, newline.
            | {escape}
            )
            '''
        self.string = verb(rf'''
                        {strprefix}
                        \"(?P<val>{schar}*)\"
                        {opt_ud_sfx}                # optional ud-suffix
                        ''')
        # c-char.  Part of character constant (C99 6.4.4.4).
        cchar = fr"""
            ((?!['\\\n]){self.srcchar}      # exclude ', \, newline.
            | {escape}
            )
            """
        self.char = verb(rf"""
                        {chrprefix}
                        \'(?P<val>{cchar}*)\'
                        {opt_ud_sfx}                # optional ud-suffix
                        """)

        # h-char.  Part of a <...> header name (C99 6.4.7).
        # Any source char other than newline or >.
        hchar = fr'((?![>\n]){self.srcchar})'
        self.hhdrname = verb(rf'<({hchar})*>')

        # q-char.  Part of a "..." header name (C99 6.4.7).
        # Any source char other than newline or ".
        qchar = fr'((?!["\n]){self.srcchar})'
        self.qhdrname = verb(rf'\"({qchar})*\"')

        # Raw string (C++14 5.13.5).  Also accepted by GCC C.
        # Delimeter in a raw string.  Named group "delim".
        delim = r'(?P<delim>[^()\\\s]*)'
        # Complete raw string.  Named group "rstring".
        self.rstring = verb(fr'''(?s)
                (?P<rstring>
                    {strprefix} R" {delim} \(.*?\) (?P=delim) ")
            ''')
        # Pattern used to find repls in the input.
        backslash = joinalts(
            r'\\',
            prep.trigraphs and r'\?\?/',
            )
        altrepls: list[str] = [        # Alternative REs for repls pattern
            fr'(?P<splice>({backslash}){ws}*\n)',             # Line splice
            # GCC changes UCNs to 8-chars lowercase.  Place before trigraphs.
            prep.emulate and rf'(?P<ucn>{ucn})',
            # The 9 trigraphs (C99 5.2.1.1)
            prep.trigraphs and r'\?\?[=\(/\)\'<\!>\-]',   # The 9 trigraphs
            unicode,                        # Possibly a unicode codepoint.
            namedchar,
        ]
        self.repls = verb(joinalts(*altrepls))

        # Digit separator.  Only for C++.
        if prep.cplus_ver:
            digsep = "[']?"                 # Optional "'".
        else:
            digsep = ""
        # Otherwise, C++ and C are the same.

        # Preprocessing number (C99 6.4.8), (C++14 5.9).
        # Note, the grammar makes use of general identifier characters
        ppnum_exp = (
            ((prep.c_ver or prep.cplus_ver >= 2014) and '[eEpP][+-]')
            or '[eE][+-]'       # C++11 doesn't have hex floats
            )
        ppnum_sep = (               # digit with separators, C23 and C++14.
            ((prep.c_ver >= 2023 or prep.cplus_ver >= 2014)
             and f"[']({digit} | {nondigit}")
            or None
            )
        ppnum_tail = joinalts(      # Anything that can follow initial digit
            '[.]',
            ppnum_exp,              # Place before ident_nondigit
            digit,
            ident_nondigit,
            ppnum_sep,
            )
        ppnum_notail = f'(?!{ppnum_tail})'
        self.ppnum = verb(rf'''
            [.]? {digit}
            ({ppnum_tail})*
            ''')

        # Integer literal (C99 6.4.4.1)
        usfx = r'([uU])'                    # unsigned-suffix
        lsfx = rf'''(
                    ll | LL                  # long-long-suffix
                    | [lL]                   # long-suffix
                    | wb | WB                # bit-precise-int-suffix
                )
                '''
        isfx = rf'''(?P<sfx>                # integer-suffix
                        {usfx} {lsfx}?
                        | {lsfx} {usfx}?
                )
                '''
        def iconst(tag: str, pfx: str, first: str, after: str = '') -> str:
            """ Regex for one flavor of integer constant. """
            return rf'''
                (?P<{tag}>
                    {pfx} {first}
                    (
                        {digsep}
                        {after or first}
                    )*
                )
                '''
        altconsts = [
            iconst('dec', '', '[1-9]', '[0-9]'),    # decimal-constant
            iconst('hex', '0[xX]', hex),            # hexadecimal-constant
            iconst('bin', '0[bB]', '[01]'),         # binary-constant
            # Put after hex and binary!
            iconst('oct', '0', '', '[0-7]'),        # octal-constant
        ]
        self.int = verb(
            rf'''
                ((?P<num>                   # integer-constant
                    {joinalts(*altconsts)}
                )
                {isfx}?                     # optional suffix
                {opt_ud_sfx}                # optional ud-suffix
                )
                {ppnum_notail}              # Not part of longer ppnum
            ''')

        # Float literal (C99 6.4.4.2)
        #   Decimal...
        #       Distinguish whether it starts with a '.' or not.
        dfdigits = r'([0-9](\'?[0-9])*)'        # digit-sequence
        dfdotfrac = rf'[.]{dfdigits}'           # '.' plus fractional-const
        dffrac = rf'{dfdigits}[.]{dfdigits}?'   # other fractional-constant
        dfexp = rf'([eE][-+]?{dfdigits})'       # exponent-part
        dfsfx = rf'([flFL]|df|dd|dl|DF|DD|DL)'  # floating-suffix
        # decimal-floating-constant...
        dfdotfloat = rf'{dfdotfrac}{dfexp}?{dfsfx}?'    # with leading '.'
        dfloat = rf'''(                                 # otherwise
                        {dffrac}{dfexp}?{dfsfx}?           
                        | {dfdigits}{dfexp}{dfsfx}?
                    )
                    '''
        #   Hexadecimal...
        hfdigits = rf'({hex}(\'?{hex})*)'       # hexadecimal-digit-seq
        hffrac = rf'''(                         # hexadecimal-frac-const
                        {hfdigits}?[.]{hfdigits}    
                        | {hfdigits}[.]
                    )
                    '''
        hfexp = rf'[pP][-+]?{dfdigits}'         # binary-exponent-part
        hfsfx = rf'[flFL]'                      # floating-suffix
        hfloat = rf'''(                         # hexadecimal-float-const
                        0[xX] {hffrac} {hfexp} {hfsfx}?
                        | 0[xX] {hfdigits} {hfexp} {hfsfx}?
                    )
                    '''
        self.float = verb(rf'({dfloat}|{hfloat}){opt_ud_sfx}{ppnum_notail}')
        self.dotfloat = verb(rf'{dfdotfloat}{opt_ud_sfx}{ppnum_notail}')


