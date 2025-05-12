""" regexes.py
Collection of regular expressions used by a lexer.
"""

from __future__ import annotations

from pcpp.common import *
from pcpp.common import *
from pcpp.escape import *

class RegExes:
    """
    Collection of various REs used by the lexer.  Most are class variables,
    but some are customized by the language.

    All characters in the lexer data are 8-bit values.  Wider characters in
    the source file have been replaced by equivalent UCN escapes.  For
    example, 'δ' becomes '\\u03b4'.
    """
    # Data pertaining to unicode.
    uni_info: UniInfo

    def __init__(self, lang: Preprocessor.Language, pasting: bool = False):
        """
        Set public REs, which vary with preprocessor options.  Optional use in
        lexing pasted values.
        """

        def set_re(**re) -> None:
            """
            For each name=regex argument, sets the self.name = regex.  Wraps
            it in (?x) to remove whitespace and (?a) to restrict to ASCII.
            """
            for name, regex in re.items():
                if regex:
                    setattr(self, name, self.wrap(regex))


        # Newline.
        self.newline = r'(\n)'
        # Whitespace, other than newline, i.e. ' ', \f, \t, \v, \r.
        ws = r'(((?!\n)\s)+)'

        # Hex char.
        hex = r'[0-9a-fA-F]'
        hexlower = r'[0-9a-f]'
        # Hex nondigit
        hexletter = r'[a-fA-F]'
        # Digit
        digit = r'[0-9]'
        self.digit = digit
        # Octal digit
        oct = r'[0-7]'


        # All escapes will be replaced in the original data by single
        # codepoint characters they represent, if possible.  self.repl_escape
        # is the regex for the Replacer.  
        # Malformed escapes will be replaced by a canonical escape sequence.
        # This allows the character to be used in an identifier.  In quoted
        # tokens, they revert to original spelling.  These include:
        # -  \u or \U with invalid value for a codepoint.  Replaced by \uxxxx
        #    or \Uxxxxxxxx.  Uses lowercase hex digits.
        # -  \N{name} with an invalid name.  With clang this is case
        #    sensitive.  Replaced by \N{name} using uppercase letters and
        #    replacing _ with -.
        #
        # Other malformed escapes will be ignored, and the \ is lexed
        # separately (as an error token):
        # -  \u or \U with not enough hex digits.
        # -  \u{} or \o{} or \x{}.
        # -  \u{num} or \x{num} with num that is not all hex digits.
        # -  likewise for \o{oct} with octal digits.

        # The canonical bad escape replacement.  Absence of
        # any hex digits means the escape could not be given a character
        # value suitable for a quoted token.

        #bad_escape = rf'\x{{{hexlower}*}}'
        # Replacement for a bad escape sequence.  In a quoted token, this will
        # be reverted to the original spelling after the token is lexed.
        badescape = rf'\\{{{hexlower}*}}'

        # Escape sequences for UCNs of all varieties.  Regexes publishes the
        # regex for the body of each variety of escape as self.escbody_xxx.

        uni = self.uni_info = self.UniInfo()

        # universal-character-name (C99 6.9.1 or C++23 5.3)
        uni.ucn4 = rf'u{self.group(f"{hex}{{1,4}}")}'
        uni.ucn8 = rf'U{self.group(f"{hex}{{1,8}}")}'

        if lang.cplus_ver >= 2023 or lang.clang:
            # Escapes new to C++23, also valid in clang for all languages...

            # named-universal-character (C++23 5.3p3)
            nchar = r'[^\n}]'  # n-char = any but '}' or '\n'.
            uni.nuc = rf"N{self.delimited(nchar)}"

            self.match_nuc = lambda m: m.group()[1] == 'N'

            # universal-character-name (C++23 5.3p3)
            uni.ucndelim = rf"u{self.delimited(hex)}"

            hexbody = rf"x{self.delimited(hex, 'hexplus')}"
            octbody = rf"o{self.delimited(hex, 'octplus')}"
            cplusescbody = self.joinalts(
                uni.ucndelim,
                uni.nuc,
                hexbody,
                octbody,
                )
        else:
            #nucbody = ''
            #nuc = ''
            #nucgroup = ''
            self.match_nuc = lambda m: False
            #ucnhexbody = hexescbody = octescbody =''
            cplusescbody = ''

        uniescbody = self.joinalts(
            uni.ucn4,
            uni.ucn8,
            uni.ucndelim,
            uni.nuc,
            )

        # General unicode codepoint.  Mostly a single character with a valid
        # unicode value with optional group.  Otherwise a canonical replacement unicode escape
        # sequence.
        def codepoint(groupname: str = '') -> str:
            """ The regex for a codepoint, with optional group if invalid. """
            num = r'[\u0080-\U0010FFFF]'        # valid codepoint
            escape = rf'\\{uniescbody}'
            #escape = codepoint_esc
            if groupname:
                num = self.group(num, groupname)
            return self.joinalts(
                # Place this first so that the capturing groups are in correct
                # order.
                escape,                 # escape sequence for codepoint
                num,                    # valid codepoint
            )

        # Identifier characters come in four flavors:
        # 1. ASCII start, or 'AS'.
        # 2. ASCII continue, or 'AC'.  This is AS or a digit.
        # 3. Unicode start, or 'US'.  This varies with the lang settings.  
        #    - C++ and C23, it has the XID_START property in the unicode
        #      database.
        #    - C11, listed in standard annex D.1, except if listed in D.2.
        # 4. Unicode continue, or 'UC.  This varies with the lang settings.  
        #    - C++ and C23, it has the XID_CONTINUE property in the unicode
        #      database.
        #    - C11, listed in standard annex D.1.

        # An identifer has the form (AS | US) (AC | UC)*.  
        # There is no regex for US or UC, so they are lexed as a general
        # unicode codepoint, which may or not be valid at their locations in
        # the identifier.  The token function will check their validity and
        # change the lexed token if it finds any invalid codepoints.

        char_asc_start = '[a-zA-Z_]'        # AS character
        char_asc_cont = '[a-zA-Z_0-9]'      # AC character
        # A codepoint char whose value is validated by the token function.
        char_uni = codepoint()              # US or UC character.
        #char_uni_group = codepoint_group    # US or UC character with 'esc'

        # Initial ascii characters (at least one) of identifier.
        ident_asc_start = f'{char_asc_start}{char_asc_cont}*'

        # Start of an identifier, AS AC* or US, with groups 'ascstart' and
        # 'unistart'.
        ident_start = self.joinalts(
            # Starts with ascii char(s) as group 'ascstart'
            self.group(ident_asc_start, 'ascstart'),
            # Starts with codepoint as group 'unistart'
            self.group(codepoint('numstart'), 'unistart'),
            #self.group(codepoint('escstart'), 'unistart'),
            )

        # Continuation character(s) of an identifier.
        #ident_cont = self.joinalts(
        #    # ascii character(s)
        #    f'({char_asc_cont}+)',
        #    # single unicode character
        #    codepoint('esccont'),
        #    )
        # Continuation character(s) of an identifier, with groups 'asc' and
        # 'uni'.  'uni' group has group 'esccont'.
        ident_cont_group = self.joinalts(
            # ascii character(s) as group 'asc'
            self.group(f'({char_asc_cont}+)', 'asc'),
            # single unicode character as group 'uni'
            self.group(codepoint('numcont'), 'uni'),
            #self.group(codepoint('esccont'), 'uni'),
            )
        # The complete RE for the identifier, before the token function
        # handles any unicode characters.  
        # After the token function, it may be empty, in the case of a
        # invalid starting codepoint.
        ident = f'''
            {ident_start}           # AS or US
            (?P<cont>
                {ident_cont_group}  # AC * or UC
                *                   # 0 or more times
            )
            '''
        set_re(identifier=ident)
        set_re(char_uni_cont=codepoint('numcont'),)
        set_re(char_uni_start=codepoint('numstart'))
        
            #ucnescbody = self.group(rf'u{{{hex}+}}', 'ucn++')
            #hexescbody = self.group(rf'x{{{hex}+}}', 'hex++')
            #octescbody = self.group(rf'o{{{oct}+}}', 'oct++')

        # Original C escapes.  
        #
        escbody = self.joinalts(
            self.group(r'[\'"?\\abfnrtv]', 'char'),
            fr'u{self.group(f"{hex}{{0,4}}", "ucn")}',
            fr'U{self.group(f"{hex}{{0,8}}", "UCN")}',
            fr'x{self.group(f"{hex}{{0,2}}", "hex")}',
            fr'{self.group(f"{oct}{{1,3}}", "oct")}',
            )
        undefescbody = self.group(r'.', 'undef')
        # Universal character name.  \uxxxx or \Uxxxxxxxx or \u{x+}.
        #   Used in an identifier or escape sequence in a quoted string.
        #ucnbody = self.joinalts(
        #    fr'u{hex}{{4}}',
        #    fr'U{hex}{{8}}',
        #    cplus_escbody,
        #    )
        #ucn = fr'\\{ucnbody}'

        # Possible Unicode codepoint.  Outside ASCII range.
        unicode_range = r'\u00C0-\uD7FF\uE000-\U0010FFFF'

        # Identifier (C99 6.4.2.1), used in a user defined suffix.
        ident_nondigit = rf'([A-Za-z_]|[{unicode_range}])'
        nondigit = rf'([A-Za-z_])'
        self.ident_body = rf'{digit}|{ident_nondigit}'
        set_re(ident=rf'{ident_nondigit}({self.ident_body})*')

        # Source character set member (C99 5.2.1p3).
        if lang.emulate:
            # Include all ASCII graphics chars.
            srcchar_range = r'\x20-\x7E'
            self.ascii_not_source = ""
        else:
            # All from 0x20 - 0x7E, except $ @ and `.
            srcchar_range = r'\x20-\x23\x25-\x3F\x41-\x5F\x61-\x7E'
            self.ascii_not_source = "$@`"
        srcchar = rf'[{srcchar_range}\n\r\f\v{unicode_range}]'

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
        if lang.cplus_ver:
            if lang.cplus_ver >= 2017:
                strprefix = chrprefix = 'L u U u8'
            elif lang.cplus_ver >= 2011:
                strprefix =  'L u U u8'
                chrprefix =  'L u U'
        # C...
        elif lang.c_ver >= 2023:
            strprefix = chrprefix = 'L u U u8'
        elif lang.c_ver >= 2011:
            strprefix = chrprefix =  'L u U u8'
        else:
            # C99
            strprefix = chrprefix = 'L'

        strprefix = f"(?P<pfx>({'|'.join(strprefix.split())})?)"
        chrprefix = f"(?P<pfx>({'|'.join(chrprefix.split())})?)"

        # User-defined suffix for number, char, or string literal (C++ only,
        # also clang).  Any identifier that begins with '_'.  However, if it
        # does not begin with '_', clang lexes it anyway, with an error
        # diagnostic. 
        if lang.cplus_ver or lang.clang:
            opt_ud_sfx = f'(?P<ud_sfx>_{self.ident})?'
        else:
            opt_ud_sfx = '(?P<ud_sfx>)'

        # Escape sequences.
        # - UCNs.  \u... \U... \N...  In quoted tokens and identifiers.
        # - Basic.  Anything else starting with \.  In quoted tokens.
        #   - Simple.  \ [ ’ " ? \ a b f n r t v ]
        #   - Numeric. \x... \o...
        #   - Conditional.  \ + any other single character.
        #
        # All are used in the ESCAPE replacement stage, and have group names:
        # ucn, nuc, simple, hex, oct, other.
        #
        # In tokens, they have been replaced by either the corresponding
        # character or a canonical escape sequence.  The canonical sequence is
        # \x{...} with one or more lowercase hex digits.

        # Quoted text, which goes between the matching quotes in a quoted
        # token.  Includes text which does not follow the standards, i.e., bad
        # escape sequences.
        quoted = r'([^\\\n]|(\\(.|\n)))*?'

        # Escape sequence (C99 6.4.4.4).  Can appear in cchar or schar.
        #escapebody = self.joinalts(      # What can follow the backslash.
        #    r'[\'"?\\abfnrtv]',
        #    r'[oct]{{1,3}}',
        #    rf'x{hex}+',
        #    ucnbody,
        #    r'.',
        #    )
        escape = rf'\\({escbody})'

        # s-char.  Part of a string literal (C99 6.4.5).  
        #   Any source char other than newline, \ or ", or escape sequence.
        schar = fr'''
            ((?!["\\\n]).               # exclude ", \, newline.
            | {badescape}
            )
            '''
        # String literal.  
        # NEW: following the original PCPP which takes anything between the
        # quotes.
        set_re(string=rf'''
            {strprefix}
            \"(?P<val>
            {quoted}
            )\"
            {opt_ud_sfx}                # optional ud-suffix
            ''')
        # c-char.  Part of character constant (C99 6.4.4.4).
        cchar = fr"""
            ((?!['\\\n]).               # exclude ', \, newline.
            | {badescape}
            )
            """
        # Character constant  
        # NEW: following the original PCPP which takes anything between the
        # quotes.
        set_re(char=rf"""
            {chrprefix}
            \'(?P<val>
            {quoted}
            )\'
            {opt_ud_sfx}                # optional ud-suffix
            """)

        # h-char.  Part of a <...> header name (C99 6.4.7).
        # Any source char other than newline or >.
        hchar = fr'((?![>\n]).)'
        set_re(hhdrname=rf'<({hchar})*>')

        # q-char.  Part of a "..." header name (C99 6.4.7).
        # Any source char other than newline or ".
        qchar = fr'((?!["\n]).)'
        set_re(qhdrname=rf'\"({qchar})*\"')

        # Raw string (C++14 5.13.5).  Also accepted by GCC C.
        # Delimeter in a raw string.  Named group "delim".
        delim = r'(?P<delim>[^()\\\s]*)'
        # Complete raw string.
        set_re(rstring=rf'''
            (?s)
            {strprefix}
            R" {delim} \(.*?\) (?P=delim) "
            {opt_ud_sfx}                # optional ud-suffix
            ''')


        # Digit separator.  Only for C++.
        if lang.cplus_ver:
            digsep = "[']?"                 # Optional "'".
        else:
            digsep = ""
        # Otherwise, C++ and C are the same.

        # Preprocessing number (C99 6.4.8), (C++14 5.9).
        # Note, the grammar makes use of general identifier characters
        ppnum_exp = (           # A sign following an exponent char
            ((lang.c_ver or lang.cplus_ver >= 2014)
             and '(?<=[eEpP])[+-]')
            or '(?<=[eE])[+-]'      # C++11 doesn't have hex floats
            )
        ppnum_sep = (               # digit with separators, C23 and C++14.
            ((lang.c_ver >= 2023 or lang.cplus_ver >= 2014)
             and f"[']( {digit} | {nondigit} )")
            or None
            )
        ppnum_tail = self.joinalts( # Anything that can follow initial digit
            '[.]',
            ppnum_exp,              # Place before ident_nondigit
            digit,
            ident_nondigit,
            ppnum_sep,
            )
        ppnum_notail = f'(?!{ppnum_tail})'
        set_re(ppnum=rf'''
            [.]? {digit}
            ({ppnum_tail})*
            ''')

        # Integer literal (C99 6.4.4.1)
        usfx = r'([uU])'                    # unsigned-suffix
        lsfx = rf'''(
                    ll | LL                  # long-long-suffix
                    | l | L                  # long-suffix
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
        set_re(int=rf'''(
            (?P<num>                    # integer-constant
                {self.joinalts(*altconsts)}
            )
            {isfx}?                     # optional suffix
            {opt_ud_sfx}                # optional ud-suffix
            {ppnum_notail}              # Not part of longer ppnum
            )
            ''')

        # Float literal (C99 6.4.4.2)
        #   Decimal...
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
        #   Either decimal or hexadecimal...
        set_re(float=rf'''
            ({dfloat} | {hfloat})
            {opt_ud_sfx}
            {ppnum_notail}
            ''')
        set_re(dotfloat=rf'''
            {dfdotfloat}
            {opt_ud_sfx}
            {ppnum_notail}
            ''')

        # Patterns used to find repls in the input for each replacement stage.

        #replbody = self.joinalts(
        #    fr'(?P<splice>{ws}*\n)',             # Line splice
        #    ucnbody,
        #    )
        #altrepls: list[str] = [        # Alternative REs for repls pattern
        #    # These escapes might have a ??/ for the \, so put them first.
        #    fr'\\({replbody})',
        #    # The 9 trigraphs (C99 5.2.1.1)
        #    lang.trigraphs and r'\?\?[=\(/\)\'<\!>\-]',   # The 9 trigraphs
        #]
        #set_re(repls=self.joinalts(*altrepls))

        # The 9 trigraphs.
        set_re(repl_trigraphs=lang.trigraphs and r'\?\?[=\(/\)\'<\!>\-]')
        # Line splices, after trigraphs replaced.
        set_re(repl_splice=fr'\\{ws}*\n')
        # Escape sequences, after trigraphs and splices replaced.
        repl_escape_body = self.joinalts(
            escbody,
            cplusescbody,
            undefescbody,
            )
        set_re(repl_escape=fr'\\{repl_escape_body}')

    class UniInfo:
        # Patterns for different kinds of UCN escapes, after the initial '\'.
        ucn4: str               # uxxxx
        ucn8: str               # Uxxxxxxxx
        ucndelim: str = None    # u{x...} - C++23 or clang only
        nuc: str = None         # N{c...} - C++23 or clang only

    # Some utility functions which can be called directly...

    @staticmethod
    def joinalts(*alts: str) -> str:
        """
        Make a non-capturing group from (non-empty) given alternatives.
        """
        alts = tuple(filter(None, alts))
        joined = '| '.join(alt + '\n' for alt in alts)
        return f'(?:{joined})'


    @staticmethod
    def group(regex: str, name: str = '') -> str:
        """ Embed regex in a named or unnamed capturing group. """
        if name:
            return rf'(?P<{name}>{regex})'
        else:
            return rf'({regex})'

    @staticmethod
    def wrap(regex: str) -> str:
        """
        Wrap a regex in (?x) to remove whitespace and (?a) to restrict to
        ASCII.  Use non-capturing group.
        """
        return '(?:(?x)(?a)' + regex + ')'

    def delimited(self, c: str, groupname: str = '') -> str:
        """
        A {c...} delimited sequence.  If data lacks the closing '}', it
        will end before a quote or escape or newline.  Includes a
        capturing group around all initial valid characters.  If there are
        other characters present, this group will be shorter than
        expected or won't end with a '}'.
        """
        chars = rf'{c}*'
        group = self.group(chars, groupname)
        others = r'[^}}\'"\n]*'
        return rf'{{{group}{others}}}?'

x = 0

