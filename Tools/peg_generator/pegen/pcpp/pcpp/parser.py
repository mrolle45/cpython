""" parser.py module.
Generates preprocessing tokens from a source file, using lex.
Specialized Lexer.
    Translation phases 1 and 2: trigraphs and line splicing.
    Tracks line and column positions.
Customization for Preprocessor via PreprocessorHooks mixin class.
Specialized parsers using yacc, for preprocessing token syntax.
    Such as 'defined' in a control expression,
    or arguments for a function macro,
    or removing whitespace tokens where they are not relevant.
"""
#!/usr/bin/python
# Python C99 conforming preprocessor parser config
# (C) 2017-2020 Niall Douglas http://www.nedproductions.biz/
# and (C) 2007-2017 David Beazley http://www.dabeaz.com/
# Started: Feb 2017
#
# This C preprocessor was originally written by David Beazley and the
# original can be found at https://github.com/dabeaz/ply/blob/master/ply/cpp.py
# This edition substantially improves on standards conforming output,
# getting quite close to what clang or GCC outputs.

from __future__ import annotations
from __future__ import generators, print_function, absolute_import, division

import sys, re, os
import dataclasses
import enum

from ply import yacc

# Python 2/3 compatible way of importing a subpackage
oldsyspath = sys.path
sys.path = [ os.path.join( os.path.dirname( os.path.abspath(__file__) ), "ply" ) ] + sys.path

sys.path = oldsyspath
del oldsyspath

in_production = 0  # Set to 0 if editing pcpp implementation!

# Some Python 3 compatibility shims
if sys.version_info.major < 3:
    STRING_TYPES = (str, unicode)
else:
    STRING_TYPES = str


# ------------------------------------------------------------------
# Preprocessor event hooks
#
# Override these to customise preprocessing
# ------------------------------------------------------------------

class PreprocessorHooks(object):
    """Override these in your subclass of Preprocessor to customise preprocessing"""
    def __init__(self):
        self.lastdirective = None

    def on_error(self, file, line, msg):
        """Called when the preprocessor has encountered an error, e.g. malformed input.
        
        The default simply prints to stderr and increments the return code.
        """
        self._error_msg(file, msg, line)

    def _error_msg(self, source: Source, msg: str, line: int = 0, col: int = 0, warn: bool = False):
        """ Prints error message to stderr  and increments the return code. """
        file = source.filename
        if self.verbose < 2:
            file = os.path.basename(file)
        if line:
            file = f"{file}:{line}"
            if col:
                file = f"{file}:{col}"

        type = warn and 'warning' or 'ERROR'
        self.log.write(f"{type}: {msg}", source=source, lineno=line, colno=col)
        msg = f"{file} {type}: {msg}"
        print(msg, file = sys.stderr)
        if not warn:
            self.return_code += 1

    def on_error_token(self, token: LexToken, msg: str, warn: bool = False):
        """ Called when the preprocessor has encountered an error or warning associated with a token.
        """
        lex = token.lexer
        self._error_msg(lex.source, msg, token.lineno, lex.colno, warn=warn)

    def on_warn_token(self, token: LexToken, msg: str):
        """ Called when the preprocessor has encountered a warning associated with a token.
        """
        self.on_error_token(token, msg, warn=True)

    def on_file_open(self,is_system_include,includepath):
        """Called to open a file for reading.
        
        This hook provides the ability to use ``chardet``, or any other mechanism,
        to inspect a file for its text encoding, and open it appropriately. Be
        aware that this function is used to probe for possible include file locations,
        so ``includepath`` may not exist. If it does not, raise the appropriate
        ``IOError`` exception.
        
        The default calls ``io.open(includepath, 'r', encoding = self.assume_encoding)``,
        examines if it starts with a BOM (if so, it removes it), and returns the file
        object opened. This raises the appropriate exception if the path was not found.
        """
        if sys.version_info.major < 3:
            assert self.assume_encoding is None
            ret = open(includepath, 'r')
        else:
            ret = open(includepath, 'r', encoding = self.assume_encoding)
        bom = ret.read(1)
        #print(repr(bom))
        if bom != '\ufeff':
            ret.seek(0)
        return ret

    def on_include_not_found(self,is_malformed,is_system_include,curdir,includepath):
        """Called when a #include wasn't found.
        
        Raise OutputDirective to pass through or remove, else return
        a suitable path. Remember that Preprocessor.add_path() lets you add search paths.
        
        The default calls ``self.on_error()`` with a suitable error message about the
        include file not found if ``is_malformed`` is False, else a suitable error
        message about a malformed #include, and in both cases raises OutputDirective
        (pass through).
        """
        if is_malformed:
            msg = f"Malformed #include statement: {includepath!r}"
        else:
            msg = f"Include file {includepath!r} not found"
        self._error_msg(self.lastdirective.source, msg, self.lastdirective.lineno)
        raise OutputDirective(Action.IgnoreAndPassThrough)
        
    def on_unknown_macro_in_defined_expr(self,tok):
        """Called when an expression passed to an #if contained a defined operator
        performed on something unknown.
        
        Return True if to treat it as defined, False if to treat it as undefined,
        raise OutputDirective to pass through without execution, or return None to
        pass through the mostly expanded #if expression apart from the unknown defined.
        
        The default returns False, as per the C standard.
        """
        return False

    def on_unknown_macro_in_expr(self,ident):
        """Called when an expression passed to an #if contained an unknown identifier.
        
        Return what value the expression evaluator ought to use, or return None to
        pass through the mostly expanded #if expression.
        
        The default returns an integer 0, as per the C standard.
        """
        return 0
    
    def on_unknown_macro_function_in_expr(self,ident):
        """Called when an expression passed to an #if contained an unknown function.
        
        Return a callable which will be invoked by the expression evaluator to
        evaluate the input to the function, or return None to pass through the
        mostly expanded #if expression.
        
        The default returns a lambda which returns integer 0, as per the C standard.
        """
        return lambda x : 0
    
    def on_directive_handle(self,directive,toks,ifpassthru,precedingtoks):
        """Called when there is one of
        
        define, include, undef, ifdef, ifndef, if, elif, else, endif
        
        Return True to execute and remove from the output, raise OutputDirective
        to pass through or remove without execution, or return None to execute
        AND pass through to the output (this only works for #define, #undef).
        
        The default returns True (execute and remove from the output).

        directive is the directive, toks is the tokens after the directive,
        ifpassthru is whether we are in passthru mode, precedingtoks is the
        tokens preceding the directive from the # token until the directive.
        """
        self.lastdirective = directive
        return True
        
    def on_directive_unknown(
            self, directive: LexToken, toks: Tokens,
            ifpassthru: bool, precedingtoks: Tokens):
        """Called when the preprocessor encounters a #directive it doesn't understand.
        This is actually quite an extensive list as it currently only understands:
        
        define, include, undef, ifdef, ifndef, if, elif, else, endif
        
        Return True to remove from the output, raise OutputDirective
        to pass through or remove, or return None to
        pass through into the output.
        
        The default handles #error and #warning by printing to stderr and returning True
        (remove from output). For everything else it returns None (pass through into output).

        directive is the directive, toks is the tokens after the directive,
        ifpassthru is whether we are in passthru mode, precedingtoks is the
        tokens preceding the directive from the # token until the directive.
        """
        def do_msg(warn: bool):
            self._error_msg(
                directive.source, self.showtoks(toks), directive.lineno, warn=warn)

        if directive.value == 'error':
            do_msg(False)
            self.return_code += 1
            return True
        elif directive.value == 'warning':
            do_msg(False)
            return True
        return None
        
    def on_potential_include_guard(self,macro):
        """Called when the preprocessor encounters an #ifndef macro or an #if !defined(macro)
        as the first non-whitespace thing in a file. Unlike the other hooks, macro is a string,
        not a token.
        """
        pass
    
    def on_comment(self,tok):
        """Called when the preprocessor encounters a comment token. You can modify the token
        in place. You must return True to let the comment pass through, else it will be removed.
        
        Returning False or None modifies the token to become whitespace, becoming a single space.
        """
        return None



