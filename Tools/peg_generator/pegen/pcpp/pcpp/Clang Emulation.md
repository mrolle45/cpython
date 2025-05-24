# Emulation of `clang` in `PCPP`.
This document describes how `clang` preprocessor operates.  It is based on the
source code for **version 20.0**.

Clang's behavior varies in some ways with the platform on which it is
installed.  In particular, Microsoft extensions and VC++ compatibility are the
default on Windows platforms.  `pcpp` emulates `clang` running without these.
Therefore, if you are running `clang` to compare it with `pcpp`, you should
supply ***-fno-ms-extensions*** on your command line.  
## Token lexing
Performed by **Lexer::Lex()**.  Returns a Token.
Delegated to **Lexer::LexTokenInternal()**.  
Input buffer is plain char[], so utf encodings have to be decoded.  
**getCharAndSize()**.  Gets next char, and how many chars were used.  Skipped
chars include trigraphs and line splices. 
    This is called in many places. Thus they operate on post-trigraph and
    post-splice input.  
Skips over horizontal whitespace.  ' ', '\t', '\f', '\v',  
Switch on next char(s)...
* *0-9: LexNumericConstant().  Gets anything matching pp-number in the Standard.  Prints as-is.
* a-z, A-Z, _: LexIdentifierContinue().  u, U, L, R first check for being a quoted literal prefix.
* $: LexIdentifierContinue(), if supported language opt.
* Quoted tokens...
    * prefix+": **LexStringLiteral()** for prefix type.  Also used for "..." header name.
    * prefix+R": **LexRawStringLiteral()** for prefix type.  Only for
      supported language opts.  These are:
      * C++11
      * C99 with GNU
    * prefix+': **LexCharConstant()** for prefix type.
    * <: when looking for file name, call **LexAngledStringLiteral()**.

    prefix is u, u8, U, L, or nothing.  u8 is in the standard only for:
    - C11, C\++11 for string literals.
    - C23, C\++14 for char constants.  Note: **u8 is not recognized in
      C++14!**

    The content of the literal is whatever is in between the quotes, verbatim.  No special
    interpretation is done, except for \\, \n, and \r.  The UTF-8 encoding is preserved.
    That is, non-ascii characters appear in the literal as their UTF-8
    encoding.  
    In C++11, lexing includes a ud-suffix (but not for <...>).
    **LexUDSuffix()** looks for an identifier, but if it doesn't start with a
    **_** it will complain.
    In a conditional expression, a char constant token is evaluated, and certain errors are detected:
	A surrogate codepoint or codepoint > maximum value.
* `\`: a unicode escape.  Call **tryReadUCN(&Result)** (see below).  Handles \uxxxx, \Uxxxxxxxx,
  \u\{x...}, and \N\{name}.  
    \u{} or \U{...} is illegal.  Values 0 - 1F and 7F - 9F and D800 - DFFF are illegal.
    All others are OK, even if not valid unicode codepoints, including those >
    10FFFF.  
    All others lex the \ by itself as an `unknown`.  It will write '\' to the output.
* **'**, **"**: Call **LexCharConstant()** or **LexStringLiteral()** for plain type.
    Basically grabs everything up to the matching ending quote or \n or \r, verbatim.
    This is after trigraphs and splices.  There is no syntax for the quoted content.
* If not an ASCII character, call convertUTF8Sequence() to try and extract a unicode
    codepoint, as a UTF32.  If it's not all whitespace, call LexUnicodeIdentifierStart().
    Thus all codepoints begin identifiers, or are illegal.
## Identifiers
An identifier consists of:
- A valid **start char**.
- Zero or more valid **continue char**'s.
 
Various standards have different definitions of these characters.  Each is
divided into ASCII and Unicode characters.  clang follows these standards.  
In all standards, ASCII start chars are `[a-zA-Z_]` , and continue chars are `[a-zA-Z_0-9]`.  
* C99.  Appendix D lists all continue chars.  Start chars are the same, except
  for `[0-9];.
* C11 .. C17.  Appendix D.1 lists all continue characters.  Appendix D.2 lists
  those continue characters which are *not* start characters.
* C23, C++.  Uses the XID_START XID_CONTINUE properties from the Unicode
  database (version 15.0).

If the first character is unicode but not an ident start,  then it is ignored,
the result is an unknown type token.  Following characters are not examined.  
	An incomplete UCN is lexed as a bare \ (type = unknown).  This could be 
	a delimited \u{x...} without proper hex digits, or \u or \U with too
	few hex digits.  
If a following character is unicode but not an ident continue, this ends the
identifier.  
Special cases:
* A surrogate codepoint is an error, and is dropped, ending the identifier.
* A value >= 110000 is an error.  However, it is ignored, and scanning the
  identifier continues.  Thus, 'X\U00110000Y' becomes 'XY'.

First char is ASCII start: **LexIdentifierContinue()** (see below).  Note, the
lexer handles L, R, u,
u8, and U as char/string literal prefixes and does not try to get an identifier.  
First char is \\: codepoint = **tryReadUCN(&Result)** (see below).  Value of 0 is failure.
* If success:
	    If unicode whitespace, ignore this and try next character.
	    Else LexUnicodeIdentifierStart(codepoint).
* If fail:
	    Result = unknown type.  Value = the char(s) examined.
## tryReadUCN(Token \* Result = nullptr):
Initial \ seen already.  Returns the codepoint if valid, else 0.  
Check next char and get the codepoint.  0 means an error.
- u, U: tryReadNumericUCN().  Looking for uxxxx, Uxxxxxxxx, or u{x...}.
		u{x...} is fail if u{}, or x... >= 10000000.
- N: tryReadNamedUCN().  Looking for N\{c...}.  Fail if newline seen, or N\{}
  or no \{.  
    Get codepoint for name 'c...'.
    If fail, but get codepoint for 'C...' (i.e., loose matching), issue a warning and use codepoint.

If codepoint < A0, fail.  Message for control char if < 20 or >= 7F, else for
ASCII char.  
If codepoint is a surrogate, fail.

Returns the codepoint if successful.
## LexUnicodeIdentifierStart(codepoint):
  codepoint is the start character.
	Check isAllowedInitiallyIDChar(codepoint).
	If success: LexIdentifierContinue()
## LexIdentifierContinue():
Loop to examine next chars.  
-   Consume ASCII continue char.  Possibly $ if options allow this.  
    - \\: **tryConsumeIdentifierUCN()**.
	Calls **tryReadUCN(nullptr)** (see above).  If OK, isAllowedIDChar(codepoint).
	This checks the unicode table.  
	ASCII codepoint or unicode whitespace: fail.  End the loop.  
    - \>= 80: **tryConsumeIdentifierUTF8Char()**.  The codepoint is encoded as UTF-8 in the data.
	Same checks as with \ above.  
    - Anything else, or fail above, end the loop. 
    
Note: UnicodeWhitespaceCharRanges in UnicodeCharSets.h defines unicode whitespace characters.
    XIDStartRanges and XIDContinueRanges used by isAllowedIDChar().  These don't include any
    whitespace characters, so there's no need to check for them.
## Conditional expressions
The condition of an `#if` or `#elif` directive is a constant expression.  It has
constants and operators.
If there are any errors in the expression, it is treated as `false`.  The rest
of the directive is skipped but invalid tokens issue diagnostics.  
Any ASCII character in the form of a UCN is an error.
### Char constant value
The token is given to a `CharLiteralParser` object, in LiteralSupport.cpp.  
    Its job is to make a vector of codepoints pieces ofthe token value.  Also sets HadError in some cases.
* Ordinary character is itself.
* A UCN (\u, \U, or \N) calls ProcessUCNEscape().  Note, in_char_string_literal is true.
* Any other escape calls ProcessCharEscape().
	\x uses the following hex digits.  Error message if no digits, but value is 0.
Any errors are issued as diagnostics, but there's still a value of 0.

HadError is set when:
*	Ordinary character or UCN codepoint exceeds maximum codepoint for the prefix.
*	Multiple characters in a literal with a prefix.
*	ProcessUCNEscape result = false.
        * codepoint D800 - DFFF or >= 110000.
        * codepoint < A0 except $, @, `, but OK for C23 and C++11.
        * Numeric: \uxxxx, \Uxxxxxxxx, \u{x...}.  
	    Missing hex digits.  
	    Value >= 1_0000_0000.  
	    Missing }, or non-hex digit seen.  
	    Too few hex digits.  
        * Named:  
	    Missing `{` or `}`, or `{}`.  
	    Name unknown.  Strict matching.

    * Delimited escape \x{ either empty or found non-hex or end of value.
    * \x value overflow, >= 1_0000_0000.
    * \o{ with missing }, or \o{} or non-octal digit, or value overflow,
    >= 1_0000_0000 or max for type.

The result value is made from the codepoints.  There are various checks for value overflow and
    truncations.  
If there were any errors (i.e. diagnostics issued), the entire conditional
expression is considered `false`.  
User-defined suffixes are not allowed.  However, `clang` just issues an error
diagnostic and handles the token without the suffix.
### Integer constant value
User-defined suffixes are not allowed.  However, `clang` just issues an error
diagnostic and handles the token without the suffix.

## `Token` class, in include/Lex/Token.h.
The term "spelling" refers to where the value of the Token is found in the source file.
    It is indicated by a SourceLocation object.
Loc = location within the source file.
The LeadingSpace flag is used to generate a single space character in front of the value.
LeadingSpace is a property of a Token and also of a Lexer.  It is indicated by the symbol '⦾' or '•'.
    •token means token.Flags & 0x02.
    Lexer.HasLeadingSpace
    TokenLexer.HasLeadingSpace
    VAOptExpansionContext.LeadingSpaceForStringifiedToken
It is set as follows:
1.  The Lex knows if it previously lexed some whitespace, and sets the token.
    Lex also tracks LeadingEmptyMacro and IsStartOfLine.
    These flags can be carried over from an earlier token by
    Lexer::PropagateLineStartLeadingSpaceInfo().
    SkipWhitespace() will set on the following Token, or with -C, returns a whitespace token.
	Called after seeing a newline or other whitespace character.
    After a // comment, the next Token has no LeadingSpace initially.
    After a /* .. */ comment, the next token has LeadingSpace.
2.  TokenLexer scans the tokens of a macro call, starting with the macro identifier and including
	the ( ... ) argument list.  Or from an array of Tokens.
    •TokenLexer.  Initially copied from the macro •call or the first •token of the array.
    TokenLexer.NextTokenGetsSpace sometimes gives a space to the next token but not always.
    At the last token of the macro, set space if either HasLeadingSpace or NextTokenGetsSpace.
	This token is passed on to the enclosing lexer.
    With # __VA_OPT___ (...), 
3.  After a #define name x ..., for an object macro, sets ⦾x.
4.  State is propagated from a macro identifier (a call) to the first replacement token.  In Preprocessor::HandleMacroExpandedIdentifier().
5.  First token of a #pragma.
6.  Lots of stuff happens during macro expansion.  Lex/TokenLexer.cpp
	By the way, HashAt refers to a #@ token, which is a Microsoft extension, resulting in ' characters
	    being escaped instead of " characters.
    This handles tokens during a single macro expansion.
    NextTokGetsSpace is set while iterating through the tokens, if token has LeadingSpace.
	This propagates to the following token after # or ## operator.

    `# VA_OPT(...)` carries •# to the result string •token.
    Likewise for `#` param.
Use of LeadingEmptyMacro...
    If macro has no expansion tokens, the macro call token sets it.
    Propagates from the EOF token of a file to the current token of the including file.
    An empty macro expansion propagates from the call token to the current lexer.
    It apparently is never actually used, only gets pushed around.

## `TokenLexer` class
This goes through a sequence of pp-tokens already obtained from a Lexer, and does some transformations.
It optionally does macro expansion of macro identifiers.  Argument list cannot go past the end.
    The DisableMacroExpansion initialization parameter turns this off.  Normally, expansion is performed.
It is created either by the constructor, or by re-using an older object and calling Init() with same args.
TokenLexer is created in one of two places...
- Preprocessor.EnterTokenStream(tokens, DisableMacroExpansion, IsReinject).  This is overloaded.
	(tokens) can be either a unique_ptr<Token> and a count, or an ArrayRef<Token>.
	Called from...
	    Printing a #pragma where the contents haven't already been expanded.
	    Expanding a function argument.  In TokenLexer::ExpandFunctionArguments().
		Called the first time the arg is needed.  Ends with an eof token.
	    An annotation token.  No expansion.
	    An #embed directive.  No expansion.
	    Preprocessor::ReadMacroCallArgumentList.  No expansion.
	    Various post-preprocessor parsing methods.
- Preprocessor.EnterMacro(EnterMacro(Token &, SourceLocation, MacroInfo *, MacroArgs *Args)).
	Tokens start from given macro call token through the token at given location, which is either
	the object macro or the ')' at the end of the function macro argument list.
	Called from Preprocessor::HandleMacroExpandedIdentifier().
	    Called from Preprocessor::HandleIdentifier() when identifier has a macro definition.
		Called from the Lexer.
## How to get the location (file and line and column) from a token:
    The source file is in the form of a FileID.
    SourceLocation::getFromRawEncoding(Token.Loc), or Token.getLocation(), returns a SourceLocation.
    This is just an integer.  We need to use the SourceManager to get a line number.
    Preprocessor::getSourceManager() gets the current SourceManager
    The SourceLocation is not the offset of the spelling in the source file, but rather a place in a global view of all source files.  The SM translates a location into a file ID and a file offset.
    SM.getFileID(location) returns the file ID.
    SM.getPresumedLoc(Loc) returns a PresumedLoc object, containing Filename, Line, and Col.
    The Filename is a plain char *, so use strcmp() to test it.

    Call SM.getLineNumber(file ID, position, NULL).

## Macro substitution and expansion
Performed by the TokenLexer.  Function macros only.
Tokens points to first token, NumTokens = number of tokens.  This is set by constructor.
ActualArgs is set by constructor.  Only for function macro.  NULL otherwise.

**ExpandFunctionArguments()** is called by Init() for a function macro.
    Transforms Tokens[NumTokens] and replaces them if anything changed.
    Does everything which only function macros do, i.e., parameter substitution and expansion,
	and stringization.  `__VA_OPT__(...)` is treated like a parameter.  
    Loops over Tokens and builds ResultTokens array.  
	CurTok = next token in Tokens in the loop.  
	•lexer is kept in lexer.NextTokGetsSpace, and is maintained during the loop.  ⦾lexer initially.
	    Set •lexer if •CurTok (except for first token, or if it follows a ##).
	    This may be carried over from previous iteration.  
	Cases of CurTok:
- `__VA_OPT__`.  Skips over following '(' and alters behavior of following tokens
	    until the closing ')'.  Builds a macro from the tokens.
	    If was after #, makes a string token, otherwise expands the macro.
	    •lexer is retained.  This is merged with existing •lexer from preceding #.
- #.  Look at next token.  This is parameter name or `__VA_OPT__`.  
	    If `__VA_OPT__`,
		Make note in the VCtx about the stringify.  This will occur after the end
		of the `__VA_OPT__(...)` is reached.
		•lexer is retained.  
	    If a parameter name,
		Create string_literal token and append to ResultToks.
		Skip the parameter name token.  It doesn't add to •lexer.
		Move •lexer to •result.
- others... NonEmptyPasteBefore = true if ResultToks ends with a π.

- Not an arg name.  Append to ResultToks.
	    Set •result if •lexer and set ◦lexer.
	    Otherwise set ◦result if follows π, except if previous ResultToks ends with π.
- An arg name.  
    - Not adjacent to π.  Line 451.  
		Append the arg tokens (expanded) to ResultToks.  
		Change any π to ## so that it won't be a paste operator later.
		If any result tokens, move •lexer to • result[0].  startOfLine = false.
		If no result tokens, •lexer remains.  Special handling in VaOpt.
    - Adjacent to π.  Line 511.  
		Get the arg tokens (unexpanded).
    	- If any tokens,
		    Append to ResultToks.
		    Change any π to ## so that it won't be a paste operator later.
		    Move •lexer to • result[0].  startOfLine = false.  
        - If no tokens,
		    if followed by π, then skip this token and the π.
		    otherwise we have π φ.  
		ResultToks ends with π if it did not follow φ.  In this case, remove from ResultToks.
		    •lexer remains.

    Note, π tokens are not handled here.  They are part of preceding and/or following token.
	    A pair π π changes the second π to ##.  
    **In summary**, if a token is replaced by something (possibly empty), its leadingSpace is added to the
	first replacement, if any.  If a token is replaced by empty, its leadingSpace is added to 
	the next token.
## Token lexing and pasting.  For all types of macros.
After any call to ExpandFunctionArguments(), **TokenLexer.Lex()** is called repeatedly to get the next token.  It gets the next token `Tok` from `Tokens`.  It may be a π operator or an ordinary token.  
Lex maintains instance variables AtStartOfLine, HasLeadingSpace and NextTokGetsSpace (a.k.a. •lexer).
If Tok is followed by π, calls **pasteTokens(Tok)** to replace Tok with the result of pasting it
	    with following tokens and π operators.  May report a failed paste.
	    If paste succeeds, new `Tok` is the result.
	    If paste fails, processing continues.  
	For the first token, it gets its startOfLine and leadingSpace properties from AtStartOfLine and
	    HasLeadingSpace.  Otherwise the token properties are added from them.
	    The instance properties are set to false.  
### pasteTokens(LHS)
Replaces LHS with a paste of LHS (π Tok)*.  If any step fails LHS = the result up
	that point and returns false.  All used tokens are skipped.
	Be sure to run clang with -fno-ms-extensions and -fno-ms-compatibility.  Otherwise,
	    If LHS follows a π, that was a failed paste, and so LHS.leadingSpace is set = false.
	Loop for next two tokens = π and RHS.
	    LHS π RHS is attempted.  New token has spellings of LHS + RHS.  Use a temporary Lexer to
	    get the token.
	    If paste succeeds, continue loop.
		Set result AtStartOfLine and HasLeadingSpace from LHS.
		Set LHS = pasted token.
	    If paste fails, exit loop.  Change a π result to ## type unknown
	    (same spelling).
## Output
Output is printed by PrintPPOutputPPCallbacks class in lib/Frontend/PrintPreprocessedOutput.cpp.

OS = the output stream.  Look for OS << ... .
SM = the current source file.  Files are distinguished by an int FileID.  Main file is ID 1.  Basic/SourceManager.cpp.
Identifiers printed using II = Tok.getIdentifierInfo(); II->getName().
### HandleWhitespaceBeforeTok()
Decides about printing whitespace before the token.

IsStartOfLine comes from Tok.isAtStartOfLine, and might be carried over from previous iteration.
    If IsStartOfLine, calls MoveToLine.  
- Finds presumed line number for Tok.  If Tok came from a macro expansion, this will be the
	    location of the macro call.  Otherwise it is the token itself in the source.
- If the line number changes, it writes up to 8 newlines or else a #line directive.
	    Also writes leading spaces to the column number.  (minimizing whitespace defeats this).
	    Column number comes from calling
	    SM.getExpansionColumnNumber(Tok.getLocation()).  
	    Changes column 1 to column 2 if
		Token is a hash # (from an expansion). Actually, it just prints an extra ' ' character without changing column number.
		or if Tok.hasLeadingSpace().  This can happen if a macro expansion in column 1 starts
		    with an empty macro argument, or an empty nested macro expansion. 
- If not on a new line, writes a space character if
	Tok.hasLeadingSpace()
	or required to separate from earlier tokens, according to
	**AvoidConcat()**.

### AvoidConcat()
This function looks at a token, and up to two previously written tokens.
These are called `PrevPrevTok`, `PrevTok`, and `Tok`.
Returns `true` if a space is required before `Tok`.
- See if `PrevTok` and `Tok` were directly adjacent (that is, no whitespace
  between) in the same source file.  If so, then  return `false`.
- Classify `Tok` based on its type, in variable `ConcatInfo`.  This has a few
  flag bits defined.
- If `ConcatInfo` has no bits, then return `false`.
- If `ConcatInfo` has 'aci__avoid_equal' bit, and `Tok` is either **=** or
  **==**, then return `true`.  Tokens with this bit are
**&**, **+**, **-**, **/**, **<**, **>**, **|**, **%**, **`*`**, **!**, **<<**, **>>**, **^**, and **=**.
- Switch on the type of `PrevTok`:
  - quoted (string literal or char constant), with or without a size prefix.

    If not C++11, return `false`.  
    If `Tok` is an identifier, return `true`.  
    If `PrevTok` has no UD-suffix, return `false`.  
    Otherwise (`PrevTok` has UD-suffix), fall through to the next case.  Treat
    the suffix as an identifier.
  - Identifier.
    - If `Tok` is a numeric constant (including any pp-number that is neither
      integer nor float), return `true` if `Tok`  begins with **.**, `false`
      otherwise.
    - If `Tok` is an identifier, or a quoted token with a size prefix, then
      return `true`.  However, `clang` doesn't recognize side prefixes
      correctly.  It recognizes *only*:
      - 'L' in all languages.  *Nothing else in C*.
      - 'u', 'u8', 'U', 'LR', 'uR', 'u8R', 'UR', in C++11.
    - If `Tok` is a quoted token without a size prefix, and `Prev` is a size
      prefix, then return `true`.
    - Else return `false`.
  - Numeric constant.  Return `true` if `Tok` starts with [A-Za-z0-9_.+-].
  - Various combinations of `PrevTok` and `Tok`[0]:  
    `..` (if `PrevPrevTok` is .), .[0-9], .\*(C\++ only), 
    &&, ++, --, -> (includes `->*`), //, `/*`, <<, <<=, <:, <%, >>, ||, %>, %:, ::
    (C\++ only), :>, \##, \#@ (Microsoft only), %%, ->* (C\++ only), <=> (C\++20 only).
## Diagnostic messages
Messages are indicated by a DiagID.
See clang/lib/Basic/DiagnosticIDs.cpp and clang/include/clang/Basic/DiagnosticIDs.h.
IDs are divided into several categories.
DiagnosticIDs::getDescription(DiagID)
    Find the category of the ID and the offset within the category.

Message texts are kept by category in build/tools/clang/include/clang/Basic/Diagnostic*Kinds.inc

Example: err_ucn_escape_incomplete
    DiagID = 1187
    Category = LEX, offset = 166
    DiagnosticLexKinds.inc line 172
    Index in all messages = 846
    Message string offset = 50221
## Command line arguments
Args are collected into a vector<const char *>.  This is passed around to various other functions.
A CompilerInvocation holds everything about the invocation of clang.
LangOpts holds lots of options relating to the language being compiled.  See clang/Basic/LangOptions.h and LangStandard.h.
LangStandard identifies which language standard is being followed.  Its Flags member has bit fields for certain characteristics.  
    LangStandards.def maps the standard name (such as "c++23") to several Flags bits (such as C99, C11, C17, and c23).
LangOptions::setLangDefaults() sets some of the fields in a LangOptions based
on the LangStandard.  
Clang`::`ConstructJob makes an argument list for the preprocessor job.  It
uses the command line args and the target triple.  
-fms-extensions and -fms-compatibility depend on the target by default, but command line can override these.
    For Windows MSVC, the default for extensions is true.  The valud for extensions is the defalt for compatibility.
    Thus, to make things independent of the target, the command line needs -fnoms-extensions.  This will make both values false.
    These correspond to LangOpts fields MicrosoftExt and MSVCCompat.
## GCC implementation of token spacing.
Spacing between tokens on output uses two mechanisms.
1.  PREV_WHITE flag on the rhs token.  This is set by the lexer when:
    - Following one or more whitespace chars.  ' ' '\t' '\f' '\v' '\0'.
    - A comment, unless saving comments to output as normal tokens.
    - Copied on from stringize # operator to the following arg name token.
    - Cleared on first expansion token of a macro.
2.  CPP_PADDING tokens between the lhs and rhs tokens.  Possibly more than one together.
    Token has a token.source field, which can be NULL.  This points to source token whose
	spacing properties are used.
    Any padding tokens between lhs and rhs cause them to be checked to avoid accidental paste.
	cpp_avoid_paste () is called to indicate a required space.
Any of these three indicators result in a space between lhs and rhs.

padding_token(source) returns a temporary CPP_PADDING token with token.source set.  In libcpp/macro.cc.
This is called:
- For a __VA_OPT__ expression, source is the __VA_OPT__ token.  But not if after a ##.
    May be stringified into a single string token.
- Likewise for an argument name.
- Following a ##, source is the ## token.
- Before entering a macro, source is the macro token.

The file.avoid_paste member is a CPP_PADDING token with NULL source.  This is a singleton.
It is used:
- After __VA_OPT__ (usually).  May be stringified.
- After argument, likewise.
- After macro expansion.

CPP_PADDING is also set for:
- file.directive_result on entering a directive.

Pasting tokens:
- A token.PASTE_LEFT flag means that it is the LHS of a paste.
    This is set when processing the macro definition, when the next token is a ##.
    copy_paste_flag copies this from another token.
      -	With # __VA_OPT__ (...), makes a new token for the string.  If the closing ')' has
	the flag, this is copied to the new string token.
- When a token is followed by a ##, its PASTE_LEFT is set and the ## is not part of the replacement.
    This token might be a padding token, which has value of "".
- With two consecutive ## tokens, the second one is ignored.
- When PASTE_LEFT is found on a token, this is LHS, and calls paste_all_tokens(LHS).
    This pastes LHS with any more tokens with PASTE_LEFT and final RHS without PASTE_LEFT.
    None of these tokens after LHS can be CPP_PADDING with PASTE_LEFT.
    If one of the pastes fails, LHS = result of the successful pastes (if any) and the
	following token is left in the replacement stream.
    Copy PREV_WHITE from old to new LHS, and clear PASTE_LEFT.
    Final result is a CPP_PADDING token with ref = LHS.

token_streamer::stream(), in c-ppoutput.cc, writes the next token and decides whether
to precede it with a space character or an indent to a new line.
It maintains a variable (called S here, stored in static print.source).  S is updated as follows:
- Initially, S = NONE.
- When stream(T) is called:
    - If T is a CPP_PADDING token:
	- if S = NULL, S = T.source
	- Else if not S.PREV_WHITE and T.source = NULL, S = NULL
	- Else no change to S.

- When stream(T) is called:
    - If T is a padding token, just update S and return.

    - Apparently, a T on a new line has T.PREV_WHITE.
    - Else if T preceded by any padding:
	- If S = NONE, set S = T.
	- If T on new line, write to new line and column, write space.
	- Else if S.PREV_WHITE, write space.
	- Else if avoid_paste(prev T, T), write space.

    - Else if T not preceded by any padding and T.PREV_WHITE:
	- If T on new line, write to new line and column.
	- Write space.
    - Reset S = NONE.
    - Record T as previous token.

# foo
