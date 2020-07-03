import collections, os
import regex as re
# from util.helpers import get_lines, recompose_program

Token = collections.namedtuple('Token', ['typ', 'value', 'line', 'column'])


########## from tokenizer.py ##########
from abc import abstractmethod

class UnexpectedTokenException(Exception):
    pass

class EmptyProgramException(Exception):
    '''In fn tokenizer:get_lines(), positions are empty, most probably the input program \
       is without any newline characters or has a special character such as ^A'''
    pass

class FailedTokenizationException(Exception):
    '''Failed to create line-wise id_sequence or literal_sequence or both'''
    pass

class Tokenizer:
    @abstractmethod
    def tokenize(self, code, keep_format_specifiers=False, keep_names=True, \
                 keep_literals=False):
        return NotImplemented
####################


# Use for a single line
class C_Tokenizer(Tokenizer):
    _keywords = set(['auto', 'break', 'case', 'const', 'continue', 'default',
                 'do', 'else', 'enum', 'extern', 'for', 'goto', 'if',
                 'register', 'return', 'sizeof', 'static', 'switch',
                 'typedef', 'void', 'volatile', 'while', 'EOF', 'NULL', 'endl',
                 'null', 'struct', 'union'] + \
                 [
                  'alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel',
                  'atomic_commit', 'atomic_noexcept', 'auto', 'bitand', 'bitor',
                  'break', 'case', 'catch',
                  'class', 'co_await', 'co_return', 'co_yield', 'compl', 'concept', 'const',
                  'const_cast', 'consteval', 'constexpr', 'continue', 'decltype', 'default',
                  'delete', 'do', 'dynamic_cast', 'else', 'enum', 'explicit',
                  'export', 'extern', 'false', 'for', 'friend', 'goto', 'if',
                  'import', 'inline', 'module', 'mutable', 'namespace', 'new',
                  'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
                  'private', 'protected', 'public', 'reflexpr', 'register', 'reinterpret_cast',
                  'requires', 'return', 'sizeof', 'static', 'static_assert',
                  'static_cast', 'struct', 'switch', 'synchronized', 'template', 'this',
                  'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename',
                  'union', 'using', 'virtual', 'void', 'volatile',
                  'while', 'xor', 'xor_eq',
                 ])
    _includes = set(['stdio.h', 'stdlib.h', 'string.h', 'math.h', 'malloc.h',
                 'stdbool.h', 'cstdio', 'cstdio.h', 'iostream', 'conio.h'])
    _includes.update(["<" +inc+ ">" for inc in _includes] + ["<string>", "<bits/stdc++.h>"])

    _calls = set(['printf', 'scanf', 'cin', 'cout', 'clrscr', 'getch', 'strlen',
              'gets', 'fgets', 'getchar', 'main', 'malloc', 'calloc', 'free', 'sort'] + \
               open(os.path.dirname(os.path.abspath(__file__)) + "/cpp_functions.txt",'r').read().split('\n')) #ADDED
    # print (_calls)
    _types = set(['char', 'double', 'float', 'int', 'long', 'short', 'unsigned'] + ['signed', 'char16_t', 'char32_t', 'char8_t', 'wchar_t', 'string', 'bool'])

    _ops = set('(|)|[|]|{|}|->|<<|>>|**|&&|--|++|-=|+=|*=|&=|%=|/=|==|<=|>=|!=|-|<|>|~|!|%|^|&|*|/|+|=|?|.|,|:|;|#'.split('|') + ['||','|=','|'])

    def _escape(self, string):
        return repr(string)[1:-1]

    def _tokenize_code(self, code):
        keywords = {'IF', 'THEN', 'ENDIF', 'FOR', 'NEXT', 'GOSUB', 'RETURN'}
        token_specification = [
            ('comment',
             r'\/\*(?:[^*]|\*(?!\/))*\*\/|\/\*([^*]|\*(?!\/))*\*?|\/\/[^\n]*'),
            ('directive', r'#\w+'),
            ('string', r'"(?:[^"\n]|\\")*"?'),
            ('char', r"'(?:\\?[^'\n]|\\')'"),
            ('char_continue', r"'[^']*"),
            ('number',  r'[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
            ('include',  r'(?<=\#include) *<([_A-Za-z]\w*(?:\.h))?>'),
            ('op',
             r'\(|\)|\[|\]|{|}|->|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=<>!]=|[-<>~!%^&*\/+=?|.,:;#]'),
            ('name',  r'[_A-Za-z]\w*'),
            ('whitespace',  r'\s+'),
            ('nl', r'\\\n?'),
            ('MISMATCH', r'.'),            # Any other character
        ]
        tok_regex = '|'.join('(?P<%s>%s)' %
                             pair for pair in token_specification)
        line_num = 1
        line_start = 0
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
            elif kind == 'SKIP':
                pass
            elif kind == 'MISMATCH':
                yield UnexpectedTokenException('%r unexpected on line %d' % (value, line_num))
            else:
                if kind == 'ID' and value in keywords:
                    kind = value
                column = mo.start() - line_start
                yield Token(kind, value, line_num, column)



    def tokenize(self, code, keep_format_specifiers=False, keep_names=True,
                 keep_literals=False):
        """
        return:
          aligned lists [tok, tok, ...], [type, type, ...]
          skip whitespace?
        """
        ret_toks = []
        ret_types = []

        # Get the iterable
        my_gen = self._tokenize_code(code)

        while True:
            try:
                token = next(my_gen)
            except StopIteration:
                break

            if isinstance(token, Exception):
                return ret_toks, ret_types

            type_ = str(token[0])
            value = str(token[1])
            if type_ == 'whitespace':
                continue

            if value in self._types:
                type_ = "type"
            elif value in self._calls:
                type_ = "call"
            elif value in self._keywords:
                type_ = "keyword"
            if len(ret_toks) >1 and ret_toks[-1] == '.' and type_ == 'name': # s.push_back()
                type_ = "call"

            ret_toks.append(value)
            ret_types.append(type_)

        return ret_toks, ret_types
