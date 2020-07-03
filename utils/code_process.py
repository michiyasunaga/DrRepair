import sys, os, shutil, re, argparse, json
from collections import defaultdict


################################################
# Basic tokenization

TEXT_TOKENIZER = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def tokenize_text(text):
    return TEXT_TOKENIZER.findall(text)

def tokenize_err_msg(text):
    return TEXT_TOKENIZER.findall(text)


################################################
# Clang interface

def fix_char_string_tok(tokens, mod_brace=True):
    res_tokens = []
    if mod_brace:
        if tokens and tokens[0] == "}":
            tokens = tokens[1:]
        if tokens and tokens[-1] == "{":
            tokens = tokens[:-1]
    for token in tokens:
        if token[0] == "\"" and token[-1] == "\"":
            res_tokens.append("\"")
            res_tokens.append(token[1:-1])
            res_tokens.append("\"")
        elif token[0] == "\'" and token[-1] == "\'":
            res_tokens.append("\'")
            res_tokens.append(token[1:-1])
            res_tokens.append("\'")
        else:
            res_tokens.append(token)
    return res_tokens

def tokenize_code(code, mod_brace=True):
    from clang.cindex import Index
    index = Index.create()
    tu = index.parse('tmp.cpp', args=['-std=c++11'], unsaved_files=[('tmp.cpp', code)])
    tokens = [token.spelling for token in tu.get_tokens(extent=tu.cursor.extent)]
    tokens = fix_char_string_tok(tokens, mod_brace)
    return tokens


################################################
# Process error message

def parse_error(msg, line_offset):
    """
    Return the first error line number and error message.
    """
    # lines = eval(msg).split('\n')
    lines = msg.split('\n')
    for line in lines:
        m = re.match(':(\d+):[:0-9 ]+error: (.*)', line)
        if not m:
            continue
        lineno, message = m.groups()
        return int(lineno) - line_offset, message.strip()
    return None, None

def filter_error_message(message, unique_id):
    if unique_id.startswith("prob"):
        return '\n'.join(x.replace(unique_id + '.c', '')
            for x in message.split('\n')
            if x.startswith(unique_id)
        )
    else:
        return '\n'.join(x.replace(unique_id + '.cpp', '')
            for x in message.split('\n')
            if x.startswith(unique_id)
        )


################################################
# For stitching (used when evaluating on test) - legacy from SPoC

def fix_strings(inp, only_char=False):
    if not only_char: #20200116
        res = ""
        temp_string = ""
        inside = False
        for i in range(len(inp)):
            if not inside:
                res += inp[i]
                if inp[i] == "\"":
                    inside = True
                continue
            if inside:
                if inp[i] == "\"":
                    inside = False
                    if len(temp_string) > 2 and temp_string[0] == " " and temp_string[-1] == " ":
                        res += temp_string[1:-1]
                    else:
                        res += temp_string
                    res += "\""
                    temp_string = ""
                else:
                    temp_string += inp[i]
        inp = res
    res = ""
    temp_string = ""
    inside = False
    for i in range(len(inp)):
        if not inside:
            res += inp[i]
            if inp[i] == "\'":
                inside = True
            continue
        if inside:
            if inp[i] == "\'":
                inside = False
                if len(temp_string) > 2 and temp_string[0] == " " and temp_string[-1] == " ":
                    res += temp_string[1:-1]
                else:
                    res += temp_string
                res += "\'"
                temp_string = ""
            else:
                temp_string += inp[i]
    return res


def fix_strings_one_whitespace(text, pred_code):
    if "\"\"" in pred_code or "\'\'" in pred_code: #this should not happen if we did " ".join(code tokensx)
        if "\" \"" in text or "\' \'" in text:
            pred_code = pred_code.replace("\"\"", "\" \"")
            pred_code = pred_code.replace("\'\'", "\' \'")
            return pred_code
    if "\' \'" in pred_code or "\" \"" in pred_code:
        if  "\"\"" in text or "\'\'" in text:
            pred_code = pred_code.replace("\" \"", "\"\"")
            pred_code = pred_code.replace("\' \'", "\'\'")
    return pred_code


def remove_braces_gold(gold):
    if len(gold) < 3:
        return gold
    gold = gold.strip()
    if gold[-1] == "{":
        gold = gold[:-1]
    if gold[0] == "}":
        gold = gold[1:]
    return gold




################################################
# For anonymize/deanonymize code - for DeepFix

sys.path.append('.')
from c_tokenizer_mod import C_Tokenizer
c_tokenizer = C_Tokenizer()

def anonymize_code_str(_code_str):
    _code_str = fix_strings(_code_str, only_char=True)
    toks, kinds = c_tokenizer.tokenize(_code_str)
    _code_tokenized_deepfix = []
    anonymize_dict = defaultdict(list)
    for tok, kind in zip(toks, kinds):
        if kind == "string":
            _code_tokenized_deepfix.append("_<string>_")
            anonymize_dict["string"].append(tok)
        elif kind == "number":
            _code_tokenized_deepfix.append("_<number>_")
            anonymize_dict["number"].append(tok)
        elif kind == "char":
            # assert len(tok)==3
            # tok = "' %s '" % tok[1] ###NOTE: do not do this. for anonymize, we need actual char like 'c'
            _code_tokenized_deepfix.append("_<char>_")
            anonymize_dict["char"].append(tok)
        else:
            _code_tokenized_deepfix.append(tok)
    return " ".join(_code_tokenized_deepfix), anonymize_dict


def deanonymize_code_str(_code_str, anonymize_dict):
    toks = _code_str.split()
    string_count_in_code = len([tok for tok in toks if tok=="_<string>_"])
    string_count_true = 0 if len(anonymize_dict)==0 else len(anonymize_dict["string"])
    number_count_in_code = len([tok for tok in toks if tok=="_<number>_"])
    number_count_true = 0 if len(anonymize_dict)==0 else len(anonymize_dict["number"])
    char_count_in_code = len([tok for tok in toks if tok=="_<char>_"])
    char_count_true = 0 if len(anonymize_dict)==0 else len(anonymize_dict["char"])
    if (string_count_in_code != string_count_true) or (number_count_in_code != number_count_true) or (char_count_in_code != char_count_true):
        return None

    string_count = 0
    number_count = 0
    char_count = 0
    ret_toks = []
    for tok in toks:
        if tok == "_<string>_":
            ret_toks.append(anonymize_dict["string"][string_count])
            string_count += 1
        elif tok == "_<number>_":
            ret_toks.append(anonymize_dict["number"][number_count])
            number_count += 1
        elif tok == "_<char>_":
            ret_toks.append(anonymize_dict["char"][char_count])
            char_count += 1
        else:
            ret_toks.append(tok)
    return " ".join(ret_toks)
