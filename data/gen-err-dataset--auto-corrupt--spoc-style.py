#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil, re, random
from collections import defaultdict, Counter


import csv, os, sys, math, time
import argparse
import heapq
import subprocess
from enum import Enum
import itertools
import traceback
import json
import numpy as np

sys.path.append("../utils")
from code_process import tokenize_code, TEXT_TOKENIZER
from code_process import parse_error, filter_error_message, fix_strings, fix_strings_one_whitespace, remove_braces_gold
from compilation import err, pass_test, compile_check

import socket


## for parallel
from joblib import Parallel, delayed
import multiprocessing as mp


# Global arguments
ARGS = None



# indices in the respective tsv_files
class _inp():
    text = 0
    code = 1
    hitid = 2
    workerid = 3
    probid = 4
    subid = 5
    line = 6
    indent = 7

class _pred():
    text = 1
    gold_score = 2
    pred_score = 3
    gold = 4
    pred_best = 5



def prepare_code_fix_spoc_bug(inp_stmt, pred_stmt, curr_code_lines_str):
    code = "#include <bits/stdc++.h>\n#include <string>\nusing namespace std;\n\n"
    code_lines = []         # For the error detection model
    prev_line = " "
    idx_count = 0
    curr_ind = 0
    for inp, pred in zip(inp_stmt, pred_stmt):
        # indent = '\t' * int(inp[_inp.indent]) ????????
        tmp_ind = int(inp[_inp.indent])
        if pred[_pred.text] == 'DUMMY':
            curr_line = remove_braces_gold(inp[_inp.code]).strip()
        else:
            curr_line = fix_strings(curr_code_lines_str[idx_count])
            curr_line = fix_strings_one_whitespace(inp[_inp.text], curr_line)
        _prev_line = prev_line.replace(" ", "")
        # handle case like
        #   cout << " YES "
        #     << " \n " ;
        if (len(curr_line) >= 2 and curr_line[:2]=="<<"):
            tmp_ind = curr_ind
        # handle "std:", "pause:", "momo:", "start:", "label:", etc.
        if (2<= len(curr_line) <=12 and re.match(r'^\w+:;?$', curr_line.replace(" ","")) is not None): #^ means start, $ means end. be careful - curr_line is tokenized (e.g. momo :)
            tmp_ind = tmp_ind + 1
        # handle
        #   10,
        #     11
        if _prev_line.endswith(",") and curr_line != "};":
            tmp_ind = curr_ind
        indent = '\t' * tmp_ind
        if tmp_ind < curr_ind:
            if not (pred[_pred.text] == 'DUMMY' and (inp[_inp.code].replace(" ", "") in ["}", "};"])): ##be careful - this function takes in line str from pred (if not DUMMY), so braces are removed
                indent += "} "
            if curr_ind - tmp_ind > 1:
                indent += (curr_ind - tmp_ind - 1) * "} "
        elif tmp_ind > curr_ind:
            if not prev_line or prev_line[-1] != "{":
                indent += "{ "
            if tmp_ind - curr_ind > 1:
                indent += (tmp_ind - curr_ind - 1) * "{ "
        curr_ind = tmp_ind

        ##handle a case like
        # if (i==10)
        # else { ... }
        if _prev_line.startswith("if(") and _prev_line.endswith(")") and curr_line.startswith("else"):
            code += ("\t" *curr_ind + ";\n")
        elif _prev_line.startswith("elseif(") and _prev_line.endswith(")") and curr_line.startswith("else"):
            code += ("\t" *curr_ind + ";\n")
        elif _prev_line =="else" and curr_line=="}":
            code += ("\t" *curr_ind + "{\n")
        elif _prev_line =="do" and curr_line.startswith("while"):
            code += ("\t" *curr_ind + "{}\n")

        if pred[_pred.text] == 'DUMMY':
            code += indent + curr_line + "\n"
            idx_count += 1
        else:
            code += indent + curr_line + " // " + inp[_inp.text].rstrip('\\') + "\n"
            idx_count += 1
        prev_line = curr_line
        code_lines.append((" ".join(TEXT_TOKENIZER.findall(inp[_inp.text].rstrip('\\'))), curr_line, curr_ind))
        # print (curr_line)
    return code, code_lines


def generate_auto_corrupt_data(inp_stmt, pred_stmt, probid, subid):
    """
    receive output from `auto_corrupt_one_program`
    run compiler on each of the corrupted programs
    """
    unique_id = probid + "-" + subid
    _return_ = [] #return this
    gold_code_lines_str = []
    for stmt_idx, (inp, pred) in enumerate(zip(inp_stmt, pred_stmt)):
        if pred[_pred.text] != 'DUMMY':
            gold_code_lines_str.append(pred[_pred.gold])
        else:
            gold_code_lines_str.append(inp[_inp.code])
    code, code_lines = prepare_code_fix_spoc_bug(inp_stmt, pred_stmt, gold_code_lines_str)
    passed, error, error_message = compile_check(ARGS, code, probid, subid, None)
    if error != err.no_err: #gold program has error
        print ("gold program has error. bye")
        print (error_message)
        return None
    else:
        print ("gold program passed!")
        # return [] ## Temporary
    #
    corrupt_vers = auto_corrupt_program(inp_stmt, pred_stmt, ARGS.n_samples) #list of curr_code_lines_str

    for ver_obj in corrupt_vers:
        curr_code_lines_str, err_lidx, corrupt_method, action_name = ver_obj
        code, code_lines = prepare_code_fix_spoc_bug(inp_stmt, pred_stmt, curr_code_lines_str)
        passed, error, error_message = compile_check(ARGS, code, probid, subid, None)
        if passed == pass_test.none and error == err.compile_err:
            error_message = filter_error_message(error_message, unique_id)
            _obj_ = {
                "rank": 1,
                "wrong_lines_idx": [int(err_lidx)],
                "wrong_lines_code": [curr_code_lines_str[err_lidx]],
                "passed": passed,
                "error": error,
                "error_message": error_message,
                "corrupt_method": corrupt_method,
                "action_name": action_name
            }
            _return_.append(_obj_)
    print ("#corrupts for %s: %d" % (unique_id, len(_return_)))
    return _return_






#############  auto_corrupt  #############
import regex, re
from c_tokenizer_mod import C_Tokenizer
c_tokenizer = C_Tokenizer()

DEBUG_MODE = False #True


def auto_corrupt_program(inp_stmt, pred_stmt, n_samples=100):
    """
    1. prepare gold program (lines)
    2. tokenize each line (using c_tokenizer_mod)
    3. sample a corruption method, and apply it to a sampled line.
    """
    #1
    gold_code_lines_str = []
    dummy_lidxs = []
    for stmt_idx, (inp, pred) in enumerate(zip(inp_stmt, pred_stmt)):
        if pred[_pred.text] != 'DUMMY':
            gold_code_lines_str.append(pred[_pred.gold])
        else:
            gold_code_lines_str.append(inp[_inp.code])
            dummy_lidxs.append(stmt_idx)
    #2
    tokenized = []
    for line in gold_code_lines_str:
        assert '\n' not in line
        toks, kinds = c_tokenizer.tokenize(line)
        tokenized.append([toks, kinds])
    #
    var_vocab = {}
    add_str_num = (np.random.uniform() < 0.5)
    for line_obj in tokenized:
        for tok, kind in zip(*line_obj):
            if kind == "name" and len(tok) <= 8:
                var_vocab[tok] = kind
            if add_str_num:
                if kind in ["string", "number", "char"]:
                    var_vocab[tok] = kind
                if tok in ["false", "true"]:
                    var_vocab[tok] = "bool"
    # primary expression := identifier | literal (:= number | string)
    # var_vocab = list(var_vocab.keys())
    #3
    corrupt_methods_fn = { 0: auto_corrupt__syntax,
                            1: auto_corrupt__var_type,
                            2: auto_corrupt__var_declare,
                            3: auto_corrupt__var_typo,
                            4: auto_corrupt__kw_typo }
    corrupt_methods_name = { 0: "syntax",
                            1: "var_type",
                            2: "var_declare",
                            3: "var_typo",
                            4: "keyword_typo" }
    corrupt_vers = {}
    cur_n_samples = 0
    cur_n_samples_success = 0
    while cur_n_samples_success < n_samples and cur_n_samples < 1.5 * n_samples:
        method_id = np.random.choice(5, p=[0.3, 0.15, 0.2, 0.3, 0.05])
        method_name =  corrupt_methods_name[method_id]
        err_lidx = None
        attempt = 0
        while err_lidx is None and attempt < 5:
            lines_str, err_lidx, action_name = corrupt_methods_fn[method_id](tokenized, dummy_lidxs, var_vocab)
            attempt += 1
        if err_lidx is not None:
            key = "%d--%s" % (err_lidx, lines_str[err_lidx]) #prevent duplicates
            if key not in corrupt_vers:
                corrupt_vers[key] = [lines_str, err_lidx, method_name, action_name]
                cur_n_samples_success += 1
        cur_n_samples += 1
    corrupt_vers = list(corrupt_vers.values())
    return corrupt_vers


## utilities ##
VAR_TYPE_for_repl = "int, long, float, double, char, string, bool".split(", ")
np.random.seed(458761)

def auto_corrupt__syntax(tokenized, dummy_lidxs, var_vocab):
    __action_pattern_map = {
        'delete(': ("\(", ""),
        'delete)': ("\)", ""),
        'delete,': (",", ""),
        'delete;': (";", ""),
        'delete{': ("\{", ""),
        'delete}': ("\}", ""),
        'delete[': ("\[", ""),
        'delete]': ("\]", ""),
        'delete+': ("\+", ""),
        'delete-': ("-", ""),
        'delete=': ("=", ""),
        'delete<': ("<", ""),
        'delete>': (">", ""),
        'duplicate(': ("\(", "( ("),
        'duplicate)': ("\)", ") )"),
        'duplicate,': (",", ", ,"),
        'duplicate{': ("\{", "{ {"),
        'duplicate}': ("\}", "} }"),
        'duplicate[': ("\[", "[ ["),
        'duplicate]': ("\]", "] ]"),
        'replace;with,': (";", ","),
        'replace,with;': (",", ";"),
        'replace;with.': (";", "."),
        'replace);with;)': ("\) ;", "; )"),
        'delete\"': ("\"", ""),
        'delete\'': ("\'", ""),
        'deleteall\"': ("\"", ""),  #NOTE
        'deleteall\'': ("\'", ""),  #NOTE
        'duplicate\"': ("\"", "\" \""),
        'duplicate\'': ("\'", "\' \'"),
    }
    _actions = list(__action_pattern_map.keys())
    _action = _actions[np.random.randint(len(_actions))]
    if DEBUG_MODE:
        print ("action picked:", _action)
    _patt = __action_pattern_map[_action]
    if np.random.randint(2) == 0:  #new 20200108
        if _action.endswith("<"):
            _action, _patt = 'delete<<', ("<<", "")
        elif _action.endswith(">"):
            _action, _patt = 'delete>>', (">>", "")
    #
    tokenized = tokenized[:]
    err_lidx = None
    line_idxs = list(set(np.arange(len(tokenized))) - set(dummy_lidxs))
    np.random.shuffle(line_idxs)
    for lidx in list(line_idxs):
        cur_line_str = " ".join(tokenized[lidx][0])
        positions = [m.span() for m in regex.finditer(_patt[0], cur_line_str)]
        if len(positions) == 0:
            continue
        if _action.startswith("deleteall"):
            cur_line_str = re.sub(_patt[0], _patt[1], cur_line_str) #replace all
        else:
            if len(positions) > 1: to_corrupt = np.random.randint(len(positions))
            else: to_corrupt = 0
            cur_line_str = cur_line_str[:positions[to_corrupt][0]] + _patt[1] + cur_line_str[positions[to_corrupt][1]:]
        #complete corruption
        cur_line_str = " ".join(cur_line_str.split())
        tokenized[lidx] = tokenized[lidx] + [cur_line_str]
        err_lidx = lidx
        break
    #prepare ret_lines
    ret_lines = [] #list(str)
    for line_obj in tokenized:
        if len(line_obj) == 3:
            ret_lines.append(line_obj[2])
            if DEBUG_MODE:
                print ("err_lidx %d: %s" % (err_lidx, line_obj[2]))
        else:
            ret_lines.append(" ".join(line_obj[0]))
    return ret_lines, err_lidx, _action

def _check_type_var_decl(cur_line, type_only=False): #cur_line: [toks, kinds]
    """
    e.g.  int a , b[2] = {1,2} , c ;
          string s = "YES" ;
    <type> <name> or <,> <name> means the start of variable
    """
    inside_type = False
    type_s = None
    type_e = None
    var_pos_list = [] #list of (var_s, var_e)
    line_len = len(cur_line[1])
    for _j_ in range(line_len): # variable def (with type) should appear only once in a line
        kind = cur_line[1][_j_]
        if kind == "type":
            if inside_type == False:
                inside_type = True
                type_s = _j_
        else:
            if inside_type == True: #end of type sequence
                if kind == "name": #make sure this is var def. Otherwise bad -- like "(int) n"
                    type_e = _j_
                break
    if type_e is None: #no var declaration
        return [None, None], []
    if type_only:
        return [type_s, type_e], []
    #
    var_s = None
    var_e = None
    for _j_ in range(type_e, line_len):
        kind = cur_line[1][_j_]
        tok =  cur_line[0][_j_]
        if kind == "name":
            prev_kind = cur_line[1][_j_-1]
            prev_tok  = cur_line[0][_j_-1]
            if prev_kind == "type":
                var_s = _j_
            elif prev_tok == ",":
                var_e = _j_ -1
                var_pos_list.append([var_s, var_e])
                #reset
                var_s = _j_
                var_e = None
        elif tok == ";":
            var_e = _j_
            var_pos_list.append([var_s, var_e])
            break
    # if var_e is None, something is wrong. var_pos_list is partial
    return [type_s, type_e], var_pos_list

def auto_corrupt__var_type(tokenized, dummy_lidxs, var_vocab):
    """
    - replace with wrong type
    - delete type
    - add random type ("<var> = ")
    be careful when type has >=2 tokens (e.g. signed char, long long)
    """
    actions = {0: "replace type", 1: "delete type", 2: "add type"}
    num_actions = len(actions)
    action_idx = np.random.randint(num_actions)
    if DEBUG_MODE:
        print ("action picked:", actions[action_idx])
    #
    tokenized = tokenized[:]
    err_lidx = None
    line_idxs = list(set(np.arange(len(tokenized))) - set(dummy_lidxs))
    np.random.shuffle(line_idxs)
    if DEBUG_MODE:
        print ("line_idxs", line_idxs)
    #
    if action_idx in [0, 1]:
        for lidx in list(line_idxs):
            cur_line = tokenized[lidx] #[toks, kinds]
            type_pos, _ = _check_type_var_decl(cur_line, type_only=True)
            type_s, type_e = type_pos
            if type_e is None: #fail
                continue
            #
            if action_idx == 0:
                type_repl = VAR_TYPE_for_repl[np.random.randint(len(VAR_TYPE_for_repl))]
                final_toks = cur_line[0][:type_s] + [type_repl] + cur_line[0][type_e:]
            else: #action_idx == 1:
                final_toks = cur_line[0][:type_s] + cur_line[0][type_e:]
            #complete corruption
            final_str = " ".join(final_toks)
            if len(final_str.strip()) == 0:
                continue
            tokenized[lidx] = tokenized[lidx] + [final_str]
            err_lidx = lidx
            break
    #
    elif action_idx == 2:
        for lidx in list(line_idxs):
            cur_line = tokenized[lidx] #[toks, kinds]
            assign_s = None #e.g. "<var> = "
            # assign_e = None
            line_len = len(cur_line[1])
            for _j_ in range(line_len): # variable def (with type) should appear only once in a line
                kind = cur_line[1][_j_]
                tok  = cur_line[0][_j_]
                if kind == "name":
                    if _j_ >= 1:
                        prev_kind = cur_line[1][_j_ -1]
                        if prev_kind == "type":
                            continue
                    if _j_ + 1 < line_len:
                        tok_next = cur_line[0][_j_ +1]
                        if tok_next == "=":
                            assign_s = _j_
                            # assign_e = _j_ + 2
                            break
            if assign_s is None: #fail
                continue
            #
            type_add = VAR_TYPE_for_repl[np.random.randint(len(VAR_TYPE_for_repl))]
            final_toks = cur_line[0][:assign_s] + [type_add] + cur_line[0][assign_s:]
            #complete corruption
            final_str = " ".join(final_toks)
            if len(final_str.strip()) == 0:
                continue
            tokenized[lidx] = tokenized[lidx] + [final_str]
            err_lidx = lidx
            break
    else:
        assert False
    #
    #prepare ret_lines
    ret_lines = [] #list(str)
    for line_obj in tokenized:
        if len(line_obj) == 3:
            ret_lines.append(line_obj[2])
            if DEBUG_MODE:
                print ("err_lidx %d: %s" % (err_lidx, line_obj[2]))
        else:
            ret_lines.append(" ".join(line_obj[0]))
    return ret_lines, err_lidx, actions[action_idx]

def auto_corrupt__var_declare(tokenized, dummy_lidxs, var_vocab):
    """
    repeat, insert, replace, or delete a var in a declare line
        corresponds to redeclaration err, undeclared err
        (insert could cause redeclaration err)
        (replace could cause undeclared err)
    """
    actions = {0: "insert var (in decl)", 1: "replace var (in decl)", 2: "delete var (in decl)"} #20200109
    num_actions = len(actions)
    action_idx = np.random.randint(num_actions)
    var_vocab = [key for key in var_vocab if var_vocab[key] == "name"]
    if len(var_vocab) > 0:
        sampled_var = var_vocab[np.random.randint(len(var_vocab))]
    else:
        action_idx = 2 #delete mode
    if DEBUG_MODE:
        print ("action picked:", actions[action_idx])
    #
    tokenized = tokenized[:]
    err_lidx = None
    line_idxs = list(set(np.arange(len(tokenized))) - set(dummy_lidxs))
    np.random.shuffle(line_idxs)
    if DEBUG_MODE:
        print ("line_idxs", line_idxs)
    #
    for lidx in list(line_idxs):
        cur_line = tokenized[lidx] #[toks, kinds]
        type_pos, var_pos_list = _check_type_var_decl(cur_line, type_only=False)
        type_s, type_e = type_pos
        if var_pos_list == []: #fail
            continue
        ## we found a var declaration part
        # num_decl_vars = len(var_pos_list)
        if action_idx == 0: #insert
            #append <,> <new var> to some <var>
            insert_pos = var_pos_list[np.random.randint(len(var_pos_list))][1]
            final_toks = cur_line[0][:insert_pos] + [","] + [sampled_var] + cur_line[0][insert_pos:]
        elif action_idx == 1: #replace
            repl_pos = var_pos_list[np.random.randint(len(var_pos_list))]
            repl_pos_s, repl_pos_e = repl_pos
            final_toks = cur_line[0][:repl_pos_s] + [sampled_var] + cur_line[0][repl_pos_e:]
        elif action_idx == 2: #delete
            del_pos = var_pos_list[np.random.randint(len(var_pos_list))]
            del_pos_s, del_pos_e = del_pos
            if cur_line[0][del_pos_e] == ";":
                if cur_line[0][del_pos_s-1] == ",":
                    final_toks = cur_line[0][:del_pos_s-1] + cur_line[0][del_pos_e:]
                else:
                    final_toks = cur_line[0][:del_pos_s] + cur_line[0][del_pos_e:]
            else: #should be ","
                final_toks = cur_line[0][:del_pos_s] + cur_line[0][del_pos_e+1:]
        else:
            assert False
        #complete corruption
        final_str = " ".join(final_toks)
        if len(final_str.strip()) == 0:
            continue
        tokenized[lidx] = tokenized[lidx] + [final_str]
        err_lidx = lidx
        break
    #prepare ret_lines
    ret_lines = [] #list(str)
    for line_obj in tokenized:
        if len(line_obj) == 3:
            ret_lines.append(line_obj[2])
            if DEBUG_MODE:
                print ("err_lidx %d: %s" % (err_lidx, line_obj[2]))
        else:
            ret_lines.append(" ".join(line_obj[0]))
    return ret_lines, err_lidx, actions[action_idx]

def auto_corrupt__var_typo(tokenized, dummy_lidxs, var_vocab):
    """
    find a line with var (except declaration line)
      apply replacement or deletion
    """
    actions = {0: "replace var (not decl)", 1: "delete var (not decl)"}
    num_actions = len(actions)
    action_idx = np.random.randint(num_actions)
    var_vocab = [key for key in var_vocab]
    if len(var_vocab) > 0:
        sampled_var = var_vocab[np.random.randint(len(var_vocab))]
    else:
        action_idx = 1 #delete mode
    if DEBUG_MODE:
        print ("action picked:", actions[action_idx])
    #
    tokenized = tokenized[:]
    err_lidx = None
    line_idxs = list(set(np.arange(len(tokenized))) - set(dummy_lidxs))
    np.random.shuffle(line_idxs)
    if DEBUG_MODE:
        print ("line_idxs", line_idxs)
    #
    for lidx in list(line_idxs):
        cur_line = tokenized[lidx] #[toks, kinds]
        if "type" in cur_line[1]: #skip var declaration
            continue
        var_pos_list = []
        for _j_, kind in enumerate(cur_line[1]):
            if kind == "name":
            # if kind in ["name", "string", "number"]:
                var_pos_list.append(_j_)
        if len(var_pos_list) == 0:
            continue
        attemps = 0
        while attemps < 3:
            attemps += 1
            picked_var_pos = var_pos_list[np.random.randint(len(var_pos_list))]
            if cur_line[0][picked_var_pos] != sampled_var: break
        if action_idx == 0:
            final_toks = cur_line[0][:picked_var_pos] + [sampled_var] + cur_line[0][picked_var_pos+1:]
        else:
            final_toks = cur_line[0][:picked_var_pos] + cur_line[0][picked_var_pos+1:]
        #complete corruption
        final_str = " ".join(final_toks)
        if len(final_str.strip()) == 0:
            continue
        tokenized[lidx] = tokenized[lidx] + [final_str]
        err_lidx = lidx
        break
    #prepare ret_lines
    ret_lines = [] #list(str)
    for line_obj in tokenized:
        if len(line_obj) == 3:
            ret_lines.append(line_obj[2])
            if DEBUG_MODE:
                print ("err_lidx %d: %s" % (err_lidx, line_obj[2]))
        else:
            ret_lines.append(" ".join(line_obj[0]))
    return ret_lines, err_lidx, actions[action_idx]

def auto_corrupt__kw_typo(tokenized, dummy_lidxs, var_vocab): #rare
    """
    find a line with keyword and apply replacement (0) -- while <-> for,  if <-> else if <-> else  # NO (break <-> continue) b/c no compiler err
                                          deletion (1) -- "if", "while", "for", "else if"  # NO "else" b/c would be empty line
                                         # insertion (2) -- insert "break;" or "continue;" to random line
    find a line with call and apply deletion (2) -- any call, as long as do not cause empty line
    """
    actions = {0: "replace keyword", 1: "delete keyword", 2: "delete call"}
    num_actions = len(actions)
    action_idx = np.random.randint(num_actions)
    if DEBUG_MODE:
        print ("action picked:", actions[action_idx])
    #
    tokenized = tokenized[:]
    err_lidx = None
    line_idxs = list(set(np.arange(len(tokenized))) - set(dummy_lidxs))
    np.random.shuffle(line_idxs)
    if DEBUG_MODE:
        print ("line_idxs", line_idxs)
    #
    if action_idx in [0, 1]:
        for lidx in list(line_idxs):
            cur_line = tokenized[lidx] #[toks, kinds]
            keyword_pos = None
            keyword_str = None
            for _j_, tok in enumerate(cur_line[0]):
                if tok == "else":
                    if _j_+1 < len(cur_line[0]) and cur_line[0][_j_+1] == "if":
                        keyword_pos = (_j_, _j_+2)
                        keyword_str = "else if"
                    else:
                        keyword_pos = (_j_, _j_+1)
                        keyword_str = "else"
                    break
                elif tok in ["if", "while", "for"]:
                    keyword_pos = (_j_, _j_+1)
                    keyword_str = tok
                    break
            if keyword_pos is None:
                continue
            if action_idx == 1: #delete mode
                if keyword_str == "else":
                    continue
                final_toks = cur_line[0][:keyword_pos[0]] + cur_line[0][keyword_pos[1]:]
            else: #repl mode
                if keyword_str == "while": sampled_keyword = "for"
                elif keyword_str == "for": sampled_keyword = "while"
                elif keyword_str == "if": sampled_keyword = "else if"
                elif keyword_str == "else if":
                    __keyword_vocab = ["if", "else"]
                    sampled_keyword = __keyword_vocab[np.random.randint(len(__keyword_vocab))]
                elif keyword_str == "else": sampled_keyword = "else if"
                final_toks = cur_line[0][:keyword_pos[0]] + [sampled_keyword] + cur_line[0][keyword_pos[1]:]
            #complete corruption
            final_str = " ".join(final_toks)
            if len(final_str.strip()) == 0:
                continue
            tokenized[lidx] = tokenized[lidx] + [final_str]
            err_lidx = lidx
            break
    elif action_idx == 2:
        for lidx in list(line_idxs):
            cur_line = tokenized[lidx] #[toks, kinds]
            cur_line = tokenized[lidx] #[toks, kinds]
            call_pos_list = []
            for _j_, kind in enumerate(cur_line[1]):
                if kind == "call":
                    call_pos_list.append(_j_)
            if len(call_pos_list) == 0:
                continue
            sampled_call_pos = call_pos_list[np.random.randint(len(call_pos_list))]
            final_toks = cur_line[0][:sampled_call_pos] + cur_line[0][sampled_call_pos+1:]
            #complete corruption
            final_str = " ".join(final_toks)
            if len(final_str.strip()) == 0:
                continue
            tokenized[lidx] = tokenized[lidx] + [final_str]
            err_lidx = lidx
            break
    else:
        assert False
    #prepare ret_lines
    ret_lines = [] #list(str)
    for line_obj in tokenized:
        if len(line_obj) == 3:
            ret_lines.append(line_obj[2])
            if DEBUG_MODE:
                print ("err_lidx %d: %s" % (err_lidx, line_obj[2]))
        else:
            ret_lines.append(" ".join(line_obj[0]))
    return ret_lines, err_lidx, actions[action_idx]

############ auto_corrupt END ##############



def get_err_data_one_json(probno, prog_fname): #for one json file

    def decide_is_dummy(code_line):
        code_line = code_line.strip()
        if re.search(r'main\s*?\(\s*?(void)?\s*?\)', code_line): return True #int main () {
        if code_line == "}": return True
        if code_line == "": return True
        if re.search(r'return\s*?0\s*?;', code_line): return True #return 0;
        return False

    folder = ARGS.folder
    inp_stmt, pred_stmt = [], []
    lines = [] #for dump to json
    probid, subid = prog_fname.rstrip(".cpp").split('/')[-2:]
    with open(prog_fname) as in_f:
        raw_lines = in_f.readlines()
        raw_lines = raw_lines[4:] #ignore header
        for raw_line in raw_lines:
            _code = raw_line.strip()
            _is_dummy = decide_is_dummy(raw_line)
            _code_str_tokenized = ' '.join(tokenize_code(_code))
            _indent = len(re.match('^ *', raw_line).group(0)) // 4
            hitid = "????????"
            workerid = "????????"
            inp = [
                "", #text
                _code, #code
                hitid, #hitid
                workerid, #workerid
                probid, #probid
                subid, #subid
                len(lines), #line
                _indent #indent
            ]
            pred = [
                len(lines), #line
                "DUMMY" if _is_dummy else "", #text
                0.0, #gold_score
                0.0, #pred_score
                _code if _is_dummy else _code_str_tokenized, #gold
                "", # pred_best
            ]
            line = {
                'line': len(lines),
                'text': "DUMMY" if _is_dummy else "",
                'code': _code if _is_dummy else _code_str_tokenized,
                'indent': _indent,
            }
            inp_stmt.append(inp)
            pred_stmt.append(pred)
            lines.append(line)

    # generate a unique id for this program
    unique_id = "{:04d}-{}-{}".format(probno, probid, subid)
    unique_id_dir = os.path.join(folder, probid, unique_id) #e.g. folder: `log--auto-corrupt--additional-codeforce/cpp-log`
    cwd = os.getcwd()

    os.system("mkdir -p %s" %(unique_id_dir))
    os.chdir(unique_id_dir) #change dir to run detailed-oracle

    detailed_oracle_out = generate_auto_corrupt_data(inp_stmt, pred_stmt, probid, subid)
    if detailed_oracle_out is None: #gold program failed
        detailed_oracle_out = []

    if detailed_oracle_out == []: #gold program failed
        os.chdir(cwd) #change dir back
        return

    expanded_detailed_oracle_out = detailed_oracle_out

    os.chdir(cwd) #change dir back
    # os.system("pwd")

    ## now dump to json
    meta = {
        'index': probno,
        'hitid': hitid,
        'workerid': workerid,
        'probid': probid,
        'subid': subid,
    }
    errors_compiler = []
    for oracle in expanded_detailed_oracle_out:
        if str(oracle["passed"]) + str(oracle["error"]) == "01": #compiler err
            error_line, error_msg = parse_error(oracle["error_message"], line_offset=LINE_OFFSET)
            if error_line is None:
                continue
            errors_compiler.append({
                'mod_line': oracle["wrong_lines_idx"],
                'mod_code': oracle["wrong_lines_code"],
                'err_line': error_line,
                'err_msg': error_msg,
                "corrupt_method": oracle["corrupt_method"],
                "action_name": oracle["action_name"],
            })
    os.system("mkdir -p %s" % ARGS.out_prefix_compiler)
    with open(ARGS.out_prefix_compiler + '/{}.json'.format(unique_id), 'w') as fout: #CHANGE to /
        json.dump({
            'meta': meta,
            'lines': lines,
            'errors': errors_compiler,
        }, fout, ensure_ascii=False, indent=2)



# tsv: text code hitid workerid probid subid line indent
# summary: index text gold_score pred_score gold pred_1 ... pred_30 prob_1 ... prob_30


# The actual code has 4 lines of preamble (#include<..> + using namespace std)
LINE_OFFSET = 5



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--max-heap', type=int, default=999999,
            help='Suicide when heap is bigger than this')
    parser.add_argument('-t', '--timeout', type=int, default=2,
            help='Timeout for execution (in seconds)')
    parser.add_argument('-T', '--gcc-timeout', type=int, default=60,
            help='Timeout for compilation (in seconds)')
    parser.add_argument('-c', '--compile-budget', type=int, default=999999,
            help='Number of maximum g++ calls')
    parser.add_argument('--n-samples', type=int, default=100) #number of corrupted samples to generate
    parser.add_argument('--input-code-dir') #e.g. codeforce/29A
    parser.add_argument('folder') #e.g. `log--auto-corrupt--additional-codeforce/cpp-log`
    parser.add_argument('out_prefix_compiler',
            help='prefix for the output JSON files')
    args = parser.parse_args()

    global ARGS
    ARGS = parser.parse_args()

    prog_fnames = [os.path.join(ARGS.input_code_dir, fn) for fn in sorted(os.listdir(ARGS.input_code_dir))]
    use_parallel = True
    if use_parallel:
        Parallel(n_jobs=2)(delayed(get_err_data_one_json)(probno_, prog_fname) for (probno_, prog_fname) in enumerate(prog_fnames)) #mp.cpu_count()
    else:
        for probno_, prog_fname in enumerate(prog_fnames):
            get_err_data_one_json(probno_, prog_fname)



if __name__ == '__main__':
    main()
