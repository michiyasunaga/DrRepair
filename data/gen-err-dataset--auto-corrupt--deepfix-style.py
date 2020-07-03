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



def prepare_code_just_as_it_is(code_lines_str):
    ret_lines = []
    for line in code_lines_str:
        line = fix_strings(line, only_char=True)
        ret_lines.append(line)
    return "\n".join(ret_lines)



























































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
    code = prepare_code_just_as_it_is(gold_code_lines_str)
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
        curr_code_lines_str, mutate_dict, action_name_list = ver_obj
        code = prepare_code_just_as_it_is(curr_code_lines_str)
        passed, error, error_message = compile_check(ARGS, code, probid, subid, None)
        if passed == pass_test.none and error == err.compile_err:
            error_message = filter_error_message(error_message, unique_id)
            wrong_lines_idx, wrong_lines_code = [], []
            for lidx in sorted(mutate_dict.keys()):
                wrong_lines_idx.append(int(lidx))
                wrong_lines_code.append(mutate_dict[lidx])
            _obj_ = {
                "rank": 1,
                "wrong_lines_idx": wrong_lines_idx,
                "wrong_lines_code": wrong_lines_code,
                "passed": passed,
                "error": error,
                "error_message": error_message,
                "action_name_list": action_name_list
            }
            _return_.append(_obj_)
    print ("#corrupts for %s: %d" % (unique_id, len(_return_)))
    # print ()
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
       repeat this ~100 times
    """
    def _tokenize_helper(code_lines_str, dummy_lidxs=[]):
        dummy_lidxs = set(dummy_lidxs)
        tokenized = []
        for lidx, line in enumerate(code_lines_str):
            assert '\n' not in line
            if lidx in dummy_lidxs:
                toks = line.split(); kinds = ["DUMMY"] * len(toks)
                tokenized.append([toks, kinds])
            else:
                line = fix_strings(line, only_char=True)
                toks, kinds = c_tokenizer.tokenize(line)
                for j, kind in enumerate(kinds):
                    if kind=="char":
                        assert toks[j][0]==toks[j][-1]=="'" and len(toks[j]) >= 3
                        toks[j] = "' %s '" % toks[j][1:-1] #we need this, because output should be tokenized in sumith's format
                tokenized.append([toks, kinds])
        return tokenized
    #1
    gold_code_lines_str = []
    dummy_lidxs = []
    for stmt_idx, (inp, pred) in enumerate(zip(inp_stmt, pred_stmt)):
        if pred[_pred.text] != 'DUMMY':
            gold_code_lines_str.append(pred[_pred.gold]) #tokenized
        else:
            gold_code_lines_str.append(inp[_inp.code]) #not tokenized
            dummy_lidxs.append(stmt_idx)
    #2
    tokenized = _tokenize_helper(gold_code_lines_str, dummy_lidxs)
    #
    var_vocab = {} #changed 20200108
    # add_str_num = (np.random.uniform() < 0.5)
    for line_obj in tokenized:
        for tok, kind in zip(*line_obj):
            if kind == "name" and len(tok) <= 8:
                var_vocab[tok] = kind
            # if add_str_num:
            #     if kind in ["string", "number", "char"]: #new 20200108
            #         var_vocab[tok] = kind
            #     if tok in ["false", "true"]:
            #         var_vocab[tok] = "bool"
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

    tokenized_save = tokenized
    while cur_n_samples_success < n_samples and cur_n_samples < 1.5 * n_samples:
        tokenized = tokenized_save[:]
        num_mutations = np.random.randint(6) + 1 #1,2,3,4,5,6
        mutate_dict = {}
        action_name_list = defaultdict(list)
        for mut_i in range(num_mutations):
            method_id = np.random.choice(5, p=[0.34, 0.15, 0.2, 0.3, 0.01])
            method_name = corrupt_methods_name[method_id]
            err_lidx = None
            attempt = 0
            while err_lidx is None and attempt < 5:
                lines_str, err_lidx, action_name = corrupt_methods_fn[method_id](tokenized, dummy_lidxs, var_vocab)
                attempt += 1
                if err_lidx is not None:
                    # key = "%d--%s" % (err_lidx, lines_str[err_lidx]) #prevent duplicates
                    mutate_dict[err_lidx] = lines_str[err_lidx]
                    action_name_list[int(err_lidx)].append(action_name)
                    tokenized = _tokenize_helper(lines_str, dummy_lidxs)

        # print ("lines_str", lines_str)
        # print ("mutate_dict", mutate_dict)

        while len(mutate_dict) > 0:
            sorted_lidxs = sorted(mutate_dict.keys())
            key_str = "--".join(["%s-%s" %(lidx, mutate_dict[lidx]) for lidx in sorted_lidxs])
            if key_str not in corrupt_vers:
                corrupt_vers[key_str] = [lines_str[:], mutate_dict.copy(), action_name_list.copy()]
                cur_n_samples_success += 1
            min_lidx = sorted_lidxs[0]
            del mutate_dict[min_lidx]
            del action_name_list[min_lidx]
            lines_str[min_lidx] = " ".join(tokenized_save[min_lidx][0])
            cur_n_samples += 1
    corrupt_vers = list(corrupt_vers.values())
    # print (len(corrupt_vers))
    return corrupt_vers


## utilities ##
VAR_TYPE_for_repl = "int, long, float, double, char".split(", ")
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
        'delete>': ("&", ""),
        'delete>': ("\*", ""),
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
        if np.random.uniform() < 0.3: #20200116
            if np.random.uniform() < 0.5:
                sampled_var = "* " + sampled_var
            else:
                sampled_var = "& " + sampled_var
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

def auto_corrupt__var_typo(tokenized, dummy_lidxs, var_vocab): ## should be common?
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
        if np.random.uniform() < 0.3: #20200116
            if np.random.uniform() < 0.5:
                sampled_var = "* " + sampled_var
            else:
                sampled_var = "& " + sampled_var
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
        if action_idx == 0:
            attemps = 0 #20200116
            while attemps < 3:
                attemps += 1
                picked_var_pos = var_pos_list[np.random.randint(len(var_pos_list))]
                if cur_line[0][picked_var_pos] != sampled_var: break
            final_toks = cur_line[0][:picked_var_pos] + [sampled_var] + cur_line[0][picked_var_pos+1:]
        else:
            picked_var_pos = var_pos_list[np.random.randint(len(var_pos_list))] #20200116
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

    def decide_is_directive(code_line):
        code_line = code_line.strip()
        if re.match(r'#\w+', code_line): return True #e.g. #include
        if re.sub(r"\s", "", code_line) == "usingnamespacestd;": return True
        return False

    from code_process import anonymize_code_str


    folder = ARGS.folder
    inp_stmt, pred_stmt = [], []
    lines = [] #for dump to json
    if prog_fname.endswith(".c"):
        probid = prog_fname.rstrip(".c").split('/')[-3]
        subid = prog_fname.rstrip(".c").split('/')[-1]
    else:
        # raise NotImplementedError
        probid, subid = prog_fname.rstrip(".cpp").split('/')[-2:]
    with open(prog_fname) as in_f:
        raw_lines = in_f.readlines()

        if prog_fname.endswith(".cpp"): #check a line with only single char (e.g. {), and move it to the end of previous line
            final_code_lines = []
            for line in raw_lines:
                line = line.rstrip()
                _line = line.strip()
                if len(_line) == 0: continue
                elif len(_line) == 1:
                    if len(final_code_lines) == 0: return
                    else: final_code_lines[-1] = final_code_lines[-1].rstrip('\\') +" "+ _line
                else:
                    final_code_lines.append(line)
            if len(final_code_lines) > 80: return   #for deepfix, max is 73, mean is 25, 95-percentile is 42
            raw_lines = final_code_lines

        # raw_lines = raw_lines[4:] #ignore header
        for raw_line in raw_lines:
            _code = raw_line.strip()
            _is_directive = decide_is_directive(raw_line)
            _code_str_tokenized = ' '.join(tokenize_code(_code, mod_brace=False))
            _code_anonymized, _anonymize_dict = anonymize_code_str(_code)
            _indent = max(len(re.match('^ *', raw_line).group(0)) // 4, len(re.match('^\t*', raw_line).group(0)))
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
                "DUMMY" if _is_directive else "", #text
                0.0, #gold_score
                0.0, #pred_score
                _code if _is_directive else _code_str_tokenized, #gold
                "", # pred_best
            ]
            line = {
                'line': len(lines),
                'text': "DUMMY" if _is_directive else "",
                'code': _code if _is_directive else _code_str_tokenized,
                'code_anonymized': _code if _is_directive else _code_anonymized,
                'anonymize_dict': {} if _is_directive else _anonymize_dict,
                'indent': _indent,
            }
            inp_stmt.append(inp)
            pred_stmt.append(pred)
            lines.append(line)

    # generate a unique id for this program
    init_run_dir = os.getcwd()
    unique_id_dir = os.path.abspath(os.path.join(folder, probid, subid))

    local_run_dir = unique_id_dir
    os.system("rm -rf " + local_run_dir)
    os.system("mkdir -p " + local_run_dir)
    assert os.path.exists(local_run_dir)
    print ("local_run_dir", local_run_dir)
    os.chdir(local_run_dir)


    detailed_oracle_out = generate_auto_corrupt_data(inp_stmt, pred_stmt, probid, subid)
    if detailed_oracle_out is None: #gold program failed
        detailed_oracle_out = []

    if detailed_oracle_out == []: #gold program failed
        os.chdir(init_run_dir) #change dir back
        print ("No data generated for", probid, subid)
        return

    expanded_detailed_oracle_out = detailed_oracle_out

    os.chdir(init_run_dir) #change dir back
    print ()
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
                print (oracle["error_message"])
                continue

            new_mod_code_anonymized = []
            new_mod_code__anonymize_dict = []
            for ml, ml_code in zip(oracle["wrong_lines_idx"], oracle["wrong_lines_code"]):
                ml_code_new_anonymized, ml_code_new_anonymize_dict = anonymize_code_str(ml_code)
                new_mod_code_anonymized.append(ml_code_new_anonymized)
                new_mod_code__anonymize_dict.append(ml_code_new_anonymize_dict)

            errors_compiler.append({
                'mod_line': oracle["wrong_lines_idx"],
                'mod_code': oracle["wrong_lines_code"],
                'mod_code_anonymized': new_mod_code_anonymized,
                'mod_code_anonymize_dict': new_mod_code__anonymize_dict,
                'err_line': error_line,
                'err_msg': error_msg, #oracle["error_message"],
                "action_name_list": oracle["action_name_list"], #ADD
            })

    os.system("mkdir -p %s" % ARGS.out_prefix_compiler)
    with open(ARGS.out_prefix_compiler + '/{}.json'.format(subid), 'w') as fout: #CHANGE to /
        json.dump({
            'meta': meta,
            'lines': lines,
            'errors': errors_compiler,
        }, fout, ensure_ascii=False, indent=2)



LINE_OFFSET = 1 #DeepFix



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
    parser.add_argument('--input-code-dir') #e.g. deepfix_data/prob11/correct
    parser.add_argument('folder') #e.g. `log--auto-corrupt--orig-deepfix/cpp-log--orig-deepfix`
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
