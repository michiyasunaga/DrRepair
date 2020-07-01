#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate JSON dataset for errmodel directly from yay.tsv and yay.summary

Prepare examples with
   - multiple error lines (1-5)
   - (no error lines.) instead, during training, predict code for err lines as well as randomly chosen other lines (which are already correct)
"""

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
from compilation import err, pass_test, compile_and_run_tests_all


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



def prepare_code_with_substitution(inp_stmt, pred_stmt, sub_lines):
    # sub_lines: {idx: codeline_str, ...} to be used for substitution
    code_header = "#include <bits/stdc++.h>\n#include <string>\nusing namespace std;\n\n" #CHANGE
    curr_j = 0
    curr_ind, prev_line = 0, " "
    code = code_header
    # generate code with everything else gold except the i-th line
    for inp_j, pred_j in zip(inp_stmt, pred_stmt):
        # find the number of tabs to insert
        tmp_ind = int(inp_j[_inp.indent])
        curr_line = remove_braces_gold(inp_j[_inp.code]).strip()
        _prev_line = prev_line.replace(" ", "")
        # handle case like
        #   cout << " YES "
        #     << " \n " ;
        if (len(curr_line) >= 2 and curr_line[:2]=="<<"):
            tmp_ind = curr_ind
        # handle "std:", "pause:", "momo:", "start:", "label:", etc.
        if (2<= len(curr_line) <=12 and re.match(r'^\w+:;?$', curr_line) is not None): #^ means start, $ means end
            tmp_ind = tmp_ind + 1
        # handle
        #   10,
        #     11
        if _prev_line.endswith(",") and curr_line != "};":
            tmp_ind = curr_ind
        indent = '\t' * tmp_ind
        # if tabs are decreasing then add } if not closing already
        if tmp_ind < curr_ind:
            if not (inp_j[_inp.code].replace(" ", "") in ["}", "};"]): ##CHANGE
                indent += "} "
            if curr_ind - tmp_ind > 1:
                indent += (curr_ind - tmp_ind - 1) * "} "
        # if tabs are increasing then add { if not open already
        elif tmp_ind > curr_ind:
            if not prev_line or prev_line[-1] != "{":
                indent += "{ "
            if tmp_ind - curr_ind > 1:
                indent += (tmp_ind - curr_ind - 1) * "{ "
        curr_ind = tmp_ind
        # pick the line of code

        ## handle a case like
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

        if pred_j[_pred.text] == 'DUMMY' or curr_j not in sub_lines:
            code += indent + curr_line + "\n"
            prev_line = curr_line
        else:
            code += indent + fix_strings(sub_lines[curr_j]) + "\n"
            prev_line = sub_lines[curr_j].strip()
        curr_j += 1
    return code


def detailed_oracle_with_test_custom(inp_stmt, pred_stmt, probid, subid):
    unique_id = probid + "-" + subid
    _return_ = [] #return this
    curr_i, prob_list_i = 0, 0
    code = prepare_code_with_substitution(inp_stmt, pred_stmt, {}) #gold code
    passed, error, error_message = compile_and_run_tests_all(ARGS, code, probid, subid, None)
    if error != err.no_err: #gold program has error
        print ("gold program has error. bye")
        print (error_message)
        return None
    else:
        print ("gold program passed!")
        # return [] ## Temporary

    for curr_i, (inp_i, pred_i) in enumerate(zip(inp_stmt, pred_stmt)):
        if pred_i[_pred.text] == 'DUMMY':
            continue
        # iterate over the i-th line predictions
        for rank in range(2): #range(ARGS.num_preds):
            sub_lines = {curr_i: pred_i[_pred.pred_best + rank]}
            code = prepare_code_with_substitution(inp_stmt, pred_stmt, sub_lines)
            passed, error, error_message = compile_and_run_tests_all(ARGS, code, probid, subid, None)
            if passed == pass_test.none and error == err.compile_err:
                error_message = filter_error_message(error_message, unique_id)
            _obj_ = {
                "rank": rank+1,
                "wrong_lines_idx": [curr_i],
                "wrong_lines_code": [pred_i[_pred.pred_best + rank]],
                "passed": passed,
                "error": error,
                "error_message": error_message,
            }
            _return_.append(_obj_)
        prob_list_i += 1
    return _return_

def findsubsets(s, n): #e.g. s = {1, 2, 3}, n = 2
    return list(itertools.combinations(s, n))

def filter_and_expand_to_multi_errs(detailed_oracle_out, inp_stmt, pred_stmt, probid, subid):
    unique_id = probid + "-" + subid
    #first filer to just get err lines
    filtered_rank1 = []
    filtered_rank2 = []
    for oracle in detailed_oracle_out:
        if oracle["error"] in [1,2,3]: #error indicator
            if oracle["rank"] == 1:
                filtered_rank1.append(oracle)
            elif oracle["rank"] == 2:
                filtered_rank2.append(oracle)

    for_prep_multi = filtered_rank1[:]
    idxs_from_rank2 = list(range(len(filtered_rank2)))
    random.shuffle(idxs_from_rank2)
    while len(for_prep_multi) < 6 and len(idxs_from_rank2) > 0:
        _idx = idxs_from_rank2.pop(0)
        for_prep_multi.append(filtered_rank2[_idx])


    ## Get tuples first, then trim, then get feedback
    num_wrong = len(for_prep_multi) # we are only considering top1 pred
    _return_ = []
    for i in range(num_wrong):
        for j in range(num_wrong):
            if i < j:
                wrong_idx_i  = for_prep_multi[i]["wrong_lines_idx"][0]
                wrong_code_i = for_prep_multi[i]["wrong_lines_code"][0]
                wrong_idx_j  = for_prep_multi[j]["wrong_lines_idx"][0]
                wrong_code_j = for_prep_multi[j]["wrong_lines_code"][0]
                if wrong_idx_i == wrong_idx_j:
                    continue
                sub_lines = {wrong_idx_i: wrong_code_i, wrong_idx_j: wrong_code_j}
                code = prepare_code_with_substitution(inp_stmt, pred_stmt, sub_lines)
                passed, error, error_message = compile_and_run_tests_all(ARGS, code, probid, subid, None)
                if passed == pass_test.none and error == err.compile_err:
                    error_message = filter_error_message(error_message, unique_id)
                _obj_ = {
                    "rank": None,
                    "wrong_lines_idx": [wrong_idx_i, wrong_idx_j],
                    "wrong_lines_code": [wrong_code_i, wrong_code_j],
                    "passed": passed,
                    "error": error,
                    "error_message": error_message,
                }
                _return_.append(_obj_)
                for k in range(num_wrong):
                    if j < k:
                        wrong_idx_k  = for_prep_multi[k]["wrong_lines_idx"][0]
                        wrong_code_k = for_prep_multi[k]["wrong_lines_code"][0]
                        if (wrong_idx_i - wrong_idx_j) * (wrong_idx_j - wrong_idx_k) * (wrong_idx_k - wrong_idx_i) == 0:
                            continue
                        sub_lines = {wrong_idx_i: wrong_code_i, wrong_idx_j: wrong_code_j, wrong_idx_k: wrong_code_k}
                        code = prepare_code_with_substitution(inp_stmt, pred_stmt, sub_lines)
                        passed, error, error_message = compile_and_run_tests_all(ARGS, code, probid, subid, None)
                        if passed == pass_test.none and error == err.compile_err:
                            error_message = filter_error_message(error_message, unique_id)
                        _obj_ = {
                            "rank": None,
                            "wrong_lines_idx": [wrong_idx_i, wrong_idx_j, wrong_idx_k],
                            "wrong_lines_code": [wrong_code_i, wrong_code_j, wrong_code_k],
                            "passed": passed,
                            "error": error,
                            "error_message": error_message,
                        }
                        _return_.append(_obj_)
                        for l in range(num_wrong):
                            if k < l:
                                wrong_idx_l  = for_prep_multi[l]["wrong_lines_idx"][0]
                                wrong_code_l = for_prep_multi[l]["wrong_lines_code"][0]
                                if (wrong_idx_i - wrong_idx_j) * (wrong_idx_j - wrong_idx_k) * (wrong_idx_k - wrong_idx_l) * (wrong_idx_l - wrong_idx_i) * (wrong_idx_i - wrong_idx_k) * (wrong_idx_j - wrong_idx_l) == 0:
                                    continue
                                sub_lines = {wrong_idx_i: wrong_code_i, wrong_idx_j: wrong_code_j, wrong_idx_k: wrong_code_k, wrong_idx_l: wrong_code_l}
                                code = prepare_code_with_substitution(inp_stmt, pred_stmt, sub_lines)
                                passed, error, error_message = compile_and_run_tests_all(ARGS, code, probid, subid, None)
                                if passed == pass_test.none and error == err.compile_err:
                                    error_message = filter_error_message(error_message, unique_id)
                                _obj_ = {
                                    "rank": None,
                                    "wrong_lines_idx": [wrong_idx_i, wrong_idx_j, wrong_idx_k, wrong_idx_l],
                                    "wrong_lines_code": [wrong_code_i, wrong_code_j, wrong_code_k, wrong_code_l],
                                    "passed": passed,
                                    "error": error,
                                    "error_message": error_message,
                                }
                                _return_.append(_obj_)
    return (filtered_rank1 + filtered_rank2 + _return_)


def get_err_data_one_json(probno): #for one json file
    folder = ARGS.folder
    count = 0
    inp_stmt, pred_stmt = [], []
    lines = [] #for dump to json
    # the following look extracts the input/pred lines for the probno specified
    # and passes it further for stitching
    with open(folder + '.tsv','r') as tsvin, open(folder + '.summary','r') as predin:
        head_t = tsvin.readline().rstrip('\n').split('\t')
        head_s = predin.readline().rstrip('\n').split('\t')
        head_s.pop()
        for _ in range(ARGS.num_preds):
            head_s.append('pred_{}'.format(_ + 1))
        for _ in range(ARGS.num_preds):
            head_s.append('score_{}'.format(_ + 1))

        probid, subid, hitid, workerid = None, None, None, None
        while True:
            inp = tsvin.readline()
            if not inp:
                # Special handling for last line
                assert count == probno, \
                    'num problems = {} but probno = {}'.format(count, probno)
                break
            inp = inp.split('\t')
            pred = predin.readline().rstrip('\n').split("\t")
            s = dict(zip(head_s, pred))
            if int(inp[_inp.line].strip()) == 0:
                if count == probno:
                    break
                count += 1
                probid, subid = inp[_inp.probid].strip(), inp[_inp.subid].strip()
                hitid = inp[_inp.hitid].strip()
                workerid = inp[_inp.workerid].strip()
            if count == probno:
                inp_stmt.append(inp)
                pred_stmt.append(pred)
                line = {
                    'line': len(lines),
                    'text': s['text'],
                    'code': s['gold'],
                    'indent': int(inp[_inp.indent]),
                }
                lines.append(line)

    # generate a unique id for this program
    unique_id = "{:04d}-{}-{}".format(probno, probid, subid)
    unique_id_dir = os.path.join("/".join(folder.split("/")[:-1]), unique_id)
    cwd = os.getcwd()

    os.system("mkdir -p %s" %(unique_id_dir))
    os.chdir(unique_id_dir) #change dir to run detailed-oracle

    detailed_oracle_out = detailed_oracle_with_test_custom(inp_stmt, pred_stmt, probid, subid)
    if detailed_oracle_out is None: #gold program failed
        detailed_oracle_out = []
    #     #### Temporary ####
    #     os.chdir(cwd)
    #     with open(ARGS.out_prefix_compiler + '/{}.txt'.format(unique_id), 'w') as fout: pass
    # else:
    #     os.chdir(cwd)
    #     with open(ARGS.out_prefix_testcase + '/{}.txt'.format(unique_id), 'w') as fout: pass
    #     ##################

    if detailed_oracle_out == []: #gold program failed
        return

    expanded_detailed_oracle_out = filter_and_expand_to_multi_errs(detailed_oracle_out, inp_stmt, pred_stmt, probid, subid)


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
            })
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
    parser.add_argument('--prog-dir', default='../raw_data/spoc_data/spoc/testcases',
            help='Path the codeforces-data repository, which contains test cases')
    parser.add_argument('--max-heap', type=int, default=999999,
            help='Suicide when heap is bigger than this')
    parser.add_argument('-t', '--timeout', type=int, default=2,
            help='Timeout for execution (in seconds)')
    parser.add_argument('-T', '--gcc-timeout', type=int, default=60,
            help='Timeout for compilation (in seconds)')
    parser.add_argument('-c', '--compile-budget', type=int, default=999999,
            help='Number of maximum g++ calls')
    parser.add_argument('--num-preds', type=int, default=30)
    parser.add_argument('folder')
    parser.add_argument('probno', type=int)
    parser.add_argument('out_prefix_compiler',
            help='prefix for the output JSON files')
    args = parser.parse_args()

    global ARGS
    ARGS = parser.parse_args()

    probno = ARGS.probno
    get_err_data_one_json(probno)
    return




if __name__ == '__main__':
    main()
