#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, os, sys, math, time, random, re
import argparse
import heapq
import subprocess
from enum import Enum
import itertools
import traceback
import json
import numpy as np

## for parallel
from joblib import Parallel, delayed
import multiprocessing as mp

import socket


# Global arguments
ARGS = None


SCORE_THRES = 1e6
ERR_BLACK_AMOUNT = -1e6


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


repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(repo_root)
sys.path.append(os.path.join(repo_root, "utils"))
from code_process import tokenize_code, TEXT_TOKENIZER
from code_process import filter_error_message, fix_strings, fix_strings_one_whitespace, remove_braces_gold
from code_process import anonymize_code_str, deanonymize_code_str
from compilation import err, pass_test, compile_check



################################################
# Utils

def prepare_lines_print(text_indent, text_str_noindt, _max_len, wrap_indent=3): #initial text_indent, text_str_noindt
    text_str = text_indent*"  " + text_str_noindt
    text_to_print = []
    if len(text_str) <= _max_len:
        text_to_print.append(text_str)
    else:
        text_str_print = text_str[:_max_len]
        text_to_print.append(text_str_print)
        text_str_noindt = text_str[_max_len:]
        text_indent += wrap_indent
        text_str = text_indent*"  " + text_str_noindt
        while len(text_str) > _max_len:
            text_str_print = text_str[:_max_len]
            text_to_print.append(text_str_print)
            text_str_noindt = text_str[_max_len:]
            text_str = text_indent*"  " + text_str_noindt
        text_to_print.append(text_str)
    return text_to_print


def prepare_code_just_as_it_is(pred_stmt, code_lines_str):
    assert len(pred_stmt) == len(code_lines_str)
    ret_lines = []
    code_lines = []
    anonymize_dicts = []
    for j, line in enumerate(code_lines_str):
        line = fix_strings(line, only_char=True)
        curr_line_for_repair_model, anonymize_dict = anonymize_code_str(line)
        ret_lines.append(line)
        code_lines.append((pred_stmt[j]["text"], curr_line_for_repair_model, pred_stmt[j]["indent"]))
        anonymize_dicts.append(anonymize_dict)
    ret_code = "\n".join(ret_lines)
    return ret_code, code_lines, anonymize_dicts



################################################
# Run repair model

def stitch_error_localize_edit_multi(inp_stmt, pred_stmt, probid, subid):
    # stmt_idx: index of inp_stmt and pred_stmt (i.e., with DUMMY lines)
    #     Note: stmt_idx = real line number

    pred_stmt = pred_stmt[:] #we might modify this with editor preds
    curr_code_lines_str = [] #list[str]: with DUMMY lines
    unique_id = probid + "-" + subid
    for stmt_idx, (inp, pred) in enumerate(zip(inp_stmt, pred_stmt)):
        if pred["text"] != 'DUMMY':
            curr_code_lines_str.append(pred["code"]) #_code_str_tokenized
        else:
            curr_code_lines_str.append(pred["code"]) #DUMMY #_code

    #load repair model
    from repair_utils import RepairPolicy
    repair_model = RepairPolicy(ARGS, for_deepfix=True)


    iter_count = 0

    # print (curr_code_lines_str)
    budget = 5
    curr_num_of_compiler_errs = 0
    with open("error_localize_edit.txt", "w") as stat_file:
        while iter_count < budget:
            stat_file.flush()
            iter_count += 1
            stat_file.write("Iteration # " + str(iter_count) + "\n")
            stat_file.write("Time: {:.3f}\n".format(time.time() - START_TIME))

            code, code_lines, anonymize_dicts = prepare_code_just_as_it_is(pred_stmt, curr_code_lines_str)

            #Print current code
            print ("Current code:", file=stat_file)
            wrong_lines = [] #list[(lineno, gold_code)]
            raw_code_lines = code.split("\n")
            for (lineno, code_line) in enumerate(code_lines):
                code_str = code_line[1]
                _code_str = code_str.replace(" ","")
                text_str = raw_code_lines[lineno] #code_line[0]
                indent = code_line[2]

                gold_to_print = []
                if gold_to_print == []: gold_to_print.append("")
                text_to_print = prepare_lines_print(indent, text_str, 50)
                code_to_print = prepare_lines_print(indent, code_str, 50)
                _text = text_to_print.pop(0)
                _code = code_to_print.pop(0)
                _gold = gold_to_print.pop(0)
                print ("{:>3}  {:<{width1}}  {:<{width2}}  {:}".format(str(lineno), _text, _code, _gold, width1=50, width2=50), file=stat_file)
                for _text, _code, _gold in itertools.zip_longest(text_to_print, code_to_print, gold_to_print):
                    if _text is None: _text = ""
                    if _code is None: _code = ""
                    if _gold is None: _gold = ""
                    print ("{:>3}  {:<{width1}}  {:<{width2}}  {:}".format("", _text, _code, _gold, width1=50, width2=50), file=stat_file)


            passed, error, error_message = compile_check(ARGS, code, probid, subid, None)


            if error == err.compile_err:
                curr_num_of_compiler_errs = len(error_message.split("\n"))
                try:
                    raw_compiler_err_msg = filter_error_message(error_message, unique_id)
                    pred_lineno, _, err_line_obj = repair_model.policy_localize(code_lines, feedback=raw_compiler_err_msg, threshold_ON=False)
                    assert pred_lineno is not None
                    _, edit_pred, _ = repair_model.policy_edit(code_lines, feedback=raw_compiler_err_msg, pred_lineno=pred_lineno, beam_size=50)
                    assert edit_pred is not None
                    pred_code_nbest = edit_pred["pred"]
                except Exception as e:
                    # Commit suicide
                    print('PANIC! {}'.format(e))
                    print('PANIC! {}'.format(e), file=stat_file)
                    stat_file.write(traceback.format_exc())
                    continue

                ##try to update the code
                print ("compiler err msg  : %s" %(err_line_obj["msg"]), file=stat_file)
                print ("compiler err line#: %d" %(err_line_obj["lineno"]), file=stat_file)
                print ("pred err line#: %d" %(pred_lineno), file=stat_file)

                accept = False
                for pred_code in pred_code_nbest:
                    pred_code_deano = deanonymize_code_str(pred_code, anonymize_dicts[pred_lineno])
                    if pred_code_deano is None:
                        continue
                    print ("pred_code_candidate:", pred_code_deano, file=stat_file)
                    tmp_curr_code_lines_str = curr_code_lines_str[:]
                    tmp_curr_code_lines_str[pred_lineno] = pred_code_deano
                    tmp_code, _, _ = prepare_code_just_as_it_is(pred_stmt, tmp_curr_code_lines_str)
                    _, _, tmp_error_message = compile_check(ARGS, tmp_code, probid, subid, None)
                    if tmp_error_message==None or len(tmp_error_message.split("\n")) < curr_num_of_compiler_errs:
                        accept = True
                        break

                if not accept:
                    print ("all edit candidates rejected.", file=stat_file)
                    return False, False

                print ("pred code (edit): %s" %(pred_code_deano), file=stat_file)
                curr_code_lines_str[pred_lineno] = pred_code_deano
                stat_file.write("\n\n")
                continue

            else:
                stat_file.write("compiled!\n\n")
                return True, True
    return False, False





################################################

def stitch_helper(prog_fname):

    def decide_is_directive(code_line):
        code_line = code_line.strip()
        if re.match(r'#\w+', code_line): return True #e.g. #include
        if re.sub(r"\s", "", code_line) == "usingnamespacestd;": return True
        return False

    inp_stmt, pred_stmt = [], []

    probid = prog_fname.rstrip(".c").split('/')[-3]
    subid = prog_fname.rstrip(".c").split('/')[-1]
    with open(prog_fname) as src_in:
        for raw_line in src_in:
            _code = raw_line.strip()
            _is_directive = decide_is_directive(raw_line)
            _code_str_tokenized = ' '.join(tokenize_code(_code, mod_brace=False))
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
                len(pred_stmt), #line
                _indent #indent
            ]
            pred = {
                'line': len(pred_stmt),
                'text': "DUMMY" if _is_directive else "",
                'code': _code if _is_directive else _code_str_tokenized,
                'indent': _indent,
            }
            inp_stmt.append(inp)
            pred_stmt.append(pred) #consider this as properly `tokenized` version


    # generate a unique id for this program
    unique_id = "{}-{}".format(probid, subid)
    print("Unique ID: " + unique_id)

    init_run_dir = os.getcwd()

    local_run_dir = unique_id
    os.system("rm -rf " + local_run_dir)
    os.system("mkdir -p " + local_run_dir)
    assert os.path.exists(local_run_dir)
    print ("local_run_dir", local_run_dir)
    os.chdir(local_run_dir)


    global START_TIME
    START_TIME = time.time()

    compiled, _ = stitch_error_localize_edit_multi(inp_stmt, pred_stmt, probid, subid)
    if compiled:
        tmp_f = open("compiled.txt", "w")
        tmp_f.close()

    os.chdir(init_run_dir)
    print ("cd backed to " + os.getcwd())
    print ()
    return


def stitch():
    prog_fnames = [os.path.join(ARGS.input_code_dir, fn) for fn in sorted(os.listdir(ARGS.input_code_dir))]
    use_parallel = False
    if use_parallel:
        Parallel(n_jobs=2)(delayed(stitch_helper)(prog_fname) for (probno_, prog_fname) in enumerate(prog_fnames)) #mp.cpu_count()
    else:
        for probno_, prog_fname in enumerate(prog_fnames):
            stitch_helper(prog_fname)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--max-heap', type=int, default=999999,
            help='Suicide when heap is bigger than this')
    parser.add_argument('-t', '--timeout', type=int, default=20,
            help='Timeout for execution (in seconds)')
    parser.add_argument('-T', '--gcc-timeout', type=int, default=60,
            help='Timeout for compilation (in seconds)')
    parser.add_argument('-c', '--compile-budget', type=int, default=999999,
            help='Number of maximum g++ calls')
    parser.add_argument('-p', '--num-preds', type=int, default=100,
            help='Number of predictions per line')
    parser.add_argument('--input-code-dir', type=str)


    group = parser.add_argument_group('error')
    group.add_argument('--err-localize-threshold', type=float, default=0.95,
            help='(advanced) Minimum probability for the advanced detector to trigger')

    group.add_argument('--repairer-server',
                    help='Server + Port. comma separated list')



    global ARGS
    ARGS = parser.parse_args()
    if ARGS.repairer_server is not None:
        ARGS.repairer_server = ARGS.repairer_server.split(",")


    stitch()

    print ("DONE")


if __name__ == "__main__":
    main()
