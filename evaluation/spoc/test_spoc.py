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
sys.path.append(os.path.join(repo_root, "utils"))
from code_process import tokenize_code, TEXT_TOKENIZER
from code_process import filter_error_message, fix_strings, fix_strings_one_whitespace, remove_braces_gold
from compilation import err, pass_test, compile_and_run_tests_all




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
            curr_line_for_repair_model = inp[_inp.code]
        else:
            curr_line_for_repair_model = curr_code_lines_str[idx_count]
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
        code_lines.append((" ".join(TEXT_TOKENIZER.findall(inp[_inp.text].rstrip('\\'))), curr_line_for_repair_model, curr_ind))
        # print (curr_line)
    return code, code_lines




################################################
# Run repair model during stitch

# add edited versions of line into heap
def stitch_error_localize_edit_search(inp_stmt, pred_stmt, probid, subid):
    # There are 2 different indexing systems (both 0-based)
    # * stmt_idx: index of inp_stmt and pred_stmt (i.e., with DUMMY lines)
    #     Note: stmt_idx = real line number minus LINE_OFFSET
    # * prob_list_idx: index of prob_list (i.e., excluding DUMMY lines)

    pred_stmt = pred_stmt[:] #we will modify this with editor preds
    curr_code_lines_str = [] #list[str]: with DUMMY lines
    gold_code_lines_str = []
    prob_list = []
    prob_list_idx_to_stmt_idx = []
    unique_id = probid + "-" + subid
    for stmt_idx, (inp, pred) in enumerate(zip(inp_stmt, pred_stmt)):
        curr_prob_list = []
        if pred[_pred.text] != 'DUMMY':
            for i in range(_pred.pred_best + ARGS.num_preds, _pred.pred_best + 2 * ARGS.num_preds):
                curr_prob_list.append(float(pred[i]))
            prob_list.append(curr_prob_list)
            prob_list_idx_to_stmt_idx.append(stmt_idx)
            curr_code_lines_str.append(pred[_pred.pred_best]) #initialize with top1 sticth
            gold_code_lines_str.append(pred[_pred.gold])
        else:
            curr_code_lines_str.append(inp[_inp.code]) #DUMMY
            gold_code_lines_str.append(inp[_inp.code])
    stmt_idx_to_prob_list_idx = {x: i for (i, x) in enumerate(prob_list_idx_to_stmt_idx)}

    iter_count, compile_count = 0, 0
    # create a heap and add the first element
    # since we want a max_heap, we add a the negative of log prob (by default it's a min heap)
    heap = FragilePrioritySet(prob_list)


    #load repair model
    from repair_utils import RepairPolicy
    repair_model = RepairPolicy(ARGS, for_deepfix=False)


    edited_lineidx = {} #idxing system = stmt list

    # blacklist[prob_list_idx] = set of candidate_idxs that are blacklisted
    blacklist = [set() for _ in range(len(prob_list))]

    # iterate until not empty
    with open("error_localize_edit.txt", "w") as stat_file:
        while not heap.empty() and iter_count < ARGS.compile_budget:
            stat_file.flush()
            iter_count += 1
            stat_file.write("\n")
            stat_file.write("Stats after iteration # " + str(iter_count) + "\n")
            stat_file.write("Time: {:.3f}\n".format(time.time() - START_TIME))
            # log_prob: float
            # curr_idx: list[int] of length len(prob_list)
            log_prob, curr_idx = heap.pop()
            stat_file.write(str(log_prob) + "\n")
            stat_file.write(str(curr_idx) + "\n")
            if log_prob >= SCORE_THRES:
                stat_file.write('Log_prob threshold reached. Committing suicide ...')
                return False, False

            # detect if there is a blacklisted candidate
            found_blacklist = None
            for prob_list_idx, candidate_idx in enumerate(curr_idx):
                if candidate_idx in blacklist[prob_list_idx]:
                    found_blacklist = (prob_list_idx, candidate_idx)
                    break

            # decide whether to proceed with code generation
            skip_synthesis = (found_blacklist is not None and ARGS.err_handling == 'black')
            if skip_synthesis:
                stat_file.write("Skip since {}.{} is in blacklist\n".format(*found_blacklist))
                iter_count -= 1
            else:
                for (linei, idx) in enumerate(curr_idx):
                    curr_code = pred_stmt[prob_list_idx_to_stmt_idx[linei]][_pred.pred_best + idx]
                    curr_code_lines_str[prob_list_idx_to_stmt_idx[linei]] = curr_code

                code, code_lines = prepare_code_fix_spoc_bug(inp_stmt, pred_stmt, curr_code_lines_str)
                assert len(gold_code_lines_str) == len(code_lines)

                print ("Current code:", file=stat_file)
                func_acc_tgts = {}
                wrong_lines = [] #list[(lineno, gold_code)]
                for (lineno, code_line) in enumerate(code_lines):
                    code_str = code_line[1]
                    _code_str = code_str.replace(" ","")
                    text_str = code_line[0]
                    indent = code_line[2]

                    gold_str = remove_braces_gold(gold_code_lines_str[lineno])
                    gold_to_print = []
                    if lineno in func_acc_tgts:
                        if (_code_str not in func_acc_tgts[lineno]):
                            gold_to_print += prepare_lines_print(0, "Gold: {}".format(gold_str), 50, wrap_indent=8)
                            wrong_lines.append(lineno)
                    else:
                        if _code_str != gold_str.replace(" ",""):
                            gold_to_print += prepare_lines_print(0, "Gold: {}".format(gold_str), 50, wrap_indent=8)
                            wrong_lines.append(lineno)
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

                # run the program
                passed, error, error_message = compile_and_run_tests_all(ARGS, code, probid, subid, None)
                if error != err.compile_err:
                    compile_count += 1
                else:
                    raw_compiler_err_msg = filter_error_message(error_message, unique_id)
                    try:
                        pred_lineno, _, err_line_obj = repair_model.policy_localize(code_lines, feedback=raw_compiler_err_msg, threshold_ON=True)
                        err_line_stmt_idx = pred_lineno
                        err_msg = err_line_obj["msg"]
                    except Exception as e:
                        # Commit suicide
                        print('PANIC (localize)! {}'.format(e))
                        print('PANIC (localize)! {}'.format(e), file=stat_file)
                        stat_file.write(traceback.format_exc())
                        err_line_stmt_idx = None
                        err_line = None
                        err_msg = None
                    # resolve the error line to prob_list_idx
                    err_line = stmt_idx_to_prob_list_idx.get(err_line_stmt_idx) if err_line_stmt_idx is not None else None

                    # after resolving, check if it's a predicted line
                    if err_line is None:
                        print('Error line UNKNOWN: {}'.format(err_msg), file=stat_file)
                    else:
                        print('Error line {} (with DUMMY): {}'.format(
                            err_line_stmt_idx, err_msg,
                        ), file=stat_file)

                        print('Blacklisting {}.{}'.format(err_line, curr_idx[err_line]), file=stat_file)
                        blacklist[err_line].add(curr_idx[err_line])

                        if ARGS.err_handling == 'gray':
                            prob_list[err_line][curr_idx[err_line]] -= ARGS.err_gray_amount
                        else:
                            prob_list[err_line][curr_idx[err_line]] = ERR_BLACK_AMOUNT

                        #Before rebuild heap, update prob_list
                        if err_line_stmt_idx not in edited_lineidx:
                            print ('Getting edit preds for line# {}'.format(err_line_stmt_idx), file=stat_file)
                            try:
                                _, edit_pred, _ = repair_model.policy_edit(code_lines, feedback=raw_compiler_err_msg, pred_lineno=err_line_stmt_idx, beam_size=50)
                            except Exception as e:
                                # Commit suicide
                                print('PANIC (edit)! {}'.format(e))
                                print('PANIC (edit)! {}'.format(e), file=stat_file)
                                stat_file.write(traceback.format_exc())
                                edit_pred = None
                            if edit_pred is not None:
                                pred_code_nbest = edit_pred["pred"]
                                # pred_code_scores = edit_pred["score"]
                                edited_lineidx[err_line_stmt_idx] = True
                                curr_highest_prob = max(prob_list[err_line])

                                topk = 1
                                pred_code_nbest = pred_code_nbest[:topk]
                                pred_code_nbest_2rank = {x:i for i, x in enumerate(pred_code_nbest)}
                                rank_to_remove = {}

                                #remove duplicates
                                for __idx in range(ARGS.num_preds -topk):
                                    __code = pred_stmt[err_line_stmt_idx][_pred.pred_best + __idx] #base pred code
                                    if __code in pred_code_nbest_2rank: #overlap btw base pred and editor pred
                                        rank = pred_code_nbest_2rank[__code]
                                        if __idx in blacklist[err_line]: #already in blacklist
                                            rank_to_remove[rank] = True
                                        else:
                                            rank_to_remove[rank] = True
                                            prob_list[err_line][__idx] = curr_highest_prob+1

                                pred_code_nbest = [code for r, code in enumerate(pred_code_nbest) if r not in rank_to_remove]
                                topk = len(pred_code_nbest)
                                if topk > 0:
                                    print ("adding {} editor preds".format(topk), file=stat_file)
                                    #add editor preds to prob list (set the highest prob in line) and pred_stmt (replace with bottom preds)
                                    prob_list[err_line][-topk:] = list(curr_highest_prob + np.arange(topk) +1)
                                    pred_stmt[err_line_stmt_idx][_pred.pred_best +ARGS.num_preds - topk: _pred.pred_best +ARGS.num_preds] = pred_code_nbest
                                print ("prob_list[err_line]:", prob_list[err_line], file=stat_file)

                        heap.rebuild()

                stat_file.write("Number of programs compiled:  " + str(compile_count) + "\n")
                stat_file.write(str(passed) + " " + str(error) + "\n")

                # if public didn't pass then proceed
                if passed == pass_test.none:
                    if error != err.compile_err:
                        test_results = error_message["test_results_public"]
                        for (i, test_result) in enumerate(test_results):
                            if test_result["error_info"] is not None:
                                x = json.dumps(test_result, ensure_ascii=False, indent=2)
                                print (x, file=stat_file)
                    stat_file.write("continuing best first search...\n\n")
                elif passed == pass_test.public:
                    stat_file.write("passed public but failed hidden!\n\n")
                    return True, False
                else:
                    stat_file.write("passed public and hidden!\n\n")
                    return True, True

        return False, False



class FragilePrioritySet(object):
    """
    Rebuild the heap every time prob_list changes.
    """

    def __init__(self, prob_list):
        # prob_list[i][j] = logprob
        # prob_list is a shared object that can be updated outside the class
        self.prob_list = prob_list
        self.L = len(self.prob_list)
        # (neg_logprob, (j1, ..., jL), (r1, ..., rL))
        # true indices (j1, ..., jL) are used to track used combinations
        # ranks (r1, ..., rL) are used for finding neighbors
        self.heap = []
        self.rebuild()
        # Store combinations (j1, ..., jL) returnd during any reincarnation
        self.used = set()
        # Store combinations (j1, ..., jL) added to heap in this reincarnation
        self.added = set()

    def rebuild(self):
        # rankings[i][r] = j
        self.rankings = []
        for i, logprobs in enumerate(self.prob_list):
            ranked = sorted((-logprob, j) for (j, logprob) in enumerate(logprobs))
            self.rankings.append([j for (_, j) in ranked])
        # print ("len(self.rankings)", len(self.rankings))
        # print (self.rankings)
        self.heap.clear()
        self.added = set()
        self._add_r_tuple(tuple([0] * self.L))

    def _add_r_tuple(self, r_tuple):
        assert isinstance(r_tuple, tuple)
        # print (r_tuple)
        j_tuple = tuple(self.rankings[i][r] for (i, r) in enumerate(r_tuple))
        if j_tuple in self.added:
            return
        assert len(j_tuple) == self.L
        logprob = sum(self.prob_list[i][j] for (i, j) in enumerate(j_tuple))
        heapq.heappush(self.heap, (-logprob, j_tuple, r_tuple))
        self.added.add(j_tuple)

    def add(self, log_prob, idx):
        raise NotImplementedError('Should not be called.')

    def pop(self):
        # Keep popping the heap until an unused item is found
        while True:
            neg_logprob, j_tuple, r_tuple = heapq.heappop(self.heap)
            # Immediately add all neighbors
            for i in range(self.L):
                if r_tuple[i] < ARGS.num_preds - 1:
                    neighbor = list(r_tuple)
                    neighbor[i] += 1
                    self._add_r_tuple(tuple(neighbor))
            # If not used, return it
            if j_tuple not in self.used:
                self.used.add(j_tuple)
                return (neg_logprob, list(j_tuple))

    def __len__(self):
        return len(self.heap)

    def empty(self):
        return len(self.heap) == 0



################################################

def stitch_helper(probno):
    input = ARGS.input
    count = 0
    inp_stmt, pred_stmt = [], []
    # print ("starting", probno)
    # the following look extracts the input/pred lines for the probno specified
    # and passes it further for stitching
    with open(input + '.tsv','r') as tsvin, open(input + '.summary','r') as predin:
        tsvin.readline(), predin.readline()
        probid, subid = None, None

        while True:
            inp = tsvin.readline()
            if not inp:
                # Special handling for last line
                assert count == probno, \
                    'num problems = {} but probno = {}'.format(count, probno)
                break
            inp = inp.split('\t')
            pred = predin.readline().split("\t")
            if int(inp[_inp.line].strip()) == 0:
                if count == probno:
                    break
                count += 1
                probid, subid = inp[_inp.probid].strip(), inp[_inp.subid].strip()
                hitid = inp[_inp.hitid].strip()
            if count == probno:
                inp_stmt.append(inp)
                pred_stmt.append(pred)

    # generate a unique id for this program
    unique_id = "{:04d}-{}-{}".format(probno, probid, subid)
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


    public, hidden = stitch_error_localize_edit_search(inp_stmt, pred_stmt, probid, subid)
    if public:
        tmp_f = open("passed_public.txt", "w")
        tmp_f.close()
    if hidden:
        tmp_f = open("passed_hidden.txt", "w")
        tmp_f.close()

    os.chdir(init_run_dir)
    print ()
    return


def stitch():
    if ARGS.n_parallel:
        probno = ARGS.probno
        n_parallel = ARGS.n_parallel
        probnos = list(np.arange(n_parallel) + probno) #e.g. [201,202,203]
        # Parallel(n_jobs=mp.cpu_count()*5)(delayed(stitch_helper)(probno) for (_, probno) in enumerate(probnos))
        for probno_ in probnos:
            stitch_helper(probno_)
    else:
        stitch_helper(ARGS.probno)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--prog-dir', default=repo_root+'/raw_data/spoc_data/spoc/testcases',
            help='Path the codeforces-data repository, which contains test cases')
    parser.add_argument('--max-heap', type=int, default=999999,
            help='Suicide when heap is bigger than this')
    parser.add_argument('-t', '--timeout', type=int, default=20,
            help='Timeout for execution (in seconds)')
    parser.add_argument('-T', '--gcc-timeout', type=int, default=60,
            help='Timeout for compilation (in seconds)')
    parser.add_argument('-c', '--compile-budget', type=int, default=100,
            help='Number of maximum g++ calls')
    parser.add_argument('-p', '--num-preds', type=int, default=100,
            help='Number of predictions per line')
    parser.add_argument('--n-parallel', type=int, default=0)
    parser.add_argument('input')
    parser.add_argument('probno', type=int)


    group = parser.add_argument_group('error')
    group.add_argument('--err-handling', choices=['black', 'gray'], default='gray',
            help='Whether to blacklist or downweight error lines')
    group.add_argument('--err-gray-amount', type=float, default=10,
            help='(for graylisting) Amount to penalize error lines')

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
