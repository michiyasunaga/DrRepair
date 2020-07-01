# Error detection methods
import json, sys
import math
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import time


sys.path.append("../utils")
from code_process import tokenize_err_msg


LINE_OFFSET_DEEPFIX=1
LINE_OFFSET_SPOC=5


def parse_error(raw_err_msg, line_offset, tokenize=True):
    """
    Return the first error line number and error message.

    Args:
        raw_err_msg (str): Raw string from g++
    Returns:
        stmt_index: real line number - line_offset.
        That is, the line number where the first non preamble line is line 0,
        and where DUMMY lines are still included.
    """
    lines = raw_err_msg.split('\n')
    for line in lines:
        m = re.match('[^:]*:(\d+):[:0-9 ]+error: (.*)', line)
        if not m:
            continue
        lineno, message = m.groups()
        if tokenize:
            message = ' '.join(tokenize_err_msg(message))
        return int(lineno) - line_offset, message.strip()

    for line in lines:
        m = re.match('[^:]*:(\d+):[:0-9 ]+: (.*)', line)
        if not m:
            continue
        lineno, message = m.groups()
        if tokenize:
            message = ' '.join(tokenize_err_msg(message))
        return int(lineno) - line_offset, message.strip()

    return None, None


def post_request(server, post_fields):
    for attempt in range(10):
        try:
            request = Request(server, urlencode(post_fields).encode())
            with urlopen(request, timeout=5) as x:
                response = x.read().decode()
            response = json.loads(response)
            return response
        except Exception as e:
            time.sleep(1 + attempt)
    return None


################################################


class RepairPolicy(object):
    """
    Ask the PyTorch server. return n-best
    """
    def __init__(self, args, for_deepfix=False):
        self.server_list = args.repairer_server
        try:
            self.info = {'probno': args.probno}
        except:
            self.info = {'probno': 'N/A'}
        self.threshold = args.err_localize_threshold
        self.for_deepfix = for_deepfix

    def policy_localize(self, code_lines, feedback, threshold_ON=False):
        if self.for_deepfix:
            DUMMY_linenos = {idx: None for (idx, code_line) in enumerate(code_lines) if code_line[0].strip() == "DUMMY"}
        else:
            DUMMY_linenos = {idx: None for (idx, code_line) in enumerate(code_lines) if code_line[0].strip() in ["", "DUMMY"]}

        if self.for_deepfix:
            lineno, msg = parse_error(feedback, LINE_OFFSET_DEEPFIX, tokenize=True)
        else:
            lineno, msg = parse_error(feedback, LINE_OFFSET_SPOC, tokenize=True)
        if msg is None:
            return None, None, None
        err_line_obj = {'lineno': lineno, 'msg': msg}
        q = {
            'info': self.info,
            'code_lines': code_lines,
            'err_line': err_line_obj,
            "gold_linenos": [0]*len(code_lines), #temporary for now
            "edit_linenos": [0]*len(code_lines), #temporary for now
            "gold_code_lines": code_lines,  #dummy
            "comment": {"method": "localize_only", "beam_size": None}
        }
        _server = self.server_list[np.random.choice(len(self.server_list))]
        response = post_request(_server, {'q': json.dumps(q)})
        if response is None:
            assert False, "server failed!"
        logit = response['logit_localize'][0]
        probs = self.softmax(logit)
        pred_lineno = None #response['pred'][0]
        for (i, idx) in enumerate((-np.array(logit)).argsort()):
            if idx not in DUMMY_linenos:
                pred_lineno = idx
                break
        if threshold_ON and probs[pred_lineno] < self.threshold: #abstain
            return None, None, err_line_obj
        return pred_lineno, None, err_line_obj


    def policy_edit(self, code_lines, feedback, pred_lineno, beam_size=10):
        # pred_lineno: int
        if self.for_deepfix:
            lineno, msg = parse_error(feedback, LINE_OFFSET_DEEPFIX, tokenize=True)
        else:
            lineno, msg = parse_error(feedback, LINE_OFFSET_SPOC, tokenize=True)
        if msg is None:
            return None, None, None
        err_line_obj = {'lineno': lineno, 'msg': msg}
        indicators = [0]*len(code_lines)
        indicators[pred_lineno] = 1
        q = {
            'info': self.info,
            'code_lines': code_lines,
            'err_line': err_line_obj,
            "gold_linenos": indicators,
            "edit_linenos": indicators,
            "gold_code_lines": code_lines,  #dummy
            "comment": {"method": "edit_only", "beam_size": beam_size}
        }
        _server = self.server_list[np.random.choice(len(self.server_list))]
        response = post_request(_server, {'q': json.dumps(q)})
        if response is None:
            assert False, "server_compiler_localize_edit failed!"
        response["pred"] = response["pred_edit"]
        return pred_lineno, response, err_line_obj


    def policy_localize_edit(self, code_lines, feedback, beam_size=10):
        # pred_lineno: int
        if self.for_deepfix:
            DUMMY_linenos = {idx: None for (idx, code_line) in enumerate(code_lines) if code_line[0].strip() == "DUMMY"}
        else:
            DUMMY_linenos = {idx: None for (idx, code_line) in enumerate(code_lines) if code_line[0].strip() in ["", "DUMMY"]}

        if self.for_deepfix:
            lineno, msg = parse_error(feedback, LINE_OFFSET_DEEPFIX, tokenize=True)
        else:
            lineno, msg = parse_error(feedback, LINE_OFFSET_SPOC, tokenize=True)
        if msg is None:
            return None, None, None
        err_line_obj = {'lineno': lineno, 'msg': msg}
        q = {
            'info': self.info,
            'code_lines': code_lines,
            'err_line': err_line_obj,
            "gold_linenos": [0]*len(code_lines), #temporary for now
            "edit_linenos": [0]*len(code_lines), #temporary for now
            "gold_code_lines": code_lines,  #dummy
            "comment": {"method": "localize_edit", "beam_size": beam_size}
        }
        _server = self.server_list[np.random.choice(len(self.server_list))]
        response = post_request(_server, {'q': json.dumps(q)})
        if response is None:
            assert False, "server_compiler_localize_edit failed!"
        logit = response['logit_localize'][0]
        probs = self.softmax(logit)
        pred_lineno = None #response['pred'][0]
        for (i, idx) in enumerate((-np.array(logit)).argsort()):
            if idx not in DUMMY_linenos:
                pred_lineno = idx
                break
        if threshold_ON and probs[pred_lineno] < self.threshold: #abstain
            return None, None, err_line_obj
        response["pred"] = response["pred_edit"]
        return pred_lineno, response, err_line_obj


    def softmax(self, numbers):
        numbers = [math.exp(x - max(numbers)) for x in numbers]
        return [x / sum(numbers) for x in numbers]


################################################
