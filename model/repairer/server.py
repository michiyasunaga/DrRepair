# Bottle does not play well with classes, so let's use plain methods
import datetime
import json
import sys
import torch
import numpy as np

from bottle import post, request, run


@post('/pred')
def process():
    q = request.forms.q
    q = json.loads(q)

    print('[{}] Received {}'.format(datetime.datetime.now().time(), q.get('info')))
    sys.stdout.flush()
    batch, comment = EXP.dataset.s_parse_request(q)

    assert len(batch) == 1
    EXP.model.eval() #IMPORTANT
    with torch.no_grad():
        if comment["method"] == "localize_only":
            all_enc_stuff = EXP.model.forward_encode(batch)
            #localize
            logit_localize, label_localize = EXP.model.forward_localize(batch, all_enc_stuff)
            pred_localize = EXP.model.get_pred_localization(logit_localize, batch)
            ret = EXP.dataset.s_generate_response(q, batch, logit_localize, (pred_localize, None))

        elif comment["method"] == "edit_only":
            all_enc_stuff = EXP.model.forward_encode(batch)
            logit_edit, label_edit = EXP.model.forward_edit(batch, all_enc_stuff, train_mode=False, beam_size=comment["beam_size"], edit_lineno_specified=None)
            pred_edit = EXP.model.get_pred_edit(logit_edit, batch, train_mode=False, retAllHyp=True)
            ret = EXP.dataset.s_generate_response(q, batch, None, (None, pred_edit))

        else: #'localize_edit'
            all_enc_stuff = EXP.model.forward_encode(batch)
            #localize
            logit_localize, label_localize = EXP.model.forward_localize(batch, all_enc_stuff)
            pred_localize = EXP.model.get_pred_localization(logit_localize, batch)
            pred_lineno = pred_localize[0].item() #one scalar
            #edit
            logit_edit, label_edit = EXP.model.forward_edit(batch, all_enc_stuff, train_mode=False, beam_size=comment["beam_size"], edit_lineno_specified=[pred_lineno])
            pred_edit = EXP.model.get_pred_edit(logit_edit, batch, train_mode=False, retAllHyp=True)
            ret = EXP.dataset.s_generate_response(q, batch, logit_localize, (pred_localize, pred_edit))
    return ret


def start_server(exp, host, port):
    global EXP
    EXP = exp
    # This will open a global port!
    print('[{}] Starting server'.format(datetime.datetime.now().time()))
    run(server='cheroot', host=host, port=port)
    print('\nGood bye!')
