import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from conll import conlleval, evaluate
from sklearn.metrics import classification_report
PAD_TOKEN = 0

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []

    for i, batch in enumerate(data):
        optimizer.zero_grad() # Zeroing the gradient

        slots, intent = model(batch['utterances'], batch['slots_len'])
        slots = slots.permute(0,2,1) # We need this for computing the loss

        loss_intent = criterion_intents(intent, batch['intents'])
        loss_slot = criterion_slots(slots, batch['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for batch in data:
            slots, intents = model(batch['utterances'], batch['slots_len'])
            slots = slots.permute(0,2,1) # We need this for computing the loss
            loss_intent = criterion_intents(intents, batch['intents'])
            loss_slot = criterion_slots(slots, batch['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())

            # Intent inference
            # Get the most probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in batch['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = batch['slots_len'].tolist()[id_seq] - 1 # -1, we ignore the CLS

                utt_ids = batch['utterances'][id_seq][:length].tolist()
                gt_ids = batch['y_slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = [lang.id2word[elem] for elem in utt_ids]

                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0, "p":0, "r":0}}
    
    # ✅ Calcolo metriche intent
    from sklearn.metrics import classification_report
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    intent_acc = report_intent.get('accuracy', 0)
    intent_f1 = report_intent.get('macro avg', {}).get('f1-score', 0)
    
    # ✅ Estrazione metriche slot
    slot_f1 = results.get('total', {}).get('f', 0)
    slot_p = results.get('total', {}).get('p', 0)
    slot_r = results.get('total', {}).get('r', 0)
    
    # ✅ Ritorno tutte le metriche
    return results, report_intent, loss_array, intent_acc, intent_f1, slot_f1, slot_p, slot_r, np.mean(loss_array)