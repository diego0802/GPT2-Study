import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from conll import conlleval, evaluate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
PAD_TOKEN = 0

# In functions.py:

def train_loop_hf(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    
    for batch in data:
        optimizer.zero_grad()
        
        slot_logits, intent_logits = model(batch['input_ids'], batch['attention_mask'])
        
        loss_slot = criterion_slots(slot_logits.transpose(1, 2), batch['slot_ids'])
        loss_intent = criterion_intents(intent_logits, batch['intents'])
        
        # Bilancia le loss
        loss = 0.5 * loss_slot + 0.5 * loss_intent
        
        loss_array.append(loss.item())
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

    return loss_array

# In functions.py, modifica eval_loop_hf per aggiungere debug:

def eval_loop_hf(data, criterion_slots, criterion_intents, model, tokenizer, 
                 slot2id, id2slot, id2intent, model_type='gpt2'):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data):
            slot_logits, intent_logits = model(batch['input_ids'], batch['attention_mask'])
            
            loss_slot = criterion_slots(slot_logits.transpose(1, 2), batch['slot_ids'])
            loss_intent = criterion_intents(intent_logits, batch['intents'])
            
            loss = 0.5 * loss_slot + 0.5 * loss_intent
            loss_array.append(loss.item())
            
            # Intent
            out_intents = [id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()]
            gt_intents = [id2intent[x] for x in batch['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot - usa word_ids per allineare correttamente
            output_slots = torch.argmax(slot_logits, dim=2)
            
            for i in range(len(batch['input_ids'])):
                # Recupera le parole originali
                words = batch['original_words'][i]
                
                # Recupera i word_ids per questo esempio
                word_ids_i = batch['word_ids'][i]
                
                # Costruisci mappe parola -> slot (solo per token che hanno word_id e non sono speciali)
                pred_slots_by_word = {}
                gt_slots_by_word = {}
                
                for j, wid in enumerate(word_ids_i):
                    if wid is None:
                        continue  # Token speciali (CLS, SEP, EOS, PAD)
                    
                    # Predizione per questo token
                    pred_id = output_slots[i][j].item()
                    pred_slot = id2slot.get(pred_id, 'O')
                    
                    # GT per questo token
                    gt_id = batch['slot_ids'][i][j].item()
                    
                    if gt_id != -100 and gt_id in id2slot:
                        gt_slot = id2slot.get(gt_id, 'O')
                    else:
                        gt_slot = None  # Non abbiamo GT per questo token
                    
                    # Salva per la parola corrispondente
                    if wid < len(words):
                        # Per la prima volta che vediamo questa parola
                        if wid not in pred_slots_by_word:
                            pred_slots_by_word[wid] = pred_slot
                        if gt_slot is not None and wid not in gt_slots_by_word:
                            gt_slots_by_word[wid] = gt_slot
                
                # Costruisci le sequenze allineate alle parole
                pred_slots_seq = [pred_slots_by_word.get(wid, 'O') for wid in range(len(words))]
                gt_slots_seq = [gt_slots_by_word.get(wid, 'O') for wid in range(len(words))]
                
                # Aggiungi alle liste
                if len(words) > 0:
                    ref_slots.append(list(zip(words, gt_slots_seq)))
                    hyp_slots.append(list(zip(words, pred_slots_seq)))
    
    # Calcola metriche
    # Converti in formato conll per valutazione
    ref_slots_conll = convert_to_conll_format(ref_slots)
    hyp_slots_conll = convert_to_conll_format(hyp_slots)
    
    try:
        results = evaluate(ref_slots_conll, hyp_slots_conll)
        if results is None:
            print("WARNING: evaluate() returned None!")
            results = {"total": {"f": 0, "p": 0, "r": 0}}
    except Exception as e:
        print("Warning in eval_loop:", e)
        results = {"total": {"f": 0, "p": 0, "r": 0}}
    
    # Usa accuracy_score per intent
    intent_acc = accuracy_score(ref_intents, hyp_intents)
    
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    intent_f1 = report_intent.get('macro avg', {}).get('f1-score', 0)
    
    # Estrai metriche slot
    if results is not None:
        slot_f1 = results.get('total', {}).get('f', 0)
        slot_p = results.get('total', {}).get('p', 0)
        slot_r = results.get('total', {}).get('r', 0)
    else:
        slot_f1 = slot_p = slot_r = 0
    
    return results, report_intent, loss_array, intent_acc, intent_f1, slot_f1, slot_p, slot_r, np.mean(loss_array)

# In functions.py, modifica convert_to_conll_format:

def convert_to_conll_format(slots_seq):
    converted = []
    for seq in slots_seq:
        converted_seq = []
        for word, slot in seq:
            if slot == 'O':
                converted_seq.append((word, 'O'))
            elif slot.startswith('B-') or slot.startswith('I-') or slot.startswith('E-') or slot.startswith('S-'):
                # Se il suffisso è 'O', convertilo a 'O' semplice
                if slot.split('-')[-1] == 'O':
                    converted_seq.append((word, 'O'))
                else:
                    converted_seq.append((word, slot))
            else:
                converted_seq.append((word, f"B-{slot}"))
        converted.append(converted_seq)
    return converted