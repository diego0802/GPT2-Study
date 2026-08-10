import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from conll import evaluate
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    for batch in data:
        optimizer.zero_grad()
        slot_logits, intent_logits = model(batch['input_ids'], batch['attention_mask'])
        
        loss_slot = criterion_slots(slot_logits.permute(0, 2, 1), batch['slot_ids'])
        loss_intent = criterion_intents(intent_logits, batch['intent_ids'])
        loss = loss_slot + loss_intent
        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, slot2id, id2slot, id2intent):
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []
    
    with torch.no_grad():
        for batch in data:
            slot_logits, intent_logits = model(batch['input_ids'], batch['attention_mask'])
            
            loss_slot = criterion_slots(slot_logits.permute(0, 2, 1), batch['slot_ids'])
            loss_intent = criterion_intents(intent_logits, batch['intent_ids'])
            loss_array.append((loss_slot + loss_intent).item())
            
            # Intent
            out_intents = [id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()]
            ref_intents.extend([id2intent[x] for x in batch['intent_ids'].tolist()])
            hyp_intents.extend(out_intents)
            
            # Slots – ALLINEAMENTO CORRETTO
            output_slots = torch.argmax(slot_logits, dim=1)  # (B, L)
            for i, seq in enumerate(output_slots):
                # Prendi solo i token validi (non padding)
                mask = batch['attention_mask'][i]
                length = mask.sum().item()
                
                # Input IDs e slot predetti (solo validi)
                utt_ids = batch['input_ids'][i][:length].tolist()
                pred_ids = seq[:length].tolist()
                
                # Slot ground truth (solo validi)
                gt_ids = batch['slot_ids'][i][:length].tolist()
                
                # Ricostruisci utterance (per conll)
                # Nota: per GPT2/BERT i token sono subword – per semplicità usiamo i token come "parole"
                utterance = [str(id) for id in utt_ids]  # o puoi decodificare con tokenizer.decode()
                
                # Ricostruisci slot predetti e ground truth
                ref_slots.append([(utterance[j], id2slot[gt_ids[j]]) for j in range(length)])
                hyp_slots.append([(utterance[j], id2slot[pred_ids[j]]) for j in range(length)])
    
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        results = {"total": {"f": 0}}
    
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array