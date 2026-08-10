import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from transformers import AutoTokenizer
from utils import *
from model import GPT2ForIntentSlots, BertForIntentSlots
from functions import train_loop, eval_loop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def experiment_2b(model_type='gpt2'):
    # Load data
    train_raw = load_data('dataset/ATIS/train.json')
    test_raw = load_data('dataset/ATIS/test.json')
    train_raw, dev_raw = create_dev_set(train_raw, portion=0.10)
    
    # Build vocab
    slot2id, intent2id, id2slot, id2intent = build_vocab(train_raw, dev_raw, test_raw)
    
    # Tokenizer
    if model_type == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForIntentSlots(
            model_name='openai-community/gpt2',
            n_intents=len(intent2id),
            n_slots=len(slot2id)
        )
    elif model_type == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = BertForIntentSlots(
            model_name='bert-base-uncased',
            n_intents=len(intent2id),
            n_slots=len(slot2id)
        )
    
    model.to(DEVICE)
    
    # Datasets
    train_dataset = ATISDataset(train_raw, tokenizer, slot2id, intent2id)
    dev_dataset = ATISDataset(dev_raw, tokenizer, slot2id, intent2id)
    test_dataset = ATISDataset(test_raw, tokenizer, slot2id, intent2id)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Training
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=0)
    criterion_intents = nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_model = None
    patience = 3
    
    for epoch in range(50):
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        
        if epoch % 5 == 0:
            results_dev, intent_res, _ = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model,
                slot2id, id2slot, id2intent
            )
            f1 = results_dev['total']['f']
            acc = intent_res['accuracy']
            print(f"Epoch {epoch}: Slot F1={f1:.3f}, Intent Acc={acc:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model.state_dict()
                patience = 3
            else:
                patience -= 1
                if patience <= 0:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    # Test
    model.load_state_dict(best_model)
    results_test, intent_test, _ = eval_loop(
        test_loader, criterion_slots, criterion_intents, model,
        slot2id, id2slot, id2intent
    )
    print(f"Test: Slot F1={results_test['total']['f']:.3f}, Intent Acc={intent_test['accuracy']:.3f}")

if __name__ == "__main__":
    experiment_2b(model_type='gpt2')
    # experiment_2b(model_type='bert')