import math
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

def train_loop(data, optimizer, criterion, model):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))
    #tqdm is a Python library that creates progress bars for loops and processes.

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad() # Zeroing the gradient
        output = model(input_ids)
        # need to reshape as (B, vocab, L)
        loss = criterion(output.permute(0,2,1), labels) #criterion is just a variable name for a loss function - a mathematical formula that measures how wrong the model's predictions are.
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array)/sum(number_of_tokens)).item())

    return sum(loss_array)/sum(number_of_tokens)

def train_model_1b(model, train_loader, dev_loader, tokenizer, lr=0.001, n_epochs=100, patience=3, device="cpu"):
    model = model.to(device)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    losses_train, losses_dev, sampled_epochs = [], [], []
    best_ppl = math.inf
    best_model = None
    patience_counter = patience
    
    for epoch in range(n_epochs):
        # Train
        loss = train_loop(train_loader, optimizer, model, tokenizer)
        losses_train.append(loss)
        sampled_epochs.append(epoch)
        
        # Dev (valuta OGNI epoca, come nell'1A)
        ppl_dev, loss_dev, acc_dev = eval_loop_1b(dev_loader, model, tokenizer)
        losses_dev.append(loss_dev)
        
        # Stampa come nell'1A
        print(f"Epoch {epoch}: Train Loss={loss:.4f} | Dev Loss={loss_dev:.4f} | PPL={ppl_dev:.2f} | Acc={acc_dev:.4f}")
        
        # Early stopping
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model)
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_model, best_ppl, losses_train, losses_dev

def eval_loop_1b(data, model, tokenizer):
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for input_ids, _, n_tokens in data:
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100
            output = model(input_ids, labels=labels)
            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

            preds = torch.argmax(output.logits, dim=-1)
            mask = labels != -100
            correct += (preds == labels).masked_select(mask).sum().item()
            total += mask.sum().item()
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    acc = correct / total if total > 0 else 0.0
    return ppl, loss_to_return, acc

def train_model_1b(model, train_loader, dev_loader, tokenizer, lr=0.001, n_epochs=100, patience=3, device="cpu"):
    model = model.to(device)
    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    losses_train, losses_dev, sampled_epochs = [], [], []
    best_ppl = math.inf
    best_model = None
    patience_counter = patience
    
    for epoch in range(n_epochs):
        loss = train_loop_1b(train_loader, optimizer, model, tokenizer)
        losses_train.append(loss)
        sampled_epochs.append(epoch)
        
        ppl_dev, loss_dev = eval_loop_1b(dev_loader, model, tokenizer)
        losses_dev.append(loss_dev)
        print(f"Epoch {epoch}: Train Loss={loss:.4f}, Dev Loss={loss_dev:.4f}, Dev PPL={ppl_dev:.2f}")
        
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model)
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                print(f"Early stopping at epoch {epoch}")
                break
    return best_model, best_ppl, losses_train, losses_dev