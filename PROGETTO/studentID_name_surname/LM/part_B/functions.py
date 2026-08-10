import math
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def train_loop_1b(data, optimizer, model, tokenizer):
    model.train()
    loss_array = []
    number_of_tokens = []
    for i, (input_ids, _, n_tokens) in enumerate(data):
        optimizer.zero_grad()
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100
        output = model(input_ids, labels=labels)
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        output.loss.backward()
        optimizer.step()
        if i % 100 == 0:
            avg_loss = sum(loss_array) / sum(number_of_tokens)
            print(f"Batch {i}: Loss = {avg_loss:.4f}")
    return sum(loss_array) / sum(number_of_tokens)

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
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

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