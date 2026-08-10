import math
import torch.nn as nn
import torch
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import GPT2
import pandas as pd
import matplotlib.pyplot as plt

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

def eval_loop(data, eval_criterion, model, tokenizer=None):   # tokenizer opzionale
    model.eval()
    loss_array = []
    number_of_tokens = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, labels, n_tokens in data:
            output = model(input_ids)                        # (B, L, vocab)
            loss = eval_criterion(output.permute(0, 2, 1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
            
            # Accuracy sui token non-pad
            preds = torch.argmax(output, dim=-1)             # (B, L)
            mask = labels != tokenizer.pad_token_id           # ignora padding
            correct += (preds == labels).masked_select(mask).sum().item()
            total += mask.sum().item()
    
    avg_loss = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(avg_loss)
    acc = correct / total if total > 0 else 0.0
    return ppl, avg_loss, acc


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias != None:
                m.bias.data.fill_(0.01)
    
def train_model(model, train_loader, dev_loader, criterion, tokenizer, lr=0.001, n_epochs=100, patience=3, device="cuda"):
    model = model.to(device)
    model.apply(init_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    losses_train, losses_dev, sampled_epochs = [], [], []
    best_ppl = math.inf
    best_model = None
    patience_counter = patience
    
    for epoch in range(n_epochs):
        # Train
        loss = train_loop(train_loader, optimizer, criterion, model)
        losses_train.append(loss)
        sampled_epochs.append(epoch)
        
        # Dev
        ppl_dev, loss_dev, acc_dev = eval_loop(dev_loader, criterion, model, tokenizer)
        losses_dev.append(loss_dev)
        
        print(f"Epoch {epoch}: Train Loss={loss:.4f} | Dev Loss={loss_dev:.4f} | PPL={ppl_dev:.2f} | Acc={acc_dev:.4f}")
        
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

def save_report(config, losses_train, losses_dev, sampled_epochs, final_ppl, save_csv=True, save_plot=True):
    """
    Salva un report CSV e un grafico dell'andamento delle loss.
    """
    # Creazione DataFrame
    df = pd.DataFrame({
        'epoch': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev
    })
    
    # Salva CSV
    if save_csv:
        csv_path = f"report_{config['d_model']}_{config['num_layers']}_{config['dropout']}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Report salvato: {csv_path}")
    
    # Salva grafico
    if save_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(sampled_epochs, losses_train, label='Train Loss')
        plt.plot(sampled_epochs, losses_dev, label='Dev Loss')
        plt.title(f"Training curves - d_model={config['d_model']}, layers={config['num_layers']}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = f"plot_{config['d_model']}_{config['num_layers']}_{config['dropout']}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Grafico salvato: {plot_path}")
    
    # Stampa riassunto a schermo
    print("\n📊 RIEPILOGO CONFIGURAZIONE")
    for k, v in config.items():
        print(f"{k}: {v}")
    print(f"Final Test PPL: {final_ppl:.2f}")