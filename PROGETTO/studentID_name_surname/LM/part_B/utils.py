import os

from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import urllib
from transformers import AutoTokenizer
from functools import partial

class PennTreeBank(Dataset):  # <- CORRETTO: usa Dataset importato
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]

def read_file(path, eos_token="<eos>"):
    #eos_token stands for "End of Sequence" token (or "End of Sentence" token). 
    # It's a special token that marks where one example/sentence ends in your data.
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

def collate_fn(batch, tokenizer, device):
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")
    
    input_ids = tokenized.input_ids[:, :-1].detach().clone()
    labels = tokenized.input_ids[:, 1:].detach().clone()

    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)
    return input_ids, labels, n_tokens

def get_dataloaders(tokenizer, train_path, dev_path, test_path, batch_size=8, device="cpu"):
    # Funzione helper che:
    # 1. Legge i file
    # 2. Crea i dataset
    # 3. Crea i dataloader con tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    train_raw = read_file(train_path)
    dev_raw = read_file(dev_path)
    test_raw = read_file(test_path)
    
    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)
    
    collate = partial(collate_fn, tokenizer=tokenizer, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True, num_workers=8,
        pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size*2, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, collate_fn=collate)
    
    return train_loader, dev_loader, test_loader, tokenizer

def download_dataset():
    """Scarica i file del Penn Treebank se non esistono"""
    base_url = "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/"
    files = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]
    os.makedirs("dataset/PennTreeBank", exist_ok=True)
    
    for file in files:
        url = base_url + file
        path = f"dataset/PennTreeBank/{file}"
        if not os.path.exists(path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"{file} already exists")

def save_report(config, losses_train, losses_dev, sampled_epochs, final_ppl, final_loss, final_acc, save_csv=True, save_plot=True):
    losses_train = [float(x) for x in losses_train] if isinstance(losses_train, list) else losses_train
    losses_dev = [float(x) for x in losses_dev] if isinstance(losses_dev, list) else losses_dev
    
    df = pd.DataFrame({
        'epoch': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev
    })
    
    if save_csv:
        # Usa rank e alpha invece di d_model e layers
        csv_name = f"report_1B_rank{config['rank']}_alpha{config['alpha']}.csv"
        df.to_csv(csv_name, index=False)
        print(f"✅ CSV salvato: {csv_name}")
    
    if save_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(sampled_epochs, losses_train, label='Train Loss')
        plt.plot(sampled_epochs, losses_dev, label='Dev Loss')
        plt.title(f"1B LoRA - rank={config['rank']}, alpha={config['alpha']}, lr={config['lr']}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        # Usa rank e alpha nel nome del plot
        plot_name = f"plot_1B_rank{config['rank']}_alpha{config['alpha']}.png"
        plt.savefig(plot_name)
        plt.close()
        print(f"✅ Grafico salvato: {plot_name}")
    
    print("\n📊 RIEPILOGO CONFIGURAZIONE")
    for k, v in config.items():
        print(f"{k}: {v}")
    print(f"Final Test PPL: {final_ppl:.2f}")
    print(f"Final Test Loss: {final_loss:.4f}")
    print(f"Final Test Acc: {final_acc:.4f}")