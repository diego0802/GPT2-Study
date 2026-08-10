import torch
from torch.utils.data import DataLoader, Dataset
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
    
    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    # count non-pad tokens
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size*2, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, collate_fn=collate)
    
    return train_loader, dev_loader, test_loader, tokenizer
