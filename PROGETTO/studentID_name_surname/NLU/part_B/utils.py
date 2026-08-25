import json
from matplotlib import pyplot as plt
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import os

import urllib

PAD_TOKEN = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load data ---
def load_data(path):
    with open(path) as f:
        return json.loads(f.read())

# --- Create dev set ---
def create_dev_set(train_raw, portion=0.10):
    intents = [x['intent'] for x in train_raw]
    count_y = Counter(intents)
    
    inputs = []
    labels = []
    mini_train = []
    
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(train_raw[id_y])
    
    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs, labels,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels
    )
    X_train.extend(mini_train)
    return X_train, X_dev

# --- Lang class ---
class Lang(): #Constructor
    #This class builds and manages vocabulary mappings for words, intents, and
    #  slots in a structured way. It's a more sophisticated version of the previous
    #  code you saw.
    def __init__(self, words, intents, slots, cutoff=0, cls=True): 
        #Builds all three vocabularies using the helper methods.
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, pad=True, cls=False)  # <-- Fix!
        self.intent2id = self.lab2id(intents, pad=False, cls=False)  # <-- Fix!
        self.id2word = {v:k for k, v in self.word2id.items()}
        # cls will have the same id as the pad token
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True, cls=True):
        #Creates mapping from words to numbers, with optional cutoff for rare words.
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        if cls:
            vocab['cls'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True, cls=False):  # <-- cls=False di default
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        # Non aggiungere 'cls' come label - è solo per i word embedding
        return vocab

class ATISDataset(Dataset):
    def __init__(self, data, tokenizer, slot2id, intent2id, model_type='gpt2'):
        self.data = data
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item['utterance'].split()
        slots = item['slots'].split()
        intent = item['intent']

        # Tokenizza
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors=None,
            add_special_tokens=True
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        word_ids = encoding.word_ids()
        
        # Crea slot_ids allineati con word_ids
        slot_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None or word_idx >= len(slots):
                slot_ids.append(-100)  # Token speciali o fuori range
            elif word_idx != previous_word_idx:
                slot_id = self.slot2id.get(slots[word_idx], self.slot2id.get('O', 0))
                slot_ids.append(slot_id)
                previous_word_idx = word_idx
            else:
                slot_ids.append(-100)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'slot_ids': slot_ids,
            'intent_id': self.intent2id[intent],
            'word_ids': word_ids,
            'original_words': words
        }

# --- Dataset class ---
class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk', cls='cls', add_cls=True):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.cls = cls
        self.add_cls = add_cls
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx]) #get utterance
        slots = torch.Tensor(self.slot_ids[idx]) #get slot
        intent = self.intent_ids[idx] #get intent
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            if self.add_cls:
                tmp_seq.append(mapper[self.cls])
            res.append(tmp_seq)
        return res

# --- Collate function ---
def collate_fn_hf(batch, pad_token_id=0, model_type='gpt2'):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    slot_ids = [item['slot_ids'] for item in batch]
    intents = [item['intent_id'] for item in batch]
    original_words = [item['original_words'] for item in batch]
    word_ids = [item['word_ids'] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    if model_type == 'gpt2':
        padding_side = "left"
    else:
        padding_side = "right"
    
    padded_input_ids = torch.LongTensor(len(batch), max_len).fill_(pad_token_id)
    padded_attention_mask = torch.LongTensor(len(batch), max_len).fill_(0)
    padded_slot_ids = torch.LongTensor(len(batch), max_len).fill_(-100)
    
    # Padded word_ids (None per padding)
    padded_word_ids = [[None] * max_len for _ in range(len(batch))]
    
    for i in range(len(batch)):
        length = len(input_ids[i])
        
        if padding_side == "left":
            offset = max_len - length
            padded_input_ids[i, offset:] = torch.tensor(input_ids[i])
            padded_attention_mask[i, offset:] = torch.tensor(attention_mask[i])
            padded_slot_ids[i, offset:] = torch.tensor(slot_ids[i])
            padded_word_ids[i][offset:] = word_ids[i]
        else:
            padded_input_ids[i, :length] = torch.tensor(input_ids[i])
            padded_attention_mask[i, :length] = torch.tensor(attention_mask[i])
            padded_slot_ids[i, :length] = torch.tensor(slot_ids[i])
            padded_word_ids[i][:length] = word_ids[i]
    
    return {
        'input_ids': padded_input_ids.to(DEVICE),
        'attention_mask': padded_attention_mask.to(DEVICE),
        'slot_ids': padded_slot_ids.to(DEVICE),
        'intents': torch.tensor(intents).to(DEVICE),
        'original_words': original_words,
        'word_ids': padded_word_ids
    }

# --- Build dataloaders ---
def get_dataloaders(train_raw, dev_raw, test_raw, batch_size=128):
    # Build vocabularies
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)
    
    # Create datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn_hf, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn_hf)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn_hf)
    
    return train_loader, dev_loader, test_loader, lang

def save_report_2b(config, losses_train, losses_dev, sampled_epochs, 
                   final_slot_f1, final_slot_p, final_slot_r, 
                   final_intent_acc, final_intent_f1):
    model_name = f"{config['model_type']}_{config['model_size']}"
    lr_str = f"{config['lr']:.0e}".replace('e-', 'e')
    
    df = pd.DataFrame({
        'epoch': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev,
        'final_slot_f1': final_slot_f1,
        'final_slot_p': final_slot_p,
        'final_slot_r': final_slot_r,
        'final_intent_acc': final_intent_acc,
        'final_intent_f1': final_intent_f1
    })
    
    csv_name = f"report_2B_{model_name}_lr{lr_str}_drop{config['dropout']}.csv"
    df.to_csv(csv_name, index=False)
    print(f"✅ CSV saved: {csv_name}")

    plt.figure(figsize=(8, 5))
    plt.plot(sampled_epochs, losses_train, label='Train Loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev Loss')
    plt.title(f"2B - {model_name} (LR={config['lr']}, Dropout={config['dropout']})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_name = f"plot_2B_{model_name}_lr{lr_str}_drop{config['dropout']}.png"
    plt.savefig(plot_name)
    plt.close()
    print(f"✅ Plot saved: {plot_name}")

    print("\nConfiguration Summary")
    for k, v in config.items():
        if k != 'hf_name':
            print(f"{k}: {v}")
    print(f"Final Slot F1: {final_slot_f1:.3f}")
    print(f"Final Slot P: {final_slot_p:.3f}")
    print(f"Final Slot R: {final_slot_r:.3f}")
    print(f"Final Intent Acc: {final_intent_acc:.3f}")
    print(f"Final Intent F1: {final_intent_f1:.3f}")

def download_atis():
    base_url = "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/ATIS/"
    files = ["train.json", "test.json"]
    os.makedirs("dataset/ATIS", exist_ok=True)
    
    for file in files:
        url = base_url + file
        path = f"dataset/ATIS/{file}"
        if not os.path.exists(path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"{file} already exists")