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
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        # cls will have the same id as the pad token
        self.id2slot = {v:k for k, v in self.slot2id.items() if not cls or k != 'cls'}
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
    
    def lab2id(self, elements, pad=True, cls=True):
        #Creates mapping from labels to numbers.
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        if cls:
            # when predicting the slots, we want to ignore the CLS
            # CLS will only be used for intent classification
            vocab['cls'] = PAD_TOKEN
        return vocab

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
def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        # Input: list of variable-length sequences (each a list of token IDs)
        # e.g., [[1,2,3], [4,5], [6,7,8,9]]
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # The pad token matrix is simply a grid of zeros used to make all sequences
        #  the same length so they can be processed together as a batch. 
        # batch_size * maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        return padded_seqs, lengths

    data_by_key = {}
    for key in data[0].keys():
        data_by_key[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(data_by_key['utterance'])
    y_slots, y_lengths = merge(data_by_key["slots"])
    intent = torch.LongTensor(data_by_key["intent"])
    
    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item = {}
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, lang

def save_report_2a(config, losses_train, losses_dev, sampled_epochs, final_slot_f1, final_slot_p, final_slot_r, final_intent_acc, final_intent_f1):
    df = pd.DataFrame({
        'epoch': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev,
        'sampled_epochs': sampled_epochs,
        'final_slot_f1': final_slot_f1,
        'final_slot_p': final_slot_p,
        'final_slot_r': final_slot_r,
        'final_intent_acc': final_intent_acc,
        'final_intent_f1': final_intent_f1
    })
    
    csv_name = f"report_2A_d{config['d_model']}_l{config['num_layers']}_d{config['dropout']}.csv"
    df.to_csv(csv_name, index=False)
    print(f"✅ CSV salvato: {csv_name}")

    plt.figure(figsize=(8, 5))
    plt.plot(sampled_epochs, losses_train, label='Train Loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev Loss')
    plt.title(f"2A - d_model={config['d_model']}, layers={config['num_layers']}, dropout={config['dropout']}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_name = f"plot_2A_d{config['d_model']}_l{config['num_layers']}_d{config['dropout']}.png"
    plt.savefig(plot_name)
    plt.close()
    print(f"✅ Grafico salvato: {plot_name}")

    print("\n📊 RIEPILOGO CONFIGURAZIONE")
    for k, v in config.items():
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