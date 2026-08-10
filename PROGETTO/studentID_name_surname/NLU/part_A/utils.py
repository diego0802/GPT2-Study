import json
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import os

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
class Lang:
    def __init__(self, words, intents, slots, cutoff=0, cls=True):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items() if not cls or k != 'cls'}
        self.id2intent = {v: k for k, v in self.intent2id.items()}
    
    def w2id(self, elements, cutoff=None, unk=True, cls=True):
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
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        if cls:
            vocab['cls'] = PAD_TOKEN
        return vocab

# --- Dataset class ---
class IntentsAndSlots(Dataset):
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
        return {
            'utterance': torch.Tensor(self.utt_ids[idx]),
            'slots': torch.Tensor(self.slot_ids[idx]),
            'intent': self.intent_ids[idx]
        }
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                tmp_seq.append(mapper[x] if x in mapper else mapper[self.unk])
            if self.add_cls:
                tmp_seq.append(mapper[self.cls])
            res.append(tmp_seq)
        return res

# --- Collate function ---
def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            padded_seqs[i, :lengths[i]] = seq
        return padded_seqs, lengths
    
    data_by_key = {key: [d[key] for d in data] for key in data[0].keys()}
    src_utt, _ = merge(data_by_key['utterance'])
    y_slots, y_lengths = merge(data_by_key['slots'])
    intent = torch.LongTensor(data_by_key['intent'])
    
    return {
        'utterances': src_utt.to(DEVICE),
        'intents': intent.to(DEVICE),
        'y_slots': y_slots.to(DEVICE),
        'slots_len': torch.LongTensor(y_lengths).to(DEVICE)
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader, lang
