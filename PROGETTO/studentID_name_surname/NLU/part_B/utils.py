import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from collections import Counter
from sklearn.model_selection import train_test_split

PAD_TOKEN = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(path):
    with open(path) as f:
        return json.loads(f.read())

def create_dev_set(train_raw, portion=0.10):
    intents = [x['intent'] for x in train_raw]
    count_y = Counter(intents)
    
    inputs, labels, mini_train = [], [], []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(train_raw[id_y])
    
    X_train, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels
    )
    X_train.extend(mini_train)
    return X_train, X_dev

class ATISDataset(Dataset):
    def __init__(self, data, tokenizer, slot2id, intent2id, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        utterance = item['utterance']
        slots = item['slots'].split()
        intent = item['intent']
        
        # Tokenizza con subwords
        tokens = self.tokenizer(
            utterance,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Allinea slots ai subword tokens
        # Per GPT2: word_ids() ci dice a quale parola appartiene ogni subword
        word_ids = tokens.word_ids()
        
        # Crea slot_ids: prendi il primo subword di ogni parola
        slot_ids = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                slot_ids.append(PAD_TOKEN)  # special tokens
            elif word_id != prev_word:
                # Primo subword della parola → assegna il slot
                if word_id < len(slots):
                    slot_ids.append(self.slot2id.get(slots[word_id], 0))
                else:
                    slot_ids.append(PAD_TOKEN)
                prev_word = word_id
            else:
                # Subword successivo → assegna PAD (ignorato nella loss)
                slot_ids.append(PAD_TOKEN)
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'slot_ids': torch.tensor(slot_ids, dtype=torch.long),
            'intent_id': torch.tensor(self.intent2id[intent], dtype=torch.long),
            'word_ids': word_ids  # Per debug
        }

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    slot_ids = torch.stack([b['slot_ids'] for b in batch])
    intent_ids = torch.stack([b['intent_id'] for b in batch])
    
    return {
        'input_ids': input_ids.to(DEVICE),
        'attention_mask': attention_mask.to(DEVICE),
        'slot_ids': slot_ids.to(DEVICE),
        'intent_ids': intent_ids.to(DEVICE)
    }

def build_vocab(train_raw, dev_raw, test_raw):
    """
    Costruisce i dizionari per slot e intent.
    Restituisce: slot2id, intent2id, id2slot, id2intent
    """
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    
    slot2id = {slot: i+1 for i, slot in enumerate(slots)}  # 0 = PAD
    slot2id['PAD'] = 0
    intent2id = {intent: i for i, intent in enumerate(intents)}
    id2slot = {v: k for k, v in slot2id.items()}
    id2intent = {v: k for k, v in intent2id.items()}
    return slot2id, intent2id, id2slot, id2intent