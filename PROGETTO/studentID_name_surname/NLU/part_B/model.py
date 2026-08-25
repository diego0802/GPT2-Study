import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, BertModel

class GPT2ForIntentSlots(nn.Module):
    def __init__(self, model_name='openai-community/gpt2', n_intents=21, n_slots=140, dropout=0.1):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.hidden_size = self.gpt2.config.hidden_size
        
        self.slot_classifier = nn.Linear(self.hidden_size, n_slots)
        self.intent_classifier = nn.Linear(self.hidden_size, n_intents)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in [self.slot_classifier, self.intent_classifier]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Slot prediction
        slot_logits = self.slot_classifier(self.dropout(hidden_states))
        
        # Intent: usa l'ultimo token (EOS con left padding)
        # Con left padding, l'ultimo token è sempre l'EOS token
        last_hidden = hidden_states[:, -1, :]  # Ultimo token
        intent_logits = self.intent_classifier(self.dropout(last_hidden))
        
        return slot_logits, intent_logits

class BertForIntentSlots(nn.Module):
    def __init__(self, model_name='bert-base-uncased', n_intents=21, n_slots=140, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        self.slot_classifier = nn.Linear(self.hidden_size, n_slots)
        self.intent_classifier = nn.Linear(self.hidden_size, n_intents)
        
        self.dropout = nn.Dropout(dropout)
        
        # Inizializza i nuovi layer
        self._init_weights()
        
    def _init_weights(self):
        for m in [self.slot_classifier, self.intent_classifier]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Slot prediction: per ogni token
        slot_logits = self.slot_classifier(self.dropout(hidden_states))
        
        # Intent: usa il CLS token (primo token)
        cls_hidden = hidden_states[:, 0, :]
        intent_logits = self.intent_classifier(self.dropout(cls_hidden))
        
        return slot_logits, intent_logits