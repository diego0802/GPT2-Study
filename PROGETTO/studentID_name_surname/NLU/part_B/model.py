import torch
import torch.nn as nn
from transformers import GPT2Model, BertModel

class GPT2ForIntentSlots(nn.Module):
    def __init__(self, model_name='openai-community/gpt2', n_intents=21, n_slots=140):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.hidden_size = self.gpt2.config.hidden_size
        
        # Classificatori aggiuntivi
        self.slot_classifier = nn.Linear(self.hidden_size, n_slots)
        self.intent_classifier = nn.Linear(self.hidden_size, n_intents)
        
        # Dropout per regolarizzazione
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, H)
        
        # Slot: usa tutti i token (ma nella loss ignoreremo i subword non primi)
        slot_logits = self.slot_classifier(self.dropout(hidden_states))
        
        # Intent: usa l'ultimo token (GPT2 è decoder-only)
        # Prendiamo l'ultimo token NON padding (attenzione alla mask)
        last_token_indices = attention_mask.sum(dim=1) - 1
        cls_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
        intent_logits = self.intent_classifier(self.dropout(cls_hidden))
        
        return slot_logits, intent_logits

class BertForIntentSlots(nn.Module):
    def __init__(self, model_name='bert-base-uncased', n_intents=21, n_slots=140):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        self.slot_classifier = nn.Linear(self.hidden_size, n_slots)
        self.intent_classifier = nn.Linear(self.hidden_size, n_intents)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        slot_logits = self.slot_classifier(self.dropout(hidden_states))
        
        # Intent: usa il [CLS] token (primo token)
        cls_hidden = hidden_states[:, 0, :]
        intent_logits = self.intent_classifier(self.dropout(cls_hidden))
        
        return slot_logits, intent_logits