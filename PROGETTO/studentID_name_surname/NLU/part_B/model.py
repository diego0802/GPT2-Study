import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, BertModel

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # 🔥 DUE dropout diversi (come richiesto dall'esercizio)
        self.dropout_attn = nn.Dropout(dropout)   # dopo softmax
        self.dropout_out = nn.Dropout(dropout)   # dopo out_proj

    def forward(self, x, mask):
        B, L, d_model = x.size()

        q = self.w_q(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        similarity = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.h_dim))
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(similarity, dim=-1)
        attn = self.dropout_attn(attn)   # ✅ dropout dopo softmax

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, L, d_model)
        y = self.out_proj(y)
        y = self.dropout_out(y)          # ✅ dropout dopo out_proj

        return y
    
#FeedForward (also called FFN - FeedForward Network) is a simple neural network 
# that processes each token independently to transform its representation. 
# It's the "thinking" part of a transformer block.
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            # LAYER 1: Linear(d_model=512, hidden_dim=2048)
            nn.Linear(d_model, hidden_dim),
            # WHAT: Expands each token's representation to 4x size
            # WHY: Gives model more capacity to learn complex patterns

            # LAYER 2: GELU (Gaussian Error Linear Unit)
            nn.GELU(),
            # WHAT: Non-linear activation function (smooth version of ReLU)
            # WHY: Allows model to learn non-linear relationships
            # GELU(x) ≈ x * Φ(x) where Φ is Gaussian CDF

            # LAYER 3: Linear(hidden_dim=2048, d_model=512)
            nn.Linear(hidden_dim, d_model),
            # WHAT: Compresses back to original dimension
            # WHY: Maintains shape for residual connections
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # applied over each token independently
        return self.net(x)
    
#TransformerBlock is the basic building unit that combines Attention 
# (communication between tokens) and FeedForward (individual processing) into a 
# single reusable component. Stacking multiple blocks creates a deep transformer 
# model.
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        # INPUT: x shape = (B, L, d_model)
        #   - B: batch size (number of sequences in parallel)
        #   - L: sequence length (number of tokens per sequence)
        #   - d_model: embedding dimension (e.g., 512)
        # 
        # OUTPUT: same shape as input (B, L, d_model)
        #   - Each token's features are normalized independently
        #   - Output has mean ≈ 0 and std ≈ 1 across the d_model dimension
        #   - Preserves the relative relationships between features
        self.ln1 = nn.LayerNorm(d_model)
        
        # INPUT: x shape = (B, L, d_model)
        #   - B: batch size
        #   - L: sequence length
        #   - d_model: embedding dimension (e.g., 512)
        #
        # OUTPUT: attended shape = (B, L, d_model)
        #   - Same shape as input
        #   - Each token now contains information from other tokens (self-attention)
        #   - Attention weights determine how much each token "looks at" others
        #
        # EXAMPLE:
        #   Input:  "The cat sat" → 3 tokens, each 512-dim
        #   Output: Same 3 tokens, each 512-dim, but "sat" now knows about "The" and "cat"
        #
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

        # INPUT: x shape = (B, L, d_model)
        #   - Output from previous layer (e.g., after attention or after residual connection)
        #
        # OUTPUT: same shape (B, L, d_model)
        #   - Normalized across the d_model dimension for each token independently
        #   - Mean ≈ 0, Standard deviation ≈ 1 per token
        #   - Ready for the next sub-layer (FeedForward)
        #
        # EXAMPLE:
        #   Input:  [10.0, -5.0, 3.0, 2.0] for one token (d_model=4)
        #   Output: [1.44, -1.73, 0.10, -0.10]  # Same shape, normalized values
        #
        self.ln2 = nn.LayerNorm(d_model)

        # INPUT: x shape = (B, L, d_model)
        #   - Normalized tokens from LayerNorm
        #
        # OUTPUT: transformed shape = (B, L, d_model)
        #   - Same shape as input
        #   - Each token processed independently (no communication between tokens)
        #   - Pattern: Expand (d_model → ff_dim) → Activate (GELU) → Compress (ff_dim → d_model)
        #   - ff_dim is typically 4× d_model (e.g., 512 → 2048 → 512)
        #
        # EXAMPLE:
        #   Input:  Token "bank" (512-dim) with context already added
        #   Output: Same token (512-dim) but transformed to emphasize its meaning
        #           ("river bank" vs "financial bank" emerges from this transformation)
        #
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        # layer norm, attention, residual
        x = x + self.attn(self.ln1(x), mask)
        # layer norm, feedforward, residual
        x = x + self.ff(self.ln2(x))
        return x
    
# In model.py:

class GPT2ForIntentSlots(nn.Module):
    def __init__(self, model_name='openai-community/gpt2', n_intents=21, n_slots=140, dropout=0.1):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.hidden_size = self.gpt2.config.hidden_size
        
        self.slot_classifier = nn.Linear(self.hidden_size, n_slots)
        
        # MLP per intent - ✅ FUNZIONA se _init_weights è corretto
        self.intent_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, n_intents)
        )
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        # Inizializza slot_classifier (è un Linear)
        torch.nn.init.uniform_(self.slot_classifier.weight, -0.01, 0.01)
        if self.slot_classifier.bias is not None:
            self.slot_classifier.bias.data.fill_(0.01)
        
        # Inizializza intent_classifier (Sequential) - ✅ CORRETTO
        for m in self.intent_classifier:
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Slot prediction: per ogni token
        slot_logits = self.slot_classifier(self.dropout(hidden_states))
        
        # Intent: usa l'ultimo token NON PAD
        last_token_indices = attention_mask.sum(dim=1) - 1
        cls_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
        intent_logits = self.intent_classifier(self.dropout(cls_hidden))
        
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