import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class CustomGPT2Attention(nn.Module):
    """Sostituisce l'attention di GPT2 con i layer LoRA"""
    def __init__(self, original_attn, rank, alpha):
        super().__init__()
        self.original_attn = original_attn
        self.rank = rank
        self.alpha = alpha
        
        # Congela i pesi originali
        for param in self.original_attn.parameters():
            param.requires_grad = False
        
        # Prendi le dimensioni
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        # Layer LoRA per Query, Key, Value
        # LoRA: W = W_original + B * A
        # A: dim_in -> rank, B: rank -> dim_out
        self.lora_q_A = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_q_B = nn.Linear(rank, self.embed_dim, bias=False)
        self.lora_k_A = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_k_B = nn.Linear(rank, self.embed_dim, bias=False)
        self.lora_v_A = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_v_B = nn.Linear(rank, self.embed_dim, bias=False)
        
        # Inizializza A con distribuzione normale, B con zeri
        nn.init.normal_(self.lora_q_A.weight, std=0.02)
        nn.init.normal_(self.lora_k_A.weight, std=0.02)
        nn.init.normal_(self.lora_v_A.weight, std=0.02)
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.zeros_(self.lora_k_B.weight)
        nn.init.zeros_(self.lora_v_B.weight)
        
        # Solo i layer LoRA sono trainabili
        for param in self.lora_q_A.parameters():
            param.requires_grad = True
        for param in self.lora_q_B.parameters():
            param.requires_grad = True
        for param in self.lora_k_A.parameters():
            param.requires_grad = True
        for param in self.lora_k_B.parameters():
            param.requires_grad = True
        for param in self.lora_v_A.parameters():
            param.requires_grad = True
        for param in self.lora_v_B.parameters():
            param.requires_grad = True
    
    def forward(self, hidden_states, *args, **kwargs):
        # Calcola query, key, value con i pesi originali + LoRA
        q_orig = self.original_attn.c_attn(hidden_states).split(self.embed_dim, dim=2)[0]
        q_lora = self.lora_q_B(self.lora_q_A(hidden_states)) * self.alpha
        q = q_orig + q_lora
        
        k_orig = self.original_attn.c_attn(hidden_states).split(self.embed_dim, dim=2)[1]
        k_lora = self.lora_k_B(self.lora_k_A(hidden_states)) * self.alpha
        k = k_orig + k_lora
        
        v_orig = self.original_attn.c_attn(hidden_states).split(self.embed_dim, dim=2)[2]
        v_lora = self.lora_v_B(self.lora_v_A(hidden_states)) * self.alpha
        v = v_orig + v_lora
        
        # Split in teste
        B, L, _ = q.shape
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcola similarità
        attn_weights = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, device=q.device))
        
        # Applica mask se presente
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            attn_weights = attn_weights + kwargs["attention_mask"]
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Calcola output
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        
        # ✅ Restituisci una tupla con due elementi!
        return attn_output, attn_weights

class GPT2_LoRA(GPT2LMHeadModel):
    def __init__(self, config, rank=1, alpha=1):
        super().__init__(config)
        self.rank = rank
        self.alpha = alpha
        
        # Sostituisci ogni attention block con la versione LoRA
        for i, block in enumerate(self.transformer.h):
            # Crea una nuova attention con LoRA
            original_attn = block.attn
            block.attn = CustomGPT2Attention(original_attn, rank, alpha)
        
        # Congela tutti i pesi originali (già fatto nelle attenzioni)
        for param in self.parameters():
            if not any(hasattr(module, 'lora_q_A') for module in self.modules() if module is param):
                param.requires_grad = False
    
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)