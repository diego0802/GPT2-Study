import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2LMHeadModel
import urllib
from model import CustomGPT2Attention
from utils import get_dataloaders
from functions import train_loop_1b, eval_loop_1b, train_model_1b

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def save_report_1b(config, losses_train, losses_dev, sampled_epochs, final_ppl):
    """Salva CSV e grafico per 1B (come in 1A)"""
    losses_train = [float(x) for x in losses_train] if isinstance(losses_train, list) else losses_train
    losses_dev = [float(x) for x in losses_dev] if isinstance(losses_dev, list) else losses_dev
    df = pd.DataFrame({
        'epoch': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev
    })
    
    csv_name = f"report_1B_rank{config['rank']}_alpha{config['alpha']}.csv"
    df.to_csv(csv_name, index=False)
    print(f"✅ CSV salvato: {csv_name}")

    plt.figure(figsize=(8, 5))
    plt.plot(sampled_epochs, losses_train, label='Train Loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev Loss')
    plt.title(f"1B LoRA - rank={config['rank']}, alpha={config['alpha']}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_name = f"plot_1B_rank{config['rank']}_alpha{config['alpha']}.png"
    plt.savefig(plot_name)
    plt.close()
    print(f"✅ Grafico salvato: {plot_name}")

def experiment_1b():
    print(f"Using device: {DEVICE}")
    download_dataset()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Dataloader
    train_loader, dev_loader, test_loader, tokenizer = get_dataloaders(
        tokenizer,
        "dataset/PennTreeBank/ptb.train.txt",
        "dataset/PennTreeBank/ptb.valid.txt",
        "dataset/PennTreeBank/ptb.test.txt",
        batch_size=8,
        device=DEVICE
    )
    
    # Configurazioni LoRA
    configs = [
        {"rank": 1, "alpha": 1, "lr": 1e-4},
        {"rank": 2, "alpha": 1, "lr": 1e-4},
        {"rank": 4, "alpha": 1, "lr": 5e-5},
        {"rank": 8, "alpha": 2, "lr": 5e-5},
        {"rank": 16, "alpha": 2, "lr": 2e-5},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}\nTesting LoRA: rank={config['rank']}, alpha={config['alpha']}, lr={config['lr']}")
        start_time = time.time()
        
        # Carica modello pre-addestrato
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        
        # Sostituisci attenzioni con LoRA
        for block in model.transformer.h:
            original_attn = block.attn
            block.attn = CustomGPT2Attention(original_attn, config["rank"], config["alpha"])
        
        # Congela tutto, sblocca solo LoRA
        for param in model.parameters():
            param.requires_grad = False
        for module in model.modules():
            if hasattr(module, 'lora_q_A'):
                for param in module.parameters():
                    param.requires_grad = True
        
        model.to(DEVICE)
        
        # Verifica parametri trainabili
        trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
                trainable += param.numel()
        print(f"Total trainable params: {trainable:,}")
        
        # Training
        best_model, best_ppl, losses_train, losses_dev = train_model_1b(
            model, train_loader, dev_loader, tokenizer,
            lr=config["lr"],
            n_epochs=1,      # breve su Colab, su Azure puoi aumentare
            patience=3,
            device=DEVICE
        )
        
        # Test
        test_ppl, _ = eval_loop_1b(test_loader, best_model, tokenizer)
        elapsed = time.time() - start_time
        
        # Salva risultati
        results.append({
            "config": config,
            "best_val_ppl": best_ppl,
            "test_ppl": test_ppl,
            "time": elapsed
        })
        
        print(f"Val PPL: {best_ppl:.2f} | Test PPL: {test_ppl:.2f} | Time: {elapsed:.1f}s")
        
        # Report
        save_report_1b(
            config=config,
            losses_train=losses_train,
            losses_dev=losses_dev,
            sampled_epochs=list(range(len(losses_train))),
            final_ppl=test_ppl
        )
        
        # Salva modello
        torch.save(best_model.state_dict(), f"best_model_rank{config['rank']}_alpha{config['alpha']}.pt")
    
    # Tabella finale
    print("\n" + "="*70)
    print("📊 TABELLA FINALE 1B")
    print("="*70)
    print(f"{'rank':<6} {'alpha':<6} {'lr':<12} {'Val PPL':<10} {'Test PPL':<10} {'Time (s)':<10}")
    print("-"*70)
    for r in results:
        c = r["config"]
        print(f"{c['rank']:<6} {c['alpha']:<6} {c['lr']:<12} "
              f"{r['best_val_ppl']:<10.2f} {r['test_ppl']:<10.2f} {r['time']:<10.1f}")
    print("="*70)
    
    df_final = pd.DataFrame(results)
    df_final.to_csv("final_results_1B.csv", index=False)
    print("✅ Tabella finale salvata in 'final_results_1B.csv'")

if __name__ == "__main__":
    experiment_1b()