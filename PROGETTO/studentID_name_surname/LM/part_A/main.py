import os
import urllib.request
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from functions import *
from utils import get_dataloaders
from model import GPT2

# ============================================================
# 1. DOWNLOAD DATASET (se non presente)
# ============================================================
def download_dataset():
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


# ============================================================
# 2. FUNZIONE PER SALVARE REPORT (CSV + GRAFICO)
# ============================================================
def save_report(config, losses_train, losses_dev, sampled_epochs, final_ppl, final_loss, final_acc, save_csv=True, save_plot=True):
    losses_train = [float(x) for x in losses_train] if isinstance(losses_train, list) else losses_train
    losses_dev = [float(x) for x in losses_dev] if isinstance(losses_dev, list) else losses_dev
    
    df = pd.DataFrame({
        'epoch': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev
    })
    
    if save_csv:
        csv_name = f"report_d{config['d_model']}_l{config['num_layers']}_d{config['dropout']}.csv"
        df.to_csv(csv_name, index=False)
        print(f"✅ CSV salvato: {csv_name}")
    
    if save_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(sampled_epochs, losses_train, label='Train Loss')
        plt.plot(sampled_epochs, losses_dev, label='Dev Loss')
        plt.title(f"Training curves - d_model={config['d_model']}, layers={config['num_layers']}")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_name = f"plot_d{config['d_model']}_l{config['num_layers']}_d{config['dropout']}.png"
        plt.savefig(plot_name)
        plt.close()
        print(f"✅ Grafico salvato: {plot_name}")
    
    print("\n📊 RIEPILOGO CONFIGURAZIONE")
    for k, v in config.items():
        print(f"{k}: {v}")
    print(f"Final Test PPL: {final_ppl:.2f}")
    print(f"Final Test Loss: {final_loss:.4f}")
    print(f"Final Test Acc: {final_acc:.4f}")


# ============================================================
# 3. ESPERIMENTO PRINCIPALE
# ============================================================
def experiment_1a():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    download_dataset()
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token ID: {tokenizer.pad_token_id}")

    train_loader, dev_loader, test_loader, tokenizer = get_dataloaders(
        tokenizer,
        "dataset/PennTreeBank/ptb.train.txt",
        "dataset/PennTreeBank/ptb.valid.txt", 
        "dataset/PennTreeBank/ptb.test.txt",
        batch_size=8,
        device=device
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    configs = [
        {"lr": 0.001, "d_model": 20, "n_heads": 1, "num_layers": 1, "ff_dim": 20, "dropout": 0.0, "weight_tying": False},
        {"lr": 0.0005, "d_model": 64, "n_heads": 1, "num_layers": 1, "ff_dim": 64, "dropout": 0.0, "weight_tying": False},
        {"lr": 0.0005, "d_model": 64, "n_heads": 4, "num_layers": 1, "ff_dim": 64, "dropout": 0.0, "weight_tying": False},
        {"lr": 0.0005, "d_model": 64, "n_heads": 4, "num_layers": 1, "ff_dim": 64, "dropout": 0.1, "weight_tying": False},
        {"lr": 0.0005, "d_model": 64, "n_heads": 4, "num_layers": 1, "ff_dim": 64, "dropout": 0.1, "weight_tying": True},
        {"lr": 0.0003, "d_model": 128, "n_heads": 4, "num_layers": 2, "ff_dim": 512, "dropout": 0.1, "weight_tying": True},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}\nTesting config: {config}\n{'='*60}")
        start_time = time.time()
        
        model = GPT2(
            vocab_size=len(tokenizer),
            pos_emb_size=1024,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            num_layers=config["num_layers"],
            ff_dim=config["ff_dim"],
            dropout=config["dropout"],
            weight_tying=config["weight_tying"]
        )
        
        best_model, best_ppl, losses_train, losses_dev = train_model(
            model, 
            train_loader, 
            dev_loader, 
            criterion,
            tokenizer,
            lr=config["lr"],
            n_epochs=1,
            patience=10,
            device=device
        )
        
        # 🔥 CORREZIONE: riceve 3 valori da eval_loop
        test_ppl, test_loss, test_acc = eval_loop(test_loader, criterion, best_model, tokenizer)
        elapsed = time.time() - start_time
        
        results.append({
            "config": config,
            "best_val_ppl": best_ppl,
            "test_ppl": test_ppl,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "time": elapsed
        })
        
        print(f"Val PPL: {best_ppl:.2f} | Test PPL: {test_ppl:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Time: {elapsed:.1f}s")
        
        save_report(
            config=config,
            losses_train=losses_train,
            losses_dev=losses_dev,
            sampled_epochs=list(range(len(losses_train))),
            final_ppl=test_ppl,
            final_loss=test_loss,
            final_acc=test_acc
        )
        
        torch.save(best_model.state_dict(), f"best_model_d{config['d_model']}_l{config['num_layers']}.pt")
        print(f"💾 Modello salvato: best_model_d{config['d_model']}_l{config['num_layers']}.pt")
    
    # ============================================================
    # 4. TABELLA FINALE DEI RISULTATI
    # ============================================================
    print("\n" + "="*80)
    print("📊 TABELLA FINALE DEI RISULTATI")
    print("="*80)
    print(f"{'d_model':<8} {'layers':<6} {'dropout':<8} {'tying':<6} {'Val PPL':<10} {'Test PPL':<10} {'Test Acc':<10} {'Time (s)':<10}")
    print("-"*80)
    for r in results:
        c = r["config"]
        print(f"{c['d_model']:<8} {c['num_layers']:<6} {c['dropout']:<8} {str(c['weight_tying']):<6} "
              f"{r['best_val_ppl']:<10.2f} {r['test_ppl']:<10.2f} {r['test_acc']:<10.4f} {r['time']:<10.1f}")
    print("="*80)

    # Salva tabella finale
    df_final = pd.DataFrame(results)
    df_final.to_csv("final_results.csv", index=False)
    print("✅ Tabella finale salvata in 'final_results.csv'")

if __name__ == "__main__":
    experiment_1a()