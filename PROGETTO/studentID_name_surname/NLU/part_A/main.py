import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from model import GPT2
from functions import init_weights, train_loop, eval_loop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_report_2a(config, losses_train, losses_dev, sampled_epochs, final_slot_f1, final_intent_acc):
    """Salva CSV e grafico per 2A"""
    df = pd.DataFrame({
        'epoch': sampled_epochs,
        'train_loss': losses_train,
        'dev_loss': losses_dev
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
    print(f"Final Intent Acc: {final_intent_acc:.3f}")

def experiment_2a():
    # --- Load data ---
    train_raw = load_data('dataset/ATIS/train.json')
    test_raw = load_data('dataset/ATIS/test.json')
    train_raw, dev_raw = create_dev_set(train_raw, portion=0.10)
    
    # --- Dataloaders ---
    train_loader, dev_loader, test_loader, lang = get_dataloaders(
        train_raw, dev_raw, test_raw, batch_size=128
    )
    
    # --- Configurazioni ---
    configs = [
        {"lr": 0.01, "d_model": 20, "n_heads": 1, "num_layers": 1, "ff_dim": 20, "dropout": 0.0},
        {"lr": 0.005, "d_model": 64, "n_heads": 1, "num_layers": 1, "ff_dim": 64, "dropout": 0.0},
        {"lr": 0.005, "d_model": 64, "n_heads": 4, "num_layers": 1, "ff_dim": 64, "dropout": 0.0},
        {"lr": 0.005, "d_model": 64, "n_heads": 4, "num_layers": 1, "ff_dim": 64, "dropout": 0.1},
        {"lr": 0.001, "d_model": 128, "n_heads": 4, "num_layers": 2, "ff_dim": 512, "dropout": 0.1},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}\nTesting config: {config}\n{'='*60}")
        start_time = time.time()
        
        model = GPT2(
            vocab_size=len(lang.word2id),
            slots_size=len(lang.id2slot),
            n_intents=len(lang.intent2id),
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            num_layers=config["num_layers"],
            ff_dim=config["ff_dim"],
            dropout=config["dropout"]
        ).to(DEVICE)
        model.apply(init_weights)
        
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        n_epochs = 100
        patience = 5
        best_f1 = 0
        best_model = None
        losses_train, losses_dev, sampled_epochs = [], [], []
        
        for epoch in range(n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            losses_train.append(np.mean(loss))
            sampled_epochs.append(epoch)
            
            if epoch % 5 == 0:
                results_dev, intent_res, loss_dev = eval_loop(
                    dev_loader, criterion_slots, criterion_intents, model, lang
                )
                losses_dev.append(np.mean(loss_dev))
                f1 = results_dev['total']['f']
                acc = intent_res['accuracy']
                print(f"Epoch {epoch}: Slot F1={f1:.3f}, Intent Acc={acc:.3f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model.state_dict()
                    patience = 3
                else:
                    patience -= 1
                    if patience <= 0:
                        break
        
        # Test
        model.load_state_dict(best_model)
        results_test, intent_test, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, model, lang
        )
        elapsed = time.time() - start_time
        
        slot_f1 = results_test['total']['f']
        intent_acc = intent_test['accuracy']
        
        results.append({
            "config": config,
            "slot_f1": slot_f1,
            "intent_acc": intent_acc,
            "time": elapsed
        })
        
        print(f"Test: Slot F1={slot_f1:.3f}, Intent Acc={intent_acc:.3f} | Time: {elapsed:.1f}s")
        
        # Report
        save_report_2a(
            config=config,
            losses_train=losses_train,
            losses_dev=losses_dev,
            sampled_epochs=sampled_epochs,
            final_slot_f1=slot_f1,
            final_intent_acc=intent_acc
        )
        
        # Salva modello
        torch.save(best_model, f"best_model_2A_d{config['d_model']}_l{config['num_layers']}.pt")
    
    # --- Tabella finale ---
    print("\n" + "="*70)
    print("📊 TABELLA FINALE 2A")
    print("="*70)
    print(f"{'d_model':<8} {'layers':<6} {'dropout':<8} {'Slot F1':<10} {'Intent Acc':<10} {'Time (s)':<10}")
    print("-"*70)
    for r in results:
        c = r["config"]
        print(f"{c['d_model']:<8} {c['num_layers']:<6} {c['dropout']:<8} "
              f"{r['slot_f1']:<10.3f} {r['intent_acc']:<10.3f} {r['time']:<10.1f}")
    print("="*70)
    
    df_final = pd.DataFrame(results)
    df_final.to_csv("final_results_2A.csv", index=False)
    print("✅ Tabella finale salvata in 'final_results_2A.csv'")

if __name__ == "__main__":
    experiment_2a()