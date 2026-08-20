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

def experiment_2a():
    # --- Load data ---
    download_atis()
    train_raw = load_data('dataset/ATIS/train.json')
    test_raw = load_data('dataset/ATIS/test.json')
    train_raw, dev_raw = create_dev_set(train_raw, portion=0.10)
    
    # --- Dataloaders ---
    train_loader, dev_loader, test_loader, lang = get_dataloaders(
        train_raw, dev_raw, test_raw, batch_size=32
    )
    
    # --- Configurazioni ---
    configs = [
        {"lr": 0.001, "d_model": 64, "n_heads": 2, "num_layers": 2, 
         "ff_dim": 128, "dropout": 0.1, "n_epochs": 100, "patience": 15},
        {"lr": 0.0005, "d_model": 128, "n_heads": 4, "num_layers": 2, 
         "ff_dim": 256, "dropout": 0.1, "n_epochs": 120, "patience": 15},
        {"lr": 0.0005, "d_model": 128, "n_heads": 8, "num_layers": 3, 
         "ff_dim": 512, "dropout": 0.1, "n_epochs": 150, "patience": 20},
        {"lr": 0.0003, "d_model": 256, "n_heads": 8, "num_layers": 3, 
         "ff_dim": 512, "dropout": 0.15, "n_epochs": 150, "patience": 20},
        {"lr": 0.0002, "d_model": 256, "n_heads": 8, "num_layers": 4, 
         "ff_dim": 1024, "dropout": 0.15, "n_epochs": 200, "patience": 25},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}\nTesting config: {config}\n{'='*60}")
        start_time = time.time()
        
        # --- Modello ---
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
        
        # Conta parametri
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        
        best_f1 = 0
        best_model = model.state_dict()
        losses_train, losses_dev, sampled_epochs = [], [], []
        epoch_times = []
        best_epoch = 0
        patience = config["patience"]
        
        for epoch in range(config["n_epochs"]):
            epoch_start = time.time()
            
            # Training
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            avg_train_loss = np.mean(loss)
            losses_train.append(avg_train_loss)
            sampled_epochs.append(epoch)
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Validation
            results_dev, intent_res, loss_dev, intent_acc, intent_f1, slot_f1, slot_p, slot_r, avg_dev_loss = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
            losses_dev.append(avg_dev_loss)
            
            print(f"Epoch {epoch}: "
                  f"Slot F1={slot_f1:.3f} | Slot P={slot_p:.3f} | Slot R={slot_r:.3f} | "
                  f"Intent Acc={intent_acc:.3f} | Intent F1={intent_f1:.3f} | "
                  f"Train Loss={avg_train_loss:.3f} | Dev Loss={avg_dev_loss:.3f} | "
                  f"Time={epoch_time:.1f}s")
            
            # Early stopping
            if slot_f1 > best_f1:
                best_f1 = slot_f1
                best_model = model.state_dict()
                best_epoch = epoch
                patience = config["patience"]  # Reset patience
            else:
                patience -= 1
                if patience <= 0:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # --- Test ---
        model.load_state_dict(best_model)
        
        results_test, intent_test, _, intent_acc, intent_f1, slot_f1, slot_p, slot_r, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, model, lang
        )
        
        # Calcola metriche aggiuntive per slot (escludendo O)
        non_o_f1s = [v['f'] for k, v in results_test.items() 
                     if k != 'O' and k != 'total' and isinstance(v, dict)]
        non_o_avg_f1 = np.mean(non_o_f1s) if non_o_f1s else 0
        
        # Metriche O
        slot_o_f1_test = results_test.get('O', {}).get('f', 0)
        
        elapsed = time.time() - start_time
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
        
        # Salva risultati
        results.append({
            "config": config,
            "d_model": config["d_model"],
            "layers": config["num_layers"],
            "dropout": config["dropout"],
            "params_total": total_params,
            "params_trainable": trainable_params,
            "slot_f1": slot_f1,
            "slot_p": slot_p,
            "slot_r": slot_r,
            "slot_o_f1": slot_o_f1_test,
            "slot_non_o_avg_f1": non_o_avg_f1,
            "intent_acc": intent_acc,
            "intent_f1": intent_f1,
            "best_epoch": best_epoch,
            "total_epochs": len(sampled_epochs),
            "time_total": elapsed,
            "time_avg_epoch": avg_epoch_time,
            "best_train_loss": min(losses_train) if losses_train else 0,
            "best_dev_loss": min(losses_dev) if losses_dev else 0
        })
        
        print(f"\n✅ Test Results: Slot F1={slot_f1:.3f} | Intent Acc={intent_acc:.3f} | Time: {elapsed:.1f}s")
        
        # Report (senza parametri inutili)
        save_report_2a(
            config=config,
            losses_train=losses_train,
            losses_dev=losses_dev,
            sampled_epochs=sampled_epochs,
            final_slot_f1=slot_f1,
            final_slot_p=slot_p,
            final_slot_r=slot_r,
            final_intent_acc=intent_acc,
            final_intent_f1=intent_f1
        )
        
        # Salva modello
        torch.save(best_model, f"best_model_2A_d{config['d_model']}_l{config['num_layers']}.pt")
    
    # --- Tabella finale ---
    print("\n" + "="*140)
    print("📊 TABELLA FINALE 2A - TUTTE LE METRICHE")
    print("="*140)
    print(f"{'d_model':<8} {'layers':<6} {'dropout':<8} {'Slot F1':<10} {'Slot P':<10} {'Slot R':<10} "
          f"{'Slot O F1':<10} {'Non-O F1':<10} {'Intent Acc':<10} {'Intent F1':<10} "
          f"{'Best Ep':<8} {'Params (M)':<12} {'Time (s)':<10}")
    print("-"*140)
    for r in results:
        params_m = r['params_total'] / 1_000_000
        print(f"{r['d_model']:<8} {r['layers']:<6} {r['dropout']:<8} "
              f"{r['slot_f1']:<10.3f} {r['slot_p']:<10.3f} {r['slot_r']:<10.3f} "
              f"{r['slot_o_f1']:<10.3f} {r['slot_non_o_avg_f1']:<10.3f} "
              f"{r['intent_acc']:<10.3f} {r['intent_f1']:<10.3f} "
              f"{r['best_epoch']:<8} {params_m:<12.2f} {r['time_total']:<10.1f}")
    print("="*140)
    
    # Salva CSV
    df_csv = pd.DataFrame(results)
    df_csv.to_csv("final_results_2A_complete.csv", index=False)
    print("✅ Tabella completa salvata in 'final_results_2A_complete.csv'")
    
    # Miglior modello per Slot F1
    best_slot = max(results, key=lambda x: x['slot_f1'])
    print("\n🏆 MIGLIOR MODELLO (Slot F1):")
    print(f"  d_model={best_slot['d_model']}, layers={best_slot['layers']}, dropout={best_slot['dropout']}")
    print(f"  Slot F1={best_slot['slot_f1']:.3f} | Slot P={best_slot['slot_p']:.3f} | Slot R={best_slot['slot_r']:.3f}")
    print(f"  Intent Acc={best_slot['intent_acc']:.3f} | Intent F1={best_slot['intent_f1']:.3f}")
    print(f"  Parametri: {best_slot['params_total']/1_000_000:.2f}M")
    print(f"  Tempo: {best_slot['time_total']:.1f}s")
    
    # Miglior modello per Intent
    best_intent = max(results, key=lambda x: x['intent_acc'])
    print("\n🏆 MIGLIOR MODELLO (Intent Acc):")
    print(f"  d_model={best_intent['d_model']}, layers={best_intent['layers']}, dropout={best_intent['dropout']}")
    print(f"  Intent Acc={best_intent['intent_acc']:.3f} | Intent F1={best_intent['intent_f1']:.3f}")
    print(f"  Slot F1={best_intent['slot_f1']:.3f}")

if __name__ == "__main__":
    experiment_2a()