import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from utils import *
from model import GPT2ForIntentSlots, BertForIntentSlots
from functions import train_loop_hf, eval_loop_hf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def experiment_2b():
    # --- Load data ---
    download_atis()
    train_raw = load_data('dataset/ATIS/train.json')
    test_raw = load_data('dataset/ATIS/test.json')
    train_raw, dev_raw = create_dev_set(train_raw, portion=0.10)
    
    # --- Costruisci Lang (come nel notebook!) ---
    print("Building vocabulary with Lang...")
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    
    lang = Lang(words, intents, slots, cutoff=0)
    
    # Estrai i dizionari
    word2id = lang.word2id
    id2word = lang.id2word
    slot2id = lang.slot2id
    id2slot = lang.id2slot
    intent2id = lang.intent2id
    id2intent = lang.id2intent
    
    print(f"   Vocabulary: {len(word2id)} words")
    print(f"   Slot: {len(slot2id)}")
    print(f"   Intent: {len(intent2id)}")
    
    configs = [
    # ===== GPT-2 BASE =====
    {
        "model_type": "gpt2",
        "model_size": "base",
        "hf_name": "openai-community/gpt2",
        "lr": 3e-5,
        "batch_size": 32,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "n_epochs": 60,
        "patience": 10,
        "warmup_ratio": 0.1,
    },
    
    # ===== GPT-2 BASE - Dropout alto =====
    {
        "model_type": "gpt2",
        "model_size": "base",
        "hf_name": "openai-community/gpt2",
        "lr": 3e-5,
        "batch_size": 32,
        "dropout": 0.2,
        "weight_decay": 0.01,
        "n_epochs": 60,
        "patience": 10,
        "warmup_ratio": 0.1,
    },
    
    # ===== GPT-2 BASE - LR basso =====
    {
        "model_type": "gpt2",
        "model_size": "base",
        "hf_name": "openai-community/gpt2",
        "lr": 2e-5,
        "batch_size": 32,
        "dropout": 0.15,
        "weight_decay": 0.02,
        "n_epochs": 70,
        "patience": 12,
        "warmup_ratio": 0.1,
    },
    
    # ===== GPT-2 MEDIUM =====
    {
        "model_type": "gpt2",
        "model_size": "medium",
        "hf_name": "openai-community/gpt2-medium",
        "lr": 2e-5,
        "batch_size": 16,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "n_epochs": 50,
        "patience": 8,
        "warmup_ratio": 0.1,
    },
    
    # ===== GPT-2 MEDIUM - Dropout alto =====
    {
        "model_type": "gpt2",
        "model_size": "medium",
        "hf_name": "openai-community/gpt2-medium",
        "lr": 2e-5,
        "batch_size": 16,
        "dropout": 0.2,
        "weight_decay": 0.01,
        "n_epochs": 50,
        "patience": 8,
        "warmup_ratio": 0.1,
    },
    
    # ===== GPT-2 MEDIUM - LR basso =====
    {
        "model_type": "gpt2",
        "model_size": "medium",
        "hf_name": "openai-community/gpt2-medium",
        "lr": 1e-5,
        "batch_size": 24,
        "dropout": 0.15,
        "weight_decay": 0.02,
        "n_epochs": 60,
        "patience": 10,
        "warmup_ratio": 0.1,
    },
    
    # ===== BERT BASE =====
    {
        "model_type": "bert",
        "model_size": "base",
        "hf_name": "bert-base-uncased",
        "lr": 3e-5,
        "batch_size": 32,
        "dropout": 0.15,
        "weight_decay": 0.01,
        "n_epochs": 60,
        "patience": 10,
        "warmup_ratio": 0.1,
    },
    
    # ===== BERT BASE - Dropout alto =====
    {
        "model_type": "bert",
        "model_size": "base",
        "hf_name": "bert-base-uncased",
        "lr": 3e-5,
        "batch_size": 32,
        "dropout": 0.2,
        "weight_decay": 0.01,
        "n_epochs": 60,
        "patience": 10,
        "warmup_ratio": 0.1,
    },
    
    # ===== BERT BASE - LR basso =====
    {
        "model_type": "bert",
        "model_size": "base",
        "hf_name": "bert-base-uncased",
        "lr": 2e-5,
        "batch_size": 32,
        "dropout": 0.15,
        "weight_decay": 0.02,
        "n_epochs": 70,
        "patience": 12,
        "warmup_ratio": 0.1,
    },
    
    # ===== BERT LARGE =====
    {
        "model_type": "bert",
        "model_size": "large",
        "hf_name": "bert-large-uncased",
        "lr": 2e-5,
        "batch_size": 16,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "n_epochs": 40,
        "patience": 8,
        "warmup_ratio": 0.1,
    },
    
    # ===== BERT LARGE - Dropout alto =====
    {
        "model_type": "bert",
        "model_size": "large",
        "hf_name": "bert-large-uncased",
        "lr": 2e-5,
        "batch_size": 16,
        "dropout": 0.2,
        "weight_decay": 0.01,
        "n_epochs": 40,
        "patience": 8,
        "warmup_ratio": 0.1,
    },
    
    # ===== BERT LARGE - LR basso =====
    {
        "model_type": "bert",
        "model_size": "large",
        "hf_name": "bert-large-uncased",
        "lr": 1e-5,
        "batch_size": 24,
        "dropout": 0.15,
        "weight_decay": 0.02,
        "n_epochs": 50,
        "patience": 10,
        "warmup_ratio": 0.1,
    },
]
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing config: {config['model_type']}-{config['model_size']}")
        print(f"  LR: {config['lr']}, Batch: {config['batch_size']}, Dropout: {config['dropout']}")
        print(f"{'='*60}")
        start_time = time.time()
        
        # --- Tokenizer e Modello ---
        if config['model_type'] == 'gpt2':
            tokenizer = AutoTokenizer.from_pretrained(config['hf_name'])
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            model = GPT2ForIntentSlots(
                model_name=config['hf_name'],
                n_intents=len(intent2id),
                n_slots=len(slot2id),
                dropout=config['dropout']
            )
        elif config['model_type'] == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(config['hf_name'])
            model = BertForIntentSlots(
                model_name=config['hf_name'],
                n_intents=len(intent2id),
                n_slots=len(slot2id),
                dropout=config['dropout']
            )
        
        model.to(DEVICE)
        
        # Conta parametri
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total Parameters: {total_params/1_000_000:.2f}M")
        
        # --- Datasets ---
        train_dataset = ATISDataset(
            train_raw, 
            tokenizer, 
            slot2id, 
            intent2id, 
            model_type=config['model_type']
        )
        dev_dataset = ATISDataset(
            dev_raw, 
            tokenizer, 
            slot2id, 
            intent2id, 
            model_type=config['model_type']
        )
        test_dataset = ATISDataset(
            test_raw, 
            tokenizer, 
            slot2id, 
            intent2id, 
            model_type=config['model_type']
        )
        
        # --- Dataloaders ---
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn_hf
        )
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=config['batch_size'], 
            collate_fn=collate_fn_hf
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            collate_fn=collate_fn_hf
        )
        
        # --- Training ---
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler con warmup
        num_training_steps = len(train_loader) * config['n_epochs']
        num_warmup_steps = int(config['warmup_ratio'] * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # --- Calcola class weights per bilanciare le classi slot ---
        print("\n=== Slot Class Weights ===")
        all_slots = []
        for example in train_raw:
            slots = example['slots'].split()
            for slot in slots:
                if slot in slot2id:
                    all_slots.append(slot)
        
        unique_slots = np.unique(all_slots)
        class_weights = compute_class_weight('balanced', classes=unique_slots, y=all_slots)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        
        # ✅ CORREZIONE: assicurati che la lunghezza corrisponda al numero di classi
        num_slots = len(slot2id)
        if class_weights_tensor.shape[0] != num_slots:
            print(f"Length is different: Corrected from {class_weights_tensor.shape[0]} to {num_slots}")
            corrected_weights = torch.ones(num_slots, dtype=torch.float32).to(DEVICE)
            for i, slot in enumerate(unique_slots):
                slot_id = slot2id.get(slot)
                if slot_id is not None:
                    corrected_weights[slot_id] = class_weights[i]
            class_weights_tensor = corrected_weights
        class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()
        class_weights_tensor = torch.clamp(class_weights_tensor, max=5.0)

        print(f"Slot Class weights shape: {class_weights_tensor.shape}")
        print(f"Slot Weight for 'O': {class_weights_tensor[slot2id.get('O', 0)].item():.4f}")
        print(f"Slot Weight for 'B-fromloc.city_name': {class_weights_tensor[slot2id.get('B-fromloc.city_name', 0)].item():.4f}")

        all_intents = []
        for example in train_raw:
                intents = example['intent'].split()
                for intent in intents:
                    if intent in intent2id:
                        all_intents.append(intent)
        unique_intents = np.unique(all_intents)
        class_weights_intents = compute_class_weight('balanced', classes=unique_intents, y=all_intents)
        class_weights_tensor_intents = torch.tensor(class_weights_intents, dtype=torch.float32).to(DEVICE)
        class_weights_tensor_intents = class_weights_tensor_intents / class_weights_tensor_intents.mean()
        class_weights_tensor_intents = torch.clamp(class_weights_tensor_intents, max=5.0)
            
            # ✅ CORREZIONE: assicurati che la lunghezza corrisponda al numero di classi
        num_intents = len(intent2id)
        if class_weights_tensor_intents.shape[0] != num_intents:
                print(f"Length is different: Corrected from {class_weights_tensor_intents.shape[0]} to {num_intents}")
                corrected_weights = torch.ones(num_intents, dtype=torch.float32).to(DEVICE)
                for i, intent in enumerate(unique_intents):
                    intent_id = intent2id.get(intent)
                    if intent_id is not None:
                        corrected_weights[intent_id] = class_weights_intents[i]
                class_weights_tensor_intents = corrected_weights
        
        print(f"Intents Class weights shape: {class_weights_tensor_intents.shape}")
        
        # --- Crea la loss con class weights ---

        criterion_slots = nn.CrossEntropyLoss(
            ignore_index=-100, 
        weight=class_weights_tensor
        )
        criterion_intents = nn.CrossEntropyLoss(weight=class_weights_tensor_intents)
        
        # --- Training loop ---
        best_f1 = 0
        best_model = model.state_dict()
        losses_train, losses_dev, sampled_epochs = [], [], []
        epoch_times = []
        best_epoch = 0
        patience = config['patience']
        
        for epoch in range(config['n_epochs']):
            epoch_start = time.time()
            
            # Training
            loss = train_loop_hf(train_loader, optimizer, criterion_slots, criterion_intents, model)
            avg_train_loss = np.mean(loss)
            losses_train.append(avg_train_loss)
            sampled_epochs.append(epoch)
            
            scheduler.step()
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Validation
            results_dev, intent_res, loss_dev, intent_acc, intent_f1, slot_f1, slot_p, slot_r, avg_dev_loss = eval_loop_hf(
                dev_loader, 
                criterion_slots, 
                criterion_intents, 
                model,
                tokenizer,
                slot2id, 
                id2slot, 
                id2intent,
                model_type=config['model_type']
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
                patience = config['patience']
            else:
                patience -= 1
                if patience <= 0:
                    print(f"⏹️ Early stopping at epoch {epoch}")
                    break
        
        # --- Test finale ---
        model.load_state_dict(best_model)
        
        results_test, intent_res, loss_dev, intent_acc, intent_f1, slot_f1, slot_p, slot_r, avg_dev_loss = eval_loop_hf(
            test_loader, 
            criterion_slots, 
            criterion_intents, 
            model,
            tokenizer,
            slot2id, 
            id2slot, 
            id2intent,
            model_type=config['model_type']
        )
        
        # Calcola metriche aggiuntive per slot (escludendo O)
        non_o_f1s = [v['f'] for k, v in results_test.items() 
                    if k != 'O' and k != 'total' and isinstance(v, dict)]
        non_o_avg_f1 = np.mean(non_o_f1s) if non_o_f1s else 0
        
        slot_o_f1_test = results_test.get('O', {}).get('f', 0)
        
        elapsed = time.time() - start_time
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
        
        # Salva risultati
        results = {
            "config": config,
            "model": f"{config['model_type']}-{config['model_size']}",
            "lr": config['lr'],
            "batch_size": config['batch_size'],
            "dropout": config['dropout'],
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
        }
        
        print(f"\n✅ Test Results: Slot F1={slot_f1:.3f} | Intent Acc={intent_acc:.3f} | Time: {elapsed:.1f}s")
        
        # Report
        save_report_2b(
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
        torch.save(best_model, f"best_model_2B_{config['model_type']}_{config['model_size']}_lr{config['lr']}.pt")
        
    # --- Tabella finale ---
    print("\n" + "="*160)
    print("Final Table 2B")
    print("="*160)
    print(f"{'Model':<14} {'LR':<10} {'Batch':<8} {'Dropout':<10} "
          f"{'Slot F1':<10} {'Slot P':<10} {'Slot R':<10} "
          f" {'Intent Acc':<12} {'Intent F1':<10} "
          f"{'Best Ep':<8} {'Params (M)':<12} {'Time (s)':<10}")
    print("-"*160)
    params_m = results['params_total'] / 1_000_000
    print(f"{results['model']:<14} {results['lr']:<10.0e} {results['batch_size']:<8} {results['dropout']:<10} "
          f"{results['slot_f1']:<10.3f} {results['slot_p']:<10.3f} {results['slot_r']:<10.3f} "
          f"{results['intent_acc']:<12.3f} {results['intent_f1']:<10.3f} "
          f"{results['best_epoch']:<8} {params_m:<12.2f} {results['time_total']:<10.1f}")
    print("="*160)
    
    # Salva CSV
    df_csv = pd.DataFrame([results])
    df_csv.to_csv("final_results_2B_all_configs.csv", index=False)
    
    print("\nBest Model (Slot F1):")
    print(f"  Model={results['model']}, LR={results['lr']:.0e}, Batch={results['batch_size']}, Dropout={results['dropout']}")
    print(f"  Slot F1={results['slot_f1']:.3f} | Slot P={results['slot_p']:.3f} | Slot R={results['slot_r']:.3f}")
    print(f"  Intent Acc={results['intent_acc']:.3f} | Intent F1={results['intent_f1']:.3f}")
    print(f"  Parameters: {results['params_total']/1_000_000:.2f}M")
    print(f"  Time: {results['time_total']:.1f}s")

if __name__ == "__main__":
    experiment_2b()