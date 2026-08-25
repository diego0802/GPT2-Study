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
        "lr": 1e-5,
        "batch_size": 32,
        "dropout": 0.25,
        "weight_decay": 0.05,
        "n_epochs": 60,
        "patience": 10,
        "warmup_ratio": 0.15,
    },
    
    # ===== GPT-2 BASE - Dropout alto =====
    {
        "model_type": "gpt2",
        "model_size": "base",
        "hf_name": "openai-community/gpt2",
        "lr": 8e-6,
        "batch_size": 32,
        "dropout": 0.3,
        "weight_decay": 0.05,
        "n_epochs": 60,
        "patience": 10,
        "warmup_ratio": 0.2,
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
        "lr": 8e-6,
        "batch_size": 24,
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
    all_results = []
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing config: {config['model_type']}-{config['model_size']}")
        print(f"  LR: {config['lr']}, Batch: {config['batch_size']}, Dropout: {config['dropout']}")
        print(f"{'='*60}")
        start_time = time.time()
        
        # --- Tokenizer e Modello ---
        if config['model_type'] == 'gpt2':
            tokenizer = AutoTokenizer.from_pretrained(config['hf_name'], add_prefix_space=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            pad_token_id = tokenizer.eos_token_id
            model = GPT2ForIntentSlots(
                model_name=config['hf_name'],
                n_intents=len(intent2id),
                n_slots=len(slot2id),
                dropout=config['dropout']
            )
        elif config['model_type'] == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(config['hf_name'])
            pad_token_id = tokenizer.pad_token_id
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
            collate_fn=lambda b: collate_fn_hf(b, pad_token_id=pad_token_id, model_type=config['model_type'])
        )
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=config['batch_size'], 
            collate_fn=lambda b: collate_fn_hf(b, pad_token_id=pad_token_id, model_type=config['model_type'])
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            collate_fn=lambda b: collate_fn_hf(b, pad_token_id=pad_token_id, model_type=config['model_type'])
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
        
        print("\n=== Slot Class Weights ===")
        all_slots = []
        for example in train_raw:
            for slot in example['slots'].split():
                if slot in slot2id:
                    all_slots.append(slot)

        # ✅ Converti in numpy array!
        present_slots = np.array(list(set(all_slots)))
        class_weights = compute_class_weight('balanced', classes=present_slots, y=all_slots)

        # Crea tensore per TUTTI gli slot
        class_weights_tensor = torch.ones(len(slot2id), dtype=torch.float32).to(DEVICE)
        for slot, weight in zip(present_slots, class_weights):
            slot_id = slot2id[slot]
            class_weights_tensor[slot_id] = weight

        class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()
        class_weights_tensor = torch.clamp(class_weights_tensor, max=5.0)

        print(f"Slot Class weights shape: {class_weights_tensor.shape}")

        # --- Calcola class weights per gli intents ---
        all_intents = [example['intent'] for example in train_raw if example['intent'] in intent2id]

        # ✅ Converti in numpy array!
        present_intents = np.array(list(set(all_intents)))
        class_weights_intents = compute_class_weight('balanced', classes=present_intents, y=all_intents)

        # Crea tensore per TUTTI gli intent
        class_weights_tensor_intents = torch.ones(len(intent2id), dtype=torch.float32).to(DEVICE)
        for intent, weight in zip(present_intents, class_weights_intents):
            intent_id = intent2id[intent]
            class_weights_tensor_intents[intent_id] = weight

        class_weights_tensor_intents = class_weights_tensor_intents / class_weights_tensor_intents.mean()
        class_weights_tensor_intents = torch.clamp(class_weights_tensor_intents, max=5.0)

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
            "intent_acc": intent_acc,
            "intent_f1": intent_f1,
            "best_epoch": best_epoch,
            "total_epochs": len(sampled_epochs),
            "time_total": elapsed,
            "time_avg_epoch": avg_epoch_time,
            "best_train_loss": min(losses_train) if losses_train else 0,
            "best_dev_loss": min(losses_dev) if losses_dev else 0
        }
        all_results.append(results)
        pd.DataFrame(all_results).to_csv("final_results_2B_all_configs.csv", index=False)
        
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
    print("Final Table 2B - ALL CONFIGURATIONS")
    print("="*160)
    print(f"{'Model':<14} {'LR':<10} {'Batch':<8} {'Dropout':<10} "
        f"{'Slot F1':<10} {'Slot P':<10} {'Slot R':<10} "
        f"{'Intent Acc':<12} {'Intent F1':<10} "
        f"{'Best Ep':<8} {'Params (M)':<12} {'Time (s)':<10}")
    print("-"*160)

    # Stampa TUTTE le configurazioni dalla lista
    for result in all_results:
        params_m = result['params_total'] / 1_000_000
        print(f"{result['model']:<14} {result['lr']:<10.0e} {result['batch_size']:<8} {result['dropout']:<10} "
            f"{result['slot_f1']:<10.3f} {result['slot_p']:<10.3f} {result['slot_r']:<10.3f} "
            f"{result['intent_acc']:<12.3f} {result['intent_f1']:<10.3f} "
            f"{result['best_epoch']:<8} {params_m:<12.2f} {result['time_total']:<10.1f}")

    print("="*160)

    # Trova il miglior modello per Slot F1
    best_result = max(all_results, key=lambda x: x['slot_f1'])

    # Salva CSV con TUTTI i risultati
    df_csv = pd.DataFrame(all_results)
    df_csv.to_csv("final_results_2B_all_configs.csv", index=False)
    print(f"\n✅ CSV saved: final_results_2B_all_configs.csv ({len(all_results)} configurations)")

    print("\n🏆 Best Model (Slot F1):")
    print(f"  Model={best_result['model']}, LR={best_result['lr']:.0e}, Batch={best_result['batch_size']}, Dropout={best_result['dropout']}")
    print(f"  Slot F1={best_result['slot_f1']:.3f} | Slot P={best_result['slot_p']:.3f} | Slot R={best_result['slot_r']:.3f}")
    print(f"  Intent Acc={best_result['intent_acc']:.3f} | Intent F1={best_result['intent_f1']:.3f}")
    print(f"  Parameters: {best_result['params_total']/1_000_000:.2f}M")
    print(f"  Time: {best_result['time_total']:.1f}s")

if __name__ == "__main__":
    experiment_2b()