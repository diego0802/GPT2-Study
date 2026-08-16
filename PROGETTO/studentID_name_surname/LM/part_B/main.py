import torch
import time
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel
from model import CustomGPT2Attention
from utils import download_dataset, get_dataloaders, save_report
from functions import eval_loop_1b, train_model_1b

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        batch_size=32,
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

        # Epoche in base al rank
        if config["rank"] >= 16:
            n_epochs = 30
            patience = 3
        elif config["rank"] >= 8:
            n_epochs = 25
            patience = 4
        else:
            n_epochs = 20
            patience = 3
        
        # Training
        best_model, best_ppl, losses_train, losses_dev = train_model_1b(
            model, train_loader, dev_loader, tokenizer,
            lr=config["lr"],
            n_epochs=1,      # breve su Colab, su Azure puoi aumentare
            patience=patience,
            device=DEVICE
        )
        
        # Test
        test_ppl, test_loss, test_acc = eval_loop_1b(test_loader, best_model, tokenizer)
        elapsed = time.time() - start_time
        
        # Salva risultati
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
            
        # Tabella finale
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
        
        df_final = pd.DataFrame(results)
        df_final.to_csv("final_results.csv", index=False)
        print("✅ Tabella finale salvata in 'final_results.csv'")

if __name__ == "__main__":
    experiment_1b()