from lora import apply_LoRA_tinyllama, configure_optimizers, save_AB_weights_tinyllama
from data_utils import DataCollator, get_tokenizer, preprocess_fn, CustomDataCollator, CustomDataLoader

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from time import time, strftime
import yaml
import argparse
from dataclasses import dataclass, asdict
from math import pi, cos
import os
import csv


@dataclass
class TrainingCfg:
    precision:str
    batch_size:int
    grad_accum_steps:int
    epochs:float
    warmup_epochs:float
    epoch_per_eval:float
    use_compile:bool
    use_autocast:bool
    weight_decay:float
    use_lr_scheduler:bool
    max_lr:float
    min_lr:float
    beta1:float
    beta2:float
    max_grad_norm:float
    max_length:int

@dataclass
class ModelCfg:
    name:str
    r:int
    alpha:float
    target_layers:list[str]

@dataclass
class Cfg:
    model: ModelCfg
    training: TrainingCfg

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    return Cfg(
        model=ModelCfg(**config_dict['model']),
        training=TrainingCfg(**config_dict['training']),
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', type=str, default=r"LoRA/cfg.yaml", help="yaml config path")
    parser.add_argument('--log_dir', type=str, default="LoRA/logs", help="logs directory")
    args = parser.parse_args()
    return args

def log_results(csv_file, metrics_list):
    with open(csv_file, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics_list)

def evaluate(model, loader:CustomDataLoader, loss_fn:nn.Module, use_autocast=True, device_type="cuda"):
    model.eval()
    val_loss = 0.
    correct = 0
    processed = 0
    t0 = time()
    with torch.no_grad():
        for i in range(1, len(loader)+1):
            batch = loader.next_batch()
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            if use_autocast:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = model(input_ids, attention_mask, use_cache=False)["logits"]
            else:
                logits = model(input_ids, attention_mask, use_cache=False)["logits"]
            labels = batch["labels"]
            lmask = batch["loss_mask"]
            loss = loss_fn(logits[lmask], labels)
            val_loss = (val_loss * (i-1) + loss.item()) / i
            correct += torch.sum(logits[lmask].argmax(dim=1) == labels).item()
            processed += len(labels)
            acc = correct / processed
            dt = time() - t0
            # validation dataset-wise metrics updated sequentially
            print(f"acc: {acc*100:.2f}% | loss:{val_loss:.4f} | eval dt: {dt:.4f}")
    return acc, val_loss, dt
    
def train():
    #Config stuff
    args = parse_args()
    cfg = load_config(args.yaml_config)
    train_cfg = cfg.training
    model_cfg = cfg.model
    print(f"Train config : {train_cfg}")
    print(f"Model config {model_cfg}")
    
    # Metrics logs stuff
    # create directory
    log_dir = args.log_dir
    today = strftime("%Y-%m-%d")
    i = 0
    run_dir = os.path.join(log_dir, today, f"run{i}")
    while os.path.exists(run_dir):
        i += 1
        run_dir = os.path.join(log_dir, today, f"run{i}")
    os.makedirs(run_dir)
    # create csv files for the losses
    val_csv_file = os.path.join(run_dir, "val_log.csv")
    with open(val_csv_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "step", "tokens_trained_on", "acc", "loss", "dt"])
    train_csv_file = os.path.join(run_dir, "train_log.csv")
    with open(train_csv_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "step", "tokens", "loss", "lr", "grad_norm", "dt", "tok_per_sec"])
    # save the yaml config used
    with open(os.path.join(run_dir, "config.yaml"), "w") as file:
        yaml.dump(asdict(cfg), file)
    print(f"Logs saved at {run_dir}!")

    #model and tokenizer
    device = torch.device("cuda")
    device_type = "cuda"
    print(f"Loading {model_cfg.name} ...")
    tokenizer = get_tokenizer()
    model = apply_LoRA_tinyllama(target_layers=model_cfg.target_layers, r=model_cfg.r, new_vocsize=len(tokenizer))
    assert model.lm_head.weight.shape == model.model.embed_tokens.weight.shape
    model.to(device)

    # Compiling the model accelerates the computations, but takes some time to compile at the 1st step. 
    # Only avaiable with cuda and python<3.12
    if train_cfg.use_compile:
        print("Compiling..."); t0 = time()
        model = torch.compile(model)
        print(f"Compiled in {time()-t0:.2f}s, note: the first iteration will take longer")
    optimizer = configure_optimizers(model, weight_decay=train_cfg.weight_decay, learning_rate=train_cfg.max_lr, betas=(train_cfg.beta1, train_cfg.beta2), device_type=device_type)
    loss_fn = nn.CrossEntropyLoss()
    # increases the computation speed
    torch.set_float32_matmul_precision(train_cfg.precision)

    #dataset
    print("Loading dataset...")
    dataset = load_dataset("tuetschek/e2e_nlg")
    print("Preprocessing the dataset...")
    dataset = dataset.map(preprocess_fn, fn_kwargs={"tokenizer":tokenizer})
    # data_collator = DataCollator(pad_token_id=tokenizer.pad_token_id, max_length=train_cfg.max_length, device=device)
    # train_loader = DataLoader(dataset["train"], batch_size=train_cfg.batch_size, collate_fn=data_collator) # type: ignore
    # val_loader = DataLoader(dataset["validation"], batch_size=train_cfg.batch_size, collate_fn=data_collator) # type: ignore
    collate_fn = CustomDataCollator(pad_token_id=tokenizer.pad_token_id, max_length=train_cfg.max_length, device=device)
    train_loader = CustomDataLoader(dataset["train"], batch_size=train_cfg.batch_size, collate_fn=collate_fn) # type: ignore
    val_loader = CustomDataLoader(dataset["validation"], batch_size=train_cfg.batch_size, collate_fn=collate_fn) # type: ignore

    # Learning rate scheduling
    N = len(dataset["train"]) # type: ignore
    steps_per_epoch = N / (train_cfg.batch_size * train_cfg.grad_accum_steps)
    warmup_steps = train_cfg.warmup_epochs * steps_per_epoch
    max_steps = round(train_cfg.epochs * steps_per_epoch)
    steps_per_eval = round(train_cfg.epoch_per_eval * steps_per_epoch)
    print(f"warmup steps: {warmup_steps:.2f} | max_steps {max_steps} | Steps per epoch: {steps_per_epoch:.2f} | Steps per eval: {steps_per_eval:.2f}")

    def get_lr(step):
        if train_cfg.use_lr_scheduler:
            '''from https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py'''
            # 1) linear warmup for warmup_iters steps
            if step < warmup_steps:
                return train_cfg.max_lr * step / warmup_steps
            # 2) after, use cosine decay down to min learning rate
            decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + cos(pi * decay_ratio)) # coeff starts at 1 and goes to 0
            return train_cfg.min_lr + coeff * (train_cfg.max_lr - train_cfg.min_lr)
        else:
            return train_cfg.max_lr
    
    print("Begin Training")
    print("="*100)
    start = time()
    step = 0
    tokens_processed = 0
    last_tokens_processed = tokens_processed
    for step in range(1, max_steps+1):
        model.train()
        train_loss = 0.
        t0 = time()
        optimizer.zero_grad()
        # gradient accumulation loop: perform grad_accum_steps loss.backward() before doing optimizer.step()
        for i in range(1, train_cfg.grad_accum_steps+1):
            batch = train_loader.next_batch()
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            if train_cfg.use_autocast:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = model(input_ids, attention_mask, use_cache=False)['logits']
            else:
                logits = model(input_ids, attention_mask, use_cache=False)['logits']
            labels = batch["labels"]
            loss = loss_fn(logits[batch["loss_mask"]], labels) / train_cfg.grad_accum_steps
            loss.backward()
            train_loss += loss.item()
            tokens_processed += labels.size(0)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm).item()
        lr = get_lr(step) * model_cfg.alpha / model_cfg.r
        optimizer.step()
        optimizer.zero_grad()
        dt = time()-t0
        tok_per_sec = (tokens_processed - last_tokens_processed) / dt
        epoch = step/steps_per_epoch
        # step-wise metrics
        print(f"Epoch: {epoch:2f} | Step: {step:4d} | tokens processed {tokens_processed:8d} | loss:{train_loss:.4f} | lr:{lr:.6f} | grad norm: {norm:.4f} | step dt: {dt:.4f}s | tok/sec {tok_per_sec:.4f}")
        log_results(train_csv_file, [epoch, step, tokens_processed, train_loss, lr, norm, dt, tok_per_sec])
        last_tokens_processed = tokens_processed
        t0 = time()
        # import code; code.interact(local=locals())

        if step % steps_per_eval == 0:
            acc, val_loss, dt = evaluate(model, val_loader, loss_fn, train_cfg.use_autocast, device_type)
            log_results(val_csv_file, [epoch, step, tokens_processed, acc, val_loss, dt])
    
    end = time() - start
    print(f"Total time : {end:.4f} | total tokens processed {tokens_processed:8d} | ratio: {tokens_processed/end:.4f}")
    print("Saving model...")
    save_AB_weights_tinyllama(os.path.join(run_dir, "model_weights"), model, model_cfg.target_layers)
    print("Saved!")
            

if __name__ == "__main__":
    train()


