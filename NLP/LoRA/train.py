from lora import apply_LoRA_tinyllama, configure_optimizers
from utils import DataCollator, get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    batch_size:int
    grad_accum_steps:int
    epochs:int
    warmup_epochs:int
    compile:bool
    weight_decay:float
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
    args = parser.parse_args()
    return args

def log_results(csv_file, metrics_list):
    with open(csv_file, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics_list)
    

def train():
    #Config stuff
    args = parse_args()
    cfg = load_config(args.yaml_config)
    train_cfg = cfg.training
    model_cfg = cfg.model
    print(f"Config : {cfg}")
    
    # Metrics logs stuff
    # create directory
    log_dir = r"LoRA/logs"
    today = strftime("%Y-%m-%d")
    i = 0
    run_dir = os.path.join(log_dir, today, f"run{i}")
    while os.path.exists(run_dir):
        i += 1
        run_dir = os.path.join(log_dir, today, f"run{i}")
    os.makedirs(run_dir)
    # create csv files for the losses
    val_csv_file = os.path.join(run_dir, "val_log.csv")
    with open(val_csv_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "step", "tokens_trained_on", "acc", "loss", "dt"])
    train_csv_file = os.path.join(run_dir, "train_log.csv")
    with open(train_csv_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "step", "loss", "lr", "grad_norm", "dt", "tokens", "batch_per_sec"])
    # save the yaml config used
    with open(os.path.join(run_dir, "config.yaml"), "w") as file:
        yaml.dump(asdict(cfg), file)
    print(f"Logs saved at {run_dir}!")

    #model and tokenizer
    device = torch.device("cuda")
    device_type = "cuda"
    print(f"Loading {model_cfg.name} ...")
    model = apply_LoRA_tinyllama(target_layers=model_cfg.target_layers, r=model_cfg.r)
    tokenizer = get_tokenizer()
    model.resize_token_embeddings(len(tokenizer))
    assert model.lm_head.weight.shape == model.model.embed_tokens.weight.shape
    model.to(device)
    # Compiling the model accelerates the computations, but takes some time to compile at first. Only avaiable with cuda and python<3.12
    if train_cfg.compile:
        print("Compiling...")
        model = torch.compile(model)
    optimizer = configure_optimizers(model, weight_decay=train_cfg.weight_decay, learning_rate=train_cfg.max_lr, betas=(train_cfg.beta1, train_cfg.beta2), device_type=device_type)
    loss_fn = nn.CrossEntropyLoss()

    #dataset
    print("Loading dataset...")
    dataset = load_dataset("tuetschek/e2e_nlg")
    data_collator = DataCollator(tokenizer=tokenizer, max_length=train_cfg.max_length, device=device)
    train_loader = DataLoader(dataset["train"], batch_size=train_cfg.batch_size, collate_fn=data_collator)
    val_loader = DataLoader(dataset["validation"], batch_size=train_cfg.batch_size, collate_fn=data_collator)

    # Learning rate scheduling
    N = len(dataset["train"])
    warmup_steps = train_cfg.warmup_epochs * N / (train_cfg.batch_size * train_cfg.grad_accum_steps)
    max_steps = train_cfg.epochs * N / (train_cfg.batch_size * train_cfg.grad_accum_steps)

    def get_lr(step):
        '''from https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py'''
        # 1) linear warmup for warmup_iters steps
        if step < warmup_steps:
            return train_cfg.max_lr * step / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if step > max_steps:
            return train_cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + cos(pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return train_cfg.min_lr + coeff * (train_cfg.max_lr - train_cfg.min_lr)
    

    print("Begin Training")
    print("="*100)
    for epoch in range(1, train_cfg.epochs+1):

        model.train()
        train_loss = 0.
        step = 1
        tokens_processed = 0
        t0 = time()
        optimizer.zero_grad()
        for it, batch in enumerate(train_loader, start=1):
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            logits = model(input_ids, attention_mask)["logits"]
            lmask = batch["loss_mask"]
            labels = batch["labels"][lmask]
            loss = loss_fn(logits[lmask], labels) / train_cfg.grad_accum_steps
            loss.backward()

            train_loss = loss.item()
            tokens_processed += labels.size(0)
            modulo = it % train_cfg.grad_accum_steps
            if modulo == 0:
                step += 1
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm)
                lr = get_lr(step)
                optimizer.step()
                optimizer.zero_grad()
                dt = time()-t0
                batch_per_sec = train_cfg.batch_size * train_cfg.grad_accum_steps / dt
                # step-wise metrics (not micro batch-wise)
                print(f"Epoch: {epoch:2d} | Step: {step:4d} | loss:{train_loss:.4f} | lr:{lr:.6f} | grad norm: {norm.item():.4f} | step dt: {dt:.4f}s | tokens processed {tokens_processed:4d} | batch/sec {batch_per_sec:.4f}")
                log_results(train_csv_file, [epoch, step, train_loss, lr, norm, dt, tokens_processed, batch_per_sec])
                t0 = time()
        # optimizer.step the remaining steps
        if modulo != 0:
            step += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.max_grad_norm)
            lr = get_lr(step)
            optimizer.step()
            optimizer.zero_grad()
            dt = time() - t0
            batch_per_sec = train_cfg.batch_size * train_cfg.grad_accum_steps / dt
            print(f"Epoch: {epoch:2d} | Step: {step:4d} | loss:{train_loss:.4f} | lr:{lr:.4f} | grad norm: {norm:.4f} | step dt: {dt:.4f}s | tokens processed: {tokens_processed:4d} | batch/sec {batch_per_sec:.4f}")
            log_results(train_csv_file, [epoch, step, train_loss, lr, norm, dt, tokens_processed, batch_per_sec])
        
        model.eval()
        val_loss = 0.
        correct = 0
        processed = 0
        with torch.no_grad():
            for it, batch in enumerate(val_loader, start=1):
                t0 = time()
                input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
                logits = model(input_ids, attention_mask)
                lmask = batch["loss_mask"]
                labels = batch["labels"][lmask]
                loss = loss_fn(logits[lmask], labels)
                val_loss = (val_loss * (it-1) + loss.item()) / it
                correct += torch.sum(logits[lmask].argmax(dim=1) == labels).item()
                processed += len(labels)
                acc = correct / processed
                dt = time() - t0
                # dataset-wise metrics updated sequentially
                print(f"Epoch: {epoch}| Step: {step:4d} | acc: {acc*100:.2f} | loss:{val_loss:.4f} | eval dt: {dt:.4f} | tokens processed {processed:4d}")
            log_results(val_csv_file, [epoch, step, tokens_processed, acc, val_loss, dt])
        import code;code.interact(local=locals())
            

if __name__ == "__main__":
    train()


