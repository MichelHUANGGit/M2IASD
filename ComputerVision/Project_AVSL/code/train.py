import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import Proxy_AVSL_Loss, AVSL_ContrastiveLoss
from model import AVSL_Similarity
from dataset import CUB_dataset
import pandas
import os

def train(
        model,
        train_dataset,
        val_dataset,
        n_layers,
        epochs,
        lr, 
        batch_size, 
        device,
        CNN_coeffs=(32,0.1),
        sim_coeffs=(32,0.1),
        margin=1.0,
        save_dir=".",
        model_name="AVSL",
    ) -> None:

    # ============== Initialization =================
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    # loss_fn = Proxy_AVSL_Loss(n_layers, *CNN_coeffs, *sim_coeffs)
    loss_fn = AVSL_ContrastiveLoss(n_layers, margin)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    losses = {"train":[],"val":[]}
    
    print("Start Training")
    # FIX LOSS IN PROGRESS BAR
    # FIX SAVING TO CSV, ONLY ONE VALUE
    for epoch in range(1, epochs+1):
        # ========================== Training ===========================
        # Train the linear projection of the graph model
        print("Epoch %d" %epoch)
        model.train()
        train_loss = 0.
        tqdmloader = tqdm(train_loader, unit="batch")
        for i, batch in enumerate(tqdmloader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            output_dict = model(images, labels)
            loss = loss_fn(output_dict)
            loss.backward()
            optimizer.step()
            train_loss = (train_loss * i * batch_size + loss.detach().cpu().item())/ ((i+1) * batch_size)
            tqdmloader.set_description("Train loss: %.5f" %train_loss)
        losses["train"].append(train_loss)
        # ========================== Validation ===========================
        # Validation on the Proxy Anchor Loss (hence no model.eval())
        with torch.no_grad():
            val_loss = 0.
            tqdmloader = tqdm(valid_loader, unit="batch")
            for i, batch in enumerate(tqdmloader):
                images, labels = batch["image"].to(device), batch["label"].to(device)
                output_dict = model(images, labels)
                loss = loss_fn(output_dict)
                val_loss = (val_loss * i * batch_size + loss.detach().cpu().item())/ ((i+1) * batch_size)
                tqdmloader.set_description("Val loss: %.5f" %val_loss)
        losses["val"].append(val_loss)
        # print("Train loss : %0.5f - Validation loss : %0.5f" %(train_loss,val_loss))
    print("Saving model and losses...")
    torch.save(model, os.path.join(save_dir, model_name))
    df = pandas.DataFrame(losses)
    df.to_csv(os.path.join(save_dir, "losses.csv"), sep=";", index=False)
    print("Saved!")

if __name__ == "__main__":
    train_args = {
        "epochs":2,
        "n_layers":3,
        "lr":1e-4,
        "batch_size":30,
        "device":torch.device("cuda"),
        "CNN_coeffs":(32,0.1),
        "sim_coeffs":(32,0.1),
        "margin":1.0,
    }
    model_args = {
        "base_model_name":"ResNet50",
        "lay_to_emb_ids":[2,3,4],
        "num_classes":30,
        "emb_dim":128,
        "topk":32,
        "momentum":0.5,
        "p":2,
        "use_proxy":False,
    }
    
    # ==================== Datasets ====================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CUB_dataset(
        root_dir='data/train_images',
        class_index_file='data/class_indexes.csv',
        transform=transform
    )
    val_dataset = CUB_dataset(
        root_dir='data/val_images',
        class_index_file='data/class_indexes.csv',
        transform=transform
    )

    model = AVSL_Similarity(**model_args).to(train_args["device"])
    train(model, train_dataset, val_dataset, **train_args)
    torch.save(
        model, 
        "test.pt"
    )


