import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import Proxy_AVSL_Loss
from models import AVSL
from model import AVSL_Similarity
from dataset import CUB_dataset

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
    ) -> None:

    # ============== Initialization =================
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    loss_fn = Proxy_AVSL_Loss(n_layers, *CNN_coeffs, *sim_coeffs)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(1, epochs+1):
        # ========================== Training ===========================
        # Train the linear projection of the graph model
        print("Epoch %d" %epoch)
        model.train()
        train_loss = 0.
        for batch in tqdm(train_loader):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            output_dict = model.forward(images, labels)
            loss = loss_fn.forward(output_dict)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()
        print("Train loss :", train_loss)
        # ========================== Validation ===========================
        # Validation on the Proxy Anchor Loss (hence no model.eval())
        with torch.no_grad():
            val_loss = 0.
            for batch in tqdm(valid_loader):
                images, labels = batch["image"].to(device), batch["label"].to(device)
                output_dict = model.forward(images, labels)
                loss = loss_fn.forward(output_dict)
                val_loss += loss.detach().cpu().item()
        # This is the anchor loss
        print("Validation loss :", val_loss)

if __name__ == "__main__":
    train_args = {
        "epochs":20,
        "n_layers":3,
        "lr":1e-4,
        "batch_size":30,
        "device":torch.device("cuda"),
        "CNN_coeffs":(32,0.1),
        "sim_coeffs":(32,0.1)
    }
    model_args = {
        "num_classes":30,
        "output_channels":[512,1024,2048],
        "emb_dim":128,
        "topk":32,
        "momentum":0.5,
        "p":2,
        "use_proxy":True,
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

    model = AVSL(**model_args).to(train_args["device"])
    train(model, train_dataset, val_dataset, **train_args)
    torch.save(
        model, 
        f"AVSL\
        -emb{model_args["emb_dim"]}\
        -batch{train_args["batch_size"]}\
        -lr{train_args["lr"]}\
        -layers{model_args["n_layers"]}\
        -topk{model_args["topk"]}\
        -m{model_args["momentum"]}.pt"
    )


