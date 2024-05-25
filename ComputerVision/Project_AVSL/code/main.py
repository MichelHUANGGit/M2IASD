import torch
from torchvision import transforms
from dataset import CUB_dataset, CUB_dataset_Test
from models import AVSL
from train import train
from inference import infer_gallery, infer_queries, precision, recall


def main(
        output_channels,
        n_layers,
        emb_dim, 
        num_classes, 
        use_proxy, 
        topk, 
        momentum, 
        p,
        epochs, 
        lr, 
        batch_size_training,
        batch_size_inference,
        device, 
        CNN_coeffs, 
        sim_coeffs,
        metrics_K,
        model_path,
        validate_on_train=False,
        pretrained=False,
    ) -> None:
    # ==================== Datasets ====================
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
    train_dataset = CUB_dataset(
        root_dir='data/train_images',
        class_index_file='data/class_indexes.csv',
        transform=train_transform)
    val_dataset = CUB_dataset(
        root_dir='data/val_images',
        class_index_file='data/class_indexes.csv',
        transform=transform)
    query_dataset = CUB_dataset_Test(
        root_dir="data/test_images",
        transform=transform,
        return_id=True,
        gallery_length=0)
    

    if pretrained:
        torch.load(model_path,device)
    else:
        model = AVSL(output_channels, emb_dim, num_classes, use_proxy, topk, momentum, p).to(device)
        print("Start Training")
        train(model, train_dataset, val_dataset, n_layers, epochs, lr, batch_size_training, device, CNN_coeffs, sim_coeffs)
        print("Saving...")
        torch.save(
            model, 
            f"AVSL\
            -emb{emb_dim}\
            -batch{batch_size_training}\
            -lr{lr}\
            -layers{n_layers}\
            -topk{topk}\
            -m{momentum}.pt"
        )
    
    # =================== Measuring performance ======================
    if validate_on_train:
        print("Measuring Recall @ K on train dataset")
        train_labels = torch.tensor(train_dataset.labels, dtype=torch.int8)
        train_similarities = infer_gallery(model, train_dataset, batch_size_inference, device),
        torch.save(train_similarities,"train_similarities.pt")
        print("Saved train similarity matrix!")
        for metric_K in metrics_K:
            print(precision(metric_K, train_similarities, train_labels))
            print(recall(metric_K, train_similarities, train_labels))

    print("Measuring Recall @ K on validation dataset")
    val_labels = torch.tensor(val_dataset.labels, dtype=torch.int8)
    validation_similarities = infer_gallery(model, val_dataset, batch_size_inference, device),
    torch.save(validation_similarities,"val_similarities.pt")
    print("Saved validation similarity matrix!")
    for metric_K in metrics_K:
        print(precision(metric_K, train_similarities, val_labels))
        print(recall(metric_K, train_similarities, val_labels))

    # ========================= Infering on test set =========================
    gallery_to_query_similarities = infer_queries(model, train_dataset, batch_size_inference, query_dataset, device)
    torch.save(gallery_to_query_similarities, "glry_to_query_sim.pt")


if __name__ == "__main__":
    args = {
        "epochs":25,
        "batch_size_training":30,
        "batch_size_inference":30,
        "emb_dim":128,
        "lr":1e-4,
        "output_channels":[512,1024,2048],
        "n_layers":3,
        "emb_dim":128,
        "num_classes":30,
        "use_proxy":True,
        "device":torch.device("cuda"),
        "topk":32,
        "momentum":0.5,
        "p":2,
        "CNN_coeffs":(32,0.1),
        "sim_coeffs":(32,0.1),
        "validate_on_train":False,
        "pretrained":False,
        "model_path":None,
        "metrics_K":[1,2,4,8],
    }
    main(**args)