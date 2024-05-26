import torch
from torchvision import transforms
from dataset import CUB_dataset, CUB_dataset_Test
# from models import AVSL
from model import AVSL_Similarity
from train import train
from inference import validate, infer_queries, get_predictions


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
        name,
        validate_on_train=False,
        validate_on_val=True,
        infer_gallery_to_queries=True,
        pretrained=False,
        train_model=False,
    ) -> None:
    # ==================== Datasets ====================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.GaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
    train_dataset = CUB_dataset(
        root_dir='data/train_images',
        class_index_file='data/class_indexes.csv',
        transform=train_transform,
        return_id=True)
    val_dataset = CUB_dataset(
        root_dir='data/val_images',
        class_index_file='data/class_indexes.csv',
        transform=transform,
        return_id=True)
    query_dataset = CUB_dataset_Test(
        root_dir="data/test_images",
        transform=transform,
        return_id=True,
        gallery_length=0)
    

    if pretrained:
        print("Loading pretrained model")
        model = torch.load(model_path, device)
    else:
        model = AVSL_Similarity(output_channels, num_classes, use_proxy, emb_dim, topk, momentum, p).to(device)
    if train_model:
        print("Start Training")
        train(model, train_dataset, val_dataset, n_layers, epochs, lr, batch_size_training, device, CNN_coeffs, sim_coeffs)
        print("Saving...")
        torch.save(
            model, 
            f"{name}-emb{emb_dim}-batch{batch_size_training}-lr{lr}-layers{n_layers}-topk{topk}-m{momentum}.pt"
        )
    
    # =================== Measuring performance ======================
    # TO FIX
    if validate_on_train:
        validate(model, train_dataset, batch_size_inference, device, metrics_K, save_matrix=False, name="train")
    if validate_on_val:
        validate(model, val_dataset, batch_size_inference, device, metrics_K, save_matrix=True, name="val")

    # ========================= Infering on test set =========================
    if infer_gallery_to_queries:
        gallery_to_query_similarities = infer_queries(model, train_dataset, batch_size_inference, query_dataset, device)
        torch.save(gallery_to_query_similarities, "glry_to_query_sim.pt")
        for metric_K in metrics_K:
            get_predictions(gallery_to_query_similarities, train_dataset.labels, metric_K, query_dataset.image_paths)


if __name__ == "__main__":
    args = {
        "epochs":25,
        "batch_size_training":100,
        "batch_size_inference":30,
        "lr":1e-4,
        "output_channels":[512,1024,2048],
        "n_layers":3,
        "emb_dim":256,
        "num_classes":30,
        "use_proxy":True,
        "device":torch.device("cuda"),
        "topk":64,
        "momentum":0.5,
        "p":2,
        "CNN_coeffs":(32,0.1),
        "sim_coeffs":(32,0.1),
        "validate_on_train":False,
        "validate_on_val":True,
        "infer_gallery_to_queries":False,
        "pretrained":True,
        "train_model":False,
        "model_path":"AVSL-emb256-batch100-lr0.0001-layers3-topk64-m0.5.pt",
        "name":"AVSL_v2",
        "metrics_K":[1,2,4,8],
    }
    main(**args)