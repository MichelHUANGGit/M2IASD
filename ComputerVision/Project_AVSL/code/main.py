import torch
from torchvision import transforms
from dataset import CUB_dataset, CUB_dataset_Test
# from models import AVSL
from model import AVSL_Similarity
from train import train
from inference import validate, infer_queries, get_predictions
import argparse


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
        # transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
    parser = argparse.ArgumentParser(description="Parse training parameters")

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size_training", type=int, default=100)
    parser.add_argument("--batch_size_inference", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_channels", type=int, nargs='+', default=[512, 1024, 2048], help="Output channels for each layer")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=30)
    parser.add_argument("--use_proxy", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=128, help="topk value AVSL")
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--p", type=int, default=2, help="norm degree for embedding distance")
    parser.add_argument("--CNN_coeffs", type=float, nargs=2, default=(32, 0.1), help="Coefficients for CNN loss")
    parser.add_argument("--sim_coeffs", type=float, nargs=2, default=(32, 0.1))
    parser.add_argument("--validate_on_train", action="store_true")
    parser.add_argument("--validate_on_val", action="store_true")
    parser.add_argument("--infer_gallery_to_queries", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--train_model", action="store_true", help="Whetheer to train the model")
    parser.add_argument("--model_path", type=str, default=None, help="if pretrained, takes the pretrained model path")
    parser.add_argument("--name", type=str, default="AVSL_v3", help="Base name")
    parser.add_argument("--metrics_K", type=int, nargs='+', default=[1, 2, 4, 8], help="Values of K for recall and precision")

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['device'] = torch.device(args_dict['device'])
    
    main(**args_dict)