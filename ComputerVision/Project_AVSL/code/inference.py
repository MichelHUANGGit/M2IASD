import torch
from torchvision import transforms
from tqdm import tqdm
from dataset import CUB_dataset, CUB_dataset_Test
from torch.utils.data import DataLoader
import os
import pandas

def precision(K, matrix:torch.tensor, labels:torch.tensor):
    '''expects pair-wise similarity matrix'''
    topK_closest_images = torch.topk(matrix, K, dim=1, largest=False).indices #shape (N,K)
    # For the N images, take the labels of K closest images
    topK_closest_labels = torch.zeros_like(topK_closest_images, dtype=torch.int8)
    for i in range(topK_closest_images.size(0)):
        for j in range(topK_closest_images.size(1)):
            topK_closest_labels[i,j] = labels[topK_closest_images[i,j]]
    labels_extended = labels.repeat(K,1).T #shape (N,K)
    correctly_retrieved = torch.eq(topK_closest_labels, labels_extended).to(torch.float64) #shape(N,K)
    return correctly_retrieved.mean().item()

def recall(K, matrix:torch.tensor, labels:torch.tensor):
    '''expects pair-wise similarity matrix'''
    topK_closest_images = torch.topk(matrix, K, dim=1, largest=False).indices #shape (N,K)
    # For the N images, take the labels of K closest images
    topK_closest_labels = torch.zeros_like(topK_closest_images, dtype=torch.int8)
    for i in range(topK_closest_images.size(0)):
        for j in range(topK_closest_images.size(1)):
            topK_closest_labels[i,j] = labels[topK_closest_images[i,j]]
    labels_extended = labels.repeat(K,1).T #shape (N,K)
    at_least_one_retrieved = (torch.sum(labels_extended == topK_closest_labels, dim=1) > 0).to(torch.float64) # shape(N) of bools
    return torch.mean(at_least_one_retrieved).item()

def infer_gallery(
        model,
        dataset,
        batch_size,
        device,
    ):
    
    N = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    similarity_matrix = torch.zeros(size=(N,N), dtype=torch.float32)
    # print(similarity_matrix.shape)

    # ==================== Inference ====================
    # leverage the symmetry of the similarity to only compute the upper diagonal and thus avoid redundant computations
    model.eval()
    with torch.no_grad():
        for i, batch1 in tqdm(enumerate(loader)):
            for j, batch2 in tqdm(enumerate(loader)):
                '''Computing roughly the upper triangle. This is not exactly the upper triangle.
                we can think of the batch_size forming a super-pixel moving down the diagonal of an image, if the super-pixel (batch)
                is much bigger than one pixel (datapoint) there will be some elements in the lower triangle computed, but that's ok.'''
                if j>=i:
                    images1 = batch1["image"].to(device)
                    images2 = batch2["image"].to(device)
                    # First and last ids of the batches, they're just used to slice the matrix
                    row_first, row_last = batch1["id"][0], batch1["id"][-1]+1
                    col_first, col_last = batch2["id"][0], batch2["id"][-1]+1
                    # Compute the similarity
                    similarity_matrix[row_first:row_last, col_first:col_last] = model(images1, None, images2).get("ovr_sim")

    # Only keep the upper triangle, the elements on the diagonal should be 0
    similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
    # Add the lower triangle
    similarity_matrix += similarity_matrix.clone().T

    return similarity_matrix.cpu()

def validate(model, dataset, batch_size, device, metrics_K, save_matrix=False, name="train", save_dir=None):
    print(f"Measuring Recall @ K on {name} dataset")
    labels = torch.tensor(dataset.labels, dtype=torch.int8)
    similarities = infer_gallery(model, dataset, batch_size, device)
    if save_matrix:
        save_path = os.path.join(save_dir, f"{name}_similarities.pt")
        torch.save(similarities, save_path)
        print(f"Saved {name} similarity matrix!")
    for metric_K in metrics_K:
        print("Precision @ %d"%metric_K, precision(metric_K, similarities, labels))
        print("Recall @ %d"%metric_K, recall(metric_K, similarities, labels))

def infer_queries(
        model,
        gallery_dataset,
        batch_size,
        query_dataset,
        device,
    ):
    # ==================== Initialization ====================
    N, M = len(gallery_dataset), len(query_dataset)
    gallery_loader = DataLoader(gallery_dataset, batch_size, shuffle=False, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size, shuffle=False, pin_memory=True)
    # Measure pairwise similarity for every (gallery/query) pair
    similarity_matrix = torch.zeros(size=(N,M))

    # ==================== Inference ====================
    model.eval()
    with torch.no_grad():
        for batch1 in tqdm(gallery_loader):
            for batch2 in tqdm(query_loader):
                images1 = batch1["image"].to(device)
                images2 = batch2["image"].to(device)
                # First and last ids of the batches, they're just used to slice the matrix
                row_first, row_last = batch1["id"][0], batch1["id"][-1]+1
                col_first, col_last = batch2["id"][0], batch2["id"][-1]+1
                # Compute the similarity
                similarity_matrix[row_first:row_last, col_first:col_last] = model(images1, None, images2).get("ovr_sim")

    return similarity_matrix.cpu()

def get_predictions(similarity_matrix, gallery_labels, K, test_paths, save_dir):
    topk_closest_images = torch.topk(similarity_matrix, K, dim=0, largest=False).indices
    topk_closest_labels = torch.zeros_like(topk_closest_images)
    for i in range(topk_closest_images.size(0)):
        for j in range(topk_closest_images.size(1)):
            topk_closest_labels[i,j] = gallery_labels[topk_closest_images[i,j]]
    predictions = torch.mode(topk_closest_labels, dim=0).values
    submissions = []
    for i in range(topk_closest_images.size(1)):
        submissions.append([test_paths[i][17:].split(".jpg")[0], predictions[i].item()])

    submissions = pandas.DataFrame(submissions, index=None)
    submissions.rename(columns={0:"ID",1:"Category"}, inplace=True)
    save_path = os.path.join(save_dir, "submissions@%d.csv"%K)
    submissions.to_csv(save_path, sep=",", index=False)
    


if __name__ == "__main__":
    args = {
        "batch_size":32,
        "device":torch.device("cuda")
    }
    # ==================== Datasets ====================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    gallery_dataset = CUB_dataset(
        root_dir="data/train_images",
        class_index_file='data/class_indexes.csv',
        transform=transform,
        return_id=True,
    )
    gallery_image_paths = gallery_dataset.image_paths.copy()
    gallery_labels = gallery_dataset.labels.copy()

    query_dataset = CUB_dataset_Test(
        root_dir="data/test_images",
        transform=transform,
        return_id=True,
        gallery_length=0
    )
    model = torch.load("model_test.pt", args["device"])

    # ==================== Inference ====================
    # Computing gallery
    similarity_matrix = infer_gallery(model, gallery_dataset, labels=gallery_labels, **args)
    torch.save(similarity_matrix, "gallery_similarities.pt")
    print(similarity_matrix.shape)
    
    # ==================== Metrics ====================
    labels = torch.tensor(gallery_labels, dtype=torch.int8)
    metrics_K = [1,2,4,8]
    for metric_K in metrics_K:
        print("Average Precision @ %d" %metric_K, precision(metric_K, similarity_matrix, labels))
        print("Average Recall @ %d" %metric_K, recall(metric_K, similarity_matrix, labels))

    # Query images (test)
    test_similarity_matrix = infer_queries(model, gallery_dataset, args["batch_size"], query_dataset, args["batch_size"], args["device"])
    torch.save(test_similarity_matrix, "gallery_query_similarity.pt")
    print(test_similarity_matrix.shape)



