import torch
from torchvision import transforms
from tqdm import tqdm
from dataset import CUB_dataset, CUB_dataset_Test
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

def precision(K, matrix:torch.tensor, labels:torch.tensor):
    topK_image_classes = torch.topk(matrix, K, dim=1, largest=False).indices #shape (N,K)
    labels_extended = labels.repeat(K,1).T #shape (N,K)
    correctly_retrieved = torch.eq(topK_image_classes, labels_extended).to(torch.float64) #shape(N,K)
    return correctly_retrieved.mean().item()

def recall(K, matrix:torch.tensor, labels:torch.tensor):
    topK_image_classes = torch.topk(matrix, K, dim=1, largest=False).indices
    labels_extended = labels.repeat(K,1).T #shape (N,K)
    at_least_one_retrieved = (torch.sum(labels_extended == topK_image_classes, dim=1) > 0).to(torch.float64) # shape(N) of bools
    # print(at_least_one_retrieved.shape)
    return torch.mean(at_least_one_retrieved, dim=0).item()

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
                    similarity_matrix[row_first:row_last, col_first:col_last] = model.forward(images1, None, images2)

    # Only keep the upper triangle, the elements on the diagonal should be 0
    similarity_matrix = torch.triu(similarity_matrix, diagonal=1)
    # Add the lower triangle
    similarity_matrix += similarity_matrix.clone().T

    return similarity_matrix

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
                similarity_matrix[row_first:row_last, col_first:col_last] = model.forward(images1, None, images2)

    return similarity_matrix

def KNN(similarity_matrix, gallery_labels, K):
    # Number of query images
    M = similarity_matrix.shape[1]
    similarity_matrix = np.array(similarity_matrix)
    gallery_labels = np.array(gallery_labels)
    unique_labels = np.unique(gallery_labels)
    num_classes = len(unique_labels)

    # Initialize the array to store the probabilities
    probabilities = np.zeros((M, num_classes))

    # Iterate over each query image
    for i in range(M):
        # Get the similarity scores for the i-th query image
        similarities = similarity_matrix[:, i]

        # Find the indices of the top K most similar gallery images
        top_k_indices = np.argsort(similarities)[-K:]

        # Retrieve the labels of the top K gallery images
        top_k_labels = gallery_labels[top_k_indices]

        # Count the occurrences of each label among the top K labels
        label_counts = Counter(top_k_labels)

        # Compute the probability distribution
        for label, count in label_counts.items():
            probabilities[i, label] = count / K

    return probabilities

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

    # Infer classes on test set :
    K_neighbors = 20
    class_probabilities = KNN(test_similarity_matrix, gallery_labels, K_neighbors)
    class_probabilities.tofile("AVSL_KNN_v0.csv",sep=";")




