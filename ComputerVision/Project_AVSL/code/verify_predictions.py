import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import pandas
from dataset import CUB_dataset, CUB_dataset_Test
from inference import get_predictions

train_dataset = CUB_dataset(
    root_dir='data/train_images',
    class_index_file='data/class_indexes.csv',
    transform=None,
    return_id=True)
test_dataset = CUB_dataset_Test(
    root_dir="data/test_images",
    transform=None,
    return_id=True,
    gallery_length=0)

gallery_labels = train_dataset.labels
query_paths = test_dataset.image_paths
sim = torch.load("glry_to_query_sim.pt")
K=20
# get_predictions(sim, gallery_labels, K, query_paths)
topk_closest_images = torch.topk(sim, K, dim=0, largest=False).indices
idx_to_class = {idx:cls for cls,idx in train_dataset.class_to_idx.items()}
# print(gallery_labels[1292])
# print(train_dataset.class_to_idx)
# print(idx_to_class)
print(idx_to_class[15])

topk_closest_labels = torch.zeros_like(topk_closest_images)
for i in range(topk_closest_images.size(0)):
    for j in range(topk_closest_images.size(1)):
        topk_closest_labels[i,j] = gallery_labels[topk_closest_images[i,j]]
# print(topk_closest_labels.shape)
# predictions = torch.mode(topk_closest_labels, dim=0).values
# print(predictions)

# Verify a single image predictions
IMAGE_ID = 187
print("Image path :", query_paths[IMAGE_ID])
print("top K similarities :", torch.topk(sim[:,IMAGE_ID], K, largest=False))
print("Top K closest images:", topk_closest_images[:,IMAGE_ID])
print("Top K closests labels:", topk_closest_labels[:,IMAGE_ID])
for label_id in topk_closest_labels[:,IMAGE_ID]:
    print(idx_to_class[label_id.item()])