import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CUB_dataset(Dataset):
    '''class with known labels'''
    def __init__(self, root_dir, class_index_file, transform=None, return_id=False, gallery_image_paths=None, gallery_labels=None):
        self.root_dir = root_dir
        self.transform = transform
        self.return_id = return_id
        
        # Read class index mapping
        class_index_df = pd.read_csv(class_index_file)
        self.class_to_idx = {row['category_cub']: row['idx'] for _, row in class_index_df.iterrows()}
        
        # Gather all image paths and their labels.
        self.image_paths = [] if gallery_image_paths is None else gallery_image_paths
        self.labels = [] if gallery_labels is None else gallery_labels
        self.gallery_length = len(self.labels)
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        output = {"image":image, "label":label}
        if self.return_id:
            output["id"] = idx + self.gallery_length
        return output
    
class CUB_dataset_Test(Dataset):
    '''test dataset with no labels'''
    def __init__(self, root_dir, transform=None, return_id=False, gallery_length=1525):
        self.root_dir = root_dir
        self.transform = transform
        self.return_id = return_id
        self.gallery_length = gallery_length
        
        self.image_paths = []
        for img_name in os.listdir(root_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root_dir, img_name)
                self.image_paths.append(img_path)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        output = {"image":image}
        if self.return_id:
            output["id"] = idx + self.gallery_length
        return output

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CUB_dataset(
        root_dir='data/train_images',
        class_index_file='data/class_indexes.csv',
        transform=transform,
        return_id=True,
    )

    test_dataset = CUB_dataset_Test(
        root_dir="data/test_images",
        transform=transform,
        return_id=True,
        gallery_length=1525
    )
    
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of test samples: {len(test_dataset)}')
    train_sample = train_dataset[0]
    test_sample = test_dataset[5]
    print(f'Training image shape: {train_sample["image"].shape}, Label: {train_sample["label"]} id : {train_sample["id"]}')
    print(f'Training image shape: {test_sample["image"].shape}, id : {test_sample["id"]}')

    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for i,batch in enumerate(loader):
        for k,v in batch.items():
            print(k, v)
        if i>1:
            break
        