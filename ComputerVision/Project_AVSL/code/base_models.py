import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

class ResNet50_(nn.Module):

    def __init__(self, lay_to_emb_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.pre_layers = [
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
        ]
        self.pre_layers += [self.model.get_submodule("layer%d"%i) for i in range(1, min(lay_to_emb_ids))]
        self.layers = [self.model.get_submodule("layer%d"%i) for i in lay_to_emb_ids]
        self.model.avgpool = PlaceHolderLayer()
        self.model.fc = PlaceHolderLayer()
        self.layers_output_channels = {1:256, 2:512, 3:1024, 4:2048}

    def forward(self, x, l):
        '''computes the feature map of the base_model at a certain layer. At the first layer, computes it starting from the image/
        At layer 2, use the feature map of layer1 to save some computation time. And so on and so on ...'''
        if l==0:
            for layer in self.pre_layers:
                x = layer(x)
        return self.layers[l](x)
    

class EfficientNet_v2_S_(nn.Module):

    def __init__(self, lay_to_emb_ids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.pre_layers = [self.model.get_submodule("features.%d"%i) for i in range(0, min(lay_to_emb_ids))]
        self.layers = [self.model.get_submodule("features.%d"%i) for i in lay_to_emb_ids]
        self.model.avgpool = PlaceHolderLayer()
        self.model.classifier = PlaceHolderLayer()
        self.layers_output_channels = {0:24, 1:24, 2:48, 3:64, 4:128, 5:160, 6:256, 7:1280}

    def forward(self, x, l):
        '''computes the feature map of the base_model at a certain layer. At the first layer, computes it starting from the image/
        At layer 2, use the feature map of layer1 to save some computation time. And so and so on ...'''
        if l==0 and len(self.pre_layers)>0:
            for layer in self.pre_layers:
                x = layer(x)
        return self.layers[l](x)


class PlaceHolderLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return None
    
if __name__ == "__main__":

    #Goal : Make sure LAYER_ID's output feature map corresponds to what we want
    LAYER_ID = 3
    B,C,H,W = 16,3,256,256
    device = torch.device("cuda")
    random_images = torch.randn(size=(B,C,H,W)).to(device)

    def fmap_hook(self, input, output):
        fmap = output.clone()
        self.register_buffer("fmap", fmap)

    # Normal model
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    layer_hooked = resnet.get_submodule(f"layer{LAYER_ID}")
    layer_hooked.register_forward_hook(fmap_hook)

    # Forward pass to get the hook going
    resnet(random_images)
    
    #Custom model
    resnet_ = ResNet50_(lay_to_emb_ids=[1,2,3,4]).to(device)
    x = random_images
    for l in range(3):
        x = resnet_(x, l)
    # Verify equality
    print(torch.norm((layer_hooked.fmap - x)))

    # Same thing for effnet
    LAYER_ID = 1
    effnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).to(device)
    layer_hooked_ = effnet.get_submodule(f"features.{LAYER_ID}")
    layer_hooked_.register_forward_hook(fmap_hook)
    random_images = torch.randn(size=(B,C,H,W)).to(device)
    effnet(random_images)
    effnet_ = EfficientNet_v2_S_(lay_to_emb_ids=[0,1,2,3,4,5,6,7]).to(device)
    x = random_images
    for l in range(2):
        x = effnet_(x,l)
    print(layer_hooked_.fmap.shape)
    print(x.shape)
    print(torch.norm((layer_hooked_.fmap - x)))
        

