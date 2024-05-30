import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetDecomposed(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.layers = [self.model.layer2, self.model.layer3, self.model.layer4]
        self.n_layers = len(self.layers)
        self.output_channels = [self.layers[i].get_submodule("0.conv3").out_channels for i in range(self.n_layers)] # [512, 1024, 2048]


    def forward(self, x):
        x = self.model.conv1(x) # B x 64 x 112 x 112
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) # B x 64 x 56 x 56
        x = self.model.layer1(x) # B x 256 x 56 x 56
        # store feature maps of interest
        feature_maps = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            feature_maps.append(x.clone())

        return feature_maps
    
if __name__ == "__main__":
    
    model = ResNetDecomposed()
    print(model)
    B, C, H, W = 16,3,256,256
    input = torch.randn(size=(B,C,W,H))
    feature_maps = model(input)
    for feature_map in feature_maps:
        print(feature_map.shape)