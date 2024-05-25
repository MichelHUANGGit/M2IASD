import torch
import torch.nn as nn
from torch.nn import functional as F

class AVSL_Graph(nn.Module):

    def __init__(self, output_channels:list[int], n_layers:int, R:int):
        super().__init__()
        self.out_channels = output_channels
        self.R = R
        self.n_layers = n_layers
        for l in range(n_layers):
            setattr(self, f"lin_proj_{l}", nn.Linear(self.out_channels[l],R))
            setattr(self, f"pool_{l}", nn.AdaptiveAvgPool2d(1))

    def get_embeddings(self, feature_maps:list[torch.Tensor]) -> list[torch.Tensor]:
        '''stores the embeddings of the layers of interest'''
        embeddings = []
        # We take a feature map (B,C,H,W) -> pool -> (B,C,1,1) -> squeeze + linear proj -> (B,r)
        for l, feature_map in enumerate(feature_maps):
            embeddings.append(
                getattr(self, f"lin_proj_{l}")(
                    getattr(self, f"pool_{l}")(feature_map).squeeze((-2,-1))
                )
            )
        return embeddings

    def get_CAMs(self, feature_maps:list[torch.Tensor]) -> list[torch.Tensor]:
        '''correponds to u in the paper (also called channel activation map (CAM))'''
        CAMs = []
        # loop through layers of interest
        for l in range(self.n_layers):
            a = getattr(self, f"lin_proj_{l}").weight.view(1,self.out_channels[l],self.R,1,1) #(C,R) -> (1,C,R,1,1)
            z = feature_maps[l].unsqueeze(2) # (B,C,H,W) -> (B,C,1,H,W)
            CAMs.append(torch.sum(a*z, dim=1)) # (1,C,R,1,1) * (B,C,1,H,W) -> (B,C,R,H,W) -> sum(..., dim=1) -> (B,R,H,W)
        return CAMs
    
    def get_CAM(self, feature_map:torch.Tensor, l:int):
        a = getattr(self, f"lin_proj_{l}").weight.view(1,self.out_channels[l],self.R,1,1) #(C,R) -> (1,C,R,1,1)
        z = feature_map.unsqueeze(2) # (B,C,H,W) -> (B,C,1,H,W)
        return torch.sum(a*z, dim=1) # (1,C,R,1,1) * (B,C,1,H,W) -> (B,C,R,H,W) -> sum(..., dim=1) -> (B,R,H,W)

    def get_certainty(self, CAM:torch.Tensor):
        return CAM.flatten(2).std(dim=-1)

    def get_certainties(self, CAMs:list[torch.Tensor]) -> list[torch.Tensor]:
        '''corresponds to the std of a CAM, returns list of torch.Tensor of shape (B,r)'''
        certainty_list = []
        for CAM in CAMs:
            # normalize
            CAM = CAM.flatten(2)
            # CAM = F.normalize(CAM, dim=-1, p=1)
            # std
            certainty_list.append(CAM.std(dim=-1))
        return certainty_list
    
    def get_link(self, low_CAM: torch.Tensor, high_CAM: torch.Tensor) -> torch.Tensor:
        '''corresponds to omega i.e. the edge values, returns a torch.Tensor() of shape (R,R)'''
        low_CAM = low_CAM.detach()
        high_CAM = high_CAM.detach()
        # pooling if necessary
        if low_CAM.size()[2:] != high_CAM.size()[2:]:
            low_CAM = F.adaptive_avg_pool2d(
                low_CAM,
                output_size=high_CAM.size()[2:]
            )
        low_CAM = low_CAM.flatten(2)
        high_CAM = high_CAM.flatten(2)
        # normalize
        low_CAM = F.normalize(low_CAM, p=2, dim=-1)
        high_CAM = F.normalize(high_CAM, p=2, dim=-1)
        # compute link
        batch_size = low_CAM.size(0)
        links = torch.einsum("bmi, bni -> mn", low_CAM, high_CAM) / batch_size
        return links
    
    def forward(self, feature_maps):
        if self.training:
            with torch.no_grad():
                embeddings = self.get_embeddings(feature_maps)
                certainties = []
                links = []
                high_CAM = None
                for l in range(self.n_layers):
                    low_CAM = high_CAM
                    high_CAM = self.get_CAM(feature_maps[l],l)
                    certainties.append(self.get_certainty(high_CAM))
                    if l>=1:
                        links.append(self.get_link(low_CAM.detach(), high_CAM.detach()))
        else:
            with torch.no_grad():
                embeddings = self.get_embeddings(feature_maps)
                certainties = []
                links = None
                for l in range(self.n_layers):
                    high_CAM = self.get_CAM(feature_maps[l],l)
                    certainties.append(self.get_certainty(high_CAM))

        return embeddings, certainties, links


'''    def forward(self, feature_maps):
        if self.training:
            with torch.no_grad():
                embeddings = self.get_embeddings(feature_maps)
                CAMs = self.get_CAMs(feature_maps)
                certainties = self.get_certainties(CAMs)
                links = []
                for idx in range(len(feature_maps)-1):
                    links.append(self.compute_link_at_i(CAMs[idx].detach(), CAMs[idx+1].detach()))
        else:
            with torch.no_grad():
                embeddings = self.get_embeddings(feature_maps)
                CAMs = self.get_CAMs(feature_maps)
                certainties = self.get_certainties(CAMs)
                links = None
        return embeddings, certainties, links
'''
if __name__ == "__main__":
    B,H,W = (2,16,16)
    n_layers, r = 3,128
    output_channels = [512,1024,2048]
    device = torch.device("cuda")
    feature_maps = [torch.randn(size=(B,out_channel,H,W)).to(device) for out_channel in output_channels]
    AVSL_model = AVSL_Graph(output_channels, n_layers, r).to(device)
    AVSL_model.train()

    embeddings, certainties, links = AVSL_model(feature_maps)
    print("Embedding shape:")
    for embedding in embeddings:
        print(embedding.shape)
    print("certainty shape:")
    for certainty in certainties:
        print(certainty.shape)
    print("links shape:")
    for link in links:
        print(link.shape)
    print(embedding)
    print(AVSL_model)