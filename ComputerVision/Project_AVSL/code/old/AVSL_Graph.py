import torch
import torch.nn as nn
from torch.nn import functional as F

class AVSL_Graph(nn.Module):

    def __init__(self, output_channels:list[int], n_layers:int, R:int):
        super().__init__()
        self.output_channels = output_channels
        self.R = R
        self.n_layers = n_layers
        self.initiate_params()
        # for l in range(n_layers):
        #     setattr(self, f"lin_proj_{l}", nn.Linear(self.output_channels[l],R))
        #     setattr(self, f"pool_{l}", nn.AdaptiveAvgPool2d(1))

    def initiate_params(self):
        """
        The maxpooling operation makes the embeddings tend towards a positive bias distribution.
        """
        # pooling
        self.adavgpool = nn.AdaptiveAvgPool2d(1)
        self.admaxpool = nn.AdaptiveMaxPool2d(1)
        self.high_CAM = None
        
        for l, dim in enumerate(self.output_channels):
            # 1x1 conv
            conv = nn.Conv2d(dim, self.R, kernel_size=(1, 1), stride=(1, 1))
            nn.init.kaiming_normal_(conv.weight, mode="fan_out")
            nn.init.constant_(conv.bias, 0)
            setattr(self, "conv1x1_{}".format(l), conv)

    # def get_embeddings(self, feature_maps:list[torch.Tensor]) -> list[torch.Tensor]:
    #     '''stores the embeddings of the layers of interest'''
    #     embeddings = []
    #     # We take a feature map (B,C,H,W) -> pool -> (B,C,1,1) -> squeeze + linear proj -> (B,r)
    #     for l, feature_map in enumerate(feature_maps):
    #         embeddings.append(
    #             getattr(self, f"lin_proj_{l}")(
    #                 getattr(self, f"pool_{l}")(feature_map).squeeze((-2,-1))
    #             )
    #         )
    #     return embeddings
    def compute_embedding_at_i(self, idx, input):
        # pooling
        ap_feat = self.adavgpool(input)
        mp_feat = self.admaxpool(input)
        output = ap_feat + mp_feat
        # get parameters
        conv = getattr(self, "conv1x1_{}".format(idx))
        # compute embeddigs
        output = conv(output)
        output = output.view(output.size(0), -1)
        return output
    
    def get_embeddings(self, feature_maps):
        embed_list = [
            self.compute_embedding_at_i(idx, item)
            for idx, item in enumerate(feature_maps)
        ]
        return embed_list
    
    def _linearize(self, input):
        H, W = input.size(2), input.size(3)
        out = F.max_unpool2d(
            *F.adaptive_max_pool2d(
                input, output_size=1, return_indices=True
            ),
            kernel_size=(H, W)
        ) * H * W
        return out

    def compute_cam_at_i(self, idx, input):
        # linearize
        ap_output = input.detach()
        am_output = self._linearize(input.detach())
        output = ap_output + am_output
        # get parameters
        conv = getattr(self, "conv1x1_{}".format(idx))
        # compute cam
        output = conv(output)
        return output    
    
    # def get_CAM(self, feature_map:torch.Tensor, l:int):
    #     a = getattr(self, f"lin_proj_{l}").weight.view(1,self.output_channels[l],self.R,1,1) #(C,R) -> (1,C,R,1,1)
    #     z = feature_map.unsqueeze(2) # (B,C,H,W) -> (B,C,1,H,W)
    #     return torch.sum(a*z, dim=1) # (1,C,R,1,1) * (B,C,1,H,W) -> (B,C,R,H,W) -> sum(..., dim=1) -> (B,R,H,W)

    def get_certainty(self, CAM:torch.Tensor):
        return CAM.flatten(2).std(dim=-1)
    
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
        # flatten and normalize
        low_CAM = F.normalize(low_CAM.flatten(2), p=2, dim=-1)
        high_CAM = F.normalize(high_CAM.flatten(2), p=2, dim=-1)
        # compute link
        batch_size = low_CAM.size(0)
        links = torch.einsum("bmi, bni -> mn", low_CAM, high_CAM) / batch_size
        return links
    
    def forward(self, feature_maps):
        embeddings = self.get_embeddings(feature_maps)
        if self.training:
            with torch.no_grad():
                certainties = []
                links = []
                high_CAM = None
                for l in range(self.n_layers):
                    low_CAM = high_CAM
                    high_CAM = self.compute_cam_at_i(l, feature_maps[l])
                    certainties.append(self.get_certainty(high_CAM))
                    if l>=1:
                        links.append(self.get_link(low_CAM.detach(), high_CAM.detach()))
        else:
            with torch.no_grad():
                certainties = []
                links = None
                for l in range(self.n_layers):
                    high_CAM = self.get_CAM(feature_maps[l],l)
                    certainties.append(self.get_certainty(high_CAM))

        return embeddings, certainties, links

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