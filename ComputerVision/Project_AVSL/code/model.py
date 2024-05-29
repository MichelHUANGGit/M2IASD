import torch
import torch.nn as nn
from torch.nn import functional as F
from base_models import ResNet50_, EfficientNet_v2_S_
from utils import topk_mask
import math


class AVSL_Graph(nn.Module):

    def __init__(self, emb_dim, base_model_name="ResNet50", lay_to_emb_ids=[2,3,4]):
        super().__init__()
        self.base_model_name = base_model_name
        if base_model_name == "ResNet50":
            self.base_model = ResNet50_(lay_to_emb_ids)
        elif base_model_name == "EfficientNet_V2_S":
            self.base_model = EfficientNet_v2_S_(lay_to_emb_ids)

        self.lay_to_emb_ids = lay_to_emb_ids
        self.n_layers = len(lay_to_emb_ids)
        self.output_channels = [self.base_model.layers_output_channels[i] for i in lay_to_emb_ids]
        self.emb_dim = emb_dim
        self.initiate_params()

    
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
            conv = nn.Conv2d(dim, self.emb_dim, kernel_size=(1, 1), stride=(1, 1))
            nn.init.kaiming_normal_(conv.weight, mode="fan_out")
            nn.init.constant_(conv.bias, 0)
            setattr(self, "conv1x1_{}".format(l), conv)
    
    def get_embedding(self, feature_map:torch.Tensor, l) -> torch.Tensor:
        ap_feat = self.adavgpool(feature_map)
        mp_feat = self.admaxpool(feature_map)
        output = ap_feat + mp_feat
        conv = getattr(self, "conv1x1_{}".format(l))
        # compute embeddigs
        output = conv(output)
        output = output.view(output.size(0), -1)
        return output

    
    def _linearize(self, input:torch.Tensor):
        H, W = input.size(2), input.size(3)
        out = F.max_unpool2d(
            *F.adaptive_max_pool2d(
                input, output_size=1, return_indices=True
            ),
            kernel_size=(H, W)
        ) * H * W
        return out
    
    def get_CAM(self, feature_map:torch.Tensor, l:int):
        # linearize
        ap_output = feature_map.detach()
        am_output = self._linearize(feature_map.detach())
        output = ap_output + am_output
        conv = getattr(self, "conv1x1_{}".format(l))
        # compute cam
        output = conv(output)
        return output

    def get_certainty(self, CAM:torch.Tensor):
        # print("CAM shape before flattening :",CAM.shape)
        # return CAM.flatten(2)
        return CAM.flatten(2).std(dim=-1)

    def get_link(self, low_CAM: torch.Tensor, high_CAM: torch.Tensor) -> torch.Tensor:
        '''corresponds to omega hat i.e. the edge values, returns a torch.Tensor of shape (emb_dim,emb_dim)'''
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
    
    def get_graph(self, feature_map, l):
        embedding = self.get_embedding(feature_map, l)
        with torch.no_grad():
            self.low_CAM = self.high_CAM
            self.high_CAM = self.get_CAM(feature_map, l)
            certainty = self.get_certainty(self.high_CAM)
            if l>=1 and self.training:
                link = self.get_link(self.low_CAM, self.high_CAM)
            else:
                link = None
        return embedding, certainty, link, feature_map
    
    def forward(self, x:torch.Tensor, l):
        feature_map = self.base_model(x, l)
        return self.get_graph(feature_map, l)
    
class AVSL_Similarity(nn.Module):

    def __init__(
            self,
            base_model_name="ResNet50",
            lay_to_emb_ids=[2,3,4],
            num_classes=30,
            use_proxy=True,
            emb_dim=128,
            topk=32,
            momentum=0.5,
            p=2,
        ):
        super().__init__()
        self.base_model_name = base_model_name
        self.lay_to_emb_ids = lay_to_emb_ids
        self.num_classes = num_classes
        self.use_proxy = use_proxy
        self.emb_dim = emb_dim
        self.topk = topk
        self.momentum = momentum
        self.p = p

        self.Graph_model = AVSL_Graph(emb_dim, base_model_name, lay_to_emb_ids)
        self.init_parameters()

    def init_parameters(self):
        self.n_layers = len(self.lay_to_emb_ids)
        self.is_link_initiated = [False] * self.n_layers
        self.register_buffer("proxy_labels", torch.arange(self.num_classes))

        d = self.emb_dim
        for l in range(self.n_layers):
            if self.use_proxy:
                proxy = nn.Parameter(nn.init.kaiming_normal_(
                            torch.zeros(self.num_classes, d), 
                            a=math.sqrt(5)
                ))
                setattr(self, f"proxy_{l}", proxy)

            # Linear coefficient for the certainty
            setattr(self, f"alpha_{l}", nn.Parameter(torch.ones(d)))
            setattr(self, f"beta_{l}", nn.Parameter(torch.zeros(d)))
            if l >= 1:
                setattr(self, f"link_{l-1}to{l}", torch.zeros(d, d))

    def update_link(self, new_link, l):
        '''Incorporate gradually all training samples links with a momentum factor(=0.5)'''
        if self.is_link_initiated[l]:
            old_link_l = getattr(self, f"link_{l-1}to{l}")
            old_link_l.data = self.momentum * old_link_l.data + (1-self.momentum) * new_link
        else:
            setattr(self, f"link_{l-1}to{l}", new_link)
            self.is_link_initiated[l]=True

    def get_similarity_matrix(self, emb1, cert1, emb2, cert2, l):
        # Normalize
        emb1, emb2 = F.normalize(emb1, self.p, dim=-1), F.normalize(emb2, self.p, dim=-1)
        emb1, emb2 = emb1.unsqueeze(1), emb2.unsqueeze(0) #(B1,R), (B1,R) -> (B1,1,R), (1,B1,R)
        nodes = torch.abs(emb1-emb2).pow(self.p) # |(1,B,R) - (B,1,R)|^2 -> (B1,B2,R)

        if l==0:
            self.nodes_hat = nodes
        else:
            '''Computing the matrix W hat in the paper containing all the "links" i.e. edges values'''
            W = getattr(self, f"link_{l-1}to{l}")
            # keep only k most correlated nodes at layer l-1
            W *= topk_mask(W, k=3, dim=0) #(R,R)
            # normalized edges
            W /= (torch.sum(W, dim=0, keepdim=True) + 1e-8) #(R,R)
            '''Computing the probability of unreliability matrix (P in the paper). Using the same trick as for the embeddings'''
            # print("cert1,2", cert1.shape, cert2.shape)
            # eta = torch.bmm(
            #     cert1.transpose(0,1), 
            #     cert2.transpose(0,1).transpose(1,2)
            #     ).transpose(0,2) #(B1,R,HW) and (B2,R,HW) -> (R,B1,HW) @ (R,HW,B2) -> (R,B1,B2) -> (B1,B2,R)
            # print("eta",eta.shape)
            eta = cert1.unsqueeze(1) * cert2.unsqueeze(0) #(B1,R) * (B2,R) -> (B1,1,R) * (1,B2,R) -> (B1,B2,R)
            P = torch.sigmoid(getattr(self, f"alpha_{l}") * eta + getattr(self, f"beta_{l}")) #(B1,B2,R)
            # print("P", P.shape)
            '''Rectified similarity nodes (delta hat in the paper)'''
            # print((self.nodes_hat @ W).shape)
            self.nodes_hat = (1-P) * (self.nodes_hat @ W) + P * nodes.detach() # (B1,B2,R) * ((B1,B2,R) @ (R,R)) + (B1,B2,R) * (B1,B2,R) -> (B1,B2,R)

        if self.training:
            # Sum along the R axis to get the similarity metric (called d in the paper) between each sample of B1 and B2
            return torch.sum(nodes, dim=-1)
    
    def forward(self, images1, labels1, images2=None):
        '''Computes the similarities between batch 1 and batch 2 if batch 2 is given (INFERENCE only), in this case labels1 is not used
        otherwise computes similarities between batch 1 and itself (TRAINING)'''
        output_dict = dict()
        # ========== Training ===========
        if self.training:
            fmap1 = images1
            for l in range(self.n_layers):
                emb1, cert1, link, fmap1 = self.Graph_model(fmap1, l)
                emb2 = getattr(self, f"proxy_{l}")
                cert2 = cert1.mean() * torch.ones_like(emb2)
                if l>=1:
                    self.update_link(link, l)
                output_dict[f"emb_sim_{l}"] = self.get_similarity_matrix(emb1, cert1, emb2, cert2, l)
            # Overall similarity
            output_dict["ovr_sim"] = torch.sum(self.nodes_hat, dim=-1)
            output_dict["row_labels"] = labels1
            output_dict["col_labels"] = getattr(self, "proxy_labels")
        # ============== Inference =============
        else:
            images2 = images1 if images2 is None else images2
            fmap1 = images1
            fmap2 = images2
            for l in range(self.n_layers):
                emb1, cert1, _, fmap1 = self.Graph_model(fmap1, l)
                emb2, cert2, _, fmap2 = self.Graph_model(fmap2, l)
                self.get_similarity_matrix(emb1, cert1, emb2, cert2, l)
            output_dict["ovr_sim"] = torch.sum(self.nodes_hat, dim=-1)

        return output_dict

if __name__ == "__main__":
    device = torch.device("cuda")
    B, C, H, W = 16,3,256,256
    n_layers = 3
    input = torch.randn(size=(B,C,W,H)).to(device)
    labels = torch.randint(0,30, size=(B,))
    n_layers, emb_dim = 7, 128
    lay_to_emb_ids = [1, 2, 3, 4, 5, 6, 7]

    model = AVSL_Similarity(
        base_model_name="EfficientNet_V2_S", 
        lay_to_emb_ids=lay_to_emb_ids,
    ).to(device)
    
    model.train()
    output_dict = model(input, labels)
    for k,v in output_dict.items():
        print(k, v.shape)

    model.eval()
    model(input, labels)