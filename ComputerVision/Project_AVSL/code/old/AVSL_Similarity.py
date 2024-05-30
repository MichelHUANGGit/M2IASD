import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import topk_mask, generate_slice
import math

class AVSL_Similarity(nn.Module):

    def __init__(
            self,
            output_channels=[512,1024,2048],
            num_classes=100,
            use_proxy_anchor=True,
            embed_dim=128,
            topk=32,
            momentum=0.5,
            p=2,
        ):
        super().__init__()
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.use_proxy_anchor = use_proxy_anchor
        self.embed_dim = embed_dim
        self.topk = topk
        self.momentum = momentum
        self.p = p

        self.register_buffer("proxy_labels", torch.arange(self.num_classes))
        self.is_link_initiated = False
        self.n_layers = len(output_channels)
        self.init_parameters()

    def init_parameters(self):
        d = self.embed_dim
        
        for l in range(self.n_layers):
            if self.use_proxy_anchor:
                proxy = nn.Parameter(nn.init.kaiming_normal_(
                            torch.zeros(self.num_classes, d), 
                            a=math.sqrt(5)
                ))
                setattr(self, f"proxy_{l}", proxy)

            # Linear coefficient for the certainty
            setattr(self, f"alpha_{l}", nn.Parameter(torch.ones(d)))
            setattr(self, f"beta_{l}", nn.Parameter(torch.zeros(d)))

            # Links from layer l to l+1
            # if l < self.n_layers-1:
            #     setattr(self, f"link_{l}to{l+1}", nn.Parameter(torch.zeros(size=(d,d))))

    def update_links(self, new_links):
        '''Incorporate gradually all training samples links with a momentum factor(=0.5)'''
        for l in range(self.n_layers-1):
            if self.is_link_initiated:
                old_link_l = getattr(self, f"link_{l}to{l+1}")
                old_link_l.data = self.momentum * old_link_l.data + (1-self.momentum) * new_links[l]
            else:
                setattr(self, f"link_{l}to{l+1}", new_links[l])
        self.is_link_initiated=True

    def get_matrix_similarity(
            self, 
            embed_list,
            certainty_list,
            embed_list2=None,
            certainty_list2=None,
        ):
        '''returns a similarity matrix between the data points in embed_list and the data points in embed_list2'''
        output_dict = dict()
        for l in range(self.n_layers):
            '''Compute the pairwise absolute difference between the first batch embedding (emb_list) and the second embedding (emb_list2)
            It uses a cool trick: by adding a dimension and leveraging PyTorch's broadcasting to compute the difference. No need to use loops!'''
            embed1 = embed_list[l]
            embed2 = embed_list2[l]
            embed1, embed2 = F.normalize(embed1, self.p, dim=-1), F.normalize(embed2, self.p, dim=-1)
            embed1, embed2 = embed1.unsqueeze(1), embed2.unsqueeze(0) #(B1,R), (B1,R) -> (B1,1,R), (1,B1,R)
            nodes = torch.abs(embed1-embed2).pow(self.p) # |(1,B,R) - (B,1,R)|^2 -> (B1,B2,R)
            # Sum along the R axis to get the similarity metric (called d in the paper) between each sample of B1 and B2
            if self.training:
                output_dict[f"emb_sim_{l}"] = torch.sum(nodes, dim=-1)

            if l>=1:
                '''Computing the matrix W hat in the paper containing all the "links" i.e. edges values'''
                W = getattr(self, f"link_{l-1}to{l}")
                # keep only k most correlated nodes at layer l-1
                W *= topk_mask(W, k=3, dim=0) #(R,R)
                # normalized edges
                W /= (torch.sum(W, dim=0, keepdim=True) + 1e-8) #(R,R)
                '''Computing the probability of unreliability matrix (P in the paper). Using the same trick as for the embeddings'''
                cert1 = certainty_list[l]
                cert2 = certainty_list2[l]
                eta = cert1.unsqueeze(1) * cert2.unsqueeze(0) #(B1,R) * (B2,R) -> (B1,1,R) * (1,B2,R) -> (B1,B2,R)
                P = torch.sigmoid(getattr(self, f"alpha_{l}") * eta + getattr(self, f"beta_{l}")) #(B1,B2,R)
                '''Rectified similarity nodes (delta hat in the paper)'''
                nodes_hat = (1-P) * (nodes_hat @ W) + P * nodes.detach() # (B1,B2,R) * ((B1,B2,R) @ (R,R)) + (B1,B2,R) * (B1,B2,R) -> (B1,B2,R)
  
            elif l==0:
                nodes_hat = nodes.detach() #(B1,B2,R)
        
        '''Final output (d hat in the paper) i.e. the similarity between the samples'''
        output_dict["ovr_sim"] = torch.sum(nodes_hat, dim=-1)

        return output_dict
    
    def forward(
            self,
            embed_list,
            certainty_list,
            labels,
            embed_list2=None,
            certainty_list2=None,
            links=None,
        ):
        # Take the new batch links and incorporate them in the current links (containing all training samples)
        self.update_links(new_links=links)
        
        # =========== Training ============
        if self.training:
            # When using proxy anchor loss, we compute the similarity between the datapoints in embed_list and the proxies
            if self.use_proxy_anchor:
                row_labels = labels
                col_labels = self.get_buffer("proxy_labels")
                embed_list2, certainty_list2 = [], []
                for l in range(self.n_layers):
                    embed_list2.append(getattr(self, f"proxy_{l}"))
                    # A tensor of shape identical to proxies[i] with values full of certainty_list[i].mean()
                    certainty_list2.append(certainty_list[l].mean() * torch.ones_like(getattr(self, f"proxy_{l}")))
            # When NOT using proxy anchor loss, we compute the similarity between the datapoints in embed_list and themselves
            else:
                embed_list2, certainty_list2 = embed_list, certainty_list
                row_labels = labels
                col_labels = labels

            # output similarities
            output_dict = self.get_matrix_similarity(
                embed_list=embed_list,
                certainty_list=certainty_list,
                embed_list2=embed_list2,
                certainty_list2=certainty_list2
            )
            output_dict["row_labels"] = row_labels
            output_dict["col_labels"] = col_labels
            return output_dict
        # ========== Inference =============
        else:
            # assert self.split_num is not None
            batch_size = embed_list[0].size(0)
            device = embed_list[0].device
            embed_list2 = embed_list if embed_list2 is None else embed_list2
            certainty_list2 = certainty_list if certainty_list2 is None else certainty_list2
            batch_size2 = embed_list2[0].size(0)
            similarities = torch.zeros((batch_size, batch_size2)).to(device)
            similarities = self.get_matrix_similarity(
                embed_list, 
                certainty_list,
                embed_list2,
                certainty_list2,
                # slice_index=slice_index
            ).get("ovr_sim")
            # slice_dict = generate_slice(batch_size, self.split_num)
            # for slice_index in slice_dict.values():
            #     print("slice_index",slice_index)
            #     similarities[slice_index, :] = self.get_matrix_similarity(
            #         embed_list, 
            #         embed_list2,
            #         certainty_list,
            #         certainty_list2,
            #         slice_index=slice_index
            #     ).get("ovr_sim")
            return similarities


if __name__ == "__main__":
    from AVSL_Graph import AVSL_Graph
    from ResNetDecomposed import ResNetDecomposed
    DEVICE = torch.device('cuda')
    
    n_layers, r = 3,128
    output_channels = [512,1024,2048]
    BaseModel = ResNetDecomposed()
    BaseModel = BaseModel.to(DEVICE)
    GraphModel = AVSL_Graph(output_channels, n_layers, r)
    GraphModel = GraphModel.to(DEVICE)
    AVSL_model = AVSL_Similarity(num_classes=69)
    AVSL_model = AVSL_model.to(DEVICE)
          
    B,H,W = (32,224,224)
    input1 = torch.randn(size=(B,3,H,W)).to(DEVICE)
    # input2 = torch.randn(size=(B,3,H,W)).to(DEVICE)
    feature_maps1 = BaseModel(input1)
    # feature_maps2 = BaseModel(input2)
    embeddings, certainties, links = GraphModel.forward(feature_maps1)
    # embeddings2, certainties2, links2 = GraphModel.forward(feature_maps2)

    AVSL_model.train()
    output = AVSL_model.forward(
        embed_list=embeddings,
        certainty_list=certainties,
        links=links,
        labels=torch.arange(32),
    )

    print(len(output))
    for key, item in output.items():
        print(key, item)