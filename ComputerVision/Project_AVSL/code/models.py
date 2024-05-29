'''obsolete'''
import torch
import torch.nn as nn
from torch.nn import functional as F
from ResNetDecomposed import ResNetDecomposed
from AVSL_Graph import AVSL_Graph
from AVSL_Similarity import AVSL_Similarity

class AVSL(nn.Module):

    def __init__(self, output_channels, emb_dim, num_classes, use_proxy, topk, momentum, p, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model = ResNetDecomposed()
        self.graph_model = AVSL_Graph(output_channels, len(output_channels), emb_dim)
        self.sim_model = AVSL_Similarity(output_channels, num_classes, use_proxy, emb_dim, topk, momentum, p)
        self.n_layers = len(output_channels)

    def forward(self, images1, labels1, images2=None):
        '''Computes the similarities between batch 1 and batch 2 if batch 2 is given (INFERENCE only), in this case labels1 is not used
        otherwise between batch 1 and itself (TRAINING)'''

        feature_maps = self.base_model.forward(images1)
        embeddings, certainties, links = self.graph_model.forward(feature_maps)

        if images2 is not None:
            feature_maps2 = self.base_model.forward(images2)
            embeddings2, certainties2, _ = self.graph_model.forward(feature_maps2)
        else:
            embeddings2, certainties2 = None, None

        # Note : During Inference, labels aren't used anyway
        output_dict = self.sim_model.forward(
            embed_list=embeddings,
            certainty_list=certainties,
            labels=labels1, 
            links=links,
            embed_list2=embeddings2,
            certainty_list2=certainties2,
        )
        return output_dict
    
if __name__ == "__main__":
    args = {
        "output_channels":[512,1024,2048],
        "emb_dim":128,
        "num_classes":30,
        "use_proxy":True,
        "topk":32,
        "momentum":0.5,
        "p":2,
    }

    device = torch.device("cuda")
    model = AVSL(**args).to(device)
    random_images = torch.randn((5,3,224,224)).to(device)
    labels = torch.randint(0,30,size=(5,)).to(device)
    output_dict = model.forward(random_images, labels)
    for key, item in output_dict.items():
        print(key, ":", item)