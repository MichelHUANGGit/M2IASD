import torch
import torch.nn as nn
import torch.nn.functional as F

class ProxyAnchorLoss(nn.Module):

    def __init__(self, a=32, b=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = a #scaler (alpha in the paper)
        self.b = b #margin (delta in the paper)

    def forward(self, similarity_matrix:torch.Tensor, row_labels:torch.Tensor, col_labels:torch.Tensor):
        '''similarity_matrix : (num_classes, b)
        example: num_classes=3, b=2 (one image of class 0 and one image of class 2)
        pos_mask = [[True, False, False],
                    [False, False, True]]
        '''
        INF = 1e10
        batch_size, num_cls = row_labels.size(0), col_labels.size(0)
        pos_mask = row_labels.repeat(num_cls,1).T == col_labels.repeat(batch_size,1)
        neg_mask = ~pos_mask
        cols_with_pos_proxy = torch.where(torch.sum(pos_mask,dim=0)>0)[0]
        num_pos_proxies = len(cols_with_pos_proxy)
        
        # print(pos_mask)
        # print(neg_mask)
        # positive
        pos_mat = similarity_matrix.clone()
        pos_mat[neg_mask] = INF
        # print("pos_mat", pos_mat)
        h_pos = torch.logsumexp(-self.a*(pos_mat[:,cols_with_pos_proxy]-self.b), dim=0)
        # print("h_pos", h_pos)
        pos_loss = torch.sum(h_pos) / num_pos_proxies

        #negative
        neg_mat = similarity_matrix.clone()
        neg_mat[pos_mask] = -INF
        # print(self.a*(neg_mat+self.b))
        h_neg = torch.logsumexp(self.a*(neg_mat+self.b), dim=0)
        # print("LSE:",h_neg)
        # h_neg_ = torch.sum(torch.exp(self.a*(neg_mat+self.b)),dim=1)
        # print("test:", torch.log(1+h_neg_))
        neg_loss = torch.sum(h_neg) / batch_size

        # print("pos loss", pos_loss)
        # print("neg loss", neg_loss)
        return pos_loss + neg_loss

class Proxy_AVSL_Loss(nn.Module):
    def __init__(self, n_layers, CNN_a, CNN_b, sim_a, sim_b, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.base_loss = ProxyAnchorLoss(CNN_a, CNN_b)
        self.similarity_loss = ProxyAnchorLoss(sim_a, sim_b)

    def forward(self, collector_output:dict):
        row_labels, col_labels = collector_output["row_labels"], collector_output["col_labels"]
        # number of similarities in the matrix (batch size1 * num_proxies)
        B = collector_output["ovr_sim"].size(0) * collector_output["ovr_sim"].size(1)
        total_loss = self.similarity_loss(collector_output["ovr_sim"], row_labels, col_labels) / B
        for l in range(self.n_layers):
            B = collector_output[f"emb_sim_{l}"].size(0) * collector_output[f"emb_sim_{l}"].size(1)
            total_loss += self.base_loss(collector_output[f"emb_sim_{l}"], row_labels, col_labels) / B
        return total_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, precision=3)
    num_classes = 30
    batch_size = 100
    row_labels = torch.randint(0,batch_size,size=(batch_size,))
    col_labels = torch.arange(num_classes)
    similarity_matrix = torch.abs(torch.randn(size=(batch_size, num_classes)))/100
    loss = ProxyAnchorLoss()
    print(loss(similarity_matrix, row_labels, col_labels))
    