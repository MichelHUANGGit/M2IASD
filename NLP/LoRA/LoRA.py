import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class LoRA_Linear(nn.Module):

    def __init__(self, base_layer:nn.Module, r:int):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.base_layer.weight.requires_grad = False
        self.output_dim, self.input_dim = base_layer.weight.shape

        self.B = nn.Linear(in_features=self.input_dim, out_features=r, bias=False)
        self.B.weight = nn.init.zeros_(self.B.weight)
        self.A = nn.Linear(in_features=r, out_features=self.output_dim, bias=False)
        self.A.weight = nn.init.normal_(self.A.weight)

    def forward(self, x:torch.Tensor):
        # print("x", x.shape)
        # print("B", self.B.weight.shape)
        # print("A", self.A.weight.shape)
        y = self.base_layer(x)
        y += self.A(self.B(x))
        return y
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total_params:,}"


def apply_LoRA_tinyllama(target_layers=["q_proj","k_proj","v_proj","o_proj"], r=4):
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("Before Trainable parameters:", count_parameters(model))

    for param in model.parameters():
        param.requires_grad = False
    n_layers = len(model.model.layers)
    
    for i in range(n_layers):
        if "q_proj" in target_layers:
            model.model.layers[i].self_attn.q_proj = LoRA_Linear(model.model.layers[i].self_attn.q_proj, r=r)
        if "k_proj" in target_layers:
            model.model.layers[i].self_attn.k_proj = LoRA_Linear(model.model.layers[i].self_attn.k_proj, r=r)
        if "v_proj" in target_layers:
            model.model.layers[i].self_attn.v_proj = LoRA_Linear(model.model.layers[i].self_attn.v_proj, r=r)
        if "o_proj" in target_layers:
            model.model.layers[i].self_attn.o_proj = LoRA_Linear(model.model.layers[i].self_attn.o_proj, r=r)

    device = torch.device("cuda")
    model.to(device)

    print("After Trainable parameters:", count_parameters(model))
    return model

if __name__ == "__main__":
    # works with a 4GB memory GPU
    LoRA_llama = apply_LoRA_tinyllama(target_layers=["q_proj","k_proj","v_proj","o_proj"], r=4)

    B,L = 4,256
    vocsize = 32000
    device = torch.device("cuda")
    random_inputs = torch.randint(low=0, high=vocsize, size=(B,L)).to(device)

    # Infer
    with torch.no_grad():
        outputs = LoRA_llama(random_inputs)
    print(outputs.logits.shape)