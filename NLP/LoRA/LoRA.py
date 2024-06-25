import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import inspect


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


    print("After Trainable parameters:", count_parameters(model))
    
    return model

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused)
    return optimizer

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