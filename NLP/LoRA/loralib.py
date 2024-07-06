import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import inspect
import os


class LoRA_Linear(nn.Module):

    def __init__(self, base_layer:nn.Module, r:int, alpha:float=1.):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.base_layer.weight.requires_grad = False
        self.output_dim, self.input_dim = base_layer.weight.shape

        self.B = nn.Linear(in_features=self.input_dim, out_features=r, bias=False)
        self.B.weight = nn.init.zeros_(self.B.weight)
        self.A = nn.Linear(in_features=r, out_features=self.output_dim, bias=False)
        self.A.weight = nn.init.normal_(self.A.weight)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # print("x", x.shape)
        # print("B", self.B.weight.shape)
        # print("A", self.A.weight.shape)
        y = self.base_layer(x)
        y += (self.A(self.B(x))) # A B x
        return y
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total_params:,}"


def apply_LoRA_tinyllama(target_layers_rank:dict, new_vocsize=None):
    '''
    Initializes a tinyllama 1B model with LoRA layers applied on target_layers.
    target_layers_rank is expected to be a dict in the following format:
    {<"layer_name">: <rank r of the layer>}
    Example:
    target_layers_rank = {"q_proj":2, "v_proj":2, "k_proj":4}
    '''

    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # resize the emb layer and the head, to add the pad token (full of zeros)
    if new_vocsize is not None:
        new_tokens = new_vocsize - 32000
        model.resize_token_embeddings(new_vocsize)
        model_dim = model.config.hidden_size
        model.model.embed_tokens.weight.data[-new_tokens:] = torch.zeros(size=(model_dim,))
        model.lm_head.weight.data[-new_tokens:] = torch.zeros(size=(model_dim,))
    print("Before LoRA, Trainable parameters:", count_parameters(model))

    for param in model.parameters():
        param.requires_grad = False
    n_layers = len(model.model.layers)
    
    for i in range(n_layers):
        if "self_attn.q_proj" in target_layers_rank.keys():
            model.model.layers[i].self_attn.q_proj = LoRA_Linear(model.model.layers[i].self_attn.q_proj, r=target_layers_rank["self_attn.q_proj"])
        if "self_attn.k_proj" in target_layers_rank.keys():
            model.model.layers[i].self_attn.k_proj = LoRA_Linear(model.model.layers[i].self_attn.k_proj, r=target_layers_rank["self_attn.k_proj"])
        if "self_attn.v_proj" in target_layers_rank.keys():
            model.model.layers[i].self_attn.v_proj = LoRA_Linear(model.model.layers[i].self_attn.v_proj, r=target_layers_rank["self_attn.v_proj"])
        if "self_attn.o_proj" in target_layers_rank.keys():
            model.model.layers[i].self_attn.o_proj = LoRA_Linear(model.model.layers[i].self_attn.o_proj, r=target_layers_rank["self_attn.o_proj"])
        if "mlp.gate_proj" in target_layers_rank.keys():
            model.model.layers[i].mlp.gate_proj = LoRA_Linear(model.model.layers[i].mlp.gate_proj, r=target_layers_rank["mlp.gate_proj"])
        if "mlp.up_proj" in target_layers_rank.keys():
            model.model.layers[i].mlp.up_proj = LoRA_Linear(model.model.layers[i].mlp.up_proj, r=target_layers_rank["mlp.up_proj"])
        if "mlp.down_proj" in target_layers_rank.keys():
            model.model.layers[i].mlp.down_proj = LoRA_Linear(model.model.layers[i].mlp.down_proj, r=target_layers_rank["mlp.down_proj"])

    print("After LoRA, Trainable parameters:", count_parameters(model))
    
    return model


@torch.no_grad
def merge_tinyllama(model:nn.Module, target_layers:list[str]):
    
    n_layers = len(model.model.layers)
    for i in range(n_layers):
        if "self_attn.q_proj" in target_layers:
            # I didn't find a simplier way to do it
            # Access layer
            lora_layer = model.model.layers[i].self_attn.q_proj
            # merge the weights by adding A@B (the shapes of the matrices are reversed in torch)
            lora_layer.base_layer.weight += (lora_layer.A.weight.data @ lora_layer.B.weight.data)
            # redefine the layer as the original layer
            model.model.layers[i].self_attn.q_proj = lora_layer.base_layer
        if "self_attn.k_proj" in target_layers:
            lora_layer = model.model.layers[i].self_attn.k_proj
            lora_layer.base_layer.weight += (lora_layer.A.weight.data @ lora_layer.B.weight.data)
            model.model.layers[i].self_attn.k_proj = lora_layer.base_layer
        if "self_attn.v_proj" in target_layers:
            lora_layer = model.model.layers[i].self_attn.v_proj
            lora_layer.base_layer.weight += (lora_layer.A.weight.data @ lora_layer.B.weight.data)
            model.model.layers[i].self_attn.v_proj = lora_layer.base_layer
        if "self_attn.o_proj" in target_layers:
            lora_layer = model.model.layers[i].self_attn.o_proj
            lora_layer.base_layer.weight += (lora_layer.A.weight.data @ lora_layer.B.weight.data)
            model.model.layers[i].self_attn.o_proj = lora_layer.base_layer
        if "mlp.gate_proj" in target_layers:
            lora_layer = model.model.layers[i].mlp.gate_proj
            lora_layer.base_layer.weight += (lora_layer.A.weight.data @ lora_layer.B.weight.data)
            model.model.layers[i].mlp.gate_proj = lora_layer.base_layer
        if "mlp.up_proj" in target_layers:
            lora_layer = model.model.layers[i].mlp.up_proj
            lora_layer.base_layer.weight += (lora_layer.A.weight.data @ lora_layer.B.weight.data)
            model.model.layers[i].mlp.up_proj = lora_layer.base_layer
        if "mlp.down_proj" in target_layers:
            lora_layer = model.model.layers[i].mlp.down_proj
            lora_layer.base_layer.weight += (lora_layer.A.weight.data @ lora_layer.B.weight.data)
            model.model.layers[i].mlp.down_proj = lora_layer.base_layer


def save_AB_weights_tinyllama(save_dir, model, target_layers:list[str]) -> None:
    n_layers = len(model.model.layers)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(n_layers):
        for target_layer in target_layers:
            module_name = f"model.layers.{i}.{target_layer}"
            lora_layer = model.get_submodule(module_name)
            torch.save(lora_layer.A.weight, os.path.join(save_dir, module_name+".A.pt"))
            torch.save(lora_layer.B.weight, os.path.join(save_dir, module_name+".B.pt"))


def load_AB_weights_tinyllama(save_dir, model, target_layers:list[str]) -> None:
    device = model.device
    n_layers = len(model.model.layers)
    for i in range(n_layers):
        for target_layer in target_layers:
            module_name = f"model.layers.{i}.{target_layer}"
            lora_layer = model.get_submodule(module_name)
            lora_layer.A.weight.data = torch.load(os.path.join(save_dir, module_name+".A.pt"), map_location=device)
            lora_layer.B.weight.data = torch.load(os.path.join(save_dir, module_name+".B.pt"), map_location=device)


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
    target_layers_rank = {
        "self_attn.q_proj" : 2,
        "self_attn.k_proj" : 2,
        "self_attn.v_proj" : 2,
        "self_attn.o_proj" : 2,
        "mlp.gate_proj" : 2,
        "mlp.up_proj" : 2,
        "mlp.down_proj" : 2,
    }
    LoRA_llama = apply_LoRA_tinyllama(target_layers_rank=target_layers_rank)

    B,L = 4,256
    vocsize = 32001
    device = torch.device("cuda")
    random_inputs = torch.randint(low=0, high=vocsize, size=(B,L)).to(device)
    LoRA_llama.to(device)
    import code; code.interact(local=locals())
    # Infer
    with torch.no_grad():
        outputs = LoRA_llama(random_inputs)
    print(outputs.logits.shape)