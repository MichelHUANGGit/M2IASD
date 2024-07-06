from datasets import load_dataset
import torch
from tqdm import tqdm
from time import time
from math import exp


def preprocess_fn_hellaswag(sample, tokenizer):
    sample['ctx_ids'] = tokenizer.encode(sample['ctx'])
    for i in range(4):
        sample[f"ending{i}"] = tokenizer.encode(sample["endings"][i] + "</s>", add_special_tokens=False)
    return sample

def load_preprocessed_hellaswag(tokenizer, split='validation'):
    dataset = load_dataset(path="Rowan/hellaswag")
    dataset = dataset.map(preprocess_fn_hellaswag, fn_kwargs={"tokenizer":tokenizer})
    return dataset[split]#type: ignore
    
class DataCollatorHellaswagTrain:

    def __init__(self, tokenizer_pad_token_id, max_length, device, batch_size):
        self.tokenizer_pad_token_id = tokenizer_pad_token_id
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size

    def __call__(self, batch:dict)-> dict:
        ''' 
        [FOR TRAINING] What we want is a dict of tensors:
        - For 'input_ids'
        ###     CONTEXT IDS + TRUE ENDING + PADDING     ###
        - For 'attention_mask', True everywhere except on paddings
        - For 'loss_mask', False everywhere except on the location of the true ending, shifted by 1.
        Example: ###    A, man, is, sitting, on, a, roof, ., he | starts, pulling, up, roofing, on, a, roof, .      ###
        loss_mask =     0   0   0   0        0   0  0     0  1    1       1        1   1        1   1  1     0
        (where | denotes the separation of the context from the ending).
        - For 'labels' the true ending token ids
        '''
        endings = [batch[f"ending{batch['label'][i]}"][i] for i in range(self.batch_size)]
        input_ids = [batch['ctx_ids'][i] + endings[i] for i in range(self.batch_size)]
        attention_mask = torch.zeros(size=(self.batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros(size=(self.batch_size, self.max_length), dtype=torch.bool)
        for i in range(self.batch_size):
            sentence_length = len(input_ids[i])
            ctx_length = len(batch['ctx_ids'][i])
            attention_mask[i, :sentence_length] = True
            loss_mask[i, ctx_length-1:sentence_length-1] = True
            input_ids[i] += [self.tokenizer_pad_token_id] * (self.max_length - sentence_length)
        labels = torch.cat([torch.tensor(ending, dtype=torch.int64) for ending in endings], dim=0)
        return dict(
            input_ids = torch.tensor(input_ids, dtype=torch.int32).to(self.device),
            attention_mask = attention_mask.to(self.device),
            loss_mask = loss_mask.to(self.device),
            labels = labels.to(self.device)
        )
    
class DataCollatorHellaswagVal:

    def __init__(self, tokenizer_pad_token_id, max_length, device, batch_size):
        self.tokenizer_pad_token_id = tokenizer_pad_token_id
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size

    def __call__(self, batch:dict)-> dict:
        '''
        [VALIDATION] what we want:
        - For 'input_ids'
        ###     CONTEXT IDS  |   ENDING 1 + PADDING     ###
        ###     CONTEXT IDS  |   ENDING 2 + PADDING     ###
        ###     CONTEXT IDS  |   ENDING 3 + PADDING     ###
        ###     CONTEXT IDS  |   ENDING 4 + PADDING     ###
        The model does its next token prediction on all 4 endings, we want to evaluate the likelihood of the 4 endings. 
        Ideally we want our model to have the highest likelihood on the true ending (or lowest neg likelihood)

        - For 'attention_mask', only False on PADDINGS, True everywhere else

        - For 'loss_mask', we want True on ENDINGs but SHIFTED by 1 because the model does next token prediction, False everywhere else
        Example: ###    A, man, is, sitting, on, a, roof, ., he | starts, pulling, up, roofing, on, a, roof, .      ###
        loss_mask =     0   0   0   0        0   0  0     0  1    1       1        1   1        1   1  1     0

        - For 'labels' (unknown for the test set), we want simply want the indexes that correspond to the true ending
        '''
        endings = [[batch[f"ending{j}"][i] for j in range(4)] for i in range(self.batch_size)]
        input_ids = [[batch['ctx_ids'][i] + endings[i][j] for j in range(4)] for i in range(self.batch_size)]
        attention_mask = torch.zeros(size=(4*self.batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros(size=(4*self.batch_size, self.max_length), dtype=torch.bool)
        for i in range(self.batch_size):
            ctx_length = len(batch['ctx_ids'][i])
            for j in range(4):
                sentence_length = len(input_ids[i][j])
                attention_mask[i*4+j, :sentence_length] = True
                loss_mask[i*4+j, ctx_length-1:sentence_length-1] = True
                input_ids[i][j] += [self.tokenizer_pad_token_id] * (self.max_length - sentence_length)
        labels = torch.tensor([int(batch['label'][i]) for i in range(self.batch_size)], dtype=torch.int64)
        return dict(
            input_ids = torch.tensor(input_ids, dtype=torch.int64).reshape(self.batch_size*4, self.max_length).to(self.device),
            attention_mask = attention_mask.to(self.device),
            loss_mask = loss_mask.to(self.device),
            labels = labels.to(self.device)
        )
    
@torch.no_grad
def eval_hellaswag(model, val_loader) -> tuple[float, float, float]:
    '''Inspired by https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py'''
    progress_bar = tqdm(range(1, len(val_loader)+1), unit="batch")
    bsz = val_loader.batch_size
    correct_preds = 0
    total = 0
    t0 = time()
    torch.set_float32_matmul_precision("high")
    for i in progress_bar:
        batch = val_loader.next_batch()
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        logits = model(input_ids, attention_mask, use_cache=False)["logits"]
        flat_shift_logits = logits.view(-1, logits.size(-1)) # (4*B,T,voc_size) -> (4*B*T, voc_size)
        flat_shift_tokens = input_ids.view(-1) # (4*B*T)
        losses = torch.nn.functional.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        losses = losses.view(4*bsz, -1) # (4*B*T) -> (4*B, T)
        # now get the average loss just for the completion region (where mask == 1), in each row
        masked_shift_losses = losses * batch['loss_mask'] # (4*B, T)
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1) # -> (4*B)
        normalized_loss = sum_loss / batch['loss_mask'].sum(dim=1) # (4*B) / (4*B) -> (4*B)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        preds = sum_loss.view(bsz, 4).argmin(dim=-1) # (4*B) -> view -> (B,4) -> argmin -> (B)
        preds_norm = normalized_loss.view(bsz, 4).argmin(dim=-1)# (4*B) -> view -> (B,4) -> argmin -> (B)
        
        correct_preds += (preds_norm == batch['labels']).sum().item()
        total += bsz
        acc = correct_preds/total
        labels_idx = torch.arange(0, 4*bsz, 4) + batch['labels'].cpu()
        avg_loss = normalized_loss[labels_idx].mean().item() # average loss on the true endings
        likelihood = exp(-avg_loss)
        progress_bar.set_description(f"acc: {acc*100:.2f}% | cross-entropy loss:{avg_loss:.4f} | true ending probability:{likelihood*100:.2f}")
    dt = time() - t0
    return acc, avg_loss, dt

if __name__ == "__main__":
    from data_utils import get_tokenizer
    tokenizer = get_tokenizer()
    dataset=  load_preprocessed_hellaswag(tokenizer, "train")
    x = dataset[0]
    import code;code.interact(local=locals())