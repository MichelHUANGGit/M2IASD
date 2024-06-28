import torch
from torch.utils.data import DataLoader

def preprocess_fn(sample, tokenizer):
    input_ids = tokenizer.encode(sample["meaning_representation"] + "\n Description:\n")
    labels = tokenizer.encode(sample["human_reference"] + "</s>", add_special_tokens=False)
    return dict(input_ids=input_ids, labels=labels)

class DataCollator:
    
    def __init__(self, pad_token_id, max_length, device):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.device = device

    def __call__(self, batch:list[dict]):
        '''
        The input of the model should be the restaurant description + some separator + human_reference, as the model is pre-trained to predict the next token
        it'll do a next token prediction, if it were perfect, we'd have a shifted (by 1) input.
        We can then define the targets as the shifted input, and compute a loss. 
        The loss should only be computed for tokens after '<sep>'.
        '''
        batch_size = len(batch)
        input_ids = [sample['input_ids'] + sample['labels'] for sample in batch]
        attention_mask = torch.zeros(size=(batch_size, self.max_length,), dtype=torch.bool)
        loss_mask = torch.zeros(size=(batch_size, self.max_length,), dtype=torch.bool)
        for i, sample in enumerate(batch):
            attention_mask[i, :len(input_ids[i])] = 1
            loss_mask[i, len(sample['input_ids']):len(sample['input_ids'])+len(sample['labels'])] = True
            current_length = len(input_ids[i])
            input_ids[i] += [self.pad_token_id]*(self.max_length-current_length)
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        # labels need to be of dtype long (=int64)
        labels = torch.cat([torch.tensor(sample["labels"], dtype=torch.int64) for sample in batch])

        return dict(
            input_ids=input_ids.to(self.device), 
            attention_mask=attention_mask.to(self.device), 
            loss_mask=loss_mask.to(self.device), 
            labels=labels.to(self.device)
        )  

def get_tokenizer():
    from transformers import AutoTokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.convert_tokens_to_ids('[PAD]')
    assert "pad_token" in tokenizer.special_tokens_map.keys()

    return tokenizer

    
if __name__ == "__main__":
    from datasets import load_dataset
    tokenizer = get_tokenizer()
    dataset = load_dataset("tuetschek/e2e_nlg")
    dataset = dataset.map(preprocess_fn, fn_kwargs={"tokenizer":tokenizer})
    device = torch.device("cuda")
    data_collator = DataCollator(pad_token_id=32000, max_length=256, device=device)
    train_loader = DataLoader(dataset=dataset["train"], batch_size=16, collate_fn=data_collator) # type: ignore
    batch = next(iter(train_loader))
    import code; code.interact(local=locals())


