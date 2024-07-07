import torch
from torch.utils.data import DataLoader
from typing import Any, DefaultDict


def get_tokenizer():
    from transformers import AutoTokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.convert_tokens_to_ids('[PAD]')
    assert "pad_token" in tokenizer.special_tokens_map.keys()
    return tokenizer

class CustomDataLoader:
    '''infinite dataloader with uniform (with replacement) sampling'''
    def __init__(self, dataset, collate_fn, batch_size:int, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.N = len(dataset)
        self.i = 0

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def next_batch(self) -> Any:
        if self.shuffle:
            random_batch_indexes = torch.randint(0, self.N, size=(self.batch_size,))
            return self.collate_fn(self.dataset[random_batch_indexes])
            
        
def get_loader(name, tokenizer, split, batch_size, max_length, device, shuffle):
    assert name in ["e2e", "hellaswag", "opus100"]
    
    print(f"Loading and preprocessing the {split} {name} dataset...")
    if name == "e2e":
        from e2e import load_preprocessed_e2e, DataCollatore2e
        dataset = load_preprocessed_e2e(tokenizer=tokenizer, split=split)
        collate_fn = DataCollatore2e(tokenizer.pad_token_id, device=device, max_length=max_length)
    
    elif name == "hellaswag":
        from hellaswag import load_preprocessed_hellaswag, DataCollatorHellaswagTrain, DataCollatorHellaswagVal
        dataset = load_preprocessed_hellaswag(tokenizer=tokenizer, split=split)
        if split == "train":
            collate_fn = DataCollatorHellaswagTrain(tokenizer.pad_token_id, device=device, max_length=max_length, batch_size=batch_size)
        elif split == "validation":
            collate_fn = DataCollatorHellaswagVal(tokenizer.pad_token_id, device=device, max_length=max_length, batch_size=batch_size)

    elif name == "opus100":
        from opus100 import load_preprocessed_opus100, DataCollatorOpus100
        dataset = load_preprocessed_opus100(tokenizer=tokenizer, split=split, max_length=max_length)
        collate_fn = DataCollatorOpus100(tokenizer.pad_token_id, device=device, max_length=max_length, batch_size=batch_size)

    loader = CustomDataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    print("============================================================================================")

    return loader

if __name__ == "__main__":
    args = dict(
        name = "opus100",
        tokenizer = get_tokenizer(),
        split = "train",
        batch_size = 2,
        max_length = 256,
        device = torch.device("cpu"),
        shuffle = True,
    )
    loader = get_loader(**args)
    batch = loader.next_batch()
    import code; code.interact(local=locals())