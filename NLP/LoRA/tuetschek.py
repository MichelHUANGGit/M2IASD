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
            t, dt = len(sample["input_ids"]), len(sample["labels"])
            attention_mask[i, :t+dt] = True
            loss_mask[i, t-1:t+dt-1] = True # -1 because the labels are shifted
            input_ids[i] += [self.pad_token_id]*(self.max_length-t-dt)
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        # labels need to be of dtype long (=int64)
        labels = torch.cat([torch.tensor(sample["labels"], dtype=torch.int64) for sample in batch])

        return dict(
            input_ids=input_ids.to(self.device), 
            attention_mask=attention_mask.to(self.device), 
            loss_mask=loss_mask.to(self.device), 
            labels=labels.to(self.device)
        )
    
class CustomDataCollator:
    '''
    what we want:
    - For 'input_ids'
    ###     MEANING REPRESENTATION + "\n Description:\n" +  HUMAN REFERENCE + [EOS] + PADDING ###

    - For 'attention_mask', only False on PADDINGS, True everywhere else

    - For 'loss_mask', we want True on ENDINGs but SHIFTED by 1 because the model does next token prediction, False everywhere else
    Example: ###    A, man, is, sitting, on, a, roof, ., he | starts, pulling, up, roofing, on, a, roof, .      ###
    loss_mask =     0   0   0   0        0   0  0     0  1    1       1        1   1        1   1  1     0

    - For 'labels' (unknown for the test set), we want the token ids of the true ending.
    '''
    
    def __init__(self, pad_token_id, device):
        self.pad_token_id = pad_token_id
        self.device = device

    def __call__(self, batch:dict):
        batch_size = len(batch['input_ids'])
        input_ids = [batch['input_ids'][i] + batch['labels'][i] for i in range(batch_size)]
        max_length = max(len(input_ids[i]) for i in range(batch_size)) # Maximum length of the batch
        attention_mask = torch.zeros(size=(batch_size, max_length), dtype=torch.bool)
        loss_mask = torch.zeros(size=(batch_size, max_length), dtype=torch.bool)
        for i in range(batch_size):
            t, dt = len(batch['input_ids'][i]), len(batch['labels'][i])
            attention_mask[i, :t+dt] = 1
            loss_mask[i, t-1:t+dt-1] = True
            input_ids[i] += [self.pad_token_id]*(max_length-t-dt) #Padding
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        # labels need to be of dtype long (=int64)
        labels = torch.cat([torch.tensor(batch["labels"][i], dtype=torch.int64) for i in range(batch_size)])

        return dict(
            input_ids=input_ids.to(self.device), 
            attention_mask=attention_mask.to(self.device), 
            loss_mask=loss_mask.to(self.device), 
            labels=labels.to(self.device)
        )
    
class CustomDataLoader:
    '''infinite dataloader with uniform (with replacement) sampling'''
    def __init__(self, dataset, batch_size:int, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.N = len(dataset)

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def next_batch(self):
        random_batch_indexes = torch.randint(0, self.N, size=(self.batch_size,))
        if self.collate_fn is not None:
            return self.collate_fn(self.dataset[random_batch_indexes])
        else:
            return self.dataset[random_batch_indexes]

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
    collate_fn = CustomDataCollator(pad_token_id=32000, device=device)
    loader = CustomDataLoader(dataset["train"], batch_size=16, collate_fn=collate_fn) # type: ignore
    batch_ = loader.next_batch()
    import code; code.interact(local=locals())


