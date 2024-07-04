import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

def preprocess_fn_tuetschek(sample, tokenizer):
    input_ids = tokenizer.encode(sample["meaning_representation"] + "\n Description:\n")
    labels = tokenizer.encode(sample["human_reference"] + "</s>", add_special_tokens=False)
    return dict(input_ids=input_ids, labels=labels)

def load_preprocessed_tuetschek(tokenizer, split='validation'):
    dataset = load_dataset(path="tuetschek/e2e_nlg")
    dataset = dataset.map(preprocess_fn_tuetschek, fn_kwargs={"tokenizer":tokenizer})
    return dataset[split]#type: ignore

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
    
class DataCollatorTuetschek:
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
    
    def __init__(self, pad_token_id, device, max_length:int):
        self.pad_token_id = pad_token_id
        self.device = device
        self.max_length = max_length

    def __call__(self, batch:dict):
        batch_size = len(batch['input_ids'])
        input_ids = [batch['input_ids'][i] + batch['labels'][i] for i in range(batch_size)]
        # max_length = max(len(input_ids[i]) for i in range(batch_size)) # Maximum length of the batch
        attention_mask = torch.zeros(size=(batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros(size=(batch_size, self.max_length), dtype=torch.bool)
        for i in range(batch_size):
            t, dt = len(batch['input_ids'][i]), len(batch['labels'][i])
            attention_mask[i, :t+dt] = 1
            loss_mask[i, t-1:t+dt-1] = True
            input_ids[i] += [self.pad_token_id]*(self.max_length-t-dt) #Padding
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        # labels need to be of dtype long (=int64)
        labels = torch.cat([torch.tensor(batch["labels"][i], dtype=torch.int64) for i in range(batch_size)])

        return dict(
            input_ids=input_ids.to(self.device), 
            attention_mask=attention_mask.to(self.device), 
            loss_mask=loss_mask.to(self.device), 
            labels=labels.to(self.device)
        )

    
if __name__ == "__main__":
    # from datasets import load_dataset
    # tokenizer = get_tokenizer()
    # dataset = load_dataset("tuetschek/e2e_nlg")
    # dataset = dataset.map(preprocess_fn, fn_kwargs={"tokenizer":tokenizer})
    # device = torch.device("cuda")
    # data_collator = DataCollator(pad_token_id=32000, device=device, max_length=256)
    # train_loader = DataLoader(dataset=dataset["train"], batch_size=16, collate_fn=data_collator) # type: ignore
    # batch = next(iter(train_loader))
    # import code; code.interact(local=locals())
    # collate_fn = CustomDataCollator(pad_token_id=32000, device=device, max_length=256)
    # loader = CustomDataLoader(dataset["train"], batch_size=16, collate_fn=collate_fn) # type: ignore
    # batch_ = loader.next_batch()
    ...


