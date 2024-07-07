import torch
from datasets import load_dataset

def preprocess_fn_opus100(sample, tokenizer, max_length, mode="FR2EN"):
    #French to English
    if mode == "FR2EN":
        tokenized = tokenizer(
            sample["translation"]["fr"] + "\n Translation:\n" + sample["translation"]["en"] + "</s>",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        t = len(tokenizer.encode(sample["translation"]["fr"] + "\n Translation:\n"))
        T = sum(tokenized["attention_mask"])
        labels = tokenized['input_ids'][t:T]
    # English to French
    elif mode == "EN2FR":
        tokenized = tokenizer(
            sample["translation"]["en"] + "\n Traduction:\n" + sample["translation"]["fr"] + "</s>",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        t = len(tokenizer.encode(sample["translation"]["en"] + "\n Traduction:\n"))
        T = sum(tokenized["attention_mask"])
        labels = tokenized['input_ids'][t:T]
    return dict(
        input_ids=tokenized['input_ids'],
        attention_mask=tokenized['attention_mask'],
        labels=labels)

def load_preprocessed_opus100(tokenizer, split:str, max_length:int, mode:str):
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")
    dataset = dataset[split]#type: ignore
    #There's 1M samples in the original trainset
    if split == "train":
        dataset = dataset.shuffle().select(range(100000))#type:ignore
    dataset = dataset.map(preprocess_fn_opus100, fn_kwargs={"tokenizer":tokenizer, "max_length":max_length, "mode":mode})#type: ignore
    return dataset

class DataCollatorOpus100:
    
    def __init__(self, pad_token_id, device, max_length:int, batch_size:int):
        self.pad_token_id = pad_token_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def __call__(self, batch:dict):
        attn_mask = torch.tensor([batch['attention_mask'][i] for i in range(self.batch_size)], dtype=torch.bool)
        loss_mask = torch.zeros(size=(self.batch_size, self.max_length), dtype=torch.bool)
        lengths = torch.sum(attn_mask, dim=1)
        for i in range(self.batch_size):
            T, dt = lengths[i], len(batch["labels"][i])
            loss_mask[i, T-dt-1:T-1] = True
        labels = torch.cat([torch.tensor(batch["labels"][i], dtype=torch.int64) for i in range(self.batch_size)])

        return dict(
            input_ids=torch.tensor(batch['input_ids'], dtype=torch.int32).to(self.device), 
            attention_mask=attn_mask.to(self.device), 
            loss_mask=loss_mask.to(self.device), 
            labels=labels.to(self.device)
        )
    
if __name__ == "__main__":

    ...
    