import torch
from datasets import load_dataset


def preprocess_fn_opus100_BE2EN(sample, tokenizer, max_length):
    tokenized = tokenizer(
        sample["translation"]["be"] + "\n Translation:\n" + sample["translation"]["en"] + "</s>",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer.encode(sample["translation"]["en"] + "</s>", add_special_tokens=False)
    return dict(
        input_ids=tokenized['input_ids'],
        attention_mask=tokenized['attention_mask'],
        labels=labels)

def load_preprocessed_opus100(tokenizer, split:str, max_length:int):
    dataset = load_dataset("Helsinki-NLP/opus-100", "be-en")
    dataset = dataset.map(preprocess_fn_opus100_BE2EN, fn_kwargs={"tokenizer":tokenizer, "max_length":max_length})
    return dataset[split]#type: ignore

class DataCollatorOpus100:
    
    def __init__(self, pad_token_id, device, max_length:int, batch_size:int):
        self.pad_token_id = pad_token_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def __call__(self, batch:dict):
        attn_mask = torch.tensor([batch['attention_mask'][i] for i in range(self.batch_size)], dtype=torch.bool)
        loss_mask = torch.zeros(size=(self.batch_size, self.max_length), dtype=torch.bool)
        for i in range(self.batch_size):
            T, dt = sum(batch["attention_mask"][i]), len(batch["labels"][i])
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
    