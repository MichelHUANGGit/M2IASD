import torch
from datasets import load_dataset


def preprocess_fn_opus100(sample, tokenizer):
    input_ids = tokenizer.encode(sample["translation"]["en"] + "\n Traduction en Français:\n")
    labels = tokenizer.encode(sample["translation"]["fr"] + "</s>", add_special_tokens=False)
    return dict(input_ids=input_ids, labels=labels)

def load_preprocessed_opus100(tokenizer, split='validation'):
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")
    dataset = dataset.map(preprocess_fn_opus100, fn_kwargs={"tokenizer":tokenizer})
    return dataset[split]#type: ignore

class DataCollatorOpus100:
    
    def __init__(self, pad_token_id, device, max_length:int, batch_size:int):
        self.pad_token_id = pad_token_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def __call__(self, batch:dict):
        input_ids = [batch['input_ids'][i] + batch['labels'][i] for i in range(self.batch_size)]
        attention_mask = torch.zeros(size=(self.batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros(size=(self.batch_size, self.max_length), dtype=torch.bool)
        for i in range(self.batch_size):
            t, dt = len(batch['input_ids'][i]), len(batch['labels'][i])
            attention_mask[i, :t+dt] = True
            loss_mask[i, t-1:t+dt-1] = True
            input_ids[i] += [self.pad_token_id]*(self.max_length-t-dt) #Padding
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        # labels need to be of dtype long (=int64)
        labels = torch.cat([torch.tensor(batch["labels"][i], dtype=torch.int64) for i in range(self.batch_size)])

        return dict(
            input_ids=input_ids.to(self.device), 
            attention_mask=attention_mask.to(self.device), 
            loss_mask=loss_mask.to(self.device), 
            labels=labels.to(self.device)
        )
    
if __name__ == "__main__":

    ...
    