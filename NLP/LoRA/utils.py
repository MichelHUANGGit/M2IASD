import torch
from torch.utils.data import DataLoader

class DataCollator:
    
    def __init__(self, tokenizer, max_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, batch):
        '''
        The input of the model should be the restaurant description + <sep> + human_reference, as the model is pre-trained to predict the next token
        it'll do a next token prediction, if it were perfect, we'd have a shifted (by 1) input.
        We can then define the targets as the shifted input, and compute a loss. 
        The loss should only be computed for tokens after '<sep>'.
        '''
        out_batch = dict()
        # concatenated input
        input_texts = [sample["meaning_representation"] + "<sep>" + sample["human_reference"] for sample in batch]
        tokenized = [self.tokenizer(
            input_texts[i],
            max_length=self.max_length, #maybe try 384
            truncation=True,
            padding="max_length",
        ) for i in range(len(batch))]

        # Convert to tensors
        out_batch["input_ids"] = torch.tensor(
            [tokenized[i]["input_ids"] for i in range(len(batch))], 
            dtype=torch.int32
        ).to(self.device)
        out_batch["attention_mask"] = torch.tensor(
            [tokenized[i]["attention_mask"] for i in range(len(batch))], 
            dtype=torch.int32
        ).to(self.device)

        #shifted labels + a column of padding, so the shape is consistent
        pad_column = torch.ones((len(batch),1), dtype=torch.int64)*self.tokenizer.pad_token_id
        out_batch["labels"] = torch.cat(
            [out_batch["input_ids"][:,1:], pad_column.to(self.device)],
            dim=1,
        ).to(torch.int64).to(self.device)

        # constructing the loss mask, only the indexes corresponding to the human reference should be computed, thus set to True
        cut_offs = torch.where(out_batch["input_ids"] == 32001)[1]
        loss_mask = out_batch["attention_mask"].clone()
        for i in range(len(batch)):
            loss_mask[i,:cut_offs[i]] = 0
        out_batch["loss_mask"] = loss_mask.to(dtype=torch.bool).to(self.device)
        return out_batch
    
def get_tokenizer():
    from transformers import AutoTokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({"pad_token":"[PAD]"})
    tokenizer.convert_tokens_to_ids('[PAD]')
    tokenizer.add_special_tokens({"sep_token":"<sep>"})
    tokenizer.convert_tokens_to_ids('<sep>')
    assert "sep_token" in tokenizer.special_tokens_map.keys()

    return tokenizer

    
if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("tuetschek/e2e_nlg")
    tokenizer = get_tokenizer()
    device = torch.device("cuda")
    data_collator = DataCollator(tokenizer=tokenizer, max_length=384, device=device)
    # train_loader = DataLoader(dataset=dataset["train"], batch_size=16, collate_fn=data_collator)

    import code; code.interact(local=locals())
