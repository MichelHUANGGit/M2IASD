from datasets import load_dataset
import torch


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
        labels = [batch[f"ending{batch['label'][i]}"][i] for i in range(self.batch_size)]
        input_ids = [batch['ctx_ids'][i] + labels[i] for i in range(self.batch_size)]
        attention_mask = torch.zeros(size=(self.batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros(size=(self.batch_size, self.max_length), dtype=torch.bool)
        for i in range(self.batch_size):
            sentence_length = len(input_ids[i])
            ctx_length = len(batch['ctx_ids'][i])
            attention_mask[i, :sentence_length] = True
            loss_mask[i, ctx_length-1:sentence_length-1] = True
            input_ids[i] += [self.tokenizer_pad_token_id] * (self.max_length - sentence_length)
        labels = torch.cat([torch.tensor(label, dtype=torch.int64) for label in labels], dim=0)
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
        Ideally our model has the highest likelihood on the true ending (or lowest neg likelihood)

        - For 'attention_mask', only False on PADDINGS, True everywhere else

        - For 'loss_mask', we want True on ENDINGs but SHIFTED by 1 because the model does next token prediction, False everywhere else
        Example: ###    A, man, is, sitting, on, a, roof, ., he | starts, pulling, up, roofing, on, a, roof, .      ###
        loss_mask =     0   0   0   0        0   0  0     0  1    1       1        1   1        1   1  1     0

        - For 'labels' (unknown for the test set), we want the token ids of the true ending.
        '''
        labels = [[batch[f"ending{j}"][i] for j in range(4)] for i in range(self.batch_size)]
        input_ids = [[batch['ctx_ids'][i] + labels[i][j] for j in range(4)] for i in range(self.batch_size)]
        attention_mask = torch.zeros(size=(4*self.batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros(size=(4*self.batch_size, self.max_length), dtype=torch.bool)
        for i in range(self.batch_size):
            ctx_length = len(batch['ctx_ids'][i])
            for j in range(4):
                sentence_length = len(input_ids[i][j])
                attention_mask[i*4+j, :sentence_length] = True
                loss_mask[i*4+j, ctx_length-1:sentence_length-1] = True
                input_ids[i][j] += [self.tokenizer_pad_token_id] * (self.max_length - sentence_length)
        labels = torch.cat([torch.tensor(label[j], dtype=torch.int64) for j in range(4) for label in labels], dim=0)
        return dict(
            input_ids = torch.tensor(input_ids, dtype=torch.int32).reshape(self.batch_size*4, self.max_length),
            attention_mask = attention_mask,
            loss_mask = loss_mask,
            labels = labels
        )

if __name__ == "__main__":
    # args = dict(
    #     tokenizer = ,
    #     split = "train",
    #     batch_size = 16,
    #     max_length = 192,
    #     device = torch.device("cpu"),
    #     shuffle = True,
    # )
    # loader = get_loader(**args)
    # batch = loader.next_batch()
    # import code; code.interact(local=locals())
    ...