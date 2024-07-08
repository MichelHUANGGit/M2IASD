import torch
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import evaluate

def preprocess_fn_e2e(sample, tokenizer):
    input_ids = tokenizer.encode(sample["meaning_representation"] + "\n Description:\n")
    labels = tokenizer.encode(sample["human_reference"] + "</s>", add_special_tokens=False)
    return dict(input_ids=input_ids, labels=labels)

def load_preprocessed_e2e(tokenizer, split='validation'):
    dataset = load_dataset(path="tuetschek/e2e_nlg")
    dataset = dataset.map(preprocess_fn_e2e, fn_kwargs={"tokenizer":tokenizer})
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
    
class DataCollatore2e:
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
    
def predict_and_save(model, tokenizer, dataset, predict_path:str, reference_path:str, generate_kwargs:dict, save_reference=True):
    predictions = dict()
    references = dict()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    for i in tqdm(range(len(dataset))):
        text = dataset[i]["meaning_representation"] + "\n Description:\n"
        inputs = tokenizer(text, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, **generate_kwargs, use_cache=True)
        # Only take the description part of the output, that is everything after '\n Description:\n' except the last EOS token
        predictions[i] = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):-1]).split()
        if save_reference:
            references[i] = tokenizer.decode(dataset[i]['labels'][:-1]).split()

    if save_reference:
        if not(os.path.exists(reference_path)):
            os.makedirs(reference_path)
        with open(os.path.join(reference_path, "references.json"), 'w') as file:
            json.dump(references, file)

    if not(os.path.exists(predict_path)):
        os.makedirs(predict_path)
    with open(os.path.join(predict_path, "predictions.json"), 'w') as file:
        json.dump(predictions, file)

def convert_token_to_text(predict_path, reference_path, tokenizer):
    with open(predict_path, "r") as file:
        predictions = json.load(file)
        predictions = list(predictions.values())
    with open(reference_path, "r") as file:
        references = json.load(file)
        references = list(references.values())
        
    predictions_text = dict()
    for i, pred in enumerate(predictions):
        text = tokenizer.decode(pred)
        predictions_text[i] = [text]
    references_text = dict()
    for i, ref in enumerate(references):
        text = tokenizer.decode(ref)
        references_text[i] = [text]

    print("Shape of predictions_text: ", len(predictions_text))
    for i in range(5):
        print(predictions_text[i])
    print("Shape of references_text: ", len(references_text))
    for i in range(5):
        print(references_text[i])
    pred_text_path = os.path.join(predict_path, "predictions_text.json")
    with open(pred_text_path, 'w') as file:
        json.dump(predictions_text, file)
    ref_text_path = os.path.join(reference_path, "references_text.json")
    with open(ref_text_path, 'w') as file:
        json.dump(references_text, file)
    
    print(f"Saved ! \nFiles converted to text have been saved to \nPrediction text path : {pred_text_path} 
          \nreference text path : {ref_text_path}")

def transform_json_to_txt(ref_json_path, ref_txt_path, pred_json_path, pred_txt_path):  
    """
    To use the library e2e-metrics :
    Transform a JSON reference file to a formatted text file.
    """
    # reference file part
    with open(ref_json_path, 'r') as json_file:
        json_data = json.load(json_file)

    with open(ref_txt_path, 'w') as output_file:
        for key, sentences in json_data.items():
            for sentence in sentences:
                output_file.write(sentence + '\n')
            output_file.write('\n')
    print(f"Reference file transformation complete. 
          The output is saved in '{ref_txt_path}'")

    # prediction file part
    with open(pred_json_path, 'r') as json_file:
        json_data = json.load(json_file)

    with open(pred_txt_path, 'w') as output_file:
        for key, sentences in json_data.items():
            for sentence in sentences:
                output_file.write(sentence + '\n')

    print(f"Prediction file transformation complete.
            The output is saved in '{pred_txt_path}'")

### Compute metrics

def compute_bleu_metric_huggingface(pred_text_path, ref_text_path):
    bleu = evaluate.load('bleu')
    results_bleu = dict()
    for i in range(len(pred_text_path)):
        results = bleu.compute(predictions=pred_text_path[i], references=ref_text_path[i])
        results_bleu[i] = results
    
    with open(os.path.join(predictions_path, "results_bleu_metric.json"), 'w') as file:
        json.dump(results_bleu, file)
    print(f"Saved !\n Results have been saved to {file}")

    # Extraire les scores BLEU et calculer leur moyenne
    bleu_scores = [entry['bleu'] for entry in results_bleu.values()]
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f'Moyenne des scores BLEU : {average_bleu}')

def compute_rouge_metric_huggingface(pred_text_path, ref_text_path):
    print("It will takes approx. 5min.")
    rouge = evaluate.load('rouge')
    results_rouge = dict()
    for i in range(len(pred_text_path)):
        results = rouge.compute(predictions=pred_text_path[i], references=ref_text_path[i])
        results_rouge[i] = results
    
    with open(os.path.join(predictions_path, "results_rouge_metric.json"), 'w') as file:
        json.dump(results_rouge, file)
    print(f"Saved !\n Results have been saved to {file}")

    # Extraire les scores ROUGE et calculer leur moyenne
    # ROUGE-1
    rouge1_scores = [entry['rouge1'] for entry in results_rouge.values()]
    average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    # ROUGE-2
    rouge2_scores = [entry['rouge2'] for entry in results_rouge.values()]
    average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    # ROUGE-L
    rougeL_scores = [entry['rougeL'] for entry in results_rouge.values()]
    average_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    # ROUGE-Lsum
    rougeLsum_scores = [entry['rougeLsum'] for entry in results_rouge.values()]
    average_rougeLsum = sum(rougeLsum_scores) / len(rougeLsum_scores)
    print(f'Moyenne des scores ROUGE 1 : {average_rouge1}')
    print(f'Moyenne des scores ROUGE 2 : {average_rouge2}')
    print(f'Moyenne des scores ROUGE L : {average_rougeL}')
    print(f'Moyenne des scores ROUGE Lsum : {average_rougeLsum}')

    
if __name__ == "__main__":
    import argparse
    from data_utils import get_tokenizer
    from loralib import load_finetuned_tinyllama

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, help="logs path of the finetuned model")
    parser.add_argument("--split", type=str, default="test", help="test or validation")
    args = parser.parse_args()
    # run_path = os.path.join("logs","2024-06-29","run3")
    tok = get_tokenizer()
    model = load_finetuned_tinyllama(args.run_path)
    device = torch.device("cuda")
    model.to(device)
    testset = load_preprocessed_e2e(tok, args.split)
    predictions_path = f'outputs/{args.split}_e2e_nlg'
    references_path = f'outputs/{args.split}_e2e_nlg'
    predict_and_save(
        model=model, 
        tokenizer=tok, 
        dataset=testset, 
        predict_path=predictions_path, 
        reference_path=references_path, 
        save_reference=True,
        generate_kwargs={
            "max_new_tokens":100,
            "do_sample":False,
            "num_beams":5
        }
    )


