from lora import apply_LoRA_tinyllama, load_AB_weights_tinyllama
from data_utils import get_tokenizer, preprocess_fn
from datasets import load_dataset
import os
import json
import yaml
import argparse
from tqdm import tqdm

import torch
import evaluate
# import nlg_eval #type: ignore

def load_finetuned_model_tokenizer(run_path):
    
    model_path = os.path.join(run_path, "model_weights","final")
    with open(os.path.join(run_path, "config.yaml"), "r") as file:
        config_dict =  yaml.safe_load(file)
    model_cfg = config_dict['model']

    tokenizer = get_tokenizer()
    model = apply_LoRA_tinyllama(target_layers=model_cfg["target_layers"], r=model_cfg["r"], new_vocsize=len(tokenizer))
    load_AB_weights_tinyllama(model_path, model, target_layers=model_cfg["target_layers"])
    return model, tokenizer

def load_preprocessed_dataset(tokenizer, split='test'):
    dataset = load_dataset("tuetschek/e2e_nlg")
    dataset = dataset.map(preprocess_fn, fn_kwargs={"tokenizer":tokenizer})
    return dataset[split] # type: ignore

def see_predictions(model, tokenizer, n_predictions:int, dataset, device:torch.device, temperature=1.0, do_sample=False):
    for i in range(n_predictions):
        idx = torch.randint(0, len(dataset), size=(1,)).item()
        sample = dataset[idx] # type: ignore
        text = sample["meaning_representation"] + "\n Description:\n"
        inputs = tokenizer(text, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=temperature, do_sample=do_sample)
        print("MR:", sample["meaning_representation"])
        print("HR:", sample["human_reference"])
        # Only take the description part of the output, that is everything after '\n Description:\n' except the last EOS token
        print("model output:", tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):-1]))
        print("=============")

def predict_and_save(model, tokenizer, dataset, device, predict_path, reference_path:str, save_reference=True, temperature=1.0, do_sample=False):
    predictions = dict()
    references = dict()
    for i in tqdm(range(len(dataset))):
        text = dataset[i]["meaning_representation"] + "\n Description:\n"
        inputs = tokenizer(text, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, max_new_tokens=500, temperature=temperature, do_sample=do_sample)
        # Only take the description part of the output, that is everything after '\n Description:\n' except the last EOS token
        predictions[i] = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):-1]).split()
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
    
def measure_metrics(predict_path, reference_path, tokenizer):#FIXME
    with open(predict_path, "r") as file:
        predictions = json.load(file)
        predictions = list(predictions.values())
    with open(reference_path, "r") as file:
        references = json.load(file)
        references = list(references.values())
    # evaluate.
    bleu = evaluate.load('bleu')
    scores = torch.zeros(size=(len(predictions),))
    for i, (prediction, reference) in enumerate(zip(predictions, references)):
        try:
            pred = tokenizer.decode(prediction)
            ref = tokenizer.decode(reference)
            scores[i] = bleu.compute(
                predictions=pred,
                references=ref,
            )['bleu'] #type: ignore
        except:
            import code; code.interact(local=locals())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, help="logs path of the finetuned model")
    parser.add_argument("--split", type=str, default="test", help="test or validation")
    args = parser.parse_args()
    # run_path = os.path.join("logs","2024-06-29","run3")
    model, tokenizer = load_finetuned_model_tokenizer(args.run_path)
    device = torch.device("cuda")
    model.to(device)
    # testset = load_preprocessed_dataset(tokenizer, args.split)
    predictions_path = f'outputs/{args.split}_tuetscheck_e2e_nlg'
    references_path = f'outputs/{args.split}_tuetscheck_e2e_nlg'
    # predict_and_save(model, tokenizer, testset, device, predict_path=predictions_path, reference_path=references_path, save_reference=True)
    measure_metrics(os.path.join(predictions_path, "predictions.json"), os.path.join(references_path, "references.json"), tokenizer=tokenizer)
        


