Re-implementation of the LoRA paper. The goal was not only to re-implement the method, but conduct experiences that complement those shown in the paper:
1. Applying LoRA to a different model (e.g. tinyllama 1B)
2. Apllying LoRA to Linear layers in the MLP block.
3. Finetuning with LoRA on a dataset in a different language than English


**Guidethrough the code**:

the important files are:
- **loralib.py** contains LoRA-related utilities such as, a LoRA-Linear nn.Module, a function that applies LoRA on some layers of tinyllama 1B, and save/load/merge util functions.

- **e2e.py, opus100.py, data_utils.py, hellaswag.py (didn't work)** contains utilities functions to load/preprocess the tuetschek/e2e_nlg, the Helsinki-NLP/opus-100 en-fr, and the hellaswag dataset. Task: **e2e:Generation, opus100:Translation, hellaswag:Multiple choice**
  
- **train.py** is the complete pipeline to train tinyllama

<br>
<br>

To reproduce the finetuning:

```bash
python train.py --yaml_config cfg.yaml --log_dir logs
```

This will finetune tinyllama1B with LoRA with the parameters given by cfg.yaml. The results are saved the logs directory.

<br>

the notebook e2e_metrics_library_for_our_project__colab measures the NLG metrics (BLUE, ROUGE, CIDER ...) for the e2e.

