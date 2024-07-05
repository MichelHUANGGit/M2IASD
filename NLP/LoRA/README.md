
Guidethrough the code:

the important files are:
- loralib.py
- hellaswag.py
- tuetschek.py
- train.py

loralib.py contains LoRA-related utilities such as, a LoRA-Linear nn.Module, a function that applies LoRA on some layers of tinyllama 1B, and save/load/merge util functions. <br>

hellaswag.py contains utilities functions to load/preprocess the hellaswag dataset (for training and evaluation). + eval function

tuetschek.py contains utilities functions to load/preprocess the tuetschek/e2e_nlg dataset

train.py is the complete pipeline to train tinyllama

To train:

This will finetune tinyllama1B with LoRA with certain parameters given by cfg.yaml. The results are saved the logs directory.
```bash
python train.py --yaml_config cfg.yaml --log_dir logs
```

