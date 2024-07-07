
Guidethrough the code:

the important files are:
- **loralib.py** contains LoRA-related utilities such as, a LoRA-Linear nn.Module, a function that applies LoRA on some layers of tinyllama 1B, and save/load/merge util functions.

- **e2e.py, opus100.py, hellaswag.py, data_utils.py** contains utilities functions to load/preprocess the tuetschek/e2e_nlg, the Helsinki-NLP/opus-100 en-fr, and the hellaswag dataset. Task: **e2e:Generation, opus100:Translation, hellaswag:Multiple choice**
  
- **train.py** is the complete pipeline to train tinyllama

<br>
<br>

To execute:

```bash
python train.py --yaml_config cfg.yaml --log_dir logs
```

This will finetune tinyllama1B with LoRA with the parameters given by cfg.yaml. The results are saved the logs directory.

