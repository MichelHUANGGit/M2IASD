
# debug
python train.py --yaml_config debugcfg.yaml --log_dir debug_logs

# real train
python train.py --yaml_config cfg.yaml --log_dir logs

python LoRA/train.py --yaml_config LoRA/cfg.yaml --log_dir LoRA/logs


# evaluation
python evaluate.py --run_path logs/2024-06-29/run3 --split test