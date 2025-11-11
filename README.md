# blood-cancer-detection


## Training

```bash
python3 src/train_enhanced_v2.py --data_root ./data/Original --save_dir ./artifacts_v2
```

## Evalaution

```bash
python src/eval_plots_enhanced_v2.py --artifacts ./artifacts_v2
```


## Inference

```bash
python3 src/app.py --artifacts ./artifacts_v2
```
