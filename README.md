storm-forecasting-convlstm/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .gitignore
├─ .pre-commit-config.yaml
├─ LICENSE
├─ Makefile                         # optional but useful
├─ configs/
│  ├─ base.yaml
│  ├─ model/
│  │  └─ convlstm_unet.yaml
│  ├─ data/
│  │  └─ vil_12in_12out.yaml
│  └─ experiments/
│     ├─ baseline_reproduction.yaml
│     ├─ weighted_mae_eval.yaml
│     └─ uncertainty_mc_dropout.yaml
├─ src/
│  └─ storm_forecasting/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ seed.py
│     ├─ paths.py
│     ├─ data/
│     │  ├─ __init__.py
│     │  ├─ io.py
│     │  ├─ windowing.py
│     │  ├─ splits.py
│     │  ├─ dataset.py
│     │  └─ transforms.py           # only if actually needed
│     ├─ models/
│     │  ├─ __init__.py
│     │  ├─ convlstm.py
│     │  ├─ blocks.py
│     │  └─ seq2seq_unet.py
│     ├─ training/
│     │  ├─ __init__.py
│     │  ├─ losses.py
│     │  ├─ engine.py
│     │  ├─ optim.py
│     │  └─ checkpoints.py
│     ├─ evaluation/
│     │  ├─ __init__.py
│     │  ├─ metrics.py
│     │  ├─ horizon_metrics.py
│     │  ├─ qualitative.py
│     │  └─ uncertainty.py
│     ├─ utils/
│     │  ├─ __init__.py
│     │  ├─ logging.py
│     │  └─ device.py
│     └─ cli/
│        ├─ train.py
│        ├─ evaluate.py
│        ├─ predict.py
│        └─ make_dataset_index.py   # optional
├─ scripts/
│  ├─ train_baseline.sh
│  ├─ evaluate_baseline.sh
│  └─ run_uncertainty.sh
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_error_analysis.ipynb
│  └─ 03_qualitative_results.ipynb
├─ reports/
│  ├─ project_report.pdf            # or markdown summary if allowed
│  └─ figures/
├─ tests/
│  ├─ test_windowing.py
│  ├─ test_splits.py
│  ├─ test_dataset_shapes.py
│  ├─ test_model_forward.py
│  └─ test_metrics.py
├─ data/
│  ├─ README.md                     # data access instructions only
│  └─ .gitkeep
├─ outputs/
│  ├─ checkpoints/
│  ├─ metrics/
│  ├─ figures/
│  └─ predictions/
└─ docs/
   └─ methodology.md                # optional if README gets too long