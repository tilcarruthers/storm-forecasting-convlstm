Raw dataset files should live here locally but should not be committed.

Expected local files:
- `data/train.h5`
- `data/events.csv`
- `data/vil_events.csv`

The repository includes code to:
1. authenticate with Hugging Face using `HF_TOKEN` or `huggingface-cli login`
2. download the raw files
3. build a VIL-only index CSV for Task 1
