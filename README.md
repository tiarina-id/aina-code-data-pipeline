# Aina Code Data Pipeline

Pipeline untuk membuat dataset training Aina Code:

- collect dataset dari Hugging Face
- normalize base code dan instruct data
- filter data buruk/secret
- exact dedup
- tokenize dan pack pretrain dataset
- shard output dataset
- upload checkpoint dan final artifact ke S3

Repo ini hanya untuk **dataset preparation**. Training model sebaiknya di repo terpisah, misalnya `aina-code-training`.

## Output

Pretrain output:

```text
train-00000.bin
train-00001.bin
val-00000.bin
manifest.json
metadata.json
checkpoint/
```

SFT output:

```text
train-00000.jsonl
train-00001.jsonl
val-00000.jsonl
manifest.json
metadata.json
checkpoint/
```

## Setup

Fish local:

```fish
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
python -m pip install -e .
```

Bash server:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Environment

```bash
export HF_TOKEN="hf_xxx"
export HF_HOME=/data/hf-cache
export HF_DATASETS_CACHE=/data/hf-cache/datasets
aws sts get-caller-identity
```

## Local Build

Pretrain 3M local dataset:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_pretrain_3m_local_18m.yaml \
  --no-resume
```

SFT 3M local dataset:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_sft_3m_local.yaml \
  --no-resume
```

## Server Build

50M model dataset:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_pretrain_50m_server_1b.yaml \
  --resume

python scripts/build_dataset.py \
  --config configs/aina_code_sft_50m_server.yaml \
  --resume
```

500M model dataset:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_pretrain_500m_server_10b.yaml \
  --resume

python scripts/build_dataset.py \
  --config configs/aina_code_sft_500m_server.yaml \
  --resume
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Notes

- Pretrain uses tokenized binary shards.
- SFT uses JSONL messages shards.
- S3 checkpoint uses `checkpoint/READY.json` to avoid restoring half-uploaded checkpoint.
- For CPU preprocessing, PyTorch is not required.

