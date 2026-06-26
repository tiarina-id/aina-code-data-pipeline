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

## VM Setup

Syarat awal:

- Pasang IAM role ke EC2 yang punya akses S3 bucket `aina-code`.
- Pastikan repo ini sudah berisi `setup.sh`.
- `AINA_AUTO_CONFIRM=1` akan format disk kosong yang terdeteksi untuk `/data`.

Clone repo dan jalankan setup:

```bash
cd ~
git clone https://github.com/tiarina-id/aina-code-data-pipeline.git preproc-pipeline
cd ~/preproc-pipeline

SWAP_GB=8 AINA_AUTO_CONFIRM=1 bash setup.sh
```

Setelah setup selesai:

```bash
source ~/.bashrc
export HF_TOKEN="hf_xxx"
export AWS_DEFAULT_REGION=ap-southeast-3

aws sts get-caller-identity
aws s3 ls s3://aina-code

sudo apt-get install -y tmux jq
deactivate 2>/dev/null || true
tmux new -s preproc

source .venv/bin/activate
```

Mini test 3M 1K:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_3m_1k_pretrain.yaml \
  --resume \
  --num-workers 6 \
  --worker-batch-size 32

python scripts/build_dataset.py \
  --config configs/aina_code_3m_1k_sft.yaml \
  --resume \
  --num-workers 6 \
  --worker-batch-size 32

aws s3 ls s3://aina-code/v1/datasets/aina-1-code-3m-1k/ --recursive --summarize
```

Opsional hapus checkpoint dataset setelah final output aman:

```bash
aws s3 rm s3://aina-code/v1/datasets/aina-1-code-3m-1k/pretrain/checkpoint/ --recursive
aws s3 rm s3://aina-code/v1/datasets/aina-1-code-3m-1k/sft/checkpoint/ --recursive
aws s3 rm s3://aina-code/v1/datasets/aina-1-code-3m-1k/pretrain/metadata.partial.json
aws s3 rm s3://aina-code/v1/datasets/aina-1-code-3m-1k/sft/metadata.partial.json
```

## Environment

```bash
export HF_TOKEN="hf_xxx"
export HF_HOME=/data/aina-code/hf-cache
export HF_DATASETS_CACHE=/data/aina-code/hf-cache/datasets
aws sts get-caller-identity
```

## Local Build

Pretrain 3M 1K mini dataset:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_3m_1k_pretrain.yaml \
  --no-resume
```

SFT 3M 1K mini dataset:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_3m_1k_sft.yaml \
  --no-resume
```

## Server Build

50M 2K model dataset:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_50m_2k_pretrain.yaml \
  --resume

python scripts/build_dataset.py \
  --config configs/aina_code_50m_2k_sft.yaml \
  --resume
```

500M 8K model dataset:

```bash
python scripts/build_dataset.py \
  --config configs/aina_code_500m_8k_pretrain.yaml \
  --resume

python scripts/build_dataset.py \
  --config configs/aina_code_500m_8k_sft.yaml \
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
