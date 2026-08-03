# Word2Vec server package (SGNS + CBOW-NS)

This directory contains only the code needed to preprocess a large plain-text corpus,
train the from-scratch PyTorch SGNS or CBOW with Negative Sampling model, resume
training, and evaluate the learned input embeddings on WordSim-353 and Google Analogy.

Hierarchical Softmax, ablation aggregation, notebooks, text8, checkpoints, and previous
run outputs are intentionally excluded.

## 1. Copy to the server

Copy the entire `word2vec_pytorch_server` directory. The large corpus must be a UTF-8 plain-text
file; whitespace is used as the tokenizer and text is lower-cased.

## 2. Environment

Install the PyTorch build appropriate for the server CUDA version first, then install
the remaining requirements. Installing `requirements.txt` directly also works when
the default PyTorch wheel matches the server.

```bash
cd word2vec_pytorch_server
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure

Edit `configs/sgns_server.yaml`, especially:

- `data.corpus_path`
- `data.processed_dir`
- `run.name`
- `training.batch_size`
- `training.precision` (`bf16` is recommended on supported server GPUs)
- `evaluation.max_vocabulary`

Use `configs/cbow_server.yaml` for CBOW. SGNS and CBOW may share the same
`data.processed_dir` when their corpus, `min_count`, split, and subsampling settings are
identical; preprocessing then needs to be run only once.

The preprocessing stage makes two streaming passes over the corpus. It holds the word
frequency dictionary in RAM, but writes encoded train/validation tokens directly to
disk as `uint32`/`uint64` binary arrays. Training memory therefore does not scale with
the number of generated center-context pairs.

## 4. Run

```bash
bash scripts/download_eval_data.sh
bash scripts/preprocess.sh configs/sgns_server.yaml
bash scripts/train.sh configs/sgns_server.yaml
bash scripts/eval.sh runs/sgns_large_corpus
```

CBOW uses the same commands with its own config and run name:

```bash
bash scripts/train.sh configs/cbow_server.yaml
bash scripts/eval.sh runs/cbow_large_corpus
```

Run in the background on a remote server if needed:

```bash
nohup bash scripts/train.sh configs/sgns_server.yaml > train_server.log 2>&1 &
tail -f train_server.log
```

Resume from the last checkpoint:

```bash
bash scripts/train.sh configs/sgns_server.yaml --resume runs/sgns_large_corpus/checkpoints/last.pt
```

## Outputs

Each run stores the resolved config, environment metadata, logs, per-step metrics,
`best.pt`, `last.pt`, and final evaluation JSON files under `runs/<run.name>/`.

Google Analogy accuracy is conditional on coverage. Always report accuracy together
with coverage, evaluated examples, and total examples. `max_vocabulary` also limits
the candidate set and therefore changes both coverage and accuracy.
