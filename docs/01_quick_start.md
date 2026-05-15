# Quick Start

This guide shows the expected high-level workflow for using the Home GPU GPT Training Toolkit.

The toolkit is designed to be run from the repository root.

---

## 1. Create a Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

## 2. Install Core Requirements

Install the lightweight core requirements:

```bash
pip install -r requirements-core.txt
```

PyTorch is intentionally not pinned in requirements-core.txt.

Install PyTorch separately according to your CUDA or CPU setup.

Use the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

A tested Pop!_OS + GTX1050 environment snapshot is available here:

`environment/tested_environment_popos_gtx1050.md`


---

## 3. Prepare a Wikipedia Dump

Download a Wikipedia XML dump such as:

https://dumps.wikimedia.org/trwiki/latest/

Example file:

`trwiki-latest-pages-articles.xml.bz2`

Large datasets are not included in this repository.

Place your dump file somewhere like:

`data/raw/trwiki-latest-pages-articles.xml.bz2`

---


## 4. Extract Wikipedia JSON Files


```bash
python scripts/data_preparation/01_extract_wikipedia_dump.py \
  --input data/raw/trwiki-latest-pages-articles.xml.bz2 \
  --output data/extracted/wikipedia \
  --processes 4
```


---

## 5. Build a JSONL Corpus

```bash
python scripts/data_preparation/02_build_wikipedia_jsonl.py \
  --input-dir data/extracted/wikipedia \
  --output data/processed/wikipedia_raw.jsonl
```


---

## 6. Clean the JSONL Corpus

```bash
python scripts/data_preparation/03_clean_wikipedia_jsonl.py \
  --input data/processed/wikipedia_raw.jsonl \
  --output data/processed/wikipedia_cleaned.jsonl \
  --min-word-count 50 \
  --max-word-length 25
```


---

## 7. Normalize the Corpus


```bash
python scripts/data_preparation/04_normalize_turkish_corpus.py \
  --input data/processed/wikipedia_cleaned.jsonl \
  --output data/processed/wikipedia_turkish_normalized.jsonl
```


---

## 8. Export Tokenizer Corpus

```bash
python scripts/data_preparation/05_export_tokenizer_corpus.py \
  --input data/processed/wikipedia_turkish_normalized.jsonl \
  --output data/tokenizer_corpus/output_tokenizer.txt
```


---

## 9. Train a Byte-Level BPE Tokenizer

```bash
python scripts/tokenizer/06_train_bytelevel_bpe_tokenizer.py \
  --corpus data/tokenizer_corpus/output_tokenizer.txt \
  --output tokenizer \
  --vocab-size 52003 \
  --min-frequency 2
```


---

## 10. Export Hugging Face GPT2TokenizerFast Files

```bash
python scripts/tokenizer/07_export_hf_gpt2_tokenizer.py \
  --input tokenizer \
  --output tokenizer
```


---

## 11. Split Corpus into Token-Safe Parts


```bash
python scripts/dataset_split/08_split_token_safe_parts.py \
  --input data/processed/wikipedia_turkish_normalized.jsonl \
  --tokenizer tokenizer \
  --output-dir data/split/wikipedia \
  --parts 120 \
  --max-tokens 512 \
  --min-tokens 100
```


---

## 12. Create Initial Schedule


```bash
python scripts/schedule/09_create_initial_schedule.py \
  --data-dir data/split/wikipedia \
  --dataset wikipedia \
  --output schedule.json \
  --eval-data data/eval/dev_set.jsonl
```


---

## 13. Add Token Counts and Max Steps


```bash
python scripts/schedule/10_generate_schedule_max_steps.py \
  --schedule schedule.json \
  --tokenizer tokenizer \
  --output schedule.json \
  --block-size 512 \
  --batch-size 1 \
  --safety-multiplier 1.1
```


---

## 14. Run Chained Training


Train one part:
```bash
python scripts/training/run_training_loop.py --parts 1
```

Train multiple parts:
```bash
python scripts/training/run_training_loop.py --parts 5
```

Interactive mode:
```bash
python scripts/training/run_training_loop.py
```


---

## Runtime Files

During training, the toolkit may create:

```text
checkpoints/
tmp/
logs_chain/
process.json
summary.json
state.json
schedule.json
```


---

## Notes

This toolkit is experimental.

Training on low-end GPUs can take a long time. The original workflow was tested with a GTX1050-class setup, but exact performance depends on your system, data size, PyTorch installation, and CUDA compatibility.