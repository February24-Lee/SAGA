# Atomas + SAGA

SAGA extensions to **Atomas** (*Hierarchical Adaptive Alignment on Molecule-Text*, ICLR 2025). The original model jointly learns SMILES and text representations with hierarchical adaptive alignment; on top of it we add **SAGA** — a Centered Kernel Alignment (CKA) regularizer that aligns the learned embeddings with classical molecular-fingerprint (Tanimoto) similarity.

> Original code: <https://github.com/yikunpku/Atomas> · Project page: <https://yikunpku.github.io/_Atomas/>

## What SAGA adds

- **SAGA / CKA regularization** — an auxiliary loss aligning the SMILES representations with Morgan / MACCS / RDKit fingerprint similarity (`include_mfp`, `ckalosswt`, `fp_type`, `mfp_queue_size`).
- **PubChemSTM-distill pretraining** — a `pubchemstm_distill` dataset mode over our 34,365-pair filtered corpus (`data/PubchemSTM_distill/`).
- **ChEBI-20 captioning finetuning** — warm-start from a PubChemSTM-distill checkpoint (`_yamls/Finetune_ChEBI20.yaml`).
- **Migration to PyTorch Lightning 2.x**, pre-tokenization caching, and retrieval metrics (MRR).

The loss itself is in `utils/losses/` (CKA, Tanimoto Gram matrix) and
`models/atomas.py::_compute_mfp_cka_loss`; fingerprints are in `utils/fingerprint/`.

## Setup

```bash
uv sync   # Python ≥ 3.12, managed via uv (see pyproject.toml)
```

## Data

- **PubChemSTM-distill** (34,365 pairs) is included at `data/PubchemSTM_distill/train.txt`.
  How it was built and the terms it is provided under are in [`../DATA.md`](../DATA.md).
- **ChEBI-20** (validation, and captioning finetuning) is loaded from the Hugging Face
  dataset [`liupf/ChEBI-20-MM`](https://huggingface.co/datasets/liupf/ChEBI-20-MM).

## Training

Pretrain on PubChemSTM-distill with the SAGA loss (config: `_yamls/Pretrain_Atomas.yaml`):

```bash
# DDP over every visible GPU; the paper used 4 GPUs x --batch_size 4 (global 16)
python main.py --project Atomas --dataset pubchemstm_distill \
    --data_dir data --model_size base --task gentext \
    --batch_size 4 --max_epochs 100 --seed 42 --start_valid_epoch 1000 \
    --include_mfp --fp_type morgan --ckalosswt 100 --mfp_queue_size 128
```

`--start_valid_epoch 1000` keeps validation to retrieval metrics, as in the paper's
pre-training runs. Without it the config default (95) switches on generation-based
validation at epoch 95, which needs a SciBERT checkpoint under
`data/pretrained/SciBERT` and fails late in the run if it is absent.

`--ckalosswt` is λ in `L_total = L_base + λ·L_SAGA`, and `--mfp_queue_size` sets the
fingerprint queue. The loss is computed over the queued samples together with the
current mini-batch of each device (128 + 4 = 132 molecules above), and the batch is
added to the queue afterwards. Both options default to other values in `main.py`; pass
them explicitly to reproduce the paper.

Finetune for ChEBI-20 captioning, warm-starting from the distill checkpoint. The paper's
captioning runs used λ = 1000 for both pre-training and fine-tuning:

```bash
python main.py --config _yamls/Finetune_ChEBI20.yaml \
    --init_from_checkpoint <pubchemstm_distill_ckpt> \
    --include_mfp --fp_type morgan --ckalosswt 1000 --mfp_queue_size 128
```

Pre-tokenize datasets first to avoid CPU bottlenecks: see `scripts/cache_tokenized_dataset.py` and pass `--token_cache_dir`.

## Evaluation

Retrieval on the ChEBI-20 test split. Pass the model options the checkpoint was trained
with:

```bash
python eval.py --resume_from_checkpoint <model.ckpt> --model_size base --task gentext \
    --include_mfp --mfp_queue_size 128 --retrieval_only
```

## Citation

Please cite the SAGA paper (see the [repository root](../README.md#citation)) and the original Atomas paper:

```bibtex
@inproceedings{zhang2025atomas,
  title     = {Atomas: Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Generation},
  author    = {Zhang, Yikun and Ye, Geyan and Yuan, Chaohao and Han, Bo and Huang, Long-Kai and Yao, Jianhua and Liu, Wei and Rong, Yu},
  booktitle = {ICLR},
  year      = {2025}
}
```
