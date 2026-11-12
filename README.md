# SAGA: Structure-Aware Guided Alignment for Molecule–Text Models

Code for **“Classical structural priors improve neural molecular representations in
molecule language models”** (Journal of Chemical Information and Modeling, 2026).

**SAGA** is a training-only auxiliary loss that aligns the representations learned by a
molecule–text model with **classical molecular-fingerprint similarity**, using Centered
Kernel Alignment (CKA) between a Gram matrix of embeddings and a Tanimoto Gram matrix of
fingerprints:

```
L_SAGA  = 1 - CKA(K_embed, K_fingerprint)
L_total = L_base + λ · L_SAGA
```

Nothing changes at inference time: the loss only shapes the embedding geometry during
training.

This repository contains what is needed to train with the SAGA loss on the two backbones
used in the paper.

| Directory | Backbone | CKA sample |
| --- | --- | --- |
| [`atomas/`](atomas/) | [Atomas](https://github.com/yikunpku/Atomas) (ICLR 2025) | fingerprint queue + device mini-batch |
| [`MolCA/`](MolCA/) | [MolCA](https://github.com/acharkq/MolCA) (EMNLP 2023) | mini-batch gathered across GPUs |

## Where the SAGA code lives

| Path | What it is |
| --- | --- |
| `atomas/utils/losses/cka.py` | CKA with the unbiased HSIC estimator; `MFPGuidedCKALoss` |
| `atomas/utils/losses/tanimoto.py` | Tanimoto Gram matrix over fingerprints |
| `atomas/utils/fingerprint/` | Morgan / MACCS / RDKit fingerprints |
| `atomas/models/atomas.py` → `_compute_mfp_cka_loss` | the queue-based CKA sample and the loss |
| `MolCA/utils/cka.py`, `MolCA/utils/loss.py` | CKA and `MiniBatchCKALoss` |
| `MolCA/utils/mfp.py` | fingerprint computation |
| `MolCA/model/blip2qformer.py` | the CKA term added to the stage-1 objective |

Fingerprints are Morgan (radius 2, 2048 bits) by default; MACCS (167) and RDKit
topological (2048) are selectable with `--fp_type`.

The first two commits import the upstream source of each backbone and every later
commit is ours, so `git diff` against those two commits shows what we changed.

## Getting started

Each backbone is a standalone project managed with [uv](https://docs.astral.sh/uv/):

```bash
cd atomas      # or: cd MolCA
uv sync
```

Training with the configuration used in the paper (Morgan fingerprints, λ = 100):

```bash
# Atomas — 4 GPUs x batch 4; CKA over the 128-entry queue + device batch (n = 132)
cd atomas
python main.py --project Atomas --dataset pubchemstm_distill \
    --data_dir data --model_size base --task gentext \
    --batch_size 4 --max_epochs 100 --seed 42 --start_valid_epoch 1000 \
    --include_mfp --fp_type morgan --ckalosswt 100 --mfp_queue_size 128

# MolCA stage 1 — 3 GPUs x batch 64; CKA over the gathered batch (n = 192)
cd MolCA
python stage1.py --root data/PubchemSTM_distill/ --gtm --lm --devices '0,1,2' \
    --mode train --filename stage1 --num_query_token 8 --tune_gnn --max_epochs 100 \
    --include_mfp --fp_type morgan --cka_weight 100
```

The CKA sample size n depends on the number of GPUs and the batch size, so a baseline
has to be matched to the same n. Details, data layout and evaluation are in
[`atomas/README.md`](atomas/README.md) and [`MolCA/README.md`](MolCA/README.md).

## Training corpus

`{atomas,MolCA}/data/PubchemSTM_distill/train.txt` is the pre-training corpus used in
the paper: 34,365 SMILES–description pairs distilled from PubChemSTM. How it was built,
its checksum, and the terms it is provided under are in [`DATA.md`](DATA.md).

## Citation

```bibtex
@article{lee2026saga,
  title   = {Classical structural priors improve neural molecular representations
             in molecule language models},
  author  = {Lee, Donghyeok and Park, Sangkwon and Chang, Munyoung and Yoon, Sungroh},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026}
}
```

Please also cite the backbone you build on — [Atomas](atomas/README.md#citation) or
[MolCA](MolCA/README.md#citation).

## License and attribution

Our contributions are MIT-licensed; see [`LICENSE`](LICENSE). The backbones, the vendored
LAVIS code and the corpus sources belong to their authors and keep their own terms; see
[`THIRD_PARTY.md`](THIRD_PARTY.md).
