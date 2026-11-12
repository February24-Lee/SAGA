# MolCA + SAGA

SAGA extensions to **MolCA** (*Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter*, EMNLP 2023). MolCA couples a 2D graph encoder to a language model through a cross-modal projector for molecule–text retrieval and captioning; on top of it we add **SAGA** — a Centered Kernel Alignment (CKA) regularizer that aligns the learned representations with classical molecular-fingerprint (Tanimoto) similarity.

> Original code: <https://github.com/acharkq/MolCA> · Paper: <https://arxiv.org/abs/2310.12798>

## What SAGA adds

- **SAGA / CKA regularization** in stage 1 (retrieval pretraining): `--include_mfp --fp_type {morgan,maccs,rdkit} --cka_weight`.
- **PubChemSTM-distill pretraining** — our 34,365-pair filtered corpus (`data/PubchemSTM_distill/`), with ChEBI-20 used for validation/test.
- **Migration to PyTorch Lightning 2.x**, retrieval metrics (MRR / MeanR), and an end-to-end captioning script (`script/run_chebi_full.sh`).

The loss itself is in `utils/cka.py` and `utils/loss.py`, fingerprints are in
`utils/mfp.py`, and the term is added to the stage-1 objective in
`model/blip2qformer.py`.

## Setup

```bash
uv sync   # Python ≥ 3.12, managed via uv (see pyproject.toml)
```

`uv sync` also installs the vendored `LAVIS/` (the BLIP-2 building blocks MolCA depends on) as an editable package **with its full dependency set** (timm, decord, transformers, omegaconf, …), pinned in `uv.lock` — so no manual LAVIS setup is required. The GIN graph-encoder weights are bundled in `gin_pretrained/graphcl_80.pth`. Fetch the nltk corpus once: `python -c "import nltk; nltk.download('wordnet')"`.

## Data

- **PubChemSTM-distill** (34,365 pairs) is included at `data/PubchemSTM_distill/train.txt`.
  How it was built and the terms it is provided under are in [`../DATA.md`](../DATA.md).
- **ChEBI-20** (stage-1 validation/test, captioning) and **PCDes / MoMu** (`kv_data`):
  unzip `data/dataset.zip` from the [original MolCA repository](https://github.com/acharkq/MolCA)
  under `data/`, which gives `data/ChEBI-20_data/` and `data/kv_data/`.
- **PubChem324k** (original MolCA pre-training set, optional): download from the
  [MolCA HF dataset](https://huggingface.co/datasets/acharkq/PubChem324kV2) into `data/`.

## Training

**Stage 1 — retrieval pretraining with SAGA** (CKA aligns the graph representations with fingerprint similarity):

```bash
# The paper used 3 GPUs x --batch_size 64 (default) for 100 epochs. CKA is computed
# over the batch gathered across GPUs, so the CKA sample size is n = 192
python stage1.py --root data/PubchemSTM_distill/ --gtm --lm --devices '0,1,2' \
    --mode train --filename stage1 --num_query_token 8 --tune_gnn --max_epochs 100 \
    --include_mfp --fp_type morgan --cka_weight 100
```

`--cka_weight` is λ in `L_total = L_base + λ·L_SAGA`; the paper's reference
configuration is Morgan fingerprints with λ=100. Changing `--devices` or
`--batch_size` changes n, so a baseline has to be matched to the same n. Stage 1
needs at least two GPUs: the contrastive loss reads the DDP rank. Checkpoints are
written every 10 epochs to `all_checkpoints/<filename>/epoch=NN.ckpt`, so the final
one above is `epoch=99.ckpt`.

**Stage 2 — captioning** (projector pretraining on the distill corpus → LoRA finetuning
on ChEBI-20). This is the original MolCA stage-2 recipe started from the stage-1
checkpoint; the SAGA loss is not applied here. `script/run_chebi_full.sh` wraps both
steps:

```bash
# Settings used in the paper: 3 GPUs, micro-batch 16, gradient accumulation 4
script/run_chebi_full.sh --stage1_path all_checkpoints/stage1/epoch=99.ckpt \
    --devices "0,1,2" --batch_size 16 --accumulate_grad_batches 4
```

See `script/run_chebi_full.sh --help` for all overrides.

## Evaluation

```bash
# Molecule–text retrieval on ChEBI-20 (validation and test); stage 1 needs two or more
# GPUs. Pass --include_mfp for a checkpoint trained with it.
python stage1.py --root data/PubchemSTM_distill/ --gtm --lm --devices '0,1' --mode eval \
    --filename stage1_eval --init_checkpoint all_checkpoints/stage1/epoch=99.ckpt \
    --num_query_token 8 --include_mfp

# ChEBI-20 captioning:
python stage2.py --root data/ChEBI-20_data --devices '[0]' --mode eval \
    --filename chebi_eval --stage2_path <ckpt> --opt_model facebook/galactica-1.3b \
    --llm_tune lora --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn
```

## Citation

Please cite the SAGA paper (see the [repository root](../README.md#citation)) and the original MolCA paper:

```bibtex
@inproceedings{liu2023molca,
  title     = {MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter},
  author    = {Liu, Zhiyuan and Li, Sihang and Luo, Yanchen and Fei, Hao and Cao, Yixin and Kawaguchi, Kenji and Wang, Xiang and Chua, Tat-Seng},
  booktitle = {EMNLP},
  year      = {2023}
}
```
