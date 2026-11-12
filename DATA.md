# PubChemSTM-distill corpus

`atomas/data/PubchemSTM_distill/train.txt` (and the identical copy under `MolCA/data/`) is
the pre-training corpus used in the paper: **34,365 SMILES–description pairs**, distilled
from PubChemSTM by CID de-duplication, a minimum description-length filter, and removal
of molecules that also occur in the ChEBI-20 or PCDes evaluation sets.

```
format  TSV with a header line: smiles <TAB> desc
lines   34,366  (header + 34,365 pairs)
sha256  a9244ce747f548e059c4e3e8e2958d2c7a8a5dd59421d389f43b69c889f35880
```

## Terms

The corpus is derived data and is **not** covered by this repository's MIT license. Its
main source, PubChemSTM, is released under the NVIDIA Source Code License, which limits
use to non-commercial research or evaluation and requires redistribution under the same
license. The corpus is provided here on those terms, for research use only. The full
list of sources is in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Filtering counts

`stats.json`, next to `train.txt`, records the counts of the run that produced it:

| Stage | Count |
| --- | --- |
| raw pairs | 408,250 |
| after CID merge | 219,645 |
| after length filter | 45,496 |
| ChEBI-20 SMILES in the filter | 32,539 |
| PCDes SMILES in the filter | 14,979 |
| pairs removed as duplicates | 11,131 |
| **final** | **34,365** |

## How it was built

Use the shipped `train.txt` to reproduce the paper. The steps below document how it was
made; all commands run inside `atomas/`.

**1. PubChemSTM.** Obtain `CID2text.json` and `CID2SMILES.csv` from the
[MoleculeSTM](https://github.com/chao1224/MoleculeSTM) release and place them under
`data/PubChemSTM_data/raw/`.

**2. ChEBI-20.** The filter reads the cached Arrow splits, which are built from
[`liupf/ChEBI-20-MM`](https://huggingface.co/datasets/liupf/ChEBI-20-MM):

```bash
python scripts/cache_tokenized_dataset.py --dataset ChEBI-20_data \
    --model-size base --cache-dir data/cache
```

**3. PCDes.** Place `align_smiles.txt` and `align_des_filt3.txt` from
[KV-PLM](https://github.com/thunlp/KV-PLM) under `data/PCdes/`, then run
`python scripts/prepare_pcdes.py`.

**4. Distill.**

```bash
python scripts/create_pubchemstm_distill.py \
    --pubchem-dir data/PubChemSTM_data/raw \
    --chebi20-dir data/cache/ChEBI-20_data/base \
    --pcdes-dir   data/PCdes \
    --output-dir  data/PubchemSTM_distill_rebuilt
```
