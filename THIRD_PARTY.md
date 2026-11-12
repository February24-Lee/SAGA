# Third-party components

Our own contributions — the SAGA loss, its integration into both backbones, the corpus
tooling and the documentation — are covered by the MIT [`LICENSE`](LICENSE). The
components below were written by others and keep their own terms.

## Code

| Component | Source | Commit | Where |
| --- | --- | --- | --- |
| Atomas (ICLR 2025) | <https://github.com/yikunpku/Atomas> | `6481c55c408311949f9b7418901b9f5586a20d0e` | `atomas/` |
| MolCA (EMNLP 2023) | <https://github.com/acharkq/MolCA> | `f728a47fae09536b9418ad21571c1b44a4a54e82` | `MolCA/` |
| LAVIS (BSD-3-Clause) | <https://github.com/salesforce/LAVIS> | `506965b9c4a18c1e565bd32acaccabe0198433f7` | `MolCA/LAVIS/` |

The first two commits of this repository import Atomas and MolCA at the commits above;
our changes come after.

`MolCA/LAVIS/` is a partial copy of LAVIS with small compatibility changes, installed by
MolCA as an editable dependency. Its copyright notice and license text are retained in
`MolCA/LAVIS/LICENSE.txt` and in the file headers.

## Training corpus

`{atomas,MolCA}/data/PubchemSTM_distill/train.txt` is derived from the sources below and
is **not** covered by our MIT license; see [`DATA.md`](DATA.md).

| Source | Obtained from | License |
| --- | --- | --- |
| PubChemSTM (`CID2text.json`, `CID2SMILES.csv`) | [MoleculeSTM](https://github.com/chao1224/MoleculeSTM) | NVIDIA Source Code License — non-commercial research or evaluation only; redistribution under the same license |
| ChEBI-20 (overlap filter, validation, test) | [`liupf/ChEBI-20-MM`](https://huggingface.co/datasets/liupf/ChEBI-20-MM) | MIT |
| PCDes (overlap filter) | [KV-PLM](https://github.com/thunlp/KV-PLM) | — |

The descriptions in PubChemSTM originate from PubChem annotations, which are themselves
aggregated from sources with their own terms.
