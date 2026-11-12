import os

import torch
from torch_geometric.data import Dataset
from transformers import BertTokenizer

from utils.mfp import smiles2mfp


def _read_primary_smiles(smiles_path):
    with open(smiles_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    return None


class RetrievalDataset(Dataset):
    def __init__(self, root, args):
        super(RetrievalDataset, self).__init__(root)
        self.root = root
        self.graph_aug = "noaug"
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root + "graph/")
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root + "text/")
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root + "smiles/")
        self.smiles_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        self.use_smiles = args.use_smiles
        self.include_mfp = getattr(args, "include_mfp", False)

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name, smiles_name = (
            self.graph_name_list[index],
            self.text_name_list[index],
            self.smiles_name_list[index],
        )
        assert (
            graph_name[len("graph_") : -len(".pt")]
            == text_name[len("text_") : -len(".txt")]
            == smiles_name[len("smiles_") : -len(".txt")]
        ), print(graph_name, text_name, smiles_name)

        graph_path = os.path.join(self.root, "graph", graph_name)
        data_graph = torch.load(graph_path)

        primary_smiles = None
        text = ""
        if self.use_smiles or self.include_mfp:
            smiles_path = os.path.join(self.root, "smiles", smiles_name)
            count = 0
            with open(smiles_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip("\n")
                    if not line:
                        continue
                    if primary_smiles is None:
                        primary_smiles = line
                    if self.use_smiles:
                        count += 1
                        if count == 1:
                            text = "This molecule is "
                        text += f" {line}"
                        if count > 1:
                            break
            if self.use_smiles and count:
                text += ". "

        text_path = os.path.join(self.root, "text", text_name)
        count = 0
        for line in open(text_path, "r", encoding="utf-8"):
            count += 1
            line = line.strip("\n")
            text += f" {line}"
            if count > 100:
                break
        text += "\n"
        # para-level
        text, mask = self.tokenizer_text(text)

        if self.include_mfp:
            if primary_smiles is None:
                smiles_path = os.path.join(self.root, "smiles", smiles_name)
                primary_smiles = _read_primary_smiles(smiles_path)
            if primary_smiles is None:
                raise ValueError(
                    f"Unable to read SMILES string for sample {graph_name}."
                )
            mfp = smiles2mfp(primary_smiles)
            return data_graph, text.squeeze(0), mask.squeeze(0), mfp

        return data_graph, text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(
            text=text,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            max_length=self.text_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = sentence_token["input_ids"]
        attention_mask = sentence_token["attention_mask"]
        return input_ids, attention_mask


class RetrievalDatasetKVPLM(Dataset):
    def __init__(self, root, args):
        super(RetrievalDatasetKVPLM, self).__init__(root)
        self.root = root
        self.graph_aug = "noaug"
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root + "graph/")
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root + "text/")
        self.text_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        self.use_smiles = args.use_smiles
        # Fingerprints are not attached on the KV-PLM (PCDes) retrieval path.
        self.include_mfp = False
        if self.use_smiles:
            smiles_dir = os.path.join(root, "smiles")
            if not os.path.isdir(smiles_dir) and self.include_mfp:
                raise FileNotFoundError(
                    f"SMILES directory not found at {smiles_dir} while include_mfp is True."
                )
            if os.path.isdir(smiles_dir):
                self.smiles_name_list = os.listdir(smiles_dir)
                self.smiles_name_list.sort()
                if self.include_mfp and len(self.smiles_name_list) != len(
                    self.graph_name_list
                ):
                    raise ValueError(
                        "SMILES and graph file counts do not match for KVPLM retrieval dataset."
                    )
            else:
                self.smiles_name_list = []
        else:
            self.smiles_name_list = []

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]

        graph_path = os.path.join(self.root, "graph", graph_name)
        data_graph = torch.load(graph_path)

        text = ""
        text_path = os.path.join(self.root, "text", text_name)
        count = 0
        for line in open(text_path, "r", encoding="utf-8"):
            count += 1
            line = line.strip("\n")
            text += f" {line}"
            break
        text += "\n"
        # para-level
        text, mask = self.tokenizer_text(text)

        if self.include_mfp:
            if not self.smiles_name_list:
                raise ValueError("SMILES filenames are not loaded for MFP computation.")
            smiles_name = self.smiles_name_list[index]
            if (
                graph_name[len("graph_") : -len(".pt")]
                != smiles_name[len("smiles_") : -len(".txt")]
            ):
                raise ValueError(
                    f"Graph file {graph_name} and SMILES file {smiles_name} do not align."
                )
            smiles_path = os.path.join(self.root, "smiles", smiles_name)
            primary_smiles = _read_primary_smiles(smiles_path)
            if primary_smiles is None:
                raise ValueError(
                    f"Unable to read SMILES string for sample {graph_name}."
                )
            mfp = smiles2mfp(primary_smiles)
            return data_graph, text.squeeze(0), mask.squeeze(0), mfp

        return data_graph, text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(
            text=text,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            max_length=self.text_max_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = sentence_token["input_ids"]
        attention_mask = sentence_token["attention_mask"]
        return input_ids, attention_mask
