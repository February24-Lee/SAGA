# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch
import torch_geometric
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from data_provider.pretrain_dataset import GINPretrainDataset
from data_provider.retrieval_dataset import RetrievalDatasetKVPLM


class KVPLMCollater(object):
    def __init__(self, include_mfp: bool = False):
        self.include_mfp = include_mfp

    def __call__(self, batch):
        if self.include_mfp:
            data_list, text_list, mask_list, mfp_list = zip(*batch)
        else:
            data_list, text_list, mask_list = zip(*batch)
            mfp_list = None

        graph_batch = Batch.from_data_list(list(data_list))
        text_batch = torch.stack(text_list, dim=0)
        mask_batch = torch.stack(mask_list, dim=0)

        if not self.include_mfp:
            return graph_batch, text_batch, mask_batch

        stacked_mfp = []
        for mfp in mfp_list:
            if mfp.ndim == 1:
                stacked_mfp.append(mfp[None, :])
            else:
                stacked_mfp.append(mfp)
        mfp_batch = np.concatenate(stacked_mfp, axis=0)
        return graph_batch, text_batch, mask_batch, mfp_batch


class Stage1KVPLMDM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = "data/",
        text_max_len: int = 128,
        graph_aug: str = "dnodes",
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        # Fingerprints are not attached on the KV-PLM (PCDes) path: it is used for
        # retrieval evaluation, and the SAGA loss is skipped for these batches.
        self.include_mfp = False

        self._persistent_workers = self.num_workers > 0
        self.collate_fn = KVPLMCollater(self.include_mfp)
        print("Loading dataset")
        self.train_dataset = GINPretrainDataset(
            root + "/train/", text_max_len, graph_aug, args.text_aug
        )
        self.val_dataset = GINPretrainDataset(
            root + "/valid/", text_max_len, graph_aug, args.text_aug
        )
        self.val_dataset_match = RetrievalDatasetKVPLM(root + "/valid/", args).shuffle()
        if args.use_phy_eval:
            self.test_dataset_match = RetrievalDatasetKVPLM(
                root + "/phy_data/", args
            ).shuffle()
        else:
            self.test_dataset_match = RetrievalDatasetKVPLM(
                root + "/test/", args
            ).shuffle()
        self.val_match_loader = torch_geometric.loader.DataLoader(
            self.val_dataset_match,
            batch_size=self.match_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
        )
        self.test_match_loader = torch_geometric.loader.DataLoader(
            self.test_dataset_match,
            batch_size=self.match_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
        )
        self.train_dataset.tokenizer = self.val_dataset_match.tokenizer
        self.val_dataset.tokenizer = self.val_dataset_match.tokenizer

    # def setup(self, stage: str = None):
    #     self.train_dataset = self.dataset

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=self._persistent_workers,
            collate_fn=self.collate_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=self._persistent_workers,
            collate_fn=self.collate_fn,
        )

        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--match_batch_size", type=int, default=64)
        parser.add_argument("--use_smiles", action="store_true", default=False)
        parser.add_argument(
            "--root", type=str, default="data/PubChemDataset/PubChem-320k"
        )
        parser.add_argument("--text_max_len", type=int, default=128)
        parser.add_argument("--graph_aug", type=str, default="dnodes")
        parser.add_argument("--text_aug", action="store_true", default=False)
        parser.add_argument("--use_phy_eval", action="store_true", default=False)
        return parent_parser
