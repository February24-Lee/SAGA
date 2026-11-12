import argparse
import os
import warnings
from datetime import timedelta
from pathlib import Path

import lightning as pl
import torch
import yaml

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message=".*barrier.*device_id.*")
warnings.filterwarnings("ignore", message=".*sync_dist=True.*")

# Use Tensor Cores efficiently
torch.set_float32_matmul_precision("high")
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset.chEBI20MMDataset import ChEBI20MMDataset
from dataset.collate import AtomasCollator
from dataset.pubChemSTMDataset import (
    PubChemSTM_Datasets_SMILES,
    PubChemSTM_Distill_Dataset,
)
from models.atomas import Atomas

mol_data_directory = "./data"
model_data_directory = "./model_data"
prediction_directory = "./output_data"


def set_seed(seed: int):
    """Set seed for reproducibility."""
    seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    cache_root = Path(args.token_cache_dir) if args.token_cache_dir else None

    if args.dataset == "ChEBI-20_data":
        chebi_cache_root = (
            (cache_root / "ChEBI-20_data" / args.model_size)
            if cache_root is not None
            else None
        )
        train_data = ChEBI20MMDataset(
            "train",
            cache_path=chebi_cache_root,
            mfp_cache_path=args.mfp_cache_path,
        )
        valid_data = ChEBI20MMDataset(
            "validation",
            cache_path=chebi_cache_root,
            mfp_cache_path=args.mfp_cache_path,
        )
        test_data = ChEBI20MMDataset(
            "test",
            cache_path=chebi_cache_root,
            mfp_cache_path=args.mfp_cache_path,
        )

    elif args.dataset == "pubchemstm":
        # train_data = PubChem_Dataset(
        #     args.data_dir,
        #     "pubchemstm",
        #     args.split,
        # )
        train_cache = (
            (cache_root / "pubchemstm" / args.model_size / "train")
            if cache_root is not None
            else None
        )
        train_data = PubChemSTM_Datasets_SMILES(
            args.data_dir,
            cache_path=train_cache,
            mfp_cache_path=args.mfp_cache_path,
        )
        chebi_cache_root = (
            (cache_root / "ChEBI-20_data" / args.model_size)
            if cache_root is not None
            else None
        )
        valid_data = ChEBI20MMDataset("validation", cache_path=chebi_cache_root)

        test_data = None

    elif args.dataset == "pubchemstm_distill":
        train_cache = (
            (cache_root / "pubchemstm_distill" / args.model_size / "train")
            if cache_root is not None
            else None
        )
        train_data = PubChemSTM_Distill_Dataset(
            args.data_dir,
            cache_path=train_cache,
            mfp_cache_path=args.mfp_cache_path,
        )
        chebi_cache_root = (
            (cache_root / "ChEBI-20_data" / args.model_size)
            if cache_root is not None
            else None
        )
        valid_data = ChEBI20MMDataset("validation", cache_path=chebi_cache_root)

        test_data = None
    else:
        raise Exception("choose pubchemstm, pubchemstm_distill, or ChEBI_20_data")

    # Create collator (with fingerprint if enabled)
    train_collator = AtomasCollator(include_mfp=args.include_mfp, fp_type=args.fp_type)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=train_collator,
        multiprocessing_context="spawn",
        persistent_workers=True if args.num_workers > 0 else False,
    )
    if valid_data is not None:
        valid_loader = DataLoader(
            valid_data,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            multiprocessing_context="spawn",
        )
    else:
        valid_loader = None

    if test_data is not None:
        test_loader = DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            multiprocessing_context="spawn",
        )
    else:
        test_loader = None

    model = Atomas(
        args=args,
    )

    if args.init_from_checkpoint:
        if args.resume_from_checkpoint:
            raise ValueError(
                "--init_from_checkpoint and --resume_from_checkpoint are mutually exclusive."
            )
        print(f"[init] loading weights from {args.init_from_checkpoint}")
        ckpt = torch.load(args.init_from_checkpoint, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[init] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[init] missing keys (first 10): {missing[:10]}")
        if unexpected:
            print(f"[init] unexpected keys (first 10): {unexpected[:10]}")

        if getattr(model, "include_mfp", False):
            print("[init] resetting CKA queue buffers")
            model.mfp_queue.zero_()
            model.smiles_cka_queue.zero_()
            model.mfp_queue_ptr.zero_()
            model.mfp_queue_filled.fill_(False)

    model_save_path = os.path.join(str(model_data_directory) + "ckpt", args.version)
    os.makedirs(model_save_path, exist_ok=True)

    if valid_loader is not None:
        if args.task == "genmol":
            # Select best ckpt by Morgan FTS (continuous, chemistry-aware), not
            # character-level SMILES BLEU. Exact/Validity shown for context since
            # FTS is averaged over valid molecules only.
            monitor = "valid_morgan_sims_score"
            filename = (
                args.version
                + "-{epoch:02d}-{step:02d}-{train_loss_tol:.4f}"
                + (
                    "-{valid_morgan_sims_score:.3f}-{valid_Exact:.3f}-{valid_validity_score:.3f}"
                )
            )
        elif args.skip_retrieval:
            monitor = "valid_BLEU2"
            filename = (
                args.version
                + "-{epoch:02d}-{step:02d}-{train_loss_tol:.4f}"
                + ("-{valid_BLEU2:.3f}-{valid_BLEU4:.3f}-{valid_ROUGE1:.3f}")
            )
        else:
            monitor = "valid_TtS_MRR"
            filename = (
                args.version
                + "-{epoch:02d}-{step:02d}-{train_loss_tol:.4f}"
                + ("-{valid_TtS_MRR:.3f}-{valid_BLEU2:.3f}-{valid_ROUGE1:.3f}")
            )
        mode = "max"
        save_top_k = 3
    else:
        monitor = "train_loss_tol"
        mode = "min"
        filename = args.version + "-{epoch:02d}-{step:02d}-{train_loss_tol:.4f}"
        save_top_k = -1

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    # Best-ckpt selector (e.g. Morgan FTS for genmol). Does NOT own last.ckpt.
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_last=False,
        save_top_k=save_top_k,
    )
    # Time-based last.ckpt, so a run can be continued with --resume_from_checkpoint.
    resume_callback = ModelCheckpoint(
        dirpath=model_save_path,
        save_last=True,
        save_top_k=0,
        train_time_interval=timedelta(minutes=15),
    )

    # Configure logger and trainer based on dry_run mode
    if args.dry_run:
        # Need enough batches to test retrieval metrics (sim_matrix needs samples)
        dry_run_batches = 5
        print("=" * 50)
        print(f"DRY RUN MODE: Running quick test with {dry_run_batches} batches")
        print("=" * 50)
        wandb_logger = None
        callbacks = []
        trainer_kwargs = dict(
            default_root_dir=model_data_directory,
            logger=False,
            callbacks=callbacks,
            accumulate_grad_batches=args.accumulate_grad_batches,
            accelerator=args.accelerator,
            strategy="ddp",
            num_nodes=args.num_nodes,
            fast_dev_run=dry_run_batches,
            precision=args.precision,
            enable_checkpointing=False,
        )
    else:
        wandb_save_path = os.path.join(
            str(model_data_directory) + "/wandb", args.version
        )
        os.makedirs(wandb_save_path, exist_ok=True)
        wandb_logger = WandbLogger(
            project=args.project,
            name=args.version,
            save_dir=wandb_save_path,
            config=args,
        )
        lr_logger = LearningRateMonitor()
        callbacks = [lr_logger, checkpoint_callback, resume_callback]
        trainer_kwargs = dict(
            default_root_dir=model_data_directory,
            logger=wandb_logger,
            callbacks=callbacks,
            accumulate_grad_batches=args.accumulate_grad_batches,
            accelerator=args.accelerator,
            strategy="ddp",
            num_nodes=args.num_nodes,
            gradient_clip_val=args.gradient_clip_val,
            log_every_n_steps=args.log_every_n_steps,
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            val_check_interval=args.val_check_interval,
            precision=args.precision,
            # https://github.com/Lightning-AI/pytorch-lightning/pull/16745
            # track_grad_norm=args.track_grad_norm,
        )

    trainer = pl.Trainer(**trainer_kwargs)
    if valid_loader is not None:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=args.resume_from_checkpoint,
        )
    else:
        trainer.fit(
            model, train_dataloaders=train_loader, ckpt_path=args.resume_from_checkpoint
        )
    # No post-fit test pass: validation already runs the same generation pipeline.
    # Use eval.py for a standalone test evaluation.


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="_yamls/Pretrain_Atomas.yaml",
        help="Path to YAML config file. Defaults to Pretrain_Atomas.yaml.",
    )
    pre_args, _ = pre_parser.parse_known_args()

    with open(pre_args.config) as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--project", type=str, default="Atomas")
    parser.add_argument("--version", type=str, default=config["version"])
    ########## for dataset ##########
    parser.add_argument("--data_dir", type=str, default=mol_data_directory)
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(config["dataset"]),
        choices=["pubchemstm", "pubchemstm_distill", "ChEBI-20_data"],
    )
    parser.add_argument("--split", type=str, default="distilled")
    parser.add_argument("--train_split", type=str, default=config["train_split"])
    parser.add_argument("--valid_split", type=str, default=config["valid_split"])
    parser.add_argument("--test_split", type=str, default=config["test_split"])
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=config.get("eval_batch_size", 1),
        help="Batch size for validation/test loaders (separate from training batch_size).",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_lenth", type=int, default=config["max_lenth"])
    parser.add_argument("--token_cache_dir", type=str, default=None)
    ########## for model ##########
    parser.add_argument(
        "--model_size",
        type=str,
        default=config["model_size"],
        choices=["base", "large"],
    )
    parser.add_argument("--queue_size", type=int, default=config["queue_size"])
    parser.add_argument(
        "--task", type=str, default=config["task"], choices=["genmol", "gentext"]
    )
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--tsclosswt", type=float, default=config["tsclosswt"])
    parser.add_argument("--lmlosswt", type=float, default=config["lmlosswt"])
    parser.add_argument("--wtilosswt", type=float, default=config["wtilosswt"])
    # MFP-guided CKA loss arguments
    parser.add_argument(
        "--include_mfp",
        action="store_true",
        default=config.get("include_mfp", False),
        help="Enable MFP-guided CKA loss",
    )
    parser.add_argument(
        "--mfp_cache_path",
        type=str,
        default=config.get("mfp_cache_path", None),
        help="Path to pre-computed MFP cache (.npy file). If not provided, MFP will be computed on-the-fly.",
    )
    parser.add_argument(
        "--mfp_queue_size",
        type=int,
        default=config.get("mfp_queue_size", 1024),
        help="Queue size for MFP (should be <= queue_size)",
    )
    parser.add_argument(
        "--ckalosswt",
        type=float,
        default=config.get("ckalosswt", 1.0),
        help="Weight for CKA loss",
    )
    parser.add_argument(
        "--fp_type",
        type=str,
        choices=["morgan", "maccs", "rdkit"],
        default=config.get("fp_type", "morgan"),
        help="Fingerprint type for CKA loss (morgan, maccs, rdkit)",
    )
    parser.add_argument(
        "--local_loss",
        action="store_true",
        default=False,
        help="Use local batch only for loss computation (disable all-gather for memory efficiency)",
    )
    parser.add_argument(
        "--skip_retrieval",
        action="store_true",
        default=config.get("skip_retrieval", False),
        help="Skip retrieval feature extraction and TtS/StT metric computation during validation/test.",
    )
    parser.add_argument("--textencoder", type=str, default="molt5")
    parser.add_argument(
        "--encode_text_lr", type=float, default=config["encode_text_lr"]
    )
    parser.add_argument(
        "--encode_smiles_lr", type=float, default=config["encode_smiles_lr"]
    )
    parser.add_argument("--molt5_lr", type=float, default=config["molt5_lr"])
    parser.add_argument("--text_lr_scale", type=float, default=config["text_lr_scale"])
    parser.add_argument(
        "--smiles_lr_scale", type=float, default=config["smiles_lr_scale"]
    )
    parser.add_argument("--decay", type=float, default=config["decay"])
    ########## for train ##########
    parser.add_argument("--precision", default=config["precision"])
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=config["accumulate_grad_batches"]
    )
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=config["max_epochs"])
    parser.add_argument(
        "--start_valid_epoch", type=int, default=config["start_valid_epoch"]
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=config.get("check_val_every_n_epoch", 1),
        help="Run validation every N epochs (Lightning Trainer arg).",
    )

    def _val_check_interval_type(v):
        # Lightning accepts either int (steps) or float (fraction of epoch).
        f = float(v)
        return int(f) if f >= 1.0 and f == int(f) else f

    parser.add_argument(
        "--val_check_interval",
        type=_val_check_interval_type,
        default=config.get("val_check_interval", 1.0),
        help="Run validation every N training steps (int) or every fraction of an epoch (float 0~1). 1.0 = once per epoch.",
    )
    parser.add_argument("--track_grad_norm", type=int, default=-1)
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=os.path.join(
            prediction_directory, parser.parse_known_args()[0].version
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=config.get("resume_from_checkpoint", None),
        help="Resume an interrupted run (Lightning ckpt_path: restores epoch, optimizer, scheduler state).",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=config.get("init_from_checkpoint", None),
        help="Warm-start weights only (no optimizer/epoch restore). For PubChemSTM-distill -> ChEBI-20 finetune.",
    )
    parser.add_argument("--seed", type=int, default=42)
    # Dry run for testing
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Run a quick test with limited batches (no wandb logging)",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    print(args)
    ########## start train ##########
    train(args)


if __name__ == "__main__":
    main()
