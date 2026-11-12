import argparse
import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent / "LAVIS")

from utils.import_handler import transformers_import_monkey_patch
from utils.mfp import turn_off_rdkit_warnings

transformers_import_monkey_patch()
turn_off_rdkit_warnings()

import warnings

import lightning as pl
import lightning.pytorch.callbacks as plc
import torch
import torch_geometric
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])

from data_provider.stage1_dm import Stage1DM
from data_provider.stage1_kvplm_dm import Stage1KVPLMDM
from model.blip2_stage1 import Blip2Stage1

os.environ["OPENBLAS_NUM_THREADS"] = "1"
## for pyg bug
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
## for A5000 gpus
torch.set_float32_matmul_precision(
    "medium"
)  # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage1.load_from_checkpoint(
            args.init_checkpoint, device=args.devices, args=args
        )
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage1(args)

    print("total params:", sum(p.numel() for p in model.parameters()))

    tokenizer = model.blip2qformer.tokenizer
    # data
    if args.root.find("kv") >= 0:
        dm = Stage1KVPLMDM(
            args.num_workers,
            args.batch_size,
            args.root,
            args.text_max_len,
            args.graph_aug,
            args,
        )
    else:
        dm = Stage1DM(
            args.num_workers,
            args.batch_size,
            args.root,
            args.text_max_len,
            args.graph_aug,
            tokenizer,
            args,
        )
    model.val_match_loader = dm.val_match_loader
    model.test_match_loader = dm.test_match_loader

    callbacks = []
    # Step-based or epoch-based checkpointing
    if args.save_every_n_steps is not None:
        callbacks.append(
            plc.ModelCheckpoint(
                dirpath="all_checkpoints/" + args.filename + "/",
                filename="step_{step}",
                every_n_train_steps=args.save_every_n_steps,
                save_top_k=-1,
            )
        )
    else:
        callbacks.append(
            plc.ModelCheckpoint(
                dirpath="all_checkpoints/" + args.filename + "/",
                filename="{epoch:02d}",
                every_n_epochs=args.save_every_n_epochs,
                save_top_k=-1,
            )
        )

    # find_unused_parameters = (not args.gtm) or (not args.lm)
    # logger = CSVLogger(save_dir=f"./all_checkpoints/{args.filename}/")
    logger = WandbLogger(
        save_dir=f"./all_checkpoints/{args.filename}/", project="molca-stage1"
    )

    # Step-based or epoch-based validation
    trainer_kwargs = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "max_epochs": args.max_epochs,
        "callbacks": callbacks,
        "logger": logger,
    }

    if args.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = args.val_check_interval
    else:
        trainer_kwargs["check_val_every_n_epoch"] = args.check_val_every_n_epoch

    trainer = Trainer(**trainer_kwargs)
    if args.mode == "train":
        trainer.fit(model, datamodule=dm)
    elif args.mode == "eval":
        trainer.fit_loop.epoch_progress.current.completed = 49  ## avoid
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", type=str, default="stage1_test")
    # GPU
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # MM settings
    parser.add_argument(
        "--gtm",
        action="store_true",
        help="use graph-text matching or not",
        default=True,
    )
    parser.add_argument(
        "--lm", action="store_true", help="use language modeling or not", default=True
    )
    parser.add_argument(
        "--include_mfp",
        action="store_true",
        help="include molecular fingerprint for CKA constraint",
        default=False,
    )
    parser.add_argument(
        "--fp_type",
        type=str,
        choices=["morgan", "maccs", "rdkit"],
        default="morgan",
        help="fingerprint type for CKA loss (morgan, maccs, rdkit)",
    )
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=None,
        help="Run validation every N steps (if set, overrides check_val_every_n_epoch)",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (if set, overrides save_every_n_epochs)",
    )
    # parser.add_argument('--save_every_n_epochs', type=int, default=1)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM.add_model_specific_args(parser)
    # parser.set_defaults(accelerator='gpu',
    #                     devices='0,1,2,3',
    #                     precision='bf16',
    #                     max_epochs=50,
    #                     check_val_every_n_epoch=1)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, "=", v)
    print("=========================================")
    main(args)
