import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.append(Path(__file__).parent / "LAVIS")

from utils.import_handler import transformers_import_monkey_patch
from utils.mfp import turn_off_rdkit_warnings

transformers_import_monkey_patch()
turn_off_rdkit_warnings()

import lightning as pl
import lightning.pytorch.callbacks as plc
import torch
import torch_geometric
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger, WandbLogger

torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])

from data_provider.stage2_dm import Stage2DM
# Stage2CheBIDM and IupacDM are imported lazily below — they pull in optional
# heavy deps (ogb) that are not needed for the default Stage2DM workflow.
from model.blip2_stage2 import Blip2Stage2

# torch.set_default_dtype(torch.float16)

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
        model = Blip2Stage2.load_from_checkpoint(
            args.init_checkpoint, strict=False, args=args
        )
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = Blip2Stage2(args)
        ckpt = torch.load(args.stage2_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)

    print("total params:", sum(p.numel() for p in model.parameters()))

    if args.opt_model.find("galactica") >= 0 or args.opt_model.find("t5") >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    elif args.opt_model.find("llama") >= 0 or args.opt_model.find("vicuna") >= 0:
        tokenizer = model.blip2opt.llm_tokenizer
    else:
        raise NotImplementedError
    # data
    if args.iupac_prediction:
        from data_provider.iupac_dm import IupacDM  # lazy: pulls optional deps
        dm = IupacDM(
            args.mode,
            args.num_workers,
            args.batch_size,
            args.root,
            args.text_max_len,
            tokenizer,
            args,
        )
    else:
        if args.root.lower().find("chebi") >= 0:
            from data_provider.stage2_chebi_dm import Stage2CheBIDM  # lazy: needs ogb
            dm = Stage2CheBIDM(
                args.mode,
                args.num_workers,
                args.batch_size,
                args.root,
                args.text_max_len,
                tokenizer,
                args,
            )
        else:
            dm = Stage2DM(
                args.mode,
                args.num_workers,
                args.batch_size,
                args.root,
                args.text_max_len,
                tokenizer,
                args,
            )

    callbacks = []
    # Keep the best checkpoint (by the monitored metric) and last.ckpt.
    # --save_every_n_epochs is accepted but not used.
    if args.save_every_n_epochs and args.save_every_n_epochs != 10:
        print(
            f"[ckpt] --save_every_n_epochs={args.save_every_n_epochs} ignored; "
            f"using best+last save policy (monitor='{args.monitor_metric}', "
            f"mode='{args.monitor_mode}')."
        )
    # The metric name may contain '/', so it is left out of the filename.
    callbacks.append(
        plc.ModelCheckpoint(
            dirpath="all_checkpoints/" + args.filename + "/",
            filename="best-{epoch:02d}",
            monitor=args.monitor_metric,
            mode=args.monitor_mode,
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
    )
    if len(args.devices.split(",")) > 1:
        if args.strategy_name == "fsdp":
            from lightning.pytorch.strategies import FSDPStrategy
            strategy = FSDPStrategy()
        elif args.strategy_name == "deepspeed":
            from lightning.pytorch.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=3)
        else:
            # Default DDP strategy, as in stage 1.
            strategy = "auto"
    else:
        strategy = "auto"
        args.devices = eval(args.devices)
    if args.logger == "wandb":
        logger = WandbLogger(
            save_dir=f"./all_checkpoints/{args.filename}/",
            project=args.wandb_project,
            name=args.filename,
        )
    else:
        logger = CSVLogger(save_dir=f"./all_checkpoints/{args.filename}/")
    # trainer = Trainer.from_argparse_args(args,
    #                                      callbacks=callbacks,
    #                                      strategy=strategy,
    #                                      logger=logger,
    #                                     #  limit_train_batches=100,
    #                                      )
    # argparse gives floats; cast to int when the user passed an integer >1.
    # Critical: 1.0 must stay a float (= "use all batches" in PL); only
    # values strictly > 1.0 are intended as concrete batch counts.
    def _maybe_int(v):
        return int(v) if isinstance(v, float) and v > 1.0 else v
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=_maybe_int(args.limit_train_batches),
        limit_val_batches=_maybe_int(args.limit_val_batches),
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
    )
    if args.mode in {"pretrain", "ft"}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode == "eval":
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="stage2_test")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # MM settings
    parser.add_argument("--mode", type=str, default="pretrain")
    parser.add_argument("--strategy_name", type=str, default=None)
    parser.add_argument("--iupac_prediction", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    parser = Stage2DM.add_model_specific_args(parser)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="Limit training batches per epoch (int=batches, float=fraction). Useful for smoke tests.",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="Limit validation batches per epoch (int=batches, float=fraction).",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "csv"],
        help="Trainer logger backend (default: wandb).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="molca-stage2",
        help="W&B project name when --logger=wandb.",
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="val molecule loss/dataloader_idx_0",
        help="Metric name to monitor for best-ckpt selection. Default targets the "
             "molecule-loss val dataloader (Lightning appends '/dataloader_idx_0' "
             "automatically because val_dataloader() returns multiple loaders). "
             "ROUGE/BLEU/METEOR are only logged when (epoch+1) %% caption_eval_epoch == 0 "
             "— set caption_eval_epoch=1 if you want to monitor those.",
    )
    parser.add_argument(
        "--monitor_mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Direction for --monitor_metric ('min' for loss, 'max' for ROUGE/BLEU/METEOR).",
    )
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, "=", v)
    print("=========================================")
    return args


if __name__ == "__main__":
    main(get_args())
