import argparse
import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.data_utils import GridSeqDataset
from modules.comp.comp_d_net_pl import comp_d_net_pl
from modules.mv.mv_depth_net_pl import mv_depth_net_pl
from modules.mono.depth_net_pl import depth_net_pl


def collate_gridseq(batch):
    # batch: list of dicts from GridSeqDataset.__getitem__
    coll = {}
    # Required keys
    coll["ref_img"] = torch.from_numpy(np.stack([b["ref_img"] for b in batch], axis=0))
    coll["src_img"] = torch.from_numpy(np.stack([b["src_img"] for b in batch], axis=0))
    coll["ref_pose"] = torch.from_numpy(np.stack([b["ref_pose"] for b in batch], axis=0))
    coll["src_pose"] = torch.from_numpy(np.stack([b["src_pose"] for b in batch], axis=0))
    coll["ref_depth"] = torch.from_numpy(np.stack([b["ref_depth"] for b in batch], axis=0))
    if "src_depth" in batch[0]:
        coll["src_depth"] = torch.from_numpy(np.stack([b["src_depth"] for b in batch], axis=0))
    # Optional masks
    if "ref_mask" in batch[0] and batch[0]["ref_mask"] is not None:
        coll["ref_mask"] = torch.from_numpy(np.stack([b["ref_mask"] for b in batch], axis=0))
    else:
        coll["ref_mask"] = None
    if "src_mask" in batch[0] and batch[0]["src_mask"] is not None:
        coll["src_mask"] = torch.from_numpy(np.stack([b["src_mask"] for b in batch], axis=0))
    else:
        coll["src_mask"] = None
    return coll


def build_datasets(dataset_root: str, dataset_name: str, L: int, depth_suffix: str):
    dataset_dir = os.path.join(dataset_root, dataset_name)
    split_file = os.path.join(dataset_dir, "split.yaml")
    with open(split_file, "r") as f:
        split = yaml.safe_load(f)

    train_set = GridSeqDataset(
        dataset_dir=dataset_dir,
        scene_names=split["train"],
        L=L,
        depth_dir=dataset_dir,
        depth_suffix=depth_suffix,
        add_rp=False,
        roll=0,
        pitch=0,
    )
    val_set = GridSeqDataset(
        dataset_dir=dataset_dir,
        scene_names=split["val"],
        L=L,
        depth_dir=dataset_dir,
        depth_suffix=depth_suffix,
        add_rp=False,
        roll=0,
        pitch=0,
    )
    return train_set, val_set


def main():
    parser = argparse.ArgumentParser(description="Train complementary (comp) network")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to 'Gibson Floorplan Localization Dataset'")
    parser.add_argument("--dataset", type=str, default="gibson_g", choices=["gibson_f", "gibson_g"], help="Dataset split to use")
    parser.add_argument("--mv_ckpt", type=str, required=True, help="Path to pretrained mv.ckpt")
    parser.add_argument("--mono_ckpt", type=str, required=True, help="Path to pretrained mono.ckpt")
    parser.add_argument("--ckpt_path", type=str, default="./logs", help="Directory to save comp checkpoints")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=20)

    parser.add_argument("--L", type=int, default=3, help="Number of source frames")
    parser.add_argument("--D", type=int, default=128, help="Number of depth planes")
    parser.add_argument("--d_min", type=float, default=0.1)
    parser.add_argument("--d_max", type=float, default=15.0)
    parser.add_argument("--d_hyp", type=float, default=-0.2)
    parser.add_argument("--F_W", type=float, default=3.0/8.0, help="focal_length / image_width")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--shape_loss_weight", type=float, default=None)

    args = parser.parse_args()

    pl.seed_everything(42)

    # depth160 is used for mvd/comp; depth40 for mono only
    depth_suffix = "depth160"

    # Datasets
    train_set, val_set = build_datasets(
        dataset_root=args.dataset_path,
        dataset_name=args.dataset,
        L=args.L,
        depth_suffix=depth_suffix,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=collate_gridseq,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=collate_gridseq,
    )

    # Load frozen mono & multi-view nets
    mv_pl = mv_depth_net_pl.load_from_checkpoint(
        checkpoint_path=args.mv_ckpt,
        D=args.D,
        d_min=args.d_min,
        d_max=args.d_max,
        d_hyp=args.d_hyp,
    )
    mono_pl = depth_net_pl.load_from_checkpoint(
        checkpoint_path=args.mono_ckpt,
        d_min=args.d_min,
        d_max=args.d_max,
        d_hyp=args.d_hyp,
        D=args.D,
    )

    model = comp_d_net_pl(
        mv_net=mv_pl.net,
        mono_net=mono_pl.encoder,
        L=args.L,
        d_min=args.d_min,
        d_max=args.d_max,
        d_hyp=args.d_hyp,
        D=args.D,
        shape_loss_weight=args.shape_loss_weight,
        F_W=args.F_W,
        use_pred=True,
        lr=args.lr,
    )

    # Callbacks and Trainer
    os.makedirs(args.ckpt_path, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename="comp",
        monitor="loss-valid",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=args.ckpt_path,
        callbacks=[checkpoint_cb, lr_cb],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main() 