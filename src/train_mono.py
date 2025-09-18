import argparse
import os
import yaml
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from torch.utils.data import Subset
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from modules.mono.depth_net_pl import depth_net_pl

# optional ClearML logger (not used directly as Lightning logger, kept for compatibility)
try:
    from lightning.pytorch.loggers import ClearMLLogger  # type: ignore
    _HAS_CLEARML_LOGGER = True
except Exception:
    ClearMLLogger = None  # type: ignore
    _HAS_CLEARML_LOGGER = False

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import random


class MonoGibsonDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        scene_names,
        depth_suffix: str = "depth40",
        add_rp: bool = False,
        roll: float = 0.0,
        pitch: float = 0.0,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.scene_names = scene_names
        self.depth_suffix = depth_suffix
        self.add_rp = add_rp
        self.roll = roll
        self.pitch = pitch

        # index across scenes
        self.scene_start_idx = [0]
        self.rgb_names_per_scene = []
        self.depth_lines_per_scene = []
        self.pose_lines_per_scene = []

        for scene in self.scene_names:
            # list RGB names sorted
            rgb_dir = os.path.join(self.dataset_dir, scene, "rgb")
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])

            # read depths (lines correspond to ascending rgb order)
            depth_file = os.path.join(self.dataset_dir, scene, f"{self.depth_suffix}.txt")
            with open(depth_file, "r") as f:
                depth_lines = [line.strip() for line in f.readlines()]

            # read poses
            pose_file = os.path.join(self.dataset_dir, scene, "poses.txt")
            with open(pose_file, "r") as f:
                pose_lines = [line.strip() for line in f.readlines()]

            assert len(rgb_files) == len(depth_lines), (
                f"RGB/depth count mismatch in scene {scene}: {len(rgb_files)} vs {len(depth_lines)}"
            )
            assert len(rgb_files) == len(pose_lines), (
                f"RGB/pose count mismatch in scene {scene}: {len(rgb_files)} vs {len(pose_lines)}"
            )

            self.rgb_names_per_scene.append(rgb_files)
            self.depth_lines_per_scene.append(depth_lines)
            self.pose_lines_per_scene.append(pose_lines)
            self.scene_start_idx.append(self.scene_start_idx[-1] + len(rgb_files))

        self.N = self.scene_start_idx[-1]

    def __len__(self):
        return self.N

    def _inverse_normalize(self, img_chw: np.ndarray) -> np.ndarray:
        # img_chw normalized with mean/std below; convert back to HWC in [0,1]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = np.transpose(img_chw, (1, 2, 0))
        img = (img * std) + mean
        img = np.clip(img, 0.0, 1.0)
        return img

    def get_observation_image(self, global_idx: int) -> np.ndarray:
        # return de-normalized image for visualization
        sample = self.__getitem__(global_idx)
        return self._inverse_normalize(sample["img"])  # HWC in [0,1]

    def __getitem__(self, global_idx: int):
        scene_idx = np.sum(global_idx >= np.array(self.scene_start_idx)) - 1
        scene = self.scene_names[scene_idx]
        local_idx = global_idx - self.scene_start_idx[scene_idx]

        rgb_name = self.rgb_names_per_scene[scene_idx][local_idx]
        depth_line = self.depth_lines_per_scene[scene_idx][local_idx]
        pose_line = self.pose_lines_per_scene[scene_idx][local_idx]

        # load and normalize image
        img_path = os.path.join(self.dataset_dir, scene, "rgb", rgb_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # (C,H,W)

        # parse ground-truth rays (depth along floorplan scanlines)
        gt_rays = np.array([float(x) for x in depth_line.split(" ")], dtype=np.float32)
        pose = np.array([float(x) for x in pose_line.split(" ")], dtype=np.float32)  # [x,y,yaw]

        return {"img": img, "gt_rays": gt_rays, "pose": pose, "scene": scene}


class MonoFlatDataset(Dataset):
    def __init__(self, dataset_dir: str, image_dir: str = "images", depth_file: str = "depth45.txt") -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, image_dir)
        self.depth_path = os.path.join(dataset_dir, depth_file)

        # gather images (support png/jpg/jpeg), sorted
        all_files = sorted(os.listdir(self.image_dir))
        self.image_names = [
            f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        with open(self.depth_path, "r") as f:
            self.depth_lines = [line.strip() for line in f.readlines()]

        assert len(self.image_names) == len(self.depth_lines), (
            f"image/depth count mismatch: {len(self.image_names)} vs {len(self.depth_lines)}"
        )

        self.N = len(self.image_names)

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        depth_line = self.depth_lines[idx]

        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # (C,H,W)

        gt_rays = np.array([float(x) for x in depth_line.split(" ")], dtype=np.float32)
        return {"img": img, "gt_rays": gt_rays}


def collate_mono(batch):
    imgs = torch.from_numpy(np.stack([b["img"] for b in batch], axis=0))
    gt = torch.from_numpy(np.stack([b["gt_rays"] for b in batch], axis=0))
    return {"img": imgs, "gt_rays": gt}


def build_mono_datasets(dataset_path: str, depth_suffix: str):
    split_file = os.path.join(dataset_path, "split.yaml")
    with open(split_file, "r") as f:
        split = yaml.safe_load(f)

    train_set = MonoGibsonDataset(dataset_path, split["train"], depth_suffix=depth_suffix)
    val_set = MonoGibsonDataset(dataset_path, split["val"], depth_suffix=depth_suffix)
    return train_set, val_set


def build_flat_datasets(dataset_path: str, image_dir: str = "images", depth_file: str = "depth45.txt", val_ratio: float = 0.1, seed: int = 42):
    full = MonoFlatDataset(dataset_path, image_dir=image_dir, depth_file=depth_file)
    N = len(full)
    indices = list(range(N))
    random.Random(seed).shuffle(indices)

    val_count = max(1, int(round(N * float(val_ratio))))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_set = Subset(full, train_idx)
    val_set = Subset(full, val_idx)
    return train_set, val_set


def _compute_rays_from_depth(d: np.ndarray, V: int = 11, dv: float = 10.0, F_W: float = 3.0/8.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lightweight 1D interpolation (avoids SciPy). Returns (angles, x_end, y_end).
    """
    W = d.shape[0]
    angles = (np.arange(0, V) - np.arange(0, V).mean()) * dv / 180.0 * np.pi
    # desired pixel coordinates along width for each angle
    w = np.tan(angles) * W * F_W + (W - 1) / 2.0
    # clamp to valid range and 1D interpolate
    w_clip = np.clip(w, 0.0, W - 1.0)
    interp_d = np.interp(w_clip, np.arange(W, dtype=np.float32), d)
    rays = interp_d / np.cos(angles)
    x = rays * np.cos(angles)
    y = rays * np.sin(angles)
    return angles, x, y


def _depth_fullwidth_to_xy(d: np.ndarray, F_W: float = 3.0/8.0) -> Tuple[np.ndarray, np.ndarray]:
    """Map per-column floorplan depths to XY using x as forward distance and y as x*tan(theta).
    This matches the definition used to later convert to rays by dividing by cos(theta).
    """
    W = d.shape[0]
    a0 = (W - 1) / 2.0
    cols = np.arange(W, dtype=np.float32)
    theta = np.arctan((cols - a0) / (W * F_W))
    x = d  # forward distance along x (not along the ray)
    y = d * np.tan(theta)
    return x, y


class ClearMLVisualizationCallback(pl.Callback):
    def __init__(self, val_dataset: MonoGibsonDataset, task=None, num_samples: int = 5, V: int = 11, dv: float = 10.0, F_W: float = 3.0/8.0) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.task = task
        self.num_samples = num_samples
        self.V = V
        self.dv = dv
        self.F_W = F_W

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: depth_net_pl) -> None:
        if self.task is None:
            
            try:
                from clearml import Task  # type: ignore
                self.task = Task.current_task()
            except Exception:
                self.task = None
        if self.task is None:
            return
        cm_logger = self.task.get_logger()

        def inv_norm(img_chw: np.ndarray) -> np.ndarray:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = np.transpose(img_chw, (1, 2, 0))
            img = (img * std) + mean
            return np.clip(img, 0.0, 1.0)

        rows = []
        device = pl_module.device
        pl_module.eval()
        with torch.no_grad():
            idxs = np.random.choice(len(self.val_dataset), size=min(self.num_samples, len(self.val_dataset)), replace=False)
            for idx in idxs:
                sample = self.val_dataset[int(idx)]
                img_np = sample["img"]
                obs = inv_norm(img_np)
                img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
                gt_rays = sample["gt_rays"]

                d_pred, attn, _ = pl_module.encoder(img_t, None)
                d_pred_np = d_pred.squeeze(0).detach().cpu().numpy()
                attn_np = attn.squeeze(0).detach().cpu().numpy()

                x_fw_gt, y_fw_gt = _depth_fullwidth_to_xy(gt_rays, F_W=self.F_W)
                x_fw_pr, y_fw_pr = _depth_fullwidth_to_xy(d_pred_np, F_W=self.F_W)
                _, x_ray_gt, y_ray_gt = _compute_rays_from_depth(gt_rays, V=self.V, dv=self.dv, F_W=self.F_W)
                _, x_ray_pr, y_ray_pr = _compute_rays_from_depth(d_pred_np, V=self.V, dv=self.dv, F_W=self.F_W)

                S = attn_np.shape[-1]
                fW = d_pred_np.shape[0]
                fH = max(1, S // fW)
                ray_indices = np.linspace(0, fW - 1, 4, dtype=int)
                attn_maps = [attn_np[v].reshape(fH, fW) for v in ray_indices]

                rows.append({
                    "obs": obs,
                    "x_fw_gt": x_fw_gt, "y_fw_gt": y_fw_gt,
                    "x_fw_pr": x_fw_pr, "y_fw_pr": y_fw_pr,
                    "x_ray_gt": x_ray_gt, "y_ray_gt": y_ray_gt,
                    "x_ray_pr": x_ray_pr, "y_ray_pr": y_ray_pr,
                    "attn_maps": attn_maps,
                    "ray_indices": ray_indices
                })

        if len(rows) == 0:
            return

        # 1) Rays/Depth combined grid (R x 4)
        fig_main, axes_main = plt.subplots(len(rows), 4, figsize=(16, 3.2 * len(rows)))
        if len(rows) == 1:
            axes_main = np.expand_dims(axes_main, 0)

        for r, row in enumerate(rows):
            # observation
            axes_main[r, 0].imshow(row["obs"])
            axes_main[r, 0].set_title("observation")
            axes_main[r, 0].axis("off")

            # floorplan depth scatter (axis orientation: X = Y_old, Y = X_old)
            px_gt = row["y_fw_gt"]; py_gt = row["x_fw_gt"]
            px_pr = row["y_fw_pr"]; py_pr = row["x_fw_pr"]
            ax = axes_main[r, 1]
            ax.scatter(px_gt, py_gt, c="green", s=8, marker="x", label="gt")
            ax.scatter(px_pr, py_pr, c="blue", s=8, marker="x", label="pred")
            ax.set_title("floorplan depth"); ax.set_aspect("equal")
            if r == 0:
                ax.legend(loc="lower right", fontsize=8)

            # ground truth ray (transformed)
            ax = axes_main[r, 2]
            for k in range(len(row["x_ray_gt"])):
                ax.plot([0, row["y_ray_gt"][k]], [0, row["x_ray_gt"][k]], color="green", linewidth=2)
            ax.set_title("ground truth ray"); ax.set_aspect("equal")

            # predicted ray (transformed)
            ax = axes_main[r, 3]
            for k in range(len(row["x_ray_pr"])):
                ax.plot([0, row["y_ray_pr"][k]], [0, row["x_ray_pr"][k]], color="blue", linewidth=2)
            ax.set_title("predicted ray"); ax.set_aspect("equal")
        plt.tight_layout()
        cm_logger.report_matplotlib_figure(title="Figures", series="Depth Rays", figure=fig_main, iteration=trainer.current_epoch)
        plt.close(fig_main)

        # 2) Attention maps grid (R x 5)
        fig_attn, axes_attn = plt.subplots(len(rows), 5, figsize=(20, 3.2 * len(rows)))
        if len(rows) == 1:
            axes_attn = np.expand_dims(axes_attn, 0)
        for r, row in enumerate(rows):
            axes_attn[r, 0].imshow(row["obs"]); axes_attn[r, 0].set_title("observation"); axes_attn[r, 0].axis("off")
            for j in range(4):
                axes_attn[r, 1 + j].imshow(row["attn_maps"][j], cmap="hot")
                axes_attn[r, 1 + j].set_title(f"N={row['ray_indices'][j]}")
                axes_attn[r, 1 + j].axis("off")
        plt.tight_layout()
        cm_logger.report_matplotlib_figure(title="Figures", series="Attention Maps", figure=fig_attn, iteration=trainer.current_epoch)
        plt.close(fig_attn)


class ClearMLMetricsCallback(pl.Callback):
    def __init__(self, task=None) -> None:
        super().__init__()
        self.task = task

    def _get_logger(self):
        if self.task is None:
            try:
                from clearml import Task  # type: ignore
                self.task = Task.current_task()
            except Exception:
                self.task = None
        return self.task.get_logger() if self.task is not None else None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        logger = self._get_logger()
        if logger is None:
            return
        metrics = trainer.callback_metrics
        # Prefer total training loss; fall back to L1 if needed. Exclude lr metrics.
        key = "loss-train" if "loss-train" in metrics else ("l1_loss-train" if "l1_loss-train" in metrics else None)
        if key and not key.lower().startswith("lr"):
            v = metrics[key]
            try:
                val = v.item() if hasattr(v, "item") else float(v)
                logger.report_scalar(title="loss", series="train", value=val, iteration=trainer.current_epoch)
            except Exception:
                pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        logger = self._get_logger()
        if logger is None:
            return
        metrics = trainer.callback_metrics
        # Prefer total validation loss; fall back to L1 if needed. Exclude lr metrics.
        key = "loss-valid" if "loss-valid" in metrics else ("l1_loss-valid" if "l1_loss-valid" in metrics else None)
        if key and not key.lower().startswith("lr"):
            v = metrics[key]
            try:
                val = v.item() if hasattr(v, "item") else float(v)
                logger.report_scalar(title="loss", series="valid", value=val, iteration=trainer.current_epoch)
            except Exception:
                pass


def main():
    # OmegaConf: load base config and merge CLI overrides (key=value)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="config.yaml")
    known, unknown = parser.parse_known_args()

    base = OmegaConf.load(known.config) if os.path.isfile(known.config) else OmegaConf.create()
    cli = OmegaConf.from_dotlist(unknown)
    conf = OmegaConf.merge(base, cli)
    OmegaConf.resolve(conf)

    pl.seed_everything(42)

    # ClearML init (Task) and push config
    task = None
    if conf.get("use_clearml", False):
        try:
            from clearml import Task  # type: ignore
            task = Task.init(project_name=conf.clearml_project, 
                             task_name=conf.clearml_task,
                             output_uri=True)
            # Make config editable in UI
            task.connect(OmegaConf.to_container(conf, resolve=True))
            try:
                task.connect_configuration(name="config.yaml", configuration=OmegaConf.to_yaml(conf))
            except Exception:
                pass
        except Exception as e:
            print("WARN: ClearML Task.init failed:", e)
            task = None

    # Resolve ClearML Dataset path if provided
    if conf.get("use_clearml", False):
        try:
            from clearml import Dataset  # type: ignore
            # Default to the new uploaded dataset if not explicitly provided
            if not conf.get("dataset_id") and not (conf.get("dataset_project") and conf.get("dataset_name")):
                conf.dataset_project = conf.get("dataset_project", "FLoc")
                conf.dataset_name = conf.get("dataset_name", "airports_malls")
            print("Using ClearML Dataset")
            ds = Dataset.get_by_id(conf.dataset_id) if conf.get("dataset_id") else Dataset.get(
                dataset_project=conf.dataset_project, dataset_name=conf.dataset_name
            )
            print("Downloading local copy of dataset")
            local_path = ds.get_local_copy()
            # Try to resolve either Gibson-style or flat-style roots
            def _resolve_root_any(p):
                # Prefer Gibson-style if present (back-compat)
                if os.path.isdir(os.path.join(p, "gibson_f")) or os.path.isdir(os.path.join(p, "gibson_g")):
                    return p
                sub = os.path.join(p, "Gibson Floorplan Localization Dataset")
                if os.path.isdir(sub):
                    return sub
                # Otherwise search for flat format: images dir + depth file
                image_dir_name = str(conf.get("image_dir", "images"))
                depth_file_name = str(conf.get("depth_file", f"{conf.get('depth_suffix', 'depth45')}.txt"))
                for root, dirs, files in os.walk(p):
                    if image_dir_name in dirs and depth_file_name in files:
                        return root
                return p
            conf.dataset_path = _resolve_root_any(local_path)
            print(f"Using ClearML dataset at: {conf.dataset_path}")
        except Exception as e:
            print("WARN: ClearML Dataset.get failed, falling back to dataset_path:", e)

    # Build datasets
    split_yaml = os.path.join(conf.dataset_path, "split.yaml")
    flat_images_dir = os.path.join(conf.dataset_path, str(conf.get("image_dir", "images")))
    flat_depth_file = os.path.join(conf.dataset_path, str(conf.get("depth_file", f"{conf.get('depth_suffix', 'depth45')}.txt")))

    if os.path.isfile(split_yaml):
        print("Detected Gibson-style dataset with split.yaml")
        train_set, val_set = build_mono_datasets(
            dataset_path=conf.dataset_path,
            depth_suffix=str(conf.get("depth_suffix", "depth40")),
        )
    elif os.path.isdir(flat_images_dir) and os.path.isfile(flat_depth_file):
        print("Detected flat dataset (images + depth file)")
        train_set, val_set = build_flat_datasets(
            dataset_path=conf.dataset_path,
            image_dir=str(conf.get("image_dir", "images")),
            depth_file=str(conf.get("depth_file", f"{conf.get('depth_suffix', 'depth45')}.txt")),
            val_ratio=float(conf.get("val_ratio", 0.1)),
            seed=int(conf.get("split_seed", 42)),
        )
    else:
        raise RuntimeError(
            f"Could not detect dataset format at {conf.dataset_path}. Expected either split.yaml (Gibson) or an images directory with a depth file."
        )

    # Optional dataset subsetting
    if int(conf.get("train_subset", 0)) > 0:
        train_set = Subset(train_set, list(range(min(int(conf.train_subset), len(train_set)))))
    if int(conf.get("val_subset", 0)) > 0:
        val_set = Subset(val_set, list(range(min(int(conf.val_subset), len(val_set)))))

    train_loader = DataLoader(
        train_set,
        batch_size=int(conf.batch_size),
        shuffle=True,
        num_workers=int(conf.num_workers),
        pin_memory=True,
        persistent_workers=int(conf.num_workers) > 0,
        drop_last=True,
        collate_fn=collate_mono,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(conf.batch_size),
        shuffle=False,
        num_workers=int(conf.num_workers),
        pin_memory=True,
        persistent_workers=int(conf.num_workers) > 0,
        drop_last=False,
        collate_fn=collate_mono,
    )

    model = depth_net_pl(
        shape_loss_weight=conf.get("shape_loss_weight", None),
        lr=float(conf.lr),
        d_min=float(conf.d_min),
        d_max=float(conf.d_max),
        d_hyp=float(conf.d_hyp),
        D=int(conf.D),
        F_W=float(conf.F_W),
    )

    os.makedirs(conf.ckpt_path, exist_ok=True)
    # Checkpoint policies
    monitor_metric = conf.get("ckpt_monitor", "loss-valid")
    monitor_mode = conf.get("ckpt_mode", "min")
    save_top_k = int(conf.get("ckpt_save_top_k", 3))
    save_last = bool(conf.get("ckpt_save_last", True))

    best_ckpt_cb = ModelCheckpoint(
        dirpath=conf.ckpt_path,
        filename="mono-best",
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=save_top_k,
        save_last=save_last,
        auto_insert_metric_name=True,
    )

    periodic_n = int(conf.get("ckpt_every_n_epochs", 0))
    periodic_cb = None
    if periodic_n > 0:
        periodic_dir = os.path.join(conf.ckpt_path, "periodic")
        os.makedirs(periodic_dir, exist_ok=True)
        periodic_cb = ModelCheckpoint(
            dirpath=periodic_dir,
            filename="mono-ep{epoch:03d}",
            save_top_k=-1,  # save all at the given interval
            every_n_epochs=periodic_n,
        )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    viz_cb = ClearMLVisualizationCallback(val_set, task=task, num_samples=int(conf.vis_samples), F_W=float(conf.F_W))
    metrics_cb = ClearMLMetricsCallback(task=task)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = pl.Trainer(
        max_epochs=int(conf.max_epochs),
        accelerator=accelerator,
        devices=devices,
        default_root_dir=conf.ckpt_path,
        callbacks=[c for c in [best_ckpt_cb, periodic_cb, lr_cb, viz_cb, metrics_cb] if c is not None],
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=int(conf.num_sanity_val_steps),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main() 