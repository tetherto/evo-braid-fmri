from pathlib import Path
from re import sub

#!/usr/bin/env python3
"""
MOSAIC multitask brain-to-CLIP training.

Features:
- canonical CLIP metadata
- no CLS token by default
- ROI-token encoder with mean/attention pooling
- contrastive branch uses unmasked ROI tokens
- MAE branch uses masked ROI tokens
- combined cosine + InfoNCE loss for retrieval
- optional dataset embedding
- optional subject embedding
- optional FiLM conditioning
- DDP-compatible InfoNCE with negatives gathered across GPUs

Main design:
    unmasked ROI tokens -> pooling -> brain projection -> CLIP loss
    masked ROI tokens   -> decoder -> ROI reconstruction -> MAE loss

The CLS token is disabled by default because retrieval uses pooled ROI tokens.
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mosaic
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger




def compute_selected_indices_from_cache(config, index_df):
    train_df = index_df[index_df["is_train"]].copy()
    val_df = index_df[index_df["is_val"]].copy()

    if config.get("dataset_tag_filter") is not None:
        train_df = train_df[train_df["dataset_name"] == config["dataset_tag_filter"]]
        val_df = val_df[val_df["dataset_name"] == config["dataset_tag_filter"]]

    if config.get("subject_filter") is not None:
        train_df = train_df[train_df["subject_name"] == config["subject_filter"]]
        val_df = val_df[val_df["subject_name"] == config["subject_filter"]]


    train_idx = train_df["real_idx"].to_numpy(dtype=np.int64).copy()
    val_idx = val_df["real_idx"].to_numpy(dtype=np.int64).copy()

    rng = np.random.default_rng(config["seed"])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    if config["max_train_samples"] is not None:
        train_idx = train_idx[: int(config["max_train_samples"])]

    if config["val_same_as_train"]:
        val_idx = train_idx.copy()
    elif config["max_val_samples"] is not None:
        val_idx = val_idx[: int(config["max_val_samples"])]

    print("Train samples selected:", len(train_idx), flush=True)
    print("Val samples selected:", len(val_idx), flush=True)
    print("Train writable:", train_idx.flags.writeable, flush=True)
    print("Val writable:", val_idx.flags.writeable, flush=True)

    return train_idx, val_idx


def build_subject_mapping_from_cache(config, index_df):
    """
    Build subject_name_to_id quickly from mosaic_index_cache.pkl.

    Uses dataset_name::subject_name so that sub-01 from different datasets
    gets different subject embeddings.
    """
    df = index_df.copy()

    # Use all valid samples so both train and val subjects are included
    if "valid_mask" in df.columns:
        df = df[df["valid_mask"] == 1].copy()
    elif "is_train" in df.columns and "is_val" in df.columns:
        df = df[df["is_train"] | df["is_val"]].copy()
    else:
        raise ValueError(
            "index_df must contain either valid_mask or both is_train/is_val columns."
        )

    # Optional dataset filter
    if config.get("dataset_tag_filter", None) is not None:
        df = df[df["dataset_name"] == config["dataset_tag_filter"]].copy()

    # Optional subject filter
    if config.get("subject_filter", None) is not None:
        df = df[df["subject_name"] == config["subject_filter"]].copy()

    # Important: dataset + subject, not subject alone
    if "subject_key" in df.columns:
        subject_keys = df["subject_key"].astype(str)
    else:
        subject_keys = (
            df["dataset_name"].astype(str)
            + "::"
            + df["subject_name"].astype(str)
        )

    unique_subjects = sorted(subject_keys.unique().tolist())

    subject_name_to_id = {
        subject_key: idx
        for idx, subject_key in enumerate(unique_subjects)
    }

    return subject_name_to_id
# ============================================================
# 1. CONFIGURATION
# ============================================================



DEFAULT_CONFIG = {
    # Paths
    "data_folder": "/srv/nfs-data/sisko/michelangelo/MOSAIC/",
    "default_root_dir": "/srv/nfs-data/sisko/michelangelo/MOSAIC/mosaic_roi_mae_runs",
    "schema_path": "/srv/nfs-data/sisko/michelangelo/MOSAIC/mosaic_roi_mae_runs/roi_schema.json",
    # "canonical_metadata_path": "/srv/nfs-data/sisko/michelangelo/MOSAIC/meta_data_compiled/MOSAIC_CANONICAL_CLIP_METADATA.pkl",
    "canonical_metadata_path": "/mount/share/moein/MOSAIC/meta_data_compiled/canonical_meta_remapped.pkl",
    "mask_path": "/srv/nfs-data/sisko/michelangelo/MOSAIC/splits/train_mask.npy",
    "valid_path": "/srv/nfs-data/sisko/michelangelo/MOSAIC/splits/valid_mask.npy",

    # Logging
    "project_name": "mosaic-roi-foundation",
    "run_name": "mosaic_multitask_mae_clip",

    # Reproducibility
    "seed": 42,

    # Dataset selection
    "dataset_tag_filter": None, # e.g. "NSD", "THINGS", "deeprecon", "BOLD5000", "BMD", "GOD", "HAD", "NOD"
    "max_train_samples": None,   # None for all
    "max_val_samples": None,     # None for all
    "val_same_as_train": False,
    "cache_selected_samples": False,

    # Train only on one subject within one dataset
    "subject_filter": None,  # e.g. "sub-01"
    "index_cache_path": "/srv/nfs-data/sisko/michelangelo/MOSAIC/mosaic_roi_mae_runs/mosaic_index_cache.pkl",

    # Training
    "batch_size": 256,            # per GPU under DDP (released checkpoint: 256 x 4 GPUs = 1024)
    "num_workers": 4,
    "pin_memory": True,
    "max_epochs": 80,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "gradient_clip_val": 1.0,


    # Model
    "embed_dim": 512,
    "encoder_depth": 4,
    "decoder_depth": 0,
    "num_heads": 4,
    "mlp_ratio": 4.0,
    "dropout": 0.1,

    # CLS token
    # Default False: retrieval uses pooled ROI tokens, and MAE reconstructs ROI tokens.
    "use_cls_token": False,

    # Pooling for retrieval: "mean" or "attention". "cls" only works if use_cls_token=True.
    "pooling": "attention",

    # Contrastive/CLIP loss
    # Multi-positive InfoNCE: treat all samples with the same stimulus_id as positives;
    # all samples with different stimulus_id remain negatives.
    "use_multipositive_infonce": True,
    "use_contrastive": True,
    "loss_mode": "infonce",  # cosine_regression, infonce, cosine_plus_infonce
    "infonce_weight": 0.1,
    "contrastive_weight": 0.5,

    

    # MAE auxiliary loss
    "use_mae": False,
    "mae_weight": 0.03,
    "mask_ratio": 0.30,
    "visible_loss_weight": 0.0,

    # Conditioning
    "dataset_name_to_id": {
        "NSD": 0,
        "THINGS": 1,
        "deeprecon": 2,
        "BOLD5000": 3,
        "BMD": 4,
        "GOD": 5,
        "HAD": 6,
        "NOD": 7,
    },
    "num_datasets": 8,
    "use_dataset_embedding": False,
    "use_subject_embedding": True,
    "use_film_conditioning": False,
    "subject_name_to_id": {},
    "num_subjects": 0,

    # Lightning
    "accelerator": "gpu",
    "devices": 4,
    "strategy": "ddp",

    # "devices": 1,
    # "strategy": "auto",
    "precision": 32,
    "log_every_n_steps": 10,
    "num_sanity_val_steps": 0,

    "use_early_stopping": True,
    "early_stopping_patience": 5,
}


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


# ============================================================
# 3. HELPERS
# ============================================================

def infer_dataset_name_from_raw_name(raw_name: str) -> str:
    parts = raw_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer dataset name from raw_name={raw_name}")
    return parts[1]


def infer_subject_name_from_raw_name(raw_name: str) -> str:
    parts = raw_name.split("_")
    if len(parts) < 1:
        raise ValueError(f"Cannot infer subject name from raw_name={raw_name}")
    return parts[0]


def make_subject_key(raw_name: str) -> str:
    dataset_name = infer_dataset_name_from_raw_name(raw_name)
    subject_name = infer_subject_name_from_raw_name(raw_name)
    return f"{dataset_name}::{subject_name}"


def filter_indices_by_dataset_tag(indices: Sequence[int], mosaic_dataset, dataset_tag: Optional[str]) -> np.ndarray:
    from tqdm.auto import tqdm

    if dataset_tag is None:
        return np.asarray(indices, dtype=np.int64)

    kept = []
    pattern = f"_{dataset_tag}_"

    for i in tqdm(indices, desc=f"Filtering {dataset_tag}", dynamic_ncols=True):
        name = mosaic_dataset[int(i)]["name"]
        if pattern in name:
            kept.append(int(i))

    return np.asarray(kept, dtype=np.int64)


def filter_indices_by_subject(indices: Sequence[int], mosaic_dataset, subject_filter: Optional[str]) -> np.ndarray:
    from tqdm.auto import tqdm

    if subject_filter is None:
        return np.asarray(indices, dtype=np.int64)

    kept = []

    for i in tqdm(indices, desc=f"Filtering subject {subject_filter}", dynamic_ncols=True):
        name = mosaic_dataset[int(i)]["name"]
        subject_name = infer_subject_name_from_raw_name(name)

        if subject_name == subject_filter:
            kept.append(int(i))

    return np.asarray(kept, dtype=np.int64)


def compute_selected_indices(config: Dict, mosaic_dataset) -> Tuple[np.ndarray, np.ndarray]:
    train_mask = np.load(config["mask_path"])
    valid_mask = np.load(config["valid_path"])

    if len(train_mask) != len(mosaic_dataset):
        raise ValueError(
            f"Mask length {len(train_mask)} does not match MOSAIC dataset length {len(mosaic_dataset)}."
        )

    train_bool = train_mask == 1
    valid_bool = valid_mask == 1

    candidate_train_idx = np.where(train_bool & valid_bool)[0]
    candidate_val_idx = np.where((~train_bool) & valid_bool)[0]

    print("Candidate train samples:", len(candidate_train_idx), flush=True)
    print("Candidate val samples:", len(candidate_val_idx), flush=True)

    # first filter dataset
    if config["dataset_tag_filter"] is not None:
        print(f"\nDataset filter: {config['dataset_tag_filter']}", flush=True)
        candidate_train_idx = filter_indices_by_dataset_tag(
            candidate_train_idx,
            mosaic_dataset,
            config["dataset_tag_filter"],
        )
        candidate_val_idx = filter_indices_by_dataset_tag(
            candidate_val_idx,
            mosaic_dataset,
            config["dataset_tag_filter"],
        )
        print("Filtered train samples:", len(candidate_train_idx), flush=True)
        print("Filtered val samples:", len(candidate_val_idx), flush=True)

    # then filter subject
    if config.get("subject_filter", None) is not None:
        print(f"\nSubject filter: {config['subject_filter']}", flush=True)

        candidate_train_idx = filter_indices_by_subject(
            candidate_train_idx,
            mosaic_dataset,
            config["subject_filter"],
        )

        candidate_val_idx = filter_indices_by_subject(
            candidate_val_idx,
            mosaic_dataset,
            config["subject_filter"],
        )

        print("Subject-filtered train samples:", len(candidate_train_idx), flush=True)
        print("Subject-filtered val samples:", len(candidate_val_idx), flush=True)
    
    rng = np.random.default_rng(config["seed"])
    rng.shuffle(candidate_train_idx)
    rng.shuffle(candidate_val_idx)

    max_train = config["max_train_samples"]
    max_val = config["max_val_samples"]

    if max_train is not None:
        train_idx = candidate_train_idx[: int(max_train)]
    else:
        train_idx = candidate_train_idx

    if config["val_same_as_train"]:
        val_idx = train_idx.copy()
    else:
        if max_val is not None:
            val_idx = candidate_val_idx[: int(max_val)]
        else:
            val_idx = candidate_val_idx

    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def build_subject_mapping_from_valid_mask(config: Dict, mosaic_dataset) -> Dict[str, int]:
    from tqdm.auto import tqdm

    valid_mask = np.load(config["valid_path"])
    all_valid_idx = np.where(valid_mask == 1)[0]

    if config["dataset_tag_filter"] is not None:
        all_valid_idx = filter_indices_by_dataset_tag(
            all_valid_idx,
            mosaic_dataset,
            config["dataset_tag_filter"],
        )

    subject_keys = []
    for i in tqdm(all_valid_idx, desc="Building subject ID mapping from valid samples", dynamic_ncols=True):
        raw_name = mosaic_dataset[int(i)]["name"]
        subject_keys.append(make_subject_key(raw_name))

    unique_subjects = sorted(set(subject_keys))
    return {name: idx for idx, name in enumerate(unique_subjects)}


# ============================================================
# 4. DATASET
# ============================================================

class FastCanonicalClipDataset(Dataset):
    def __init__(
        self,
        canonical_metadata: pd.DataFrame,
        items: list,
        roi_names: List[str],
        dataset_name_to_id: Dict[str, int],
        subject_name_to_id: Optional[Dict[str, int]] = None,
        mosaic_dataset=None,
    ):
        if canonical_metadata["clean"].duplicated().sum() != 0:
            raise ValueError("Canonical metadata still contains duplicated clean keys.")

        self.meta_list = {
            row.clean: row
            for row in canonical_metadata.itertuples(index=False)
        }

        self.stimulus_clean_to_id = {
            clean: i
            for i, clean in enumerate(canonical_metadata["clean"].tolist())
        }

        self.items = items
        self.roi_names = list(roi_names)
        self.dataset_name_to_id = dict(dataset_name_to_id)
        self.subject_name_to_id = dict(subject_name_to_id or {})
        self.mosaic_dataset = mosaic_dataset

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _extract_betas_name(raw_name: str) -> str:
        return (
            raw_name
            .split("stimulus-")[1]
            .split("_phase")[0]
            .replace(" ", "_")
            .split(".")[0]
        )

    @staticmethod
    def _process_betas(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32).flatten()
        arr[~np.isfinite(arr)] = 0.0
        mu = np.mean(arr)
        sd = np.std(arr)
        return np.clip((arr - mu) / (sd + 1e-6), -5.0, 5.0).astype(np.float32)

    def _get_sample(self, item):
        if item["sample"] is not None:
            return item["sample"]

        if self.mosaic_dataset is None:
            raise ValueError("mosaic_dataset is required when item['sample'] is None.")

        return self.mosaic_dataset[int(item["real_idx"])]

    def __getitem__(self, idx):
        item = self.items[idx]
        real_idx = int(item["real_idx"])
        sample = self._get_sample(item)

        raw_name = sample["name"]
        clean = self._extract_betas_name(raw_name)
        stimulus_id = int(self.stimulus_clean_to_id[clean])

        if clean not in self.meta_list:
            raise KeyError(
                f"Could not find clean={clean} in canonical metadata. "
                f"Raw beta name: {raw_name}"
            )

        dataset_name = infer_dataset_name_from_raw_name(raw_name)
        subject_name = infer_subject_name_from_raw_name(raw_name)
        subject_key = f"{dataset_name}::{subject_name}"

        if dataset_name not in self.dataset_name_to_id:
            raise KeyError(
                f"Dataset name '{dataset_name}' inferred from '{raw_name}' is not in dataset_name_to_id. "
                f"Available keys: {list(self.dataset_name_to_id.keys())}"
            )

        dataset_id = int(self.dataset_name_to_id[dataset_name])

        if len(self.subject_name_to_id) > 0:
            if subject_key not in self.subject_name_to_id:
                raise KeyError(
                    f"Subject key '{subject_key}' inferred from '{raw_name}' is not in subject_name_to_id."
                )
            subject_id = int(self.subject_name_to_id[subject_key])
        else:
            subject_id = -1

        row = self.meta_list[clean]
        betas_dict = sample["betas"]

        roi_list = [
            torch.from_numpy(self._process_betas(betas_dict[r])).float()
            for r in self.roi_names
        ]

        clip_features = torch.from_numpy(
            np.asarray(row.canonical_embedding, dtype=np.float32)
        ).float()

        has_clip = bool(row.embedding_ok)

        return {
            "roi_list": roi_list,
            "clip_features": clip_features,
            "has_clip": has_clip,
            "real_idx": real_idx,
            "raw_betas_name": raw_name,
            "stimulus_clean": clean,
            "canonical_filename": row.filename,
            "n_duplicate_rows": int(row.n_duplicate_rows),
            "all_data_sources": row.all_data_sources,
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "subject_name": subject_name,
            "subject_key": subject_key,
            "subject_id": subject_id,
            "stimulus_id": stimulus_id,
        }


def roi_collate_fn(batch):
    num_rois = len(batch[0]["roi_list"])

    roi_tensors = [
        torch.stack([item["roi_list"][r] for item in batch])
        for r in range(num_rois)
    ]

    stimulus_id = torch.tensor(
        [item["stimulus_id"] for item in batch],
        dtype=torch.long,
    )

    clip_features = torch.stack([item["clip_features"] for item in batch])

    has_clip = torch.tensor(
        [item["has_clip"] for item in batch],
        dtype=torch.bool,
    )

    dataset_id = torch.tensor(
        [item["dataset_id"] for item in batch],
        dtype=torch.long,
    )

    subject_id = torch.tensor(
        [item["subject_id"] for item in batch],
        dtype=torch.long,
    )

    return {
        "roi_tensors": roi_tensors,
        "clip_features": clip_features,
        "has_clip": has_clip,
        "dataset_id": dataset_id,
        "stimulus_id": stimulus_id,
        "dataset_name": [item["dataset_name"] for item in batch],
        "subject_id": subject_id,
        "subject_name": [item["subject_name"] for item in batch],
        "subject_key": [item["subject_key"] for item in batch],
        "real_idx": torch.tensor([item["real_idx"] for item in batch], dtype=torch.long),
        "raw_betas_name": [item["raw_betas_name"] for item in batch],
        "stimulus_clean": [item["stimulus_clean"] for item in batch],
        "canonical_filename": [item["canonical_filename"] for item in batch],
        "n_duplicate_rows": torch.tensor(
            [item["n_duplicate_rows"] for item in batch],
            dtype=torch.long,
        ),
        "all_data_sources": [item["all_data_sources"] for item in batch],
    }


class MosaicMultitaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mosaic_dataset,
        roi_names,
        config,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ):
        super().__init__()
        self.mosaic_dataset = mosaic_dataset
        self.roi_names = roi_names
        self.config = config
        self.train_idx = np.asarray(train_idx, dtype=np.int64)
        self.val_idx = np.asarray(val_idx, dtype=np.int64)

    def setup(self, stage=None):
        from tqdm.auto import tqdm

        with open(self.config["canonical_metadata_path"], "rb") as f:
            canonical_meta = pickle.load(f)

        print("\n==============================", flush=True)
        print("Canonical metadata", flush=True)
        print("==============================", flush=True)
        print("Shape:", canonical_meta.shape, flush=True)
        print("Duplicated clean keys:", canonical_meta["clean"].duplicated().sum(), flush=True)

        train_idx = self.train_idx
        val_idx = self.val_idx

        print("\nFinal split", flush=True)
        print("Train samples:", len(train_idx), flush=True)
        print("Val samples:", len(val_idx), flush=True)
        print("Val same as train:", self.config["val_same_as_train"], flush=True)
        print("Cache selected samples:", self.config["cache_selected_samples"], flush=True)

        if self.config["cache_selected_samples"]:
            unique_idx = np.unique(np.concatenate([train_idx, val_idx]))

            print("\n==============================", flush=True)
            print("Caching selected MOSAIC samples once", flush=True)
            print("==============================", flush=True)
            print("Unique samples to cache:", len(unique_idx), flush=True)

            sample_cache = {}
            for real_idx in tqdm(unique_idx, desc="Caching selected samples", dynamic_ncols=True):
                real_idx = int(real_idx)
                sample_cache[real_idx] = self.mosaic_dataset[real_idx]

            train_items = [
                {"real_idx": int(i), "sample": sample_cache[int(i)]}
                for i in train_idx
            ]

            val_items = [
                {"real_idx": int(i), "sample": sample_cache[int(i)]}
                for i in val_idx
            ]

            mosaic_for_lazy = None

        else:
            print("\nUsing lazy MOSAIC loading inside Dataset.", flush=True)
            print("This saves RAM but can be slower per epoch.", flush=True)

            train_items = [{"real_idx": int(i), "sample": None} for i in train_idx]
            val_items = [{"real_idx": int(i), "sample": None} for i in val_idx]
            mosaic_for_lazy = self.mosaic_dataset

        self.train_ds = FastCanonicalClipDataset(
            canonical_metadata=canonical_meta,
            items=train_items,
            roi_names=self.roi_names,
            dataset_name_to_id=self.config["dataset_name_to_id"],
            subject_name_to_id=self.config.get("subject_name_to_id", {}),
            mosaic_dataset=mosaic_for_lazy,
        )

        self.val_ds = FastCanonicalClipDataset(
            canonical_metadata=canonical_meta,
            items=val_items,
            roi_names=self.roi_names,
            dataset_name_to_id=self.config["dataset_name_to_id"],
            subject_name_to_id=self.config.get("subject_name_to_id", {}),
            mosaic_dataset=mosaic_for_lazy,
        )

        rank = int(os.environ.get("LOCAL_RANK", 0))
        if rank == 0:
            self._print_first_examples(n=3)

    def _print_first_examples(self, n: int = 3):
        print("\n==============================", flush=True)
        print("First dataset examples", flush=True)
        print("==============================", flush=True)

        for i in range(min(n, len(self.train_ds))):
            item = self.train_ds[i]
            print("----------------------------------------", flush=True)
            print("raw beta name:      ", item["raw_betas_name"], flush=True)
            print("dataset_name:       ", item["dataset_name"], flush=True)
            print("dataset_id:         ", item["dataset_id"], flush=True)
            print("subject_name:       ", item["subject_name"], flush=True)
            print("subject_key:        ", item["subject_key"], flush=True)
            print("subject_id:         ", item["subject_id"], flush=True)
            print("stimulus_clean:     ", item["stimulus_clean"], flush=True)
            print("canonical filename: ", item["canonical_filename"], flush=True)
            print("n_duplicate_rows:   ", item["n_duplicate_rows"], flush=True)
            print("clip norm:          ", item["clip_features"].norm().item(), flush=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            collate_fn=roi_collate_fn,
            pin_memory=self.config["pin_memory"],
            drop_last=True,
            persistent_workers=self.config["num_workers"] > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=roi_collate_fn,
            pin_memory=self.config["pin_memory"],
            drop_last=True,
            persistent_workers=self.config["num_workers"] > 0,
        )


# ============================================================
# 5. MODEL
# ============================================================

class ROITokenizer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class ROIReconstructionHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        hidden = max(32, embed_dim // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, roi_tokens):
        weights = torch.softmax(self.score(roi_tokens).squeeze(-1), dim=1)
        pooled = torch.sum(roi_tokens * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class ROIEncoderMAEContrastive(nn.Module):
    def __init__(
        self,
        roi_names,
        roi_dims,
        embed_dim=512,
        encoder_depth=4,
        decoder_depth=1,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        mask_ratio=0.3,
        use_cls_token=False,
        num_datasets=8,
        use_dataset_embedding=True,
        num_subjects=0,
        use_subject_embedding=False,
        use_film_conditioning=False,
    ):
        super().__init__()

        self.roi_names = list(roi_names)
        self.roi_dims = list(roi_dims)
        self.num_rois = len(self.roi_names)
        self.embed_dim = embed_dim
        self.mask_ratio = float(mask_ratio)
        self.use_cls_token = bool(use_cls_token)

        self.use_dataset_embedding = bool(use_dataset_embedding)
        self.use_subject_embedding = bool(use_subject_embedding)
        self.use_film_conditioning = bool(use_film_conditioning)

        self.tokenizers = nn.ModuleList([
            ROITokenizer(d, embed_dim)
            for d in self.roi_dims
        ])

        self.roi_embeddings = nn.Parameter(
            torch.randn(self.num_rois, embed_dim) * 0.02
        )

        self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if self.use_dataset_embedding:
            self.dataset_embedding = nn.Embedding(num_datasets, embed_dim)
            nn.init.normal_(self.dataset_embedding.weight, std=0.02)
        else:
            self.dataset_embedding = None

        if self.use_subject_embedding:
            if num_subjects is None or int(num_subjects) <= 0:
                raise ValueError("num_subjects must be > 0 when use_subject_embedding=True.")
            self.subject_embedding = nn.Embedding(int(num_subjects), embed_dim)
            nn.init.normal_(self.subject_embedding.weight, std=0.02)
        else:
            self.subject_embedding = None

        if self.use_film_conditioning:
            if not (self.use_dataset_embedding or self.use_subject_embedding):
                raise ValueError("FiLM conditioning requires dataset and/or subject embeddings.")

            self.condition_norm = nn.LayerNorm(embed_dim)
            self.film_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim * 2),
            )
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)
        else:
            self.condition_norm = None
            self.film_mlp = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_depth,
        )

        if decoder_depth > 0:
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerEncoder(
                decoder_layer,
                num_layers=decoder_depth,
            )
        else:
            self.decoder = nn.Identity()

        self.final_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.attn_pool = AttentionPool(embed_dim)

        self.reconstruction_heads = nn.ModuleList([
            ROIReconstructionHead(embed_dim, d)
            for d in self.roi_dims
        ])

    def sample_mask(self, batch_size, device):
        num_mask = max(1, int(round(self.mask_ratio * self.num_rois)))
        noise = torch.rand(batch_size, self.num_rois, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, self.num_rois, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        return mask

    def _build_condition_embedding(self, dataset_id=None, subject_id=None):
        cond = None

        if self.use_dataset_embedding:
            if dataset_id is None:
                raise ValueError("dataset_id must be provided when use_dataset_embedding=True.")
            cond = self.dataset_embedding(dataset_id)

        if self.use_subject_embedding:
            if subject_id is None:
                raise ValueError("subject_id must be provided when use_subject_embedding=True.")
            if (subject_id < 0).any():
                raise ValueError("subject_id contains -1 but use_subject_embedding=True.")
            subj = self.subject_embedding(subject_id)
            cond = subj if cond is None else cond + subj

        return cond

    def _apply_conditioning(self, tokens, cond):
        if cond is None:
            return tokens

        if self.use_film_conditioning:
            cond = self.condition_norm(cond)
            gamma_beta = self.film_mlp(cond)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = 1.0 + 0.1 * torch.tanh(gamma)
            beta = 0.1 * beta
            tokens = gamma.unsqueeze(1) * tokens + beta.unsqueeze(1)
        else:
            tokens = tokens + cond.unsqueeze(1)

        return tokens

    def encode_roi_tokens(self, roi_tensors, dataset_id=None, subject_id=None, mask=None):
        batch_size = roi_tensors[0].shape[0]
        device = roi_tensors[0].device

        tokens = torch.stack(
            [tok(x) for tok, x in zip(self.tokenizers, roi_tensors)],
            dim=1,
        )

        if mask is not None:
            tokens = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.expand(batch_size, self.num_rois, -1),
                tokens,
            )

        tokens = tokens + self.roi_embeddings.unsqueeze(0)

        cond = self._build_condition_embedding(
            dataset_id=dataset_id,
            subject_id=subject_id,
        )
        tokens = self._apply_conditioning(tokens, cond)

        if self.use_cls_token:
            x = torch.cat(
                [self.cls_token.expand(batch_size, -1, -1), tokens],
                dim=1,
            )
        else:
            x = tokens

        x = self.encoder(x)
        x = self.final_norm(x)

        if self.use_cls_token:
            cls_latent = x[:, 0]
            roi_latents = x[:, 1:]
        else:
            cls_latent = None
            roi_latents = x

        return cls_latent, roi_latents

    def decode_and_reconstruct(self, roi_latents):
        dec = self.decoder(roi_latents)
        dec = self.decoder_norm(dec)
        reconstructions = [
            head(dec[:, i, :])
            for i, head in enumerate(self.reconstruction_heads)
        ]
        return reconstructions

    def forward(self, roi_tensors, dataset_id=None, subject_id=None, do_mae=True):
        batch_size = roi_tensors[0].shape[0]
        device = roi_tensors[0].device

        # Unmasked branch for contrastive retrieval
        cls_full, roi_full = self.encode_roi_tokens(
            roi_tensors,
            dataset_id=dataset_id,
            subject_id=subject_id,
            mask=None,
        )

        roi_mean_latent = roi_full.mean(dim=1)
        roi_attn_latent, roi_attn = self.attn_pool(roi_full)

        out = {
            "cls_latent": cls_full,
            "roi_latents": roi_full,
            "roi_mean_latent": roi_mean_latent,
            "roi_attn_latent": roi_attn_latent,
            "roi_attn": roi_attn,
        }

        if do_mae:
            mask = self.sample_mask(batch_size, device)
            _, roi_masked = self.encode_roi_tokens(
                roi_tensors,
                dataset_id=dataset_id,
                subject_id=subject_id,
                mask=mask,
            )
            reconstructions = self.decode_and_reconstruct(roi_masked)
            out["mask"] = mask
            out["reconstructions"] = reconstructions

        return out


# ============================================================
# 6. LIGHTNING MODULE
# ============================================================

class LitMultitaskMAEContrastive(pl.LightningModule):
    def __init__(self, roi_names, roi_dims, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["roi_names", "roi_dims"])

        self.model = ROIEncoderMAEContrastive(
            roi_names=roi_names,
            roi_dims=roi_dims,
            embed_dim=kwargs["embed_dim"],
            encoder_depth=kwargs["encoder_depth"],
            decoder_depth=kwargs["decoder_depth"],
            num_heads=kwargs["num_heads"],
            mlp_ratio=kwargs["mlp_ratio"],
            dropout=kwargs["dropout"],
            mask_ratio=kwargs["mask_ratio"],
            use_cls_token=kwargs["use_cls_token"],
            num_datasets=kwargs["num_datasets"],
            use_dataset_embedding=kwargs["use_dataset_embedding"],
            num_subjects=kwargs.get("num_subjects", 0),
            use_subject_embedding=kwargs.get("use_subject_embedding", False),
            use_film_conditioning=kwargs.get("use_film_conditioning", False),
        )

        self.brain_projector = nn.Sequential(
            nn.LayerNorm(kwargs["embed_dim"]),
            nn.Linear(kwargs["embed_dim"], kwargs["embed_dim"]),
            nn.GELU(),
            nn.Linear(kwargs["embed_dim"], 512),
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
        )

    def _select_brain_latent(self, out):
        pooling = self.hparams.pooling

        if pooling == "mean":
            return out["roi_mean_latent"]
        if pooling == "attention":
            return out["roi_attn_latent"]
        if pooling == "cls":
            if out["cls_latent"] is None:
                raise ValueError("pooling='cls' requires use_cls_token=True.")
            return out["cls_latent"]

        raise ValueError(f"Unknown pooling mode: {pooling}")

    @staticmethod
    def cosine_regression_loss(pred_clip, target_clip):
        pred_clip = F.normalize(pred_clip, dim=-1)
        target_clip = F.normalize(target_clip, dim=-1)
        return 1.0 - F.cosine_similarity(pred_clip, target_clip, dim=-1).mean()

    def _gather_with_grad(self, x):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = torch.distributed.nn.functional.all_gather(x)
            return torch.cat(gathered, dim=0)
        return x

    def _global_labels_for_local_batch(self, batch_size, device):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return torch.arange(batch_size, device=device)

        local_bs = torch.tensor([batch_size], device=device, dtype=torch.long)
        all_bs = torch.distributed.nn.functional.all_gather(local_bs)
        all_bs = torch.cat(all_bs, dim=0)

        offsets = torch.cumsum(all_bs, dim=0) - all_bs
        offset = offsets[self.global_rank]

        return torch.arange(batch_size, device=device) + offset

    def infonce_loss_distributed(self, pred_clip, target_clip):
        pred_clip = F.normalize(pred_clip, dim=-1)
        target_clip = F.normalize(target_clip, dim=-1)

        batch_size = pred_clip.shape[0]
        device = pred_clip.device

        all_pred_clip = self._gather_with_grad(pred_clip)
        all_target_clip = self._gather_with_grad(target_clip)

        labels = self._global_labels_for_local_batch(batch_size, device)

        logit_scale = self.logit_scale.exp().clamp(max=15.0)

        logits_b2c = logit_scale * (pred_clip @ all_target_clip.T)
        logits_c2b = logit_scale * (target_clip @ all_pred_clip.T)

        loss_b2c = F.cross_entropy(logits_b2c, labels)
        loss_c2b = F.cross_entropy(logits_c2b, labels)

        return 0.5 * (loss_b2c + loss_c2b)

    def _gather_no_grad(self, x):
        """
        Non-autograd distributed gather.
        Useful for integer IDs such as stimulus_id.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            x = x.contiguous()
            world_size = torch.distributed.get_world_size()
            gathered = [torch.zeros_like(x) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, x)
            return torch.cat(gathered, dim=0)

        return x


    @staticmethod
    def soft_cross_entropy(logits, targets):
        """
        Cross-entropy with soft/multi-positive targets.

        logits:  [B_local, B_global]
        targets: [B_local, B_global], rows sum to 1
        """
        log_probs = F.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()


    def multipositive_infonce_loss_distributed(
        self,
        pred_clip,
        target_clip,
        stimulus_id,
    ):
        """
        Multi-positive symmetric brain-to-CLIP / CLIP-to-brain InfoNCE.

        All samples with the same stimulus_id are treated as positives.
        This avoids treating repeated presentations of the same stimulus as negatives.
        """
        pred_clip = F.normalize(pred_clip, dim=-1)
        target_clip = F.normalize(target_clip, dim=-1)

        stimulus_id = stimulus_id.long()

        all_pred_clip = self._gather_with_grad(pred_clip)
        all_target_clip = self._gather_with_grad(target_clip)
        all_stimulus_id = self._gather_no_grad(stimulus_id).long().to(pred_clip.device)

        # targets[i, j] = 1 if local sample i and global sample j share the same stimulus
        targets = stimulus_id[:, None].eq(all_stimulus_id[None, :]).float()

        # Normalize so each row is a probability distribution.
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1.0)

        logit_scale = self.logit_scale.exp().clamp(max=15.0)

        logits_b2c = logit_scale * (pred_clip @ all_target_clip.T)
        logits_c2b = logit_scale * (target_clip @ all_pred_clip.T)

        loss_b2c = self.soft_cross_entropy(logits_b2c, targets)
        loss_c2b = self.soft_cross_entropy(logits_c2b, targets)

        return 0.5 * (loss_b2c + loss_c2b)


    def compute_clip_loss(self, pred_clip, target_clip, stimulus_id=None):
        cosine_loss = self.cosine_regression_loss(pred_clip, target_clip)

        if getattr(self.hparams, "use_multipositive_infonce", False):
            if stimulus_id is None:
                raise ValueError(
                    "stimulus_id must be provided when use_multipositive_infonce=True."
                )

            infonce_loss = self.multipositive_infonce_loss_distributed(
                pred_clip=pred_clip,
                target_clip=target_clip,
                stimulus_id=stimulus_id,
            )
        else:
            infonce_loss = self.infonce_loss_distributed(
                pred_clip=pred_clip,
                target_clip=target_clip,
            )

        loss_mode = self.hparams.loss_mode

        if loss_mode == "cosine_regression":
            total_loss = cosine_loss

        elif loss_mode == "infonce":
            total_loss = infonce_loss

        elif loss_mode == "cosine_plus_infonce":
            total_loss = cosine_loss + self.hparams.infonce_weight * infonce_loss

        else:
            raise ValueError(f"Unknown loss mode: {loss_mode}")

        return total_loss, cosine_loss, infonce_loss

    def compute_mae_loss(self, out, roi_tensors):
        mask = out["mask"]
        visible = ~mask

        masked_losses = []
        visible_losses = []

        for r, (pred, tgt) in enumerate(zip(out["reconstructions"], roi_tensors)):
            m_idx = mask[:, r]
            if m_idx.any():
                masked_losses.append(F.mse_loss(pred[m_idx], tgt[m_idx]))

            v_idx = visible[:, r]
            if self.hparams.visible_loss_weight > 0 and v_idx.any():
                visible_losses.append(F.mse_loss(pred[v_idx], tgt[v_idx]))

        masked_loss = torch.stack(masked_losses).mean() if masked_losses else torch.tensor(0.0, device=self.device)

        if visible_losses:
            visible_loss = torch.stack(visible_losses).mean()
        else:
            visible_loss = torch.tensor(0.0, device=self.device)

        if self.hparams.visible_loss_weight > 0:
            mae_loss = (
                masked_loss + self.hparams.visible_loss_weight * visible_loss
            ) / (1.0 + self.hparams.visible_loss_weight)
        else:
            mae_loss = masked_loss

        return mae_loss, masked_loss, visible_loss

    @staticmethod
    @torch.no_grad()
    def retrieval_metrics(pred_clip, target_clip):
        pred_clip = F.normalize(pred_clip, dim=-1)
        target_clip = F.normalize(target_clip, dim=-1)

        logits = pred_clip @ target_clip.T
        labels = torch.arange(logits.shape[0], device=logits.device)

        ranks = logits.argsort(dim=1, descending=True)
        correct = ranks.eq(labels[:, None])

        top1 = correct[:, :1].any(dim=1).float().mean()
        top5 = correct[:, :5].any(dim=1).float().mean()
        top10 = correct[:, :10].any(dim=1).float().mean()

        pos_sim = logits.diag().mean()
        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        neg_sim = logits[~eye].mean()

        return top1, top5, top10, pos_sim, neg_sim

    def _shared_step(self, batch, stage):
        roi_tensors = [x.to(self.device, non_blocking=True) for x in batch["roi_tensors"]]
        target_clip = batch["clip_features"].to(self.device, non_blocking=True)
        dataset_id = batch["dataset_id"].to(self.device, non_blocking=True)
        subject_id = batch["subject_id"].to(self.device, non_blocking=True)
        stimulus_id = batch["stimulus_id"].to(self.device, non_blocking=True)
        with torch.no_grad():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                all_stimulus_id_for_log = self._gather_no_grad(stimulus_id).long().to(self.device)
            else:
                all_stimulus_id_for_log = stimulus_id.long()

            num_pos = stimulus_id[:, None].eq(all_stimulus_id_for_log[None, :]).sum(dim=1).float()
            avg_num_pos = num_pos.mean()

        do_mae = bool(self.hparams.use_mae)

        out = self.model(
            roi_tensors,
            dataset_id=dataset_id,
            subject_id=subject_id,
            do_mae=do_mae,
        )

        total_loss = torch.tensor(0.0, device=self.device)
        clip_loss = torch.tensor(0.0, device=self.device)
        cosine_loss = torch.tensor(0.0, device=self.device)
        infonce_loss = torch.tensor(0.0, device=self.device)
        mae_loss = torch.tensor(0.0, device=self.device)
        masked_loss = torch.tensor(0.0, device=self.device)
        visible_loss = torch.tensor(0.0, device=self.device)

        if self.hparams.use_contrastive:
            brain_latent = self._select_brain_latent(out)
            pred_clip = self.brain_projector(brain_latent)

            clip_loss, cosine_loss, infonce_loss = self.compute_clip_loss(
                pred_clip=pred_clip,
                target_clip=target_clip,
                stimulus_id=stimulus_id,
            )

            total_loss = self.hparams.contrastive_weight * total_loss + self.hparams.contrastive_weight * clip_loss

            top1, top5, top10, pos_sim, neg_sim = self.retrieval_metrics(
                pred_clip.detach(),
                target_clip.detach(),
            )
        else:
            top1 = top5 = top10 = pos_sim = neg_sim = torch.tensor(0.0, device=self.device)

        if self.hparams.use_mae:
            mae_loss, masked_loss, visible_loss = self.compute_mae_loss(out, roi_tensors)
            total_loss =  total_loss + self.hparams.mae_weight * mae_loss

        sync_dist = torch.distributed.is_available() and torch.distributed.is_initialized()

        self.log(f"{stage}/loss", total_loss, prog_bar=True, sync_dist=sync_dist)
        self.log(f"{stage}/clip_loss", clip_loss, sync_dist=sync_dist)
        self.log(f"{stage}/cosine_loss", cosine_loss, sync_dist=sync_dist)
        self.log(f"{stage}/infonce_loss", infonce_loss, sync_dist=sync_dist)
        self.log(f"{stage}/avg_num_positives", avg_num_pos, sync_dist=sync_dist)

        self.log(f"{stage}/mae_loss", mae_loss, sync_dist=sync_dist)
        self.log(f"{stage}/masked_loss", masked_loss, sync_dist=sync_dist)
        self.log(f"{stage}/visible_loss", visible_loss, sync_dist=sync_dist)

        self.log(f"{stage}/top1", top1, prog_bar=True, sync_dist=sync_dist)
        self.log(f"{stage}/top5", top5, prog_bar=True, sync_dist=sync_dist)
        self.log(f"{stage}/top10", top10, sync_dist=sync_dist)
        self.log(f"{stage}/pos_sim", pos_sim, sync_dist=sync_dist)
        self.log(f"{stage}/neg_sim", neg_sim, sync_dist=sync_dist)
        self.log(f"{stage}/pos_minus_neg", pos_sim - neg_sim, prog_bar=True, sync_dist=sync_dist)
        self.log(f"{stage}/logit_scale", self.logit_scale.exp().detach(), sync_dist=sync_dist)

        return total_loss

    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and self.current_epoch == 0 and self.global_rank == 0:
            print("\n==============================")
            print("First training batch check")
            print("==============================")
            for i in range(min(5, len(batch["stimulus_clean"]))):
                print("----------------------------------------")
                print("raw beta name:      ", batch["raw_betas_name"][i])
                print("dataset_name:       ", batch["dataset_name"][i])
                print("dataset_id:         ", batch["dataset_id"][i].item())
                print("subject_name:       ", batch["subject_name"][i])
                print("subject_key:        ", batch["subject_key"][i])
                print("subject_id:         ", batch["subject_id"][i].item())
                print("stimulus_clean:     ", batch["stimulus_clean"][i])
                print("canonical filename: ", batch["canonical_filename"][i])
                print("n_duplicate_rows:   ", batch["n_duplicate_rows"][i].item())
                print("clip norm:          ", batch["clip_features"][i].norm().item())

        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if (
                name.endswith(".bias")
                or "norm" in name.lower()
                or "layernorm" in name.lower()
                or "logit_scale" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "weight_decay": 0.0,
                },
            ]
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ============================================================
# 7. ARGPARSE OVERRIDES
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--max_train_samples", type=str, default=None)
    parser.add_argument("--max_val_samples", type=str, default=None)
    parser.add_argument("--dataset_tag_filter", type=str, default=None)

    parser.add_argument("--subject_filter", type=str, default=None)

    parser.add_argument("--cache_selected_samples", action="store_true")
    parser.add_argument("--lazy", action="store_true")

    parser.add_argument("--pooling", type=str, default=None, choices=["mean", "attention", "cls"])
    parser.add_argument("--loss_mode", type=str, default=None,
                        choices=["cosine_regression", "infonce", "cosine_plus_infonce"])
    parser.add_argument("--infonce_weight", type=float, default=None)

    parser.add_argument("--use_multipositive_infonce", action="store_true")
    parser.add_argument("--no_multipositive_infonce", action="store_true")
    parser.add_argument("--contrastive_weight", type=float, default=None)

    parser.add_argument("--use_mae", action="store_true")
    parser.add_argument("--no_mae", action="store_true")
    parser.add_argument("--mae_weight", type=float, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--visible_loss_weight", type=float, default=None)

    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--no_contrastive", action="store_true")

    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--no_cls_token", action="store_true")

    parser.add_argument("--use_dataset_embedding", action="store_true")
    parser.add_argument("--no_dataset_embedding", action="store_true")
    parser.add_argument("--use_subject_embedding", action="store_true")
    parser.add_argument("--no_subject_embedding", action="store_true")
    parser.add_argument("--use_film_conditioning", action="store_true")
    parser.add_argument("--no_film_conditioning", action="store_true")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--encoder_depth", type=int, default=None)
    parser.add_argument("--decoder_depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)

    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=None)

    return parser.parse_args()


def _parse_optional_int(value: Optional[str], default_value):
    if value is None:
        return default_value
    value_lower = str(value).lower()
    if value_lower in {"all", "none", "null"}:
        return None
    return int(value)


def apply_cli_overrides(config: Dict, args) -> Dict:
    config = dict(config)

    if args.run_name is not None:
        config["run_name"] = args.run_name

    config["max_train_samples"] = _parse_optional_int(args.max_train_samples, config["max_train_samples"])
    config["max_val_samples"] = _parse_optional_int(args.max_val_samples, config["max_val_samples"])

    if args.dataset_tag_filter is not None:
        if args.dataset_tag_filter.lower() in {"none", "all", "null"}:
            config["dataset_tag_filter"] = None
        else:
            config["dataset_tag_filter"] = args.dataset_tag_filter

    if args.subject_filter is not None:
        if args.subject_filter.lower() in {"none", "all", "null"}:
            config["subject_filter"] = None
        else:
            config["subject_filter"] = args.subject_filter
    
    if args.cache_selected_samples:
        config["cache_selected_samples"] = True
    if args.lazy:
        config["cache_selected_samples"] = False

    if args.pooling is not None:
        config["pooling"] = args.pooling
    if args.loss_mode is not None:
        config["loss_mode"] = args.loss_mode
    if args.infonce_weight is not None:
        config["infonce_weight"] = args.infonce_weight

    if args.use_multipositive_infonce:
        config["use_multipositive_infonce"] = True

    if args.no_multipositive_infonce:
        config["use_multipositive_infonce"] = False

    if args.use_mae:
        config["use_mae"] = True
    if args.no_mae:
        config["use_mae"] = False
    if args.mae_weight is not None:
        config["mae_weight"] = args.mae_weight
    if args.mask_ratio is not None:
        config["mask_ratio"] = args.mask_ratio
    if args.visible_loss_weight is not None:
        config["visible_loss_weight"] = args.visible_loss_weight
    if args.contrastive_weight is not None:
        config["contrastive_weight"] = args.contrastive_weight

    if args.use_contrastive:
        config["use_contrastive"] = True
    if args.no_contrastive:
        config["use_contrastive"] = False

    if args.use_cls_token:
        config["use_cls_token"] = True
    if args.no_cls_token:
        config["use_cls_token"] = False

    if args.use_dataset_embedding:
        config["use_dataset_embedding"] = True
    if args.no_dataset_embedding:
        config["use_dataset_embedding"] = False

    if args.use_subject_embedding:
        config["use_subject_embedding"] = True
    if args.no_subject_embedding:
        config["use_subject_embedding"] = False

    if args.use_film_conditioning:
        config["use_film_conditioning"] = True
    if args.no_film_conditioning:
        config["use_film_conditioning"] = False

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay

    if args.embed_dim is not None:
        config["embed_dim"] = args.embed_dim
    if args.encoder_depth is not None:
        config["encoder_depth"] = args.encoder_depth
    if args.decoder_depth is not None:
        config["decoder_depth"] = args.decoder_depth
    if args.num_heads is not None:
        config["num_heads"] = args.num_heads
    if args.dropout is not None:
        config["dropout"] = args.dropout

    if args.devices is not None:
        config["devices"] = args.devices
        if args.devices > 1 and config["strategy"] == "auto":
            config["strategy"] = "ddp"
    if args.strategy is not None:
        config["strategy"] = args.strategy

    if args.use_early_stopping:
        config["use_early_stopping"] = True
    if args.early_stopping_patience is not None:
        config["early_stopping_patience"] = args.early_stopping_patience

    return config


# ============================================================
# 8. MAIN
# ============================================================

def main():
    args = parse_args()
    config = apply_cli_overrides(DEFAULT_CONFIG, args)

    seed_everything(config["seed"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    Path(config["default_root_dir"]).mkdir(parents=True, exist_ok=True)

    if wandb.run is not None:
        wandb.finish()

    # print("\n==============================")
    # print("Loading MOSAIC datasets")
    # print("==============================")

    # dataset = mosaic.load(
    #     names_and_subjects={
    #         "NaturalScenesDataset": "all",
    #         "THINGS": "all",
    #         "deeprecon": "all",
    #         "BOLD5000": "all",
    #         "BOLDMomentsDataset": "all",
    #         "GenericObjectDecoding": "all",
    #         "HumanActionsDataset": "all",
    #         "NaturalObjectDataset": "all",
    #     },
    #     folder=config["data_folder"],
    #     parse_betas=True,
    # )

    # print("MOSAIC dataset length:", len(dataset))

    # print("\n==============================")
    # print("Selecting train/val indices")
    # print("==============================")

    # train_idx, val_idx = compute_selected_indices(config, dataset)

    print("\n==============================")
    print("Building/loading fast index cache")
    print("==============================")

    index_df = pd.read_pickle(config["index_cache_path"])

    print("\n==============================")
    print("Selecting train/val indices from cache")
    print("==============================")

    train_idx, val_idx = compute_selected_indices_from_cache(config, index_df)

    print("Train samples selected:", len(train_idx))
    print("Val samples selected:", len(val_idx))

    print("\n==============================")
    print("Loading MOSAIC datasets for training")
    print("==============================")

    dataset = mosaic.load(
        names_and_subjects={
            "NaturalScenesDataset": "all",
            "THINGS": "all",
            "deeprecon": "all",
            "BOLD5000": "all",
            "BOLDMomentsDataset": "all",
            "GenericObjectDecoding": "all",
            "HumanActionsDataset": "all",
            "NaturalObjectDataset": "all",
        },
        folder=config["data_folder"],
        parse_betas=True,
    )

    print("MOSAIC dataset length:", len(dataset))

    print("Train samples selected:", len(train_idx))
    print("Val samples selected:", len(val_idx))

    # if config["use_subject_embedding"]:
    #     print("\n==============================")
    #     print("Building subject mapping")
    #     print("==============================")
    #     subject_name_to_id = build_subject_mapping_from_valid_mask(config, dataset)
    #     config["subject_name_to_id"] = subject_name_to_id
    #     config["num_subjects"] = len(subject_name_to_id)
    #     print("Number of subject keys:", config["num_subjects"])
    #     print("First 10 subject keys:", list(subject_name_to_id.items())[:10])
    # else:
    #     config["subject_name_to_id"] = {}
    #     config["num_subjects"] = 0
    #     print("\nSubject embedding disabled: no subject mapping is built.")

    if config["use_subject_embedding"]:
        print("\n==============================")
        print("Building subject mapping from index cache")
        print("==============================")

        subject_name_to_id = build_subject_mapping_from_cache(config, index_df)

        config["subject_name_to_id"] = subject_name_to_id
        config["num_subjects"] = len(subject_name_to_id)

        print("Number of subject keys:", config["num_subjects"])
        print("First 10 subject keys:", list(subject_name_to_id.items())[:10])

        if config["num_subjects"] == 0:
            raise ValueError(
                "No subjects found for the current dataset/subject filters. "
                "Check dataset_tag_filter and subject_filter."
            )
    else:
        config["subject_name_to_id"] = {}
        config["num_subjects"] = 0
        print("\nSubject embedding disabled: no subject mapping is built.")

    print("\n==============================")
    print("Loading ROI schema")
    print("==============================")

    with open(config["schema_path"], "r") as f:
        schema = json.load(f)

    roi_names = schema["roi_names"]
    roi_dims = [schema["roi_dims"][r] for r in roi_names]

    print("Number of ROIs:", len(roi_names))
    print("First 5 ROIs:", roi_names[:5])
    print("First 5 ROI dims:", roi_dims[:5])

    data_module = MosaicMultitaskDataModule(
        mosaic_dataset=dataset,
        roi_names=roi_names,
        config=config,
        train_idx=train_idx,
        val_idx=val_idx,
    )

    lit_model = LitMultitaskMAEContrastive(
        roi_names=roi_names,
        roi_dims=roi_dims,
        **config,
    )

    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=config["run_name"],
        save_dir=config["default_root_dir"],
        log_model=False,
    )



    checkpoint_dir = Path(config["default_root_dir"]) / "checkpoints_multitask_mae_clip"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Retrieval is the target, so you may prefer val/top5 once stable.
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        # filename="mt--{epoch:03d}-{val_loss:.4f}-{val_top1:.4f}",
        filename = (
            
            f"MT-dataset{DEFAULT_CONFIG['dataset_tag_filter']}"
            f"subject{DEFAULT_CONFIG['subject_filter']}"
            f"batchsize{DEFAULT_CONFIG['batch_size']}"
            f"lr{DEFAULT_CONFIG['lr']}"
            f"Wdecay{DEFAULT_CONFIG['weight_decay']}"
            f"emb_dim{DEFAULT_CONFIG['embed_dim']}"
            f"-encdepth{DEFAULT_CONFIG['encoder_depth']}"
            f"-decdepth{DEFAULT_CONFIG['decoder_depth']}"
            f"-numHeads{DEFAULT_CONFIG['num_heads']}"
            f"-dropout{DEFAULT_CONFIG['dropout']}"
            f"-pooling{DEFAULT_CONFIG['pooling']}"
            f"-ContlossMode{DEFAULT_CONFIG['loss_mode']}"
            f"-infonce_w{DEFAULT_CONFIG['infonce_weight']}"
            f"-contr_w{DEFAULT_CONFIG['contrastive_weight']}"
            f"-mae{DEFAULT_CONFIG['use_mae']}"
            f"-mae_weight{DEFAULT_CONFIG['mae_weight']}"
            f"-mask_ratio{DEFAULT_CONFIG['mask_ratio']}"
            f"-visibleLossW{DEFAULT_CONFIG['visible_loss_weight']}"
            f"-multipositive{DEFAULT_CONFIG['use_multipositive_infonce']}"
            f"-epoch{{epoch:03d}}"
        ),

        # monitor="val/loss",
        # mode="min",
        monitor="val/top5",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if config["use_early_stopping"]:
        callbacks.append(
            EarlyStopping(
                # monitor="val/loss",
                # mode="min",
                monitor="val/top5",
                mode="max",
                patience=config["early_stopping_patience"],
                min_delta=1e-4,
            )
                )

    trainer = pl.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        strategy=config["strategy"],
        max_epochs=config["max_epochs"],
        precision=config["precision"],
        logger=wandb_logger,
        gradient_clip_val=config["gradient_clip_val"],
        log_every_n_steps=config["log_every_n_steps"],
        callbacks=callbacks,
        num_sanity_val_steps=config["num_sanity_val_steps"],
        default_root_dir=config["default_root_dir"],
    )

    print("\n==============================")
    print("Starting training")
    print("==============================")
    print("Run name:", config["run_name"])
    print("Subject filter:", config.get("subject_filter", None))
    print("Dataset tag:", config["dataset_tag_filter"])
    print("Pooling:", config["pooling"])
    print("Use CLS token:", config["use_cls_token"])
    print("Use contrastive:", config["use_contrastive"])
    print("Loss mode:", config["loss_mode"])
    print("InfoNCE weight:", config["infonce_weight"])
    print("Use MAE:", config["use_mae"])
    print("MAE weight:", config["mae_weight"])
    print("Mask ratio:", config["mask_ratio"])
    print("Visible loss weight:", config["visible_loss_weight"])
    print("Use dataset embedding:", config["use_dataset_embedding"])
    print("Use subject embedding:", config["use_subject_embedding"])
    print("Use FiLM conditioning:", config["use_film_conditioning"])
    print("Max train samples:", config["max_train_samples"])
    print("Max val samples:", config["max_val_samples"])
    print("Cache selected samples:", config["cache_selected_samples"])
    print("Val same as train:", config["val_same_as_train"])
    print("Devices:", config["devices"])
    print("Strategy:", config["strategy"])

    trainer.fit(lit_model, datamodule=data_module)
    wandb.finish()


if __name__ == "__main__":
    main()

