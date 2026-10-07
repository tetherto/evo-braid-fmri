# Many brains, one geometry — model, training code and figure reproduction

This folder contains the trained model, the code used to train it, and a notebook that reproduces every data figure in the paper.

```
many_brains_one_geometry_release/
├── checkpoints/
│   ├── checkpoint.ckpt           # model used for all results in the paper
│   └── last_checkpoint.ckpt      # last checkpoint of the same run, compared in Fig. S3
├── training/
│   └── train_mosaic_brain_clip.py
├── reproduce_figures.ipynb       # reproduces Fig. 2–5 and Supplementary Fig. S1–S3
├── results/                      # intermediate results read by the notebook
├── resources/atlases/            # HCP-MMP1.0 atlas (volumetric, MNI) for the cortical maps; see its LICENSE.md
├── figures/                      # every panel, written by the last section of the notebook
└── requirements.txt
```
## Training

`training/train_mosaic_brain_clip.py` is the training script. Its defaults reproduce `checkpoint.ckpt`:

- 4 GPUs with DDP, 256 samples per GPU (effective batch 1,024), fp32, gradient clipping at 1.0, seed 42
- AdamW, learning rate 1e-4, weight decay 1e-3 (no decay on biases, normalization parameters and the logit scale), cosine schedule.
- Early stopping and checkpoint selection on `val/top5` (patience 5, min delta 1e-4)
- 379 ROI tokens (360 HCP-MMP1.0 cortical parcels + 19 CIFTI subcortical structures), transformer encoder (depth 4, width 512, 4 heads), attention pooling, learned participant embeddings, multi-positive InfoNCE

Edit the data paths at the top of the script (`DEFAULT_CONFIG`) before running. It needs the MOSAIC data and the `mosaic` data-loader package.

## Checkpoints


| File | Use | SHA-256 |
|---|---|---|
| `checkpoint.ckpt` | Selected checkpoint; all results and figures in the paper | `7f5d1634553356903f923f38b5f21a2e13e96a6fc26e893e80e77386a0efbfa0` |
| `last_checkpoint.ckpt` | Last checkpoint| `30dc1283d474ccca2703a3590b659f877f08e006d2763dc909a536d3d89a085c` |

Both files are standard PyTorch Lightning checkpoints with the model weights, hyperparameters and training metadata; the optimizer state was removed to keep them under GitHub's file-size limit, so they load for evaluation (`LitMultitaskMAEContrastive.load_from_checkpoint`) but cannot resume training. They are stored with Git LFS: after cloning, run `git lfs pull` if the files are only small pointers.


## Reproducing the results

1. Create an environment with the packages in `requirements.txt` (suggested with Python 3.10).
2. Open `reproduce_figures.ipynb` from this folder and run all cells; a GPU is not required.

The notebook reads the intermediate results in `results/`, so it does not need the model or a GPU. Fig. S3 runs both checkpoints on all held-out trials and needs a GPU and the MOSAIC data. Three panels show stimulus images or video frames (Fig. 2d, the collages in Fig. 3a and Fig. S1a); these need the MOSAIC stimuli. Set the `MOSAIC_ROOT` environment variable (or edit the setup cell) to point to the MOSAIC data folder. Without it, those panels show placeholders and everything else still runs. Fig. 4d is rendered with BrainSpace/VTK, which needs a working (offscreen) OpenGL setup.


## Notes for users of the data

- HAD has no held-out trials (all HAD trials are in the training split), so it does not appear in retrieval analyses.
- The ROI-to-network grouping (`results/.../analysis5_glasser_roi_group_mapping.csv`) assigns Glasser parcels by name-prefix matching, which places a few parcels in unexpected groups (e.g. v23ab in early visual, a10p in temporal/auditory, PF/PFm/PFcm/PFop/PFt in frontal/control). The published results use this mapping as is.

## Third-party data

`resources/atlases/The-HCP-MMP1.0-atlas-in-FSL/` is a copy of https://github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL (volumetric HCP-MMP1.0 atlas in MNI space), distributed under its own licence (`LICENSE.md` in that folder).
