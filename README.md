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

## Checkpoints

Both checkpoints come from one training run (`training/train_mosaic_brain_clip.py` with its default settings).

| File | Use | SHA-256 |
|---|---|---|
| `checkpoint.ckpt` | Selected checkpoint; all results and figures in the paper | `7f5d1634553356903f923f38b5f21a2e13e96a6fc26e893e80e77386a0efbfa0` |
| `last_checkpoint.ckpt` | Last checkpoint; compared with the selected checkpoint in Fig. S3 | `30dc1283d474ccca2703a3590b659f877f08e006d2763dc909a536d3d89a085c` |

Both files are standard PyTorch Lightning checkpoints with the model weights, hyperparameters and training metadata; the optimizer state was removed to keep them under GitHub's file-size limit, so they load for evaluation (`LitMultitaskMAEContrastive.load_from_checkpoint`) but cannot resume training. They are stored with Git LFS: after cloning, run `git lfs pull` if the files are only small pointers.

`checkpoint.ckpt` is the checkpoint with the highest value of the monitored metric (in-batch Top-5 retrieval on the held-out split). No separate validation split was held out from the training data, so this selection used the held-out evaluation split (see Methods).

## Training

`training/train_mosaic_brain_clip.py` is the training script. Its defaults reproduce `checkpoint.ckpt`:

- 4 GPUs with DDP, 256 samples per GPU (effective batch 1,024), fp32, gradient clipping at 1.0, seed 42
- AdamW, learning rate 1e-4, weight decay 1e-3 (no decay on biases, normalization parameters and the logit scale), cosine schedule over at most 80 epochs
- Early stopping and checkpoint selection on `val/top5` (patience 5, min delta 1e-4)
- 379 ROI tokens (360 HCP-MMP1.0 cortical parcels + 19 CIFTI subcortical structures), transformer encoder (depth 4, width 512, 4 heads), attention pooling, learned participant embeddings, multi-positive InfoNCE

Edit the data paths at the top of the script (`DEFAULT_CONFIG`) before running. It needs the MOSAIC data and the `mosaic` data-loader package.

## Reproducing the figures

1. Create an environment with the packages in `requirements.txt` (tested with Python 3.10).
2. Open `reproduce_figures.ipynb` from this folder and run all cells. It takes about 6 minutes on a CPU; a GPU is not required.
3. Every panel is written to `figures/` (e.g. `figures/Fig3d.pdf`).

The notebook reads the intermediate results in `results/`, so apart from Fig. S3 it does not need the model or a GPU. Fig. S3 runs both checkpoints on all held-out trials and needs a GPU and the MOSAIC data. Three panels show stimulus images or video frames (Fig. 2d, the collages in Fig. 3a and Fig. S1a); these need the MOSAIC stimuli. Set the `MOSAIC_ROOT` environment variable (or edit the setup cell) to point to the MOSAIC data folder. Without it, those panels show placeholders and everything else still runs. Fig. 4d is rendered with BrainSpace/VTK, which needs a working (offscreen) OpenGL setup.

## Where each panel comes from

| Panel | Notebook section | Output in `figures/` |
|---|---|---|
| Fig. 2a | Figure 2 › 2a | `Fig2a.pdf` |
| Fig. 2b | Figure 2 › 2b | `Fig2b.pdf` |
| Fig. 2c | Figure 2 › 2c | `Fig2c.pdf` |
| Fig. 2d | Figure 2 › 2d | `Fig2d.pdf` (one page per row) |
| Fig. 3a | Figure 3 › 3a | `Fig3a_tsne.pdf`, `Fig3a_collages.pdf` |
| Fig. 3b | Figure 3 › 3b | `Fig3b.pdf` |
| Fig. 3c | Figure 3 › 3c | `Fig3c_top.pdf`, `Fig3c_bottom.pdf` |
| Fig. 3d–f | Figure 3 › 3d, 3e, 3f | `Fig3d.pdf`, `Fig3e.pdf`, `Fig3f.pdf` |
| Fig. 3g | not included | — |
| Fig. 4a–c | Figure 4 › 4a–c | `Fig4abc.pdf` |
| Fig. 4d | Figure 4 › 4d | `Fig4d.pdf` |
| Fig. 4e | Figure 4 › 4e | `Fig4e_radars.pdf`, `Fig4e_bar.pdf` |
| Fig. 4f | Figure 4 › 4f | `Fig4f.pdf` |
| Fig. 5a–e | Figure 5 › 5a–5e | `Fig5a.pdf` … `Fig5e.pdf` |
| Fig. S1a–c | Supplementary Figure S1 | `FigS1a.pdf`, `FigS1b_top.pdf`, `FigS1b_bottom.pdf`, `FigS1c.pdf` |
| Fig. S2a–c | Supplementary Figure S2 | `FigS2a.pdf`, `FigS2b.pdf` (panel b of the saved figure), `FigS2c.pdf` |
| Fig. S3 | Supplementary Figure S3 | `FigS3.pdf` |

Fig. 1 is a schematic and has no code.

### Differences from the published figures

The published figures were assembled and lightly edited in a vector editor after export. The notebook produces the underlying panels, so the following differ in appearance only:

- Panel letters, some titles, the "(Cxx)" suffixes on some cluster labels, and line breaks in a few axis labels were edited or removed.
- Label wording: "Subject" → "Participant" (Fig. 2b,c), "deeprecon" → "Deeprecon", and the NOD panel was cropped from Fig. 2b.
- Fig. 2d: rows were restyled; the notebook draws each row as a separate page.
- Fig. 4d: the colour bar was redrawn.
- Fig. 4e: the radar plots and the central bar plot are two separate PDFs.
- Fig. S2b is panel b of the saved three-panel figure.
- Fig. S1 cluster names: a few names were shortened in the published figure (e.g. "Stone bridges" for "Stone bridges & viaducts").

The notebook also corrects or updates the following:

- **Fig. 5d**: the dashed chance line is drawn at 10/125 = 8%, matching the candidate pool of 125 stimuli used for the curves and the Methods. The published panel draws it at 10%.
- **Fig. S1b (bottom)**: the y-axis label reads "Number of unique stimuli". The published panel reads "Fraction of unique stimuli", although it shows counts.

## Numerical check

Each reproduced panel was compared with the PDF that produced the published panel by extracting all embedded text (every number, tick label and annotation). All panels match exactly except the intended changes listed above. Examples: global Top-10 = 15.4%, identity margin 0.094 (P = 6.4 × 10⁻¹²), semantic margin 0.043 (P = 3.9 × 10⁻¹⁰), and every value in the ablation heatmaps.

## Notes for users of the data

- Fig. 2c ranks each participant's queries against that participant's own unique held-out stimuli (e.g. 200 or 824 candidates for NOD participants), not against all stimuli of the participant's dataset.
- HAD has no held-out trials (all HAD trials are in the training split), so it does not appear in retrieval analyses.
- The ROI-to-network grouping (`results/.../analysis5_glasser_roi_group_mapping.csv`) assigns Glasser parcels by name-prefix matching, which places a few parcels in unexpected groups (e.g. v23ab in early visual, a10p in temporal/auditory, PF/PFm/PFcm/PFop/PFt in frontal/control). The published results use this mapping as is.

## Third-party data

`resources/atlases/The-HCP-MMP1.0-atlas-in-FSL/` is a copy of https://github.com/mbedini/The-HCP-MMP1.0-atlas-in-FSL (volumetric HCP-MMP1.0 atlas in MNI space), distributed under its own licence (`LICENSE.md` in that folder).
