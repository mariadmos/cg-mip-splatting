<p align="center">
  <h1 align="center">Mip-Splatting — comparison pipeline</h1>
  <p align="center">Course project based on <a href="https://niujinshuchong.github.io/mip-splatting/">Mip-Splatting: Alias-free 3D Gaussian Splatting</a> (Yu, Chen, Huang, Sattler &amp; Geiger — CVPR 2024 Best Student Paper)</p>
</p>

This repository is a fork of the official Mip-Splatting code. In addition to the original method, it contains our
**own comparison pipeline** that reproduces the paper's central experiment (train at one scale, test
at several) on a real object we captured on video, comparing Mip-Splatting against the original 3DGS
and 3DGS+EWA.

3D Gaussian Splatting (3DGS) renders fast but produces aliasing artifacts when the scene is viewed at a
scale different from the training scale (zoom in/out, or resolution changes). Mip-Splatting solves this
with two new filters: a **3D smoothing filter** (limits each Gaussian's frequency to the sampling
rate of the training cameras) and a **2D Mip filter** (replaces the fixed dilation of 3DGS with a
per-pixel low-pass filter). Both inherit the rasterization and optimization pipeline of the original 3DGS.

## Our pipeline

Entry point: [`pipeline_comparacao_mipsplatting.ipynb`](pipeline_comparacao_mipsplatting.ipynb).
Object of study: a stone lion filmed on video (`data/leao_video/`). Notebook steps:

1. **Data**: extracts the sharpest frames from the video (Laplacian variance, reduces motion blur) and
   reconstructs cameras + sparse point cloud with COLMAP (`convert.py`).
2. **Mip-Splatting**: trains the paper's method (3D filter enabled, `--kernel_size 0.1`) at `-r 2`.
3. **Baselines**: trains vanilla **3DGS** and **3DGS+EWA** on the same COLMAP, same split and same
   hyperparameters, using the official 3DGS repository (`baselines/gaussian-splatting/`).
4. **Multi-scale evaluation**: renders and measures (PSNR/SSIM/LPIPS-VGG, same implementation for all three)
   the three models at `-r 1, 2, 4, 8`. `-r 1` is zoom-in, `-r 8` is extreme zoom-out; `-r 2` is the
   training scale.
5. **Visualization**: side-by-side crops of the three models + ground truth at each scale.

Each step is automatically skipped if the artifact already exists (frames, COLMAP, checkpoints, renders,
metrics), so re-running the notebook from scratch is safe.

### Result

On the lion held-out set (trained at `-r 2`, 30k iterations), we reproduced the paper's central claim on our
own object:

| model | PSNR -r8 (zoom-out) | PSNR -r2 (train) | PSNR -r1 (zoom-in) | LPIPS -r8 |
|---|---|---|---|---|
| **Mip-Splatting** | **22.91** | 22.50 | **22.16** | **0.089** |
| 3DGS | 19.19 | 22.08 | 21.12 | 0.165 |
| 3DGS+EWA | 22.43 | 22.44 | 21.86 | 0.112 |

At the training scale the three are tied. Outside it, vanilla 3DGS collapses (-3.7 dB at zoom-out); Mip-Splatting is the sharpest and most stable across all scales.

The notebook is parameterized (`VIDEO`, `SOURCE`, `N_FRAMES` in the configuration cell) and runs on
any object captured on video the same way: it has also been applied to a bust (`data/busto_video/`).

## Our additions

Besides the code inherited from 3DGS/Mip-Splatting (`train.py`, `render.py`, `scene/`,
`gaussian_renderer/`, `submodules/`, ...):
- [`pipeline_comparacao_mipsplatting.ipynb`](pipeline_comparacao_mipsplatting.ipynb): complete pipeline, main entry point.

## Installation

Two conda environments: one for Mip-Splatting (rasterizer with `kernel_size`) and another for the 3DGS
baselines (rasterizer with `antialiasing`). The two filters require different builds of the CUDA rasterizer and
cannot coexist in the same installed package — hence the separate environments. COLMAP must be installed and on
the PATH (used by `convert.py`).

### 1. Mip-Splatting
```
git clone git@github.com:autonomousvision/mip-splatting.git
cd mip-splatting

conda create -y -n mip-splatting python=3.8
conda activate mip-splatting

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
```

### 2. Official 3DGS
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive baselines/gaussian-splatting

conda create -y -n gs3dgs --clone mip-splatting
conda activate gs3dgs
CUDA_HOME=$CONDA_PREFIX pip install \
    baselines/gaussian-splatting/submodules/diff-gaussian-rasterization \
    baselines/gaussian-splatting/submodules/simple-knn
```
Use the environment's own CUDA (`$CONDA_PREFIX`) to compile the rasterizer; a version different from the one used to compile torch breaks the build.

## Other datasets

Besides our video, the code accepts the original formats from the paper:

```
# Blender / NeRF synthetic multi-scale
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale

# Mip-NeRF 360: download from https://jonbarron.info/mipnerf360/
```

## Credits

Training/rendering/rasterization code is from the original [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/),
which in turn is a fork of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

### Citation
```bibtex
@InProceedings{Yu2024MipSplatting,
    author    = {Yu, Zehao and Chen, Anpei and Huang, Binbin and Sattler, Torsten and Geiger, Andreas},
    title     = {Mip-Splatting: Alias-free 3D Gaussian Splatting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {19447-19456}
}
```
