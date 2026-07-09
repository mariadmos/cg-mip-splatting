<p align="center">
  <h1 align="center">Mip-Splatting — pipeline de comparação</h1>
  <p align="center">Projeto de curso baseado em <a href="https://niujinshuchong.github.io/mip-splatting/">Mip-Splatting: Alias-free 3D Gaussian Splatting</a> (Yu, Chen, Huang, Sattler &amp; Geiger — CVPR 2024 Best Student Paper)</p>
</p>

Este repositório é um fork do código oficial do Mip-Splatting. Além do método original, ele contém um
**pipeline próprio de comparação** que reproduz o experimento central do paper (treino numa escala, teste
em várias) sobre um objeto real capturado por nós em vídeo, comparando o Mip-Splatting contra o 3DGS
original e o 3DGS+EWA.

3D Gaussian Splatting (3DGS) renderiza rápido mas gera artefatos de aliasing quando a cena é vista numa
escala diferente da escala de treino (zoom in/out, ou mudança de resolução). Mip-Splatting resolve isso
com dois filtros novos: um **filtro 3D de suavização** (limita a frequência de cada Gaussiana à taxa de
amostragem das câmeras de treino) e um **filtro 2D Mip** (substitui a dilatação fixa do 3DGS por um
filtro passa-baixa por pixel). Ambos herdam a rasterização e o pipeline de otimização do 3DGS original.

## Nosso pipeline

Ponto de entrada: [`pipeline_comparacao_mipsplatting.ipynb`](pipeline_comparacao_mipsplatting.ipynb).
Objeto de estudo: um leão de pedra filmado em vídeo (`data/leao_video/`). Etapas do notebook:

1. **Dados**: extrai os quadros mais nítidos do vídeo (variância do Laplaciano, reduz motion blur) e
   reconstrói câmeras + nuvem esparsa com COLMAP (`convert.py`).
2. **Mip-Splatting**: treina o método do paper (filtro 3D ligado, `--kernel_size 0.1`) em `-r 2`.
3. **Baselines**: treina **3DGS** puro e **3DGS+EWA** no mesmo COLMAP, mesmo split e mesmos
   hiperparâmetros, usando o repositório oficial do 3DGS (`baselines/gaussian-splatting/`).
4. **Avaliação multi-escala**: renderiza e mede (PSNR/SSIM/LPIPS-VGG, mesma implementação para os três)
   os três modelos em `-r 1, 2, 4, 8`. `-r 1` é zoom-in, `-r 8` é zoom-out extremo; `-r 2` é a escala de
   treino.
5. **Visualização**: crops lado a lado dos três modelos + ground truth em cada escala.

Cada etapa pula automaticamente se o artefato já existe (frames, COLMAP, checkpoints, renders,
métricas), então re-rodar o notebook do zero é seguro.

### Resultado

No held-out do leão (treino em `-r 2`, 30k iterações), reproduzimos a alegação central do paper no nosso
próprio objeto:

| modelo | PSNR -r8 (zoom-out) | PSNR -r2 (treino) | PSNR -r1 (zoom-in) | LPIPS -r8 |
|---|---|---|---|---|
| **Mip-Splatting** | **22.91** | 22.50 | **22.16** | **0.089** |
| 3DGS | 19.19 | 22.08 | 21.12 | 0.165 |
| 3DGS+EWA | 22.43 | 22.44 | 21.86 | 0.112 |

Na escala de treino os três empatam. Fora dela, o 3DGS puro desaba (-3.7 dB no zoom-out); o 3DGS+EWA
estabiliza mas borra (LPIPS pior); o Mip-Splatting é o mais nítido e estável em toda escala. Detalhes,
gotchas e o histórico completo do experimento (incluindo tentativas anteriores e por que foram
descartadas) estão em
[`wiki/analyses/leao-video-pipeline.md`](wiki/analyses/leao-video-pipeline.md).

O notebook é parametrizado (`VIDEO`, `SOURCE`, `N_FRAMES` na célula de configuração) e roda sobre
qualquer objeto capturado em vídeo do mesmo jeito: já foi aplicado também a um busto (`data/busto_video/`).

## Estrutura própria

Além do código herdado do 3DGS/Mip-Splatting (`train.py`, `render.py`, `scene/`,
`gaussian_renderer/`, `submodules/`, ...):
- [`pipeline_comparacao_mipsplatting.ipynb`](pipeline_comparacao_mipsplatting.ipynb): pipeline completo, ponto de entrada principal.

## Instalação

Dois ambientes conda: um para o Mip-Splatting (rasterizer com `kernel_size`) e outro para os baselines
3DGS (rasterizer com `antialiasing`). Os dois filtros exigem builds diferentes do rasterizer CUDA e não
convivem no mesmo pacote instalado — por isso ambientes separados. COLMAP precisa estar instalado e no
PATH (usado por `convert.py`).

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

### 2. 3DGS oficial 
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive baselines/gaussian-splatting

conda create -y -n gs3dgs --clone mip-splatting
conda activate gs3dgs
CUDA_HOME=$CONDA_PREFIX pip install \
    baselines/gaussian-splatting/submodules/diff-gaussian-rasterization \
    baselines/gaussian-splatting/submodules/simple-knn
```
Use o CUDA do próprio ambiente (`$CONDA_PREFIX`) para compilar o rasterizer, uma versão diferente da usada para compilar o torch quebra o build.

## Outros datasets (suportados sem alteração)

Além do nosso vídeo, o código aceita os formatos originais do paper:

```
# Blender / NeRF synthetic multi-escala
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale

# Mip-NeRF 360: baixe de https://jonbarron.info/mipnerf360/
```

## Comandos manuais (fora do notebook)

```
python train.py -s data/leao -m output/leao --eval -r 2 --kernel_size 0.1
python render.py -m output/leao --skip_train -r 2
python metrics.py -m output/leao
```

Depois de treinado, funde o filtro 3D nos parâmetros para visualizar no
[viewer online do Mip-Splatting](https://niujinshuchong.github.io/mip-splatting-demo):
```
python create_fused_ply.py -m output/leao --output_ply fused/leao_fused.ply
```

## Créditos

Código de treino/renderização/rasterização é do [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/)
original, que por sua vez é um fork do [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

### Citação
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