This repository implements Structure-Guided, **Physics-Aware Reconstruction (SPARC)** as a practical framework for extending the functional operating range of conventional multiphoton microscopes under scan-limited conditions.

**Requirements**

**Hardware**

A CUDA-capable GPU (recommended but not required)

≥8 GB RAM (≥16 GB recommended for large TIFFs)


**Software**

**Windows 10/11**

Anaconda or Miniconda (Python ≥ 3.8)

Optional: NVIDIA drivers + CUDA toolkit (for GPU acceleration)


**Installation (Windows + Anaconda)**

Follow these steps to set up the environment on Windows using Anaconda:

1\. Create a new conda environment

conda create -n SPARC python=3.9

conda activate SPARC

2\. Install PyTorch with CUDA support (or CPU-only)

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3\. Install dependencies

pip install numpy=1.23.5 scipy pillow tifffile visdom cv2

4\. (Optional) Start Visdom server for training monitoring

conda activate periodic\_denoise

python -m visdom.server -port 8097

**Datasets**
A set of simulated calcium imaging data and the pre-trained model are available for download (14 Hz sampling rate; SNR = 5 dB, 10.5281/zenodo.19673550).

For model training:

raw_iso.tif (128 × 512 × 1000) should be placed in the ./train/Input directory,

and raw_iso_label.tif (512 × 512) should be placed in the ./train/Reference directory.

For model inference:

The pre-trained weight file E30_loss167.0068.pth should be placed in the ./pth/model directory,

and the raw anisotropically sampled data raw_iso.tif should be placed in the ./test directory.

**Important Notes**

1\.  Raw anisotropically sampling data (e.g., raw\_iso.tif, 128×512×1000) are placed in the ./train/Input directory.

Structural reference data (e.g., raw\_iso\_label.tif, 512×512) are placed in the ./train/Reference directory.

Test data are placed in the ./test.

2\. Visdom Requirement: Training will fail if Visdom server isn’t running (unless disabled in code).

3\. Reproducibility: Use --seed to fix random states for data sampling and training.

