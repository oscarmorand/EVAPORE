<div align="center">

# EVAPORE

[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)](https://docs.python.org/3.12/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=lightning&logoColor=white)](https://lightning.ai/pytorch-lightning)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![lightning-hydra-template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/nathanpainchaud/lightning-hydra-template)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

</div>

# Description
 
Accurate segmentation of retinal blood vessels in 2D fundus images is essential for reconstructing complete vascular trees and enabling downstream analyses that rely on a reliable vascular topology. However, most deep learning approaches rely on pixel-wise segmentation, which can lead to vessel discontinuities, particularly in thin or low-contrast regions — fragmentation errors that affect the structural integrity of the reconstructed network and impact subsequent modeling tasks.
 
Tracking-based methods and automated post-processing approaches have been proposed to reconnect fragmented segments, but they typically require knowing beforehand which points along the vascular skeleton should be reconnected — a problem that in practice remains difficult and often requires manual intervention.
 
**EVAPORE** addresses this limitation with an automated framework that identifies vessel point pairs that should be reconnected. Rather than relying on handcrafted geometric criteria, it formulates reconnection as a **path classification problem**: given a candidate path between two vessel points in a binary mask, the model determines whether a direct vessel segment exists between them. By encoding complete paths as single entities, the approach leverages image information along the entire potential connection.
 
Experimental results on the [FIVES dataset](#fives-dataset) show that this approach effectively identifies disconnected segments in predicted vascular masks, substantially outperforming geometric baselines and providing reliable guidance for downstream reconnection methods.


## Table of Contents
 
- [Description](#description)
- [Publications](#publications)
  - [ShapeMI — MICCAI 2026 Workshop](#shapemi--miccai-2026-workshop)
- [Project Structure](#project-structure)
- [FIVES dataset](#fives-dataset)
- [Installation](#installation)
  - [Using uv](#using-uv)
  - [Environment variables](#environment-variables)
- [License](#license)
- [Contact](#contact)

# Publications

## ShapeMI — MICCAI 2026 Workshop

**Paper:** *[Publication link — to be added]*

This repository contains the code associated with our paper presented at the **ShapeMI Workshop at MICCAI 2026**.

The version of the project corresponding to the experiments and results reported in the paper is available on the `ShapeMI` branch.

After completing the installation described in the [Installation](#installation) section, the experiments can be reproduced by following the notebooks located in the `notebooks/` directory.

The notebooks are intended to be run sequentially:

```text
notebooks/
├── 01_prepare_data.ipynb
├── 02_pretrain_unet.ipynb
├── 03_generate_segmentation_preds.ipynb
├── 04_path_train_data_generation.ipynb
├── 05_train_path_classification_model.ipynb
└── 06_test_path_classification_model.ipynb
```

# Project Structure
 
<!-- TODO: adjust to match the actual repository layout -->
```text
EVAPORE/
├── notebooks/                   # Notebooks reproducing the ShapeMI experiments
├── images/                      # Images included in the notebooks
├── checkpoints/                 # Pretrained model checkpoints (tracked with git lfs)
├── data/
│   └── FIVES/
│       └── clean_files_idx.txt  # Idx of the files we used in the train/val set         
├── src/                         # Main source code
│   ├── graph/
│   ├── image_segmentation/
│   ├── path_neural_networks/
│   └── utils/
├── pyproject.toml
└── README.md
```

# FIVES dataset

The experiments use the **FIVES (A Fundus Image Dataset for AI-based Vessel Segmentation)** dataset.

- **Dataset:** [FIVES on Figshare](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1)
- **Direct download:** [Download FIVES](https://figshare.com/ndownloader/files/34969398)

# Installation

## Using uv

1. Install [uv](https://github.com/astral-sh/uv) if it is not already available on your system.

2. Download the repository.
    ```bash
    git clone https://github.com/oscarmorand/EVAPORE
    cd EVAPORE
    ```

3. From the repository root, create the environment and install the project dependencies:

    Create a virtual environment and install the project and its dependencies. You must specify as an extra the desired compute platform for PyTorch (i.e. CPU/CUDA). Supported values are: `cpu`, `cu128`, `cu126`, `cu118`.
    ```bash
    # e.g. to install the project with the PyTorch version built for CPU
    uv sync --extra cpu

    # e.g. to install the project with the PyTorch version built for CUDA 12.6
    uv sync --extra cu126
    ```

4. Activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```

## Environment variables

EVAPORE uses environment variables to specify the locations of the dataset and model checkpoints.

### EVAPORE_DATA_DIR
the directory where the data is located (ex: /home/user/EVAPORE/data).

### EVAPORE_CHECKPOINTS_DIR
the directory where the checkpoints are located (ex: /home/user/EVAPORE/checkpoints)

We recommend using the `checkpoints\` folder of this repository as several pretrained models checkpoints are already provided in this folder. To get these checkpoints with git lfs, please follow this procedure to import them locally:

```bash
git lfs install
git lfs pull
```

# License
 
This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 — see the [LICENSE](LICENSE) file for details.
 
# Contact
 
Oscar Morand (oscar.morand@epita.fr), [GitHub](https://github.com/oscarmorand), [LRE (Laboratoire de Recherche de l'EPITA)](https://www.lre.epita.fr/), [CREATIS](https://www.creatis.insa-lyon.fr/site/en)