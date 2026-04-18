# Physical-Bounded Multimodal Mode Discovery

An unsupervised machine learning system for discovering hidden operational modes in complex industrial equipment. The project is designed for high-reliability systems such as turbofans, bearings, and motors, and combines multimodal sensor analysis with explicit physical constraints so that discovered regimes and fault modes remain physically plausible.

## Overview

The pipeline correlates multiple sensor modalities, clusters latent operating states, and then filters or relabels those states using thermodynamic or electromechanical feasibility rules. This helps distinguish meaningful operational modes from mathematically convenient but physically invalid clusters.

Core ideas in the repository include:

- Hierarchical consensus arbitration across modalities instead of relying on early feature fusion alone.
- Physics feasibility filtering to reject clusters that violate known operating limits.
- Physics-informed auto-labeling of discovered modes.
- Constraint-bounded synthetic fault generation for low-density failure regions.
- Cross-domain alignment utilities for comparing learned modes across datasets.

## Quick Start

If you want to run the project with the least manual setup, use the platform-specific bootstrap script. The script creates a virtual environment, installs Python dependencies, downloads the CWRU dataset, and launches the CWRU demo pipeline.

### 1. Clone the repository

```bash
git clone https://github.com/burntcookiedough/Physical-Bounded-Multimodal-Mode-Discovery.git
cd Physical-Bounded-Multimodal-Mode-Discovery
```

### 2. Run the automated setup script

Windows:

```cmd
setup_and_run.bat
```

Linux / macOS:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

After the script finishes, generated outputs are written to the `results/` directory.

## Manual Setup

Use the manual flow if you want more control over the environment or if you plan to run the NASA CMAPSS experiment in addition to CWRU.

### Prerequisites

- Python 3.9 or higher

### 1. Create and activate a virtual environment

Windows:

```cmd
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare datasets

The bootstrap scripts already download the CWRU dataset automatically by running `src/download_cwru.py`.

If you also want to run the NASA CMAPSS experiment, download the dataset manually from NASA's prognostics data repository and place the extracted files in `data/cmapss/`.

Expected local layout:

```text
data/
+-- cwru/      # downloaded automatically by the setup script
\-- cmapss/    # add NASA CMAPSS train/test/RUL text files here
```

Recommended CMAPSS files include:

- `train_FD001.txt` through `train_FD004.txt`
- `test_FD001.txt` through `test_FD004.txt`
- `RUL_FD001.txt` through `RUL_FD004.txt`

## Running the Pipeline

From the repository root, run one of the following:

```bash
# CWRU demo (electromechanical constraints)
python -m src.pipeline --demo cwru

# NASA CMAPSS demo (thermodynamic constraints)
python -m src.pipeline --demo cmapss

# Run both demos sequentially
python -m src.pipeline --demo both
```

Optional arguments:

- `--data-dir` to point to a custom dataset folder
- `--config-dir` to use alternate constraint files
- `--results-dir` to write outputs somewhere other than `results/`

## Outputs

Pipeline runs create artifacts under `results/`, including text summaries and auxiliary files such as conflict pattern logs. The evaluation utilities under `evaluation_framework/` can also generate plots inside `evaluation_framework/output_plots/`.

## Repository Structure

```text
.
+-- configs/                     # Physical constraint definitions for each demo
+-- evaluation_framework/        # Supplementary evaluation and visualization modules
|   +-- main.py
|   +-- physics_constraints.py
|   +-- multimodal_consensus.py
|   \-- output_plots/           # Generated evaluation plots
+-- src/                         # Main end-to-end pipeline implementation
|   +-- baseline_pipeline.py
|   +-- conflict_pattern_library.py
|   +-- consensus_arbiter.py
|   +-- dataset_adapter.py
|   +-- download_cwru.py
|   +-- feature_extractor.py
|   +-- imdti.py
|   +-- modality_clusterer.py
|   +-- physics_filter.py
|   +-- pipeline.py
|   +-- visualization.py
|   \-- whitespace_features.py
+-- requirements.txt             # Python dependencies
+-- setup_and_run.bat            # Automated Windows setup + execution
+-- setup_and_run.sh             # Automated Linux/macOS setup + execution
+-- Project.txt                  # Project notes / reference material
\-- README.md
```

Generated local directories such as `data/`, `results/`, and `venv/` are created during setup or execution and are not expected to be committed.

## Key Features

- Hierarchical consensus arbitration to detect disagreement between modalities during state transitions.
- Physics feasibility filtering using known operating constraints such as vibration, current, temperature, and turbofan state relationships.
- Physics-Informed Auto Labeling (PIAML) for assigning semantic meaning to newly discovered clusters.
- Physics-bounded synthetic augmentation for sparse fault-state regions.
- Cross-domain mode alignment utilities for comparing analogous behaviors between datasets.

## Reported Observations

- NASA CMAPSS validation can uncover physically viable operational regimes while rejecting noise-like clusters that do not satisfy the imposed constraints.
- CWRU validation separates coherent operating loads and fault-related behavior under strict electromechanical boundary checks.
- Cross-modal conflict tracking helps expose transitions and disagreement patterns that single-space clustering can miss.
