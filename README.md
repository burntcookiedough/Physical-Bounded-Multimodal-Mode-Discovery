# Physical-Bounded Multimodal Mode Discovery

An unsupervised machine learning system for discovering hidden operational modes in complex industrial equipment. Designed for high-reliability systems (e.g., turbofans, bearings, and motors), this architecture correlates disparate multimodal sensor data while applying strict, explicit physical constraints (thermodynamic and electromechanical) to ensure discovered operational regimes and fault modes are physically feasible.

---

## ⚡ Quick Start (Zero-Friction Execution)

Want to see it work immediately without manual setup? Clone the repository and run the automated execution script for your platform. The script will automatically create an isolated environment, install dependencies, fetch the required CWRU sensor dataset, and run the pipeline test.

**1. Clone the repository:**
```bash
git clone https://github.com/burntcookiedough/physical-bounded-mode-discovery.git
cd physical-bounded-mode-discovery
```

**2. Run the automated script:**

*   **Windows:**
    Double-click `setup_and_run.bat` from Windows Explorer, or run in terminal:
    ```cmd
    setup_and_run.bat
    ```

*   **Linux / macOS:**
    ```bash
    chmod +x setup_and_run.sh
    ./setup_and_run.sh
    ```
    
The pipeline will execute, parse the data, evaluate the models, and output the final validation reports to the `results/` folder.

---

## Key Features

*   **Hierarchical Consensus Arbitration:** Resolves sensor disagreement during state transitions without early-fusion feature smearing. Conflicting states trigger flags that reveal transitional failure onsets.
*   **Physics Feasibility Filter:** Validates data-driven density clusters against known conservation laws (Joule heating, ISO vibration, operational margins), directly removing mathematically spurious clusters.
*   **Physics-Informed Auto Labeling (PIAML):** Auto-labels new geometric modes using severity metrics derived from pre-defined boundary conditions.
*   **Generative Physics-Bounded Augmentation (PFBSFA):** In-domain interpolative generation of fault-state data strictly bounded by the physical boundaries of the operation limits. 
*   **Cross-Domain Alignment & Causality Tracking:** Advanced structural fingerprinting for unsupervised mode sequence tracking and predictive temporal causality analysis across differing datasets.

## Manual Setup & Reproducibility 

If you prefer to manually control the environment or wish to test the NASA CMAPSS dataset sequentially, follow these instructions:

### Prerequisites

*   Python 3.9 or higher

**1. Create and activate a Virtual Environment:**

*   **Windows:**
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```
*   **Linux / macOS:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

### Data Placement & Preparation

The automatic script already downloads the `CWRU` database. If you would like to run the second validation experiment on `NASA CMAPSS`:

Download the CMAPSS dataset manually from NASA's prognostic data repository, extract the archive, and place the `train_FD00X.txt`, `test_FD00X.txt`, and `RUL_FD00X.txt` text files inside the `data/cmapss/` directory.

```text
data/
├── cwru/             # Excludes via execution script pulling CWRU .mat files 
└── cmapss/           # Contains NASA CMAPSS FD001.txt - FD004.txt and RUL files
```

### Usage

Run the complete unsupervised evaluation pipeline covering multimodal baseline extraction, HDBSCAN clustering, Physics Arbitration, and Generative Feature Synthesis from the root directory:

```bash
# Evaluate on CWRU Dataset (Electromechanical constraints)
python -m src.pipeline --demo cwru

# Evaluate on NASA CMAPSS Dataset (Thermodynamic constraints)
python -m src.pipeline --demo cmapss

# Evaluate Both Demos sequentially
python -m src.pipeline --demo both
```

All outputs and evaluation reports will be parsed into the `results/` folder.

## Repository Structure

```text
├── configs/                # Constraint bounds (e.g., cmapss_params.json, cwru_params.json)
├── data/                   # Raw sensor datasets (excluded via .gitignore)
├── results/                # Output evaluation summaries and cluster performance metrics
├── src/                    # Source Code
│   ├── dataset_adapter.py      # Parsers for physical datasets (CWRU, CMAPSS)
│   ├── feature_extractor.py    # Time-domain and frequency-domain feature processing
│   ├── modality_clusterer.py   # Individual Sub-system HDBSCAN processing
│   ├── consensus_arbiter.py    # Modality voting and conflict detection architecture
│   ├── physics_filter.py       # Constraint boundary checking, scoring, and PIAML labeling
│   ├── whitespace_features.py  # Generative fault modeling and Cross-Domain DCFP Alignment
│   ├── baseline_pipeline.py    # Standard K-Means flat models for comparative ablation
│   ├── imdti.py                # Inter-Mode Degradation Trajectory Index analysis 
│   ├── download_cwru.py        # Automated CWRU dataset fetcher
│   ├── conflict_pattern_library.py # Cross-modal violation pattern tracking
│   ├── visualization.py        # Heatmaps and t-SNE generation outputs
│   └── pipeline.py             # Evaluation Orchestrator
├── tests/                  # Unit and integration tests for physical boundary components
├── requirements.txt        # PIP dependencies
├── setup_and_run.bat       # 1-click Execution Script (Windows)
├── setup_and_run.sh        # 1-click Execution Script (Linux/macOS)
└── README.md
```

## Results and Observations

*   **NASA CMAPSS Validation:** Successfully uncovers $k=7$ physically viable regimes out of the multi-sensor temporal progression, cleanly rejecting mathematical noise modes. Highlights extensive cross-modal conflict rates (near 99%) emphasizing the need for grouped multimodal arbitration over generic single-space clustering.
*   **CWRU Electromechanical Validation:** Defines $k=8$ coherent operational loads vs baseline $k=4$, maintaining roughly an 80% constraint coherence verification rate under strict bounding constraints (ISO 10816, Joule Heating Limits).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
