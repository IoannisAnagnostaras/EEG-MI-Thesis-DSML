 BCI SSL Experiment Reproduction

This repository contains the code and experimental results developed for the Master's thesis of **Ioannis Anagnostaras**, as part of the **Data Science and Machine Learning** Master's program at the **National Technical University of Athens**.

This package contains the exact code and run artifacts used to answer:
- Benchmark policy
- Datasets used
- Preprocessing
- New architecture and SSL setup

## Folder Map

The repository is organized into the following structure to separate logic from outputs:

- `code/`: Contains all Python code. This includes the core pipeline modules (config, datasets, preprocessing, augmentations, models, objectives, training) as well as the reproducibility scripts.
- `results/`: Contains the run configurations and summary outputs from the final comparisons. Results are organized into specific subfolders (`deep_supervised/`, `bnci_only/`, `bnci_plus_stieger/`).
- `graphs/`: Contains all evaluation plots, subject deltas, and ablation visualizations.

*(Note: Local test runs and exploratory visualizations are kept out of the main version control to maintain a clean production state).*

## Main Reproducibility Scripts

To reproduce the experiments, navigate to the root directory and run the scripts located in the `code/` folder:

- `python code/run_repro_deep_supervised_sweep.py`
- `python code/run_repro_ssl_comparison.py`
- `python code/build_locked_reference_table.py`

##  Core Reference Outputs

The absolute reference files for the final leaderboard and performance metrics can be found here:

- **Official Comparison:** - `results/official_comparison_repro_locked_latest.csv`
  - `results/official_comparison_repro_locked_latest.json`
- **Deep Supervised Baseline:** - `results/deep_supervised/deep_supervised_leaderboard.csv`
  - `results/deep_supervised/best_config_summary.json`

## Environment & System Specifications

- Please refer to `requirements.txt` for the core dependencies and `python_version.txt` for the exact Python version used (Python 3.11.15).
- The hardware specifications of the production server used for the final runs (NVIDIA RTX 5090, 64GB RAM) are fully documented in `system_info.txt`.
