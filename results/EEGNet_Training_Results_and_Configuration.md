### Baseline Results (Supervised EEGNet)

Performance metrics calculated using Leave-One-Subject-Out (LOSO) cross-validation on the BCI Competition IV-2a dataset.

| Subject | Accuracy | Kappa |
| :---: | :---: | :---: |
| **1** | 64.58% | 0.5278 |
| **2** | 29.17% | 0.0556 |
| **3** | 69.79% | 0.5972 |
| **4** | 38.02% | 0.1736 |
| **5** | 33.16% | 0.1088 |
| **6** | 33.68% | 0.1157 |
| **7** | 42.88% | 0.2384 |
| **8** | 61.98% | 0.4931 |
| **9** | 66.15% | 0.5486 |
| **AVG** | **48.82%** | **0.3176** |

#### Experimental Setup
* **Epochs:** 300 (Patience: 100)
* **Batch Size:** 64
* **Optimizer:** Adam (LR: 0.001, Cosine Annealing)
* **Dropout:** 0.25
* **Preprocessing:** 4-38Hz Filter, 128Hz Resample, Exponential Moving Standardization
* **Segmentation:** 0.5s - 2.5s post-cue
