# BabyCry

**Generalization, Dataset Shift, and Acoustic Representations in Infant Cry Classification**

## Overview

BabyCry is a research project investigating the generalization and reproducibility of machine learning models for infant cry classification.

Although machine learning models have been developed to classify infant cries according to their presumed causes, their performance may depend on the recording session, infant identity, and dataset characteristics.

This study systematically evaluates these factors using multiple validation strategies, cross-dataset transfer experiments, acoustic feature analyses, and pretrained audio representations.

The primary objectives are to:

1. Evaluate how classification performance changes across different data-splitting strategies.
2. Examine whether cry-reason classification models generalize across independent datasets.
3. Investigate whether dataset identity can be predicted from acoustic features.
4. Identify acoustic feature groups associated with dataset-specific differences.
5. Assess whether the observed generalization patterns persist when using pretrained wav2vec2 representations.

This repository provides the Python scripts used for the analyses and figures presented in the accompanying manuscript.

## Datasets

The analyses use infant cry recordings from the following datasets:

| Dataset | Description |
|---------|-------------|
| EnesBabyCries1 | Infant cry recordings with parent-reported cry reasons and associated annotations |
| Donate-a-Cry | Publicly contributed infant cry recordings with cry-reason labels |
| Dunstan-related recordings | Infant vocalizations associated with Dunstan-style vocalization categories |

The datasets differ in recording conditions, annotation procedures, and label structures.

These differences are considered when interpreting within-dataset generalization and cross-dataset transfer performance.

### Data availability

Original audio recordings are **not included** in this repository.

Users must obtain the datasets independently and comply with the relevant access conditions, licenses, and ethical requirements.

The exact data acquisition procedures and expected input directory structure should be verified before attempting full reproduction.

## Repository Structure

```text
BabyCry/
├── src/
│   ├── 00_dataset/
│   │   └── Dataset preparation and preprocessing
│   │
│   ├── 01_figure1/
│   │   └── Dataset overview and workflow
│   │
│   ├── 02_figure2/
│   │   └── Generalization hierarchy and infant identity
│   │
│   ├── 03_figure3/
│   │   └── Cross-dataset cry-reason classification
│   │
│   ├── 04_figure4/
│   │   └── Dataset identity and acoustic feature families
│   │
│   ├── 05_figure5/
│   │   └── Group-aware acoustic feature analysis
│   │
│   ├── 06_figure6/
│   │   └── wav2vec2 representation analysis
│   │
│   ├── 90_supplementary/
│   │   └── Supplementary analyses and tables
│   │
│   └── 99_utils/
│       └── Utility scripts
│
├── data/
│   └── Input and processed data (not distributed)
│
├── results/
│   └── Analysis outputs (generated locally)
│
├── figures/
│   └── Generated figures
│
├── run_final.sh
├── make_figures.sh
├── requirements.txt
└── README.md
```

## Analysis Overview

The main analyses are organized according to the manuscript figures.

### Figure 1. Dataset Overview and Analysis Workflow

Script:

```bash
python src/01_figure1/01_make_figure1.py
```

This figure summarizes the datasets and the overall analytical workflow.

### Figure 2. Generalization Hierarchy

Scripts:

```bash
python src/02_figure2/01_generalization_hierarchy.py
python src/02_figure2/02_infant_identity.py
python src/02_figure2/03_make_figure2.py
```

Cry-reason classification performance is evaluated using different data-splitting strategies:

- Random clip-level splitting
- Session-level splitting
- Infant-level splitting
- Leave-one-infant-out (LOIO) evaluation

Infant identity classification is also investigated to assess the extent to which acoustic features contain infant-specific information.

### Figure 3. Cross-Dataset Transfer

Scripts:

```bash
python src/03_figure3/01_cross_dataset_reason.py
python src/03_figure3/02_cluster_inference.py
python src/03_figure3/03_make_figure3.py
```

Cry-reason classification models are evaluated across independent datasets.

Cluster-aware statistical inference is used to account for dependencies among recordings originating from the same infant or recording group.

Bootstrap confidence intervals and permutation tests are used to assess the uncertainty of cross-dataset classification performance.

### Figure 4. Dataset Identity Classification

Final analysis scripts:

```bash
python src/04_figure4/01_dataset_identity_groupcv.py --repeats 200
python src/04_figure4/02_feature_family_groupcv_all.py --repeats 200
python src/04_figure4/03_reason_vs_dataset_qc.py
python src/04_figure4/04_make_figure4_qc.py
```

This analysis investigates whether acoustic characteristics can distinguish recordings originating from different datasets.

Dataset identity classification is evaluated using group-aware cross-validation.

The analysis also examines the contribution of different acoustic feature families.

The final evaluation uses 200 repeated group-aware assessments.

### Figure 5. Acoustic Feature Importance and Distributions

Final analysis scripts:

```bash
python src/05_figure5/01_feature_importance_group.py
python src/05_figure5/02_feature_distributions_matched.py
python src/05_figure5/03_make_figure5_group.py
```

This analysis examines acoustic features associated with dataset-specific differences.

Feature importance and feature distributions are evaluated using group-aware and reason-matched procedures.

The analysis is intended to characterize acoustic differences between datasets rather than establish their underlying recording or physiological causes.

### Figure 6. Representation Robustness Using wav2vec2

Final analysis scripts:

```bash
python src/06_figure6/05_extract_wav2vec2_single.py --dataset both
python src/06_figure6/single/02_reason_generalization.py
python src/06_figure6/single/04_dataset_identity_groupcv.py
python src/06_figure6/04_make_figure6_qc.py
```

This analysis evaluates whether the observed generalization and dataset identity patterns persist when using pretrained wav2vec2 audio representations.

Embeddings are extracted from individual audio recordings to avoid batch-padding-dependent variation.

Cry-reason generalization and dataset identity classification are subsequently evaluated using the extracted embeddings.

## Supplementary Analyses

### Supplementary Figure S1. Harmonization Sensitivity

Scripts:

```bash
python src/90_supplementary/S01_harmonization/01_run_sensitivity_qc.py
python src/90_supplementary/S01_harmonization/02_make_figureS1_qc.py
```

This analysis evaluates the sensitivity of cross-dataset classification results to alternative acoustic feature harmonization conditions.

### Supplementary Tables

The supplementary tables are generated using:

```bash
python src/90_supplementary/tables/01_make_tableS1.py
python src/90_supplementary/tables/03_make_tableS3.py
python src/90_supplementary/tables/04_make_tableS4_qc.py
```

The final workflow generates Supplementary Tables S1, S3, and S4.

## Software Requirements

The analyses were developed in Python.

The main software dependencies include:

| Package | Version |
|---------|---------|
| NumPy | 2.4.6 |
| pandas | 3.0.5 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.10.9 |
| librosa | 0.11.0 |
| soundfile | 0.14.0 |
| PyTorch | 2.5.1 |
| torchaudio | 2.5.1 |
| Transformers | 5.16.1 |
| tqdm | 4.70.0 |
| joblib | 1.5.3 |

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Depending on the operating system and GPU configuration, a platform-specific installation of PyTorch may be necessary.

Pretrained model weights used for wav2vec2 embedding extraction must also be available.

## Reproducing the Analyses

### Step 1. Clone the Repository

```bash
git clone https://github.com/BioRegDS/BabyCry.git
cd BabyCry
```

### Step 2. Install Dependencies

Create and activate an appropriate Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 3. Prepare the Datasets

Obtain the original audio datasets and prepare them using the scripts located in:

```text
src/00_dataset/
```

The preprocessing workflow includes dataset preparation, label processing, audio quality control, and acoustic feature extraction.

**Important:** The final analysis script assumes that the required processed datasets and acoustic features have already been generated.

Raw audio files and processed research datasets are not provided in this repository.

### Step 4. Run the Final Analyses

From the repository root, execute:

```bash
bash run_final.sh
```

This script runs the analyses used for the final manuscript results, including:

- Generalization hierarchy
- Infant identity classification
- Cross-dataset transfer
- Cluster-aware statistical inference
- Repeated group-aware dataset identity classification
- Acoustic feature-family analysis
- Group-aware feature importance
- Reason-matched feature distributions
- Single-audio wav2vec2 embedding extraction
- wav2vec2-based generalization analysis
- Supplementary harmonization sensitivity analysis

Some analyses are computationally intensive, particularly repeated cross-validation and wav2vec2 embedding extraction.

### Step 5. Generate Figures and Tables

After the analyses have completed, execute:

```bash
bash make_figures.sh
```

This script generates the final manuscript figures and supplementary tables from the analysis outputs.

The generated files are written to the corresponding figure and result directories.

## Evaluation and Statistical Considerations

### Data Leakage Prevention

Infant cry recordings are not necessarily independent observations.

Multiple recordings may originate from the same infant or recording session.

Accordingly, group-aware splitting strategies are used to evaluate generalization beyond individual recordings.

### Cross-Dataset Generalization

Within-dataset classification performance and cross-dataset transfer performance are evaluated separately.

This distinction is important because successful classification within one dataset does not necessarily imply that the learned relationships generalize to independently collected recordings.

### Repeated Evaluation

Repeated group-aware cross-validation is used for dataset identity analyses.

Results are summarized using the mean and standard deviation across repetitions.

These standard deviations should not be interpreted as 95% confidence intervals.

### Cluster-Aware Inference

For cross-dataset cry-reason classification, cluster-aware bootstrap and permutation procedures are used to quantify statistical uncertainty.

Confidence intervals and hypothesis tests are interpreted with consideration of the relevant clustering structure.

### Interpretation of Dataset Differences

High dataset identity classification performance indicates that dataset-specific information is present in the evaluated acoustic representations.

However, these analyses alone do not establish whether the differences arise from recording devices, acoustic environments, infant characteristics, annotation procedures, or other factors.

## Reproducibility Notes

The repository contains both final analysis scripts and some earlier scripts retained for methodological traceability or as dependencies of the final workflow.

For reproduction of the manuscript results, use:

```bash
bash run_final.sh
bash make_figures.sh
```

The figure-specific `run_all.sh` scripts should not be assumed to reproduce the final manuscript analyses.

The final workflow prioritizes group-aware evaluation, corrected statistical inference, and single-audio wav2vec2 embedding extraction.

## Data and Privacy

Infant cry recordings and associated infant-level information require appropriate handling.

This repository does not distribute original audio recordings or participant-level research data.

Any reuse of the datasets must comply with the original providers' conditions and applicable ethical and privacy requirements.

## Citation

If you use this repository, please cite the associated manuscript once its bibliographic information becomes available.

## Authors

BioRegDS  
Faculty of Medicine, Tottori University, Japan