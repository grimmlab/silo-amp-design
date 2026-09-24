
# SILO for AMP: Self-Improvement Imitation Learning for Antimicrobial Peptide Design

This repository contains the official implementation and submission of **SILO for AMP** for the [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027). SILO (Self-Improvement Imitation Learning for Protein Optimization) is a generative protein sequence optimization framework adapted for de-novo antimicrobial peptide (AMP) design. The framework combines a transformer-based generative policy, incremental stochastic beam search, surrogate-guided activity evaluation, and self-imitation learning to explore the antimicrobial peptide sequence space.

For the AMP Challenge, SILO generates a library of 50,000 unique peptide candidates, evaluates their predicted antimicrobial activity against 11 bacterial pathogens using APEX, and applies physicochemical, synthesizability, novelty, and diversity filters to select a final set of 100 candidates.

The repository provides the code required to train the SILO policy, perform inference using a pretrained checkpoint, generate the candidate library, and verify the final submission.

The underlying method is described in:

> **Self-Improvement Imitation with Biologically Guided Search for Protein Design Under Oracle Budgets**
>
> [Read the paper on arXiv](https://arxiv.org/abs/2605.26690)

## 1. TL;DR: What the organizers should run

This section provides the main entry points for reproducing and verifying our AMP Challenge submission. The submitted inference checkpoint is used to generate the peptide library and select the final 100 candidates. The inference procedure does not perform additional model training or update the policy weights.

### 1.1. Clone the repository

```bash
git clone https://github.com/grimmlab/silo-amp-design.git
cd silo-amp-design
```

### 1.2. Environment and Installation

SILO uses `uv` for Python dependency management. The Python dependencies are defined in `pyproject.toml` and are installed automatically when running the inference pipeline. However, MMseqs2 is required for sequence similarity evaluation and **must be installed separately**.

### 1. Install MMseqs2
On Linux, MMseqs2 can be installed using Conda:

```bash
conda install -c conda-forge -c bioconda mmseqs2

```
### 2. Install Python dependencies

From the repository root:

```bash
uv sync
```
This automatically creates the project virtual environment and installs the dependencies specified in `pyproject.toml`.

### 1.3. Prepare the pretrained checkpoint

Place the supplied competition checkpoint in the designated checkpoint directory:

```text
inference_model/
└── best_model.pt
```

The checkpoint contains the pretrained SILO policy used for competition inference. The model weights are provided separately because of their file size. The checkpoint should be placed in the expected directory before running the inference pipeline.

### 1.4. Verify submission

The organizers can either run the following command for inference or run the uv run verify_submission.py as an additional entry point for checking the generated submission artifacts:

```bash
 uv run python verify_submission.py https://github.com/ashi198/silo-amp-design.git

```
or 

```bash
python -m SILO_amp.generate \
    --checkpoint ./inference_model \
    --output_dir ./generate \
    --seed 42 \
    --device cuda:0 \
    --total-peptide-count 50000 \
    --top-k 100
```
These commands loads the pretrained SILO policy, generates the peptide library, evaluates candidate sequences, and selects the final submission. The inference pipeline produces:

- A library of 50,000 unique peptide sequences.
- Predicted antimicrobial activity and physicochemical properties for the generated candidates.
- A final set of 100 selected peptide sequences.
- FASTA and CSV files containing the generated library and final submission under /generate folder.

**Important:** Run the command from the repository root to ensure that all relative paths to model assets, reference datasets, and configuration files resolve correctly.

---

## 2. Repository Structure

The repository is organized into separate directories for training, inference, generated artifacts, and submission verification.

```text
silo-amp-design/
│
├── data/
│
├── generate/
│
├── inference_model/
│   └── best_model.pt
│
├── SILO_amp/
│   │
│   ├── __init__.py
│   ├── generate.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── model/
│   ├── core/
│   │
│   ├── sequence_design.py
│   ├── sequence_dataset.py
│   ├── sequence_evaluator.py
│   │
│   ├── evaluation_metrics/
│   │
│   ├── apex/
│   ├── OmegAMP/
│   │
│   ├── pretrain/
│   │
│   ├── training/
│   │   └── training_data/
│   │
│   └── data/
│       ├── antibacterial.fasta
│       ├── marlys.fasta
│       └── training.fasta
│
├── pyproject.toml
├── README.md
└── verify_submission.py
```

### 2.1. Directory descriptions

| Directory or file | Description |
|---|---|
| `data/` | Antibacterial.fasta, training.fasta and MarLys.fasta files for novelty and diversity checks |
| `generate/` | Output directory for generated peptide libraries and final submission artifacts. |
| `inference_model/` | Directory containing the finetuned SILO policy checkpoint used for inference. |
| `SILO_amp/` | Main Python module containing the SILO implementation for antimicrobial peptide design. |
| `pyproject.toml` | Python project configuration and package dependency information. |
| `README.md` | Documentation for installation, inference, training, and submission verification. |
| `verify_submission.py` | Entry point for verifying the competition submission. |

### 2.2. SILO_amp package

The `SILO_amp` directory contains the implementation of the generative policy, stochastic beam search, sequence evaluation, and self-imitation learning.

| Module or directory | Description |
|---|---|
| `generate.py` | Inference entry point for generating and selecting competition candidates. |
| `main.py` | Main training and fine-tuning entry point for SILO. |
| `config.py` | Configuration of model architecture, generation parameters, data paths, evaluation, and optimization. |
| `model/` | Transformer-based generative policy and associated model components. |
| `core/` | Sequence generation and incremental stochastic beam-search implementation. |
| `sequence_design.py` | Sequence representation and action-space definition used by the generative policy. |
| `sequence_dataset.py` | Dataset utilities for policy training and sequence representation. |
| `sequence_evaluator.py` | Candidate evaluation and selection using antimicrobial activity predictions and sequence constraints. |
| `evaluation_metrics/` | Computational metrics for antimicrobial activity, sequence novelty, diversity, and physicochemical properties. |
| `apex/` | APEX pathogen MIC prediction models and associated evaluation utilities. |
| `OmegAMP/` | OmegAMP implementation and model assets for AMP likelihood prediction. |
| `pretrain/` | Data preparation and supervised pretraining utilities. |
| `training/` | SILO training utilities and associated training datasets. |
| `data/` | Reference FASTA files used for candidate filtering and sequence novelty checks. |

*** Module execution**

The SILO implementation is organized as a Python module. All main entry points should therefore be executed from the repository root using Python's module syntax.

For example:

```bash
python -m SILO_amp.generate
```

rather than executing the source file directly.

---

## 3. Environment and Installation

### 3.1. System requirements

The inference pipeline requires a Python environment with PyTorch and the scientific computing packages used by SILO, APEX, and OmegAMP.

The original development configuration used:

- Python 3.10
- PyTorch 2.8.0+cu128
- NVIDIA A40 GPU
- CUDA-enabled inference

A CUDA-compatible NVIDIA GPU is recommended for generating and evaluating the full 50,000-sequence library.

The APEX pathogen prediction ensemble and transformer-based generative policy are the main model components used during inference.

### 3.2. Create a Python environment

Create and activate a dedicated Conda environment:

```bash
conda create -n silo-amp python=3.10 -y

conda activate silo-amp
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### 3.3. Install the SILO package

From the repository root, install the project and its dependencies:

```bash
python -m pip install -e .
```

This installs the project in editable mode using the configuration defined in `pyproject.toml`. Editable installation allows the `SILO_amp` package to be imported directly while preserving the repository structure. Ensure that MMseqs2 are installed and accessible in the environment.

---

## 4. Method Overview

SILO is an iterative optimization framework that improves a generative protein design policy by learning from its own high-performing solutions.

The framework consists of four main components.

### 4.1. Generative policy

SILO uses a transformer-based autoregressive decoder as its generative policy. The policy is first pretrained on a curated collection of known antimicrobial peptide sequences. During sequence optimization, the policy generates new peptide candidates through a sequence of actions in the peptide design space.

### 4.2. Incremental stochastic beam search

SILO employs incremental stochastic beam search to generate diverse candidate sequences. Unlike deterministic beam search, stochastic beam search introduces randomness into candidate generation, allowing the policy to explore multiple promising regions of the peptide sequence space. The search procedure produces candidate generation trajectories that can subsequently be evaluated and used for policy improvement.

### 4.3. Activity-guided evaluation

Generated candidates are evaluated using APEX, a machine learning-based antimicrobial activity predictor. APEX predicts minimum inhibitory concentrations (MICs) against 11 bacterial pathogens. These predictions are used to identify promising candidates and guide the optimization process toward peptide sequences with favorable predicted antimicrobial activity. Additional physicochemical, sequence validity, and diversity criteria are used to constrain the generated sequences.

### 4.4. Self-imitation learning

Rather than learning an explicit value function or applying policy-gradient optimization, SILO improves its policy by learning from its own best-performing generation trajectories. High-performing trajectories are selected based on oracle evaluations and used to fine-tune the transformer policy through next-action cross-entropy learning. **Importantly, SILO learns in the action space rather than directly optimizing sequences as independent objects.** The policy learns to increase the likelihood of the actions that generated promising candidates, allowing the knowledge gained from previous search iterations to guide subsequent generation.

The overall optimization process alternates between:

1. Generating diverse candidate sequences.
2. Evaluating candidates using the activity prediction model.
3. Selecting high-performing generation trajectories.
4. Updating the policy through next-action imitation learning.

The updated policy is then used to generate candidates in the next optimization iteration.

---

## 5. Training Data and Pretraining

Before iterative optimization, the SILO generative policy is pretrained on a curated collection of known antimicrobial peptide sequences.

### 5.1. Pretraining datasets

The pretraining dataset was assembled from three publicly available data sources.

| Dataset | Description |
|---|---|
| AMPDiffusion | A collection of 19,670 unique AMP sequences derived from DRAMP 3.0, APD3, and DBAASP. |
| GRAMPA | AMP sequences obtained from the GRAMPA dataset (`mic_data.csv`), as distributed through the HydrAMP repository. |
| Known AMP positives | Additional known AMP sequences from `unlabelled_positive.csv`, comprising positive AMP examples from dbAMP, DRAMP, and AMP Scanner. |

Dataset references:

- [AMPDiffusion training data](https://github.com/szczurek-lab/ampdiffusion-starter-kit/tree/main/data/training.fasta)
- [HydrAMP repository and GRAMPA data](https://github.com/szczurek-lab/hydramp-starter-kit/tree/main/data)

After merging, deduplication, and length filtering, the resulting pretraining dataset contained **21,881 unique antimicrobial peptide sequences.** The curated dataset was used to pretrain the transformer-based generative policy before SILO optimization.

### 5.2. Pretraining implementation

The pretraining utilities are located in:

```text
SILO_amp/pretrain/
```

The main pretraining implementation is:

```text
SILO_amp/pretrain/pretrain.py
```

Pretraining initializes the generative policy on known AMP sequences before subsequent activity-guided optimization. Pretraining is not required for competition inference when the supplied SILO checkpoint is used.

---

## 6. Scoring

### 6.1. Antimicrobial activity prediction as objective to optimize

Each candidate in the generated library is evaluated using the APEX pathogen prediction model. APEX estimates the minimum inhibitory concentration (MIC) of each peptide against 11 bacterial pathogens. The predicted MIC values are used to calculate activity metrics for candidate ranking and final selection. Lower predicted MIC values indicate stronger predicted antimicrobial activity.

The evaluated activity profiles include:

- Overall predicted antimicrobial activity across the pathogen panel.
- Predicted activity against Gram-positive pathogens.
- Predicted activity against Gram-negative pathogens.
- Predicted activity against the multidrug-resistant (MDR) pathogen subset.

These activity profiles are used to construct the different candidate selection pools.

### 6.2. AMP likelihood prediction

Candidate sequences are additionally evaluated using OmegAMP. OmegAMP provides machine learning-based AMP likelihood scores, which are recorded as complementary indicators of predicted antimicrobial peptide characteristics. The OmegAMP implementation and associated model assets are located in:

```text
SILO_amp/OmegAMP/

```

### 6.3. Initial sequence filtering

Generated sequences for 50K library are filtered to satisfy the following requirements:

- Only the 20 standard proteinogenic amino acids (`ACDEFGHIKLMNPQRSTVWY`) are permitted.
- Peptide length must be between 8 and 50 residues.
- Duplicate sequences are removed.
- Sequences identical to any reference sequence in `antibacterial.fasta`, `marlys.fasta`, or `training.fasta` are excluded.

The reference files are located in:

```text
SILO_amp/data/antibacterial.fasta
SILO_amp/data/marlys.fasta
SILO_amp/data/training.fasta
```

This initial filtering step prevents exact duplication of sequences already present in the designated reference datasets.


### 6.4. Physicochemical property calculation

The physicochemical properties of generated peptides are computed using the modlamp library and additional sequence-based calculations.

The evaluated properties include:

| Property | Description |
|---|---|
| Net charge | Calculated peptide charge. |
| Hydrophobicity | Peptide hydrophobicity calculated using the Eisenberg scale. |
| Hydrophobic moment | Measure of the spatial distribution of hydrophobic residues, calculated using the Eisenberg scale. |
| Isoelectric point | Estimated pH at which the peptide has zero net charge. |
| Cysteine count | Number of cysteine residues in the peptide. |
| Proline content | Fraction of proline residues in the peptide sequence. |
| Maximum hydrophobic run | Length of the longest consecutive stretch of hydrophobic residues. |
| Consecutive glycine content | Presence of consecutive glycine residues within the sequence. |

These properties are used to filter candidates before final sequence selection.

---


## 7. Final Candidate Filtering and Selection

Following generation and computational evaluation, the candidate library undergoes additional physicochemical, synthesizability-related, novelty, and diversity filtering. The remaining candidates are ranked and selected according to their predicted antimicrobial activity profiles.

### 7.1. Physicochemical and synthesizability-related filtering

Only candidates satisfying all predefined physicochemical and sequence composition constraints are retained.

The filtering criteria include:

| Property | Selection criterion |
|---|---|
| Net charge | Between +2.0 and +10.0. |
| Hydrophobicity | Between -0.5 and 0.8. |
| Hydrophobic moment | Between 0.3 and 0.6. |
| Cysteine content | Must satisfy the configured maximum cysteine count. |
| Hydrophobic runs | Must not exceed the configured maximum consecutive hydrophobic stretch. |
| Proline content | No more than 20% of the total sequence length. |
| Consecutive glycines | No more than two consecutive glycine residues. |
| MarLys sequence identity | No local alignment with greater than 80% identity covering at least 80% of the candidate sequence. |

These computational filters are intended to prioritize candidates with acceptable physicochemical properties and reduce potentially undesirable sequence characteristics.

### 7.2. Activity-based candidate categories

The remaining candidates are organized into five selection categories to capture different predicted antimicrobial activity profiles.

| Category | Selection criteria |
|---|---|
| Gram-positive selective | Gram-positive selectivity score below 0.8 and predicted MIC90 against the Gram-positive pathogen subset below 64 µM. |
| Gram-negative selective | Gram-negative selectivity score below 0.5 and predicted MIC90 against the Gram-negative pathogen subset below 64 µM. |
| Broad-spectrum | Similar predicted selectivity against Gram-positive and Gram-negative bacteria, with an overall predicted MIC90 below 64 µM. |
| Multidrug-resistant (MDR) pathogens | Predicted MIC90 against the MDR pathogen subset below 64 µM. |
| Overall high-activity candidates | Overall predicted MIC90 below 64 µM. |

For the broad-spectrum category, comparable predicted activity is defined using the ratio of Gram-positive and Gram-negative MIC50 values:

```text
min(GP_MIC50, GN_MIC50) / max(GP_MIC50, GN_MIC50) >= 0.9
```

Within each category, eligible candidates are ranked according to the corresponding predicted activity metric, prioritizing lower predicted MIC values.

### 7.3. Greedy selection and diversity constraints

Final candidates are selected greedily from the ranked category-specific pools.

Before a candidate is included in the submission, it must pass additional sequence novelty and diversity checks.

**Novelty against known antimicrobial peptides**

Candidate sequences are compared against the `antibacterial.fasta` reference database using Levenshtein sequence similarity. Candidates with greater than 80% Levenshtein similarity to a reference sequence are excluded.

**Diversity within the selected library**

Each candidate is also compared against all previously selected candidates, including those selected from other activity categories. A candidate is rejected if its local sequence similarity to any previously selected sequence strictly exceeds 0.60. This procedure reduces redundancy within the final library while retaining candidates with distinct predicted antimicrobial activity profiles.

### 7.4. Final submission

The greedy selection procedure continues until the requested number of valid and sufficiently diverse candidates has been obtained.

For the AMP Challenge submission, the final output consists of **100 selected antimicrobial peptide sequences.** The final library is designed to represent multiple predicted activity profiles while satisfying the predefined sequence validity, novelty, physicochemical, and diversity constraints.

---

## 8. Output Files

A successful inference run writes the generated library, computed evaluation metrics, and final submission artifacts to the specified output directory.
With the default output path:

```text
generate/
```

the expected output files are:

| File | Description |
|---|---|
| `generated_50k_peptides_library.fasta` | Intermediate generated peptide library before metric-based selection. |
| `library_50k.fasta` | Validated 50,000-sequence library in FASTA format. |
| `library_50k.csv` | Generated library with computed antimicrobial activity scores, physicochemical properties, and associated metadata. |
| `top_100.fasta` | Final 100 peptide sequences selected for the AMP Challenge submission. |
| `top_100.csv` | Final selected sequences with their predicted activity scores, selection categories, and associated evaluation metrics. |
| `manifest.json` | Submission metadata, artifact counts, and SHA-256 checksums for the designated submission files. |
| `diagnostic_error.json` | Diagnostic information written if the declared library-size or final-selection requirements cannot be satisfied. |

### 8.1. Generated library

The generated library files contain the candidate sequences produced by the pretrained policy after the initial sequence validation procedure.
The corresponding CSV file includes the computational metrics used during candidate evaluation and selection.

### 8.2. Final submission

The `top_100.fasta` file contains the final peptide sequences selected for the competition. The corresponding `top_100.csv` file provides the evaluation results and associated metadata for these candidates.

### 8.3. Artifact validation

The inference pipeline is configured to detect failures to satisfy the requested library size or final candidate count.
If the required number of valid candidates cannot be obtained, the runtime reports the failure rather than silently returning an incomplete submission.
Diagnostic information may be written to `diagnostic_error.json` to support troubleshooting.

---

## 9. Training and Fine-Tuning

The full SILO finetuning implementation is provided for reproducibility and further research.

The main training entry point is:

```text
SILO_amp/main.py
```

Training can be launched from the repository root using:

```bash
python -m SILO_amp.main \
    --seed 42 \
    --epoches 100 \
    --device cuda:0 \
    --results ./results/experiment \
    --comments experiment
```


## 10. Citation and Acknowledgements

If you use SILO for AMP, please cite the associated SILO paper:
**Self-Improvement Imitation with Biologically Guided Search for Protein Design Under Oracle Budgets**
[https://arxiv.org/abs/2605.26690](https://arxiv.org/abs/2605.26690)

The implementation builds on or incorporates ideas, methods, and software from the following projects:
- [SILO] (https://github.com/grimmlab/SILO): The original SILO framework for self-improvement imitation learning and protein sequence optimization.
- [Gumbeldore](https://github.com/grimmlab/gumbeldore): Initial self improvement learning framework.
- [Stochastic Beam Search](https://github.com/wouterkool/stochastic-beam-search): Stochastic beam-search methodology and reference implementation.
- [APEX Pathogen] (https://gitlab.com/machine-biology-group-public/apex-pathogen): Antimicrobial activity prediction model used to estimate pathogen-specific MIC values and guide SILO optimization.
- [OmegAMP](https://openreview.net/forum?id=hAq3XLZ9ex): AMP classification and likelihood scoring for generated peptide sequences.
- [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027): Competition resources and submission specifications.

---

## 15. Contact

For questions regarding the implementation, inference pipeline, or reproducibility of the AMP Challenge submission, please open an issue in this repository.