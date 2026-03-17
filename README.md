# 1BM110 Data-driven Artificial Intelligence

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/78/Eindhoven_University_of_Technology_logo_new.png?20231008195526" alt="TU/e Logo" width="180" />
</p>

<p align="center">
  Coursework repository for <strong>1BM110 - Data-driven Artificial Intelligence</strong><br/>
  Eindhoven University of Technology
</p>

---

## Overview

This repository contains the working materials, notebooks, datasets, and deliverables for two course assignments:

- **Assignment 1** focuses on energy generation prediction using time-series and deep learning workflows.
- **Assignment 2** focuses on reinforcement learning for the **bounded knapsack problem** using Gymnasium and Stable-Baselines.

The structure is intentionally assignment-based so that each folder keeps its own notebook, inputs, and deliverables together.

---

## Repository Structure

```text
1BM110-dataDrivenAI/
├── README.md
├── .gitignore
├── assignment 1/
│   ├── main.ipynb
│   ├── energy_generation_train.csv
│   ├── energy_generation_test.csv
│   ├── deliverables/
│   │   ├── Assignment1_Group02.ipynb
│   │   └── Assignment1_Group02.html
│   └── part4/
│       └── lstm_random_search/
├── assignment 2/
│   ├── main.ipynb
│   ├── knapsack_env.py
│   └── instructions.pdf
└── d
```

---

## File Guide

### Root

| File / Folder | Purpose |
| --- | --- |
| `README.md` | Project-level documentation and navigation guide. |
| `.gitignore` | Ignore rules for macOS artifacts, Python caches, notebook checkpoints, environments, and build outputs. |
| `assignment 1/` | Materials related to the first assignment on energy prediction. |
| `assignment 2/` | Materials related to the second assignment on reinforcement learning and knapsack optimization. |
| `d` | Miscellaneous local file currently present in the repository root. |

### Assignment 1

| File / Folder | Purpose |
| --- | --- |
| `assignment 1/main.ipynb` | Main working notebook for Assignment 1. Contains the analysis and implementation pipeline. |
| `assignment 1/energy_generation_train.csv` | Training dataset for the energy prediction task. |
| `assignment 1/energy_generation_test.csv` | Test dataset used for evaluation or final prediction steps. |
| `assignment 1/deliverables/Assignment1_Group02.ipynb` | Final notebook deliverable submitted for Assignment 1. |
| `assignment 1/deliverables/Assignment1_Group02.html` | Exported HTML version of the final Assignment 1 notebook. |
| `assignment 1/part4/lstm_random_search/` | Hyperparameter search artifacts for the LSTM tuning workflow. This folder stores tuner outputs and checkpoint files generated during experimentation. |

### Assignment 2

| File / Folder | Purpose |
| --- | --- |
| `assignment 2/main.ipynb` | Main notebook for Assignment 2. Structured into three parts matching the assignment brief. |
| `assignment 2/knapsack_env.py` | Custom Gymnasium environment implementation for the unbounded and bounded knapsack problems. |
| `assignment 2/instructions.pdf` | Official assignment brief describing the problem, required experiments, and submission expectations. |

---

## Assignment Breakdown

### Assignment 1: Energy Prediction

The first assignment is organized around a notebook-based machine learning workflow. Based on the repository contents, it includes:

- data loading and preprocessing,
- exploratory analysis,
- deep learning experiments,
- LSTM hyperparameter tuning,
- and final deliverables exported as notebook and HTML.

### Assignment 2: Reinforcement Learning for the Bounded Knapsack Problem

The second assignment is centered on reinforcement learning in a custom environment. The notebook is organized into three parts:

1. **Environment / MDP description**
   Explain the bounded knapsack problem as a Markov Decision Process.
2. **Training DQN and PPO**
   Train, evaluate, and compare reinforcement learning agents.
3. **Invalid action masking**
   Train a masked policy variant and compare it against standard PPO.

The environment logic is implemented in `knapsack_env.py`, while the official requirements remain in `instructions.pdf`.

---

## Working Style in This Repository

- **Notebooks are the primary interface** for analysis, experiments, and reporting.
- **Raw data and assignment inputs stay close to the notebook that uses them.**
- **Deliverables are preserved separately** from working notebooks where applicable.
- **Generated tuning artifacts** are stored alongside the related assignment rather than in the repository root.

---

## Suggested Setup

This repository does not currently include a pinned environment file such as `requirements.txt` or `environment.yml`, so dependencies are inferred from the notebooks and source files.

Likely packages used in this project include:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `keras`
- `keras-tuner`
- `gymnasium`
- `stable-baselines3`
- `sb3-contrib`

If needed, create a virtual environment and install the required packages manually for the assignment you want to run.

---

## Quick Start

### Run Assignment 1

Open:

- `assignment 1/main.ipynb`

Use it together with:

- `assignment 1/energy_generation_train.csv`
- `assignment 1/energy_generation_test.csv`

### Run Assignment 2

Open:

- `assignment 2/main.ipynb`

Use it together with:

- `assignment 2/knapsack_env.py`
- `assignment 2/instructions.pdf`

---

## Notes

- The repository currently mixes final deliverables and working files, which is practical for coursework and traceability.
- Some experiment artifacts in Assignment 1 are generated files rather than hand-edited source files.
- If this repository is later reused beyond coursework, adding a `requirements.txt` and a short reproducibility section would be the next useful cleanup.

