# GrainLegumes-PINO: Physics-Informed Neural Operators for Porous Media Flow  
### *Specialization Project (VP1) – MSE Data Science, Autumn 2025*

**Master of Science in Engineering – Major Data Science**  
**Eastern Switzerland University of Applied Sciences (OST)**  
**Author:** Rino M. Albertin  
**Supervisor:** Prof. Dr. Christoph Würsch  

---

## 📌 Project Overview

This specialization project investigates the performance and applicability of **Physics-Informed Neural Operators (PINOs)** for simulating air flow in **porous granular media** such as agricultural grain beds.  

Permeability fields κ(x) are **generated in MATLAB**, solved via **COMSOL Multiphysics**, and exported as structured **PyTorch datasets (.pt)**.  
The goal is to train a 2-D PINO model that learns the mapping from permeability to pressure and velocity fields — effectively replacing expensive CFD solvers for design and optimization tasks.

The repository provides a complete, modular workflow covering:
- 🧩 **Data generation** (MATLAB → COMSOL → PyTorch conversion)  
- 📊 **Exploratory Data Analysis (EDA)** Spectral analysis of permeability κ, pressure p, and velocity U  
- ⚙️ **PINO training and evaluation** using the `neuraloperator` library  
- 📈 **Visualization and diagnostics** for convergence, residuals, and spectral errors  
- 🧱 **Reproducible setup** via Docker and VS Code Dev Container  

---

## 🧭 Data Flow Overview

```mermaid
graph LR
A[MATLAB – Permeability field generator] --> B[COMSOL – Brinkman flow solver]
B --> C[PyTorch dataset (.pt)]
C --> D[EDA – Spectral analysis]
D --> E[PINO training – Physics-informed neural operator]
E --> F[Evaluation – Residuals & spectral error maps]
F --> G[Model checkpoints & reproducible results]
```

---

## ⚙️ Local Execution

<details>
<summary><strong>Option A – Run in Visual Studio Code with Docker Dev Container (recommended)</strong></summary>

**Requirements**
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Visual Studio Code](https://code.visualstudio.com/)
- VS Code extension **“Dev Containers”**

**Steps**
```bash
git clone https://github.com/Rinovative/grainlegumes-pino.git
cd grainlegumes-pino
```
1. Open the folder in VS Code  
2. Reopen in Container (via prompt or `F1 → Dev Containers: Reopen in Container`)  
3. Launch `PINO_Project_Rino_Albertin_GrainLegumes.ipynb` and run all cells  

</details>

<details>
<summary><strong>Option B – Run via Docker CLI (without VS Code)</strong></summary>

```bash
git clone https://github.com/Rinovative/grainlegumes-pino.git
cd grainlegumes-pino

docker build -t pino-dev .
docker run -it --rm -p 8888:8888 -v $(pwd):/app pino-dev
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

Then open the URL shown in the terminal.

</details>

---

## 📂 Repository Structure
<details>
<summary><strong>Show project tree</strong></summary>

```bash
.
├── .devcontainer/                                      # VS Code Dev Container configuration
│   └── devcontainer.json                               # Container setup and environment definition
│
├── data/                                               # Final trained modelss and batch training datasets
│   ├── processed/                                      # Final trained models
│   └── raw/                                            # COMSOL output and metadata for batch before preprocessing
│       ├── samples_uniform_var10_N1000/                # Example batch of simulation cases
│       │   ├── cases/                                  # Individual case files with (κ, p, U)
│       │   └── meta.pt                                 # Batch generation parameters
│       └── ...                                         
│
├── data_generation/                                    # MATLAB → COMSOL → PyTorch data creation pipeline
│   ├── comsol/                                         # COMSOL model templates for automated simulation
│   │   ├── template_brinkman.mph                       # Base Brinkman model file
│   │   ├── template_brinkman_cluster.mph               # Cluster version
│   │   └── template_brinkman_tensor.mph                # Tensor variant for permeability field
│   │
│   ├── data/                                           # Generated datasets
│   │   ├── meta/                                       # Metadata describing batch
│   │   │   ├── samples_uniform_var10_N1000.csv         # Generation parameters for cases of batch
│   │   │   ├── samples_uniform_var10_N1000.json        # Metadata for batch generation
│   │   │   └── ...                                     
│   │   │
│   │   ├── processed/                                  # COMSOL outputs
│   │   │   ├── samples_uniform_var10_N1000/            # Processed dataset directory
│   │   │   │   ├── case_0001_sol.csv                   # Example processed field solution
│   │   │   │   └── ...                                 
│   │   │   └── ...                                     
│   │   │
│   │   └── raw/                                        # MATLAB permability-field
│   │       ├── samples_uniform_var10_N1000/            # Individual batch
│   │       │   ├── case_0001.csv                       # Raw permeability field data
│   │       │   ├── case_0001.json                      # Associated metadata for this case
│   │       │   └── ...                                 
│   │       └── ...                                     
│   │
│   └── matlab/                                         # MATLAB scripts for permeability generation and COMSOL coupling
│       ├── functions/                                  # Modularized MATLAB functions
│       │   ├── core/                                   # Core utilities for data generation and visualization
│       │   │   ├── gen_permeability.m                  # Generates synthetic permeability fields κ(x)
│       │   │   ├── run_comsol_case.m                   # Executes a single COMSOL simulation case
│       │    │   ├── sample_parameters.m                 # Creates randomized parameter sets for DoE
│       │   │   └── visualize_case.m                    # Visualization helper for MATLAB/COMSOL outputs
│       │   │
│       │   └── test/                                   # MATLAB test routines for validation
│       │       ├── test_generate_permeability_fields.m # Test for permeability generation
│       │       ├── test_run_comsol_case.m              # Test for COMSOL automation routine
│       │       └── test_visualize_case.m               # Test for visualization and output integrity
│       │
│       ├── batch_run.m                                 # Batch execution for full dataset generation
│       ├── build_batch_dataset.py                      # Python converter for merging raw COMSOL outputs into .pt
│       ├── merge_batch_cases.py                        # Combines multiple cases into unified datasets
│       ├── permeability_field_viewer.mlx               # MATLAB Live Script for permeability-field inspection
│       └── singel_run.m                                # Single test run for debugging and prototyping
│   
├── docs/                                               # Project documentation, plots, and figures
│
├── model_training/                                     # Core training and analysis environment
│   ├── data/                                           # Training datasets and model checkpoints
│   │   ├── meta/                                       # 
│   │   ├── processed/                                  # 
│   │   └── raw/                                        # Merged datasets used as input
│   │       ├── samples_uniform_var10_N1000/            # Example batch
│   │       │   ├── meta.pt                             # Batch generation parameters
│   │       │   └── samples_uniform_var10_N1000.pt      # Main training tensor data
│   │       └── ...                                     
│   │
│   ├── notebooks/                                      # Interactive notebooks for analysis and visualization
│   │   └── EDA.ipynb                                   # Exploratory Data Analysis for PINO input fields
│   │
│   ├── src/                                            
│   │   ├── eda/                                        # Spectral and statistical analysis utilities
│   │   │   ├── __init__.py                             
│   │   │   └── eda_spectral_analysis.py                # Main EDA routines for PSD and field spectra
│   │   │
│   │   ├── model/                                      # 
│   │   │   ├── __init__.py                             
│   │   │   └── XXX.py                                  #
│   │   │
│   │   └── util/                                       # Shared helper functions
│   │       ├── __init__.py                             
│   │       ├── util_data.py                            # Data loading and preprocessing routines
│   │       └── util_nb.py                              # Notebook utilities (visualization, widgets)
│   │
│   └── train_pino.py                                   # Main training entry script for PINO
│
├── .dockerignore                                       # Docker build exclusion list
├── .gitignore                                          # Git exclusion list
├── Dockerfile                                          # Docker image setup for reproducible environment
├── environment.yml                                     # Conda/Mamba environment specification
├── pyproject.toml                                      # Poetry configuration for dependencies
└── README.md                                           # Project overview and documentation
```
</details>

---

## 🧠 Methodology

1. **Data Generation (MATLAB + COMSOL)**  
   Random κ fields are generated in MATLAB and solved for p and U in COMSOL (Brinkman flow).  
2. **Data Preparation (Python)**  
   Case files and metadata are merged into structured `.pt` datasets.  
3. **Exploratory Data Analysis (EDA)**  
   Statistical and spectral inspection of fields using Matplotlib and ipywidgets.  
4. **Model Training (PINO)**  
   Train a Fourier-based Physics-Informed Neural Operator to learn the mapping κ → (p, U).  
5. **Evaluation and Diagnostics**  
   Visualize residual loss, convergence curves, and spectral error maps.

---

## 📊 Visualizations

---

## 📄 License

This project is released under the []().

---

## 📚 Reference

```bibtex
@article{kossaifi2025librarylearningneuraloperators,
   author    = {Jean Kossaifi and Nikola Kovachki and Zongyi Li and David Pitt and 
                 Miguel Liu-Schiaffini and Valentin Duruisseaux and Robert Joseph George and 
                 Boris Bonev and Kamyar Azizzadenesheli and Julius Berner and Anima Anandkumar},
   title     = {A Library for Learning Neural Operators},
   journal   = {arXiv preprint arXiv:2412.10354},
   year      = {2025}
}
```