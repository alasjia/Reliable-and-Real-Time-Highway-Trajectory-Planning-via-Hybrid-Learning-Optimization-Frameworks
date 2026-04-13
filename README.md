# Reliable and Real-Time Highway Trajectory Planning via Hybrid Learning-Optimization Frameworks

This project implements and evaluates a hybrid trajectory planning framework for autonomous vehicles on highways. The core of the system is a Graph Neural Network (GNN), specifically VectorNet, trained on the HighD dataset to predict the future trajectories of surrounding vehicles. This prediction is then utilized by a downstream trajectory planning module. The repository also includes an end-to-end (E2E) replanning evaluation pipeline.

## Key Components

*   **GNN-based Trajectory Prediction:** A VectorNet model that processes vectorized scene context (vehicle trajectories, lane lines) to predict future vehicle velocities.
*   **Hybrid Trajectory Planning:** A planning module that integrates the GNN predictions to generate safe and efficient trajectories for the ego vehicle.
*   **Data Processing Pipeline:** Scripts to process the raw HighD dataset into a graph-based format suitable for PyTorch Geometric.
*   **Evaluation and Visualization:** Tools for quantitative evaluation (ADE/FDE metrics, replanning success rate) and qualitative visualization (BEV plots, dynamic GIFs).

## Repository Structure

```
.
├── H_HTP/
│   ├── data_processing/              # Scripts to process raw HighD data into PyG datasets.
│   ├── velocity_prediction_training/   # Core module for training and evaluating the VectorNet prediction model.
│   │   ├── PyG_DataSet/                # Pre-processed .pt datasets.
│   │   ├── train_myway.py              # Main training script.
│   │   └── test_for_de.py              # Evaluation script (ADE/FDE).
│   └── trajectory_planning/            # Hybrid planning implementation and visualization.
│       └── utils/
│           ├── res_exec_shots.py       # Generates multi-shot BEV plots of planning results.
│           └── res_exec_dynamically.py # Generates dynamic GIFs of planning execution.
├── VectorNet_HighD/
│   └── replanning/                   # End-to-end (E2E) replanning evaluation.
│       ├── e2e_replanning_batch.py     # Batch evaluation of E2E replanning success rate.
│       └── paper_pics_generation/      # Scripts to generate figures for publication.
└── readme.md                         # This file.
```

## Workflow

### 1. Data Preparation

The project uses the HighD dataset. The raw data needs to be converted into graph-structured `.pt` files for the GNN model.

*   **Option A (Recommended): Use Pre-processed Data**
    The pre-processed dataset is available in `H_HTP/velocity_prediction_training/PyG_DataSet/`. You can skip the manual processing step.

*   **Option B: Process Data Manually**
    If you need to customize the data processing, navigate to `H_HTP/velocity_prediction_training/` and run the `HighD_datapre.py` script. For more details, refer to the README in `H_HTP/data_processing/`.
    ```bash
    cd H_HTP/velocity_prediction_training/
    python HighD_datapre.py
    ```

### 2. Trajectory Prediction Model

The core of the prediction module is in `H_HTP/velocity_prediction_training/`. For a detailed explanation, please see its dedicated README.

*   **Training:** To train the VectorNet model, run `train_myway.py`. Hyperparameters and paths can be configured in `config_nw.py`.
    ```bash
    cd H_HTP/velocity_prediction_training/
    python train_myway.py
    ```
*   **Evaluation:** After training, a model checkpoint (`.pth` file) will be saved. Evaluate its performance (ADE/FDE) using `test_for_de.py`.
    ```bash
    python test_for_de.py
    ```

### 3. Trajectory Planning and Evaluation

This project explores two approaches for planning and evaluation:

*   **Hybrid Planning Visualization:** The results of the hybrid planner can be visualized using scripts in `H_HTP/trajectory_planning/utils/`. These scripts generate static multi-shot BEV plots (`res_exec_shots.py`) and dynamic GIFs (`res_exec_dynamically.py`) of specific planning scenarios.

*   **E2E Replanning Evaluation:** An end-to-end batch evaluation can be run to measure the replanning success rate over many cases. This uses the script in `VectorNet_HighD/replanning/`.
    ```bash
    cd VectorNet_HighD/replanning/
    python e2e_replanning_batch.py --ckpt /path/to/your/model.pth
    ```

## Reproducing Paper Figures

The project contains code to reproduce figures and results from the associated publication.

*   **Specific Scenarios:** The following scenarios were highlighted for demonstration:
    *   **Scene A:** `rec 54`, `ep 699`, `t = 3.0s -> 8.5s`
    *   **Scene B:** `rec 53`, `ep 1189`, `t = 1.5s -> 7.0s`
    *   **Scene C:** `rec 53`, `ep 1091`, `t = 4.5s -> 10.0s`

*   **Figure Generation:**
    *   For **Hybrid Planning** results, see `H_HTP/trajectory_planning/utils/res_exec_shots.py` and `draw_trajectory_replanning` in `H_HTP/trajectory_planning/utils/res_exec_dynamically.py`.
    *   For **E2E Planning** results, a refined script for generating publication-quality figures is available at `VectorNet_HighD/replanning/paper_pics_generation/trajectory_multi_BEV.py`.

## Dependencies

The main dependencies for this project are:
*   Python 3.x
*   PyTorch
*   PyTorch Geometric (`torch-geometric`)
*   NumPy, Pandas, Matplotlib, tqdm

Ensure these are installed in your Python environment before running the scripts.
