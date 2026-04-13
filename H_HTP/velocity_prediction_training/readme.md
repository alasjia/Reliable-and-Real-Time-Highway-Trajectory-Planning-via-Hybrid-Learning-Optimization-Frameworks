# velocity_predicition Directory Documentation

This directory implements highway vehicle trajectory prediction based on Graph Neural Networks (GNNs), with core processes including data preprocessing, model training, and evaluation. The main dependencies are PyTorch, PyTorch Geometric, and custom modules.

> Note: If you are not interested in the process of converting data in "dataset_after_dp" to ".pt" files, you can skip [HighD_datapre.py](HighD_datapre.py) and directly use [PyG_DataSet](PyG_DataSet).

---

## 1. Core Files Overview

### 1.1 HighD_datapre.py

**Function:**  
Preprocesses the original HighD dataset into graph-structured data suitable for GNNs and saves it as `.pt` files for efficient loading.

**Main Workflow:**
- Reads raw CSV files (vehicle and lane information).
- Sliding window augmentation: Generates multiple trajectory samples for each scenario by sliding along the time axis, greatly enriching the dataset.
- Coordinate normalization: The last observed point of the target vehicle is set as the origin to facilitate model learning.
- Feature extraction and missing value handling, unifying the feature format for vehicles and lane lines.
- Constructs PyTorch Geometric Data objects (node features, edges, labels, etc.).
- Merges all samples, splits them into training/validation/test sets using a fixed random seed, and generates corresponding DataLoaders.

**When to Use:**  
Run this script for the first time or when custom data preprocessing is needed. If `.pt` files already exist, you can skip this step.

**How to Run:**
```bash
python HighD_datapre.py
```

---

### 1.2 train_myway.py

**Function:**  
Main training script for the VectorNet trajectory prediction model using preprocessed data.

**Main Workflow:**
- Loads `.pt` data and constructs training/validation/test sets.
- Initializes the model (VectorNet, see vectornet.py), optimizer, learning rate scheduler, etc.
- Training loop:  
  - `train_epoch`: Iterates over the training set each epoch, performs forward pass, computes loss, backpropagation, and parameter updates.
  - `eval_epoch`: Evaluates model performance on the validation set without updating parameters.
- Logging and visualization: Records loss curves and supports high-quality plotting.
- Supports automatic GPU switching and flexible hyperparameter configuration.

**How to Run:**
```bash
python train_myway.py
```

---

### 1.3 test_for_de.py

**Function:**  
Evaluates the trained model, computes various trajectory prediction metrics, and visualizes prediction results.

**Main Workflow:**
- Loads trained model parameters.
- Loads test set data and performs batch inference.
- Computes metrics such as Average Displacement Error (ADE) and Final Displacement Error (FDE).
- Optional: Visualizes predicted and ground-truth trajectories for intuitive analysis.

**How to Run:**
```bash
python test_for_de.py
```

---

## 2. Other Important Modules

- **basic_module.py**  
  Defines basic neural network modules (e.g., MLPs) as building blocks for the main model.

- **vectornet.py**  
  Implements the main VectorNet architecture, including subgraph aggregation, global graph aggregation, and other core GNN logic.

- **global_graph.py, subgraph.py**  
  Implement global and subgraph aggregation operations for VectorNet.

- **config_nw.py**  
  Configuration file for managing data paths, model input/output dimensions, hyperparameters, and other global constants.

- **utils/**  
  Utility functions, including loss functions, visualization, data processing, and other helper tools.

- **PyG_DataSet/**  
  Stores preprocessed PyTorch Geometric dataset files, ready for training and evaluation.

- **results_vp/**  
  Stores model prediction results, visualization images, and other outputs.

The construction of VectorNet is reflected in velocity_predicition/basic_module.py, velocity_predicition/subgraph.py, velocity_predicition/global_graph.py, and velocity_predicition/vectornet.py. The implementation refers to xxx.github.com.

---

## 3. Recommended Usage Workflow

1. **Data Preprocessing**  
   If `.pt` files are not available, run HighD_datapre.py to generate the data.

2. **Model Training**  
   Configure parameters as needed and run train_myway.py to train the model.

3. **Model Evaluation**  
   After training, run test_for_de.py to evaluate model performance.

---

## 4. Notes

- All core processes support GPU acceleration.
- Data/model parameter paths and hyperparameters can be flexibly configured in config_nw.py or at the top of the main scripts.
- If you only need to reproduce the training/testing workflow, you can directly use the preprocessed data in PyG_DataSet/ without re-running preprocessing.