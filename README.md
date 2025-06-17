# Cosmic Ray Classification, Energy Inference and Angular Reconstruction (CONDOR)

This repository contains code to process simulated cosmic ray shower data (from CORSIKA) and train a deep learning model for **multi-task learning**:  
- **Classifying the primary particle** (photon vs proton),  
- **Inferring the energy class** (binary), and  
- **Reconstructing the zenith angle** of the cosmic ray.

---

## 🚀 Key Features (Updated)

- **Raw Sequence-Based Input**: Each input sequence corresponds to a unique particle shower and preserves its temporal-spatial structure. All sequences are padded with zeros based on the longest sequence, ensuring consistency without mixing data from different events.

- **Feature Normalization**:
  - `x`, `y` spatial coordinates are scaled with `MinMaxScaler`.
  - `t` (arrival time) is scaled with `StandardScaler` to preserve physical significance.

- **Updated Model Architecture**:
  - Replaces previous Conv1D+LSTM hybrid with a **deep Conv1D stack** of 5 layers followed by **Multi-Head Attention**.
  - Three output branches: particle classification, energy classification, and angle regression.
  - Each branch includes skip connections, dense layers, normalization, and dropout for robustness.

- **Unified Modeling & Visualization**:
  - All model training, prediction, evaluation, and plotting are handled in a single notebook: `CNN_Transformer.ipynb`.

---

## 📁 Repository Structure

### `CONDOR_Binary_Translation/`

- **`ReadBinary.py`**:
  - Parses CORSIKA `.DAT` binary simulation files using the `panama` library.
  - Filters particles within the CONDOR detector area.
  - Adds metadata (e.g., energy, angle, shower area).
  - Outputs preprocessed `.csv` files for each shower (both binned and unbinned).
  
### `CNN_Transformer.ipynb`

- **Main notebook for the full modeling pipeline**:
  - Reads the processed CSVs.
  - Constructs padded sequences.
  - Applies normalization.
  - Builds and trains a multi-output neural network model.
  - Evaluates and visualizes model performance directly within the notebook.

### `datapreprocessing.py`

- A helper script for testing preprocessing methods on **individual showers**.  
  *Note: Not part of the main training pipeline.*

---

## 🧠 Model Architecture Summary

```
Input (padded variable-length sequence with 3 features: x_bin, y_bin, t_bin)
│
├── Masking (0-padding ignored)
├── Conv1D (×5 with increasing filters: 32 → 512)
├── Shared feature encoding
│
├── Particle Classification Head:    Attention → Dense → Sigmoid
├── Energy Classification Head:      Attention → Dense → Sigmoid
└── Angle Regression Head:           Attention → Dense → Linear
```

- Each output branch includes:
  - `LayerNormalization`
  - `GlobalAveragePooling1D`
  - `Dropout`
  - `BatchNormalization`

- Losses used:
  - `BinaryCrossentropy` for both classification outputs.
  - `MeanSquaredError (MSE)` for regression output.

---

## 🔁 Workflow Summary

1. **Data Generation**:
   - Place `.DAT` CORSIKA simulations in the specified input directory.

2. **Binary Translation (`ReadBinary.py`)**:
   - Convert binary files to `.csv` with spatial, temporal and metadata fields.

3. **Modeling & Visualization (`CNN_Transformer.ipynb`)**:
   - Load CSV data.
   - Build padded, normalized input sequences.
   - Train the multi-output model.
   - Evaluate and visualize predictions (F1-score, MAE, confusion matrices, angle reconstruction error, etc.).

---

## ⚙️ Requirements

- Python 3.x  
- Required libraries:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `panama`

You can install them with:

```bash
pip install -r requirements.txt
```

---

## 📌 Usage

1. Clone this repo.
2. Run `ReadBinary.py` to generate structured CSVs from CORSIKA `.DAT` files.
3. Open and run `CNN_Transformer.ipynb` for:
   - Data preprocessing,
   - Model training,
   - Performance evaluation and visualization.

---

## 🤝 Contributions

Contributions are welcome!  
Feel free to open issues or submit pull requests with improvements, bug fixes, or suggestions.

---

## 📬 Contact

For questions, collaboration opportunities, or academic use cases, feel free to contact:

**Luis Navarro**  
📧 luis.navarrof@usm.cl
