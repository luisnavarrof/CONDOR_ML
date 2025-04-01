# Cosmic Ray Classification and Angular Reconstruction (CONDOR)

This repository contains code to process simulation data of cosmic ray showers (photons and protons) and train machine learning models for both classifying the type of cosmic ray and reconstructing the incidence angle of these particles.

## Repository Structure

The repository is organized into the following main folders:

* **CONDOR Angular Reconstruction:** Contains the codes to train and evaluate models for angular reconstruction and cosmic ray classification.
* **CONDOR Binary Translation, Data Processing and Visualization:** Contains codes to process simulation data in DAT format, preprocess data, and visualize results.

## Folder Contents

### CONDOR Angular Reconstruction

* **model.py / model.ipynb:**
    * This code implements a hybrid neural network model, combining 1D convolutional layers (Conv1D) and Long Short-Term Memory (LSTM) networks, to classify cosmic ray particles (photon or proton) and reconstruct their incidence angle.
    * **Main functionality:**
        * Loads preprocessed data from CSV files, which include information about the position, time, energy, and type of particles.
        * Normalizes the data using `MinMaxScaler`.
        * Defines and compiles a hybrid model that uses Conv1D layers to extract spatial patterns and LSTM layers to model temporal dependencies. The model has two outputs: one for particle type classification (using a sigmoid activation function) and the other for angle regression (using a linear activation function).
        * Trains the model using the training data and validates its performance with a validation set.
        * Saves the trained model and prediction statistics to CSV files.
        * Generates plots to visualize the model's results, including angular reconstruction and the confusion matrix for particle classification.
    * **Dependencies:** TensorFlow, pandas, numpy, scikit-learn, matplotlib.

### CONDOR Binary Translation, Data Processing and Visualization

* **ReadBinary.py:**
    * This code is responsible for reading simulation data files in DAT format, which contain information about cosmic ray showers.
    * **Main functionality:**
        * Reads DAT files using the `panama` library.
        * Filters relevant particles (protons, electrons, muons, pions, kaons, etc.) based on their particle ID (pdgid).
        * Extracts and normalizes the coordinates (x, y), time (t), and energy (E) of the particles.
        * Filters particles that fall within the CONDOR detector area.
        * Calculates additional metadata, such as the percentage of particles within the detector area and the shower area.
        * Adds metadata to the data, including particle ID, incidence energy, incidence angle, and simulation number.
        * Saves the processed data to CSV files, both in binned format (grouping particles into coordinate and time bins) and unbinned.
        * The data is organized by particle type (photon or proton), incidence angle, and simulation number, facilitating subsequent analysis and modeling.
    * **Dependencies:** panama, numpy, pandas, pathlib.
* **datapreprocessing.py:** Code to visualize aspects of cosmic ray shower simulation data.
* **models_plots.ipynb:** Notebook to visualize the results and predictions of trained models.
* **density_plots.ipynb:** Notebook to visualize the particle density per square meter as a function of incidence angle and energy.

## Workflow

1.  **Data Processing (ReadBinary.py):**
    * Simulation files in DAT format are processed using `ReadBinary.py`.
    * The data is filtered, organized, and saved to CSV files, both in binned and unbinned formats.
    * The data is organized by particle type (photon/proton), incidence angle, and simulation number.

2.  **Preprocessing and Visualization (datapreprocessing.py, density_plots.ipynb):**
    * `datapreprocessing.py` allows to explore and visualize specific aspects of the generated datasets.
    * `density_plots.ipynb` generates visualizations of particle density to better understand the distribution of particle showers.

3.  **Model Training (model.py / model.ipynb):**
    * The processed data is used to train a hybrid Conv1D-LSTM model that learns to classify the type of cosmic ray and reconstruct the incidence angle.
    * The model is evaluated and saved for later use.

4.  **Results Visualization (models_plots.ipynb):**
    * `models_plots.ipynb` is used to load the trained model and visualize its predictions and performance metrics.

## Requirements

* Python 3.x
* Libraries: pandas, numpy, scikit-learn, TensorFlow, matplotlib, panama.

## Usage

1.  Ensure that Python and the necessary libraries are installed.
2.  Place the simulation files in DAT format in the input directories specified in `ReadBinary.py`.
3.  Execute `ReadBinary.py` to process the data and generate CSV files.
4.  Use `datapreprocessing.py` and `density_plots.ipynb` to explore and visualize the data.
5.  Execute `model.py` or `model.ipynb` to train the model.
6.  Use `models_plots.ipynb` to visualize the model's results.

## Contribution

Contributions are welcome! If you find errors or have suggestions for improvement, feel free to create a pull request.
