Email: daniel.alexander.arendt@student.hu-berlin.de
Email: domi.tarno+github@gmail.com
Email: oezkarak@student.hu-berlin.de

# Deconvolutional Neural Network for Fluorescence Spectrometer

This repository contains Python code developed as part of a semester project aimed at creating a Deconvolutional Neural Network (DeconvNet) for processing data from a fluorescence spectrometer.

## Overview

The main objectives of this project are to:

- Load and preprocess fluorescence data from an HDF5 file.
- Implement a Deconvolutional Neural Network architecture using PyTorch.
- Train the network on the provided dataset and optimize its performance.
- Analyze the network's output using clustering techniques.
- Visualize training loss and inference results during the training process.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- h5py
- Optuna
- Scikit-learn

You can install the required packages using pip:

```bash
pip install torch torchvision numpy matplotlib h5py optuna scikit-learn
```

## Dataset

The dataset used in this project is stored in an HDF5 format. The `H5Dataset` class is implemented to load and manage the data efficiently.

### Dataset Structure

The HDF5 file should contain groups with the following structure:

- Each group should have two datasets:
  - `X`: Input features.
  - `Y`: Target outputs.

## Training Module

### Code Structure

The main components of the training module include:

- **H5Dataset Class**: Manages loading of the HDF5 data.
- **DeconvNet Class**: Implements the Deconvolutional Neural Network architecture.
- **train Function**: Contains the logic for training the model, including data loading, forward propagation, loss calculation, and model optimization.

### Hyperparameters

- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Number of Epochs**: 1000

### Training

To start training the model, simply run the following command:

```bash
python your_training_script.py
```

Make sure to replace `your_training_script.py` with the actual name of your script file.

### Outputs

During training, the model will produce:

- Loss plots saved at intervals.
- Inference results saved as images.
- Model checkpoints saved every 25 epochs.

The final model will be saved as `model_final.pth`.

## Clustering Module

The clustering function analyzes the output data from the neural network and categorizes it into distinct groups using K-Means clustering.

### Code Explanation

The clustering functionality is encapsulated in the `cluster` function, which takes the following parameters:

- **yData**: The output data from the Deconvolutional Neural Network that you want to cluster.
- **method**: A placeholder for future clustering methods (currently not utilized).
- **ySize**: The size of the output data.
- **noise**: A threshold value to filter out low-intensity data points.

### Key Steps in the Clustering Function

1. **Data Preparation**: The function flattens the input data and identifies indices of data points that exceed the specified noise threshold.
2. **Early Exit**: If there are fewer than two valid data points above the noise threshold, the function returns `0`.
3. **Coordinate Calculation**: It calculates the coordinates of valid data points in a 2D space.
4. **K-Means Initialization**: The initial cluster centers are set using the positions of the first, last, and mid-point valid data.
5. **K-Means Fitting**: The K-Means model is trained on the resulting coordinates.
6. **Silhouette Score Calculation**: The silhouette score is calculated to evaluate the quality of the clustering.

### Usage

To use the clustering function, you can call it with the required parameters:

```python
score = cluster(yData, method, ySize, noise)
```

### Notes

- The current implementation is designed for 2D data clustering using K-Means.
- The `method` parameter is a placeholder for potential extensions to include other clustering algorithms in the future.

## Optimization Module

This section provides the implementation of an optimization module that utilizes the Optuna library to fine-tune parameters for the Deconvolutional Neural Network.

### Code Explanation

The optimization functionality is encapsulated in the `objective` function, which takes advantage of the Optuna library to perform the following steps:

1. **Hyperparameter Suggestion**: The function suggests values for a tensor containing 7 elements within the range of [0.0, 1.0].
2. **Model Prediction**: The suggested values are passed to the DeconvNet to obtain the predicted outputs.
3. **Clustering Evaluation**: The predicted outputs are clustered using the `cluster` function, and the silhouette score is calculated to evaluate the clustering quality.
4. **Optuna Study**: An Optuna study is created to minimize the objective value over 1000 trials.
5. **Best Parameters**: After optimization, the best hyperparameter values are printed and used to generate final model predictions.
6. **Visualization**: The predicted output corresponding to the best parameters is visualized using Matplotlib.

### Usage

To perform optimization, run the following command:

```python
python your_optimization_script.py
```

### Notes

- The model path should be set correctly to point to the pre-trained model checkpoint.
- The clustering algorithm currently used is K-Means, and you can modify it based on your specific clustering needs.
- The visualization will display the best model output after optimization.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch community for providing an excellent deep learning framework.
