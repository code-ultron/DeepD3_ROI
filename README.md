[![DeepD3 project website](https://img.shields.io/website-up-down-green-red/https/naereen.github.io.svg)](https://deepd3.forschung.fau.de/)
[![Documentation Status](https://readthedocs.org/projects/deepd3/badge/?version=latest)](https://deepd3.readthedocs.io/en/latest/?badge=latest)

# 3D ROI Generation of dendritic spines using DeepD3

This repository provides an implementation of a Dendritic Spine Detection system using the Deep3D framework. The system is capable of detecting dendritic spines and generating 3D Regions of Interest (ROIs) of these spines for further analysis.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

1. Python 3.7 or later.
2. Other dependencies specified in the requirements.txt file.

## DeepD3

DeepD3, is a dedicated framework for the segmentation and detection of dendritic spines and dendrites.

Utilizing the power of deep learning, DeepD3 offers a robust and reliable solution for neuroscientists and researchers interested in analyzing the intricate structures of neurons.

With DeepD3, you are able to

* train a deep neural network for dendritic spine and dendrite segmentation
* use pre-trained DeepD3 networks for inference
* build 2D and 3D ROIs
* export results to your favourite biomedical image analysis platform
* use command-line or graphical user interfaces

### How to install and run DeepD3

DeepD3 is written in Python. First, please download and install any Python-containing distribution, such as [Anaconda](https://www.anaconda.com/products/distribution). We recommend Python 3.7 and more recent version.

Then, installing DeepD3 is as easy as follows:

    pip install deepd3

Now, you have access to almost all the DeepD3 functionalities.

### Custom-trained Model 

The model, built with TensorFlow and Keras, exhibits an architecture that comprises a single encoder and dual decoders. The input shape of this model is set to (384, 1472, 1) with the base filters being 32.
The custom-trained Model can be downloaded from the below link on google drive.

https://drive.google.com/file/d/1U-ZaXJyK-c5yB28xAt_iFRZupNgSIctv/view?usp=drive_link 

## Workflow

### Training and Validation Dataset Guide

This guide will show you how to access and use the DeepD3 training and validation datasets from the DeepD3 website for dendritic spine and dendrite detection.

Downloading Datasets

* Visit the [DeepD3 Website](https://deepd3.forschung.fau.de/)
* Navigate to the `Datasets` section.
* Look for the `DeepD3_Training.d3set` and `DeepD3_Validation.d3set` datasets. They should be clearly labeled.
* Click on the download link or button for each dataset.
* Save the datasets in your local directory or in a directory accessible to your Python environment.

### Train DeepD3 on your own dataset

We have prepared a Jupyter notebook `Training_deepd3.ipynb` in the folder `examples`. Follow the instructions to train your own deep neural network for DeepD3 use.
Steps we follow below:
* Import all necessary libraries. This includes TensorFlow and other related packages.
* Specify the paths for your training and validation datasets and prepare your datasets.

``` from deepd3.training.stream import DataGeneratorStream

# Specify the paths to your training and validation datasets
TRAINING_DATA_PATH = r"/path/to/your/downloaded/training/data"
VALIDATION_DATA_PATH = r"/path/to/your/downloaded/validation/data"

# Create data generators
dg_training = DataGeneratorStream(TRAINING_DATA_PATH, batch_size=32, target_resolution=0.094, min_content=50)
dg_validation = DataGeneratorStream(VALIDATION_DATA_PATH, batch_size=32, target_resolution=0.094, min_content=50, augment=False, shuffle=False)
```
Replace `/path/to/your/downloaded/training/data` and `/path/to/your/downloaded/validation/data` with the actual paths to your downloaded training and validation datasets.

* Visualize your input data to ensure it has been loaded correctly.
* Initialize your DeepD3 model with appropriate settings.
* Specify the callbacks and train your model.
* Plot the loss and accuracy metrics for both training and validation sets to evaluate your model.

This guide should get you started with training a deep learning model using DeepD3. If you have any questions, feel free to open an issue on this repository.


### Model Predictions and 3D ROI generation for dendritic spines

This guide will explain how to use a trained model to generate predictions on a new data stack and 3d roi generation of spines.
We have prepared a Jupyter notebook `3Droi_generation.ipynb` in the folder `examples`. Please refer to this notebook for detailed code execution.

DeepD3 Benchmark Dataset

The DeepD3 Benchmark dataset is used for prediction model and performance evaluation. It was collected from the [DeepD3 Website](https://deepd3.forschung.fau.de/). This dataset contains annotated samples of dendrites and their corresponding spines, serving as an ideal testbed for the development and evaluation of the machine-learning model. The dataset is used here to predict dendritic spine structures in 3D images and evaluate the performance of the model.

Please make sure to download and place the dataset in the correct directory as per the file paths specified in the code. The link to the DeepD3 dataset can be found here:

[DeepD3 Dataset](https://zenodo.org/record/7590773)


### Performance evaluation

This guide will explain how to use the 3D ROIs for the performance evaluation of the model. 
We have prepared a Jupyter notebook `performance_evaluation.ipynb` in the folder `examples`. Please refer to this notebook for detailed code execution.

The script utilizes techniques such as DBSCAN for initial clustering, KMeans for splitting large clusters, and distance-based thresholds to clean up clusters and merge adjacent ones. The final output is a labeled 3D scatter plot of the identified clusters and performance metrics such as Recall, Precision, and F1 Score calculated by comparing deepd3 roi results with human annotations.


Input

The script requires two  CSV files as input:

* `roidata.csv`: Contains information about the ROIs identified by the deepd3.You will get this file as the output of previous results.
* `Annotations_and_Clusters.csv` : Contains the manually annotated data. You will get this from [DeepD3 website](https://zenodo.org/record/7590773)
  
