
# CIFAR-10 Image Classification with CNN Ensemble

  

This project implements an image classification system using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. It features three different CNN architectures and combines them using ensemble methods to achieve better classification accuracy.

  

## Overview

  

The project consists of:

- Three CNN models with different architectures

- Data augmentation techniques

- Ensemble methods (Simple Averaging and Majority Voting)

- A Streamlit web interface for real-time predictions

  

CNN_model_1.h5: Basic CNN architecture

CNN_model_2.h5: Enhanced CNN with batch normalization

CNN_model_3.h5: ResNet-inspired architecture with residual connections

ensemble_classifier.py: Streamlit interface for predictions

  

**Models Architecture**

  

* CNN Model 1:

* Basic CNN with multiple convolutional layers

* Dropout for regularization

* Dense layers for classification

* CNN Model 2:

* Enhanced architecture with batch normalization

* Deeper network with additional convolutional blocks

* L2 regularization

* CNN Model 3:

* ResNet-inspired architecture

* Residual connections

* Extensive regularization techniques

  

**Training Process**

  

* Data Preparation:

* Load CIFAR-10 dataset

* Normalize pixel values

* Split into training, validation, and test sets

* Data Augmentation:

* Rotation

* Width/height shifts

* Horizontal flips

* Zoom and shear transformations

* Training:

* Batch size: 32

* Maximum epochs: 100

* Learning rate scheduling

* Early stopping

  

**Ensemble Methods**

  

The project implements two ensemble techniques:

  

* Simple Averaging

* Majority Voting

  
  

# Running the Interface

  

## Local Deployment

  

```bash

streamlit  run  ensemble_classifier.py

```

  

## Google Colab Deployment

  

1. Install required packages:

  

```python

!pip install streamlit

```

  

2. Run the interface:

  

```python

!streamlit run ensemble_classifier.py & npx localtunnel --port 8501

```

  

# Usage

  
1. Download the three models first in the directory of your notebook
2. Launch the Streamlit interface.
3. Upload an image (32x32 RGB).
4. Click "Make Prediction."
5. View the classification result.

  

# Performance

  

The ensemble model achieves improved accuracy compared to individual models:

  

-  **Model 1**: ~84% accuracy

-  **Model 2**: ~86% accuracy

-  **Model 3**: ~89% accuracy

-  **Ensemble (Majority Voting)**: ~87% accuracy

  


