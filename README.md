# CIFAR-10 Image Classification with CNN Ensemble 🖼️🤖

This project implements an advanced image classification system using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset. By combining three distinct CNN architectures through ensemble methods, we achieve robust and accurate image classification results. 🎯

## Overview 📋

The project consists of:
- 🧠 Three CNN models with different architectures
- 🔄 Advanced data augmentation techniques
- 🤝 Ensemble methods (Simple Averaging and Majority Voting)
- 🌐 A Streamlit web interface for real-time predictions

### Project Components 📦
```
├── CNN_model_1.h5       # Basic CNN architecture
├── CNN_model_2.h5       # Enhanced CNN with batch normalization
├── CNN_model_3.h5       # ResNet-inspired architecture
└── ensemble_classifier.py # Streamlit interface
```

## Model Architectures 🏗️

### 1. CNN Model 1 (Base Model) 📊
* Foundational CNN architecture:
  * 4 convolutional layers with increasing filters
  * MaxPooling after each conv layer
  * Dropout (0.25) for regularization
  * Dense layers: 512 → 10 (output)

### 2. CNN Model 2 (Enhanced) 🚀
* Sophisticated architecture:
  * Batch normalization after each conv layer
  * 6 convolutional blocks
  * L2 regularization (lambda=0.01)
  * Advanced dropout patterns (0.3, 0.4)

### 3. CNN Model 3 (ResNet-Inspired) ⭐
* State-of-the-art features:
  * Residual connections (skip connections)
  * Identity mappings
  * Deep supervision
  * Bottleneck layers

## Training Pipeline 🛠️

### Data Preparation 📥
* Dataset: CIFAR-10 (60,000 32x32 color images)
  * 50,000 training images
  * 10,000 testing images
* Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
* Preprocessing:
  * Normalization: [0,1] range
  * Train/Val/Test split: 80/10/10

### Data Augmentation 🔄
```python
augmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1
)
```

### Training Configuration ⚙️
* Batch size: 32
* Epochs: 100 (with early stopping)
* Optimizer: Adam (lr=0.001)
* Loss: Categorical Crossentropy
* Metrics: Accuracy, Top-5 Accuracy

## Running the Interface 🖥️

### Local Deployment 🏠
```bash
streamlit run ensemble_classifier.py
```

### Google Colab Deployment ☁️
```python
!pip install streamlit
!streamlit run ensemble_classifier.py & npx localtunnel --port 8501
```

## Usage Guide 📱

1. 📥 Download the pre-trained models
2. 🚀 Launch Streamlit interface
3. 🖼️ Upload a 32x32 RGB image
4. 🔍 Click "Make Prediction"
5. 📊 View detailed classification results

## Performance Metrics 📈

Model Performance Comparison:
```
┌────────────────────────┬────────────┐
│ Model                  │ Accuracy   │
├────────────────────────┼────────────┤
│ CNN Model 1 (Base)     │    84%     │
│ CNN Model 2 (Enhanced) │    86%     │
│ CNN Model 3 (ResNet)   │    89%     │
│ Ensemble (Majority)    │    87%     │
└────────────────────────┴────────────┘
```

## Future Improvements 🔮

* 🔧 Implement more advanced ensemble techniques
* 🎯 Add model interpretability features
* 🚀 Optimize for mobile deployment
* 📊 Add confusion matrix visualization
* 🔄 Real-time data augmentation

## Requirements 📋

* Python 3.7+
* TensorFlow 2.x
* Streamlit
* NumPy
* OpenCV
* Pillow

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

---
Made with ❤️ for the Computer Vision community
