# CIFAR-10 CNN Classifier

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. CIFAR-10 consists of 60,000 32×32 color images across 10 categories such as airplanes, cars, birds, and cats.

##  Features

- Loads and preprocesses CIFAR-10 dataset using torchvision
- Custom CNN architecture with convolution, pooling, dropout, and fully connected layers
- GPU acceleration with CUDA support
- Training and validation with cross-entropy loss and Adam optimizer
- Accuracy evaluation on test dataset
- Visualization of images, predictions, and training curves

## Project Structure

- `cifar10_cnn_solution.ipynb` → Main Jupyter notebook with code and explanations
- `notebook_ims/` → Sample images used for visualization in the notebook

## Model Architecture

The CNN model (`Net` class) is defined as follows:

**Convolutional Layer 1**
- Input: 3 × 32 × 32 (RGB image)
- Conv2d: 3 → 16 channels, kernel size = 3, padding = 1
- Activation: ReLU
- MaxPool2d: 2×2, stride = 2 → output: 16 × 16 × 16

**Convolutional Layer 2**
- Input: 16 × 16 × 16
- Conv2d: 16 → 32 channels, kernel size = 3, padding = 1
- Activation: ReLU
- MaxPool2d: 2×2, stride = 2 → output: 32 × 8 × 8

**Convolutional Layer 3**
- Input: 32 × 8 × 8
- Conv2d: 32 → 64 channels, kernel size = 3, padding = 1
- Activation: ReLU
- MaxPool2d: 2×2, stride = 2 → output: 64 × 4 × 4

**Fully Connected Layers**
- Flatten: 64 × 4 × 4 → 1024
- Linear(1024 → 500), Dropout(0.5), ReLU
- Linear(500 → 10) (final output for CIFAR-10 classes)

This structure allows the network to progressively extract spatial features before flattening into a vector for classification.

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Matplotlib
- Numpy
- Jupyter

##  Results

The model achieves strong performance (~70–80% test accuracy depending on training time and hyperparameters).

Results may vary based on batch size, epochs, and dropout rate.

## Future Improvements

- Use data augmentation (random crop, rotation, flip) for better generalization
- Try deeper architectures like ResNet or VGG
- Experiment with learning rate scheduling and regularization techniques

##  Author

Developed as part of a deep learning practice project using PyTorch for image classification tasks.
