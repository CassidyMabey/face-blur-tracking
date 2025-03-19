# Face Blur

## Overview

This is a small project using a neural network to locate and then blur your face.
<br>
### Navigation
**[Requirements](#requirements)**
<br>
[GPU requirements](#gpu)
<br>
[Dataset size](#dataset)
<br>
[Epochs amount](#epochs)

## Setup
### Training your own model

## Requirements
### Dependencies

To get started, you'll need to install the required libraries. You can install them using `pip`:

```bash
pip install -r requirements.txt
```
## Reccomended
### GPU
Any NVIDIA GPU will do better than most AMD or Intel GPU. NVIDIA has Compute Unified Device Architecture or CUDA which allows it to have better performance with neural networks. This will increase the speed of which your neural network will run and the better it will blur your face.
<div>
  <img src="./assets/nvidia_training.png" width="700" height="350">
  
</div>
This shows the training performance of lots of consumer cards of NVIDIA, AMD and Intel. Bear in mind the higher score, the better it will perform in training which includes epochs, processing and most other things.

### Dataset
The dataset is the amount of images which you are putting into it.
**Increase Dataset Size When:**

1. Your model achieves high accuracy very quickly like the image below
<img src="./assets/high_accuracy.png" width="250" height="200">
This basically means that your model is too smart for the amount of data your giving it (giving a spelling book to someone in college etc).

2. Theres a large gap between training and validation accuracy. This is refered to as overfitting
<img src="./assets/gap_between_training_validation.PNG" width="250" height="200">

3. Your model when being used doesnt detect faces very well. you will see this yourself

### Epochs
**Increase Epochs When:**

1. You’re training on a larger dataset. (>1000 images  start to consider it)
  
2. Loss is decreasing slowly (could also be a sign of a small data set)
<img src="./assets/gap_between_training_validation.PNG" width="250" height="200">


