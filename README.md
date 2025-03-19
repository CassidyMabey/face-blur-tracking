# Face Blur

## Overview

This is a small project using a neural network to locate and then blur your face.

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

### When to increase / decrease the dataset (amount of images)
Increase Dataset Size When:

    
    There’s a large gap between training and validation accuracy — overfitting warning sign.
    You’re seeing unstable validation accuracy/loss — could be due to lack of variety in your data.
    Your model performs well on training data but poorly in the real world.
    You’re using a complex model (like deep CNNs) — they need lots of data to generalize well.

1. Your model achieves high accuracy very quickly like the image below
<img src="./assets/high_accuracy.png" width="200" height="175">
This basically means that your model is too smart for the amount of data your giving it (giving a spelling book to someone in college etc).


2️⃣ Number of Epochs
🔼 Increase Epochs When:

    The model is still learning — loss is decreasing steadily, and accuracy is still improving.
    You’re using data augmentation (it slows down learning a bit, so more epochs help).
    You’re training on a larger dataset or using a deeper model.
    You use regularization (like dropout, weight decay) — helps avoid overfitting but often needs more epochs to converge.

🔍 Example:

If loss is still dropping at epoch 15 and accuracy is improving, training longer may help.
🔽 Decrease Epochs When:

    Your model overfits quickly (loss on validation increases while training loss decreases).
    Training and validation accuracy plateau early, like in your case — no need to go on.
    You use early stopping — this automatically stops training when validation performance stops improving.
    You’re running a quick prototype or test.

⚖️ Balance: Dataset vs Epochs
Small Dataset	Big Dataset
Needs fewer epochs (model memorizes quickly)	Needs more epochs (model takes longer to generalize)
High risk of overfitting	Better generalization
Data augmentation recommended	May or may not need augmentation
Few Epochs	Many Epochs
Good for fast testing	Better for full training
Risk underfitting on large datasets	Risk overfitting on small datasets
Combine with early stopping	Combine with regularization
🔑 Rules of Thumb:
Situation	What to Do
Fast overfitting, high accuracy early	Add more data or reduce epochs
Training and validation both low	Increase data, epochs, or model complexity
Validation accuracy stagnant or dropping	Use early stopping, reduce epochs, or regularize
Loss still decreasing, accuracy rising	Increase epochs
Using data augmentation	Increase epochs to allow more learning
💡 Tips:

    Always visualize your learning curves like you did — they tell you exactly what’s happening.
    Combine data size, model complexity, epochs, and regularization for best results.
    Try early stopping with patience=3-5 epochs as a safety net.

If you’d like, I can analyze your current training logs and suggest a new training strategy (epochs, batch size, augmentation setup). Want that?
