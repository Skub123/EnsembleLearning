# Classification using Transfer Learning and Ensemble in PyTorch

We implements a deep learning pipeline using transfer learning with ResNet-50, VGG-16, and EfficientNet-B0 on a balanced  radiography dataset. The final classification result is obtained through ensemble learning.

## Models Used for Fine-Tuning

The following deep learning models were fine-tuned on our dataset:

- **ResNet50 / ResNet101** – Employ residual connections to prevent vanishing gradients  
- **DenseNet121** – Enables efficient gradient flow through dense connections; used in *CheXNet*  
- **EfficientNet-B3** – Optimizes width, depth, and resolution for high performance with fewer parameters  
- **Xception** – Leverages depthwise separable convolutions for computational efficiency  
- **InceptionV3** – Captures multi-scale features using inception modules

