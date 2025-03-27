# Generative Adversarial Networks (GANs) - PyTorch Implementation

## Project Overview

This repository contains a **PyTorch-based implementation** of **Generative Adversarial Networks (GANs)** for image generation using the **MNIST dataset**. The project automates dataset downloading, trains both **Generator** and **Discriminator** models, and provides visualization and performance evaluation.

### Implemented GAN Models:

- **Deep Convolutional GAN (DCGAN)** – Image generation with CNN-based networks.
- **Custom GAN Architecture** – Utilizing transposed convolutions and batch normalization.

## Features

- **Automatic downloading & preprocessing** of MNIST dataset.
- **Training pipeline** for GAN models with configurable hyperparameters.
- **GPU acceleration support** (CUDA-enabled training).
- **Visualization of generated images** at each epoch.
- **Comparison of real vs. generated images**.
- **Loss curve analysis** for Generator and Discriminator.
- **Model saving & checkpointing** for further use.

## Technologies Used

- **Python**
- **PyTorch**
- **Torchvision** (Dataset loading & transformations)
- **Matplotlib & NumPy** (Visualization & numerical operations)
- **OpenCV & PIL** (Image processing)

## Dataset

This implementation uses the **MNIST dataset**, which is automatically downloaded from **Yann LeCun’s website** into the specified directory.

## Project Structure

```
├── data/                   # MNIST dataset storage
├── models/                 # GAN model architectures
├── notebooks/              # Jupyter notebooks for training & evaluation
├── scripts/                # Training and evaluation scripts
├── results/                # Generated images and training logs
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
└── LICENSE                 # Open-source license
```

## Installation & Setup

Clone this repository:

```bash
git clone https://github.com/your-username/GANs-PyTorch.git
cd GANs-PyTorch
```

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

## Training the GAN

Run the training script:

```bash
python scripts/train_gan.py --dataset mnist --epochs 200
```

## Example Generated Images

During training, the generator progressively improves in creating realistic handwritten digits. Below is a sample of images generated at various epochs.

(Add example images here once trained.)

## Model Performance Evaluation

- **Loss Curves**: Monitor Generator and Discriminator losses.
- **Final Generated Images**: Compare generated images with real MNIST samples.
- **Checkpointing**: Save trained models for inference and further fine-tuning.

## Usage of Saved Models

To load the trained generator model:

```python
import torch
from models.generator import Generator

latent_dim = 100  # Must match training setting
model = Generator(latent_dim=latent_dim)
model.load_state_dict(torch.load("path/to/model.pth"))
model.eval()
```

## License

This project is open-source under the **MIT License**.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any inquiries or collaborations, reach out via:

- **Email:** [Adedayom70@gmail.com](mailto\:Adedayom70@gmail.com)
- **GitHub:** dayoxy
- **LinkedIn:** Adedayo Oguntonade

---

### ⭐ If you find this project useful, please star this repository! ⭐

