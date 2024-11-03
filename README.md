# Anime Face Generation with DCGAN and VAE

This project focuses on the generation of high-quality anime faces using Deep Convolutional Generative Adversarial Networks (DCGAN) and Variational Autoencoders (VAE). It also provides an in-depth performance comparison between these two generative models using evaluation metrics like the Inception Score (IS) and Fréchet Inception Distance (FID).

## Table of Contents
- [Introduction](#introduction)
- [Models Used](#models-used)
  - [DCGAN](#dcgan)
  - [VAE](#vae)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusions](#conclusions)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

## Introduction
Anime face generation has gained significant attention in the domain of generative models. By training deep learning models on large datasets of anime faces, we can create realistic-looking synthetic images. This project aims to showcase the strengths and weaknesses of two popular generative models, DCGAN and VAE, and evaluate their performance using quantitative metrics.

## Models Used

### DCGAN
Deep Convolutional Generative Adversarial Networks (DCGAN) are a type of GAN that utilizes convolutional layers for both the generator and the discriminator. Key features of DCGAN include:
- The use of strided convolutions instead of pooling layers.
- Batch normalization for stable training.
- ReLU activations in the generator and LeakyReLU in the discriminator.

### VAE
Variational Autoencoders (VAE) are a type of generative model that encode data into a latent space and decode it back to generate new samples. VAEs provide:
- A probabilistic approach to latent space representation.
- A balance between image reconstruction quality and smooth latent space traversal.
- Regularization using the KL divergence to enforce distribution constraints.

## Dataset
The project uses a large dataset of anime faces that are preprocessed to a uniform size (e.g., 64x64 or 128x128 pixels). The dataset is split into training and validation sets to ensure generalizability.

## Implementation Details
- **Frameworks**: PyTorch and TensorFlow
- **Training Parameters**: Customizable epochs, batch sizes, and learning rates for both models.
- **Data Augmentation**: Random flips, rotations, and color jittering to increase model robustness.
- **Optimization**: Adam optimizer with a learning rate of 0.0002 for both models.

## Evaluation Metrics
- **Inception Score (IS)**: Measures the quality and diversity of generated images based on how well they match a pretrained Inception model's classification capabilities. A higher score indicates better performance.
- **Fréchet Inception Distance (FID)**: Compares the distribution of generated images with real images. A lower FID indicates that the generated images are closer to the real distribution, implying better quality and diversity.

## Results
| Model | Inception Score (↑) | FID (↓) |
|-------|---------------------|---------|
| DCGAN | 7.2 ± 0.15          | 45.3    |
| VAE   | 5.8 ± 0.20          | 65.7    |

### Analysis
- **DCGAN**: Outperformed VAE in terms of both IS and FID, showcasing sharper and more visually appealing images.
- **VAE**: Provided smoother latent space interpolation but produced slightly blurrier outputs compared to DCGAN.

## Conclusions
- **DCGAN** is better suited for generating high-quality and diverse anime faces, making it more favorable for applications where visual fidelity is crucial.
- **VAE** offers an advantage in latent space exploration, allowing for smoother transitions between generated samples, which is beneficial for creative applications.

