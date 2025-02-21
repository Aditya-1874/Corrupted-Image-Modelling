# Corrupted Image Modeling

## Overview
This project focuses on **Corrupted Image Modeling (CiM)**, where the goal is to train a model capable of reconstructing or enhancing images with missing or corrupted regions. The approach involves learning a robust representation of images to infer and restore missing details.

## Methodology
- Utilizes a **BEiT-based masked image modeling approach**, where images are divided into patches, and specific patches are masked out.
- The model is trained to predict the masked patches using **self-supervised learning**, enabling it to learn meaningful representations of image structures.
- A **DALL-E decoder** is used to reconstruct the missing parts based on learned embeddings.

## Progress
- Successfully implemented the **generator model**, which processes corrupted images and produces meaningful reconstructions.
- Currently working on integrating a more refined enhancement module to improve restoration quality.

## Future Work
- Optimize the reconstruction pipeline by experimenting with different architectures.
- Evaluate performance on various datasets to measure the model’s effectiveness.
- Extend the model for real-world applications such as **medical imaging, satellite image restoration, and digital forensics**.

## Reference
This project is inspired by **BEiT: BERT Pre-Training of Image Transformers**. You can read more about it here:  
🔗 [BEiT Paper (arXiv:2202.03382)](https://arxiv.org/abs/2202.03382)

## Usage
To be updated once the full pipeline is complete.
