# Image Generation and Embedding Distillation Project
## Overview
This repository contains the code and resources for a project focused on improving the performance and efficiency of the VQGAN+CLIP image generation model. The project aims to optimize the model's scale and performance without significantly compromising accuracy, making it more suitable for applications requiring efficient training processes, such as game development and virtual reality.

## Key Features
**VQGAN+CLIP Model Optimization**: Fine-tuning and architectural enhancements to improve the model's adaptability and efficacy across various tasks and datasets.

**Image Embedding Distillation**: Development of a compact convolutional model (ImageEmbeddingNetwork) to replace the computationally expensive Vision Transformer (ViT) in the CLIP image decoder.

**Text-to-Image Generation**: Implementation of a generative model that produces images from natural language descriptions using a combination of linear layers, transformers, and convolutional layers.

**Dataset**: Utilizes the MS COCO dataset for training and validation, providing a rich source of captioned images for model training.

## Repository Structure
**Data Collection and Preprocessing**: Scripts for loading and preprocessing the MS COCO dataset.

**Model Training**: Code for training the VQGAN+CLIP model and the ImageEmbeddingNetwork.

**Image Generation**: Implementation of the generative model for producing images from text descriptions.

**Evaluation**: Scripts for evaluating model performance using Mean Squared Error (MSE) and Cosine Similarity metrics.

##Results
**Generator Model**: Achieved a test dataset error of 0.0605089515 using the MSE loss function.

**Image Embedding Network**: Achieved a test dataset error of 0.0001325494, demonstrating the effectiveness of the compact model in reproducing embeddings.

##Usage
**Clone the Repository**:

bash
Copy
git clone https://github.com/your-username/image-generation-distillation.git
cd image-generation-distillation

**Install Dependencies**:

bash
Copy
pip install -r requirements.txt

**Run the Training Script**:

bash
Copy
python train_generator.py
python train_image_embedding.py

**Generate Images from Text**:

bash
Copy
python generate_image.py --text "A painting of an apple in a fruit bowl" --width 400 --height 400 --model vqgan-imagenet-f16-16384 --seed 42 --max-iterations 500
##Contributors
Jiayu Yuan

Chun-Chih Yang

Robel Abdissa

Bala Sujith Potineni

Chi-Ao Chen

##References
**VQGAN+CLIP**: Open Domain Image Generation and Editing with Natural Language Guidance

**Attention is All You Need**

**Distilling the Knowledge in a Neural Network**

##License
This project is licensed under the MIT License - see the LICENSE file for details.
