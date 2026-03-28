# AIRL Internship — Final Round Coding Assignment  

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Candidate:** Akash Kumar Pandey  

---

## Overview  

This repository contains the implementation of the **Final Round Coding Assignment** for the AIRL Internship.  

The project focuses on two key areas of modern computer vision:  

- Vision Transformer (ViT) for image classification  
- Text-driven image segmentation using Segment Anything Model (SAM 2)  

The objective is to demonstrate:  
- Strong understanding of deep learning fundamentals  
- Ability to implement research papers from scratch  
- Experience with modern foundation models  
- End-to-end machine learning pipeline development  

---

## Repository Structure  


├── q1.ipynb # Vision Transformer (ViT) implementation and training
├── q2.ipynb # Text-driven segmentation using SAM 2
├── README.md # Project documentation

---

## Environment  

- Language: Python 3  
- Framework: PyTorch  
- Platform: Google Colab (GPU enabled)  
- GPU Used: Tesla T4  
- Libraries:
  - torchvision  
  - numpy  
  - matplotlib  
  - OpenCV  
  - transformers  

---

## How to Run  

1. Open Google Colab  
2. Upload `q1.ipynb` or `q2.ipynb`  
3. Enable GPU:  
   Runtime → Change runtime type → GPU  
4. Run all cells sequentially  

---

# Q1 — Vision Transformer on CIFAR-10  

## Objective  

Implement a Vision Transformer (ViT) from scratch and achieve high classification accuracy on CIFAR-10 dataset.  

---

## Methodology  

Based on:  
**“An Image is Worth 16x16 Words” — Dosovitskiy et al., ICLR 2021**  

### Architecture Overview  

1. Patchify input image (32×32 → 4×4 patches)  
2. Flatten patches into embeddings  
3. Add learnable positional embeddings  
4. Add CLS token  
5. Pass through Transformer encoder layers:
   - Multi-Head Self Attention (MHSA)  
   - Feedforward MLP  
   - Residual connections  
   - Layer Normalization  
6. Classification using CLS token  

---

## Training Configuration  

- Patch Size: 4  
- Embedding Dimension: 256  
- Transformer Depth: 8  
- Attention Heads: 8  
- MLP Ratio: 4.0  
- Batch Size: 128  
- Optimizer: AdamW  
- Learning Rate: 3e-4  
- Weight Decay: 0.05  
- Scheduler: Cosine Annealing  
- Epochs: 30  

---

## Results  

| Model        | Patch | Depth | Dim | Best Test Accuracy (%) |
|--------------|-------|-------|-----|-------------------------|
| ViT-starter  | 4x4   | 8     | 256 | 80.26                  |

---

## Analysis  

- Accuracy improves steadily across epochs  
- Smaller patch size improves spatial understanding  
- Data augmentation enhances generalization  
- Transformer depth and embedding size affect performance  

### Improvements  

- Train for 60–100 epochs  
- Use MixUp / CutMix  
- Apply learning rate warmup  
- Increase model capacity  

---

# Q2 — Text-Driven Image Segmentation with SAM 2  

## Objective  

Perform segmentation of objects in an image using natural language prompts.  

---

## Pipeline  

1. Load image  
2. Input text prompt (e.g., "a dog")  
3. Convert text → region proposals using:
   - CLIPSeg  
   - GroundingDINO  
4. Provide region seeds to SAM  
5. Generate segmentation mask  
6. Overlay mask on image  

---

## Technologies Used  

- Segment Anything Model (SAM)  
- CLIPSeg / GroundingDINO  
- PyTorch  
- OpenCV  
- Matplotlib  

---

## Limitations  

- Depends on quality of text-to-region model  
- Ambiguous prompts reduce accuracy  
- Requires pretrained weights download  
- Computationally intensive  

---

## Submission Details  

- Repository contains:
  - q1.ipynb  
  - q2.ipynb  
  - README.md  

- Both notebooks run end-to-end on Colab with GPU  
- Best CIFAR-10 Accuracy: **80.26%**  

---

## Conclusion  

This project demonstrates:  

- Implementation of Vision Transformers from scratch  
- Training and optimization of deep learning models  
- Use of modern foundation models (SAM)  
- Integration of multiple AI components into a pipeline  

---

## Author  

**Akash Kumar Pandey**  
Computer Science Undergraduate  
