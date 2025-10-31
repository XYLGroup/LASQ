# 🚀 LASQ: Unsupervised Hierarchical Learning for Illumination Enhancement

---

## 🏗️ 1. Introduction

LASQ reformulates low-light image enhancement as a statistical sampling process over hierarchical luminance distributions, leveraging a diffusion-based forward process to autonomously model luminance transitions and achieve unsupervised, generalizable light restoration across diverse illumination conditions.
The overall architecture is illustrated below 👇  

<p align="center">
  <img src="Figures/pipeline.png" alt="pipeline" width="80%">
</p>

---
## 📦 2. Create Environment

```bash
# Environment setup
conda create -n LSAQ python=3.11
conda activate LASQ

# Install dependencies
pip install -r requirements.txt
```
---
## 📂 3. Data Preparation

**Core Steps 🧵**

### 3.1 💾 Data Preparation  
LOL dataset: Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. "Deep Retinex Decomposition for Low-Light Enhancement". BMVC, 2018. [🌐Google Drive](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view)

### 3.2 🧱 Model Initialization  
Load backbone and fusion modules.  
Initialize with pre-trained weights if available.  
