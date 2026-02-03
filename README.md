# Industrial Surface Defect Segmentation

> 🎯 **Project Objective**  
> This project investigates the behavior of industrial surface defect segmentation models under realistic production-line constraints, with a focus on common practical challenges such as class imbalance, micro-scale defects, illumination variation, and annotation inconsistency.  
> Due to the limited availability of real industrial datasets, this project adopts procedurally generated synthetic data to simulate real-world factory imaging conditions.

## 🏭 Project Overview

This project implements a **U-Net–based image segmentation pipeline** using synthetic metal texture images with various types of surface defects, including scratches, cracks, and pits.  
Rather than aiming solely to maximize benchmark metrics, the emphasis is placed on **model design choices and engineering trade-offs** that are commonly encountered in industrial applications.

## 🔧 Key Features

### Synthetic Metal Texture Generation
- Procedural generation of grayscale metal backgrounds  
- Randomized textures, metal streaks, contrast, and brightness variations  

### Defect Simulation
- Scratches (including 1-pixel–level micro defects)  
- Cracks with irregular elongated structures  
- Pits / small circular defects  
- Natural blending of defects into metal surfaces  

### Class Imbalance Handling
- Adjustable defect ratio via `p_defect`  
- Optional use of `WeightedRandomSampler` to avoid trivial “all-good” predictions  

### Illumination & Texture Augmentation
- Gamma intensity variation  
- 2D illumination gradients  
- Local glare simulation  
- Gaussian noise injection  

### Annotation Inconsistency Simulation
- Boundary blurring  
- Morphological dilation / erosion  
- Mimics subjective differences in human annotation  

### Configurable U-Net Architecture
- Adjustable depth (1–3 downsampling stages)  
- Lightweight channel design  
- Enhanced preservation of fine-grained defects  

### Hybrid Loss Functions
- Binary Cross-Entropy (BCE)  
- Dice Loss (for pixel-level imbalance)  
- Focal Loss (to emphasize hard samples)  

## 📁 Project Structure

```
.
├── train.py / notebook # Training and evaluation pipeline
├── model.py # Configurable U-Net architecture
├── dataset.py # Synthetic data generation and Dataset definition
├── losses.py # BCE / Dice / Focal loss functions
├── outputs_synth_metal/
│ └── viz_samples/ # Visualization results
├── README.md # Project documentation
```

> 💬 **Note**: The actual file structure may vary depending on the implementation.

## 🚀 Quick Start

### Requirements
- Python 3.x  
- PyTorch  
- NumPy  
- PIL  
- Matplotlib  

### Run Training
```bash
python train.py

Alternatively, the notebook version can be executed on **Google Colab**.
```
## 🧠 Model Design

- **Architecture**: U-Net (configurable depth)  
- **Input**: 1 × 256 × 256 grayscale image  
- **Output**: Pixel-wise defect probability map  

### Design Focus
- Preservation of micro-scale defects  
- Balance between local detail and global context  
- Lightweight model design for potential real-time deployment  

## 📊 Evaluation & Visualization

During training, the pipeline visualizes:
- Input images  
- Ground truth defect masks  
- Model prediction results  

Outputs are automatically saved to:
outputs_synth_metal/viz_samples/


### Evaluation includes:
- Dice Score  
- Defect pixel ratio sanity checks  

## 🔧 Key Configuration Parameters

The following parameters can be adjusted:
- Defect ratio (`p_defect`)  
- U-Net depth (`unet_depth`)  
- Base channel size (`base_channels`)  
- Weights for BCE / Dice / Focal losses  
- Data augmentation and annotation noise switches  

## ⚠️ Limitations
- Training relies exclusively on synthetic data  
- No fine-tuning with real industrial images  
- Sim-to-real domain gap not yet addressed  
- Inference speed not yet optimized  

## 🔮 Future Improvements
- Domain Adaptation (synthetic → real)  
- Attention-based or Feature Pyramid architectures  
- Model distillation for real-time inference  
- Fine-tuning with limited real-world data  

## 📌 Notes

This project is intended for **learning, research, and portfolio demonstration purposes**, and is not designed as a production-ready system.