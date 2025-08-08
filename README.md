# SPARSE: Few-Shot Semi-Supervised Learning via Class-Conditioned Image Translation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXXX)
[![Code](https://img.shields.io/badge/Code-GitHub-green)](https://github.com/username/SPARSE)

This repository contains the official implementation of **"SPARSE Data, Rich Results: Few-Shot Semi-Supervised Learning via Class-Conditioned Image Translation"** published at [Venue Name].

## 📋 Overview

SPARSE introduces a novel GAN-based semi-supervised learning framework specifically designed for extremely low labeled-data regimes (5-50 labeled samples per class). Our approach achieves robust classification performance in medical imaging scenarios where annotation costs are prohibitive.

### Key Features

- **Three-Phase Training Framework**: Alternating supervised and unsupervised training phases
- **Image-to-Image Translation**: Modifies real unlabeled images to preserve authentic anatomical features 
- **Ensemble-Based Pseudo-Labeling**: Combines confidence-weighted predictions with temporal consistency
- **Medical Imaging Focus**: Evaluated on 11 MedMNIST datasets with strong performance in 5-shot settings

![Framework](assets/framework.png)

## 🏗️ Architecture

Our framework integrates three specialized neural networks:

- **Generator (G)**: Attention U-Net for class-conditioned image translation
- **Discriminator (D)**: Dual-purpose network for authenticity assessment and classification  
- **Classifier (C)**: EfficientNet-B3 dedicated to classification tasks

The training consists of three main phases:

1. **Supervised Training Phase**: Joint training of all networks on limited labeled data
2. **Self-Supervised Pre-training**: Ensemble-based pseudo-labeling + class-conditioned image translation
3. **Synthetic Data Enhancement**: Training classifier on generated samples with known target classes

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/username/SPARSE.git
cd SPARSE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Preparation

The codebase includes utilities for processing MedMNIST datasets:

1. **Download and organize raw data**:
```bash
python src/dataset/process_medmnist.py -i /path/to/raw/data -o /path/to/organized/data
```

2. **Split into supervised/unsupervised sets**:
```bash
python src/dataset/split_datasets.py --path_to_dataset /path/to/organized/data --percentage_of_supervised_dataset 0.05 --saving_path ./dataset
```

3. **Create few-shot folders**:
```bash
python src/dataset/create_shot_folder.py -i /path/to/organized/data -o ./dataset
```

## 🎯 Usage

### Training

Run training with default parameters for 5-shot learning:

```bash
python src/train.py \
    --data_flag bloodmnist \
    --path_to_dt ./dataset \
    --percentage_supervised 5 \
    --epochs 1000 \
    --batch_size 24 \
    --lr 0.0002 \
    --project_id SPARSE_experiment \
    --exp_id bloodmnist_5shot
```

### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_flag` | Dataset name (e.g., bloodmnist, pathmnist) | bloodmnist |
| `--percentage_supervised` | Number of shots per class (5, 10, 20, 50) | 5 |
| `--epochs` | Total training epochs | 1000 |
| `--n_unsupervised` | Unsupervised training frequency (every N epochs) | 5 |
| `--percentile` | Confidence threshold for pseudo-labeling | 0.75 |
| `--batch_size` | Training batch size | 24 |
| `--lr` | Learning rate | 0.0002 |

### Example Commands

**5-shot learning on PathMNIST**:
```bash
python src/train.py --data_flag pathmnist --percentage_supervised 5 --exp_id path_5shot
```

**20-shot learning on DermaMNIST**:
```bash
python src/train.py --data_flag dermamnist --percentage_supervised 20 --exp_id derma_20shot
```

**Custom unsupervised training frequency**:
```bash
python src/train.py --data_flag bloodmnist --n_unsupervised 10 --exp_id blood_custom
```

## 📊 Results

### Performance Comparison (Accuracy %)

| Method | 5-shot | 10-shot | 20-shot | 50-shot |
|--------|--------|---------|---------|---------|
| **SPARSE** | **63.21** | **68.50** | **73.44** | **77.15** |
| **SPARSE_ens** | **66.22** | **70.95** | **75.71** | **78.28** |
| TripleGAN | 64.23 | 68.79 | 71.40 | 76.25 |
| SEC-GAN | 58.73 | 63.79 | 66.95 | 73.30 |
| CISSL | 45.84 | 46.80 | 49.19 | 51.20 |
| MatchGAN | 39.88 | 48.73 | 51.90 | 54.15 |
| EC-GAN | 35.09 | 34.66 | 34.29 | 47.41 |
| SGAN | 25.80 | 25.96 | 24.92 | 27.28 |

*Results averaged across 11 MedMNIST datasets*

### Key Findings

- **Statistical Significance**: SPARSE_ens achieves statistically significant improvements over most competitors in 5-shot settings (p < 0.05)
- **Ensemble Advantage**: Combining discriminator and classifier predictions consistently improves performance
- **Optimal Training Schedule**: Unsupervised training every 10 epochs (μ=10) provides best performance across all shot settings

## 🏥 Supported Datasets

The framework has been evaluated on 11 MedMNIST datasets:

| Dataset | Modality | Task | Classes | Samples |
|---------|----------|------|---------|---------|
| BloodMNIST | Blood Cell Microscopy | Multi-class | 8 | 17,092 |
| PathMNIST | Colon Pathology | Multi-class | 9 | 107,180 |
| DermaMNIST | Dermatoscopy | Multi-class | 7 | 10,015 |
| RetinaMNIST | Retinal OCT | Multi-class | 5 | - |
| TissueMNIST | Kidney Cortex | Multi-class | 8 | 236,386 |
| OrganAMNIST | Abdominal CT | Multi-class | 11 | 58,830 |
| ChestMNIST | Chest X-Ray | Binary | 2 | 112,120 |
| PneumoniaMNIST | Chest X-Ray | Binary | 2 | 5,856 |
| BreastMNIST | Breast Ultrasound | Binary | 2 | 780 |
| OrganCMNIST | Abdominal CT | Multi-class | 11 | 23,583 |
| OrganSMNIST | Abdominal CT | Multi-class | 11 | 25,211 |

## 🔧 Implementation Details

### Model Architectures

- **Generator**: Attention U-Net with encoder-decoder architecture
- **Discriminator**: PatchGAN with dual outputs (authenticity + classification)
- **Classifier**: EfficientNet-B3 with custom classification head

### Loss Functions

The supervised training combines four specialized losses:

1. **Prototype Loss**: Creates discriminative class-specific features
2. **Mutual Learning Loss**: Enables knowledge sharing between models  
3. **Entropy Minimization**: Encourages confident predictions
4. **Mixup Loss**: Provides regularization through data augmentation

### Training Schedule

- **Supervised Phase**: Every epoch using limited labeled data
- **Unsupervised Phase**: Every μ epochs (default μ=10) with pseudo-labeling and image translation
- **Optimizer**: AdamW with learning rate 0.0002
- **Total Epochs**: 1000

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{manni2024sparse,
  title={SPARSE Data, Rich Results: Few-Shot Semi-Supervised Learning via Class-Conditioned Image Translation},
  author={Manni, Guido and Lauretti, Clemente and Zollo, Loredana and Soda, Paolo},
  journal={arXiv preprint arXiv:XXXXX},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This work was partially funded by:
- Piano Nazionale Ripresa e Resilienza (PNRR) - HEAL ITALIA Extended Partnership - SPOKE 2 Cascade Call
- PNRR MUR project PE0000013-FAIR
- Resources provided by NAISS and SNIC at Alvis @ C3S

## 📧 Contact

For questions or issues, please contact:
- Guido Manni: [email](mailto:g.manni@unicampus.it)
- Paolo Soda: [email](mailto:p.soda@unicampus.it)

---

**Keywords**: Semi-supervised learning, Few-shot learning, Medical imaging, Deep learning, GAN-based methods, Class-conditioned image translation