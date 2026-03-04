# Ensemble-Based Blood Cancer Stage Classification from Smear Images

## Overview
This project presents an interpretable and computationally efficient machine learning framework for multi-stage leukemia classification from peripheral blood smear (PBS) images.

Unlike deep CNN-based black-box approaches, this system emphasizes:
- Clinical interpretability
- Feature transparency
- CPU-level deployability
- Ensemble robustness

The model classifies smear images into four diagnostic stages:
- Benign
- Early-Leukemic
- Pre-Leukemic
- Pro-Leukemic

---

## Dataset

Source: Mehrad Aria’s Leukemia Blood Smear Dataset (Kaggle)  
https://www.kaggle.com/datasets/mehradaria/leukemia

Total Images: 3,256  

Class Distribution:
- Benign – 504
- Early – 985
- Pre – 963
- Pro – 804

All images resized to 256×256 resolution.

---

## Pipeline Architecture

### 1️⃣ Preprocessing
- Gray-World White Balancing (illumination normalization)
- Bilateral Filtering (edge-preserving denoising)
- CLAHE (local contrast enhancement)

### 2️⃣ Segmentation
- Enhanced Fuzzy C-Means (EFCM) with spatial regularization
- Morphological opening and closing operations
- Robust nucleus-cytoplasm separation

### 3️⃣ Feature Engineering (102 Features)
- Shape: Area, Perimeter, Circularity, Solidity, Hu Moments
- Color: HSV & LAB statistics
- Texture: LBP histograms, GLCM (Contrast, Energy, Homogeneity, Correlation)

### 4️⃣ Feature Selection
- Recursive Feature Elimination (RFE)
- Linear SVM estimator
- Reduced from 102 → 48 optimal features

### 5️⃣ Ensemble Classification
Base Models:
- Support Vector Machine (RBF)
- Random Forest
- Extra Trees

Fusion Strategies:
- Weighted Soft Voting (F1-based weights)
- Stacked Ensemble (Logistic Regression meta-learner)

10-fold Out-of-Fold validation used for stacking.

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| SVM | 96.5% | 0.964 |
| Random Forest | 94.6% | 0.945 |
| Extra Trees | 94.8% | 0.946 |
| Weighted Ensemble | 96.2% | 0.961 |
| **Stacked Ensemble** | **96.8%** | **0.967** |

ROC-AUC (Macro): 0.9975

- Early, Pre, Pro: AUC = 1.00
- Benign: AUC = 0.99

---

## Key Highlights

- Interpretable handcrafted feature pipeline
- Reduced multicollinearity via RFE
- Balanced precision-recall across minority and majority classes
- CPU-efficient and deployable in low-resource environments
- Suitable for clinical decision-support integration

---

## Future Improvements

- Transfer learning comparison with CNN models
- Grad-CAM / SHAP for explainability
- Multi-modal hematology feature integration

---

## Tech Stack

Python, OpenCV, scikit-learn, NumPy, Matplotlib, Gradio

http://127.0.0.1:7860/

## Training

```bash
python3 src/train_enhanced_v2.py --data_root ./data/Original --save_dir ./artifacts_v2
```

## Evalaution

```bash
python src/eval_plots_enhanced_v2.py --artifacts ./artifacts_v2
```


## Inference

```bash
python3 src/app.py --artifacts ./artifacts_v2
```

## License

This project is licensed under the MIT License.
