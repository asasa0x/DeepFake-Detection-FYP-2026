# DEEPFAKE DETECTION SYSTEM
## Data Preprocessing & Dataset Quality Audit Report
**Final Year Project (FYP)**

---

### 1. Introduction

The purpose of the data preprocessing phase was to prepare a clean, balanced, reliable, and reproducible dataset for training a deepfake detection model.

The dataset contains two classes:
- **Real:** Authentic human face images.
- **Fake:** AI-generated and synthetically manipulated human face images.

The complete preprocessing pipeline included dataset sourcing, cleaning, image integrity verification, duplicate removal, watermark auditing, class balancing, quality analysis, bias mitigation, and stratified dataset splitting.

---

### 2. Dataset Overview

**Dataset location:**
`backend/DF-DATASET/Cleaned`

| Category | Images |
|---|---:|
| Real | 72,326 |
| Fake | 72,326 |
| Total | 144,652 |
| Class balance | 50% / 50% |
| Dataset size | 4.390 GB |

#### Real image sources

| Source | Description | Images |
|---|---|---:|
| RCA | CelebA | 36,228 |
| RFF | FFHQ | 36,098 |
| **Total** | | **72,326** |

#### Fake image sources

| Source | Description | Images |
|---|---|---:|
| FSG | StyleGAN | 61,667 |
| FSD | Stable Diffusion | 9,971 |
| AI | Other AI-generated images | 688 |
| **Total** | | **72,326** |

---

### 3. Dataset Cleaning and Integrity Verification

The dataset underwent several integrity checks:

- Corrupted images checked.
- Zero-byte files checked.
- Unsupported formats checked.
- Unreadable images checked.
- SHA-256 duplicate detection performed.
- Cross-class and within-class duplicate checks performed.
- Image dimensions and color modes verified.

**Results:**

| Check | Result |
|---|---:|
| Corrupted images remaining | 0 |
| Zero-byte files | 0 |
| Unsupported files | 0 |
| Unreadable images | 0 |
| Remaining duplicate hashes | 0 |
| Exact duplicates removed | 8,336 |

All 144,652 remaining images have unique SHA-256 hashes.

---

### 4. Watermark Detection and Removal

A targeted audit identified 313 Fake images containing the `pollinations.ai` watermark.

These images were backed up and removed to reduce the possibility of the model learning watermark patterns instead of genuine forensic characteristics.

- Identified watermark images: 313
- Backed up and removed: 313
- Watermarked images remaining: 0

---

### 5. Image Dimensions and Color Analysis

All images were verified at 299 × 299 pixels.

The model preprocessing pipeline resizes images to 224 × 224 pixels.

| Color mode | Images |
|---|---:|
| RGB | 143,681 |
| Grayscale (L) | 971 |
| Total | 144,652 |

A grayscale imbalance was identified, particularly among the Stable Diffusion Fake images. This was addressed through training-time RandomGrayscale augmentation.

---

### 6. Dataset Quality and Bias Audit

A statistical audit was performed at 224 × 224 resolution to identify possible differences between Real and Fake images.

| Feature | Real Median | Fake Median | AUC |
|---|---:|---:|---:|
| Laplacian Blur | 268.06 | 344.49 | 0.6391 |
| Sobel Sharpness | 52.01 | 52.26 | 0.5109 |
| Brightness | 112.53 | 115.47 | 0.5226 |
| Contrast | 58.95 | 55.63 | 0.5790 |
| Noise Level | 1.297 | 1.575 | 0.6289 |
| File Size | 29.24 KB | 33.57 KB | 0.6827 |
| Color Saturation | 89.11 | 85.50 | 0.5368 |
| JPEG Blockiness | 1.030 | 1.023 | 0.5554 |

The audit identified differences in blur, noise, and file size distributions. These findings were considered when designing the training augmentation strategy.

---

### 7. Bias Mitigation and Data Augmentation

Two targeted augmentations were introduced:

**Gaussian Blur**
- Applied using RandomApply.
- Probability: 25%.
- Kernel size: 3.
- Sigma range: 0.3–1.4.

Purpose: Reduce the model's dependence on differences in image sharpness and blur between data sources.

**Random Grayscale**
- Probability: 10%.

Purpose: Reduce the effect of grayscale distribution differences between Real and Fake classes.

---

### 8. Dataset Splitting

A reproducible stratified split was generated using seed 42.

The dataset was divided into:

| Split | Percentage | Total | Real | Fake |
|---|---:|---:|---:|---:|
| Train | 70% | 101,256 | 50,628 | 50,628 |
| Validation | 15% | 21,698 | 10,849 | 10,849 |
| Test | 15% | 21,698 | 10,849 | 10,849 |
| **Total** | **100%** | **144,652** | **72,326** | **72,326** |

The split was stratified by both class and sub-source prefix.

Generated files:
- `splits/train.csv`
- `splits/val.csv`
- `splits/test.csv`

Verification confirmed:
- Train/validation overlap: 0
- Train/test overlap: 0
- Validation/test overlap: 0
- Missing file paths: 0

---

### 9. Image Preprocessing Pipeline

Images are processed using the following sequence:

1. Convert image to RGB.
2. Resize to 224 × 224.
3. Convert image to PyTorch tensor.
4. Normalize using ImageNet statistics.

**Normalization parameters:**

Mean:
`[0.485, 0.456, 0.406]`

Standard deviation:
`[0.229, 0.224, 0.225]`

#### Training transformations

- RGB conversion
- Resize to 224 × 224
- Random Horizontal Flip (p=0.5)
- Random Rotation (±10°)
- Color Jitter
- Random Gaussian Blur (p=0.25)
- Random Grayscale (p=0.10)
- ToTensor
- ImageNet Normalization

#### Validation and testing transformations

- RGB conversion
- Resize to 224 × 224
- ToTensor
- ImageNet Normalization

Validation and testing transformations are deterministic. Repeated transformations produced a maximum absolute difference of 0.0.

---

### 10. Pipeline Verification

The data pipeline was tested before full model training.

Verified results:
- Training samples: 101,256
- Validation samples: 21,698
- Testing samples: 21,698
- Missing files: 0
- Dataset overlap: 0
- Input tensor shape: `[8, 3, 224, 224]`
- Model output shape: `[8, 2]`
- Probability output shape: `[8, 2]`
- Sample loss: 0.71772

---

### 11. Final Dataset Readiness

The preprocessing and quality audit stages were successfully completed.

The final dataset contains 144,652 verified images with an exact 50/50 Real/Fake class balance. Duplicate removal, watermark auditing, quality analysis, augmentation design, and stratified splitting have been completed.

**Status: Dataset preparation and pipeline verification completed.**

---

### 12. Tools and Technologies

- Python
- PyTorch
- Torchvision
- Pillow (PIL)
- NumPy
- OpenCV
- Pandas
- SHA-256 hashing
- CSV manifests
