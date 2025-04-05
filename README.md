# 🥔 Potato Plant Disease Classifier

A deep learning project for classifying potato leaf diseases using image data.  
We compare the performance of multiple CNN architectures before and after data augmentation, and apply fine-tuning to selected models to boost performance.

---

## 📌 Project Overview

- **Objective:** Classify potato leaf images into 3 categories:
  - 🌿 Healthy
  - 🍂 Early Blight
  - 🌫️ Late Blight

- **Challenge:** Class imbalance  
  → Healthy (152 images) vs. Disease classes (1,000 each)

- **Key Questions:**
  1. Does data augmentation improve model performance?
  2. How effective is fine-tuning on pre-trained models for this task?
  3. Can the models generalize to real-world potato leaf images?

---

## 🧪 Experiments

### 1. **Model Comparison (Before vs. After Augmentation)**

#### Code File: potato_diseases_classification.ipynb


| Model          | Type             | Pretrained | Augmentation | Fine-tuned |
|----------------|------------------|------------|--------------|------------|
| MobileNetV2    | Lightweight CNN  | ✅          | ✅ / ❌        | ✅          |
| VGG16          | Deep CNN         | ✅          | ✅ / ❌        | ✅          |
| ResNet50       | Residual Network | ✅          | ✅ / ❌        | ❌          |
| EfficientNetB0 | Scalable CNN     | ✅          | ✅ / ❌        | ❌          |

- **Observation:** Augmentation had mixed effects. Not all models improved significantly despite severe imbalance.
- **Insight:** Some architectures are more robust to imbalance than others.

---

### 2. **Fine-tuning (VGG16, MobileNetV2)**

#### Code File: potato_diseases_classification_tunned.ipynb

- **Step 1:** Freeze base model → Train classifier
- **Step 2:** Unfreeze top layers → Fine-tune with `CosineDecay` learning rate scheduler
- **Result:** Fine-tuned models showed improved validation performance but didn't reach perfect accuracy on real-world images.

---

## 🗃️ Features

- ✅ Data augmentation using `ImageDataGenerator` (rotation, zoom, flip)
- ✅ Model evaluation with confusion matrix & classification report
- ✅ Real-world image test (from Google images)
- ✅ Database integration:
  - Model metadata
  - Layer configurations
  - Training logs (loss, accuracy per epoch)

---

## 🔧 Tech Stack

- **Framework:** TensorFlow / Keras
- **Models:** VGG16, MobileNetV2, ResNet50, EfficientNetB0 (all ImageNet pretrained)
- **Data:** [Potato Leaf Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/hafiznouman786/potato-plant-diseases-data)
- **Storage:** SQLite3 for storing model, layers & training metadata
- **Scheduler:** `CosineDecay` learning rate schedule

---

