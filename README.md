
# 🧠 PneumoScan – Pneumonia Detection from Chest X-rays

> Deep Learning Project for Medical Imaging  
> Detects pneumonia from chest X-ray images using CNNs and Transfer Learning.

---

## 📌 Project Overview

PneumoScan is a deep learning-based diagnostic tool that classifies chest X-ray images as either **NORMAL** or **PNEUMONIA**.  
It leverages both a **custom-built CNN** and a **pretrained VGG16 model** to evaluate and compare performance, ultimately selecting the most reliable model for real-world deployment.

---

## 🗂️ Project Structure

```
PneumoScan/
├── data/
│   └── raw/                # Original dataset
├── models/                 # Saved models (.keras)
├── notebooks/              # Step-by-step development notebooks
├── utils/                  # Helper scripts (optional)
├── app/                    # Streamlit/Flask web app (Step 8)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```



 

---

## 🧪 Dataset

- 📍 **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 👨‍⚕️ Images: 5,000+ X-ray scans labeled as:
  - `NORMAL`
  - `PNEUMONIA`

---

## 🚀 How to Run (Locally or on Colab)

1. **Clone the repo**:
```bash
git clone https://github.com/your-username/pneumoscan.git
cd pneumoscan
```
## 🚀 Setup Instructions

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run notebooks**:
- `01_explore_data.ipynb`
- `02_preprocess_data.ipynb`
- `03_custom_CNN.ipynb`
- `05_transfer_learning_vgg16.ipynb`
- `06_fine_tuning.ipynb`
- `07_evaluate_compare_models.ipynb`

---

## 📊 Model Evaluation Summary

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Custom CNN         | 0.84     | 0.86      | 0.84   | 0.83     |
| VGG16 (Frozen) ✅   | 0.91     | 0.91      | 0.91   | 0.91     |
| VGG16 (Fine-Tuned) | 0.91     | 0.91      | 0.91   | 0.91     |

> ✅ **Final Model Selected:** VGG16 (Frozen)

---

## 🧠 Key Concepts Used

- Convolutional Neural Networks (CNNs)
- Transfer Learning (VGG16 from ImageNet)
- Fine-Tuning
- Data Augmentation (rotation, flipping, zoom)
- Evaluation Metrics (Accuracy, Precision, Recall, F1-score)
- Confusion Matrix

---

## 📦 Tools & Frameworks

- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Matplotlib / Seaborn
- Google Colab (for GPU training)
- Streamlit or Flask (coming in Step 8)

---

## 📈 Sample Output

<p align="center">
  <img src="app/static/sample_confusion_matrix.png" width="400"/>
  <br><i>Confusion Matrix – Final Model (VGG16)</i>
</p>

---

## ✅ Next Steps (Step 8 – In Progress)

- Build a web app to let users upload chest X-rays
- Live prediction from the trained model
- Deploy via Streamlit Cloud or Render

---

## 👨‍💻 Author

**[Your Name]**  
- 💼 Software Engineer & AI Enthusiast  
- 🌐 [LinkedIn](https://www.linkedin.com/in/your-name)  
- 🐍 [Portfolio](https://yourportfolio.com)
