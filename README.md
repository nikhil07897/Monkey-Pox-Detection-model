# 🐒✨ Monkeypox Classifier Using Machine Learning  

> **AI Meets Healthcare**: An intelligent system to detect Monkeypox from images using the power of deep learning.  

---

## 🌟 Highlights  

- 🚀 **State-of-the-Art Model**: Powered by MobileNetV2 for high performance and efficiency.  
- 📊 **In-Depth Analysis**: Features accuracy, loss graphs, and confusion matrix visualizations.  
- 🖼️ **Interactive Visuals**: Displays sample classified images with prediction labels.  
- 💾 **Ready-to-Use**: Preprocessing pipeline included for a seamless experience.  
- 🔄 **Automated Training**: Incorporates callbacks like early stopping and learning rate reduction.

---

## 📜 Table of Contents  

1. [Project Overview](#-project-overview)  
2. [Features](#-features)  
3. [Sample Images](#-sample-images)  
4. [Data Preparation](#-data-preparation)  
5. [Model Architecture](#-model-architecture)  
6. [Results](#-results)  
7. [Installation and Usage](#-installation-and-usage)  
8. [Future Enhancements](#-future-enhancements)  
9. [Acknowledgments](#-acknowledgments)  
10. [Contributors](#-contributors)  

---

## 🧐 Project Overview  

This project aims to create an automated classification model for **Monkeypox diagnosis** using machine learning. With the increasing demand for accurate and timely medical imaging solutions, this classifier can play a pivotal role in healthcare analytics.  

💡 **Goal**: Classify images into two categories:  
1. **Monkeypox Positive** 🐒  
2. **Others** 🌟  

---

## ✨ Features  

- 📈 **High Accuracy**: Leverages MobileNetV2 to achieve exceptional performance.  
- 🧹 **Data Augmentation**: Handles imbalanced datasets with augmentation techniques.  
- 🔍 **Explainability**: Visualizations like confusion matrices and accuracy/loss plots.  
- 📂 **Reusable Dataset Pipeline**: Easily extendable for other skin conditions or datasets.  
- 🔧 **Customizability**: Modify hyperparameters, layers, and configurations as needed.  

---

## 🖼️ Sample Images  

**Monkeypox Samples**:  
![Monkeypox Sample Images](link_to_image_1)  

**Others Samples**:  
![Other Sample Images](link_to_image_2)  

---

## 📂 Data Preparation  

1. **Dataset**: The dataset includes images from the categories "Monkeypox" and "Others".  
2. **Preprocessing Steps**:  
   - Images resized to 224x224.  
   - Data normalized (pixel values scaled to [0, 1]).  
   - Labels encoded as categorical (One-Hot Encoding).  
3. **Split Ratio**:  
   - **80% Training** 🏋️‍♂️  
   - **20% Testing** 🧪  

---

## 🧠 Model Architecture  

Our classifier is built using **MobileNetV2** and features:  

- 🌐 **Base Layers**: Pre-trained MobileNetV2 with ImageNet weights.  
- 📏 **Global Average Pooling**: Reduces feature maps to meaningful averages.  
- 🔠 **Fully Connected Layers**: Dense layers for classification.  
- 🛡️ **Dropout Layers**: Prevents overfitting by regularization.  

**Hyperparameters**:  
- **Epochs**: 40  
- **Batch Size**: 32  
- **Optimizer**: Adam  
- **Loss Function**: Binary Crossentropy  

---

## 📊 Results  

- **Training Accuracy**: 🌟 **98%**  
- **Validation Accuracy**: 🌟 **96%**  

**📈 Performance Graphs**:  
- Accuracy:  
![Accuracy Plot](link_to_accuracy_plot)  

- Loss:  
![Loss Plot](link_to_loss_plot)  

**Confusion Matrix**:  
![Confusion Matrix](link_to_confusion_matrix)  

**Sample Predictions**:  
Randomly selected images with predicted and true labels:  
![Predicted Samples](link_to_sample_predictions)  

---

## 🛠️ Installation and Usage  

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/nikhil07897/monkeypox-classifier.git
   ```  

2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```  

3. **Run the script**:  
   ```bash
   python nikhil_monkey_pox_classifier_using_machine_learning.py
   ```  

4. **Use your own images for classification**:  
   Place your image in the `data/` folder and modify the script to load it.  

---

## 🚀 Future Enhancements  

- 🔬 Expand the dataset for higher diversity and accuracy.  
- 📡 Deploy the model as a web app using Flask or FastAPI.  
- 🧠 Integrate Grad-CAM for explainable AI to visualize important features.  
- 📱 Create a mobile-friendly version for real-time diagnosis.  

---

## 📢 Acknowledgments  

This project is built on the shoulders of giants:  
- TensorFlow  
- Keras  
- OpenCV  
- Matplotlib  
- MobileNetV2 (Pre-trained on ImageNet)  

Thanks to the open-source community for providing amazing tools and datasets! 🌟  

---

## 🤝 Contributors  

| 👨‍💻 Nikhil |  
| :---: |  
| [![GitHub followers](https://img.shields.io/github/followers/nikhil07897?style=social)](https://github.com/nikhil07897) |  

---
