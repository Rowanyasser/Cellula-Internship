# Cellula-Internship
# 🦷 Teeth Classification with Deep Learning

##  Overview
This project focuses on developing a deep learning model for classifying dental images into **seven categories** related to oral health conditions:  

- **CaS**
- **CoS** 
- **Gum** 
- **MC**
- **OC**
- **OLP** 
- **OT**

The goal is to assist in **automated dental diagnostics** potentially helping dentists in early detection and treatment planning.

---

##  Dataset
- Images are stored in a folder structure with **train / validation / test splits**, where each subfolder represents a class.  
- The dataset was loaded using `tf.keras.utils.image_dataset_from_directory`.  
- Preprocessing included:  
  - **Normalization** (rescaling pixel values to [0,1])  
  - **Data Augmentation**: RandomFlip, RandomRotation, RandomZoom, RandomTranslation  

---

##  Model Architecture
- Built with **TensorFlow & Keras**  
- CNN with:  
  - Multiple **Conv2D + BatchNormalization + MaxPooling** blocks  
  - Dense layers with **Dropout (0.5, 0.3)**  
  - **L2 regularization** to reduce overfitting  
- Trained using:  
  - **Categorical Cross-Entropy loss** with **Label Smoothing**  
  - **Adam Optimizer**  
  - **50 epochs** with callbacks (EarlyStopping, ReduceLROnPlateau)  

---

##  Results
- **Test Accuracy:** **95%**  
- **Confusion Matrix:** strong diagonal dominance, meaning most predictions were correct.  
- **Classification Report:**  
  - Precision, Recall, F1-score all **above 0.90** for most classes.  
  - Slightly lower recall for OLP due to similarity with OC/MC.  

---

##  Visualizations
- **Histograms of Pixel Intensities** (RGB) → showed discriminative color features across classes.  
- **Sobel Edge Maps** → highlighted boundaries of teeth and lesions.  
- **t-SNE plots** → showed good class separability in feature space.  
- **Confusion Matrix** → revealed strengths and misclassifications.  

---

## Future Improvements
- Experiment with **deeper CNNs or transfer learning** (ResNet, EfficientNet).  
- Use **class balancing** and **advanced augmentations** (CutMix, MixUp). 
- Deploy as a **web or mobile app** for real-time use in dental clinics.  

---

## Repository Structure
```
├── teethClasses.ipynb          # Jupyter Notebook (training, evaluation, visualization)
├── Teeth Classification Project Report.pdf   # Detailed project report
├── README.md                   # Project documentation
├── training.log                # Log file recording training process
├── visualization_outputs               # visualization outputs are saved in this folder (couldn't be uploaded because of the size)     
└── Teeth_Dataset                      # Dataset (couldn't be uploaded because of the size)
```

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Rowanyasser/teeth-classification.git
   cd teeth-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook teethClasses.ipynb
   ```
4. Train the model and evaluate on the test set.

---

