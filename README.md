# 🍎🥦 Fruit and Vegetable Image Recognition Model  

## 📌 Project Overview  
This project builds a custom Convolutional Neural Network (CNN) to classify images of fruits and vegetables. The model is trained on a dataset containing 36 classes, with 100 images or less per class.  

## [Click here to use the app](https://adenrele-fruit-and-vegetable-image-recognition-main-jrfvxi.streamlit.app)

---

## 📊 Summary, Metrics & Optimizations  

### **📝 Model Summary**  

| Layer (type)   | Output Shape        | Param #   |
|---------------|--------------------|----------|
| Conv2d-1      | [-1, 24, 128, 128] | 672    |
| MaxPool2d-2   | [-1, 24,64, 64]   | 0        |
| Linear-4      | [-1, 320]          | 31,457,600 |
| Dropout-5     | [-1, 320]          | 0        |
| Linear-6      | [-1, 64]           | 20,544    |
| Dropout-7      | [-1, 64]           |    |
| Linear-8      | [-1, 32]           | 2,080   |

Total params: 31,480,896

Trainable params: 31,480,896 



### 📊 Model Performance Metrics  

| Metric                 | Value |
|------------------------|-------|
| **Test Accuracy**      | 82%   |
| **Validation Accuracy** | 87%   |
| **Best Model Epoch**   | 70    |
| **Precision**          | 0.87   |
| **Recall**             | 0.86    |
| **F1-Score**          | 0.86   |


![Confusion Matrix](results/ConfusionMatrix.png) 


![ROACAUC](results/RocCurve.png)

### 🚀 Installation & Setup
```git clone https://github.com/Adenrele/Fruit-and-Vegetable-Image-Recognition.git```

```cd fruit-veg-classification```

```python3.10 -m venv myenv```

```source myenv/bin/activate``` 

```pip install -r requirements.txt```

Download Kaggle dataset
```python3 KaggleDownload.py script ```

Convert all images to RGBA to avoid error during training 
```python3 ConvertAllFIlesRGBA.py```

Run streamlit app in localhost
```sh shellScripts/launchStreanlit.sh```

## 🚀 Optimizations and Strategies  

To improve model performance and ensure efficient training, the following optimization techniques were applied:  

### 🔹 Data Augmentation  
- Applied **RandomResizedCrop(112, scale=(0.8, 1.0))** to introduce variability in training images.  
- Used **RandomHorizontalFlip(p=0.5)** to simulate real-world conditions.  
- Added **ColorJitter** to enhance color variations and improve generalization.  

### 🔹 Regularization Techniques  
- Implemented **Dropout (p=0.5)** in fully connected layers to reduce overfitting.  

### 🔹 Learning Rate Scheduling  
- Utilized **ReduceLROnPlateau** to **adjust the learning rate dynamically** based on validation loss.  
- If the validation loss did not improve for 4 epochs, the learning rate was **reduced by a factor of 0.5**, helping the model converge better.  

### 🔹 Early Stopping  
- **Monitored validation loss** and stopped training when performance stopped improving for **7 consecutive epochs**.  
- This helped **prevent overfitting** and **saved computational resources** by avoiding unnecessary training cycles.  

### 🔹 Model Architecture Improvements  
- Adjusted **kernel sizes and number of filters** to capture richer features.  
- Used **MaxPooling layers** to reduce spatial dimensions efficiently.  
- Experimented with different **activation functions** (ReLU, LeakyReLU).  

### 🔹 Loss Function & Optimization  
- Used **CrossEntropyLoss** as the loss function for multi-class classification.  
- Optimized with **SGD**, known for good generalistion. 

![Loss Over Epochs](results/LossEachEpoch.png) 

### 📈 Future Improvements  
- Experiment with **pretrained models** (e.g., ResNet, EfficientNet).  
- Implement **fine-tuning** on deeper architectures.  
- Try **advanced augmentation** methods like Mixup and Cutout.  
- Explore **hyperparameter tuning** using techniques like Grid Search or Bayesian Optimization.
- Consider deploying a database and an API to serve on any application.
- Consider a feedback and monitoring system.  
