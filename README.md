# **Emotion Detection with Real-time Camera Access**  

## **Overview**  
This project detects human emotions in real-time using a **Convolutional Neural Network (CNN)** trained on the **FER-2013 dataset**. The model classifies facial expressions into seven different categories and processes real-time input from a webcam using **OpenCV**.

---

## **Dataset Structure**  
The **FER-2013 dataset** contains labeled images of facial expressions divided into **training** and **testing** sets.  

```
dataset/
│── train/
│   ├── happy/
│   ├── anger/
│   ├── fear/
│   ├── surprise/
│   ├── neutral/
│   ├── disgust/
│   ├── sad/
│── test/
│   ├── happy/
│   ├── anger/
│   ├── fear/
│   ├── surprise/
│   ├── neutral/
│   ├── disgust/
│   ├── sad/
```

Each folder contains thousands of images categorized based on emotion labels.

---

## **Project Structure**  
```
emotiondetection/
│── dataset/               # FER-2013 dataset
│── models/                # Saved trained model
│── src/
│   ├── train_model.py      # Model training script
│   ├── test_model.py       # Model testing script
│   ├── realtime_emotion.py # Real-time emotion detection
│── requirements.txt        # Required dependencies
│── README.md               # Project documentation
```

---

## **Installation & Setup**  
### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/your-username/emotion-detection-ai.git
cd emotion-detection-ai
```

### **2️⃣ Install Dependencies**  
Ensure you have Python installed, then run:  
```sh
pip install -r requirements.txt
```

---

## **How to Run?**  

### **🔹 Train the Model**  
```sh
python src/train_model.py
```
- Trains the CNN model on the FER-2013 dataset.
- Saves the trained model in the `models/` directory.

### **🔹 Test the Model**  
```sh
python src/test_model.py
```
- Evaluates the trained model on the test dataset.
- Outputs accuracy, loss, and sample predictions.

### **🔹 Run Real-time Emotion Detection**  
```sh
python src/realtime_emotion.py
```
- Accesses the webcam using OpenCV.
- Detects and classifies facial expressions in real time.

---

## **Features**  
✅ **Trained on FER-2013 dataset** (7 emotion classes: Happy, Sad, Anger, Fear, Disgust, Neutral, Surprise).  
✅ **Real-time emotion detection** using **OpenCV**.  
✅ **CNN-based deep learning model** with high accuracy.  
✅ **Fast & lightweight implementation** with optimized performance.  

---

## **Technologies Used**  
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy & Pandas**
- **Matplotlib (for visualization)**  

---

## **License**  
This project is licensed under the **MIT License**.  

---

## **Contributing**  
Feel free to contribute by opening issues or pull requests. If you find this project helpful, give it a ⭐ on GitHub!  

🚀 Happy Coding! 😊 
