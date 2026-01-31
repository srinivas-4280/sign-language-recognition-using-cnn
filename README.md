# Sign Language Recognition Using CNN

This project is a deep-learning based system that recognizes American Sign Language (ASL) alphabet gestures.  
A Convolutional Neural Network (CNN) is trained on the Sign MNIST dataset, and real-time detection is performed using a webcam.

---

## 📁 Files Included

### 🧠 Model & Training
- cnn_model_train.py — Builds and trains the CNN model  
- sign_mnist_train.csv — Training dataset  
- sign_mnist_test.csv — Test dataset  

### 🎥 Detection & Prediction
- live_detection.py — Real-time sign language detection using webcam  
- detect_hand.py — Detects and preprocesses hand region  
- asl_result.py — Predicts ASL letter from a single image  
- generate_result_image.py — Generates output images with predictions  

---

## ⚠️ Note About Model File
The file **trained_model.h5** is NOT uploaded because it exceeds GitHub's 25MB file limit.

You can recreate the model using:

```bash
python cnn_model_train.py
