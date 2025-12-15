# Driver Drowsiness Detection System

A real-time driver drowsiness detection system that combines classical computer vision techniques with deep learning-based eye-state classification to enhance road safety.

## 📋 Project Overview

This project implements a vision-based driver drowsiness detection system that monitors driver alertness through eye-state analysis. The system uses Haar cascade classifiers for face and eye detection, followed by CNN-based classification to determine if eyes are open or closed. When prolonged eye closure is detected, an audible alert is triggered to warn the driver.

**Course:** ECE-5831 - Pattern Recognition and Neural Networks  
**Institution:** University of Michigan - Dearborn  
**Team Members:**
- Parvathi Primilla
- Venkata Sesha Sai Raj Nanduri
- Pradeepa Hari

## 🎯 Key Features

- **Real-time Detection:** Processes live webcam feed for immediate drowsiness detection
- **Dual Model Support:** Implements both custom CNN and MobileNetV2 architectures
- **Temporal Analysis:** Aggregates predictions across 15 consecutive frames to reduce false positives from blinking
- **Alert System:** Triggers audible alarm when drowsiness is detected
- **High Accuracy:** Custom CNN achieves 91.02% accuracy, MobileNetV2 achieves 86.58% accuracy

## 🏗️ System Architecture

The system follows a layered architecture:

1. **Video Acquisition:** Real-time capture from webcam
2. **Face & Eye Detection:** Haar cascade classifiers for region localization
3. **Preprocessing:** Image resizing, normalization, and optional grayscale conversion
4. **Eye-State Classification:** CNN-based prediction (open/closed)
5. **Temporal Aggregation:** Frame-level decision making to reduce false alarms
6. **Alert Generation:** Audio warning when drowsiness threshold is exceeded

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | 91.02% | 0.91 | 0.91 | 0.91 |
| MobileNetV2 | 86.58% | 0.89 | 0.87 | 0.86 |

## 📁 Repository Contents

- `Driver_Drowsiness_Detection.ipynb` - Complete implementation with executed cells
- `cnn_drowsiness_model.h5` - Trained custom CNN model weights
- `mobilenet_drowsiness_model.h5` - Trained MobileNetV2 model weights
- `live_demo.py` - Real-time drowsiness detection script
- `.gitignore` - Git ignore configuration
- `README.md` - Project documentation

## 🔗 Project Resources

- **📹 Demo Video:** https://youtu.be/E05-RDMoDzg
- **🎥 Presentation Video:** https://youtu.be/Z907nrLepk8 ; https://drive.google.com/file/d/1K0Ik6LjSpuQFNPheaCAMi9z7ZwmSsnG2/view?usp=drive_link
- **📊 Presentation Slides:** https://drive.google.com/drive/folders/1_KgcQHQtdtvkNrBMRGW3KqRMej1L1G0z?usp=drive_link
- **📄 Project Report:** https://drive.google.com/file/d/1yjUx8WlIBN5IU0JN9Efdqmd76-375BU9/view?usp=drive_link
- **💾 Dataset:** https://drive.google.com/drive/folders/192jIgznDinHjgEtUXNb4X9McoAkxDCZi?usp=drive_link
- [MRL Eye Dataset](https://www.kaggle.com/datasets/your-dataset-link)

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
OpenCV
NumPy
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/parvathiprimilla/ece5831-2025-final-project.git
cd ece5831-2025-final-project
```

2. Install required packages:
```bash
pip install tensorflow opencv-python numpy matplotlib
```

### Running the System

1. **Using Jupyter Notebook:**
   - Open `Driver_Drowsiness_Detection.ipynb`
   - Run all cells sequentially

2. **Real-time Detection:**
```bash
python live_demo.py --model_path cnn_drowsiness_model.h5
```

The system will:
- Access your webcam
- Detect your face and eyes using Haar cascades
- Classify eye state (open/closed) using the trained models
- Trigger an audio alert if eyes remain closed for 15 consecutive frames

## 🔬 Technical Details

### Dataset
- **MRL Eye Dataset:** 80,000 eye images (approx.)
- **Classes:** Open eyes and closed eyes (binary classification)
- **Split:** Training and validation sets

### CNN Architecture
- Custom convolutional neural network
- Multiple Conv2D layers with ReLU activation
- MaxPooling for spatial dimension reduction
- Dropout layers for regularization
- Dense layers for classification

### Detection Logic
- Eyes detected using Haar cascade classifiers
- Predictions made on each detected eye region
- Threshold-based classification (open/closed)
- Drowsiness triggered after 15 consecutive frames of closed eyes
- Audio beep alert for driver warning

## 📈 Results

### CNN Model
- Validation Accuracy: 91.04%
- Validation Loss: 0.3738
- Correctly classified: 7,726 closed eyes, 7,720 open eyes
- False positives: 663 closed misclassified as open
- False negatives: 870 open misclassified as closed

### MobileNetV2 Model
- Validation Accuracy: 86.58%
- Validation Loss: 1.1066
- High recall (0.99) for closed eyes
- Lower recall (0.74) for open eyes
- Tendency to misclassify open eyes as closed

## 🔮 Future Improvements

- Replace Haar cascades with deep learning-based detectors (MTCNN, MediaPipe)
- Integrate LSTM networks for temporal modeling
- Add multi-modal features:
  - Yawning detection
  - Head pose estimation
  - Blink rate analysis
- Fine-tune MobileNetV2 or experiment with EfficientNet
- Improve robustness to lighting conditions and occlusions

## 🙏 Acknowledgments

We would like to thank:
- **Dr. Jaerock Kwon** - Course instructor for guidance and feedback
- **Aydin Zaboli** - Teaching assistant for implementation support
- Contributors to **TensorFlow** and **OpenCV** frameworks
- MRL Eye Dataset creators

## 📝 References

1. Y. Suresh et al., "Driver Drowsiness Detection Using Deep Learning," IEEE ICOSEC, 2021
2. R. Jabbar et al., "Driver Drowsiness Detection Model Using CNN Techniques," IEEE ICIoT, 2020
3. V. R. R. Chirra et al., "Deep CNN: A Machine Learning Approach for Driver Drowsiness Detection," IJACSA, 2021
4. D. F. Dinges and R. Grace, "PERCLOS: A Valid Psychophysiological Measure of Alertness," 1998
5. A. G. Howard et al., "MobileNets: Efficient CNNs for Mobile Vision Applications," arXiv:1704.04861, 2017

## 📄 License

This project is developed for educational purposes as part of ECE-5831 coursework at University of Michigan - Dearborn.

## 📧 Contact

For questions or collaboration:
- Parvathi Primilla - pparu@umich.edu
- Venkata Sesha Sai Raj Nanduri - nsairaj@umich.edu
- Pradeepa Hari - prade@umich.edu

---

**Note:** This is an academic project developed for ECE-5831 Pattern Recognition and Neural Networks course (2025). The system is designed for research and educational purposes.
