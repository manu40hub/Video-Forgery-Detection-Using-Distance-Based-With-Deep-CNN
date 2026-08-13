# Video Forgery Detection Using Distance-Based Deep CNN & GRU

[![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg)](https://www.tensorflow.org/)
[![Computer Vision](https://img.shields.io/badge/Library-OpenCV-green.svg)](https://opencv.org/)
[![GUI](https://img.shields.io/badge/GUI-Tkinter-lightgrey.svg)](https://docs.python.org/3/library/tkinter.html)

A deep learning project for detecting video frame tampering and forgery. The system combines **Spatial Feature Extraction** via 2D Convolutional Neural Networks (CNN) wrapped in `TimeDistributed` layers with **Temporal Sequence Learning** using Gated Recurrent Units (GRU). It features a desktop GUI built with `Tkinter` to facilitate dataset loading, frame normalization, model training, evaluation, and real-time video forgery detection.

---

## 📌 Key Features

- **Interactive Desktop GUI**: Built using `Tkinter` with a clean step-by-step workflow interface.
- **Hybrid Deep Learning Model (CNN + GRU)**:
  - **CNN**: Extracts spatial features from individual video frames (`Conv2D` + `MaxPooling2D` with `TimeDistributed`).
  - **GRU**: Captures temporal dynamics and frame-to-frame temporal correlations across the sequence.
- **Dataset Preprocessing & Caching**:
  - Automated frame extraction, resizing ($32 \times 32$), and pixel normalization ($[0, 1]$).
  - Fast feature caching into `X.txt.npy` and `Y.txt.npy` numpy binary files for rapid re-runs.
- **Comprehensive Evaluation**: Calculates and displays Accuracy, Precision, Recall, and F1-Score.
- **Real-Time Video Forgery Detection**:
  - Live video stream display labeling frames as **`Forge Frame`** or **`Real Frame`**.
- **Automated Video Reconstruction**:
  - Filters out forged frames in real-time and exports a cleaned video file (`video_after_removing_forge.mp4`) containing only genuine frames.

---

## 🛠 Architecture & Methodology

```
+-------------------------------------------------------------------+
|                        Input Video Sequence                       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Frame Extraction & Resizing (32x32)                 |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                  Normalization & Shuffling (0-1)                  |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|       Spatial Extraction: TimeDistributed(Conv2D + MaxPool)       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               Temporal Sequence Modeling: GRU Layer               |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|          Classification (Softmax): Real vs. Forged Frame          |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|           Reconstructed Clean Video (Forged Frames Removed)       |
+-------------------------------------------------------------------+
```

### Model Summary
- **Input Shape**: `(batch_size, 1, 32, 32, 3)`
- **Layer 1**: `TimeDistributed(Conv2D(32, (3, 3), activation='relu'))` + `MaxPooling2D((4, 4))` + `Dropout(0.5)`
- **Layer 2**: `TimeDistributed(Conv2D(64, (3, 3), activation='relu'))` + `MaxPooling2D((4, 4))` + `Dropout(0.5)`
- **Layer 3**: `TimeDistributed(Conv2D(128, (3, 3), activation='relu'))` + `MaxPooling2D((2, 2))` + `Dropout(0.5)`
- **Layer 4**: `TimeDistributed(Conv2D(256, (2, 2), activation='relu'))` + `MaxPooling2D((1, 1))` + `Dropout(0.5)`
- **Flatten**: `TimeDistributed(Flatten())`
- **Recurrent Layer**: `GRU(32)`
- **Output Layer**: `Dense(units=classes, activation='softmax')`
- **Loss Function**: `categorical_crossentropy`
- **Optimizer**: `adam`

---

## 📁 Repository Structure

```
Video-Forgery-Detection-Using-Distance-Based-With-Deep-CNN-main/
│
├── README.md                           # Comprehensive documentation
└── VideoForgery/
    ├── Main.py                         # GUI Application and Deep Learning Pipeline
    ├── requirements.txt                # Python package dependencies
    ├── run.bat                         # Batch execution script for Windows
    ├── DatasetLink.txt                 # Kaggle Dataset download link
    ├── SCREENS.docx                    # GUI Screenshots document
    └── model/                          # Saved models & preprocessed data arrays
        ├── X.txt.npy                   # Cached preprocessed frame data
        ├── Y.txt.npy                   # Cached class labels
        ├── dnn_weights.hdf5            # Saved DNN model weights
        ├── best_dnn_weights.hdf5       # Checkpointed best model weights
        └── dnn_history.pckl            # Training history dictionary
```

---

## 📊 Dataset Information

The project uses the **Video Forgery Dataset**, available on Kaggle:
- **Kaggle Link**: [Video Forgery Dataset](https://www.kaggle.com/datasets/neetusingla5/video-forgery-dataset)

### Dataset Directory Layout
Organize dataset videos into subdirectories representing classes (e.g., `Forge` and `Real`):
```
Video_Dataset/
├── Forge/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── Real/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

---

## ⚡ Setup & Installation

### 1. Prerequisites
- **Python**: `3.7` or `3.8` (recommended due to TensorFlow 1.x / Keras 2.3.1 requirements)
- **Virtual Environment** (Optional but recommended)

### 2. Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Video-Forgery-Detection-Using-Distance-Based-With-Deep-CNN.git
   cd Video-Forgery-Detection-Using-Distance-Based-With-Deep-CNN-main/VideoForgery
   ```

2. **Create & Activate a Virtual Environment** (Windows):
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

   *Dependencies in `requirements.txt`:*
   - `numpy==1.20.3`
   - `pandas==1.3.5`
   - `matplotlib==3.1.1`
   - `keras==2.3.1`
   - `tensorflow==1.14.0`
   - `h5py==2.10.0`
   - `protobuf==3.16.0`
   - `scikit-learn==0.22.2.post1`
   - `seaborn==0.10.1`
   - `opencv-python==4.1.1.26`
   - `opencv-contrib-python==4.3.0.36`

---

## 🚀 How to Run

### Method 1: Using Windows Batch File
Double-click `run.bat` inside the `VideoForgery` folder or run from terminal:
```cmd
run.bat
```

### Method 2: Using Python Direct Command
Navigate to the `VideoForgery` directory and execute:
```cmd
python Main.py
```

---

## 📖 Step-by-Step User Guide

1. **Upload Video Dataset**:
   - Click **`Upload Video Dataset`** button.
   - Select the folder containing your dataset subfolders (`Forge`, `Real`).
   - The system extracts frames, resizes them, caches features in `model/X.txt.npy` and labels in `model/Y.txt.npy`.

2. **Shuffle & Normalize Video Frames**:
   - Click **`Shuffle & Normalize Video Frames`**.
   - Converts frame pixel values to range $[0, 1]$, shuffles frames, and converts target labels into categorical one-hot vectors.

3. **Split Frames Train & Test**:
   - Click **`Split Frames Train & Test`**.
   - Reshapes input tensors and splits the data: **80% Training**, **20% Testing**.

4. **Train DNN Model**:
   - Click **`Train DNN Model`**.
   - Builds and trains the CNN + GRU model over 150 epochs (or loads existing pre-trained weights from `model/dnn_weights.hdf5`).
   - Outputs evaluation metrics (Accuracy, Precision, Recall, F1-Score) in the console text box.

5. **Video Based Forgery Detection**:
   - Click **`Video Based Forgey Detection`**.
   - Select an input video file (`.mp4`, `.avi`, etc.) for testing.
   - View frame-by-frame live detection (**`Forge Frame`** or **`Real Frame`**).
   - Generates a cleaned video output saved as `video_after_removing_forge.mp4`.

---

## 📈 Evaluation Metrics

The system computes standard classification metrics on test video frames:
- **Accuracy**: Overall correct frame predictions percentage.
- **Precision**: Macro-averaged precision across Real/Forge classes.
- **Recall**: Macro-averaged recall score.
- **F1-Score**: Harmonic mean of Precision and Recall.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
