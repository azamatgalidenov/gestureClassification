# Gesture Classification

## Overview
This project implements a gesture classification system using TensorFlow Lite on microcontrollers, specifically targeting the ESP32 platform. The system is capable of recognizing hand gestures (rock, paper, scissors), which can be utilized in various applications such as robotics, smart home devices, and interactive systems.

## Project Structure
```
gestureClassification/
├── 1_Collect_data.py          # Script to collect gesture data from ESP32 camera
├── 2_Train.ipynb              # Jupyter notebook for training the model
├── 3_OnnxToTf.py              # Script for converting ONNX model to TensorFlow format
├── 4_TFtofloat32.py           # Script for converting TensorFlow model to float32 format
├── 5_TftoArrayquantized(4_alternative).py  # Script for converting TensorFlow model to TFLite format
├── README.md                   # Project overview and documentation
├── requirements.txt            # Python dependencies
├── Arduino_TensorFlowLite_ESP32/ # Arduino Libs
├── CameraWebServer/            # Web server for streaming camera data from ESP32
├── data/                       # Directory for storing gesture training data
│   ├── paper/
│   ├── rock/
│   └── scissors/
├── main/                       # Main application files
│   ├── main.ino                # Arduino sketch for gesture classification
│   ├── modelnooptimized.h      # Model header file (unoptimized)
│   └── modelquantized.h        # Model header file (quantized)
└── trained_models/             # Directory for storing trained models
    ├── esp32_gesture_model.pth # PyTorch model
    └── esp32_model.onnx        # ONNX model format
```

## Workflow

### 1. Data Collection (`1_Collect_data.py`)
Collect training data by capturing images of gestures using an ESP32 camera.

**Setup:**
- Configure your ESP32 IP address in the script
- Ensure ESP32 is running the CameraWebServer sketch
- Connect to the same network as your ESP32

**Usage:**
```bash
python 1_Collect_data.py
```

The script captures images and organizes them into folders for each gesture class (rock, paper, scissors).

### 2. Model Training (`2_Train.ipynb`)
Train a neural network model using the collected gesture data.

**Features:**
- Data preprocessing and augmentation
- Model architecture definition
- Training and validation
- Model evaluation and metrics

**Usage:**
Open the notebook in Jupyter and follow the cells sequentially:
```bash
jupyter notebook 2_Train.ipynb
```

### 3. ONNX to TensorFlow Conversion (`3_OnnxToTf.py`)
Convert the model from ONNX format to TensorFlow format for further optimization.

**Purpose:**
- Converts ONNX model (esp32_model.onnx) to TensorFlow SavedModel format
- Enables compatibility with TensorFlow conversion tools
- Required intermediate step before TensorFlow Lite conversion

**Usage:**
```bash
python 3_OnnxToTf.py
```

### 4. TensorFlow to Float32 Conversion (`4_TFtofloat32.py`)
Convert the TensorFlow model to float32 format for compatibility.

**Purpose:**
- Converts TensorFlow model to float32 precision
- Prepares model for quantization in the next step
- Ensures model weights are in the correct format

**Usage:**
```bash
python 4_TFtofloat32.py
```

### 5. Model Conversion (`5_TftoArrayquantized(4_alternative).py`)
Convert the trained TensorFlow model to TensorFlow Lite format optimized for microcontrollers.

**Features:**
- Automatic input shape detection
- Model quantization for reduced size
- C header file generation for embedded deployment

**Usage:**
```bash
python 5_TftoArrayquantized(4_alternative).py
```

### Deployment on ESP32
Deploy the quantized model to ESP32 using the Arduino IDE.

**Steps:**
1. Install TensorFlow Lite for Microcontrollers library
2. Copy the generated model header file to your sketch
3. Upload the Arduino sketch to your ESP32

## Installation

### Prerequisites
- Python 3.7+
- Arduino IDE
- ESP32 development board
- Camera module (optional, for CameraWebServer)

### Python Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gestureClassification
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Arduino Setup
1. Install the TensorFlow Lite for Microcontrollers library via Arduino IDE
2. Install ESP32 board support in Arduino IDE
3. Copy the library files from `Arduino_TensorFlowLite_ESP32` to your Arduino libraries folder

## Hardware Requirements
- **ESP32 Development Board** (e.g., ESP32-CAM for camera support)
- **Camera Module** (optional, for image capture)
- **USB Cable** for programming
- **Network Connection** (WiFi)

## Examples

### person_detection
Real-time person detection using the camera module.

### main.ino
Main gesture classification application that runs on ESP32.

## Technologies Used
- **TensorFlow Lite** - Machine learning framework for microcontrollers
- **PyTorch** - Model training framework
- **Python** - Data collection and preprocessing
- **Arduino** - Embedded development
- **ESP32** - Microcontroller platform
- **OpenCV** - Computer vision library