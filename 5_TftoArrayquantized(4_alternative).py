import tensorflow as tf
import numpy as np
import os
import cv2

# ==================================================================================
# CONFIGURATION
# ==================================================================================
MODEL_PATH = "models/model_tf"
DATASET_DIR = "dataset"
IMG_SIZE = 96
CALIBRATION_IMAGES = 50  # Number of images to use for calibration (more = better accuracy)

# ==================================================================================
# 1. ANALYZE INPUT SHAPE
# Why? We must feed data to the converter in the exact shape the model expects.
# PyTorch usually exports NCHW (Channels First), while TensorFlow expects NHWC (Channels Last).
# This block automatically detects which one the model currently has.
# ==================================================================================
print("Loading model signature...")
model = tf.saved_model.load(MODEL_PATH)
signature = model.signatures["serving_default"]
input_details = list(signature.inputs)[0]
shape = input_details.shape.as_list() # e.g., [1, 96, 96, 1] or [1, 1, 96, 96]

print(f"Model expects shape: {shape}")

# Logic: If the 2nd dimension (index 1) is small (1 or 3), it's Channels-First (NCHW).
# Otherwise, it's standard TensorFlow Channels-Last (NHWC).
is_nchw = (shape[1] == 1 or shape[1] == 3)
if is_nchw: print("Format: NCHW (Channels First)")
else:       print("Format: NHWC (Channels Last)")

# ==================================================================================
# 2. REPRESENTATIVE DATASET GENERATOR
# Why? To convert Float32 to Int8, the converter needs to measure the "dynamic range"
# of activations. It runs these sample images through the network to record 
# min/max values. This ensures the Int8 integers accurately represent the real data.
# ==================================================================================
def representative_data_gen():
    count = 0
    # Walk through dataset folders
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if count >= CALIBRATION_IMAGES: break
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')): continue

            # 1. Load and Resize
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Force Grayscale

            # 2. Normalize (0.0 -> 1.0)
            # Crucial: Must match the preprocessing done during training!
            img = img.astype(np.float32) / 255.0

            # 3. Reshape based on detected format
            if is_nchw:
                # [1, Channels, Height, Width]
                img = np.reshape(img, (1, 1, IMG_SIZE, IMG_SIZE))
            else:
                # [1, Height, Width, Channels]
                img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 1))

            # Yield: Passes data to the converter one by one
            yield [img]
            count += 1

# ==================================================================================
# 3. CONVERT TO INT8 (QUANTIZATION)
# Why? 
# 1. Size: Float32 (4 bytes) -> Int8 (1 byte). Model becomes 4x smaller.
# 2. Speed: ESP32 CPU is much faster at integer math than floating point math.
# ==================================================================================
print("Quantizing...")
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_PATH)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force Internal Operations to Int8
# If an op isn't supported in Int8, it will crash (ensures pure optimization)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Keep Input/Output as Float32
# Why? If we set this to Int8, the ESP32 code would need to manually "dequantize" 
# input numbers. Keeping it Float makes the C++ code simpler (just copy float buffer).
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
model_len = len(tflite_model)
print(f"Quantized Size: {model_len / 1024:.2f} KB")

# ==================================================================================
# 4. EXPORT TO C HEADER
# Why? Embeds the binary directly into Flash memory.
# ==================================================================================
print("Generating header file...")
with open("models/modelquantized.h", "w", encoding="utf-8") as f:
    f.write("// Auto-generated Int8 TFLite Model\n")
    f.write("#include <pgmspace.h>\n\n")
    f.write(f"const int g_model_len = {model_len};\n")
    
    # aligned(16) is required for TFLite Micro memory access
    f.write("const unsigned char g_model[] __attribute__((aligned(16))) PROGMEM = {\n")
    
    hex_values = [f"0x{b:02x}" for b in tflite_model]
    
    # Write in chunks of 12 for readability
    for i in range(0, len(hex_values), 12):
        chunk = hex_values[i:i+12]
        is_last = (i + 12 >= len(hex_values))
        f.write("  " + ", ".join(chunk) + ("\n" if is_last else ",\n"))
        
    f.write("};\n")

print("Success: models/modelquantized.h ready for ESP32.")